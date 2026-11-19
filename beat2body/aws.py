"""SageMaker training jobs (AWS): upload data, submit, check status, download results.

    python -m beat2body aws upload-data --tool dwpose
    python -m beat2body aws submit --instance ml.g5.2xlarge --max-hours 4 -- --model diffusion --steps 60000
    python -m beat2body aws status [JOB]          # latest beat2body job if JOB is omitted
    python -m beat2body aws download JOB          # model.pt, metrics, samples -> runs/JOB/

Arguments after `--` are passed to `beat2body train` inside the job. Defaults: bucket
beat2body-<account>-<region>, role SageMakerTraining-beat2body, PyTorch 2.9 training image
pinned by digest. Needs: pip install boto3 'sagemaker>=3.22,<4' 'botocore[crt]' and AWS credentials.
Price per instance hour: https://aws.amazon.com/sagemaker/ai/pricing/
"""
import argparse
import io
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import uuid

from beat2body import paths

DOC = __doc__
REGION = os.environ.get("AWS_REGION", "us-east-1")
ROLE_NAME = "SageMakerTraining-beat2body"
# pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker, verified in ECR on 2026-11-18
IMAGE = ("763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training@"
         "sha256:86fbaa7cef705829a5cb92de9abca974817ca8b274b3b3d978cc2778b458ee67")
JOB_PREFIX = "beat2body"
os.environ.setdefault("AWS_SDK_UA_APP_ID", "AWSSkill-SageMaker")
# The SDK's client-side role check requires a generic permission list, including EC2 network
# permissions that only VPC jobs need. The role is deliberately least-privilege (no VPC), and
# SageMaker validates the role itself when the job starts, so the client-side check is skipped.
os.environ.setdefault("SAGEMAKER_VALIDATE_ROLE", "false")


def add_args(p):
    sub = p.add_subparsers(dest="aws_cmd", required=True)
    up = sub.add_parser("upload-data", help="pack the training data and upload it to S3")
    up.add_argument("--tool", default="dwpose", choices=paths.TOOLS)
    sb = sub.add_parser("submit", help="start a SageMaker training job")
    sb.add_argument("--tool", default="dwpose", choices=paths.TOOLS)
    sb.add_argument("--instance", default="ml.g5.2xlarge")
    sb.add_argument("--spot", action="store_true", help="managed spot training (cheaper, may be interrupted)")
    sb.add_argument("--max-hours", type=float, default=4.0, help="hard stop (spend bound)")
    sb.add_argument("--name", default="", help="suffix for the job name")
    sb.add_argument("--dry-run", action="store_true", help="print the job configuration, do not submit")
    sb.add_argument("train_args", nargs=argparse.REMAINDER, help="-- then options for `beat2body train`")
    st = sub.add_parser("status", help="job status and latest training metrics")
    st.add_argument("job", nargs="?")
    dl = sub.add_parser("download", help="download a finished job's outputs to runs/JOB/")
    dl.add_argument("job")
    for sp in (up, sb, st, dl):
        sp.add_argument("--bucket", default=None)


def run(args):
    {"upload-data": upload_data, "submit": submit, "status": status, "download": download}[args.aws_cmd](args)


def _clients():
    import boto3
    s = boto3.Session(region_name=REGION)
    return s, s.client("sts").get_caller_identity()["Account"]


def _bucket(args, account):
    return args.bucket or f"beat2body-{account}-{REGION}"


def data_key(tool):
    return f"data/{tool}/2d.tar"


def make_archive(path, members):
    with tarfile.open(path, "w") as t:
        for m in members:
            t.add(os.path.join(paths.data_root(), m), arcname=os.path.join("2d", m))


def upload_data(args):
    session, account = _clients()
    bucket = _bucket(args, account)
    root = paths.data_root()
    members = ["categories.json", "features", os.path.join("keypoints", args.tool),
               os.path.join("clips", args.tool, "meta.csv"), os.path.join("clips", args.tool, "split.json")]
    for m in members:
        if not os.path.exists(os.path.join(root, m)):
            sys.exit(f"{m} missing under {root}; run `python -m beat2body hf download --tool {args.tool}`")
    tmp = tempfile.NamedTemporaryFile(suffix=".tar", delete=False)
    tmp.close()
    make_archive(tmp.name, members)
    size = os.path.getsize(tmp.name) / 1e6
    print(f"[aws] uploading {size:.0f} MB to s3://{bucket}/{data_key(args.tool)}")
    session.client("s3").upload_file(tmp.name, bucket, data_key(args.tool))
    os.remove(tmp.name)
    print("[aws] done")


def _stage_source():
    """Copy the package and the entry script to a clean folder (what the job runs)."""
    d = tempfile.mkdtemp(prefix="beat2body_src_")
    shutil.copytree(paths.PACKAGE_DIR, os.path.join(d, "beat2body"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(os.path.join(paths.PACKAGE_DIR, "sagemaker_entry.py"), os.path.join(d, "sagemaker_entry.py"))
    return d


def train_hyperparameters(train_args):
    """['--model', 'diffusion', '--tiny'] -> {'model': 'diffusion', 'tiny': True}."""
    a = [x for x in train_args if x != "--"]
    hps, i = {}, 0
    while i < len(a):
        if not a[i].startswith("--"):
            sys.exit(f"unexpected train argument {a[i]!r}; use --key value")
        key = a[i][2:].replace("-", "_")
        if i + 1 < len(a) and not a[i + 1].startswith("--"):
            hps[key], i = a[i + 1], i + 2
        else:
            hps[key], i = True, i + 1
    for forbidden in ("out", "checkpoint_dir", "download", "data_root"):
        if forbidden in hps:
            sys.exit(f"--{forbidden} is set by the job; remove it")
    return hps


def submit(args):
    from sagemaker.core.helper.session_helper import Session
    from sagemaker.core.training.configs import (
        CheckpointConfig,
        Compute,
        InputData,
        OutputDataConfig,
        SourceCode,
        StoppingCondition,
    )
    from sagemaker.train import ModelTrainer

    session, account = _clients()
    bucket = _bucket(args, account)
    s3 = session.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=data_key(args.tool))
    except Exception:
        sys.exit(f"s3://{bucket}/{data_key(args.tool)} not found; run `python -m beat2body aws upload-data` first")
    hps = train_hyperparameters(args.train_args)
    hps.setdefault("tool", args.tool)
    name = re.sub(r"[^A-Za-z0-9-]", "-", f"{JOB_PREFIX}-{hps.get('model', 'diffusion')}-{args.name}").strip("-")
    prefix = f"{name[:30].rstrip('-')}-{uuid.uuid4().hex[:8]}"
    role = f"arn:aws:iam::{account}:role/{ROLE_NAME}"
    max_runtime = int(args.max_hours * 3600)
    summary = {"job_prefix": prefix, "instance": args.instance, "spot": args.spot, "max_runtime_h": args.max_hours,
               "image": IMAGE, "role": role, "data": f"s3://{bucket}/{data_key(args.tool)}",
               "output": f"s3://{bucket}/output/", "checkpoints": f"s3://{bucket}/checkpoints/{prefix}/",
               "hyperparameters": hps}
    for k, v in summary.items():
        print(f"  {k:15s} {v}")
    if args.dry_run:
        print("[aws] dry run: nothing submitted")
        return
    sm = session.client("sagemaker")
    trainer = ModelTrainer(
        training_image=IMAGE,
        source_code=SourceCode(source_dir=_stage_source(), entry_script="sagemaker_entry.py"),
        compute=Compute(instance_type=args.instance, instance_count=1, volume_size_in_gb=30,
                        enable_managed_spot_training=args.spot),
        output_data_config=OutputDataConfig(s3_output_path=f"s3://{bucket}/output/"),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=max_runtime,
                                             max_wait_time_in_seconds=2 * max_runtime if args.spot else None),
        checkpoint_config=CheckpointConfig(s3_uri=f"s3://{bucket}/checkpoints/{prefix}/"),
        hyperparameters={k: str(v) if not isinstance(v, bool) else v for k, v in hps.items()},
        sagemaker_session=Session(boto_session=session, sagemaker_client=sm, default_bucket=bucket,
                                  default_bucket_prefix="code"),
        role=role,
        base_job_name=prefix,
    )
    trainer.train(input_data_config=[InputData(channel_name="training",
                                               data_source=f"s3://{bucket}/data/{args.tool}/")],
                  wait=False, logs=False)
    job = _find_job(sm, prefix)
    print(f"[aws] submitted {job}\n      python -m beat2body aws status {job}")


def _find_job(sm, prefix, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        jobs = [j["TrainingJobName"] for j in sm.list_training_jobs(NameContains=prefix, SortBy="CreationTime",
                                                                  SortOrder="Descending", MaxResults=10)
                ["TrainingJobSummaries"] if j["TrainingJobName"].startswith(prefix)]
        if len(jobs) == 1:
            return jobs[0]
        time.sleep(2)
    sys.exit(f"job with prefix {prefix} not visible after {timeout}s")


def _latest_job(sm):
    jobs = sm.list_training_jobs(NameContains=JOB_PREFIX, SortBy="CreationTime", SortOrder="Descending",
                                 MaxResults=1)["TrainingJobSummaries"]
    if not jobs:
        sys.exit("no beat2body training jobs")
    return jobs[0]["TrainingJobName"]


def status(args):
    import json
    session, _ = _clients()
    sm = session.client("sagemaker")
    job = args.job or _latest_job(sm)
    d = sm.describe_training_job(TrainingJobName=job)
    secs = d.get("TrainingTimeInSeconds") or 0
    print(f"{job}: {d['TrainingJobStatus']} / {d.get('SecondaryStatus')}  "
          f"instance {d['ResourceConfig']['InstanceType']}  trained {secs / 60:.0f} min, "
          f"billable {(d.get('BillableTimeInSeconds') or 0) / 60:.0f} min")
    if d.get("FailureReason"):
        print("  failure:", d["FailureReason"][:500])
    logs = session.client("logs")
    try:
        streams = logs.describe_log_streams(logGroupName="/aws/sagemaker/TrainingJobs", logStreamNamePrefix=job)
        lines = []
        for s in streams["logStreams"]:
            ev = logs.get_log_events(logGroupName="/aws/sagemaker/TrainingJobs", logStreamName=s["logStreamName"],
                                     startFromHead=False, limit=200)["events"]
            lines += [e["message"] for e in ev]
        metrics = [json.loads(m) for m in lines if m.startswith("{") and '"step"' in m]
        for key in ("loss", "val_loss", "sample_speed_ratio"):
            last = [m for m in metrics if key in m]
            if last:
                print(f"  {key:18s} {last[-1][key]:.4f} at step {last[-1]['step']}")
        if not metrics:
            print("  last log lines:\n    " + "\n    ".join(lines[-8:]))
    except logs.exceptions.ResourceNotFoundException:
        print("  (no logs yet)")


def download(args):
    session, _ = _clients()
    sm = session.client("sagemaker")
    d = sm.describe_training_job(TrainingJobName=args.job)
    if d["TrainingJobStatus"] != "Completed":
        sys.exit(f"{args.job} is {d['TrainingJobStatus']}; model artifacts exist only for completed jobs")
    uri = d["ModelArtifacts"]["S3ModelArtifacts"]
    bucket, key = uri[5:].split("/", 1)
    body = session.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    out = os.path.join(paths.REPO_ROOT, "runs", args.job)
    os.makedirs(out, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(body)) as t:
        t.extractall(out, filter="data")
    raw = os.path.join(out, "model_raw.pt")
    ck = d.get("CheckpointConfig", {}).get("S3Uri")
    if ck and not os.path.exists(raw) and os.path.exists(os.path.join(out, "model.pt")):
        # older jobs saved only the EMA weights: rebuild the plain weights from the resume checkpoint
        import torch
        cb, ckey = ck[5:].split("/", 1)
        try:
            data = session.client("s3").get_object(Bucket=cb, Key=ckey.rstrip("/") + "/last.pt")["Body"].read()
            last = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
            bundle = torch.load(os.path.join(out, "model.pt"), map_location="cpu", weights_only=False)
            bundle["model"] = last["model"]
            torch.save(bundle, raw)
            print(f"[aws] wrote {raw} (plain weights from the checkpoint at step {last['step']})")
        except Exception as e:  # noqa: BLE001
            print(f"[aws] no plain weights: {type(e).__name__}: {str(e)[:120]}")
    print(f"[aws] {uri} -> {out}: {sorted(os.listdir(out))}")
