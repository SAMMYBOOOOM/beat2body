"""SageMaker training jobs (AWS): upload data, submit, check status, download results.

    python -m beat2body aws upload-data --tool dwpose
    python -m beat2body aws submit --instance ml.g5.2xlarge --max-hours 4 -- --model diffusion --steps 60000
    python -m beat2body aws status [JOB]          # latest beat2body job if JOB is omitted
    python -m beat2body aws progress [JOB ...]    # per-checkpoint table (speed, beat power, ...) of running jobs
    python -m beat2body aws download JOB          # model.pt, metrics, samples -> runs/JOB/
    python -m beat2body aws download JOB --live   # progress of a running job (metrics, samples, snapshots)
    python -m beat2body aws pose --videos ids.txt --shards 4 --instances ml.g6.xlarge ml.g6.xlarge:spot ...
    python -m beat2body aws pose-download PREFIX...   # DWPose raw outputs -> <data>/keypoints/<category>/

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
# pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker, verified in ECR on 2026-11-18 (us-east-1) and
# 2026-11-20 (us-west-2, same digest). Another region: set AWS_REGION; the job uses that region's copy.
IMAGE = (f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training@"
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
    pg = sub.add_parser("progress", help="per-checkpoint training checks of jobs (default: all running ones)")
    pg.add_argument("jobs", nargs="*", help="job names or any part of them (e.g. the 8-character id)")
    pg.add_argument("--last", type=int, default=3, help="checkpoints shown per job (0 = all)")
    dl = sub.add_parser("download", help="download a job's outputs to runs/JOB/")
    dl.add_argument("job")
    dl.add_argument("--live", action="store_true",
                    help="also / only (while running) the synced checkpoint folder: metrics, samples, snapshots")
    dl.add_argument("--no-snapshots", action="store_true", help="with --live: skip the model snapshots")
    po = sub.add_parser("pose", help="DWPose on AWS GPUs for a list of AIST video ids (sharded jobs)")
    po.add_argument("--videos", required=True, help="text file with one AIST video id per line")
    po.add_argument("--shards", type=int, default=1)
    po.add_argument("--instances", nargs="+", default=["ml.g6.xlarge"],
                    help="one per shard (cycled); append :spot for managed spot, e.g. ml.g6.xlarge:spot")
    po.add_argument("--max-hours", type=float, default=6.0)
    po.add_argument("--name", default="pose")
    po.add_argument("--only", type=int, nargs="*", default=None, help="submit only these shard numbers")
    po.add_argument("--dry-run", action="store_true")
    pd = sub.add_parser("pose-download", help="download pose jobs' raw outputs (also while they run)")
    pd.add_argument("prefixes", nargs="+", help="checkpoint prefixes printed by `aws pose`")
    for sp in (up, sb, st, pg, dl, po, pd):
        sp.add_argument("--bucket", default=None)


def run(args):
    {"upload-data": upload_data, "submit": submit, "status": status, "download": download, "pose": pose,
     "pose-download": pose_download, "progress": progress}[args.aws_cmd](args)


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
    members = ["categories.json", "features"] + [os.path.join("keypoints", t) for t in paths.TOOLS
                                                   if os.path.isdir(os.path.join(root, "keypoints", t))] + [
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
    job = _launch(session, bucket, prefix, args.instance, args.spot, max_runtime, hps,
                  [("training", f"s3://{bucket}/data/{args.tool}/")])
    print(f"[aws] submitted {job}\n      python -m beat2body aws status {job}")


def _launch(session, bucket, prefix, instance, spot, max_runtime, hps, inputs, extra_files=None):
    """Start one training job running sagemaker_entry.py; extra_files {name: text} go next to it."""
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
    src = _stage_source()
    for name, text in (extra_files or {}).items():
        with open(os.path.join(src, name), "w") as f:
            f.write(text)
    sm = session.client("sagemaker")
    trainer = ModelTrainer(
        training_image=IMAGE,
        source_code=SourceCode(source_dir=src, entry_script="sagemaker_entry.py"),
        compute=Compute(instance_type=instance, instance_count=1, volume_size_in_gb=30,
                        enable_managed_spot_training=spot),
        output_data_config=OutputDataConfig(s3_output_path=f"s3://{bucket}/output/"),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=max_runtime,
                                             max_wait_time_in_seconds=2 * max_runtime if spot else None),
        checkpoint_config=CheckpointConfig(s3_uri=f"s3://{bucket}/checkpoints/{prefix}/"),
        hyperparameters={k: str(v) if not isinstance(v, bool) else v for k, v in hps.items()},
        sagemaker_session=Session(boto_session=session, sagemaker_client=sm, default_bucket=bucket,
                                  default_bucket_prefix="code"),
        role=f"arn:aws:iam::{session.client('sts').get_caller_identity()['Account']}:role/{ROLE_NAME}",
        base_job_name=prefix,
    )
    trainer.train(input_data_config=[InputData(channel_name=c, data_source=uri) for c, uri in inputs] or None,
                  wait=False, logs=False)
    return _find_job(sm, prefix)


def pose(args):
    session, account = _clients()
    bucket = _bucket(args, account)
    with open(args.videos) as f:
        ids = [x.strip() for x in f if x.strip()]
    stamp = uuid.uuid4().hex[:6]
    for i in range(args.shards):
        if args.only is not None and i not in args.only:
            continue
        inst = args.instances[i % len(args.instances)]
        instance, spot = inst.split(":")[0], inst.endswith(":spot")
        prefix = f"{JOB_PREFIX}-{args.name}-{stamp}-s{i}of{args.shards}"
        n = len(ids[i::args.shards])
        print(f"  shard {i}/{args.shards}: {n} videos on {instance}{' (spot)' if spot else ''}  "
              f"checkpoints s3://{bucket}/checkpoints/{prefix}/")
        if args.dry_run:
            continue
        job = _launch(session, bucket, prefix, instance, spot, int(args.max_hours * 3600),
                      {"task": "pose", "shard": f"{i}/{args.shards}"}, [], {"pose_videos.txt": "\n".join(ids)})
        print(f"[aws] submitted {job}")
    if args.dry_run:
        print("[aws] dry run: nothing submitted")


def pose_download(args):
    session, account = _clients()
    bucket = _bucket(args, account)
    s3 = session.client("s3")
    n = 0
    for prefix in args.prefixes:
        base = f"checkpoints/{prefix}/keypoints/"
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=base):
            for o in page.get("Contents", []):
                dst = os.path.join(paths.keypoints_dir(), o["Key"][len(base):])
                if os.path.exists(dst) and os.path.getsize(dst) == o["Size"]:
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                s3.download_file(bucket, o["Key"], dst)
                n += 1
    print(f"[aws] {n} new raw pose outputs -> {paths.rel(paths.keypoints_dir())}")


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


def _job_metrics(logs, job):
    """All JSON metric lines of a job's CloudWatch log, in order."""
    import json
    out = []
    try:
        streams = logs.describe_log_streams(logGroupName="/aws/sagemaker/TrainingJobs", logStreamNamePrefix=job)
    except logs.exceptions.ResourceNotFoundException:
        return out
    for s in streams["logStreams"]:
        token = None
        while True:
            kw = dict(logGroupName="/aws/sagemaker/TrainingJobs", logStreamName=s["logStreamName"], startFromHead=True)
            if token:
                kw["nextToken"] = token
            r = logs.get_log_events(**kw)
            for e in r["events"]:
                if e["message"].startswith('{"step"'):
                    try:
                        out.append(json.loads(e["message"]))
                    except ValueError:
                        pass
            if r["nextForwardToken"] == token:
                break
            token = r["nextForwardToken"]
    return out


def progress(args):
    """Table of the training checks (every --sample-every steps) of one or more jobs."""
    session, _ = _clients()
    sm, logs = session.client("sagemaker"), session.client("logs")
    if args.jobs:
        jobs = []
        for q in args.jobs:
            found = [j for page in sm.get_paginator("list_training_jobs").paginate(
                NameContains=q, SortBy="CreationTime", SortOrder="Descending") for j in page["TrainingJobSummaries"]]
            if not found:
                sys.exit(f"no training job matches {q!r}")
            jobs.append(found[0]["TrainingJobName"])
    else:
        jobs = [j["TrainingJobName"] for j in sm.list_training_jobs(
            NameContains=JOB_PREFIX, StatusEquals="InProgress", SortBy="CreationTime", SortOrder="Ascending",
            MaxResults=20)["TrainingJobSummaries"]]
        if not jobs:
            print("no running beat2body jobs; name some: python -m beat2body aws progress <job> ...")
    for job in jobs:
        d = sm.describe_training_job(TrainingJobName=job)
        rows = _job_metrics(logs, job)
        start = next((r for r in rows if r.get("event") == "start"), {})
        last = [r for r in rows if "loss" in r]
        total = start.get("steps") or d.get("HyperParameters", {}).get("steps", "?")
        line = (f"\n{job}\n  {d['TrainingJobStatus']} / {d.get('SecondaryStatus')}, "
                f"{d['ResourceConfig']['InstanceType']}")
        if d.get("EnableManagedSpotTraining"):
            line += " (spot)"
        if last:
            x = last[-1]
            line += f", step {x['step']}/{total}, {x.get('steps_per_s', 0):.1f} steps/s"
            if isinstance(total, int) and x.get("steps_per_s"):
                line += f", ~{(total - x['step']) / x['steps_per_s'] / 60:.0f} min left"
        print(line)
        if start:
            keys = ("recipe", "d_model", "layers", "bone_weight", "energy_weight", "scale_norm", "select")
            print("  " + ", ".join(f"{k} {start.get(k)}" for k in keys if k in start))
        val = {r["step"]: r["val_loss"] for r in rows if "val_loss" in r}
        sel = {r["step"]: r["select"].get("val_fid_k") for r in rows if isinstance(r.get("select"), dict)}
        checks = [(r["step"], r["sample"]) for r in rows if isinstance(r.get("sample"), dict)]
        if args.last:
            checks = checks[-args.last:]
        if not checks:
            print("  (no training checks yet)")
            continue
        print(f"  {'step':>7} {'val_loss':>9} {'speed':>6} {'beat_pow':>9} {'body_cv':>8} {'(real)':>7} "
              f"{'val_fid_k':>10}")
        for step, x in checks:
            fid = sel.get(step)
            print(f"  {step:>7} {val.get(step, float('nan')):9.3f} {x.get('speed_ratio', float('nan')):6.2f} "
                  f"{x.get('beat_power', float('nan')):9.2f} {x.get('bone_cv', float('nan')):8.3f} "
                  f"{x.get('bone_cv_real', float('nan')):7.3f} {'' if fid is None else f'{fid:10.1f}'}")
    print("\nspeed / beat_pow: generated / real on 8 val clips (1 = real); body_cv: body-size variation (match real)")


def download_live(session, d, out, snapshots=True):
    """Copy the job's checkpoint folder (synced to S3 while it runs) to <out>/live/, except last.pt."""
    ck = d.get("CheckpointConfig", {}).get("S3Uri")
    if not ck:
        print("[aws] this job has no checkpoint folder")
        return
    bucket, prefix = ck[5:].split("/", 1)
    s3 = session.client("s3")
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix.rstrip("/")) + 1:]
            if rel == "last.pt" or (not snapshots and rel.startswith("snapshots/")):
                continue
            dst = os.path.join(out, "live", rel)
            if os.path.exists(dst) and os.path.getsize(dst) == obj["Size"]:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            s3.download_file(bucket, obj["Key"], dst)
            n += 1
    print(f"[aws] {n} new files from {ck} -> {os.path.join(out, 'live')}")


def download(args):
    session, _ = _clients()
    sm = session.client("sagemaker")
    d = sm.describe_training_job(TrainingJobName=args.job)
    out = os.path.join(paths.REPO_ROOT, "runs", args.job)
    os.makedirs(out, exist_ok=True)
    if args.live:
        download_live(session, d, out, not args.no_snapshots)
    if d["TrainingJobStatus"] != "Completed":
        if args.live:
            print(f"[aws] {args.job} is {d['TrainingJobStatus']}: final model not available yet")
            return
        sys.exit(f"{args.job} is {d['TrainingJobStatus']}; use --live for progress, the model comes when it completes")
    uri = d["ModelArtifacts"]["S3ModelArtifacts"]
    bucket, key = uri[5:].split("/", 1)
    body = session.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
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
