"""beat2body on Google Colab: paste this whole file into ONE Colab cell and run it.

Runtime -> Change runtime type -> GPU first. The script:
  1. clones (or updates) the code, without the 2.1 GB of raw videos, and installs it
  2. downloads the training data from Hugging Face (public, no login, ~120 MB)
  3. mounts Google Drive; every run gets its own folder MyDrive/beat2body_runs/<name>
  4. trains all RUNS at the same time on the one GPU (a small model uses ~2 GB of a T4's 15 GB), printing
     a progress table every minute; each run's full log is in <name>/train.log
  5. evaluates each run (realism, energy, foot contact, drift detector) and writes shareable videos
     (<name>/report/: training progress + 20 s dances)

If Colab disconnects, run the cell again: finished runs are skipped and unfinished ones resume from
their last checkpoint. Mixed precision (--amp) is on: float16 on a T4, bfloat16 on an A100 / L4.
"""
import json
import os
import subprocess
import sys
import time

# ----------------------------------------------------------------------------- settings
SMALL = ["--d-model", "256", "--ff", "512", "--layers", "4", "--heads", "4"]   # 7.1M parameters
COMMON = ["--steps", "20000", "--val-every", "2000", "--sample-every", "2000", "--amp"]
# one entry per run, trained in parallel; [] instead of SMALL = full EDGE size (46.9M).
# --recipe regularized (the default): dropout 0.2, weight decay 0.1, cosine LR, 3 pose tools, time-stretch +
# jitter augmentation, Min-SNR, model.pt = best val FID_k. --recipe edge: EDGE's original settings.
# Energy fixes (docs/pipeline_2d.md "Energy"): --normalizer zscore (fine motion stands out from the noise),
# --param v (skip path at low noise), --energy-weight (phase-invariant speed / amplitude loss).
FIX = ["--normalizer", "zscore", "--param", "v"]
RUNS = {
    "reg_small": SMALL + COMMON,                                         # control: regularized recipe only
    "fix_small": SMALL + COMMON + FIX,                                   # + z-score + v-prediction
    "fix_energy_small": SMALL + COMMON + FIX + ["--energy-weight", "0.25"],   # + energy loss
}
# Larger model: EDGE size (46.9M) with every fix. ~4x slower per step; on a T4 next to the small runs it
# takes most of a session, so only uncomment it on an A100 / L4 runtime (or run it on AWS g6.xlarge).
# RUNS["fix_energy_edge"] = COMMON + FIX + ["--energy-weight", "0.25"]
REPO_URL = "https://github.com/novaz03/beat2body"
BRANCH = "claude/relaxed-knuth-fc6tqf"
REPO = os.environ.get("BEAT2BODY_REPO", "/content/beat2body")
POLL_SECONDS = 60
# -----------------------------------------------------------------------------
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:      # local dry run: use an existing checkout, write runs under it
    IN_COLAB = False


def sh(cmd, cwd=REPO, check=True):
    """Run a command and stream its output into the cell."""
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0 and check:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}")


# 1. code
if IN_COLAB:
    if not os.path.isdir(REPO):
        sh(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse",
            REPO_URL, REPO], cwd="/content")
        sh(["git", "sparse-checkout", "set", "--no-cone", "/*", "!/2d/raw_video/"])
    else:
        sh(["git", "pull", "-q", "--depth", "1", "origin", BRANCH])
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".[train]"])

# 2. data (+ the other pose tools' keypoints, used as extra training views by the regularized recipe)
if not os.path.exists(f"{REPO}/2d/clips/dwpose/meta.csv"):
    sh([sys.executable, "-m", "beat2body", "hf", "download", "--tool", "dwpose",
        "--parts", "keypoints", "features", "meta"])
for tool in ("openpose", "mediapipe"):
    if not os.path.isdir(f"{REPO}/2d/keypoints/{tool}"):
        sh([sys.executable, "-m", "beat2body", "hf", "download", "--tool", tool, "--parts", "keypoints"])

# 3. Drive
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = "/content/drive/MyDrive/beat2body_runs"
else:
    ROOT = os.environ.get("BEAT2BODY_RUNS", f"{REPO}/runs/colab")
os.makedirs(ROOT, exist_ok=True)

import torch  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("GPU:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "none", "| runs:", ", ".join(RUNS))


# 4. train all runs in parallel
def last_metrics(run_dir):
    """Latest training line, val loss and sample check from <run>/metrics.jsonl."""
    out = {}
    path = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "loss" in m:
                    out["train"] = m
                if "val_loss" in m:
                    out["val"] = m
                if "sample" in m:
                    out["sample"] = m
    return out


procs = {}
for name, args in RUNS.items():
    run_dir = os.path.join(ROOT, name)
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, "model.pt")):
        print(f"[{name}] already finished")
        continue
    log = open(os.path.join(run_dir, "train.log"), "a")
    cmd = [sys.executable, "-m", "beat2body", "train", *args, "--out", run_dir,
           "--checkpoint-dir", os.path.join(run_dir, "ckpt")]
    if DEVICE != "cuda":
        cmd = [c for c in cmd if c != "--amp"]
    procs[name] = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    print(f"[{name}] started: {' '.join(cmd[3:])}")

while procs:
    time.sleep(POLL_SECONDS)
    print(f"\n{time.strftime('%H:%M:%S')}  {'run':16s} {'step':>13s} {'loss':>7s} {'steps/s':>8s} "
          f"{'val_loss':>9s} {'speed':>6s} {'beat_pw':>8s} {'contact':>8s} {'bone_cv':>8s} {'(real)':>7s}")
    for name in list(procs):
        m = last_metrics(os.path.join(ROOT, name))
        t, v, s = m.get("train", {}), m.get("val", {}), m.get("sample", {}).get("sample", {})
        steps = next((RUNS[name][i + 1] for i, a in enumerate(RUNS[name]) if a == "--steps"), "?")
        fmt = lambda d, k, f: format(d[k], f) if k in d else "-"  # noqa: E731
        print(f"{'':10s}{name:16s} {str(t.get('step', 0)) + '/' + str(steps):>13s} {fmt(t, 'loss', '7.3f')} "
              f"{fmt(t, 'steps_per_s', '8.2f')} {fmt(v, 'val_loss', '9.3f')} {fmt(s, 'speed_ratio', '6.2f')} "
              f"{fmt(s, 'beat_power', '8.2f')} {fmt(s, 'contact_rate', '8.2f')} "
              f"{fmt(s, 'bone_cv', '8.3f')} {fmt(s, 'bone_cv_real', '7.3f')}")
        code = procs[name].poll()
        if code is not None:
            print(f"[{name}] {'finished' if code == 0 else f'FAILED (exit {code})'}")
            if code != 0:
                with open(os.path.join(ROOT, name, "train.log")) as f:
                    print("".join(f.readlines()[-20:]))
            del procs[name]

# 5. evaluate + shareable videos for every finished run
for name in RUNS:
    run_dir = os.path.join(ROOT, name)
    if not os.path.exists(os.path.join(run_dir, "model.pt")):
        continue
    if not os.path.exists(os.path.join(run_dir, "eval_val.json")):
        sh([sys.executable, "-m", "beat2body", "evaluate", "--checkpoint", f"{run_dir}/model.pt", "--device", DEVICE])
    if not os.path.isdir(os.path.join(run_dir, "report")):
        sh([sys.executable, "-m", "beat2body", "report", run_dir, "--device", DEVICE])
print(f"\nDone. Results in {ROOT}/<run>/: model.pt, metrics.jsonl, train.log, eval_val.json, report/*.mp4")
