"""beat2body on Google Colab: paste this whole file into ONE Colab cell and run it.

Runtime -> Change runtime type -> GPU first. The script:
  1. clones (or updates) the code, without the 2.1 GB of raw videos, and installs it
  2. downloads the training data from Hugging Face (public, no login, ~120 MB)
  3. mounts Google Drive; results and resumable checkpoints go to MyDrive/beat2body_runs/<RUN_NAME>
  4. trains (rerun the cell after a disconnect: it resumes from the last checkpoint)
  5. evaluates on the val songs (realism, energy, foot contact, drift detector) and renders one dance

Edit the settings below if needed. Mixed precision (--amp) is switched on automatically on GPUs that
support bfloat16 (A100, L4, ...) and off on a T4.
"""
import os
import subprocess
import sys

# ----------------------------------------------------------------------------- settings
RUN_NAME = "small_bone03"          # folder under MyDrive/beat2body_runs/
# small model (7.1M parameters); use [] for the full EDGE size (46.9M)
MODEL_ARGS = ["--d-model", "256", "--ff", "512", "--layers", "4", "--heads", "4"]
STEPS = 20000                      # small: 20000 (~45-60 min on a T4); full EDGE size: 66000
BONE_WEIGHT = 0.3                  # projected bone-length loss (0 = off); AWS runs the 1.0 version
EXTRA_ARGS = ["--val-every", "2000", "--sample-every", "2000"]
DEMO_VIDEO = "gLH_sFM_c01_d16_mLH4_ch05"   # a held-out test song (mLH4) to render at the end
REPO_URL = "https://github.com/novaz03/beat2body"
BRANCH = "claude/relaxed-knuth-fc6tqf"
REPO = "/content/beat2body"
# -----------------------------------------------------------------------------


def sh(cmd, cwd=REPO, check=True):
    """Run a command and stream its output into the cell."""
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0 and check:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}")


# 1. code
if not os.path.isdir(REPO):
    sh(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse",
        REPO_URL, REPO], cwd="/content")
    sh(["git", "sparse-checkout", "set", "--no-cone", "/*", "!/2d/raw_video/"])
else:
    sh(["git", "pull", "-q", "--depth", "1", "origin", BRANCH])
sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".[train]"])

# 2. data
if not os.path.exists(f"{REPO}/2d/clips/dwpose/meta.csv"):
    sh([sys.executable, "-m", "beat2body", "hf", "download", "--tool", "dwpose",
        "--parts", "keypoints", "features", "meta"])

# 3. Drive
from google.colab import drive  # noqa: E402

drive.mount("/content/drive")
RUN = f"/content/drive/MyDrive/beat2body_runs/{RUN_NAME}"
os.makedirs(RUN, exist_ok=True)

# 4. train (resumes from RUN/ckpt/last.pt if present)
import torch  # noqa: E402

amp = ["--amp"] if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else []
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU!)", "| amp:", bool(amp))
if not os.path.exists(f"{RUN}/model.pt"):
    sh([sys.executable, "-m", "beat2body", "train", *MODEL_ARGS, "--steps", str(STEPS),
        "--bone-weight", str(BONE_WEIGHT), *EXTRA_ARGS, *amp, "--out", RUN, "--checkpoint-dir", f"{RUN}/ckpt"])
else:
    print(f"{RUN}/model.pt exists: training already finished (delete it to train again)")

# 5. evaluate + one rendered dance
sh([sys.executable, "-m", "beat2body", "evaluate", "--checkpoint", f"{RUN}/model.pt", "--device", "cuda"])
# shareable videos: training progress (same clip at each check) + 20 s dances, one per genre (silent on Colab)
sh([sys.executable, "-m", "beat2body", "report", RUN, "--device", "cuda"])
sh([sys.executable, "-m", "beat2body", "sample", "--checkpoint", f"{RUN}/model.pt", "--video-id", DEMO_VIDEO,
    "--seconds", "20", "--device", "cuda", "-o", f"{RUN}/generated/{DEMO_VIDEO}.npz"])
sh([sys.executable, "-m", "beat2body", "render", "--video-id", DEMO_VIDEO,
    "--keypoints", f"{RUN}/generated/{DEMO_VIDEO}.npz", "--labels", "real dancer", "generated",
    "--frames", "600", "-o", f"{RUN}/{DEMO_VIDEO}.mp4"])
print(f"\nDone. Results in Google Drive: MyDrive/beat2body_runs/{RUN_NAME}/ "
      f"(model.pt, model_raw.pt, snapshots/, metrics.jsonl, eval_val.json, report/*.mp4, {DEMO_VIDEO}.mp4)")
