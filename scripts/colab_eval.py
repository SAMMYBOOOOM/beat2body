"""beat2body evaluations on Google Colab: paste this whole file into ONE Colab cell and run it.

Runtime -> Change runtime type -> GPU (L4 / A100 / T4). Add a Colab secret HF_TOKEN (key icon on the left): a
Hugging Face token with write access to SizheZ03/beat2body-runs. The script:
  1. clones (or updates) the code, without the raw videos, and installs it
  2. downloads the evaluation data from Hugging Face (public dataset, ~150 MB)
  3. reads the queue evals/queue.json in SizheZ03/beat2body-runs: a list of
        {"name": "6.1", "job": "<run folder>", "checkpoint": "model.pt", "tag": "feet2_raw", "args": [...]}
  4. for every entry whose result runs/<job>/eval_test_<tag>.json is not in the repo yet: downloads the checkpoint,
     runs `python -m beat2body evaluate` on the held-out test songs (seed 0, 50 DDIM steps, eta 1, guidance 2,
     20 s long dances), and uploads the result JSON and the printed log next to the checkpoint
Run the cell again whenever new entries are queued; finished entries are skipped. Nothing runs on a local machine.
"""
import json
import os
import subprocess
import sys

REPO_URL = "https://github.com/novaz03/beat2body"
BRANCH = "claude/relaxed-knuth-fc6tqf"
REPO = os.environ.get("BEAT2BODY_REPO", "/content/beat2body")
HF_REPO = "SizheZ03/beat2body-runs"
COMMON = ["--split", "test", "--subset", "all", "--seed", "0", "--steps", "50", "--eta", "1", "--guidance", "2",
          "--batch-size", "64", "--long-seconds", "20", "--device", "cuda"]


def sh(cmd, cwd=None, check=True, log=None):
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd or REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    if log:
        with open(log, "w") as f:
            f.writelines(lines)
    if p.wait() != 0 and check:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return p.returncode


def token():
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:                                       # noqa: BLE001  (local test: environment variable)
        return os.environ.get("HF_TOKEN")


# 1. code
if not os.path.exists(REPO):
    sh(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO_URL, REPO],
       cwd="/content")
    sh(["git", "sparse-checkout", "set", "--no-cone", "/*", "!/2d/raw_video/"])
else:
    sh(["git", "pull", "-q", "--ff-only"])
sh([sys.executable, "-m", "pip", "-q", "install", "-e", ".[train]", "huggingface_hub"])

# 2. data (public dataset)
if not os.path.exists(os.path.join(REPO, "2d", "clips", "dwpose", "meta.csv")):
    sh([sys.executable, "-m", "beat2body", "hf", "download", "--tool", "dwpose", "--parts", "keypoints", "features",
        "meta"])

# 3. queue
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

tok = token()
if not tok:
    raise SystemExit("add a Colab secret HF_TOKEN (write access to SizheZ03/beat2body-runs)")
api = HfApi(token=tok)
queue = json.load(open(hf_hub_download(HF_REPO, "evals/queue.json", token=tok, force_download=True)))
have = set(api.list_repo_files(HF_REPO))
todo = [e for e in queue if f"runs/{e['job']}/eval_test_{e['tag']}.json" not in have]
print(f"\n{len(queue)} queued evaluations, {len(todo)} to run: {[e['name'] + ' ' + e['tag'] for e in todo]}\n")

# 4. evaluate and upload
for e in todo:
    ck = hf_hub_download(HF_REPO, f"runs/{e['job']}/{e.get('checkpoint', 'model.pt')}", token=tok,
                         local_dir="/content/hf")
    log = os.path.join(os.path.dirname(ck), f"eval_test_{e['tag']}.log")
    rc = sh([sys.executable, "-m", "beat2body", "evaluate", "--checkpoint", ck, "--tag", e["tag"], *COMMON,
             *e.get("args", [])], check=False, log=log)
    out = os.path.join(os.path.dirname(ck), f"eval_test_{e['tag']}.json")
    if rc != 0 or not os.path.exists(out):
        print(f"!! {e['name']} {e['tag']} failed (see {log}); skipping its upload", flush=True)
        continue
    for f in (out, log):
        api.upload_file(path_or_fileobj=f, path_in_repo=f"runs/{e['job']}/{os.path.basename(f)}", repo_id=HF_REPO,
                        commit_message=f"Evaluation {e['name']} {e['tag']} (Colab)")
    print(f"== uploaded {e['name']} {e['tag']}", flush=True)
print("\nall queued evaluations done")
