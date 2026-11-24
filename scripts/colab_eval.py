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
  5. for every video set in evals/videos.json not on the repo yet: runs scripts/make_videos.py (real | model | ...
     on the 10 test songs, with music) and uploads the mp4 files to the set's "out" folder
Run the cell again whenever new entries are queued; finished entries are skipped. Nothing runs on a local machine.

A runtime reset (changing the runtime type, a disconnect, the idle timeout) empties /content. So the data (one tar
file, ~150 MB) and the downloaded checkpoints are also kept on Google Drive (MyDrive/beat2body_cache; Colab asks once
per session for Drive access) and restored from there instead of Hugging Face. Set USE_DRIVE = False to skip Drive.
"""
import json
import os
import subprocess
import sys

REPO_URL = "https://github.com/novaz03/beat2body"
BRANCH = "claude/relaxed-knuth-fc6tqf"
REPO = os.environ.get("BEAT2BODY_REPO", "/content/beat2body")
HF_REPO = "SizheZ03/beat2body-runs"
USE_DRIVE = True
CACHE = "/content/drive/MyDrive/beat2body_cache"
WORKER = ""            # your name (e.g. "alice"): marks the entries you are running, so teammates skip them
CLAIM_HOURS = 3        # a claim older than this (a crashed or closed session) is taken over by the next person
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


# 0. a GPU runtime is required (Runtime -> Change runtime type -> GPU)
try:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No GPU in this Colab session. Runtime -> Change runtime type -> Hardware accelerator: GPU "
                         "-> Save, then run this cell again.")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
except ImportError:
    pass

# 1. code
if not os.path.exists(REPO):
    sh(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO_URL, REPO],
       cwd="/content")
    sh(["git", "sparse-checkout", "set", "--no-cone", "/*", "!/2d/raw_video/"])
else:
    sh(["git", "pull", "-q", "--ff-only"])
INSTALLED = "/content/.beat2body_installed"            # editable install: a git pull is enough afterwards
if not os.path.exists(INSTALLED):
    sh([sys.executable, "-m", "pip", "-q", "install", "-e", ".[train]", "huggingface_hub"])
    open(INSTALLED, "w").close()
else:
    print("package already installed in this runtime", flush=True)

# 2. data (public dataset): this runtime, else the Drive copy, else Hugging Face (then saved to Drive)
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        os.makedirs(CACHE, exist_ok=True)
    except Exception as err:                                  # noqa: BLE001  (not on Colab, or access refused)
        print("Drive not available, no cache:", err, flush=True)
        USE_DRIVE = False
DATA_TAR = os.path.join(CACHE, "data_dwpose.tar")
if os.path.exists(os.path.join(REPO, "2d", "clips", "dwpose", "meta.csv")):
    print("data already in this runtime", flush=True)
elif USE_DRIVE and os.path.exists(DATA_TAR):
    print("restoring the data from Drive", flush=True)
    sh(["tar", "-xf", DATA_TAR, "-C", REPO])
else:
    sh([sys.executable, "-m", "beat2body", "hf", "download", "--tool", "dwpose", "--parts", "keypoints", "features",
        "meta"])
    if USE_DRIVE:
        sh(["tar", "-cf", DATA_TAR, "2d/categories.json", "2d/keypoints/dwpose", "2d/features", "2d/clips/dwpose"])
        print("saved the data to Drive:", DATA_TAR, flush=True)

# 3. queue
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

tok = token()
if not tok:
    raise SystemExit("add a Colab secret HF_TOKEN (write access to SizheZ03/beat2body-runs)")
api = HfApi(token=tok)
# heel / toe stance labels (7 MB, private runs repo): the contact-head evaluation of stage-C models uses them
if not os.path.isdir(os.path.join(REPO, "2d", "features", "contact")):
    from huggingface_hub import snapshot_download  # noqa: E402
    snapshot_download(HF_REPO, allow_patterns=["2d/features/contact/*"], token=tok, local_dir=REPO)
import time  # noqa: E402
import uuid  # noqa: E402

ME = (WORKER or os.environ.get("BEAT2BODY_WORKER") or "unnamed") + "-" + uuid.uuid4().hex[:4]


def claim(path):
    """Several people can run this cell at once: before starting an entry, write <path> = {worker, time} to the
    repo. False (skip the entry) if someone else's claim younger than CLAIM_HOURS is there, or the result
    appeared meanwhile."""
    files = set(api.list_repo_files(HF_REPO))
    if path.endswith(".claim") and path[:-len(".claim")] + ".json" in files:
        print("   finished by someone else meanwhile: skipped", flush=True)
        return False
    if path in files:
        try:
            c = json.load(open(hf_hub_download(HF_REPO, path, token=tok, force_download=True)))
            if c.get("worker") != ME and time.time() - c.get("time", 0) < CLAIM_HOURS * 3600:
                print(f"   {c.get('worker')} is running it (claimed {(time.time() - c['time']) / 60:.0f} min ago):"
                      " skipped", flush=True)
                return False
        except Exception:                                     # noqa: BLE001  (unreadable claim: take it)
            pass
    api.upload_file(path_or_fileobj=json.dumps({"worker": ME, "time": time.time()}).encode(), path_in_repo=path,
                    repo_id=HF_REPO, commit_message=f"{ME} claims {path}")
    return True


def release(path):
    try:
        api.delete_file(path, repo_id=HF_REPO, commit_message=f"{ME} releases {path}")
    except Exception:                                         # noqa: BLE001
        pass


print(f"worker: {ME}", flush=True)
queue = json.load(open(hf_hub_download(HF_REPO, "evals/queue.json", token=tok, force_download=True)))
have = set(api.list_repo_files(HF_REPO))
todo = [e for e in queue if f"runs/{e['job']}/eval_test_{e['tag']}.json" not in have]
print(f"\n{len(queue)} queued evaluations, {len(todo)} to run: {[e['name'] + ' ' + e['tag'] for e in todo]}\n")

# 4. evaluate and upload (a failed evaluation uploads its log as eval_test_<tag>.failed.log, for diagnosis)
ok, failed = [], []
for e in todo:
    lock = f"runs/{e['job']}/eval_test_{e['tag']}.claim"
    print(f"\n-- {e['name']} {e['tag']}", flush=True)
    if not claim(lock):
        continue
    rel = f"runs/{e['job']}/{e.get('checkpoint', 'model.pt')}"
    root = CACHE if USE_DRIVE else "/content/hf"              # checkpoints persist on Drive across resets
    ck = os.path.join(root, rel)
    if os.path.exists(ck):
        print(f"checkpoint already here: {ck}", flush=True)
    else:
        ck = hf_hub_download(HF_REPO, rel, token=tok, local_dir=root)
    log = os.path.join(os.path.dirname(ck), f"eval_test_{e['tag']}.log")
    rc = sh([sys.executable, "-m", "beat2body", "evaluate", "--checkpoint", ck, "--tag", e["tag"], *COMMON,
             *e.get("args", [])], check=False, log=log)
    out = os.path.join(os.path.dirname(ck), f"eval_test_{e['tag']}.json")
    if rc != 0 or not os.path.exists(out):
        print(f"!! {e['name']} {e['tag']} failed (see {log})", flush=True)
        failed.append(f"{e['name']} {e['tag']}")
        try:
            api.upload_file(path_or_fileobj=log, repo_id=HF_REPO,
                            path_in_repo=f"runs/{e['job']}/eval_test_{e['tag']}.failed.log",
                            commit_message=f"Failed evaluation log {e['name']} {e['tag']} (Colab)")
        except Exception as err:                              # noqa: BLE001
            print("   could not upload the log:", err, flush=True)
        release(lock)
        continue
    for f in (out, log):
        api.upload_file(path_or_fileobj=f, path_in_repo=f"runs/{e['job']}/{os.path.basename(f)}", repo_id=HF_REPO,
                        commit_message=f"Evaluation {e['name']} {e['tag']} (Colab, {ME})")
    release(lock)
    print(f"== uploaded {e['name']} {e['tag']}", flush=True)
    ok.append(f"{e['name']} {e['tag']}")
print(f"\nfinished: {len(ok)} uploaded {ok}; {len(failed)} FAILED {failed}")

# 5. videos: evals/videos.json lists {"name": "6.x", "out": "reports/<folder>/videos",
#    "models": {"6.1": "runs/<job>/model.pt", ...}}; done when <out> on Hugging Face holds the 10 test-song videos.
#    The music comes from the 10 raw test videos kept in media/test_videos/ (Colab has ffmpeg, the Mac may not).
try:
    vqueue = json.load(open(hf_hub_download(HF_REPO, "evals/videos.json", token=tok, force_download=True)))
except Exception:                                             # noqa: BLE001  (no video queue)
    vqueue = []
vtodo = [v for v in vqueue if sum(f.startswith(v["out"] + "/") and f.endswith(".mp4") for f in have) < 10]
print(f"\n{len(vqueue)} queued video sets, {len(vtodo)} to make: {[v['name'] for v in vtodo]}\n")
for v in vtodo:
    vlock = v["out"] + "/.claim"
    print(f"\n-- videos {v['name']}", flush=True)
    if not claim(vlock):
        continue
    from huggingface_hub import snapshot_download  # noqa: E402
    snapshot_download(HF_REPO, allow_patterns=["media/test_videos/*"], token=tok, local_dir="/content/hf_media")
    src = "/content/hf_media/media/test_videos"
    for cat in os.listdir(src):
        os.makedirs(os.path.join(REPO, "2d", "raw_video", cat), exist_ok=True)
        for f in os.listdir(os.path.join(src, cat)):
            dst = os.path.join(REPO, "2d", "raw_video", cat, f)
            if not os.path.exists(dst):
                os.symlink(os.path.join(src, cat, f), dst)
    args = []
    for label, rel in v["models"].items():
        root = CACHE if USE_DRIVE else "/content/hf"
        ck = os.path.join(root, rel)
        if not os.path.exists(ck):
            ck = hf_hub_download(HF_REPO, rel, token=tok, local_dir=root)
        args += ["--model", f"{label}={ck}"]
    out = os.path.join(REPO, v["out"])
    rc = sh([sys.executable, "scripts/make_videos.py", "--out", out, "--device", "cuda", *args], check=False)
    if rc != 0:
        print(f"!! videos {v['name']} failed", flush=True)
        release(vlock)
        continue
    api.upload_folder(folder_path=out, path_in_repo=v["out"], repo_id=HF_REPO, allow_patterns=["*.mp4"],
                      commit_message=f"Videos {v['name']} (Colab, {ME})")
    release(vlock)
    print(f"== uploaded videos {v['name']} -> {v['out']}", flush=True)
