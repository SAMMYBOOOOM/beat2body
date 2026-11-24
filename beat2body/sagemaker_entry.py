"""Entry point inside a SageMaker training job (copied to the root of the uploaded source).

Training (default): unpacks the data archive from the "training" channel and runs `beat2body train` with
the job's hyperparameters (keys are train options without the leading dashes, e.g. {"model": "diffusion"}).

Pose (hyperparameter task=pose, shard=i/n): DWPose on the GPU for every n-th video (offset i) of
pose_videos.txt (AIST video ids, shipped with the source). Videos are downloaded from AIST; each raw
output goes to the checkpoint folder as soon as it is done (synced to S3, so a restarted spot job resumes)
and to the model archive at the end.
"""
import json
import os
import shutil
import sys
import tarfile

CHANNEL = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
HPS_FILE = os.path.join(os.environ.get("SM_INPUT_CONFIG_DIR", "/opt/ml/input/config"), "hyperparameters.json")
DATA = "/tmp/beat2body_data"


def hyperparameters():
    with open(HPS_FILE) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if k.startswith("sagemaker_") or k.startswith("_"):
            continue
        try:
            v = json.loads(v) if isinstance(v, str) else v
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def pose_main(hps):
    import subprocess
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from beat2body import aist
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu[cuda,cudnn]", "huggingface_hub",
                    "opencv-python-headless"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "easy-dwpose"], check=True)
    if not shutil.which("ffmpeg"):           # frames are decoded with the ffmpeg binary
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg"], check=True)
        import imageio_ffmpeg
        bin_dir = "/tmp/beat2body_bin"
        os.makedirs(bin_dir, exist_ok=True)
        if not os.path.exists(os.path.join(bin_dir, "ffmpeg")):
            os.symlink(imageio_ffmpeg.get_ffmpeg_exe(), os.path.join(bin_dir, "ffmpeg"))
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
    print(f"[pose] ffmpeg: {shutil.which('ffmpeg')}", flush=True)
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        sys.exit(f"onnxruntime has no CUDA provider ({ort.get_available_providers()}); stopping instead of "
                 "running DWPose on the CPU")
    i, n = (int(x) for x in str(hps.get("shard", "0/1")).split("/"))
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "pose_videos.txt")) as f:
        ids = [x.strip() for x in f if x.strip()][i::n]
    root = "/tmp/beat2body_pose/2d"
    ckpt = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
    backup = os.path.join(ckpt, "keypoints")
    os.makedirs(backup, exist_ok=True)
    done = {os.path.basename(p)[:-len("_dwpose.npz")] for _, _, fs in os.walk(backup) for p in fs
            if p.endswith("_dwpose.npz")}
    todo = [v for v in ids if v not in done]
    print(f"[pose] shard {i}/{n}: {len(ids)} videos, {len(done & set(ids))} already done", flush=True)

    def fetch(v):
        d = os.path.join(root, "raw_video", aist.GENRES[v[1:3]])
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, v + ".mp4")
        given = os.path.join(CHANNEL, aist.GENRES[v[1:3]], v + ".mp4")     # aws pose --upload-videos
        if os.path.exists(given):
            shutil.copy(given, out)
            return out
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(aist.VIDEO_URL.format(video_id=v), out + ".part")
                os.replace(out + ".part", out)
                return out
            except Exception as e:                       # noqa: BLE001  (retry any network error)
                print(f"[pose] download {v} failed ({e}), retry {attempt + 1}", flush=True)
        return None
    with ThreadPoolExecutor(8) as ex:
        videos = [p for p in ex.map(fetch, todo) if p]
    print(f"[pose] got {len(videos)} of {len(todo)} videos", flush=True)
    if todo and not videos:
        sys.exit("[pose] no video could be read or downloaded; failing the job")
    from beat2body.cli import main as cli_main
    if videos:
        cli_main(["--data-root", root, "pose", "dwpose", "--device", "cuda", "--videos", *videos,
                  "--backup-dir", backup])
    out = os.path.join(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"), "keypoints")
    shutil.copytree(backup, out, dirs_exist_ok=True)
    print(f"[pose] done: {sum(len(fs) for _, _, fs in os.walk(out))} raw outputs in the model archive", flush=True)


def main():
    hps = hyperparameters()
    if hps.get("task") == "pose":
        return pose_main(hps)
    archives = [f for f in os.listdir(CHANNEL) if f.endswith((".tar", ".tar.gz"))]
    if len(archives) != 1:
        sys.exit(f"expected one data archive in {CHANNEL}, found {archives}")
    os.makedirs(DATA, exist_ok=True)
    with tarfile.open(os.path.join(CHANNEL, archives[0])) as t:
        t.extractall(DATA, filter="data")
    argv = ["--data-root", os.path.join(DATA, "2d"), "train"]
    for k, v in hps.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            argv += [flag] if v else []
        else:
            argv += [flag, str(v)]
    print("[entry] python -m beat2body " + " ".join(argv), flush=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from beat2body.cli import main as cli_main
    cli_main(argv)


if __name__ == "__main__":
    main()
