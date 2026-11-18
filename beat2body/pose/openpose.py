"""OpenPose, two backends.

  --backend official : the official CMU OpenPose (C++, everything on the GPU), BODY_25 model.
                       Build it first; on Colab: bash scripts/colab_build_openpose.sh.
                       The published OpenPose dataset was made with this backend.
  --backend pytorch  : (default) PyTorch port of the BODY-18 model, no build needed, but the
                       post-processing runs on the CPU and is slow on 1080p video. Its source
                       files (huggingface/controlnet_aux) are fetched into ~/.cache/beat2body/ and
                       the weights come from the public Hugging Face repo lllyasviel/Annotators.

License: OpenPose models and code are for academic / non-commercial research use only.

Install (pytorch backend):
    pip install torch scipy opencv-python huggingface_hub

    python -m beat2body pose openpose --backend official --openpose-dir /content/openpose
"""
import glob
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np

from beat2body import paths
from beat2body.pose.runner import (
    add_common_args,
    backup,
    pending_videos,
    report,
    run_pose_tool,
    save_raw_keypoints,
    wants_preview,
)
from beat2body.skeleton import BODY25_NAMES, BODY25_TO_COCO17, CONF_THRESHOLD, OPENPOSE18_NAMES, OPENPOSE18_TO_COCO17
from beat2body.video import PreviewWriter, iter_frames, probe_video, scaled_size

TOOL = "openpose"
DOC = __doc__
COLOR = (255, 80, 80)
CACHE = os.path.expanduser("~/.cache/beat2body")
PKG = "openpose_body"
SRC_COMMIT = "b1aac318b15c4e65a21bc516dc236ad4bf593d46"
SRC_URL = ("https://raw.githubusercontent.com/huggingface/controlnet_aux/"
           f"{SRC_COMMIT}/src/controlnet_aux/open_pose/{{f}}")


def add_args(p):
    add_common_args(p, TOOL)
    p.add_argument("--backend", default="pytorch", choices=["pytorch", "official"],
                   help="pytorch: PyTorch port, network on the GPU and post-processing on the CPU; "
                        "official: the official CMU OpenPose binary (C++, GPU), BODY_25 model")
    p.add_argument("--openpose-dir", default=os.environ.get("OPENPOSE_DIR", "/content/openpose"),
                   help="official backend: OpenPose source folder with build/ and models/")
    p.add_argument("--net-resolution", default="-1x368", help="official backend: network input size")
    p.add_argument("--device", default="auto", help="pytorch backend: auto | cuda | cpu")
    p.add_argument("--scales", nargs="+", type=float, default=[1.0],
                   help="pytorch backend: inference scales relative to a 368 px tall input "
                        "(multi-scale such as 0.5 1.0 1.5 2.0 is more accurate but slower)")


def load_body_class(scales):
    """Fetch body.py / model.py / util.py once and import them as a small package."""
    pkg_dir = os.path.join(CACHE, PKG)
    os.makedirs(pkg_dir, exist_ok=True)
    for f in ["body.py", "model.py", "util.py"]:
        dst = os.path.join(pkg_dir, f)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(SRC_URL.format(f=f), dst)
    open(os.path.join(pkg_dir, "__init__.py"), "w").close()
    body = os.path.join(pkg_dir, "body.py")
    src = open(body).read()
    # scipy.ndimage.filters is deprecated; inference scale is made configurable.
    src = src.replace("from scipy.ndimage.filters import gaussian_filter", "from scipy.ndimage import gaussian_filter")
    src = src.replace("scale_search = [0.5]", "scale_search = SCALES")
    if "SCALES = " not in src:
        src = src.replace("from . import util", "from . import util\n\nSCALES = [1.0]", 1)
    open(body, "w").write(src)
    sys.path.insert(0, CACHE)
    # util.py and body.py import each other; util must be imported first (as in controlnet_aux).
    importlib.import_module(f"{PKG}.util")
    mod = importlib.import_module(f"{PKG}.body")
    mod.SCALES = list(scales)
    return mod.Body


def run_official(args):
    """Each video is resampled to the fixed fps (same frames as the other tools), then
    openpose.bin writes one JSON per frame, which is converted to the shared format."""
    root = os.path.abspath(args.openpose_dir)
    binary = os.path.join(root, "build", "examples", "openpose", "openpose.bin")
    if not os.path.exists(binary):
        sys.exit(f"official OpenPose not found at {binary}; build it first or use --backend pytorch")
    print(f"[{TOOL}] running the official OpenPose (BODY_25) from {root}")

    for n, cat, vid, path in pending_videos(TOOL, args):
        w, h, _ = probe_video(path)
        w, h = scaled_size(w, h, args.height)
        tmp = tempfile.mkdtemp(prefix="openpose_")
        try:
            clip = os.path.join(tmp, "clip.mp4")
            vf = f"fps={args.fps}" + (f",scale={w}:{h}" if args.height else "")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vf", vf, "-an",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", clip], check=True)
            json_dir = os.path.join(tmp, "json")
            t0 = time.perf_counter()
            subprocess.run([binary, "--video", clip, "--write_json", json_dir, "--display", "0",
                            "--render_pose", "0", "--model_pose", "BODY_25", "--number_people_max", "1",
                            "--net_resolution", args.net_resolution,
                            "--model_folder", os.path.join(root, "models", "")],
                           check=True, cwd=root, stdout=subprocess.DEVNULL)
            dt = time.perf_counter() - t0
            native = []
            for f in sorted(glob.glob(os.path.join(json_dir, "*_keypoints.json"))):
                with open(f) as fh:
                    people = json.load(fh)["people"]
                if not people:
                    native.append(np.full((25, 3), np.nan, np.float32))
                    continue
                k = np.array(people[0]["pose_keypoints_2d"], np.float32).reshape(25, 3)
                k[k[:, 2] <= 0] = np.nan  # OpenPose writes 0, 0, 0 for joints it did not find
                native.append(k)
            native = np.stack(native)
            coco = native[:, BODY25_TO_COCO17]
            if wants_preview(args, n):
                preview = PreviewWriter(paths.preview_path(TOOL, cat, vid), args.fps, w, h, f"{TOOL}  {cat}/{vid}")
                for i, frame in enumerate(iter_frames(clip, args.fps)):
                    if i < len(coco):
                        preview.write_frame(frame, coco[i], CONF_THRESHOLD[TOOL], COLOR)
                preview.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        out = paths.raw_keypoint_path(TOOL, cat, vid)
        save_raw_keypoints(out, coco, native, BODY25_NAMES, TOOL, args.fps, w, h, dt)
        backup(out, args.backup_dir)
        report(TOOL, cat, vid, coco, dt)


def run(args):
    if args.backend == "official":
        run_official(args)
        return

    import torch
    from huggingface_hub import hf_hub_download

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    Body = load_body_class(args.scales)
    model = Body(hf_hub_download("lllyasviel/Annotators", "body_pose_model.pth")).to(device)
    print(f"[{TOOL}] running on {device}, scales {args.scales}")

    def make_infer(fps):
        def infer(frame_bgr):
            candidate, subset = model(frame_bgr)  # expects BGR
            if len(subset) == 0:
                return None, None
            best = int(np.argmax(subset[:, 18]))  # person with the highest total score
            native = np.full((18, 3), np.nan, np.float32)
            for j, ci in enumerate(subset[best, :18].astype(int)):
                if ci != -1:
                    native[j] = candidate[ci, :3]
            return native[OPENPOSE18_TO_COCO17], native
        return infer

    run_pose_tool(TOOL, args, make_infer, OPENPOSE18_NAMES, color=COLOR)
