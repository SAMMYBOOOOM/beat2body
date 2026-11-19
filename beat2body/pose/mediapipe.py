"""MediaPipe Pose Landmarker (GPU if available, else CPU).

Install:
    pip install mediapipe opencv-python

The model file (pose_landmarker_heavy.task, ~30 MB) is downloaded from Google
on first run to ~/.cache/beat2body/. Use --model full or lite for a faster,
less accurate variant. On Colab use --device cpu (no OpenGL for the GPU path).

    python -m beat2body pose mediapipe --categories la_hiphop house --preview 2

native: (T, 33, 3) MediaPipe's 33 landmarks (x, y in pixels, visibility).
"""
import os
import urllib.request

import numpy as np

from beat2body.pose.runner import add_common_args, run_pose_tool
from beat2body.skeleton import MEDIAPIPE33_NAMES, MEDIAPIPE33_TO_COCO17

TOOL = "mediapipe"
DOC = __doc__
CACHE = os.path.expanduser("~/.cache/beat2body")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")


def add_args(p):
    add_common_args(p, TOOL)
    p.add_argument("--model", default="heavy", choices=["lite", "full", "heavy"])
    p.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"],
                   help="auto: try the GPU and fall back to the CPU if MediaPipe cannot use it")


def model_path(variant):
    path = os.path.join(CACHE, f"pose_landmarker_{variant}.task")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        print(f"[{TOOL}] downloading {variant} model")
        urllib.request.urlretrieve(MODEL_URL.format(v=variant), path)
    return path


def run(args):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    path = model_path(args.model)
    delegates = {"gpu": mp_python.BaseOptions.Delegate.GPU, "cpu": mp_python.BaseOptions.Delegate.CPU}

    def gpu_works():
        """MediaPipe's GPU support depends on the platform and build; test it once on a blank image."""
        try:
            probe = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=path, delegate=delegates["gpu"]),
                running_mode=vision.RunningMode.IMAGE, num_poses=1))
            probe.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((256, 256, 3), np.uint8)))
            probe.close()
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[{TOOL}] GPU not available in MediaPipe ({str(e).splitlines()[0][:120]})")
            return False

    device = args.device
    if device in ("auto", "gpu"):
        device = "gpu" if gpu_works() else "cpu"
        if args.device == "gpu" and device == "cpu":
            print(f"[{TOOL}] warning: --device gpu was requested; using the CPU")
    print(f"[{TOOL}] running on {device}")

    def make_infer(fps):
        # VIDEO mode tracks the person across frames and needs strictly increasing
        # timestamps, so a fresh landmarker is created for every video.
        landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path, delegate=delegates[device]),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))
        state = {"frame": 0}

        def infer(frame_bgr):
            ts = int(round(state["frame"] * 1000.0 / fps))
            state["frame"] += 1
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            if not res.pose_landmarks:
                return None, None
            h, w = frame_bgr.shape[:2]
            native = np.array([[lm.x * w, lm.y * h, 1.0 if lm.visibility is None else lm.visibility]
                               for lm in res.pose_landmarks[0]], np.float32)
            return native[MEDIAPIPE33_TO_COCO17], native
        return infer

    run_pose_tool(TOOL, args, make_infer, MEDIAPIPE33_NAMES, color=(0, 200, 255))
