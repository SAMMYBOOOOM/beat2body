"""Where everything lives. All data sits under one data root (default: <repo>/2d).

The layout under the data root is the same as in the Hugging Face datasets, whose files
all start with "2d/". Paths written to meta.csv are relative to the data root's parent
(e.g. "2d/clips/dwpose/train/..."), so they also resolve inside a Hugging Face snapshot.

Override the data root with --data-root on the command line or BEAT2BODY_DATA_ROOT.
"""
import os

TOOLS = ("dwpose", "mediapipe", "openpose")
SPLITS = ("train", "val", "test")
FEATURE_KINDS = ("audio_mel", "audio_onset", "beats")

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)
_data_root = os.path.abspath(os.environ.get("BEAT2BODY_DATA_ROOT") or os.path.join(REPO_ROOT, "2d"))


def set_data_root(path):
    global _data_root
    _data_root = os.path.abspath(path)


def data_root():
    return _data_root


def base_dir():
    """Parent of the data root; meta.csv paths and Hugging Face paths are relative to it."""
    return os.path.dirname(_data_root)


def rel(path):
    return os.path.relpath(path, base_dir())


def _d(*parts):
    return os.path.join(_data_root, *parts)


# folders
def raw_video_dir():
    return _d("raw_video")


def frames_dir():
    return _d("frames")


def keypoints_dir():
    return _d("keypoints")


def audio_dir():
    return _d("audio")


def features_dir():
    return _d("features")


def clips_dir():
    return _d("clips")


def categories_json():
    return _d("categories.json")


# files
def raw_keypoint_path(tool, category, video_id):
    """Pose tool output (4.3): keypoints/<category>/<video_id>_<tool>.npz"""
    return _d("keypoints", category, f"{video_id}_{tool}.npz")


def preview_path(tool, category, video_id):
    return _d("keypoints", category, f"{video_id}_{tool}.mp4")


def clean_keypoint_path(tool, category, video_id):
    """Cleaned training keypoints (4.4): keypoints/<tool>/<category>/<video_id>.npz"""
    return _d("keypoints", tool, category, video_id + ".npz")


def normalization_path(tool):
    return _d("keypoints", tool, "normalization.json")


def audio_path(category, video_id):
    return _d("audio", category, video_id + ".wav")


def feature_path(kind, video_id):
    """Audio features (4.5), shared by all tools: features/<kind>/<video_id>.npy"""
    return _d("features", kind, video_id + ".npy")


def tool_clips_dir(tool):
    return _d("clips", tool)


def clip_path(tool, split, category, video_id, start):
    return _d("clips", tool, split, f"{category}__{video_id}__{start:06d}.npz")


def meta_csv_path(tool):
    return _d("clips", tool, "meta.csv")


def split_json_path(tool):
    return _d("clips", tool, "split.json")


def audio_stats_path(tool):
    """Per-channel audio normalization computed on the train split (beat2body.data)."""
    return _d("clips", tool, "audio_stats.json")
