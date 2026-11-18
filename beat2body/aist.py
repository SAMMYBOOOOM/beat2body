"""AIST Dance Video Database naming and the list of videos / categories.

Video ids look like gLH_sFM_c01_d16_mLH0_ch01: genre LH, advanced choreography (sFM),
front camera (c01), dancer d16, song mLH0, choreography 01. Each genre has 3 dancers
and 6 songs; every dancer dances to all 6 songs of their genre.
"""
import glob
import json
import os
import re

from beat2body import paths

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# AIST genre code -> category folder name
GENRES = {
    "LH": "la_hiphop", "HO": "house", "MH": "middle_hiphop", "BR": "break", "PO": "pop",
    "LO": "lock", "WA": "waack", "KR": "krump", "JS": "street_jazz", "JB": "ballet_jazz",
}
VIDEO_URL = "https://aistdancedb.ongaaccel.jp/v1.0.0/video/2M/{video_id}.mp4"


def dancer_id(video_id):
    """'d16' for gLH_sFM_c01_d16_mLH0_ch01, None for non-AIST names."""
    m = re.search(r"_(d\d\d)_", video_id)
    return m.group(1) if m else None


def music_id(video_id):
    """'mLH0' for gLH_sFM_c01_d16_mLH0_ch01, None for non-AIST names."""
    m = re.search(r"_(m[A-Z]{2}\d)_", video_id)
    return m.group(1) if m else None


def list_categories(video_dir=None):
    """Categories are the sub-folders of raw_video/ that contain videos."""
    video_dir = video_dir or paths.raw_video_dir()
    if not os.path.isdir(video_dir):
        return []
    return [d for d in sorted(os.listdir(video_dir))
            if os.path.isdir(os.path.join(video_dir, d))
            and any(f.lower().endswith(VIDEO_EXTS) for f in os.listdir(os.path.join(video_dir, d)))]


def list_videos(categories=None, limit=None):
    """[(category, video_id, path)] of the raw videos, sorted by category and name."""
    out = []
    for c in categories or list_categories():
        files = sorted(f for f in glob.glob(os.path.join(paths.raw_video_dir(), c, "*"))
                       if f.lower().endswith(VIDEO_EXTS))
        out += [(c, os.path.splitext(os.path.basename(f))[0], f) for f in files[:limit]]
    return out


def write_categories_json(categories):
    """Fixed name -> index mapping for the category embedding (alphabetical)."""
    mapping = {c: i for i, c in enumerate(sorted(categories))}
    with open(paths.categories_json(), "w") as f:
        json.dump(mapping, f, indent=2)
        f.write("\n")
    return mapping


def load_categories():
    with open(paths.categories_json()) as f:
        return json.load(f)
