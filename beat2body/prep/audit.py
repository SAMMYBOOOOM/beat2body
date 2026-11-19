"""Audit the training data of one tool: labels, split, lengths, normalization, clips, beats.

Every check compares independent sources (file names vs labels, clips vs the video-level files
they were cut from, frame counts vs the raw videos, beat tempo vs the published AIST tempo).

    python -m beat2body audit --tool dwpose
Exits with an error if a check fails; long fast motions (breaking, kicks) are reported, not failed.
"""
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from beat2body import aist, paths
from beat2body.paths import SPLITS, TOOLS
from beat2body.skeleton import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER

DOC = __doc__
CHECKS = ["label", "split", "meta", "files", "nan", "length", "norm", "clip", "beats"]


def add_args(p):
    p.add_argument("--tool", default="dwpose", choices=TOOLS)
    p.add_argument("--no-video", action="store_true", help="skip the frame-count check against raw videos")


def _video_seconds(path):
    import cv2
    cap = cv2.VideoCapture(path)
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n / fps if fps else None


def run(args):
    tool = args.tool
    issues = defaultdict(list)
    cats = aist.load_categories()
    with open(paths.meta_csv_path(tool)) as f:
        meta = list(csv.DictReader(f))
    print(f"[audit] {tool}: {len(meta)} clips, {len(cats)} categories")

    # labels, split, metadata
    vid_split, vid_cat = {}, {}
    for r in meta:
        v, c, s = r["video_id"], r["category"], r["split"]
        code = v[1:3]
        if aist.GENRES.get(code) != c:
            issues["label"].append(f"{v}: genre code {code} but category {c}")
        if vid_split.setdefault(v, s) != s:
            issues["split"].append(f"{v} in {vid_split[v]} and {s}")
        if f"/{tool}/{s}/" not in r["clip_path"]:
            issues["split"].append(f"{r['clip_path']} not in folder {s}")
        if r["music"] != (aist.music_id(v) or "") or r["dancer"] != (aist.dancer_id(v) or ""):
            issues["meta"].append(f"{v}: music / dancer columns")
        vid_cat[v] = c
    songs = {s: {aist.music_id(v) for v, x in vid_split.items() if x == s} for s in SPLITS}
    with open(paths.split_json_path(tool)) as f:
        info = json.load(f)
    if info.get("test_songs") and songs["test"] & (songs["train"] | songs["val"]):
        issues["split"].append("a test song also appears in train / val")
    for s in SPLITS:
        n = len({vid_cat[v] for v, x in vid_split.items() if x == s})
        if n != len(cats):
            issues["split"].append(f"{s} has {n} of {len(cats)} genres")
    on_disk = set(glob.glob(os.path.join(paths.tool_clips_dir(tool), "*", "*.npz")))
    listed = {os.path.join(paths.base_dir(), r["clip_path"]) for r in meta}
    if on_disk != listed:
        issues["files"].append(f"{len(on_disk - listed)} clip files not in meta.csv, "
                               f"{len(listed - on_disk)} rows without file")

    # video-level files
    fast = Counter()
    torso_med = {}                       # video -> median torso length (dataset-normalized)
    for v, c in sorted(vid_cat.items()):
        d = np.load(paths.clean_keypoint_path(tool, c, v))
        kp, valid = d["keypoints"], d["valid"]
        mel, on = np.load(paths.feature_path("audio_mel", v)), np.load(paths.feature_path("audio_onset", v))
        T = len(kp)
        if not np.isfinite(kp).all():
            issues["nan"].append(v)
        if abs(T - len(mel)) > 3 or abs(T - len(on)) > 3:
            issues["length"].append(f"{v}: keypoints {T}, mel {len(mel)}, onset {len(on)}")
        raw = glob.glob(os.path.join(paths.raw_video_dir(), c, v + ".*"))
        if raw and not args.no_video:
            sec = _video_seconds(raw[0])
            if sec and abs(T - sec * 30) > 3:
                issues["length"].append(f"{v}: {T} frames, video {sec * 30:.0f}")
        hips = valid[:, L_HIP] & valid[:, R_HIP]
        mid_hip = (kp[:, L_HIP] + kp[:, R_HIP]) / 2
        if hips.any():
            if np.abs(mid_hip[hips]).max() > 1e-3:
                issues["norm"].append(f"{v}: mid-hip not at the origin")
            torso = np.linalg.norm((kp[:, L_SHOULDER] + kp[:, R_SHOULDER]) / 2 - mid_hip, axis=1)[hips]
            torso_med[v] = float(np.median(torso))
        fast[v[:3]] += int((np.linalg.norm(np.diff(kp, axis=0), axis=-1) > 0.6).sum())
        # beats vs published tempo
        b = np.load(paths.feature_path("beats", v)).astype(np.float64)
        bpm = aist.music_bpm(v)
        if bpm and len(b) > 2:
            est = 60 * 30 / np.median(np.diff(b))
            if abs(est - bpm) > 0.5:
                issues["beats"].append(f"{aist.music_id(v)}: {est:.1f} BPM, published {bpm}")

    # every clip equals its source slice
    same = 0
    for r in meta:
        c = np.load(os.path.join(paths.base_dir(), r["clip_path"]))
        v, s, e = r["video_id"], int(r["start_frame"]), int(r["end_frame"])
        d = np.load(paths.clean_keypoint_path(tool, r["category"], v))
        b = np.load(paths.feature_path("beats", v))
        ok = (np.array_equal(c["keypoints"], d["keypoints"][s:e]) and np.array_equal(c["valid"], d["valid"][s:e])
              and np.array_equal(c["mel"], np.load(paths.feature_path("audio_mel", v))[s:e])
              and np.array_equal(c["onset"], np.load(paths.feature_path("audio_onset", v))[s:e])
              and np.allclose(c["beats"], b[(b >= s) & (b < e)] - s)
              and str(c["category"]) == r["category"] and int(c["category_index"]) == cats[r["category"]])
        same += ok
        if not ok and len(issues["clip"]) < 5:
            issues["clip"].append(r["clip_path"])
    if same != len(meta):
        issues["clip"].append(f"{len(meta) - same} clips differ from their source")

    # body size: each dancer's median torso must be ~1 (the dataset scale); single videos may be far lower
    # (floor work, e.g. break basics, where the torso points at the camera), which is reported, not failed
    by_dancer = defaultdict(list)
    for v, t in torso_med.items():
        by_dancer[aist.dancer_id(v) or v].append(t)
    for d, ts in sorted(by_dancer.items()):
        if not 0.6 < np.median(ts) < 1.5:
            issues["norm"].append(f"dancer {d}: median torso {np.median(ts):.2f}")
    low = sorted(v for v, t in torso_med.items()
                 if t < 0.6 * np.median(by_dancer[aist.dancer_id(v) or v]))

    print(f"[audit] videos per split: {dict(Counter(vid_split.values()))}; test songs: {sorted(songs['test'])}")
    for k in CHECKS:
        n = len(issues[k])
        print(f"  {'OK  ' if n == 0 else 'FAIL'} {k:7s}" + ("" if n == 0 else f" {n}: {issues[k][:3]}"))
    print(f"  clips identical to their source: {same}/{len(meta)}")
    print(f"  info: joint moves > 0.6 torso lengths per frame (fast real moves; inspect with render): "
          f"{dict(fast.most_common(3))} ...")
    if low:
        print(f"  info: {len(low)} videos with a much shorter torso than their dancer's (floor work / foreshortening; "
              f"use --scale-norm dancer, not video): {low[:3]} ...")
    if any(issues[k] for k in CHECKS):
        sys.exit("[audit] FAILED")
    print("[audit] passed")
