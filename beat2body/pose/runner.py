"""Shared loop and options of the pose tools: video selection, raw output files,
skeleton previews, and an optional backup folder (e.g. Google Drive on Colab)."""
import argparse
import glob
import os
import shutil
import time

import numpy as np

from beat2body import aist, paths
from beat2body.skeleton import COCO17_NAMES, CONF_THRESHOLD
from beat2body.video import DEFAULT_FPS, PreviewWriter, iter_frames, probe_video, scaled_size


def add_common_args(p, tool):
    p.add_argument("--categories", nargs="*", default=None,
                   help="sub-folders of <data>/raw_video to process (default: all)")
    p.add_argument("--videos", nargs="*", default=None,
                   help="process these video files instead (category = parent folder name)")
    p.add_argument("--limit", type=int, default=None, help="at most N videos per category")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS, help="resample videos to this frame rate")
    p.add_argument("--height", type=int, default=None,
                   help="resize frames to this height before inference (default: original size)")
    p.add_argument("--preview", type=int, nargs="?", const=-1, default=0, metavar="N",
                   help=f"also write skeleton preview videos (<video_id>_{tool}.mp4): "
                        "--preview alone = every video, --preview N = the first N videos")
    p.add_argument("--overwrite", action="store_true", help="re-run videos that already have output")
    p.add_argument("--backup-dir", default=None,
                   help="copy every finished output here, and restore missing outputs from here first "
                        "(e.g. a Google Drive folder, so a disconnected run resumes)")


def wants_preview(args, n):
    return args.preview < 0 or n < args.preview


def select_videos(args):
    if args.videos:
        return [(os.path.basename(os.path.dirname(os.path.abspath(v))),
                 os.path.splitext(os.path.basename(v))[0], v) for v in args.videos]
    return aist.list_videos(args.categories, limit=args.limit)


def save_raw_keypoints(path, keypoints, native, native_names, tool, fps, width, height, seconds):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        keypoints=keypoints.astype(np.float32),
        native=native.astype(np.float32),
        native_joint_names=np.array(native_names),
        joint_names=np.array(COCO17_NAMES),
        tool=tool, fps=fps, width=width, height=height,
        processing_seconds=seconds,
    )


def _backup_path(backup_dir, path):
    return os.path.join(backup_dir, os.path.relpath(path, paths.keypoints_dir()))


def restore_backup(tool, backup_dir):
    """Copy raw outputs of `tool` from the backup folder that are missing locally."""
    if not backup_dir or not os.path.isdir(backup_dir):
        return
    n = 0
    for f in glob.glob(os.path.join(backup_dir, "*", f"*_{tool}.npz")):
        dst = os.path.join(paths.keypoints_dir(), os.path.relpath(f, backup_dir))
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(f, dst)
            n += 1
    print(f"[{tool}] restored {n} earlier outputs from {backup_dir}")


def backup(path, backup_dir):
    if backup_dir:
        dst = _backup_path(backup_dir, path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(path, dst)


def report(tool, cat, vid, coco, seconds):
    found = 100 * (~np.isnan(coco[:, :, 0]).all(1)).mean()
    print(f"[{tool}] {cat}/{vid}: {len(coco)} frames, {len(coco) / seconds:.1f} fps, "
          f"person found in {found:.0f}% of frames")


def pending_videos(tool, args):
    """Videos to process, after restoring the backup; skips videos that already have output."""
    restore_backup(tool, args.backup_dir)
    videos = select_videos(args)
    print(f"[{tool}] {len(videos)} videos")
    for n, (cat, vid, path) in enumerate(videos):
        if os.path.exists(paths.raw_keypoint_path(tool, cat, vid)) and not args.overwrite:
            print(f"[{tool}] skip {cat}/{vid} (exists)")
            continue
        yield n, cat, vid, path


def run_pose_tool(tool, args, make_infer, native_names, color):
    """Frame-by-frame loop. make_infer(fps) is called once per video and returns
    infer(frame_bgr) -> (coco17 (17, 3) or None, native (J, 3) or None)."""
    for n, cat, vid, path in pending_videos(tool, args):
        w, h, _ = probe_video(path)
        w, h = scaled_size(w, h, args.height)
        preview = None
        if wants_preview(args, n):
            preview = PreviewWriter(paths.preview_path(tool, cat, vid), args.fps, w, h, f"{tool}  {cat}/{vid}")
        infer = make_infer(args.fps)
        coco, native = [], []
        t0 = time.perf_counter()
        for frame in iter_frames(path, args.fps, args.height):
            k, kn = infer(frame)
            coco.append(np.full((17, 3), np.nan, np.float32) if k is None else k)
            native.append(np.full((len(native_names), 3), np.nan, np.float32) if kn is None else kn)
            if preview:
                preview.write_frame(frame, coco[-1], CONF_THRESHOLD[tool], color)
        dt = time.perf_counter() - t0
        if preview:
            preview.close()
        coco, native = np.stack(coco), np.stack(native)
        out = paths.raw_keypoint_path(tool, cat, vid)
        save_raw_keypoints(out, coco, native, native_names, tool, args.fps, w, h, dt)
        backup(out, args.backup_dir)
        report(tool, cat, vid, coco, dt)


def make_parser(tool, doc):
    """Stand-alone parser (used by the CLI as a sub-command)."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, tool)
    return p
