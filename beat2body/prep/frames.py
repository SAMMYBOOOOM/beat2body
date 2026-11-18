"""4.2 Frame extraction: dump JPEG frames at a fixed frame rate.

Optional. The pose tools read videos directly at the same fixed frame rate, so JPEGs
are only needed for inspection. 1080p JPEGs of all 210 AIST videos take 20+ GB;
use --categories / --limit / --height to keep it small.

    python -m beat2body frames --categories la_hiphop --limit 1
    -> <data>/frames/la_hiphop/<video_id>/000001.jpg ...
"""
import os
import subprocess

from beat2body import aist, paths
from beat2body.video import DEFAULT_FPS

DOC = __doc__


def add_args(p):
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None, help="at most N videos per category")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--height", type=int, default=None, help="resize to this height (default: original)")
    p.add_argument("--quality", type=int, default=3, help="ffmpeg JPEG quality, 2 (best) to 31")


def run(args):
    for cat, vid, path in aist.list_videos(args.categories, limit=args.limit):
        out_dir = os.path.join(paths.frames_dir(), cat, vid)
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            print("skip", out_dir)
            continue
        os.makedirs(out_dir, exist_ok=True)
        vf = f"fps={args.fps}" + (f",scale=-2:{args.height}" if args.height else "")
        subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-q:v", str(args.quality),
                        os.path.join(out_dir, "%06d.jpg")], check=True)
        print(f"{cat}/{vid}: {len(os.listdir(out_dir))} frames")
