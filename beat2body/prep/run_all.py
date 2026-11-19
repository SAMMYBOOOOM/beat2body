"""4.4 + 4.5 + 4.6 + 4.7 for one or more pose tools: audio -> clean -> clips.

Run the pose tools first (python -m beat2body pose <tool>), then:

    python -m beat2body all --tools dwpose mediapipe openpose
    python -m beat2body all --tools dwpose --categories la_hiphop --delete-raw

Before cleaning it checks that every category has the same number of raw outputs per tool.
--delete-raw removes the raw pose outputs (<video_id>_<tool>.npz) once a tool's clips are
written. Every option of the clean, audio and clips steps is accepted as well.
"""
import argparse
import os
from collections import Counter

from beat2body.paths import TOOLS
from beat2body.prep import audio, clean, clips

DOC = __doc__


def add_args(p):
    p.add_argument("--tools", nargs="+", choices=TOOLS, required=True)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None, help="at most N videos per category")
    p.add_argument("--skip-audio", action="store_true", help="audio features already exist")
    p.add_argument("--delete-raw", action="store_true", help="delete raw pose outputs after each tool")
    g = p.add_argument_group("clean (4.4)")
    clean.add_args(g, with_tool=False)
    g = p.add_argument_group("audio (4.5)")
    audio.add_args(g, with_categories=False)
    g = p.add_argument_group("clips (4.6 + 4.7)")
    clips.add_args(g, with_tool=False)


def check_raw(tools, categories):
    counts = {t: Counter(os.path.basename(os.path.dirname(f)) for f in clean.raw_files(t, categories)) for t in tools}
    cats = sorted(set().union(*counts.values()))
    for c in cats:
        print(f"  {c:15s} " + "  ".join(f"{t}={counts[t][c]}" for t in tools))
    if len({tuple(counts[t][c] for c in cats) for t in tools}) > 1:
        print("  warning: the tools have different numbers of raw outputs per category")


def run(args):
    if not args.skip_audio:
        print("== 4.5 audio")
        audio.run(args)
    print("== raw pose outputs per category")
    check_raw(args.tools, args.categories)
    for tool in args.tools:
        targs = argparse.Namespace(**{**vars(args), "tool": tool})
        raw = clean.raw_files(tool, args.categories)
        if not raw:
            print(f"[{tool}] no raw pose output, skipped")
            continue
        print(f"== 4.4 clean ({tool})")
        clean.run(targs)
        print(f"== 4.6 + 4.7 clips ({tool})")
        clips.run(targs)
        if args.delete_raw:
            for f in raw:
                os.remove(f)
            print(f"[{tool}] deleted {len(raw)} raw outputs")
