"""Command line: python -m beat2body <command> [options]   (run from anywhere)

Data preparation (README 4.2-4.7), in order:
    pose <tool>   4.3 pose estimation (dwpose | mediapipe | openpose)
    clean         4.4 keypoint cleaning + normalization (one tool)
    audio         4.5 audio features (shared by all tools)
    clips         4.6 + 4.7 windows and train / val / test split (one tool)
    all           audio + clean + clips for several tools
    frames        4.2 optional JPEG dump
Data on Hugging Face:
    hf download | card | upload
Training (needs torch):
    data          check the PyTorch data loader (shapes, NaNs, audio stats, speed)
Inspection:
    render        stick-figure video of real or generated keypoints

Use `python -m beat2body <command> --help` for options. The data root defaults to
<repo>/2d; change it with --data-root or BEAT2BODY_DATA_ROOT.
"""
import argparse
import sys

from beat2body import paths

sys.dont_write_bytecode = True


def _commands():
    # imported lazily so `--help` works without the optional dependencies of each step
    import beat2body.pose as pose_pkg
    from beat2body import render
    from beat2body.prep import audio, clean, clips, frames, hf, run_all
    commands = {"clean": clean, "audio": audio, "clips": clips, "all": run_all, "frames": frames,
                "hf": hf, "render": render}
    try:  # training commands need torch (pip install -e ".[train]")
        from beat2body import data
        commands["data"] = data
    except ImportError:
        pass
    return commands, pose_pkg


def build_parser():
    commands, pose_pkg = _commands()
    fmt = argparse.RawDescriptionHelpFormatter
    p = argparse.ArgumentParser(prog="python -m beat2body", description=__doc__, formatter_class=fmt)
    p.add_argument("--data-root", default=None, help=f"data folder (default: {paths.data_root()})")
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    pose = sub.add_parser("pose", help="4.3 pose estimation", formatter_class=fmt,
                          description=pose_pkg.__doc__)
    pose_sub = pose.add_subparsers(dest="tool", required=True, metavar="tool")
    for name, mod in pose_pkg.TOOLS.items():
        sp = pose_sub.add_parser(name, help=mod.DOC.splitlines()[0], description=mod.DOC, formatter_class=fmt)
        mod.add_args(sp)
        sp.set_defaults(run=mod.run)

    for name, mod in commands.items():
        sp = sub.add_parser(name, help=mod.DOC.splitlines()[0], description=mod.DOC, formatter_class=fmt)
        mod.add_args(sp)
        sp.set_defaults(run=mod.run)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.data_root:
        paths.set_data_root(args.data_root)
    args.run(args)


if __name__ == "__main__":
    main()
