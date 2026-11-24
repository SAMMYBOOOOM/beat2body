"""Side-by-side videos (real dancer | model A | model B | ...) on the 10 held-out test songs, with music.

    python scripts/make_videos.py --out reports/2026-11-23_v6/videos --device cuda \
        --model 6.1=runs/<job>/model.pt --model 6.2=runs/<job>/model.pt --model 6.3=runs/<job>/model.pt

Same songs and settings as reports/2026-11-19_2.1b_variants (20 s, 50 DDIM steps, eta 1, guidance 2, seed 0), raw
output (no foot lock). Generated motion is cached as <out>/../npz/<label>_<video>.npz, so a rerun only writes videos.
The music is taken from 2d/audio/<genre>/<video>.wav, else from the raw video 2d/raw_video/<genre>/<video>.*.
Runs on Colab (scripts/colab_eval.py, video queue), not on a local machine.
"""
import argparse
import json
import os

import numpy as np
import torch

from beat2body import paths
from beat2body.data import video_motion
from beat2body.report import absolute, audio_source, write_panels
from beat2body.train import generate_long, load_generator

SONGS = os.path.join(paths.REPO_ROOT, "reports", "2026-11-19_2.1b_variants", "stats.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="folder for the mp4 files")
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=CHECKPOINT")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--size", type=int, default=300, help="panel size in pixels")
    a = ap.parse_args(argv)
    models = [m.split("=", 1) for m in a.model]
    songs = json.load(open(SONGS))["songs"]
    npz = os.path.join(os.path.dirname(os.path.abspath(a.out)), "npz")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(npz, exist_ok=True)

    for label, ck in models:                      # one model in memory at a time
        gen, bundle = load_generator(ck, a.device)
        for i, vid in enumerate(songs.values()):
            f = os.path.join(npz, f"{label}_{vid}.npz")
            if not os.path.exists(f):
                motion, _, _ = generate_long(gen, bundle, vid, a.seconds, seed=0)
                np.savez(f, motion=motion.numpy())
            print(f"[videos] {label}: song {i + 1}/{len(songs)}", flush=True)
        del gen, bundle

    tag = "_".join(label for label, _ in models).replace(".", "")
    for cat, vid in songs.items():
        seqs = [np.load(os.path.join(npz, f"{label}_{vid}.npz"))["motion"] for label, _ in models]
        T = min(len(s) for s in seqs)
        real = video_motion("dwpose", cat, vid, "dancer")[0][:T]
        beats = np.load(paths.feature_path("beats", vid))
        out = os.path.join(a.out, f"compare_{tag}_{cat}_{vid}.mp4")
        audio = audio_source(cat, vid)
        if audio is None:
            print(f"[videos] no music for {vid}: silent video", flush=True)
        write_panels(out, [absolute(real)] + [absolute(s[:T]) for s in seqs],
                     ["real dancer"] + [label for label, _ in models], beats[beats < T], a.size, audio)
        print("[videos] wrote", out, flush=True)


if __name__ == "__main__":
    main()
