"""Videos of run 4.4 next to the base model 4.1 on the 10 held-out test songs. Run from the repo root:

    python reports/2026-11-20_feet/make_41_videos.py

Same songs, seed and settings as reports/2026-11-19_2.1b_variants (20 s, 50 DDIM steps, eta 1, guidance 2, seed 0);
4.1 is read from this folder's npz/ cache (make_41_videos.py). Writes videos/compare44_<genre>_<video>.mp4
(real | 4.1 | 4.4, with music) and npz/4.4_<video>.npz.
"""
import json
import os

import numpy as np
import torch

from beat2body import paths
from beat2body.data import video_motion
from beat2body.report import absolute, audio_source, write_panels
from beat2body.train import generate_long, load_generator

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(paths.REPO_ROOT, "reports", "2026-11-19_2.1b_variants")
CK = os.path.join(paths.REPO_ROOT, "runs", "beat2body-diffusion-v44-5df86e13-2026-11-20-21-00-54-698", "model.pt")
PANELS = [("4.1", "4.1 (base)"), ("4.4", "4.4 (aligned labels + new energy)")]


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    songs = json.load(open(os.path.join(SRC, "stats.json")))["songs"]
    for d in ("videos", "npz"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    gen, bundle = load_generator(CK, dev)
    for cat, vid in songs.items():
        f = os.path.join(HERE, "npz", f"4.4_{vid}.npz")
        if not os.path.exists(f):
            motion, _, _ = generate_long(gen, bundle, vid, 20.0, seed=0)
            np.savez(f, motion=motion.numpy())
        seqs = {"4.4": np.load(f)["motion"], "4.1": np.load(os.path.join(HERE, "npz", f"4.1_{vid}.npz"))["motion"]}
        T = min(len(s) for s in seqs.values())
        real = video_motion("dwpose", cat, vid, "dancer")[0][:T]
        beats = np.load(paths.feature_path("beats", vid))
        out = os.path.join(HERE, "videos", f"compare44_{cat}_{vid}.mp4")
        write_panels(out, [absolute(real)] + [absolute(seqs[n][:T]) for n, _ in PANELS],
                     ["real dancer"] + [t for _, t in PANELS], beats[beats < T], 300, audio_source(cat, vid))
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
