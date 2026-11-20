"""Foot locking on the cached 2.1b-family dances (reports/2026-11-19_2.1b_variants/npz). Run from the repo root:

    python reports/2026-11-20_feet/make_lock_videos.py [model ...]

Writes videos/lock_<model>_<genre>_<video>.mp4 (real | model | model + foot lock | + root smoothing) and
lock_stats.json (feet / root measures for each).
"""
import json
import os
import sys

import numpy as np

from beat2body import paths
from beat2body.data import CONTACT_THRESHOLD, video_motion
from beat2body.model.footlock import feet_stats, foot_lock
from beat2body.report import absolute, audio_source, write_panels

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(paths.REPO_ROOT, "reports", "2026-11-19_2.1b_variants")


def main(models):
    songs = json.load(open(os.path.join(SRC, "stats.json")))["songs"]
    os.makedirs(os.path.join(HERE, "videos"), exist_ok=True)
    stats = {}
    for name in models:
        acc = {"real": [], "model": [], "lock": [], "lock_s1": []}
        for cat, vid in songs.items():
            m = np.load(os.path.join(SRC, "npz", f"{name}_{vid}.npz"))["motion"]
            real = video_motion("dwpose", cat, vid, "dancer")[0][:len(m)]
            seqs = {"real": real, "model": m, "lock": foot_lock(m), "lock_s1": foot_lock(m, root_smooth=1.0)}
            for k, s in seqs.items():
                acc[k].append(feet_stats(absolute(s), CONTACT_THRESHOLD))
            if cat in ("house", "krump", "lock", "break"):
                b = np.load(paths.feature_path("beats", vid))
                write_panels(os.path.join(HERE, "videos", f"lock_{name}_{cat}_{vid}.mp4"),
                             [absolute(seqs[k]) for k in seqs],
                             ["real dancer", name, f"{name} + foot lock", "+ root smoothing"],
                             b[b < len(m)], 300, audio_source(cat, vid))
        stats[name] = {k: {s: float(np.mean([r[s] for r in v])) for s in v[0]} for k, v in acc.items()}
        print(name, json.dumps(stats[name]))
    json.dump({"songs": songs, "stats": stats}, open(os.path.join(HERE, "lock_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["2.1b", "2.1b.2"])
