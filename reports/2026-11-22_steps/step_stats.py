"""How do the feet move? Steps (lift, travel, put down) vs slides, real dancers vs models. Run from the repo root:

    python reports/2026-11-22_steps/step_stats.py

Per foot, on the 10 held-out test songs (20 s, the cached dances of the 2.1b-variants and feet reports): contacts by
the ankle-speed rule (as the training labels); every movement between two contacts is a "move". Per move: travel =
ankle displacement from lift-off to landing, lift = highest point above the local floor during the move (physics.
local_floor), duration. A move with lift < 0.05 torso lengths is a slide. Writes step_stats.json.
"""
import json
import os

import numpy as np

from beat2body import paths
from beat2body.data import contact_labels, video_motion
from beat2body.physics import local_floor
from beat2body.report import absolute

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(paths.REPO_ROOT, "reports")
NPZ = {"2.1b.2": "2026-11-19_2.1b_variants", "4.1": "2026-11-20_feet", "4.4": "2026-11-20_feet"}
SLIDE = 0.05


def runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def moves(j):
    an = j[:, [15, 16]]
    lab = contact_labels(an) > 0.5
    slow = lab.copy()
    slow[:-1] |= lab[1:]
    floor = local_floor(an[..., 1], slow, 15)
    out = []
    for k in range(2):
        for a, b in runs(~slow[:, k]):                       # frames a .. b-1 move; a-1 and b are planted
            if a == 0 or b >= len(j):
                continue                                     # incomplete at the clip edges
            travel = float(np.linalg.norm(an[b, k] - an[a - 1, k]))
            lift = float(np.max(floor[a:b] - an[a:b, k, 1]))
            out.append({"travel": travel, "lift": lift, "frames": int(b - a)})
    return out


def summary(ms, seconds):
    if not ms:
        return {}
    tr = np.array([m["travel"] for m in ms])
    li = np.array([m["lift"] for m in ms])
    fr = np.array([m["frames"] for m in ms])
    big = tr > 0.1                                            # moves that change place
    return {"moves_per_s_per_foot": len(ms) / seconds / 2, "travel_median": float(np.median(tr)),
            "relocating_moves_share": float(big.mean()), "relocations_per_s_per_foot": float(big.sum() / seconds / 2),
            "lift_median": float(np.median(li)), "slide_share": float((li < SLIDE).mean()),
            "slide_share_of_relocations": float((li[big] < SLIDE).mean()) if big.any() else float("nan"),
            "move_frames_median": float(np.median(fr))}


def main():
    songs = json.load(open(os.path.join(R, "2026-11-19_2.1b_variants", "stats.json")))["songs"]
    res, all_m = {}, {"real": [], **{n: [] for n in NPZ}}
    secs = 0.0
    for cat, vid in songs.items():
        ms = {n: np.load(os.path.join(R, f, "npz", f"{n}_{vid}.npz"))["motion"] for n, f in NPZ.items()}
        T = min(len(m) for m in ms.values())
        secs += T / 30
        all_m["real"] += moves(absolute(video_motion("dwpose", cat, vid, "dancer")[0][:T]))
        for n, m in ms.items():
            all_m[n] += moves(absolute(m[:T]))
    keys = ["moves_per_s_per_foot", "relocations_per_s_per_foot", "travel_median", "relocating_moves_share",
            "lift_median", "slide_share", "slide_share_of_relocations", "move_frames_median"]
    print("".ljust(8) + "".join(k[:14].rjust(15) for k in keys))
    for n, ms in all_m.items():
        res[n] = summary(ms, secs)
        print(n.ljust(8) + "".join(f"{res[n][k]:15.3f}" for k in keys))
    json.dump(res, open(os.path.join(HERE, "step_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
