"""Are the dances physically plausible in 2D? Run from the repo root:

    python reports/2026-11-22_physics/physics_check.py

1. Calibration on real dancers: all advanced (sFM) videos, full length, per genre. In flight, the centre of mass
   should fall at g (in torso lengths / frame^2: g / torso length / 30^2) with no horizontal acceleration.
2. Real vs models on the 10 held-out test songs (20 s, seed 0; cached dances from the 2.1b-variants and feet
   reports): flight share, flight gravity, horizontal acceleration in flight, floor penetration, balance.
Penetration and balance are depth-confounded (a foot swinging toward the camera projects lower), so they are read
only against the real dancers. Writes physics.json.
"""
import json
import os
from collections import defaultdict

import numpy as np

from beat2body import paths
from beat2body.data import load_meta, video_motion
from beat2body.physics import physics_stats, summarize
from beat2body.report import absolute

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(paths.REPO_ROOT, "reports")
NPZ = {"2.1b": "2026-11-19_2.1b_variants", "2.1b.1": "2026-11-19_2.1b_variants", "2.1b.2": "2026-11-19_2.1b_variants",
       "4.1": "2026-11-20_feet", "4.4": "2026-11-20_feet"}
G_EXPECTED = 9.81 / 0.5 / 30 ** 2     # torso ~0.5 m


def fmt(s):
    return (f"flight {100 * s['flight_share']:5.2f}% ({s['flights']:3d} jumps) | acc_y {s['flight_acc_y_median']:+.4f} "
            f"(IQR {s['flight_acc_y_iqr']:.4f}) | |acc_x| {s['flight_acc_x_abs_median']:.4f} | "
            f"penetration {s['penetration_mean']:.4f} ({100 * s['penetration_share']:.1f}% > 0.05) | "
            f"off-balance {100 * s['balance_out_share']:.1f}%")


def main():
    out = {"g_expected_torso_per_frame2": G_EXPECTED}
    # 1. calibration: all advanced videos
    vids = {}
    for split in ("train", "val", "test"):
        for r in load_meta("dwpose", split):
            if "_sFM_" in r["video_id"]:
                vids[r["video_id"]] = r["category"]
    per_genre = defaultdict(list)
    for vid, cat in sorted(vids.items()):
        m = video_motion("dwpose", cat, vid, "dancer")[0]
        per_genre[cat].append(physics_stats(absolute(m)))
    print(f"expected g ~ {G_EXPECTED:.4f} torso lengths / frame^2 (torso 0.5 m)\n\nReal, all advanced videos:")
    out["real_all"] = {}
    for cat in sorted(per_genre):
        s = summarize(per_genre[cat])
        out["real_all"][cat] = s
        print(f"  {cat:14s} {fmt(s)}")
    s = summarize([x for v in per_genre.values() for x in v])
    out["real_all"]["all"] = s
    print(f"  {'all':14s} {fmt(s)}")

    # 2. the 10 test songs: real vs models
    songs = json.load(open(os.path.join(R, "2026-11-19_2.1b_variants", "stats.json")))["songs"]
    rows = {"real": []}
    for name, folder in NPZ.items():
        rows[name] = []
        for cat, vid in songs.items():
            f = os.path.join(R, folder, "npz", f"{name}_{vid}.npz")
            if not os.path.exists(f):
                continue
            m = np.load(f)["motion"]
            rows[name].append(physics_stats(absolute(m)))
            if name == "4.4":
                rows["real"].append(physics_stats(absolute(video_motion("dwpose", cat, vid, "dancer")[0][:len(m)])))
    print("\nThe 10 test songs (20 s each):")
    out["test_songs"] = {}
    for name, st in rows.items():
        if st:
            s = summarize(st)
            out["test_songs"][name] = s
            print(f"  {name:8s} {fmt(s)}")
    json.dump(out, open(os.path.join(HERE, "physics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
