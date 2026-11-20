"""Table of the foot-fix runs on the test songs, raw and with foot locking. Run from the repo root:

    python reports/2026-11-20_feet/summarize.py

Reads runs/<job>/eval_test_feet2_{raw,lock}.json (python -m beat2body evaluate ... --tag feet2_raw | feet2_lock
--foot-lock --root-smooth 1) and writes metrics/feet2_summary.csv and a Markdown table to stdout.
"""
import csv
import json
import os

from beat2body import paths

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {
    "2.1b.2": "beat2body-diffusion-e-gauss-b9934c0c-2026-11-19-22-46-15-486",
    "4.2": "beat2body-diffusion-v42-98ca09dc-2026-11-20-16-22-50-670",
    "4.2-allt": "beat2body-diffusion-r42-od-ddd05822-2026-11-20-15-54-53-814",
    "4.3": "beat2body-diffusion-v43-c03b2cc0-2026-11-20-17-51-52-776",
    "4.1": "beat2body-diffusion-v41-863026e8-2026-11-20-16-20-14-033",
}
COLS = [("FID_k", "row"), ("FID_p", "row"), ("speed", "row"), ("BeatPow", "row"), ("BoneCV", "row"), ("PFC", "row"),
        ("planted_foot_speed", "feet"), ("no_foot_planted", "feet"), ("root_jitter", "feet"), ("stance_drift", "feet"),
        ("contact_frames", "feet"), ("contact_share", "feet"), ("swing_speed", "feet"), ("torso", "drift")]


def load(run, mode):
    f = os.path.join(paths.REPO_ROOT, "runs", RUNS[run], f"eval_test_feet2_{mode}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    row = d["rows"][0]
    out = {}
    for k, src in COLS:
        if src == "row":
            out[k] = row[k]
        elif src == "feet":
            out[k] = d["feet"]["generated"].get(k)
        else:
            out[k] = (d.get("long_drift") or {}).get("generated", {}).get(k)
    real = {k: d["feet"]["real"].get(k) for k, src in COLS if src == "feet"}
    real.update(PFC=d["rows"][2]["PFC"], BoneCV=d["rows"][2]["BoneCV"], speed=1.0, BeatPow=1.0, FID_k=0.0, FID_p=0.0,
                torso=(d.get("long_drift") or {}).get("real", {}).get("torso"))
    return out, real


def main():
    rows, real = [], None
    for run in RUNS:
        for mode in ("raw", "lock"):
            r = load(run, mode)
            if r:
                rows.append({"run": run, "mode": mode, **r[0]})
                real = real or r[1]
    if not rows:
        print("no evaluations yet")
        return
    os.makedirs(os.path.join(HERE, "metrics"), exist_ok=True)
    with open(os.path.join(HERE, "metrics", "feet2_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "mode"] + [k for k, _ in COLS])
        w.writeheader()
        w.writerows(rows)
    names = [k for k, _ in COLS]
    print("| Run | Mode | " + " | ".join(names) + " |")
    print("|---|---|" + "---:|" * len(names))
    fmt = (lambda v: "" if v is None else (f"{v:.4f}" if abs(v) < 0.1 else f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"))
    print("| Real | | " + " | ".join(fmt(real.get(k)) for k in names) + " |")
    for r in rows:
        print(f"| {r['run']} | {r['mode']} | " + " | ".join(fmt(r[k]) for k in names) + " |")


if __name__ == "__main__":
    main()
