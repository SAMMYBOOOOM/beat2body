"""Per-genre comparison of the evaluated models, from the eval_test_<tag>.json files on Hugging Face (light: no GPU).

    python scripts/summarize_genres.py                     # models of the per-genre queue, tag feet2_raw_genre
    python scripts/summarize_genres.py --out reports/2026-11-24_genres/README.md

Needs read access to SizheZ03/beat2body-runs (`hf auth login` or HF_TOKEN). Writes markdown:
  1. genre difficulty: each model's FID divided by its genre's real reference (real training dances of the same
     genre, same number of windows), averaged over the models; FIDs of different genres are not comparable
     directly (50-120 windows per genre bias FID upwards by different amounts);
  2. one table per genre: every model next to the real reference values.
"""
import argparse
import json
import os
import statistics

from huggingface_hub import HfApi, hf_hub_download

HF_REPO = "SizheZ03/beat2body-runs"
COLS = [  # (key, real reference key, header, format)
    ("FID_k", "real_train_FID_k", "FID_k", "{:.1f}"),
    ("FID_p", "real_train_FID_p", "FID_p", "{:.1f}"),
    ("Div_k", "real_Div_k", "Div_k", "{:.2f}"),
    ("speed", None, "speed", "{:.2f}"),
    ("BeatPow", None, "BeatPow", "{:.2f}"),
    ("BoneCV", "real_BoneCV", "BoneCV", "{:.3f}"),
    ("stance_drift", "real_stance_drift", "stance drift", "{:.3f}"),
    ("swing_speed", "real_swing_speed", "swing", "{:.3f}"),
    ("no_foot_planted", "real_no_foot_planted", "no foot", "{:.0%}"),
    ("root_jitter", "real_root_jitter", "jitter", "{:.4f}"),
    ("flight_acc_y_median", "real_flight_acc_y_median", "fall rate", "{:.4f}"),
    ("flights", "real_flights", "jumps", "{}"),
    ("penetration_mean", "real_penetration_mean", "below floor", "{:.4f}"),
    ("drift_height", "real_drift_height", "height drift", "{:.3f}"),
]


def fmt(f, v):
    if v is None:
        return "–"
    try:
        return f.format(v)
    except (TypeError, ValueError):
        return "–"


def load(tag, token=None):
    api = HfApi(token=token)
    queue = json.load(open(hf_hub_download(HF_REPO, "evals/queue.json", token=token, force_download=True)))
    files = set(api.list_repo_files(HF_REPO))
    out = {}
    for e in queue:
        if e["tag"] != tag:
            continue
        path = f"runs/{e['job']}/eval_test_{tag}.json"
        if path not in files:
            print(f"not evaluated yet: {e['name']} ({path})")
            continue
        d = json.load(open(hf_hub_download(HF_REPO, path, token=token)))
        if d.get("by_genre"):
            out[e["name"]] = d["by_genre"]
    return out


def report(models):
    genres = sorted({g for m in models.values() for g in m})
    lines = ["# Per-genre comparison", "",
             f"Models: {', '.join(models)}. Real reference = real training dances of the same genre, as many windows "
             "as the genre's test windows. Speed and BeatPow are relative to the genre's real test dances (1 = real).",
             "", "## Genre difficulty", "",
             "Model FID ÷ the genre's real-reference FID (1 = as close as real dance), mean over the models; "
             "higher = harder.", "",
             "| genre | windows | FID_k ratio | FID_p ratio | " + " | ".join(f"{m} FID_k÷ref" for m in models) + " |",
             "|---|---:|---:|---:|" + "---:|" * len(models)]
    rows = []
    for g in genres:
        rk, rp, per = [], [], []
        for m in models.values():
            v = m.get(g, {})
            if v.get("real_train_FID_k"):
                rk.append(v["FID_k"] / v["real_train_FID_k"])
                rp.append(v["FID_p"] / v["real_train_FID_p"])
            per.append(v["FID_k"] / v["real_train_FID_k"] if v.get("real_train_FID_k") else None)
        n = next((m[g]["windows"] for m in models.values() if g in m), 0)
        rows.append((statistics.mean(rk) if rk else float("nan"), g, n, rp, per))
    for k, g, n, rp, per in sorted(rows, reverse=True):
        lines.append(f"| {g} | {n} | {k:.2f} | {statistics.mean(rp) if rp else float('nan'):.2f} | "
                     + " | ".join(fmt("{:.2f}", x) for x in per) + " |")
    for g in genres:
        lines += ["", f"## {g}", "", "| | " + " | ".join(h for _, _, h, _ in COLS) + " |",
                  "|---|" + "---:|" * len(COLS)]
        ref = next((m[g] for m in models.values() if g in m), {})
        lines.append("| **real** | " + " | ".join(fmt(f, ref.get(r)) if r else "1" for _, r, _, f in COLS) + " |")
        for name, m in models.items():
            v = m.get(g, {})
            lines.append(f"| {name} | " + " | ".join(fmt(f, v.get(k)) for k, _, _, f in COLS) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="feet2_raw_genre")
    ap.add_argument("--out", default=None, help="markdown file (default: print)")
    a = ap.parse_args(argv)
    models = load(a.tag, os.environ.get("HF_TOKEN"))
    if not models:
        raise SystemExit("no per-genre results yet")
    text = report(models)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(text)
        print("wrote", a.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
