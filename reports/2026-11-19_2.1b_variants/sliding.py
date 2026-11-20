"""Where does the whole-body shifting and foot sliding come from? Run from the repo root:

    python reports/2026-11-19_2.1b_variants/sliding.py

Same 10 test songs and seed as make_comparison.py. For real dancers and each model, measures:
  root speed / root jitter      pelvis speed and its frame-to-frame change (torso lengths / frame)
  planted-foot speed            per frame the slower ankle's image speed; real dancers almost always have one
                                foot still, so this is the foot-sliding measure (no contact labels needed)
  no foot planted               share of frames where both ankles move faster than the contact threshold
  slide on predicted contact    ankle speed on frames where the model's own contact output says "planted"
  window disagreement           root distance between the two windows over their shared frames, before the
                                cross-fade (long dances are 150-frame windows overlapping by 75)
  seam glide                    root speed in the middle of the cross-fades vs at their ends
Writes sliding.json and figures/sliding.png.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from beat2body import aist  # noqa: E402
from beat2body.data import CONTACT_THRESHOLD, video_motion  # noqa: E402
from beat2body.model.diffusion import EDGEDiffusion  # noqa: E402
from beat2body.report import absolute  # noqa: E402
from beat2body.train import load_generator, music_input  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402

sys.path.insert(0, HERE)
from make_comparison import COLORS, INK, INK2, MODELS, R  # noqa: E402

ANK = [15, 16]


def windows_for(gen, bundle, vid, seconds=20.0, seed=0):
    """generate_long, but also returning the un-stitched windows (N, L, 38) denormalized."""
    cfg, cats = bundle["config"], bundle["categories"]
    cond = torch.from_numpy(music_input(vid, cfg["feature_set"], bundle["audio_stats"]))
    T = min(len(cond), int(seconds * 30))
    L, half = cfg["seq_len"], cfg["seq_len"] // 2
    n = max(1, int(np.ceil((T - L) / half)) + 1)
    need = half * (n + 1)
    cond = torch.cat([cond, cond[-1:].expand(max(0, need - len(cond)), -1)])[:need]
    w = torch.stack([cond[i * half:i * half + L] for i in range(n)])
    genre = torch.full((n,), cats[aist.GENRES[vid[1:3]]], dtype=torch.long)
    g = torch.Generator(device=gen.device).manual_seed(seed)
    m = gen.generate(w, genre, steps=50, link_windows=True, generator=g, eta=1.0).cpu()
    return m, EDGEDiffusion.stitch(m)[:T], half


def foot_stats(motion):
    j = absolute(motion)
    ank = np.linalg.norm(np.diff(j[:, ANK], axis=0), axis=-1)            # (T-1, 2)
    slow = ank.min(1)
    root = motion[:, 2:4]
    rv = np.diff(root, axis=0)
    out = {"root_speed": float(np.linalg.norm(rv, axis=-1).mean()),
           "root_jitter": float(np.linalg.norm(np.diff(rv, axis=0), axis=-1).mean()),
           "planted_foot_speed": float(np.median(slow)),
           "no_foot_planted": float((slow > CONTACT_THRESHOLD).mean())}
    c = motion[:-1, :2] > 0.5
    out["slide_on_contact"] = float(ank[c].mean()) if c.any() else float("nan")
    out["contact_rate"] = float(c.mean())
    return out, slow


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    songs = json.load(open(os.path.join(HERE, "stats.json")))["songs"]
    res, slow_all = {"real": []}, {"real": []}
    for cat, vid in songs.items():
        m = video_motion("dwpose", cat, vid, "dancer")[0][:600]
        s, slow = foot_stats(m)
        res["real"].append(s)
        slow_all["real"].append(slow)
    for name, (ck, _) in MODELS.items():
        gen, bundle = load_generator(os.path.join(R, ck), dev)
        res[name], slow_all[name] = [], []
        for vid in songs.values():
            win, motion, half = windows_for(gen, bundle, vid)
            win, motion = win.numpy(), motion.numpy()
            s, slow = foot_stats(motion)
            # root disagreement on the shared frames, before the cross-fade
            d = np.linalg.norm(win[:-1, half:, 2:4] - win[1:, :half, 2:4], axis=-1)       # (N-1, half)
            s["window_root_gap"] = float(d.mean())
            s["window_root_gap_max"] = float(d.max(1).mean())
            # root speed by position inside the cross-fade (fades cover frames half .. T-half)
            rv = np.linalg.norm(np.diff(motion[:, 2:4], axis=0), axis=-1)
            pos = (np.arange(len(rv)) - half) % half
            inside = (np.arange(len(rv)) >= half) & (np.arange(len(rv)) < len(rv) - half)
            mid, ends = inside & (pos > half * 0.33) & (pos < half * 0.67), inside & ((pos < 8) | (pos > half - 8))
            s["root_speed_fade_mid"], s["root_speed_fade_ends"] = float(rv[mid].mean()), float(rv[ends].mean())
            # the same within single windows (no stitching): how much root speed is the cross-fade's
            s["root_speed_in_window"] = float(np.linalg.norm(np.diff(win[:, :, 2:4], axis=1), axis=-1).mean())
            res[name].append(s)
            slow_all[name].append(slow)
        print("done", name, flush=True)
        del gen
    summary = {n: {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]} for n, rs in res.items()}
    json.dump({"summary": summary, "per_song": res, "songs": songs}, open(os.path.join(HERE, "sliding.json"), "w"),
              indent=1)
    for n, s in summary.items():
        print(f"{n:7}", " ".join(f"{k} {v:.4f}" for k, v in s.items()))

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.logspace(-3.5, -0.5, 60)
    for n in ["real", *MODELS]:
        v = np.concatenate(slow_all[n])
        h, _ = np.histogram(v, bins)
        ax.plot(np.sqrt(bins[1:] * bins[:-1]), h / h.sum(), color=INK if n == "real" else COLORS[n],
                lw=2.4 if n == "real" else 1.5, label=n)
    ax.axvline(CONTACT_THRESHOLD, color=INK2, lw=1, ls="--")
    ax.text(CONTACT_THRESHOLD * 1.08, ax.get_ylim()[1] * 0.92, "contact threshold", color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("speed of the slower ankle (torso lengths / frame)")
    ax.set_ylabel("share of frames")
    ax.set_title("Planted foot: real dancers keep one foot still, the models do not")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "sliding.png"), dpi=150)


if __name__ == "__main__":
    main()
