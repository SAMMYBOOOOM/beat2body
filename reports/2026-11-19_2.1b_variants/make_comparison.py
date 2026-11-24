"""Side-by-side videos and per-song statistics for the 2.1b family. Run from the repo root:

    python reports/2026-11-19_2.1b_variants/make_comparison.py [--seconds 20] [--no-video]

For one advanced (sFM) test song per genre: generates a 20 s dance with each model (same seed, 50 DDIM steps,
eta 1, the model's own guidance), renders real | 2.1b | 2.1b.1 | 2.1b.2 | 2.1b.3 with the music, and measures
per song: speed, beat-band power, limb amplitude, body-size pulsing at 1-4 Hz, slow body-size drift, bone
variation and root travel (each as generated / real). Writes videos/, npz/, stats.csv, stats.json, figures/.
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from beat2body import aist, paths  # noqa: E402
from beat2body.data import load_meta, video_motion  # noqa: E402
from beat2body.evaluate import beat_power, bone_cv  # noqa: E402
from beat2body.report import absolute, audio_source, write_panels  # noqa: E402
from beat2body.train import generate_long, load_generator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(paths.REPO_ROOT, "runs")
MODELS = {  # name -> (checkpoint, short description)
    "2.1b": ("beat2body-diffusion-c-small-co-best-so-far/model.pt", "base"),
    "2.1b.1": ("beat2body-diffusion-2.1b.1-large-2f9d37ca/model.pt", "large"),
    "2.1b.2": ("beat2body-diffusion-e-gauss-b9934c0c-2026-11-19-22-46-15-486/model.pt", "gauss"),
    "2.1b.3": ("beat2body-diffusion-e-square-5f731354-2026-11-19-22-46-28-136/model.pt", "square"),
}
# reference palette (light): categorical slots 1-4 in order, text and surface tokens
COLORS = {"2.1b": "#2a78d6", "2.1b.1": "#eb6834", "2.1b.2": "#1baf7a", "2.1b.3": "#eda100"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
})
METRICS = {  # key -> (label, what "good" is)
    "speed": ("Speed", "joint speed"),
    "beatpow": ("Beat-band power", "1-4 Hz movement, pelvis-centred"),
    "amplitude": ("Limb amplitude", "std of pelvis-centred joints"),
    "size_pulse": ("Body-size pulsing", "1-4 Hz power of torso length"),
    "torso_cv": ("Body-size drift", "torso length std / mean over 20 s"),
    "bone_cv": ("Bone variation", "BoneCV"),
    "root_travel": ("Root travel", "pelvis speed"),
}


def torso(j):
    return np.linalg.norm(j[:, [5, 6]].mean(1) - j[:, [11, 12]].mean(1), axis=-1)


def band(x, lo=1.0, hi=4.0, fps=30):
    f = np.abs(np.fft.rfft(x - x.mean())) ** 2
    fr = np.fft.rfftfreq(len(x), 1 / fps)
    return f[(fr >= lo) & (fr < hi)].sum() / len(x)


def stats(j):
    """j (T, 17, 2) absolute joints in body units."""
    pel = j[:, [11, 12]].mean(1)
    rel = j - pel[:, None]
    t = torso(j)
    return {
        "speed": float(np.linalg.norm(np.diff(j, axis=0), axis=-1).mean()),
        "beatpow": beat_power(j[None]),
        "amplitude": float(rel.std(0).mean()),
        "size_pulse": band(t / t.mean()),
        "torso_cv": float(t.std() / t.mean()),
        "bone_cv": float(np.mean(bone_cv(j[None]))),
        "root_travel": float(np.linalg.norm(np.diff(pel, axis=0), axis=-1).mean()),
        "torso_mean": float(t.mean()),
    }


def pick_songs(seconds):
    rows = load_meta("dwpose", "test")
    length = {}
    for r in rows:
        length[r["video_id"]] = max(length.get(r["video_id"], 0), int(r["end_frame"]))
    out = {}
    for v in sorted(length):
        if "_sFM_" in v and length[v] >= seconds * 30:
            out.setdefault(aist.GENRES[v[1:3]], v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--size", type=int, default=300)
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    for d in ("videos", "npz", "figures"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    songs = pick_songs(args.seconds)
    gen_j = {}
    for name, (ck, _) in MODELS.items():
        gen, bundle = load_generator(os.path.join(R, ck), dev)
        assert bundle["config"].get("scale_norm") == "dancer", name
        for vid in songs.values():
            f = os.path.join(HERE, "npz", f"{name}_{vid}.npz")
            if not os.path.exists(f):
                motion, _, _ = generate_long(gen, bundle, vid, args.seconds, seed=0)
                np.savez(f, motion=motion.numpy())
            gen_j[name, vid] = absolute(np.load(f)["motion"])
        print("generated", name, flush=True)
        del gen
    rows, curves = [], {}
    for cat, vid in songs.items():
        T = min(len(gen_j[n, vid]) for n in MODELS)
        real = absolute(video_motion("dwpose", cat, vid, "dancer")[0][:T])
        rs = stats(real)
        curves[vid] = {"real": torso(real)}
        for n in MODELS:
            j = gen_j[n, vid][:T]
            s = stats(j)
            curves[vid][n] = torso(j)
            rows.append({"genre": cat, "video_id": vid, "music": aist.music_id(vid), "model": n,
                         **{k: float(s[k] / rs[k]) for k in METRICS},
                         "torso_vs_real": float(s["torso_mean"] / rs["torso_mean"]),
                         **{"real_" + k: float(rs[k]) for k in METRICS}})
        if not args.no_video:
            beats = np.load(paths.feature_path("beats", vid))
            path = os.path.join(HERE, "videos", f"compare_{cat}_{vid}.mp4")
            write_panels(path, [real] + [gen_j[n, vid][:T] for n in MODELS],
                         ["real dancer"] + [f"{n} {MODELS[n][1]}" for n in MODELS],
                         beats[beats < T], args.size, audio_source(cat, vid))
            print("wrote", path, flush=True)
    with open(os.path.join(HERE, "stats.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {n: {k: float(np.median([r[k] for r in rows if r["model"] == n])) for k in [*METRICS, "torso_vs_real"]}
               for n in MODELS}
    json.dump({"songs": songs, "seconds": args.seconds, "median_ratio_to_real": summary, "rows": rows},
              open(os.path.join(HERE, "stats.json"), "w"), indent=1)
    figures(rows, songs, curves)
    for n, s in summary.items():
        print(n, " ".join(f"{k} {v:.2f}" for k, v in s.items()))


def figures(rows, songs, curves):
    names = list(MODELS)
    keys = ["speed", "beatpow", "amplitude", "size_pulse", "torso_cv", "bone_cv"]
    # 1. ratio to real per metric: one dot per song, median bar per model (log scale, 1 = real)
    fig, axes = plt.subplots(1, len(keys), figsize=(15, 3.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, k in zip(axes, keys):
        for i, n in enumerate(names):
            v = np.array([r[k] for r in rows if r["model"] == n])
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=14, color=COLORS[n], alpha=0.55, lw=0)
            ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=COLORS[n], lw=3, solid_capstyle="round")
        ax.axhline(1, color=INK2, lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_title(METRICS[k][0])
        ax.set_xticks(range(len(names)), names, rotation=30)
    axes[0].set_ylabel("generated / real (1 = real)")
    axes[0].text(0.02, 1.08, "real", color=INK2, fontsize=8, transform=axes[0].get_yaxis_transform())
    fig.suptitle("10 held-out test songs, 20 s each (dots: songs; bars: median)", x=0.01, ha="left", color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "ratios.png"), dpi=150)
    plt.close(fig)

    # 2. per-genre beat-band power and body-size pulsing (heatmaps, diverging around real = 1)
    genres = list(songs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, k in zip(axes, ["beatpow", "size_pulse"]):
        m = np.array([[next(r[k] for r in rows if r["model"] == n and r["genre"] == g) for g in genres] for n in names])
        lm = np.log2(m)
        im = ax.imshow(lm, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(len(genres)), genres, rotation=40, ha="right")
        ax.set_yticks(range(len(names)), names)
        for a in range(len(names)):
            for b in range(len(genres)):
                ax.text(b, a, f"{m[a, b]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(lm[a, b]) > 1.5 else INK)
        ax.set_title(f"{METRICS[k][0]}, generated / real")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_yticklabels(["¼×", "½×", "real", "2×", "4×"])
    fig.savefig(os.path.join(HERE, "figures", "per_genre.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. torso length over the dance for two songs: pulsing and drift
    show = [songs[g] for g in ("house", "krump") if g in songs] or list(songs.values())[:2]
    fig, axes = plt.subplots(len(show), 1, figsize=(11, 2.6 * len(show)), sharex=True)
    for ax, vid in zip(np.atleast_1d(axes), show):
        c = curves[vid]
        t = np.arange(len(c["real"])) / 30
        ax.plot(t, c["real"] / c["real"].mean(), color=INK, lw=2.2, label="real")
        for n in names:
            ax.plot(t, c[n] / c[n].mean(), color=COLORS[n], lw=1.2, label=n)
        ax.set_title(f"{aist.GENRES[vid[1:3]]} ({vid}): torso length / its mean")
    np.atleast_1d(axes)[0].legend(ncol=5, loc="upper right", bbox_to_anchor=(1, 1.3))
    np.atleast_1d(axes)[-1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "torso_traces.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
