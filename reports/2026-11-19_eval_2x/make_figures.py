"""Figures for reports/2026-11-19_eval_2x/README.md. Run from the repo root:

    python reports/2026-11-19_eval_2x/make_figures.py

Reads the runs' metrics.jsonl (final or live/), probe_*.json and the beat / onset features; writes figures/*.png.
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from beat2body import aist, paths  # noqa: E402
from beat2body.prep.audio import _grid_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
RUNS = {
    "2.1b": "beat2body-diffusion-c-small-co-best-so-far",
    "2.3": "beat2body-diffusion-c-small-pu-eb0ffe3f-2026-11-19-16-17-55-187",
    "3.2": "beat2body-diffusion-ba-co-48ba8a4d-2026-11-19-18-12-54-636",
    "3.1": "beat2body-diffusion-ba-pure-8a92bce1-2026-11-19-18-05-40-302",
}
TEST_SONGS = {"mBR4", "mHO4", "mJB3", "mJS0", "mKR5", "mLH4", "mLO2", "mMH4", "mPO0", "mWA5"}
# reference palette (light): categorical slots 1-3, text and surface tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
})


def run_dir(name):
    return os.path.join(paths.REPO_ROOT, "runs", RUNS[name])


def checks(name):
    """Training checks per step: val_loss, beat_power, bone_cv, speed_ratio."""
    d = run_dir(name)
    f = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(f):
        f = os.path.join(d, "live", "metrics.jsonl")
    out = {}
    for line in open(f):
        r = json.loads(line)
        s = out.setdefault(r["step"], {})
        if "val_loss" in r:
            s["val_loss"] = r["val_loss"]
        if "sample" in r:
            s.update(r["sample"])
    steps = sorted(k for k, v in out.items() if "val_loss" in v and "beat_power" in v)
    return steps, {k: [out[s][k] for s in steps] for k in ("val_loss", "beat_power", "bone_cv", "speed_ratio")}, \
        float(np.mean([out[s]["bone_cv_real"] for s in steps]))


def fig_training():
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for row, (base, new) in enumerate((("2.1b", "3.2"), ("2.3", "3.1"))):
        for name, color in ((base, BLUE), (new, ORANGE)):
            steps, c, real_cv = checks(name)
            k = np.array(steps) / 1000
            label = f"{name}" + (" + beat attention" if name in ("3.1", "3.2") else "")
            for ax, key in zip(axes[row], ("val_loss", "beat_power", "bone_cv")):
                ax.plot(k, c[key], color=color, label=label, marker="o", markersize=3.5)
        axes[row, 1].axhline(1.0, color=MUTED, lw=1, ls="--")
        axes[row, 1].text(0.3, 1.0, "real", color=INK2, va="bottom", fontsize=9)
        axes[row, 2].axhline(real_cv, color=MUTED, lw=1, ls="--")
        axes[row, 2].text(0.3, real_cv, "real", color=INK2, va="bottom", fontsize=9)
        axes[row, 0].legend(loc="upper right")
        axes[row, 0].set_ylabel(f"{base} vs {new}", fontweight="bold", color=INK)
    for ax, t in zip(axes[0], ("Validation loss", "BeatPow (1 = real)", "Body-size variation (BoneCV)")):
        ax.set_title(t)
    for ax in axes[1]:
        ax.set_xlabel("training step (thousands)")
    fig.suptitle("Training checks: beat attention vs its baseline (8 val clips per check)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_checks.png"), dpi=150)
    plt.close(fig)


def fig_guidance():
    p = json.load(open(os.path.join(run_dir("2.1b"), "probe_guidance.json")))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for vary, color, label in (("music", BLUE, "vary music weight (genre = 2)"),
                               ("genre", ORANGE, "vary genre weight (music = 2)")):
        rows = [r for r in p["rows"] if r["vary"] == vary]
        w = [r["w_music" if vary == "music" else "w_genre"] for r in rows]
        for ax, key in zip(axes, ("genre_acc", "music_resp", "FID_k")):
            ax.plot(w, [r[key] for r in rows], color=color, marker="o", markersize=4, label=label)
    axes[0].axhline(p["real_genre_acc"], color=MUTED, lw=1, ls="--")
    axes[0].text(0, p["real_genre_acc"], "real dances", color=INK2, va="bottom", fontsize=9)
    best = next(r for r in p["rows"] if r["vary"] == "music" and r["w_music"] == 1.0)
    for ax, key in zip(axes, ("genre_acc", "music_resp", "FID_k")):
        ax.scatter([1.0], [best[key]], s=90, facecolor="none", edgecolor=INK, lw=1.2, zorder=5)
    axes[0].annotate("genre 2, music 1", (1.0, best["genre_acc"]), xytext=(1.5, best["genre_acc"] + 0.06),
                     color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    axes[0].set_title("Genre accuracy (requested genre)")
    axes[1].set_title("Motion change when the track changes")
    axes[1].set_ylabel("torso lengths")
    axes[2].set_title("FID_k (lower = more realistic)")
    for ax in axes:
        ax.set_xlabel("guidance weight")
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Separate music / genre guidance on run 2.1b (64 val windows, same noise)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "guidance_sweep.png"), dpi=150)
    plt.close(fig)


def fig_beat_lock():
    p = {n: json.load(open(os.path.join(run_dir(n), "probe_beat_shift_all.json"))) for n in ("2.3", "2.1b")
         if os.path.exists(os.path.join(run_dir(n), "probe_beat_shift_all.json"))}
    p.update({n: json.load(open(os.path.join(run_dir(n), "probe_beat_shift_all.json"))) for n in ("3.1", "3.2")
              if os.path.exists(os.path.join(run_dir(n), "probe_beat_shift_all.json"))})
    ref = p["2.1b"]
    items = [("Real dance", ref["real_lock"], BLUE), ("Real dance vs a random grid (chance)",
                                                      ref["random_grid_lock"], MUTED),
             ("Jittered frozen pose", ref["frozen_lock"], MUTED)]
    items += [(f"Model {n}", p[n]["rows"][0]["lock"], ORANGE) for n in ("2.3", "2.1b", "3.1", "3.2") if n in p]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(items) + 1.2))
    y = np.arange(len(items))[::-1]
    ax.barh(y, [v for _, v, _ in items], color=[c for _, _, c in items], height=0.6)
    ax.axvline(ref["random_grid_lock"], color=INK2, lw=1, ls="--")
    for yi, (_, v, _) in zip(y, items):
        ax.text(v + 0.001, yi, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [n for n, _, _ in items])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("beat lock (resultant length of the accent beat-phases; 0 = no relation)")
    ax.set_title(f"Beat locking, all {ref['windows']} val windows (dashed = chance)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "beat_lock.png"), dpi=150)
    plt.close(fig)


def fig_beat_grid():
    songs = {}
    for f in sorted(glob.glob(os.path.join(os.path.dirname(paths.feature_path("beats", "x")), "*.npy"))):
        v = os.path.basename(f)[:-4]
        songs.setdefault(aist.music_id(v), v)
    rows = []
    for m, v in sorted(songs.items()):
        b = np.load(paths.feature_path("beats", v)).astype(float)
        on = np.load(paths.feature_path("audio_onset", v)).astype(float).ravel()
        per = np.median(np.diff(b))
        ph = b[0] % per
        rows.append((m, _grid_score(on, ph, per) / max(_grid_score(on, (ph + per / 2) % per, per), 1e-9)))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(rows))
    for xi, (m, v) in zip(x, rows):
        test = m in TEST_SONGS
        ax.scatter(xi, v, s=36, color=ORANGE if v <= 1.05 else BLUE, marker="D" if test else "o",
                   edgecolor=INK if test else "none", lw=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 50], ["0.5", "1", "2", "5", "10", "50"])
    ax.axhline(1.05, color=INK2, lw=1, ls="--")
    ax.text(0, 25, "ambiguous at or below 1.05 (orange): onsets as strong half a beat away", color=INK2,
            fontsize=9)
    ax.set_xticks(x, [m for m, _ in rows], rotation=90, fontsize=7.5)
    for lbl in ax.get_xticklabels():
        if lbl.get_text() in TEST_SONGS:
            lbl.set_fontweight("bold")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("onset strength on grid /\nhalf a beat away")
    ax.set_title("Beat-grid phase confidence per song (diamonds, bold = held-out test songs)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "beat_grid_confidence.png"), dpi=150)
    plt.close(fig)




def fig_proposal_guidance(path=os.path.join(paths.REPO_ROOT, "reports", "ieee-proposal", "fig_guidance.pdf")):
    """Single-column guidance figure for the IEEE proposal (its motionblue / audioteal colors, 8 pt text)."""
    p = json.load(open(os.path.join(run_dir("2.1b"), "probe_guidance.json")))
    blue, teal = "#27517C", "#246C63"
    with plt.rc_context({"font.size": 7.5, "axes.titlesize": 8, "lines.linewidth": 1.4, "font.family": "serif",
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.12))
        for vary, color, label in (("music", blue, r"vary $w_m$ ($w_g$ = 2)"), ("genre", teal, r"vary $w_g$ ($w_m$ = 2)")):
            rows = [r for r in p["rows"] if r["vary"] == vary]
            w = [r["w_music" if vary == "music" else "w_genre"] for r in rows]
            axes[0].plot(w, [r["genre_acc"] for r in rows], color=color, marker="o", markersize=2.5, label=label)
            axes[1].plot(w, [r["FID_k"] for r in rows], color=color, marker="o", markersize=2.5)
        best = next(r for r in p["rows"] if r["vary"] == "music" and r["w_music"] == 1.0)
        axes[0].scatter([1], [best["genre_acc"]], s=40, facecolor="none", edgecolor="black", lw=0.8, zorder=5)
        axes[1].scatter([1], [best["FID_k"]], s=40, facecolor="none", edgecolor="black", lw=0.8, zorder=5)
        axes[0].axhline(p["real_genre_acc"], color="#888888", lw=0.8, ls="--")
        axes[0].text(4, p["real_genre_acc"] + 0.01, "real", fontsize=6.5, color="#555555", ha="right", va="bottom")
        axes[0].set_title("Genre accuracy")
        axes[1].set_title(r"FD$_k$ (lower is better)")
        for ax in axes:
            ax.set_xlabel("guidance weight", labelpad=1)
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.tick_params(length=2, pad=1.5)
        axes[0].set_ylim(0, 0.6)
        h, lab = axes[0].get_legend_handles_labels()
        axes[1].legend(h, lab, loc="upper left", fontsize=6, handlelength=1.2, borderaxespad=0.2)
        fig.tight_layout(pad=0.3, w_pad=0.8)
        fig.savefig(path)
        plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_training, fig_guidance, fig_beat_lock, fig_beat_grid, fig_proposal_guidance):
        f()
        print("wrote", f.__name__)
