"""Controlled probes of a trained diffusion model: does the motion follow the conditions?

    python -m beat2body probe guidance --checkpoint runs/<job>/model.pt      # separate music / genre guidance
    python -m beat2body probe beat-shift --checkpoint runs/<job>/model.pt    # shifted beat grid

Both use fixed val windows (default: the advanced dances, 64 windows spread over the split) and the same
initial noise for every variant (fixed seed), so differences between rows come from the conditions alone.

guidance   Sweeps the separate guidance weights x0 = uncond + w_genre (genre_only - uncond)
           + w_music (joint - genre_only), one weight at a time around --base (equal weights = the
           single-weight guidance used so far). Per setting:
             music_resp  how much the motion changes when the track is swapped (another window's music,
                         same genre): mean joint distance, torso lengths; higher = follows the music more
             genre_resp  the same when another genre is requested on the same music
             genre_acc   a nearest-centroid genre classifier (kinetic + pose features of the real train
                         windows) applied to the generated windows: share classified as the requested genre;
                         genre_acc_swap: the same after requesting the other genre. Its accuracy on the real
                         val windows is printed as the reference
             lock        beat locking: resultant length of the motion accents' beat phases (0 = no relation
                         to the beat, 1 = every accent at the same phase)
             FID_k, BeatPow, speed   as in `evaluate`
guidance with --base-music / --base-genre sweeps around other centres.

beat-shift Moves the beat grid by a fraction of a beat (the beat phase and EDGE's beat one-hot; the rest of the
           music unchanged; same tempo, genre and seed) and measures where the motion accents (local minima
           of the smoothed joint speed, as in BAS) fall relative to the ORIGINAL grid. A model that follows
           the grid moves its accents by the same fraction: follow = measured shift / applied shift (1 = fully,
           0 = ignores the beat features), with a bootstrap standard error over windows. BeatPow is reported
           alongside, so more movement alone does not count as better timing. Calibration rows: the beat lock of
           the real dances, of the real dances against randomly shifted grids (chance level) and of a jittered
           frozen pose; the accent detector is the BAS one, which a jittered frozen pose can fool, so read a
           model's lock against these.
"""
import json
import os

import numpy as np
import torch

from beat2body.data import DanceClips, music_input, normalize_audio
from beat2body.evaluate import (
    SUBSETS,
    beat_power,
    frechet,
    kinematic_beats,
    kinetic_features,
    mean_speed,
    pose_features,
    windows,
)
from beat2body.model.diffusion import absolute_joints

DOC = __doc__
EDGE_BEAT_COL = 34          # EDGE features: envelope 1, MFCC 20, chroma 12, onset peaks 1, beat one-hot 1


def add_args(p):
    sub = p.add_subparsers(dest="probe", required=True)
    for name in ("guidance", "beat-shift"):
        sp = sub.add_parser(name)
        sp.add_argument("--checkpoint", required=True)
        sp.add_argument("--subset", choices=list(SUBSETS), default="advanced")
        sp.add_argument("--windows", type=int, default=64, help="val windows, spread over the split")
        sp.add_argument("--steps", type=int, default=50)
        sp.add_argument("--eta", type=float, default=1.0)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--batch-size", type=int, default=64)
        sp.add_argument("--device", default="auto")
        sp.add_argument("--tag", default="")
        if name == "guidance":
            sp.add_argument("--weights", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0, 4.0],
                            help="values for the weight being varied")
            sp.add_argument("--base-genre", type=float, default=None, help="default: the trained guidance")
            sp.add_argument("--base-music", type=float, default=None, help="default: the trained guidance")
        else:
            sp.add_argument("--shifts", type=float, nargs="+", default=[0.25, 0.5], help="beats")
            sp.add_argument("--guidance", type=float, default=None)
            sp.add_argument("--music-guidance", type=float, default=None)


# ---------------------------------------------------------------------------
def beat_phases(frames, beats):
    """Phase in [0, 1) within the beat grid of each frame (NaN outside the grid)."""
    b = np.sort(np.asarray(beats, np.float64))
    t = np.asarray(frames, np.float64)
    out = np.full(len(t), np.nan)
    if len(b) < 2:
        return out
    i = np.clip(np.searchsorted(b, t, side="right") - 1, 0, len(b) - 2)
    ph = (t - b[i]) / (b[i + 1] - b[i])
    inside = (t >= b[0]) & (t <= b[-1])
    out[inside] = ph[inside]
    return out % 1.0


def accent_phases(j_windows, beats_list):
    """Per window: beat phases of the motion accents (local minima of the smoothed joint speed)."""
    out = [beat_phases(kinematic_beats(j), b) for j, b in zip(j_windows, beats_list)]
    return [p[~np.isnan(p)] for p in out]


def circular(phase_list, idx=None):
    """(angle in beats in [0, 1), resultant length) of the pooled phases of the windows idx (default all)."""
    ph = np.concatenate([phase_list[i] for i in (range(len(phase_list)) if idx is None else idx)] + [[]])
    if len(ph) == 0:
        return float("nan"), 0.0
    z = np.exp(2j * np.pi * ph).mean()
    return float((np.angle(z) / (2 * np.pi)) % 1.0), float(np.abs(z))


def accent_phase(j_windows, beats_list):
    """Circular mean of the accent phases over all windows -> (angle in beats in [0, 1), resultant length)."""
    return circular(accent_phases(j_windows, beats_list))


def wrap(x):
    """Phase difference in beats -> (-0.5, 0.5]."""
    return (x + 0.5) % 1.0 - 0.5


def pick_windows(ds, n):
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).round().astype(int)
    ds.rows = [ds.rows[i] for i in sorted(set(idx.tolist()))]
    return ds


def load(args):
    from beat2body.train import load_generator
    dev = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device
    gen, bundle = load_generator(args.checkpoint, dev)
    cfg = bundle["config"]
    kw = dict(seq_len=cfg["seq_len"], feature_set=cfg["feature_set"], stats=bundle["audio_stats"], augment=False,
              scale_norm=cfg.get("scale_norm", "dataset"))

    def subset(ds):
        tag = SUBSETS[args.subset]
        if tag:
            ds.rows = [r for r in ds.rows if f"_{tag}_" in r["video_id"]]
        return ds
    val = pick_windows(subset(DanceClips(cfg["tool"], "val", **kw)), args.windows)
    return gen, bundle, cfg, val, kw, subset, dev


def generate(gen, audio, genre, args, guidance=None, music_guidance=None):
    """Same initial noise for every call (generator re-seeded per batch)."""
    out = []
    for s in range(0, len(audio), args.batch_size):
        g = torch.Generator(device=gen.device).manual_seed(args.seed + s)
        out.append(gen.generate(audio[s:s + args.batch_size], genre[s:s + args.batch_size], guidance=guidance,
                                music_guidance=music_guidance, steps=args.steps, eta=args.eta, generator=g).cpu())
    return torch.cat(out)


def save(args, result):
    name = f"probe_{args.probe.replace('-', '_')}" + (f"_{args.tag}" if args.tag else "") + ".json"
    path = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), name)
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
def run_guidance(args):
    gen, bundle, cfg, val, kw, subset, _ = load(args)
    real_m, audio, genre, beats = windows(val)
    n, n_genres = len(real_m), cfg["n_genres"]
    real = absolute_joints(real_m).numpy()
    # swapped music: another window of the same genre where possible (else any other window)
    rng = np.random.default_rng(0)
    perm = np.arange(n)
    for gidx in np.unique(genre.numpy()):
        members = np.flatnonzero(genre.numpy() == gidx)
        if len(members) > 1:
            perm[members] = np.roll(rng.permutation(members), 1)
    lonely = perm == np.arange(n)
    perm[lonely] = (np.flatnonzero(lonely) + n // 2) % n
    other_genre = (genre + 1 + torch.from_numpy(rng.integers(0, n_genres - 1, n))) % n_genres

    # nearest-centroid genre classifier on standardized kinetic + pose features of real train windows
    train_m, _, train_g, _ = windows(subset(DanceClips(cfg["tool"], "train", **kw)))
    tj = absolute_joints(train_m).numpy()
    tf = np.concatenate([kinetic_features(tj), pose_features(tj)], 1)
    mu, sd = tf.mean(0), tf.std(0) + 1e-8
    cents = np.stack([((tf - mu) / sd)[train_g.numpy() == k].mean(0) if (train_g.numpy() == k).any()
                      else np.full(tf.shape[1], np.inf) for k in range(n_genres)])

    def classify(j):
        f = (np.concatenate([kinetic_features(j), pose_features(j)], 1) - mu) / sd
        return np.linalg.norm(f[:, None] - cents[None], axis=-1).argmin(1)

    real_acc = float((classify(real) == genre.numpy()).mean())
    rk = kinetic_features(real)
    kz = (rk.mean(0), rk.std(0) + 1e-8)

    def dist(a, b):
        return float(np.linalg.norm(a - b, axis=-1).mean())

    g0 = cfg["guidance"]
    bg = g0 if args.base_genre is None else args.base_genre
    bm = g0 if args.base_music is None else args.base_music
    settings = [("music", bg, w) for w in args.weights] + [("genre", w, bm) for w in args.weights]
    rows = []
    print(f"{n} val windows ({args.subset}); nearest-centroid genre accuracy on the real windows: {real_acc:.2f}")
    cols = ["music_resp", "genre_resp", "genre_acc", "genre_acc_swap", "lock", "FID_k", "BeatPow", "speed"]
    print(f"{'vary':6s}{'w_genre':>8s}{'w_music':>8s}" + "".join(f"{c:>15s}" for c in cols))
    for vary, wg, wm in settings:
        base = absolute_joints(generate(gen, audio, genre, args, wg, wm)).numpy()
        swap_m = absolute_joints(generate(gen, audio[perm], genre, args, wg, wm)).numpy()
        swap_g = absolute_joints(generate(gen, audio, other_genre, args, wg, wm)).numpy()
        k = kinetic_features(base)
        row = {"vary": vary, "w_genre": wg, "w_music": wm,
               "music_resp": dist(base, swap_m), "genre_resp": dist(base, swap_g),
               "genre_acc": float((classify(base) == genre.numpy()).mean()),
               "genre_acc_swap": float((classify(swap_g) == other_genre.numpy()).mean()),
               "lock": accent_phase(base, beats)[1],
               "FID_k": frechet((k - kz[0]) / kz[1], (rk - kz[0]) / kz[1]),
               "BeatPow": beat_power(base) / beat_power(real), "speed": mean_speed(base) / mean_speed(real)}
        rows.append(row)
        print(f"{vary:6s}{wg:8.2f}{wm:8.2f}" + "".join(f"{row[c]:15.3f}" for c in cols), flush=True)
    real_lock = accent_phase(real, beats)[1]
    print(f"real: lock {real_lock:.3f}; responses are mean joint distances in torso lengths (same noise)")
    save(args, {"checkpoint": args.checkpoint, "subset": args.subset, "windows": n, "real_genre_acc": real_acc,
                "real_lock": real_lock, "rows": rows})


# ---------------------------------------------------------------------------
def shifted_cond(val, i, shift, cfg, stats):
    """Music input of window i with the beat grid moved by `shift` beats (phase + EDGE beat one-hot)."""
    r = val.rows[i]
    vid, s, L = r["video_id"], int(r["start_frame"]), val.seq_len
    b = val.music[vid]["beats"]
    period = float(np.median(np.diff(b))) if len(b) > 1 else 15.0
    nb = b + shift * period
    full = val.music[vid]["audio"] if shift == 0 else music_input(vid, cfg["feature_set"], stats, nb)
    cond = full[s:s + L].copy()
    if shift != 0 and cfg["feature_set"] == "edge":         # (music_input already moved the beat phase)
        onehot = np.zeros(len(full), np.float32)
        bi = np.round(nb).astype(int)
        onehot[bi[(bi >= 0) & (bi < len(full))]] = 1
        raw = np.zeros((len(full), len(stats["mean"])), np.float32)
        raw[:, EDGE_BEAT_COL] = onehot
        cond[:, EDGE_BEAT_COL] = normalize_audio(raw, stats)[s:s + L, EDGE_BEAT_COL]
    return torch.from_numpy(cond), b - s


def run_beat_shift(args):
    gen, bundle, cfg, val, _, _, _ = load(args)
    stats = bundle["audio_stats"]
    real_m, _, genre, _ = windows(val)
    n = len(real_m)
    real = absolute_joints(real_m).numpy()
    from beat2body.train import guidance_weights
    gw, mw = guidance_weights(args, cfg["guidance"])
    rng = np.random.default_rng(0)
    boot = [rng.integers(0, n, n) for _ in range(200)]          # window resamples for the error bars
    grid = [shifted_cond(val, i, 0.0, cfg, stats)[1] for i in range(n)]
    period = [float(np.median(np.diff(b))) if len(b) > 1 else 15.0 for b in grid]

    # calibration: beat locking of real dances on their own grid vs on a randomly shifted grid (no relation:
    # the lock that chance alone gives with this many accents), and of a jittered frozen pose
    real_ph = accent_phases(real, grid)
    real_angle, real_lock = circular(real_ph)
    rand_lock = float(np.mean([circular(accent_phases(real, [b + rng.uniform(0, 1) * p for b, p in
                                                             zip(grid, period)]))[1] for _ in range(20)]))
    frozen = np.repeat(real.mean((0, 1), keepdims=True), n, 0).repeat(real.shape[1], 1)
    frozen = frozen + 1e-3 * rng.standard_normal(frozen.shape)
    frozen_lock = circular(accent_phases(frozen, grid))[1]
    print(f"{n} val windows ({args.subset}). Beat lock: real dance {real_lock:.3f} (accent phase {real_angle:.3f}),"
          f" real vs a random grid {rand_lock:.3f}, jittered frozen pose {frozen_lock:.3f}")
    print(f"{'shift':>7s}{'phase':>9s}{'moved':>9s}{'+/-':>7s}{'follow':>9s}{'lock':>8s}{'BeatPow':>9s}{'speed':>8s}")
    results, base_ph = [], None
    for shift in [0.0] + list(args.shifts):
        audio = torch.stack([shifted_cond(val, i, shift, cfg, stats)[0] for i in range(n)])
        j = absolute_joints(generate(gen, audio, genre, args, gw, mw)).numpy()
        ph = accent_phases(j, grid)                        # accents measured against the ORIGINAL grid
        angle, lock = circular(ph)
        if shift == 0.0:
            base_ph = ph
        moved = wrap(angle - circular(base_ph)[0])
        se = float(np.std([wrap(circular(ph, idx)[0] - circular(base_ph, idx)[0]) for idx in boot])) if shift else 0.0
        row = {"shift": shift, "phase": angle, "moved": moved, "moved_se": se,
               "follow": moved / shift if shift else None, "follow_se": se / shift if shift else None,
               "lock": lock, "BeatPow": beat_power(j) / beat_power(real), "speed": mean_speed(j) / mean_speed(real)}
        results.append(row)
        f = f"{row['follow']:9.2f}" if shift else f"{'':9s}"
        print(f"{shift:7.2f}{angle:9.3f}{moved:9.3f}{se:7.3f}{f}{lock:8.3f}{row['BeatPow']:9.3f}{row['speed']:8.3f}",
              flush=True)
    print("phase / moved: accent phase vs the original grid, beats (+/-: bootstrap s.e. over windows); follow = moved /"
          " shift (1 = follows the grid). A lock near the random-grid value means no measurable beat relation.")
    save(args, {"checkpoint": args.checkpoint, "subset": args.subset, "windows": n, "guidance": gw,
                "music_guidance": mw, "real_phase": real_angle, "real_lock": real_lock, "random_grid_lock": rand_lock,
                "frozen_lock": frozen_lock, "rows": results})


def run(args):
    {"guidance": run_guidance, "beat-shift": run_beat_shift}[args.probe](args)
