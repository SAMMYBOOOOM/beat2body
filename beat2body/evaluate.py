"""Evaluate a trained model on held-out music (Task 4): realism, diversity, rhythm, physics.

    python -m beat2body evaluate --checkpoint runs/<job>/model.pt              # val split
    python -m beat2body evaluate --checkpoint runs/<job>/model.pt --split test # final numbers only

For every window of the split (150 frames, the clips' first 5 s), the model generates a dance for
that window's music and genre. Metrics, adapted to 2D from AIST++ / Bailando / EDGE:

    FID_k   Frechet distance between generated and real *kinetic* features (per joint: mean
            horizontal and vertical speed^2, mean |acceleration|), lower = more realistic dynamics
    FID_p   the same for *pose* features (per joint: mean and std of the pelvis-centred x, y)
    Div_k / Div_p   mean pairwise distance between generated windows (higher = more varied)
    BAS     beat alignment: for each music beat, exp(-d^2 / 2 sigma^2) to the nearest kinematic beat
            (local minimum of the smoothed joint speed); sigma = 3 frames
    PFC     EDGE's physical foot contact score: mean over frames of root acceleration x the slower
            ankle's speed (each normalized by its max); lower = fewer sliding / floating feet
    speed   mean joint speed relative to the real val windows (1 = same energy, ~0 = frozen)
    BeatPow power of the pelvis-centred joint motion at 1-4 Hz (the beat and half-beat band at 80-135 BPM)
            relative to the real windows: 1 = as much rhythmic movement as real; a model that sways
            slowly instead of dancing on the beat scores far below its speed
    BoneCV  drift detector: for each window, the variation (std / mean over time) of each body bone's
            on-screen length, averaged over bones and windows. Real dance is not 0 (foreshortening);
            generated dance should match the real value, higher = the body changes size

Long-dance drift check (--long-seconds, default 20): one video per genre of the split; the model
generates that many seconds of the video's music (EDGE long-form sampling, chained windows) and the
variation over the whole piece (std / mean) of torso length, thigh length and body height is
compared with the real dancer on the same music. This catches size drift that builds up across
windows, which the per-window BoneCV cannot see.

Features are standardized with the real val windows' mean / std before FID and diversity.
Reference rows give the scale: real train windows vs real val ("Real (train)"), the real val
windows themselves, real val dances paired with other songs (BAS only), and a frozen mean pose.
"""
import json
import os

import numpy as np
import torch
from scipy import linalg
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

from beat2body.data import ANKLES, CONTACT, DanceClips
from beat2body.model.diffusion import absolute_joints, bone_lengths

DOC = __doc__


def add_args(p):
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--guidance", type=float, default=None, help="default: the trained value")
    from beat2body.train import add_guidance_args, add_postprocess_args
    add_guidance_args(p)
    add_postprocess_args(p)
    p.add_argument("--style", choices=["none", "genre", "real"], default="none",
                   help="style-conditioned models: manner descriptors to generate with: none (the model chooses), "
                        "genre (the genre's median over training windows) or real (each real window's own; an "
                        "upper bound and a controllability check)")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--eta", type=float, default=1.0, help="DDIM noise: 1 = EDGE (stochastic), 0 = deterministic")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-windows", type=int, default=None, help="evaluate on fewer windows (quick check)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--long-seconds", type=float, default=20.0, help="long-dance drift check length; 0 = skip")
    p.add_argument("--subset", choices=["all", "advanced", "basic"], default="all",
                   help="only the advanced (sFM) or basic (sBM) dances of the split, e.g. to compare with models "
                        "evaluated before the basic dances were added (advanced)")
    p.add_argument("--tag", default="", help="suffix for the result file: eval_<split>_<tag>.json (sweeps)")
    p.add_argument("--device", default="auto")


# ---------------------------------------------------------------------------
# features and metrics on absolute joints (N, L, 17, 2), torso lengths
# ---------------------------------------------------------------------------
def kinetic_features(j):
    v = np.diff(j, axis=1)
    a = np.diff(v, axis=1)
    return np.concatenate([(v[..., 0] ** 2).mean(1), (v[..., 1] ** 2).mean(1),
                           np.linalg.norm(a, axis=-1).mean(1)], axis=1)              # (N, 51)


def pose_features(j):
    rel = j - j[:, :, [11, 12]].mean(2, keepdims=True)                              # pelvis-centred
    return np.concatenate([rel.mean(1).reshape(len(j), -1), rel.std(1).reshape(len(j), -1)], axis=1)   # (N, 68)


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)          # SciPy >= 1.16 removed `disp`; returns only the matrix
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def diversity(f, n_pairs=2000, seed=0):
    rng = np.random.default_rng(seed)
    i, k = rng.integers(0, len(f), n_pairs), rng.integers(0, len(f), n_pairs)
    keep = i != k
    return float(np.linalg.norm(f[i[keep]] - f[k[keep]], axis=1).mean())


def kinematic_beats(j, sigma=5):
    speed = gaussian_filter1d(np.linalg.norm(np.diff(j, axis=0), axis=-1).mean(1), sigma)
    return argrelextrema(speed, np.less)[0]


def bas(j, music_beats, sigma=3.0):
    mb = music_beats[(music_beats >= 0) & (music_beats < len(j) - 1)]
    kb = kinematic_beats(j)
    if len(mb) == 0 or len(kb) == 0:
        return 0.0
    d = np.abs(mb[:, None] - kb[None, :]).min(1)
    return float(np.exp(-d ** 2 / (2 * sigma ** 2)).mean())


def pfc(j):
    """EDGE's PFC in 2D: root acceleration x min ankle speed, each normalized by its max."""
    root = j[:, [11, 12]].mean(1)
    acc = np.linalg.norm(np.diff(root, n=2, axis=0), axis=-1)
    foot_v = np.linalg.norm(np.diff(j[:, list(ANKLES)], axis=0), axis=-1)[1:]          # (L-2, 2)
    acc = acc / max(acc.max(), 1e-8)
    foot_v = foot_v / np.maximum(foot_v.max(0, keepdims=True), 1e-8)
    return float((acc * foot_v.min(1)).mean())


def bone_cv(j):
    lengths = bone_lengths(torch.as_tensor(j)).numpy()                                # (N, L, 12)
    return float((lengths.std(1) / np.maximum(lengths.mean(1), 1e-6)).mean())


def size_variation(j):
    """Whole-piece variation (std / mean) of torso length, mean thigh length and body height; j (T, 17, 2)."""
    sh, hp = j[:, [5, 6]].mean(1), j[:, [11, 12]].mean(1)
    torso = np.linalg.norm(sh - hp, axis=-1)
    thigh = (np.linalg.norm(j[:, 11] - j[:, 13], axis=-1) + np.linalg.norm(j[:, 12] - j[:, 14], axis=-1)) / 2
    height = j[:, :, 1].max(1) - j[:, :, 1].min(1)
    parts = (("torso", torso), ("thigh", thigh), ("height", height))
    return {k: float(v.std() / max(v.mean(), 1e-6)) for k, v in parts}


def long_drift(gen, bundle, split, seconds, seed, guidance=None, music_guidance=None, args=None):
    """Generated vs real whole-piece size variation, one video per genre (the first one at least `seconds` long)."""
    from beat2body.data import load_meta, video_motion
    from beat2body.train import generate_long, postprocess
    cfg = bundle["config"]
    rows = load_meta(cfg["tool"], split)
    length = {}
    for r in rows:
        length[r["video_id"]] = max(length.get(r["video_id"], 0), int(r["end_frame"]))
    vids = {}
    for r in rows:
        if length[r["video_id"]] >= seconds * 30:
            vids.setdefault(r["category"], r["video_id"])
    gen_v, real_v = [], []
    for i, (cat, vid) in enumerate(sorted(vids.items())):
        print(f"[evaluate] long dance {i + 1}/{len(vids)}: {vid}", flush=True)
        motion, _, _ = generate_long(gen, bundle, vid, seconds, guidance=guidance, seed=seed,
                                     music_guidance=music_guidance)
        motion = postprocess(motion, args)
        real, _, _ = video_motion(cfg["tool"], cat, vid, cfg.get("scale_norm", "dataset"))
        gen_v.append(size_variation(absolute_joints(motion).numpy()))
        real_v.append(size_variation(absolute_joints(torch.from_numpy(real[:len(motion)])).numpy()))
    keys = ("torso", "thigh", "height")
    return ({k: float(np.mean([g[k] for g in gen_v])) for k in keys},
            {k: float(np.mean([r[k] for r in real_v])) for k in keys}, len(vids))


def beat_power(j, lo=1.0, hi=4.0, fps=30):
    """Mean spectral power of the pelvis-centred joints' motion between lo and hi Hz; j (N, L, 17, 2)."""
    rel = j - j[:, :, [11, 12]].mean(2, keepdims=True)
    f = np.abs(np.fft.rfft(rel - rel.mean(1, keepdims=True), axis=1)) ** 2
    fr = np.fft.rfftfreq(j.shape[1], 1 / fps)
    return float(f[:, (fr >= lo) & (fr < hi)].sum(1).mean())


def mean_speed(j):
    return float(np.linalg.norm(np.diff(j, axis=1), axis=-1).mean())


def score(name, j, real_j, beats, z_k, z_p, contact=None):
    fk, fp = (kinetic_features(j) - z_k[0]) / z_k[1], (pose_features(j) - z_p[0]) / z_p[1]
    rk, rp = (kinetic_features(real_j) - z_k[0]) / z_k[1], (pose_features(real_j) - z_p[0]) / z_p[1]
    row = {"name": name, "FID_k": frechet(fk, rk), "FID_p": frechet(fp, rp), "Div_k": diversity(fk),
           "Div_p": diversity(fp), "BAS": float(np.mean([bas(x, b) for x, b in zip(j, beats)])),
           "PFC": float(np.mean([pfc(x) for x in j])), "speed": mean_speed(j) / mean_speed(real_j),
           "BeatPow": beat_power(j) / beat_power(real_j), "BoneCV": bone_cv(j)}
    if contact is not None:
        row["contact"] = float(contact)
    return row


SUBSETS = {"all": None, "advanced": "sFM", "basic": "sBM"}


def windows(ds, n=None):
    """Deterministic windows of a split: absolute joints, music, genre, beat frames, contacts."""
    items = [ds[i] for i in range(len(ds) if n is None else min(n, len(ds)))]
    motion = torch.stack([x["motion"] for x in items])
    beats = [torch.nonzero(x["beats"]).flatten().numpy().astype(float) for x in items]
    return motion, torch.stack([x["audio"] for x in items]), torch.stack([x["category"] for x in items]), beats


def style_inputs(gen, cfg, mode, real_m, genre, train_m, train_genre):
    """(N, D) standardized descriptors for --style, or None."""
    if mode == "none" or not getattr(gen.model, "style_dim", 0):
        return None
    from beat2body.model.style import descriptors, standardize

    def desc(m):
        v = torch.ones(m.shape[0], m.shape[1], 17, dtype=torch.bool)
        return standardize(descriptors(m.float(), v, torch.ones(m.shape[:2], dtype=torch.bool)), cfg["style_stats"])
    if mode == "real":
        return desc(real_m)
    d = desc(train_m)
    med = {int(c): d[train_genre == c].median(0).values for c in train_genre.unique()}
    return torch.stack([med.get(int(c), d.median(0).values) for c in genre])


def run(args):
    from beat2body.train import load_generator
    dev = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device
    from beat2body.train import guidance_weights
    gen, bundle = load_generator(args.checkpoint, dev)
    cfg = bundle["config"]
    gw, mw = guidance_weights(args, cfg["guidance"])
    kw = dict(seq_len=cfg["seq_len"], feature_set=cfg["feature_set"], stats=bundle["audio_stats"], augment=False,
              scale_norm=cfg.get("scale_norm", "dataset"))
    def subset(ds):
        tag = SUBSETS[args.subset]
        if tag:
            ds.rows = [r for r in ds.rows if f"_{tag}_" in r["video_id"]]
        return ds
    real_m, audio, genre, beats = windows(subset(DanceClips(cfg["tool"], args.split, **kw)), args.max_windows)
    train_m, _, train_genre, train_beats = windows(subset(DanceClips(cfg["tool"], "train", **kw)), len(real_m))
    real = absolute_joints(real_m).numpy()
    z_k = (kinetic_features(real).mean(0), kinetic_features(real).std(0) + 1e-8)
    z_p = (pose_features(real).mean(0), pose_features(real).std(0) + 1e-8)

    style = style_inputs(gen, cfg, getattr(args, "style", "none"), real_m, genre, train_m, train_genre)
    torch.manual_seed(args.seed)
    out = []
    import time
    t0 = time.time()
    for s in range(0, len(audio), args.batch_size):
        print(f"[evaluate] generating test windows {s}/{len(audio)} ({time.time() - t0:.0f} s)", flush=True)
        out.append(gen.generate(audio[s:s + args.batch_size], genre[s:s + args.batch_size],
                                first=gen.normalizer.norm(real_m[s:s + args.batch_size, 0].to(dev)),
                                guidance=gw, steps=args.steps, eta=args.eta, music_guidance=mw,
                                style=None if style is None else style[s:s + args.batch_size]).cpu())
    from beat2body.train import postprocess
    gen_m = postprocess(torch.cat(out), args)
    generated = absolute_joints(gen_m).numpy()
    static = np.repeat(real.mean((0, 1), keepdims=True), len(real), 0).repeat(real.shape[1], 1)
    static = static + 1e-4 * np.random.default_rng(0).standard_normal(static.shape)
    shuffled = [beats[i] for i in np.random.default_rng(1).permutation(len(beats))]

    rows = [
        score(f"Model ({cfg['model']}, {sum(p.numel() for p in gen.model.parameters()) / 1e6:.1f}M)",
              generated, real, beats, z_k, z_p, (gen_m[..., CONTACT] > 0.5).float().mean()),
        score("Real (train windows)", absolute_joints(train_m).numpy(), real, train_beats, z_k, z_p),
        score(f"Real ({args.split}) itself", real, real, beats, z_k, z_p, real_m[..., CONTACT].mean()),
        score("Real dance + other music", real, real, shuffled, z_k, z_p),
        score("Frozen mean pose", static, real, beats, z_k, z_p),
    ]
    print(f"\n{args.split} ({args.subset}): {len(real)} windows of {cfg['seq_len']} frames; "
          f"checkpoint {args.checkpoint}")
    cols = ["FID_k", "FID_p", "Div_k", "Div_p", "BAS", "PFC", "speed", "BeatPow", "BoneCV"]
    print(f"{'':34s}" + "".join(f"{c:>9s}" for c in cols))
    for r in rows:
        print(f"{r['name']:34s}" + "".join(f"{r[c]:9.3f}" for c in cols))
    print("FID: lower = closer to real; Div: higher = more varied; BAS: higher = more on the beat;"
          " PFC: lower = less foot sliding; speed: 1 = as energetic as real;\n"
          "BeatPow: 1 = as much movement on the beat (1-4 Hz) as real; BoneCV: should match real")
    from beat2body.data import CONTACT_THRESHOLD
    from beat2body.model.footlock import feet_stats
    feet = {"generated": feet_stats(generated, CONTACT_THRESHOLD, (gen_m[..., CONTACT] > 0.5).numpy()),
            "real": feet_stats(real, CONTACT_THRESHOLD, (real_m[..., CONTACT] > 0.5).numpy())}
    print("\nFeet and root (torso lengths / frame): planted = median speed of the slower ankle; no foot = share of"
          " frames with both ankles moving; jitter = mean |root acceleration|;\nper contact run (model output /"
          " labels): drift = stance drift (torso lengths), frames = contact length, share = contact share,"
          " swing = ankle speed outside contact")
    keys = ("planted_foot_speed", "no_foot_planted", "root_jitter", "stance_drift", "contact_frames", "contact_share",
            "swing_speed")
    print(f"{'':12s}" + "".join(f"{h:>9s}" for h in ("planted", "no foot", "jitter", "drift", "frames", "share",
                                                       "swing")))
    for name, v in feet.items():
        print(f"{name:12s}" + "".join(f"{v[k]:9.4f}" for k in keys))
    from beat2body.physics import physics_stats, summarize
    phys = {"generated": summarize([physics_stats(x) for x in generated]),
            "real": summarize([physics_stats(x) for x in real])}
    print("\nPhysics (2D; g ~0.0218 torso lengths / frame^2): jumps = flights with the CoM rising then falling; "
          "acc_y = median CoM acceleration in flight (real jumps ~g); down > g = share of frames accelerating "
          "down faster than g")
    print(f"{'':12s}{'jumps':>9s}{'acc_y':>9s}{'|acc_x|':>9s}{'down>g':>9s}")
    for name, v in phys.items():
        print(f"{name:12s}{v['flights']:9d}{v['flight_acc_y_median']:9.4f}{v['flight_acc_x_abs_median']:9.4f}"
              f"{100 * v['acc_down_over_g']:8.2f}%")
    drift = None
    if args.long_seconds > 0:
        g, r, n = long_drift(gen, bundle, args.split, args.long_seconds, args.seed, gw, mw, args)
        drift = {"generated": g, "real": r, "videos": n, "seconds": args.long_seconds}
        print(f"\nLong-dance drift ({n} videos x {args.long_seconds:.0f} s, variation std/mean over the piece):")
        print(f"{'':12s}{'torso':>9s}{'thigh':>9s}{'height':>9s}")
        for name, v in (("generated", g), ("real", r)):
            print(f"{name:12s}" + "".join(f"{v[k]:9.3f}" for k in ("torso", "thigh", "height")))
    name = f"eval_{args.split}" + (f"_{args.tag}" if args.tag else "") + ".json"
    path = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), name)
    with open(path, "w") as f:
        json.dump({"split": args.split, "subset": args.subset, "windows": len(real), "guidance": gw,
                   "music_guidance": mw, "rows": rows, "long_drift": drift,
                   "feet": feet, "foot_lock": bool(args.foot_lock), "root_smooth": args.root_smooth,
                   "style": getattr(args, "style", "none"), "physics": phys}, f, indent=1)
    print(f"wrote {path}")
