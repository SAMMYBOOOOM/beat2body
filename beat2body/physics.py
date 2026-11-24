"""2D physical-plausibility measures for real and generated dances (fixed, level camera: image down = gravity).

All inputs are absolute joints (T, 17, 2) in body units (torso lengths, image y pointing down). The same contact rule
(ankle speed, data.contact_labels) is used for real and generated motion.

    centre of mass   segment-weighted (Winter's mass fractions) from the COCO-17 joints
    flight           frames where neither ankle is planted and both are >= air_margin (0.15 torso lengths) above
                     the local floor, in runs of >= min_flight frames in which the CoM rises then falls; per run, a
                     parabola fitted to the CoM gives its vertical and horizontal acceleration (in flight: vertical
                     = g, horizontal = 0). Calibrated on the real advanced dances: 1,061 jumps, CoM falls at 0.0179
                     torso lengths / frame^2 vs g ~0.0218 (torso ~0.5 m); a looser rule (any height above the floor)
                     catches toe stance and shuffles and measures 0.003, so it is not a jump detector
    floor            local floor height = the lowest planted ankle within +/- `floor_window` frames; a slow ankle
                     counts as planted only near that floor (an ankle at the top of a jump is slow but airborne);
                     penetration = how far a swinging ankle goes below it. Depth-confounded: a foot swinging forward
                     (toward the camera) projects lower in the image, so compare with real dancers only, never use it
                     as a training penalty. Balance shares the caveat (the support is flattened onto one image axis)
    balance          when standing (not in flight, a foot planted), the share of frames whose CoM is horizontally
                     outside the planted ankles' span by more than `foot_margin` (a foot's length, in torso units)

    python -m beat2body physics ...   (see reports/2026-11-22_physics/physics_check.py)
"""
import numpy as np

from beat2body.data import contact_labels

# Winter (2009) mass fractions; segment CoM as a fraction of the way from the proximal joint
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
SEGMENTS = (                      # (proximal, distal, mass fraction, CoM position from proximal)
    ((L_SH, L_EL), 0.028, 0.436), ((R_SH, R_EL), 0.028, 0.436),
    ((L_EL, L_WR), 0.022, 0.43), ((R_EL, R_WR), 0.022, 0.43),           # forearm + hand
    ((L_HIP, L_KN), 0.100, 0.433), ((R_HIP, R_KN), 0.100, 0.433),
    ((L_KN, L_AN), 0.061, 0.5), ((R_KN, R_AN), 0.061, 0.5),             # shank + foot
)
HEAD, TRUNK = 0.081, 0.497


def centre_of_mass(j):
    """(..., 17, 2) -> (..., 2)."""
    sh, hp = j[..., [L_SH, R_SH], :].mean(-2), j[..., [L_HIP, R_HIP], :].mean(-2)
    head = j[..., 0:5, :].mean(-2)
    com = HEAD * head + TRUNK * (sh + 0.5 * (hp - sh))
    for (a, b), m, f in SEGMENTS:
        com = com + m * (j[..., a, :] + f * (j[..., b, :] - j[..., a, :]))
    return com


def _runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def local_floor(ankle_y, planted, window=None):
    """(T, 2) ankle heights and planted flags -> (T,) floor height: the lowest planted ankle (largest y) per frame,
    linearly interpolated between planted frames (NaN if never planted); with `window`, the lowest planted ankle
    within +/- window frames (robust to slow ankles at the top of a jump)."""
    T = len(ankle_y)
    y = np.where(planted, ankle_y, -np.inf).max(1)
    if window:
        y = np.array([y[max(0, t - window):t + window + 1].max() for t in range(T)])
    known = np.isfinite(y)
    if not known.any():
        return np.full(T, np.nan)
    return np.interp(np.arange(T), np.flatnonzero(known), y[known])


def video_floor(video_id, category, tool="dwpose"):
    """(horizon, k) of a video's floor in the training data's body units (scale_norm "dancer"), from
    features/floor/<video>.json: floor_y(t) = horizon + k * torso(t), torso(t) = the on-screen torso length in body
    units. None if the video has no floor fit."""
    import json
    import os

    from beat2body import paths
    from beat2body.data import dancer_torso
    from beat2body.prep.floor import floor_path
    f = floor_path(video_id)
    if not os.path.exists(f):
        return None
    r = json.load(open(f))
    if r.get("horizon_y") is None or r.get("k") is None:
        return None
    k = float(r["k"])
    if not r.get("k_ok", True):                     # implausible own fit: the dancer's typical k
        k = _dancer_k(video_id)
        if k is None:
            return None
    clean = np.load(paths.clean_keypoint_path(tool, category, video_id))
    unit = float(clean["scale"]) * (dancer_torso(tool, video_id) or 1.0)          # body unit in pixels
    return (r["horizon_y"] - float(clean["height"]) / 2) / unit, k


_DANCER_K = {}


def _dancer_k(video_id):
    """Median plausible k over the dancer's videos (cached)."""
    import glob
    import json
    import os

    from beat2body import aist, paths
    d = aist.dancer_id(video_id)
    if d not in _DANCER_K:
        ks = []
        for f in glob.glob(os.path.join(paths.features_dir(), "floor", "*.json")):
            r = json.load(open(f))
            if aist.dancer_id(r["video_id"]) == d and r.get("k_ok") and r.get("k") is not None:
                ks.append(r["k"])
        _DANCER_K[d] = float(np.median(ks)) if ks else None
    return _DANCER_K[d]


def model_floor(j, horizon, k, planted=None, window=15):
    """Floor height per frame (T,) for absolute joints j (T, 17, 2) in body units: horizon + k * body size, body size
    = the median torso length over +/- window frames; with planted (T, 2), corrected toward the planted ankles (the
    correction, which also absorbs the ankle's height above the sole, is interpolated between contacts)."""
    torso = np.linalg.norm(j[:, [L_SH, R_SH]].mean(1) - j[:, [L_HIP, R_HIP]].mean(1), axis=-1)
    T = len(j)
    size = np.array([np.median(torso[max(0, t - window):t + window + 1]) for t in range(T)])
    floor = horizon + k * size
    if planted is not None and planted.any():
        ay = j[:, [L_AN, R_AN], 1]
        r = np.where(planted, ay - floor[:, None], np.nan)
        rt = np.nanmean(np.where(planted, r, np.nan), 1)                          # per frame, NaN if no contact
        known = np.isfinite(rt)
        if known.any():
            idx = np.flatnonzero(known)
            corr = np.interp(np.arange(T), idx, rt[known])
            ker = np.ones(2 * window + 1) / (2 * window + 1)                       # smooth the correction
            corr = np.convolve(np.pad(corr, window, mode="edge"), ker, mode="valid")
            floor = floor + corr
    return floor


def physics_stats(j, min_flight=6, air_margin=0.15, foot_margin=0.25, floor_window=15, contact_margin=0.05,
                  floor_model=None, all_joints=True, acc_window=7):
    """One sequence j (T, 17, 2) -> dict of raw measures (lists, so sequences can be pooled)."""
    j = np.asarray(j, np.float64)
    T = len(j)
    ankles = j[:, [L_AN, R_AN]]
    lab = contact_labels(ankles) > 0.5                          # lab[t]: the step t-1 -> t is planted
    slow = lab.copy()
    slow[:-1] |= lab[1:]                                         # frame t is slow if the step into or out of it is
    if floor_model is None:
        floor = local_floor(ankles[..., 1], slow, floor_window)
    else:                                                        # (horizon, k) of the video's camera
        near = local_floor(ankles[..., 1], slow, floor_window)[:, None] - ankles[..., 1]
        on_floor = slow & (near <= contact_margin)
        floor = model_floor(j, *floor_model, planted=on_floor, window=floor_window)
    planted = slow & ((floor[:, None] - ankles[..., 1]) <= contact_margin)   # slow AND on the floor
    com = centre_of_mass(j)
    above = (floor[:, None] - ankles[..., 1]) > air_margin      # image y down: above the floor = smaller y
    flight = ~slow.any(1) & above.all(1) & np.isfinite(floor)   # no slow ankle at all (a held, lifted leg is no jump)
    if all_joints:                                              # hands, knees or head on the floor = support
        flight &= (floor - j[..., 1].max(1)) > air_margin
    # downward CoM acceleration on every frame (local quadratic fit over acc_window frames); the floor can only push
    # up, so it should never exceed g by more than measurement noise
    h = acc_window // 2
    tt = np.arange(-h, h + 1, dtype=np.float64)
    acc_down = [2 * np.polyfit(tt, com[i - h:i + h + 1, 1], 2)[0] for i in range(h, T - h)]
    out = {"frames": T, "flight_frames": 0, "flight_acc_y": [], "flight_acc_x": [], "flight_len": [],
           "acc_down": acc_down,
           "penetration": [], "swing_frames": 0, "balance_out": 0, "standing_frames": 0}
    t = np.arange(T, dtype=np.float64)
    for a, b in _runs(flight):
        if b - a < min_flight:
            continue
        k = int(np.argmin(com[a:b, 1]))                         # apex (image y down): must rise, then fall
        if not 0 < k < b - a - 1:
            continue
        out["flight_frames"] += b - a
        out["flight_len"].append(b - a)
        cy = np.polyfit(t[a:b] - a, com[a:b, 1], 2)
        cx = np.polyfit(t[a:b] - a, com[a:b, 0], 2)
        out["flight_acc_y"].append(2 * cy[0])
        out["flight_acc_x"].append(2 * cx[0])
    swing = ~planted & np.isfinite(floor)[:, None]
    pen = (ankles[..., 1] - floor[:, None])[swing]
    out["penetration"] = np.clip(pen, 0, None).tolist()
    out["swing_frames"] = int(swing.sum())
    standing = planted.any(1) & ~flight
    for f in np.flatnonzero(standing):
        xs = ankles[f, planted[f], 0]
        lo, hi = xs.min() - foot_margin, xs.max() + foot_margin
        out["standing_frames"] += 1
        out["balance_out"] += int(not (lo <= com[f, 0] <= hi))
    return out


def summarize(stats, g=0.0218):
    """Pool per-sequence dicts -> summary numbers (g in torso lengths / frame^2, for the downward-acceleration
    check)."""
    ad = np.concatenate([s["acc_down"] for s in stats]) if stats else np.zeros(0)
    acc_y = np.concatenate([s["flight_acc_y"] for s in stats]) if stats else np.zeros(0)
    acc_x = np.concatenate([s["flight_acc_x"] for s in stats]) if stats else np.zeros(0)
    pen = np.concatenate([s["penetration"] for s in stats]) if stats else np.zeros(0)
    frames = sum(s["frames"] for s in stats)
    return {
        "flight_share": sum(s["flight_frames"] for s in stats) / max(frames, 1),
        "flights": int(len(acc_y)),
        "flight_acc_y_median": float(np.median(acc_y)) if len(acc_y) else float("nan"),
        "flight_acc_y_iqr": float(np.subtract(*np.percentile(acc_y, [75, 25]))) if len(acc_y) else float("nan"),
        "flight_acc_x_abs_median": float(np.median(np.abs(acc_x))) if len(acc_x) else float("nan"),
        "penetration_mean": float(pen.mean()) if len(pen) else float("nan"),
        "penetration_share": float((pen > 0.05).mean()) if len(pen) else float("nan"),
        "balance_out_share": sum(s["balance_out"] for s in stats) / max(sum(s["standing_frames"] for s in stats), 1),
        "acc_down_p99": float(np.percentile(ad, 99)) if len(ad) else float("nan"),
        "acc_down_over_g": float((ad > g).mean()) if len(ad) else float("nan"),
    }


# ---------------------------------------------------------------------------------------------------------------
# Gravity as a training loss (torch). Uses the same definitions as the measures above, on the generated motion
# itself; the flight mask is not differentiated (it only selects where the law applies).
# ---------------------------------------------------------------------------------------------------------------
def flight_mask(j, contact_threshold, air_margin=0.15, min_flight=6, floor_window=15):
    """(B, L, 17, 2) absolute joints (body units, image y down) -> (B, L) bool: frames in flight (no slow ankle, both
    ankles and every joint >= air_margin above the window's local floor, in runs of >= min_flight frames)."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        an = j[:, :, [L_AN, R_AN]]
        sp = (an[:, 1:] - an[:, :-1]).norm(dim=-1)                                   # (B, L-1, 2)
        sp = torch.cat([sp[:, :1], sp], 1)
        slow = sp < contact_threshold
        slow[:, :-1] |= slow[:, 1:].clone()
        y = torch.where(slow, an[..., 1], torch.full_like(an[..., 1], -1e9)).amax(-1)  # (B, L) lowest planted ankle
        floor = F.max_pool1d(y[:, None], 2 * floor_window + 1, 1, floor_window)[:, 0]
        known = floor > -1e8
        above = (floor[..., None] - an[..., 1]).amin(-1) > air_margin
        clear = (floor - j[..., 1].amax(-1)) > air_margin
        f = (~slow.any(-1) & above & clear & known).float()[:, None]
        k = min_flight
        core = F.avg_pool1d(F.pad(f, (k // 2, k - 1 - k // 2)), k, 1) > 0.999          # opening: drop short runs
        f = F.max_pool1d(F.pad(core.float(), (k - 1 - k // 2, k // 2)), k, 1)[:, 0] > 0.5
    return f


def gravity_loss(j, valid, w, g=0.02, tol=0.25, contact_threshold=0.012, smooth=1.0):
    """(B, L, 17, 2) absolute joints, valid (B, L, 17), w (B,) -> (flight term, downward term), both relative to g.

    flight:   per jump (a flight run of >= 6 frames in which the CoM rises then falls, as in physics_stats), the
              CoM's acceleration from a least-squares parabola over the whole jump (differentiable: a fixed matrix
              per jump length): ((a_y - g) / g)^2 + (a_x / g)^2, averaged over jumps. This is the measure that
              recovers g on real dancers; per-frame accelerations mostly measure tracking noise and the take-off /
              landing impulses.
    downward: everywhere, relu((a_y - (1 + tol) g) / g)^2 on the per-frame CoM acceleration (smoothed over
              `smooth` frames): the floor only pushes up, so nothing accelerates the body down faster than g."""
    import torch

    from beat2body.model.diffusion import smooth_time
    com = centre_of_mass(j)                                                            # (B, L, 2)
    cs = smooth_time(com, smooth)
    acc = cs[:, 2:] - 2 * cs[:, 1:-1] + cs[:, :-2]                                     # frames 1 .. L-2
    ok = valid.all(-1)
    okf = (ok[:, 2:] & ok[:, 1:-1] & ok[:, :-2]).float() * w[:, None]
    e_dn = torch.relu((acc[..., 1] - (1 + tol) * g) / g) ** 2
    down = (e_dn * okf).sum() / okf.sum().clamp(min=1)
    fm = flight_mask(j, contact_threshold).cpu().numpy()
    okn, yn = ok.cpu().numpy(), com[..., 1].detach().cpu().numpy()
    terms, weights, cache = [], [], {}
    for b in range(j.shape[0]):
        if float(w[b]) <= 0:
            continue
        for a, e in _runs(fm[b]):
            if e - a < 6 or not okn[b, a:e].all():
                continue
            k = int(np.argmin(yn[b, a:e]))                                             # apex: rise, then fall
            if not 0 < k < e - a - 1:
                continue
            n = e - a
            if n not in cache:                                                         # parabola coefficients
                t = np.arange(n, dtype=np.float64) - (n - 1) / 2
                A = np.stack([t ** 2, t, np.ones(n)], 1)
                cache[n] = torch.as_tensor(2 * np.linalg.pinv(A)[0], dtype=j.dtype, device=j.device)   # -> 2a
            a_y, a_x = cache[n] @ com[b, a:e, 1], cache[n] @ com[b, a:e, 0]
            terms.append(((a_y - g) / g) ** 2 + (a_x / g) ** 2)
            weights.append(w[b])
    if terms:
        tw = torch.stack(weights)
        flight = (torch.stack(terms) * tw).sum() / tw.sum().clamp(min=1e-8)
    else:
        flight = com.sum() * 0
    return flight, down
