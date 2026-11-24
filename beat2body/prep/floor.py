"""The floor of each AIST video, from the raw video (pinhole camera, level, fixed per video).

AIST's studio is white; the only dark objects are the other cameras' tripods (identical, standing on the floor at
different depths) and their cables. For a level camera, an object of height H standing on the floor at depth Z has
its base at y = y_h + f h / Z and an on-screen height of f H / Z, so

    base_y = horizon + (h / H) * on_screen_height          (h: camera height)

The same holds for the dancer: the point where they meet the floor is at horizon + k * (on-screen body size), with
k = camera height / body size, one number per video. Steps:
  1. background = the 90th percentile of ~60 frames (the white studio wherever the dancer was absent at least
     sometimes; the tripods are always there);
  2. tripods = dark connected components with a camera head on top; base = their lowest point near the head;
  3. horizon from the tripods by least squares (needs >= 2 different heights, i.e. depths);
  4. per video, k from the dancer's planted heel / toe points (DWPose, features/contact) against the torso length
     (body size), with the horizon fixed; the dancer's own horizon (fitted freely) is kept as a check.

Writes <data>/features/floor/<video_id>.json (pixel units of the original video):
    horizon_y, tripods [[x, top, base, height]], k, dancer_horizon_y, fit residual, n contacts
The floor under the dancer at frame t is then horizon_y + k * torso_px(t).

    python -m beat2body floor [--videos ID ...] [--frames 60]
"""
import glob
import json
import os
import subprocess

import numpy as np

from beat2body import paths

DOC = __doc__
WIDTH = 960                     # analysis width (the videos are 1920 x 1080)
HORIZON_RANGE = (0.44, 0.56)    # plausible horizon, fraction of the image height (the AIST cameras: ~0.50)
K_RANGE = (1.5, 3.5)            # plausible floor scale (camera height / torso length: ~2.4)


def add_args(p):
    p.add_argument("--videos", nargs="*", default=None, help="only these video ids")
    p.add_argument("--frames", type=int, default=60, help="frames for the background")
    p.add_argument("--overwrite", action="store_true")


def floor_path(video_id):
    return os.path.join(paths.features_dir(), "floor", video_id + ".json")


def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def background(path, n=60, width=WIDTH):
    """90th percentile (bright) of n evenly spaced grayscale frames -> (h, w) uint8, and the video's (W, H)."""
    ff = _ffmpeg()
    info = subprocess.run([ff, "-i", path], capture_output=True, text=True).stderr
    hh, mm, ss = info.split("Duration: ")[1].split(",")[0].split(":")
    dur = int(hh) * 3600 + int(mm) * 60 + float(ss)
    size = [s for s in info.split("Video:")[1].split(",")
            if "x" in s and s.strip().split(" ")[0].replace("x", "").isdigit()]
    W, H = (int(v) for v in size[0].strip().split(" ")[0].split("x"))
    raw = subprocess.run([ff, "-loglevel", "error", "-i", path, "-vf",
                          f"fps={n / max(dur, 1e-3)},scale={width}:-2,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    h = int(round(width * H / W / 2) * 2)
    fr = np.frombuffer(raw, np.uint8)
    fr = fr[: len(fr) // (h * width) * h * width].reshape(-1, h, width)
    return np.percentile(fr, 90, axis=0).astype(np.uint8), (W, H)


def tripods(bg, dark=110, min_height=40, leg=45, max_fill=0.38):
    """Dark components with a camera head on top -> list of (x_head, top, base) in background pixels. A tripod is thin
    legs under a small head (fill 0.23-0.30 of the box); one that fills more than max_fill is solid (a dancer, ~0.45,
    who stood still long enough to stay in the background, as in the basic videos) and is skipped."""
    from scipy import ndimage
    lab, _ = ndimage.label(ndimage.binary_dilation(bg < dark, iterations=1))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        ys, xs = np.nonzero(lab[sl] == i)
        ys, xs = ys + sl[0].start, xs + sl[1].start
        if ys.max() - ys.min() < min_height:
            continue
        box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        if len(ys) / box > max_fill:
            continue
        top = ys.min()
        hx = xs[ys < top + 10].mean()
        base = ys[np.abs(xs - hx) < leg].max()
        out.append((float(hx), int(top), int(base)))
    return out


def fit_horizon(tri, min_spread=8):
    """Least squares base = horizon + c * height over the tripods -> (horizon, c) or None (one depth only)."""
    if len(tri) < 2:
        return None
    hgt = np.array([b - t for _, t, b in tri], float)
    base = np.array([b for _, _, b in tri], float)
    if hgt.max() - hgt.min() < min_spread:
        return None
    c, horizon = np.polyfit(hgt, base, 1)
    return float(horizon), float(c)


def dancer_fit(video_id, category, horizon_px):
    """k (and a free horizon as a check) from planted heel / toe points vs torso length, in original pixels."""
    from beat2body.prep.contact import contact_path
    raw = np.load(paths.raw_keypoint_path("dwpose", category, video_id))
    names = list(raw["native_joint_names"])
    nat = raw["native"]
    lab = np.load(contact_path(video_id))
    feet = {0: ("left_heel", "left_big_toe", "left_small_toe"), 1: ("right_heel", "right_big_toe", "right_small_toe")}
    sh = (nat[:, names.index("left_shoulder"), :2] + nat[:, names.index("right_shoulder"), :2]) / 2
    hp = (nat[:, names.index("left_hip"), :2] + nat[:, names.index("right_hip"), :2]) / 2
    body = [names.index(n) for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    ok_t = (nat[:, body, 2] > 0.5).all(1)
    torso = np.linalg.norm(sh - hp, axis=1)
    # body size: median torso over +/- 0.5 s (robust to bending)
    T = len(torso)
    size = np.array([np.median(torso[max(0, t - 15):t + 16][ok_t[max(0, t - 15):t + 16]])
                     if ok_t[max(0, t - 15):t + 16].any() else np.nan for t in range(T)])
    ys, ss = [], []
    n = min(T, len(lab["heel"]))
    for k, pts in feet.items():
        idx = [names.index(p) for p in pts]
        planted = (lab["heel"][:n, k] == 1) | (lab["toe"][:n, k] == 1)
        conf = nat[:n, idx, 2] > 0.6
        low = np.where(conf, nat[:n, idx, 1], -np.inf).max(1)            # lowest confident foot point (y down)
        sel = planted & np.isfinite(low) & np.isfinite(size[:n])
        ys.append(low[sel])
        ss.append(size[:n][sel])
    y, s = np.concatenate(ys), np.concatenate(ss)
    if len(y) < 30:
        return None
    k = float(np.sum((y - horizon_px) * s) / np.sum(s * s)) if horizon_px is not None else None
    slope, free_h = np.polyfit(s, y, 1)
    res = float(np.median(np.abs(y - (horizon_px + k * s)))) if k is not None else None
    return {"k": k, "dancer_horizon_y": float(free_h), "dancer_k_free": float(slope), "median_abs_residual_px": res,
            "contacts": int(len(y)), "size_range_px": [float(np.percentile(s, 5)), float(np.percentile(s, 95))]}


def run(args):
    from beat2body import aist
    os.makedirs(os.path.join(paths.features_dir(), "floor"), exist_ok=True)
    # pass 1: the horizon from the tripods of every local raw video
    for path in sorted(glob.glob(os.path.join(paths.raw_video_dir(), "*", "*.mp4"))):
        vid = os.path.basename(path)[:-4]
        if (args.videos and vid not in args.videos) or (os.path.exists(floor_path(vid)) and not args.overwrite):
            continue
        bg, (W, H) = background(path, args.frames)
        scale = W / bg.shape[1]
        tri = tripods(bg)
        fit = fit_horizon(tri)
        rec = {"video_id": vid, "width": W, "height": H,
               "tripods": [[x * scale, t * scale, b * scale, (b - t) * scale] for x, t, b in tri],
               "tripod_horizon_y": None if fit is None else fit[0] * scale,
               "tripod_slope": None if fit is None else fit[1]}
        with open(floor_path(vid), "w") as fh:
            json.dump(rec, fh, indent=1)
        h = rec["tripod_horizon_y"]
        print(f"{vid}: {len(tri)} tripods, horizon {'-' if h is None else round(h)}", flush=True)
    # pass 2: every cleaned video: horizon (own, else the dancer's median, else the overall median) and k
    recs = {}
    for f in glob.glob(os.path.join(paths.features_dir(), "floor", "*.json")):
        r = json.load(open(f))
        h = r.get("tripod_horizon_y")
        if h is not None and HORIZON_RANGE[0] <= h / r["height"] <= HORIZON_RANGE[1]:   # plausible only
            recs[r["video_id"]] = h
    by_dancer = {}
    for v, h in recs.items():
        by_dancer.setdefault(aist.dancer_id(v), []).append(h)
    overall = float(np.median(list(recs.values()))) if recs else None
    n = 0
    for f in sorted(glob.glob(os.path.join(paths.keypoints_dir(), "dwpose", "*", "*.npz"))):
        cat, vid = os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-4]
        if args.videos and vid not in args.videos:
            continue
        rec = json.load(open(floor_path(vid))) if os.path.exists(floor_path(vid)) else {"video_id": vid}
        if vid in recs:
            h, src = recs[vid], "tripods"
        elif by_dancer.get(aist.dancer_id(vid)):
            h, src = float(np.median(by_dancer[aist.dancer_id(vid)])), "dancer median"
        else:
            h, src = overall, "overall median"
        rec.update(horizon_y=h, horizon_source=src)
        if h is not None and os.path.exists(paths.raw_keypoint_path("dwpose", cat, vid)):
            d = dancer_fit(vid, cat, h)
            if d:
                rec.update(d)
                rec["k_ok"] = bool(K_RANGE[0] <= d["k"] <= K_RANGE[1])
        with open(floor_path(vid), "w") as fh:
            json.dump(rec, fh, indent=1)
        n += 1
    ks = [json.load(open(floor_path(v)))["k"] for v in [os.path.basename(f)[:-4] for f in
          glob.glob(os.path.join(paths.keypoints_dir(), "dwpose", "*", "*.npz"))]
          if os.path.exists(floor_path(v)) and json.load(open(floor_path(v))).get("k") is not None]
    hs = list(recs.values())
    where = os.path.join(paths.rel(paths.features_dir()), "floor")
    print(f"{n} videos -> {where}/; tripod horizons: {len(hs)} videos, "
          f"median {np.median(hs):.1f} px, IQR {np.subtract(*np.percentile(hs, [75, 25])):.1f}; "
          f"k: {len(ks)} videos, median {np.median(ks):.2f}, IQR {np.subtract(*np.percentile(ks, [75, 25])):.2f}")
