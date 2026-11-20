"""Foot-contact labels from DWPose's heel and toe points (plan: reports/2026-11-20_feet/CONTACT_PLAN.md, stage B).

Reads the raw DWPose output (native 134 points: heel, big toe and small toe per foot, with confidence) and the
cleaned keypoints, and writes per video

    <data>/features/contact/<video_id>.npz
        heel (T, 2) int8   1 planted, 0 moving, -1 unknown, for the step t-1 -> t (left, right; same convention
        toe  (T, 2) int8   as the ankle-speed labels: contact[0] copies contact[1])
        heel_speed, toe_speed (T, 2) float16   torso lengths / frame (NaN = unknown)

Steps per foot point: the confidence threshold, spike removal, short-gap interpolation and Savitzky-Golay smoothing
of the keypoint cleaning (4.4), then the same body units as the training data (image centre at 0, divided by the
cleaning scale and the dancer's torso). The toe is the mean of the big and small toe. A step is known only where the
point was observed (confidence >= --conf, not interpolated) at both ends, and where the point is close enough to the
cleaned ankle (a far-off heel / toe is a detection error). Planted = speed < CONTACT_THRESHOLD; planted runs shorter
than --min-run steps are dropped and single moving steps between planted ones are filled (detector jitter).

    python -m beat2body contact                 # all videos with raw DWPose output; prints a per-genre report
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from beat2body import paths
from beat2body.prep.clean import clean_video

DOC = __doc__
FOOT_NAMES = ("left_heel", "left_big_toe", "left_small_toe", "right_heel", "right_big_toe", "right_small_toe")
UNKNOWN = -1


def add_args(p):
    p.add_argument("--conf", type=float, default=0.6, help="foot point confidence threshold")
    p.add_argument("--max-heel-ankle", type=float, default=0.5,
                   help="heel farther than this from the cleaned ankle (torso lengths) = detection error")
    p.add_argument("--max-toe-ankle", type=float, default=0.8, help="the same for the toe")
    p.add_argument("--min-run", type=int, default=2, help="shortest planted run (steps)")
    p.add_argument("--videos", nargs="*", default=None, help="only these video ids")


def contact_path(video_id):
    return os.path.join(paths.features_dir(), "contact", video_id + ".npz")


def _clean_points(native, idx, conf, sg_window=7):
    """(T, 134, 3) raw -> (T, k, 2) cleaned pixels (NaN where unknown) and (T, k) observed mask."""
    pts = native[:, idx]
    observed = pts[..., 2] >= conf
    xy, _ = clean_video(pts, conf, max_gap=5, outlier_factor=3.0, sg_window=sg_window)
    observed &= np.isfinite(xy).all(-1)
    return xy, observed


def hysteresis(lab, min_run=2):
    """(T,) int8 labels (1 / 0 / -1) -> planted runs shorter than min_run become 0; a single 0 between two 1s
    becomes 1. Unknown steps are left alone and break runs."""
    lab = lab.copy()
    T = len(lab)
    for t in range(1, T - 1):
        if lab[t] == 0 and lab[t - 1] == 1 and lab[t + 1] == 1:
            lab[t] = 1
    t = 0
    while t < T:
        if lab[t] == 1:
            e = t
            while e < T and lab[e] == 1:
                e += 1
            if e - t < min_run:
                lab[t:e] = 0
            t = e
        else:
            t += 1
    return lab


def step_labels(pos, ok, threshold, min_run):
    """pos (T, 2) body units, ok (T,) observed -> (T,) labels for the steps t-1 -> t and speeds."""
    T = len(pos)
    sp = np.full(T, np.nan)
    sp[1:] = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
    known = np.zeros(T, bool)
    known[1:] = ok[1:] & ok[:-1]
    sp[~known] = np.nan
    sp[0], known[0] = sp[1], known[1]
    lab = np.where(known, (sp < threshold).astype(np.int8), np.int8(UNKNOWN)).astype(np.int8)
    return hysteresis(lab, min_run), sp


def video_labels(video_id, category, args):
    from beat2body.data import CONTACT_THRESHOLD, dancer_torso, video_motion
    raw = np.load(paths.raw_keypoint_path("dwpose", category, video_id))
    clean = np.load(paths.clean_keypoint_path("dwpose", category, video_id))
    names = list(raw["native_joint_names"])
    idx = [names.index(n) for n in FOOT_NAMES]
    xy, observed = _clean_points(raw["native"], idx, args.conf)
    centre = np.array([float(clean["width"]) / 2, float(clean["height"]) / 2])
    t = dancer_torso("dwpose", video_id) or 1.0
    body = (xy - centre) / float(clean["scale"]) / t                          # (T, 6, 2), same units as the data
    motion, valid, root_valid = video_motion("dwpose", category, video_id, "dancer")
    ankles = motion[:, 4:].reshape(len(motion), 17, 2)[:, [15, 16]] + motion[:, None, 2:4]
    out = {"heel": [], "toe": [], "heel_speed": [], "toe_speed": []}
    for k in range(2):                                                        # left, right
        h, b, s = 3 * k, 3 * k + 1, 3 * k + 2
        heel, heel_ok = body[:, h], observed[:, h]
        toes_ok = observed[:, b] | observed[:, s]
        toe = np.where(observed[:, [b]] & observed[:, [s]], (body[:, b] + body[:, s]) / 2,
                       np.where(observed[:, [b]], body[:, b], body[:, s]))
        ank_ok = valid[:, 15 + k] & root_valid
        heel_ok = heel_ok & ank_ok & (np.linalg.norm(np.nan_to_num(heel - ankles[:, k]), axis=-1)
                                      <= args.max_heel_ankle)
        toes_ok = toes_ok & ank_ok & (np.linalg.norm(np.nan_to_num(toe - ankles[:, k]), axis=-1)
                                      <= args.max_toe_ankle)
        for name, pos, ok in (("heel", heel, heel_ok), ("toe", toe, toes_ok)):
            lab, sp = step_labels(np.nan_to_num(pos), ok, CONTACT_THRESHOLD, args.min_run)
            out[name].append(lab)
            out[name + "_speed"].append(sp)
    res = {k: np.stack(v, 1) for k, v in out.items()}
    res["heel_speed"] = res["heel_speed"].astype(np.float16)
    res["toe_speed"] = res["toe_speed"].astype(np.float16)
    ankle_lab = motion[:, :2] > 0.5
    return res, ankle_lab


def stance(heel, toe):
    """Per foot and step: 'flat', 'heel', 'toe', 'swing' or 'unknown' (as small ints 0..4)."""
    s = np.full(heel.shape, 4, np.int8)
    known = (heel != UNKNOWN) & (toe != UNKNOWN)
    s[known & (heel == 1) & (toe == 1)] = 0
    s[known & (heel == 1) & (toe == 0)] = 1
    s[known & (heel == 0) & (toe == 1)] = 2
    s[known & (heel == 0) & (toe == 0)] = 3
    return s


def run(args):
    files = sorted(glob.glob(os.path.join(paths.keypoints_dir(), "*", "*_dwpose.npz")))
    os.makedirs(os.path.join(paths.features_dir(), "contact"), exist_ok=True)
    report = defaultdict(lambda: np.zeros(5))
    agree = defaultdict(lambda: np.zeros(2))
    n = 0
    for f in files:
        cat = os.path.basename(os.path.dirname(f))
        vid = os.path.basename(f)[:-len("_dwpose.npz")]
        if args.videos and vid not in args.videos:
            continue
        if not os.path.exists(paths.clean_keypoint_path("dwpose", cat, vid)):
            continue
        res, ankle_lab = video_labels(vid, cat, args)
        np.savez_compressed(contact_path(vid), **res, conf=args.conf, min_run=args.min_run)
        st = stance(res["heel"], res["toe"])
        g = f"{cat} ({'adv' if '_sFM_' in vid else 'basic'})"
        report[g] += np.bincount(st.ravel(), minlength=5)
        planted = (st <= 2)
        known = st != 4
        agree[g] += [((planted == ankle_lab) & known).sum(), known.sum()]
        n += 1
    print(f"{n} videos -> {os.path.join(paths.rel(paths.features_dir()), 'contact')}/")
    print(f"{'':24s}{'flat':>7s}{'heel':>7s}{'toe':>7s}{'swing':>7s}{'unknown':>8s}{'agree w/ ankle':>16s}")
    for g in sorted(report):
        r = report[g] / report[g].sum()
        ag = agree[g][0] / max(agree[g][1], 1)
        print(f"{g:24s}" + "".join(f"{x:7.1%}" for x in r[:4]) + f"{r[4]:8.1%}" + f"{ag:16.1%}")
    tot = sum(report.values())
    hits, known = sum(a[0] for a in agree.values()), sum(a[1] for a in agree.values())
    shares = dict(zip(["flat", "heel", "toe", "swing", "unknown"], (tot / tot.sum()).tolist()))
    summary = {"videos": n, "shares": shares,
               "agreement_with_ankle_labels": float(hits / max(known, 1)),
               "conf": args.conf, "min_run": args.min_run}
    with open(os.path.join(paths.features_dir(), "contact", "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
