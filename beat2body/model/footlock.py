"""Contact-driven foot locking for generated 2D motion (post-processing, no retraining).

Diffusion samples slide: a foot the model itself marks as planted keeps moving in the image. For each frame
the model's contact output (first two features) says which ankle is planted. Then:
  1. (optional) the root is smoothed with a Gaussian of `root_smooth` frames (root jitter);
  2. the root's velocity absorbs the planted ankles' velocity, so a planted ankle stays still and the body moves
     over it, as in a real step (both planted: their mean velocity);
  3. a planted ankle that still moves (both feet planted, or the blend at a contact edge) is pinned to where it
     landed with 2-bone IK on hip-knee-ankle, keeping the thigh and shin lengths and the side the knee bends to.
Contacts shorter than `min_frames` are ignored; the correction fades in and out over `blend` frames around each
contact, so there are no jumps.
"""
import numpy as np

from beat2body.data import CONTACT, JOINTS, ROOT

LEGS = ((11, 13, 15), (12, 14, 16))          # (hip, knee, ankle), left and right


def _runs(mask):
    """(T,) bool -> list of [start, end) runs of True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def contact_weights(contact, min_frames=3, blend=4):
    """(T, 2) bool -> (T, 2) weights in [0, 1] (1 = planted, linear fade over `blend` frames outside a contact)
    and (T, 2) index of the frame each weight's contact run starts at (-1 = none)."""
    T = len(contact)
    w = np.zeros((T, 2), np.float32)
    start = np.full((T, 2), -1)
    t = np.arange(T)
    for k in range(2):
        for a, b in _runs(contact[:, k]):
            if b - a < min_frames:
                continue
            dist = np.maximum(np.maximum(a - t, t - (b - 1)), 0)                       # 0 inside the run
            wk = np.clip(1 - dist / (blend + 1), 0, 1)
            take = wk > w[:, k]
            w[take, k], start[take, k] = wk[take], a
    return w, start


def _gauss(x, sigma):
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(x.shape[1])], 1)


def two_bone_ik(hip, knee, ankle, target):
    """Move knee and ankle so the ankle reaches `target` (as far as the leg allows), keeping both bone lengths and
    the side the knee bends to. All (2,)."""
    l1, l2 = np.linalg.norm(knee - hip), np.linalg.norm(ankle - knee)
    v = target - hip
    d = np.linalg.norm(v)
    if d < 1e-6 or l1 < 1e-6 or l2 < 1e-6:
        return knee, ankle
    u = v / d
    d = np.clip(d, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
    a = (l1 ** 2 - l2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(max(l1 ** 2 - a ** 2, 0.0))
    n = np.array([-u[1], u[0]])
    ah, kh = ankle - hip, knee - hip                                     # bend side of the current leg
    side = np.sign(ah[0] * kh[1] - ah[1] * kh[0]) or 1.0
    k_new = hip + a * u + side * h * n
    return k_new, hip + d * u


def foot_lock(motion, threshold=0.5, min_frames=3, blend=4, root_smooth=0.0, ik=True):
    """(T, 38) denormalized motion (numpy or torch) -> foot-locked copy (numpy float32)."""
    m = np.array(motion, dtype=np.float32, copy=True)
    T = len(m)
    if T < 2:
        return m
    root = _gauss(m[:, ROOT], root_smooth)
    rel = m[:, JOINTS].reshape(T, -1, 2)
    w, start = contact_weights(m[:, CONTACT] > threshold, min_frames, blend)

    # 2. root velocity absorbs the planted ankles' velocity
    ank = np.stack([rel[:, LEGS[k][2]] for k in range(2)], 1) + root[:, None]       # (T, 2, 2)
    va = ank[1:] - ank[:-1]                                                          # (T-1, 2, 2)
    wt = w[1:]                       # the step t -> t+1 is planted when contact[t+1] is (contact[t]: step t-1 -> t)
    corr = (wt[..., None] * va).sum(1) / np.maximum(wt.sum(1), 1.0)[:, None]
    root = np.concatenate([root[:1], root[0] + np.cumsum(root[1:] - root[:-1] - corr, 0)])

    # 3. pin what still moves with IK, toward where each contact started
    if ik:
        absj = rel + root[:, None]
        for k, (hi, ki, ai) in enumerate(LEGS):
            for t in np.flatnonzero(w[:, k] > 0):
                anchor = absj[start[t, k], ai]
                goal = w[t, k] * anchor + (1 - w[t, k]) * absj[t, ai]
                if np.linalg.norm(goal - absj[t, ai]) < 1e-6:
                    continue
                absj[t, ki], absj[t, ai] = two_bone_ik(absj[t, hi], absj[t, ki], absj[t, ai], goal)
        rel = absj - root[:, None]
    m[:, ROOT] = root
    m[:, JOINTS] = rel.reshape(T, -1)
    return m


def feet_stats(j, contact_threshold):
    """Sliding measures of absolute joints j (N, L, 17, 2) (or (L, 17, 2)): median speed of the slower ankle,
    share of frames with no ankle slower than the contact threshold, and root jitter (mean |root acceleration|)."""
    j = np.asarray(j)
    if j.ndim == 3:
        j = j[None]
    slow = np.linalg.norm(np.diff(j[:, :, [15, 16]], axis=1), axis=-1).min(-1)          # (N, L-1)
    pel = j[:, :, [11, 12]].mean(2)
    acc = np.linalg.norm(np.diff(pel, n=2, axis=1), axis=-1)
    return {"planted_foot_speed": float(np.median(slow)), "no_foot_planted": float((slow > contact_threshold).mean()),
            "root_jitter": float(acc.mean())}
