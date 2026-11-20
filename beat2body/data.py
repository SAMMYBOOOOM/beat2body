"""PyTorch data loading for training, in an EDGE-style motion representation.

    from beat2body.data import make_loader
    train = make_loader("dwpose", "train", batch_size=64, seq_len=150)

Each item is a dict of tensors for one window of L = seq_len frames (EDGE: 150 = 5 s):
    motion     (L, 38)  float32  [contact L/R ankle (0/1) | root x, y | 17 joints x (x, y)]
                                 EDGE's layout [4 contacts | 3 root | 24 x 6D rotations] in 2D:
                                 root = pelvis position in the image, relative to the image centre;
                                 joints = pelvis-centred; all lengths in torso lengths (dataset scale)
    valid      (L, 17)  bool     observed joints (loss mask)
    root_valid (L,)     bool     both hips observed
    audio      (L, A+2) float32  music features (normalized with train-split stats) + sin/cos of the
                                 beat phase (where in the beat each frame falls)
                                 feature_set="edge": EDGE's 35 baseline features (A = 35)
                                 feature_set="mel":  log-mel 64 + onset 1 (A = 65)
    beats      (L,)     float32  1.0 on the frame nearest each beat
    category   ()       int64    index from categories.json
    video_id, start              bookkeeping (start = first frame in the video)

Windows: the clips in clips/<tool>/meta.csv (240 frames, stride 60) define where windows may come
from and their split; the data itself is read from the per-video files (keypoints/<tool>/,
features/), so changing the representation needs no rebuild. Train items take a random
seq_len crop of a clip and a random left/right flip; val / test items are deterministic.
Everything is held in memory; keep num_workers=0.

Body size (scale_norm): "video" (default) divides each video's root and joints by that video's
median torso length, so every dancer has the same body size (taller dancers and dancers closer to
the camera no longer look bigger); within a video, real foreshortening is kept. "dataset" keeps
the single dataset-wide scale of the cleaned keypoints.

Foot contact: an ankle is "in contact" when its image speed (with root motion) is below
CONTACT_THRESHOLD torso lengths per frame (~1.5 px at 1080p, ~40% of frames). contact[t] describes the step
t-1 -> t (frame 0 copies frame 1), so the step t -> t+1 is planted when contact[t+1] is set.

    python -m beat2body data --tool dwpose --download      # check: shapes, NaNs, stats, speed
"""
import csv
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from beat2body import aist, paths
from beat2body.paths import SPLITS, TOOLS
from beat2body.skeleton import FLIP_PERM, L_HIP, L_SHOULDER, R_HIP, R_SHOULDER

DOC = __doc__
N_JOINTS = 17
ANKLES = (15, 16)
CONTACT_THRESHOLD = 0.012
MOTION_DIM = 2 + 2 + 2 * N_JOINTS              # 38
CONTACT, ROOT, JOINTS = slice(0, 2), slice(2, 4), slice(4, MOTION_DIM)
FEATURE_SETS = {"edge": (["audio_edge"], 35), "mel": (["audio_mel", "audio_onset"], 65)}
PHASE_DIM = 2
SCALE_NORMS = ("dancer", "video", "dataset")


def audio_dim(feature_set):
    return FEATURE_SETS[feature_set][1] + PHASE_DIM


# ---------------------------------------------------------------------------
# metadata and data availability
# ---------------------------------------------------------------------------
def load_meta(tool, split=None):
    """Rows of <data>/clips/<tool>/meta.csv (optionally of one split)."""
    path = paths.meta_csv_path(tool)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run `python -m beat2body hf download --tool {tool}` "
                                f"or `python -m beat2body clips --tool {tool}`")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split is None or r["split"] == split]


def ensure_data(tool, org=None):
    """Download keypoints, features and clips of `tool` from Hugging Face if meta.csv is missing."""
    if os.path.exists(paths.meta_csv_path(tool)):
        return
    import argparse

    from beat2body.prep import hf
    print(f"[data] {paths.rel(paths.meta_csv_path(tool))} missing, downloading from Hugging Face")
    hf.download(argparse.Namespace(tool=[tool], org=org, parts=["keypoints", "features", "meta"], revision=None))


# ---------------------------------------------------------------------------
# per-video arrays
# ---------------------------------------------------------------------------
def torso_length(kp, valid):
    """Median mid-shoulder to mid-hip distance over frames where all four joints were observed."""
    ok = valid[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]].all(1)
    t = np.linalg.norm((kp[:, L_SHOULDER] + kp[:, R_SHOULDER]) / 2 - (kp[:, L_HIP] + kp[:, R_HIP]) / 2, axis=-1)
    return float(np.median(t[ok])) if ok.any() else 1.0


_DANCER_TORSO = {}


def dancer_torso(tool, video_id):
    """Median torso length of this dancer over all of their videos (per-video medians), or None for non-AIST
    names. Robust to videos that are mostly floor work, where a single video's torso is foreshortened."""
    d = aist.dancer_id(video_id)
    if d is None:
        return None
    if (tool, d) not in _DANCER_TORSO:
        meds = []
        for f in glob.glob(os.path.join(paths.keypoints_dir(), tool, "*", f"*_{d}_*.npz")):
            z = np.load(f)
            meds.append(torso_length(z["keypoints"], z["valid"]))
        _DANCER_TORSO[(tool, d)] = float(np.median(meds)) if meds else None
    return _DANCER_TORSO[(tool, d)]


def video_motion(tool, category, video_id, scale_norm="video"):
    """(T, 38) motion, (T, 17) valid, (T,) root_valid for one video. scale_norm: dancer = this dancer's
    median torso over all their videos = 1; video = this video's median torso = 1 (wrong for floor work);
    dataset = one scale for all."""
    d = np.load(paths.clean_keypoint_path(tool, category, video_id))
    kp, valid, scale = d["keypoints"].astype(np.float32), d["valid"], float(d["scale"])
    centre = np.array([float(d["width"]) / 2, float(d["height"]) / 2], np.float32)
    root = (d["pelvis"].astype(np.float32) - centre) / scale
    if scale_norm in ("video", "dancer"):
        t = (dancer_torso(tool, video_id) if scale_norm == "dancer" else None) or torso_length(kp, valid)
        kp, root = kp / t, root / t
    contact = contact_labels(kp[:, ANKLES] + root[:, None])
    motion = np.concatenate([contact, root, kp.reshape(len(kp), -1)], 1)
    return motion, valid.astype(bool), valid[:, L_HIP] & valid[:, R_HIP]


def contact_labels(ankles):
    """(T, 2, 2) absolute ankle positions -> (T, 2) float 0/1: contact[t] = the ankle moved less than
    CONTACT_THRESHOLD over the step t-1 -> t (contact[0] = contact[1])."""
    speed = np.linalg.norm(np.diff(ankles, axis=0), axis=-1)
    speed = np.concatenate([speed[:1], speed])                   # (T, 2)
    return (speed < CONTACT_THRESHOLD).astype(np.float32)


def video_audio(video_id, feature_set):
    """Raw (unnormalized) music features (T, A)."""
    kinds = FEATURE_SETS[feature_set][0]
    arrs = [np.load(paths.feature_path(k, video_id)).astype(np.float32) for k in kinds]
    T = min(len(a) for a in arrs)
    return np.concatenate([a[:T].reshape(T, -1) for a in arrs], 1)


def beat_phase(beats, T):
    """sin / cos of the position within the beat for frames 0..T-1 (0 outside the beat grid)."""
    out = np.zeros((T, PHASE_DIM), np.float32)
    b = np.sort(np.asarray(beats, np.float64))
    if len(b) < 2:
        return out
    t = np.arange(T, dtype=np.float64)
    i = np.clip(np.searchsorted(b, t, side="right") - 1, 0, len(b) - 2)
    ph = (t - b[i]) / (b[i + 1] - b[i])
    inside = (t >= b[0]) & (t <= b[-1])
    out[inside, 0] = np.sin(2 * np.pi * ph[inside])
    out[inside, 1] = np.cos(2 * np.pi * ph[inside])
    return out


# ---------------------------------------------------------------------------
# audio normalization (train split only)
# ---------------------------------------------------------------------------
def _fingerprint(video_ids):
    return hashlib.sha1("\n".join(sorted(video_ids)).encode()).hexdigest()[:16]


def audio_stats(tool, feature_set="edge", recompute=False):
    """Per-channel mean / std of the music features over the frames of the train videos."""
    vids = sorted({r["video_id"] for r in load_meta(tool, "train")})
    fp = _fingerprint(vids)
    path = paths.audio_stats_path(tool).replace(".json", f"_{feature_set}.json")
    if os.path.exists(path) and not recompute:
        with open(path) as f:
            stats = json.load(f)
        if stats.get("train_fingerprint") == fp:
            return stats
    a = np.concatenate([video_audio(v, feature_set) for v in vids]).astype(np.float64)
    stats = {"tool": tool, "feature_set": feature_set, "videos": len(vids), "frames": int(len(a)),
             "train_fingerprint": fp, "mean": a.mean(0).tolist(), "std": np.maximum(a.std(0), 1e-6).tolist()}
    with open(path, "w") as f:
        json.dump(stats, f, indent=1)
    return stats


def normalize_audio(a, stats):
    return ((a - np.asarray(stats["mean"], np.float32)) / np.asarray(stats["std"], np.float32)).astype(np.float32)


def music_input(video_id, feature_set, stats, beats=None):
    """Normalized features + beat phase for a whole video: (T, A + 2)."""
    a = normalize_audio(video_audio(video_id, feature_set), stats)
    if beats is None:
        beats = np.load(paths.feature_path("beats", video_id))
    return np.concatenate([a, beat_phase(beats, len(a))], 1)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
def flip_motion(motion, valid, root_valid):
    """Mirror left / right: negate x of root and joints, swap left / right joints and contacts."""
    m = motion.copy()
    m[:, CONTACT] = m[:, [1, 0]]
    m[:, ROOT.start] *= -1
    j = m[:, JOINTS].reshape(len(m), N_JOINTS, 2)[:, FLIP_PERM]
    j[..., 0] *= -1
    m[:, JOINTS] = j.reshape(len(m), -1)
    return m, valid[:, FLIP_PERM].copy(), root_valid


def jitter_motion(motion, scale, angle, shift):
    """Scale and rotate the whole figure about the image centre, then shift the root (torso lengths)."""
    m = motion.copy()
    c, s_ = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s_], [s_, c]], np.float32) * scale
    m[:, ROOT] = m[:, ROOT] @ rot.T + np.asarray(shift, np.float32)
    j = m[:, JOINTS].reshape(len(m), N_JOINTS, 2) @ rot.T
    m[:, JOINTS] = j.reshape(len(m), -1)
    return m


class DanceClips(Dataset):
    """Windows of one split (see the module docstring for the item format).

    Training augmentation (train split only; val / test items are deterministic, main tool only):
      extra_tools  the same videos' keypoints from other pose tools; each item picks one tool at random
                   (the tools track the same dancer slightly differently: a cheap source of variety)
      stretch      time-stretch motion and music together by a factor in [1 - s, 1 + s]; beats follow
      scale, rotate, shift   random global scale (+/- scale), in-plane rotation (+/- rotate degrees) and
                   root shift (+/- shift torso lengths) of the whole figure
      flip_prob    left / right mirror
    """

    def __init__(self, tool="dwpose", split="train", seq_len=150, feature_set="edge", augment=None,
                 flip_prob=0.5, stats=None, scale_norm="video", extra_tools=(), stretch=0.0,
                 scale=0.0, rotate=0.0, shift=0.0):
        if tool not in TOOLS or split not in SPLITS or feature_set not in FEATURE_SETS:
            raise ValueError(f"tool in {TOOLS}, split in {SPLITS}, feature_set in {list(FEATURE_SETS)}")
        self.tool, self.split, self.feature_set = tool, split, feature_set
        self.rows = load_meta(tool, split)
        if not self.rows:
            raise ValueError(f"no {split} clips in {paths.meta_csv_path(tool)}")
        self.augment = (split == "train") if augment is None else augment
        self.flip_prob, self.stretch, self.scale, self.rotate, self.shift = flip_prob, stretch, scale, rotate, shift
        self.stats = stats or audio_stats(tool, feature_set)
        self.cats = aist.load_categories()
        shortest = min(int(r["end_frame"]) - int(r["start_frame"]) for r in self.rows)
        self.seq_len = seq_len or shortest
        if self.seq_len > shortest:
            raise ValueError(f"seq_len {self.seq_len} is longer than the shortest clip ({shortest} frames)")
        tools = [tool] + ([t for t in extra_tools if t != tool] if self.augment else [])
        self.videos, self.music, self.tools_of = {}, {}, {}
        for r in self.rows:
            v = r["video_id"]
            if v in self.music:
                continue
            beats = np.load(paths.feature_path("beats", v)).astype(np.float64)
            audio = music_input(v, feature_set, self.stats, beats)
            avail = []
            for t in tools:
                if not os.path.exists(paths.clean_keypoint_path(t, r["category"], v)):
                    continue
                motion, valid, root_valid = video_motion(t, r["category"], v, scale_norm)
                self.videos[(t, v)] = {"motion": motion, "valid": valid, "root_valid": root_valid}
                avail.append(t)
            T = min([len(audio)] + [len(self.videos[(t, v)]["motion"]) for t in avail])
            for t in avail:
                d = self.videos[(t, v)]
                for k in d:
                    d[k] = d[k][:T]
            self.music[v] = {"audio": audio[:T], "beats": beats, "category": self.cats[r["category"]], "T": T}
            self.tools_of[v] = avail

    def __len__(self):
        return len(self.rows)

    def _rand(self, lo, hi):
        # torch RNG: DataLoader seeds it differently in every worker (numpy's global RNG is not)
        return lo + (hi - lo) * float(torch.rand(()))

    def window(self, i):
        """(video_id, tool, start, source length) of item i; the source is resampled to seq_len frames."""
        r = self.rows[i]
        vid, L = r["video_id"], self.seq_len
        s, e = int(r["start_frame"]), min(int(r["end_frame"]), self.music[vid]["T"])
        if not self.augment:
            return vid, self.tool, s, L
        tool = self.tools_of[vid][int(torch.randint(0, len(self.tools_of[vid]), ()))]
        src = L
        if self.stretch > 0:
            src = int(round(L * self._rand(1 - self.stretch, 1 + self.stretch)))
            src = max(2, min(src, e - s))
        off = int(torch.randint(0, e - s - src + 1, ())) if e - s > src else 0
        return vid, tool, s + off, src

    def __getitem__(self, i):
        vid, tool, s, src = self.window(i)
        L, m, v = self.seq_len, self.music[vid], self.videos[(tool, vid)]
        if src == L:
            idx = np.arange(s, s + L)
            motion, valid, root_valid = v["motion"][idx], v["valid"][idx], v["root_valid"][idx]
            audio = m["audio"][idx]
        else:                                   # time-stretch: resample src frames to L frames
            t = np.linspace(s, s + src - 1, L)
            near = np.round(t).astype(int)
            lo_i = np.floor(t).astype(int)
            hi_i = np.minimum(lo_i + 1, s + src - 1)
            w = (t - lo_i)[:, None]
            motion = (1 - w) * v["motion"][lo_i] + w * v["motion"][hi_i]
            motion[:, CONTACT] = v["motion"][near][:, CONTACT]
            audio = ((1 - w) * m["audio"][lo_i] + w * m["audio"][hi_i]).astype(np.float32)
            valid, root_valid = v["valid"][near], v["root_valid"][near]
        b = (m["beats"] - s) * (L - 1) / max(src - 1, 1)
        beats = np.zeros(L, np.float32)
        bi = np.round(b).astype(int)
        beats[bi[(bi >= 0) & (bi < L)]] = 1
        motion = motion.astype(np.float32)
        if self.augment:
            if float(torch.rand(())) < self.flip_prob:
                motion, valid, root_valid = flip_motion(motion, valid, root_valid)
            if self.scale or self.rotate or self.shift:
                motion = jitter_motion(motion, self._rand(1 - self.scale, 1 + self.scale),
                                       np.deg2rad(self._rand(-self.rotate, self.rotate)),
                                       (self._rand(-self.shift, self.shift), self._rand(-self.shift, self.shift)))
        return {
            "motion": torch.from_numpy(np.ascontiguousarray(motion)),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
            "root_valid": torch.from_numpy(np.ascontiguousarray(root_valid)),
            "audio": torch.from_numpy(np.ascontiguousarray(audio)),
            "beats": torch.from_numpy(beats),
            "category": torch.tensor(m["category"], dtype=torch.long),
            "video_id": vid,
            "start": s,
        }

    def motion_range(self, mode="minmax"):
        """Per-dimension (lo, hi) over this split's frames (all tools) for the [-1, 1] normalizer.
        minmax (EDGE): min / max, made symmetric in x so that flipped items stay in range.
        zscore: mean -/+ std over the frames and their mirror images (so lo / hi are flip-symmetric).
        Contacts are always 0 / 1."""
        frames = np.concatenate([self.videos[(t, r["video_id"])]["motion"][int(r["start_frame"]):int(r["end_frame"])]
                                 for r in self.rows for t in self.tools_of[r["video_id"]]])
        if mode == "zscore":
            ok = np.isfinite(frames).all(1)
            f = frames[ok]
            f = np.concatenate([f, flip_motion(f, np.ones((len(f), N_JOINTS), bool), np.ones(len(f), bool))[0]])
            mu, sd = f.mean(0), f.std(0) + 1e-6
            lo, hi = mu - sd, mu + sd
            lo[CONTACT], hi[CONTACT] = 0.0, 1.0
            return lo.astype(np.float32), hi.astype(np.float32)
        assert mode == "minmax", mode
        lo, hi = frames.min(0), frames.max(0)
        xs = np.r_[ROOT.start, np.arange(JOINTS.start, MOTION_DIM, 2)]
        lo[xs], hi[xs] = np.minimum(lo[xs], -hi[xs]), np.maximum(hi[xs], -lo[xs])
        joints = np.arange(JOINTS.start, MOTION_DIM)
        pairs = joints.reshape(N_JOINTS, 2)[FLIP_PERM].ravel()   # mirrored joint takes the partner's range
        lo[joints], hi[joints] = np.minimum(lo[joints], lo[pairs]), np.maximum(hi[joints], hi[pairs])
        lo[CONTACT], hi[CONTACT] = 0.0, 1.0
        return lo.astype(np.float32), hi.astype(np.float32)


def make_loader(tool="dwpose", split="train", batch_size=64, seq_len=150, num_workers=0, **dataset_kw):
    """DataLoader with the usual settings: shuffle + drop_last for train, pinned memory on CUDA."""
    ds = DanceClips(tool, split, seq_len=seq_len, **dataset_kw)
    train = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)


# ---------------------------------------------------------------------------
# command line check
# ---------------------------------------------------------------------------
def add_args(p):
    p.add_argument("--tool", default="dwpose", choices=TOOLS)
    p.add_argument("--feature-set", default="edge", choices=list(FEATURE_SETS))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=150)
    p.add_argument("--download", action="store_true", help="download from Hugging Face if missing")
    p.add_argument("--org", default=None, help="Hugging Face account for --download")
    p.add_argument("--recompute-stats", action="store_true")


def run(args):
    if args.download:
        ensure_data(args.tool, args.org)
    stats = audio_stats(args.tool, args.feature_set, recompute=args.recompute_stats)
    print(f"[data] {args.feature_set} music stats from {stats['videos']} train videos / {stats['frames']} frames")
    ok = True
    for split in SPLITS:
        t0 = time.perf_counter()
        loader = make_loader(args.tool, split, args.batch_size, args.seq_len, feature_set=args.feature_set,
                             stats=stats)
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        cats, contact, beat_rate, amu, asd = set(), [], [], [], []
        for n, batch in enumerate(loader):
            if n == 0:
                shapes = {k: (tuple(v.shape), str(v.dtype).replace("torch.", ""))
                          for k, v in batch.items() if torch.is_tensor(v)}
            for k in ("motion", "audio", "beats"):
                if not torch.isfinite(batch[k]).all():
                    print(f"[data] {split}: non-finite values in {k}")
                    ok = False
            cats |= set(batch["category"].tolist())
            contact.append(batch["motion"][..., CONTACT].mean().item())
            beat_rate.append(batch["beats"].mean().item())
            a = batch["audio"][..., :-PHASE_DIM]
            amu.append(a.mean().item())
            asd.append(a.std().item())
        print(f"[data] {split}: {len(loader.dataset)} clips, {len(loader.dataset.videos)} videos, "
              f"{len(cats)} genres; load {t_load:.1f}s, epoch {time.perf_counter() - t0:.2f}s; "
              f"contact {np.mean(contact):.0%}, beats {1800 * np.mean(beat_rate):.0f}/min; "
              f"audio mean {np.mean(amu):+.2f} std {np.mean(asd):.2f}")
        if split == "train":
            for k, (shape, dtype) in shapes.items():
                print(f"         {k:10s} {shape} {dtype}")
        ok &= len(cats) == len(aist.load_categories())
    if not ok:
        sys.exit("[data] check FAILED")
    print("[data] check passed")
