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

Foot contact: an ankle is "in contact" when its image speed (with root motion) is below
CONTACT_THRESHOLD torso lengths per frame (~1.5 px at 1080p, ~40% of frames).

    python -m beat2body data --tool dwpose --download      # check: shapes, NaNs, stats, speed
"""
import csv
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
from beat2body.skeleton import FLIP_PERM, L_HIP, R_HIP

DOC = __doc__
N_JOINTS = 17
ANKLES = (15, 16)
CONTACT_THRESHOLD = 0.012
MOTION_DIM = 2 + 2 + 2 * N_JOINTS              # 38
CONTACT, ROOT, JOINTS = slice(0, 2), slice(2, 4), slice(4, MOTION_DIM)
FEATURE_SETS = {"edge": (["audio_edge"], 35), "mel": (["audio_mel", "audio_onset"], 65)}
PHASE_DIM = 2


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
    hf.download(argparse.Namespace(tool=[tool], org=org, parts=list(hf.PARTS), revision=None))


# ---------------------------------------------------------------------------
# per-video arrays
# ---------------------------------------------------------------------------
def video_motion(tool, category, video_id):
    """(T, 38) motion, (T, 17) valid, (T,) root_valid for one video."""
    d = np.load(paths.clean_keypoint_path(tool, category, video_id))
    kp, valid, scale = d["keypoints"].astype(np.float32), d["valid"], float(d["scale"])
    centre = np.array([float(d["width"]) / 2, float(d["height"]) / 2], np.float32)
    root = (d["pelvis"].astype(np.float32) - centre) / scale
    ankles = kp[:, ANKLES] + root[:, None]
    speed = np.linalg.norm(np.diff(ankles, axis=0), axis=-1)
    speed = np.concatenate([speed[:1], speed])                   # (T, 2)
    contact = (speed < CONTACT_THRESHOLD).astype(np.float32)
    motion = np.concatenate([contact, root, kp.reshape(len(kp), -1)], 1)
    return motion, valid.astype(bool), valid[:, L_HIP] & valid[:, R_HIP]


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


class DanceClips(Dataset):
    """Windows of one split of one pose tool (see the module docstring for the item format)."""

    def __init__(self, tool="dwpose", split="train", seq_len=150, feature_set="edge", augment=None,
                 flip_prob=0.5, stats=None):
        if tool not in TOOLS or split not in SPLITS or feature_set not in FEATURE_SETS:
            raise ValueError(f"tool in {TOOLS}, split in {SPLITS}, feature_set in {list(FEATURE_SETS)}")
        self.tool, self.split, self.feature_set = tool, split, feature_set
        self.rows = load_meta(tool, split)
        if not self.rows:
            raise ValueError(f"no {split} clips in {paths.meta_csv_path(tool)}")
        self.augment = (split == "train") if augment is None else augment
        self.flip_prob = flip_prob
        self.stats = stats or audio_stats(tool, feature_set)
        self.cats = aist.load_categories()
        shortest = min(int(r["end_frame"]) - int(r["start_frame"]) for r in self.rows)
        self.seq_len = seq_len or shortest
        if self.seq_len > shortest:
            raise ValueError(f"seq_len {self.seq_len} is longer than the shortest clip ({shortest} frames)")
        self.videos = {}
        for r in self.rows:
            v = r["video_id"]
            if v in self.videos:
                continue
            motion, valid, root_valid = video_motion(tool, r["category"], v)
            beats = np.load(paths.feature_path("beats", v))
            audio = music_input(v, feature_set, self.stats, beats)
            T = min(len(motion), len(audio))
            mask = np.zeros(T, np.float32)
            bi = np.round(beats).astype(int)
            mask[bi[(bi >= 0) & (bi < T)]] = 1
            self.videos[v] = {"motion": motion[:T], "valid": valid[:T], "root_valid": root_valid[:T],
                              "audio": audio[:T], "beats": mask, "category": self.cats[r["category"]]}

    def __len__(self):
        return len(self.rows)

    def window(self, i):
        r = self.rows[i]
        s = int(r["start_frame"])
        e = min(int(r["end_frame"]), len(self.videos[r["video_id"]]["motion"]))
        L = self.seq_len
        # torch RNG: DataLoader seeds it differently in every worker (numpy's global RNG is not)
        off = int(torch.randint(0, e - s - L + 1, ())) if self.augment and e - s > L else 0
        return r["video_id"], s + off, s + off + L

    def __getitem__(self, i):
        vid, s, e = self.window(i)
        v = self.videos[vid]
        motion, valid, root_valid = v["motion"][s:e], v["valid"][s:e], v["root_valid"][s:e]
        if self.augment and float(torch.rand(())) < self.flip_prob:
            motion, valid, root_valid = flip_motion(motion, valid, root_valid)
        return {
            "motion": torch.from_numpy(np.ascontiguousarray(motion)),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
            "root_valid": torch.from_numpy(np.ascontiguousarray(root_valid)),
            "audio": torch.from_numpy(v["audio"][s:e]),
            "beats": torch.from_numpy(v["beats"][s:e]),
            "category": torch.tensor(v["category"], dtype=torch.long),
            "video_id": vid,
            "start": s,
        }

    def motion_range(self):
        """Per-dimension min / max over this split's frames, made symmetric in x so that flipped
        items stay in range (EDGE normalizes with MinMax to [-1, 1]); contacts are 0 / 1."""
        frames = np.concatenate([self.videos[r["video_id"]]["motion"][int(r["start_frame"]):int(r["end_frame"])]
                                 for r in self.rows])
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
