"""PyTorch data loading for training (Task 2).

    from beat2body.data import make_loader
    train = make_loader("dwpose", "train", batch_size=64, seq_len=120)
    batch = next(iter(train))

Each item is a dict of tensors for one clip (W = seq_len or the clip length, 240):
    keypoints  (W, 17, 2) float32  COCO-17, pelvis-centred, in torso lengths
    valid      (W, 17)    bool     observed joints (loss mask)
    audio      (W, 65)    float32  log-mel (64) + onset strength (1), normalized with train-split stats
    beats      (W,)       float32  1.0 on beat frames, else 0.0
    category   ()         int64    index from categories.json
    video_id, start                for bookkeeping / rendering (start = first frame in the video)

Train items get a random temporal crop (when seq_len < clip length) and a random left/right
flip; val / test items are deterministic (crop at the clip start, no flip).

The audio statistics are computed once per split definition on the train videos only and cached
in <data>/clips/<tool>/audio_stats.json together with a fingerprint of the train videos, so they are
recomputed automatically after a re-split. Store them with the model checkpoint for inference.

Everything is held in memory (~200 MB for DWPose), so an epoch over the train split takes well under
a second on one CPU core: keep num_workers=0 (worker processes only add start-up cost).

On a fresh machine (e.g. AWS) `ensure_data(tool)` / `--download` fetches the dataset from
Hugging Face first; private datasets need HF_TOKEN.

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
from beat2body.skeleton import FLIP_PERM

DOC = __doc__
N_MELS = 64
AUDIO_DIM = N_MELS + 1


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
# audio normalization (train split only)
# ---------------------------------------------------------------------------
def _fingerprint(video_ids):
    return hashlib.sha1("\n".join(sorted(video_ids)).encode()).hexdigest()[:16]


def _video_audio(tool, video_id, rows):
    """Full-video mel / onset; falls back to stitching the clips if the feature files are absent."""
    mel_path, onset_path = paths.feature_path("audio_mel", video_id), paths.feature_path("audio_onset", video_id)
    if os.path.exists(mel_path) and os.path.exists(onset_path):
        return np.load(mel_path), np.load(onset_path)
    mel, onset = {}, {}
    for r in rows:
        with np.load(os.path.join(paths.base_dir(), r["clip_path"])) as d:
            s = int(r["start_frame"])
            for i in range(len(d["mel"])):
                mel[s + i], onset[s + i] = d["mel"][i], d["onset"][i]
    idx = sorted(mel)
    return np.stack([mel[i] for i in idx]), np.array([onset[i] for i in idx])


def audio_stats(tool, recompute=False):
    """Per-channel mean / std of mel and onset over the frames of the train videos (each frame once)."""
    train = load_meta(tool, "train")
    vids = sorted({r["video_id"] for r in train})
    fp = _fingerprint(vids)
    path = paths.audio_stats_path(tool)
    if os.path.exists(path) and not recompute:
        with open(path) as f:
            stats = json.load(f)
        if stats.get("train_fingerprint") == fp:
            return stats
        print("[data] train split changed, recomputing audio statistics")
    by_vid = {}
    for r in train:
        by_vid.setdefault(r["video_id"], []).append(r)
    mels, onsets = [], []
    for v in vids:
        m, o = _video_audio(tool, v, by_vid[v])
        mels.append(m)
        onsets.append(o)
    mel, onset = np.concatenate(mels).astype(np.float64), np.concatenate(onsets).astype(np.float64)
    stats = {
        "tool": tool, "split": "train", "videos": len(vids), "frames": int(len(mel)), "train_fingerprint": fp,
        "mel_mean": mel.mean(0).tolist(), "mel_std": np.maximum(mel.std(0), 1e-6).tolist(),
        "onset_mean": float(onset.mean()), "onset_std": float(max(onset.std(), 1e-6)),
    }
    with open(path, "w") as f:
        json.dump(stats, f, indent=1)
    return stats


def normalize_audio(mel, onset, stats):
    """(W, 64), (W,) -> (W, 65) float32 normalized audio features."""
    m = (mel - np.asarray(stats["mel_mean"], np.float32)) / np.asarray(stats["mel_std"], np.float32)
    o = (onset - stats["onset_mean"]) / stats["onset_std"]
    return np.concatenate([m, o[:, None]], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
def flip_keypoints(kp, valid):
    """Mirror left / right: negate x (coordinates are pelvis-centred) and swap the joint pairs."""
    kp = kp[:, FLIP_PERM].copy()
    kp[..., 0] *= -1
    return kp, valid[:, FLIP_PERM].copy()


class DanceClips(Dataset):
    """Clips of one split of one pose tool (see the module docstring for the item format)."""

    def __init__(self, tool="dwpose", split="train", seq_len=None, augment=None, flip_prob=0.5,
                 stats=None, in_memory=True):
        if tool not in TOOLS or split not in SPLITS:
            raise ValueError(f"tool must be one of {TOOLS}, split one of {SPLITS}")
        self.tool, self.split = tool, split
        self.rows = load_meta(tool, split)
        if not self.rows:
            raise ValueError(f"no {split} clips in {paths.meta_csv_path(tool)}")
        self.augment = (split == "train") if augment is None else augment
        self.flip_prob = flip_prob
        self.stats = stats or audio_stats(tool)
        lengths = {int(r["end_frame"]) - int(r["start_frame"]) for r in self.rows}
        self.seq_len = seq_len or min(lengths)
        if self.seq_len > min(lengths):
            raise ValueError(f"seq_len {self.seq_len} is longer than the shortest clip ({min(lengths)} frames)")
        self._cache = [self._load(i) for i in range(len(self.rows))] if in_memory else None

    def __len__(self):
        return len(self.rows)

    def _load(self, i):
        r = self.rows[i]
        with np.load(os.path.join(paths.base_dir(), r["clip_path"])) as d:
            W = len(d["keypoints"])
            beats = np.zeros(W, np.float32)
            b = d["beats"]
            beats[b[(b >= 0) & (b < W)]] = 1.0
            return {
                "keypoints": d["keypoints"].astype(np.float32),
                "valid": d["valid"].astype(bool),
                "audio": normalize_audio(d["mel"].astype(np.float32), d["onset"].astype(np.float32), self.stats),
                "beats": beats,
                "category": int(d["category_index"]),
                "video_id": r["video_id"],
                "start": int(r["start_frame"]),
            }

    def __getitem__(self, i):
        c = self._cache[i] if self._cache is not None else self._load(i)
        W, L = len(c["keypoints"]), self.seq_len
        # torch RNG: DataLoader seeds it differently in every worker (numpy's global RNG is not)
        off = int(torch.randint(0, W - L + 1, ())) if self.augment and W > L else 0
        kp, valid = c["keypoints"][off:off + L], c["valid"][off:off + L]
        if self.augment and float(torch.rand(())) < self.flip_prob:
            kp, valid = flip_keypoints(kp, valid)
        return {
            "keypoints": torch.from_numpy(np.ascontiguousarray(kp)),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
            "audio": torch.from_numpy(c["audio"][off:off + L]),
            "beats": torch.from_numpy(c["beats"][off:off + L]),
            "category": torch.tensor(c["category"], dtype=torch.long),
            "video_id": c["video_id"],
            "start": c["start"] + off,
        }


def make_loader(tool="dwpose", split="train", batch_size=64, seq_len=None, num_workers=0, **dataset_kw):
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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=None, help="frames per item (default: clip length)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--download", action="store_true", help="download from Hugging Face if missing")
    p.add_argument("--org", default=None, help="Hugging Face account for --download (see `hf --help`)")
    p.add_argument("--recompute-stats", action="store_true")


def run(args):
    if args.download:
        ensure_data(args.tool, args.org)
    stats = audio_stats(args.tool, recompute=args.recompute_stats)
    print(f"[data] audio stats from {stats['videos']} train videos / {stats['frames']} frames: "
          f"mel mean {np.mean(stats['mel_mean']):.1f} dB, onset mean {stats['onset_mean']:.2f} "
          f"({paths.rel(paths.audio_stats_path(args.tool))})")
    ok = True
    for split in SPLITS:
        t0 = time.perf_counter()
        loader = make_loader(args.tool, split, args.batch_size, args.seq_len, args.workers, stats=stats)
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        n, cats, beat_rate, audio_mu, audio_sd = 0, torch.zeros(64, dtype=torch.long), [], [], []
        for batch in loader:
            if n == 0:
                shapes = {k: (tuple(v.shape), str(v.dtype).replace("torch.", ""))
                          for k, v in batch.items() if torch.is_tensor(v)}
            for k in ("keypoints", "audio", "beats"):
                if not torch.isfinite(batch[k]).all():
                    print(f"[data] {split}: non-finite values in {k}")
                    ok = False
            cats += torch.bincount(batch["category"], minlength=64)
            beat_rate.append(batch["beats"].mean().item())
            audio_mu.append(batch["audio"].mean().item())
            audio_sd.append(batch["audio"].std().item())
            n += len(batch["category"])
        t_epoch = time.perf_counter() - t0
        n_cat = int((cats > 0).sum())
        print(f"[data] {split}: {len(loader.dataset)} clips, {len(loader)} batches, {n_cat} genres; "
              f"load {t_load:.1f}s, epoch {t_epoch:.2f}s; beats {60 * 30 * np.mean(beat_rate):.0f}/min; "
              f"audio mean {np.mean(audio_mu):+.2f} std {np.mean(audio_sd):.2f}")
        if split == "train":
            for k, (shape, dtype) in shapes.items():
                print(f"         {k:9s} {shape} {dtype}")
        ok &= n_cat == len(aist.load_categories())
    if not ok:
        sys.exit("[data] check FAILED")
    print("[data] check passed")
