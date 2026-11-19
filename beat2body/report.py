"""Turn a training run into shareable videos and a summary (for the team).

    python -m beat2body report runs/<job>              # after `aws download <job>` or a Colab run folder
    python -m beat2body report runs/<job> --split test --songs 6

Writes <run>/report/:
    progress_<genre>_<video>.mp4   the same val clip generated at each training check, next to the
                                   real dancer (from the samples saved during training)
    dance_<genre>_<video>.mp4      a long dance (default 20 s) for one song per genre: real vs generated
    summary.json                   training curves (metrics.jsonl), evaluation (eval_*.json), video list
Music is added when the audio is available locally (2d/audio/ or 2d/raw_video/); on Colab / AWS the
videos are silent.
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch

from beat2body import aist, paths
from beat2body.render import PANEL_COLORS, StickPanel
from beat2body.video import DEFAULT_FPS, VideoWriter

DOC = __doc__


def add_args(p):
    p.add_argument("run_dir", metavar="RUN",
                   help="run folder with model.pt, metrics.jsonl, samples/ (and live/ from aws download)")
    p.add_argument("--checkpoint", default="model.pt", help="model file inside the run folder")
    p.add_argument("--split", default="val", choices=["val", "test"], help="songs for the long dances")
    p.add_argument("--songs", type=int, default=4, help="number of long dances (one genre each)")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--progress-clips", type=int, default=2, help="number of training-progress videos")
    p.add_argument("--max-panels", type=int, default=6, help="training checkpoints shown per progress video")
    p.add_argument("--size", type=int, default=360, help="panel height in pixels")
    p.add_argument("--device", default="auto")


def absolute(motion):
    m = np.asarray(motion, np.float32)
    return m[:, 4:].reshape(len(m), -1, 2) + m[:, None, 2:4]


def audio_source(category, video_id):
    wav = paths.audio_path(category, video_id)
    if os.path.exists(wav):
        return wav
    raw = glob.glob(os.path.join(paths.raw_video_dir(), category, video_id + ".*"))
    return raw[0] if raw else None


def write_panels(path, seqs, titles, beats, size, audio=None, audio_offset=0.0):
    """Side-by-side stick figures; seqs: list of (T, 17, 2)."""
    T = min(len(s) for s in seqs)
    panels = [StickPanel(s[:T], size, t, PANEL_COLORS[min(i, 1)] if i < 2 else PANEL_COLORS[1])
              for i, (s, t) in enumerate(zip(seqs, titles))]
    flash = set()
    for b in beats:
        flash.update(range(int(round(b)), int(round(b)) + 3))
    with VideoWriter(path, DEFAULT_FPS, size * len(panels), size, audio=audio, audio_offset=audio_offset) as w:
        for t in range(T):
            w.write(np.concatenate([p.draw(s[t], t in flash) for p, s in zip(panels, seqs)], axis=1))


def sample_files(run):
    files = sorted(set(glob.glob(os.path.join(run, "samples", "*.npz")) +
                       glob.glob(os.path.join(run, "live", "samples", "*.npz"))), key=os.path.basename)
    groups = defaultdict(dict)
    for f in files:
        m = re.match(r"step(\d+)_(.+)_(\d{4})\.npz$", os.path.basename(f))
        if m:
            groups[(m.group(2), int(m.group(3)))][int(m.group(1))] = f
    return groups


def run(args):
    from beat2body.data import video_motion
    from beat2body.train import generate_long, load_generator
    dev = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device
    out = os.path.join(args.run_dir, "report")
    os.makedirs(out, exist_ok=True)
    gen, bundle = load_generator(os.path.join(args.run_dir, args.checkpoint), dev)
    cfg = bundle["config"]
    norm = cfg.get("scale_norm", "dataset")
    videos = []

    # 1. training progress: the same val clip at each check
    groups = sample_files(args.run_dir)
    # prefer clips from different videos
    ranked, seen = [], set()
    for key, by_step in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if key[0] not in seen:
            ranked.append((key, by_step))
            seen.add(key[0])
    for (vid, start), by_step in ranked[:args.progress_clips]:
        steps = sorted(by_step)
        pick = [steps[i] for i in np.linspace(0, len(steps) - 1, min(args.max_panels, len(steps))).round().astype(int)]
        pick = sorted(set(pick))
        cat = aist.GENRES[vid[1:3]]
        first = np.load(by_step[pick[-1]])
        L = len(first["motion"])
        real = first["real"] if "real" in first.files else video_motion(cfg["tool"], cat, vid, norm)[0][start:start + L]
        seqs = [absolute(real)] + [absolute(np.load(by_step[s])["motion"]) for s in pick]
        titles = ["real dancer"] + [f"step {s:,}" for s in pick]
        b = np.load(paths.feature_path("beats", vid)) - start
        path = os.path.join(out, f"progress_{cat}_{vid}_{start:04d}.mp4")
        write_panels(path, seqs, titles, b[(b >= 0) & (b < L)], args.size, audio_source(cat, vid), start / DEFAULT_FPS)
        videos.append({"file": os.path.basename(path), "kind": "progress", "video_id": vid, "genre": cat,
                       "music": aist.music_id(vid), "steps": pick})
        print(f"wrote {path} ({len(pick)} checkpoints)")

    # 2. long dances, one song per genre
    from beat2body.data import load_meta
    per_genre = {}
    for r in load_meta(cfg["tool"], args.split):
        per_genre.setdefault(r["category"], r["video_id"])
    for cat in list(sorted(per_genre))[:args.songs]:
        vid = per_genre[cat]
        motion, n, seam = generate_long(gen, bundle, vid, args.seconds)
        real = video_motion(cfg["tool"], cat, vid, norm)[0][:len(motion)]
        b = np.load(paths.feature_path("beats", vid))
        path = os.path.join(out, f"dance_{cat}_{vid}.mp4")
        write_panels(path, [absolute(real), absolute(motion.numpy())], ["real dancer", "generated"],
                     b[b < len(motion)], args.size, audio_source(cat, vid))
        np.savez(path.replace(".mp4", ".npz"), motion=motion.numpy())
        videos.append({"file": os.path.basename(path), "kind": "dance", "video_id": vid, "genre": cat,
                       "music": aist.music_id(vid), "split": args.split, "seconds": round(len(motion) / 30, 1),
                       "windows": n, "overlap_disagreement": round(seam, 4)})
        print(f"wrote {path} ({len(motion) / 30:.0f} s, {n} windows)")

    # 3. summary
    metrics = []
    for f in (os.path.join(args.run_dir, "metrics.jsonl"), os.path.join(args.run_dir, "live", "metrics.jsonl")):
        if os.path.exists(f) and not metrics:
            with open(f) as fh:
                metrics = [json.loads(line) for line in fh if line.strip()]
    evals = {}
    for f in glob.glob(os.path.join(args.run_dir, "eval_*.json")):
        with open(f) as fh:
            evals[os.path.basename(f)] = json.load(fh)
    summary = {"run": os.path.abspath(args.run_dir), "config": cfg, "videos": videos, "metrics": metrics,
               "evaluations": evals}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"wrote {os.path.join(out, 'summary.json')}: {len(videos)} videos, {len(metrics)} metric lines")

