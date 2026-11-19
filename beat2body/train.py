"""Train a dance generator (Task 3). Runs locally or as a SageMaker training job.

    python -m beat2body train --model diffusion --steps 60000          # main model
    python -m beat2body train --model gru --steps 20000                # README baseline
    python -m beat2body train --model diffusion --steps 200 --tiny     # quick smoke test (CPU ok)

Outputs in --out (on SageMaker: $SM_MODEL_DIR, uploaded to S3 at the end):
    model.pt        weights (EMA for diffusion), config, normalization, audio stats, categories
    metrics.jsonl   one JSON line per log / validation step (also printed, so CloudWatch has it)
    samples/        generated val clips (.npy, (L, 17, 2)) at every --sample-every
Checkpoints for resuming (spot interruptions) go to --checkpoint-dir (SageMaker: /opt/ml/checkpoints).

Validation reports the loss on val clips and, for generated clips, the mean joint speed relative
to real motion ("speed ratio"): far below 1 means the model collapsed to a frozen pose.
"""
import copy
import json
import math
import os
import time

import numpy as np
import torch

from beat2body import aist, paths
from beat2body.data import AUDIO_DIM, DanceClips, audio_stats, ensure_data
from beat2body.model.diffusion import GaussianDiffusion, gru_generate, gru_loss
from beat2body.model.nets import POSE_DIM, DanceDenoiser, GRUBaseline
from beat2body.paths import TOOLS

DOC = __doc__


def add_args(p):
    p.add_argument("--model", choices=["diffusion", "gru"], default="diffusion")
    p.add_argument("--tool", default="dwpose", choices=TOOLS)
    p.add_argument("--seq-len", type=int, default=120, help="frames per training window (4 s)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=6, help="transformer layers (diffusion) / GRU layers (gru: use 2)")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--vel-weight", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--sample-every", type=int, default=10000)
    p.add_argument("--sample-steps", type=int, default=50, help="DDIM steps when sampling")
    p.add_argument("--out", default=os.environ.get("SM_MODEL_DIR", os.path.join(paths.REPO_ROOT, "runs", "latest")))
    p.add_argument("--checkpoint-dir", default=os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
                   if os.environ.get("SM_MODEL_DIR") else None)
    p.add_argument("--download", action="store_true", help="download the dataset from Hugging Face if missing")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tiny", action="store_true", help="small model, frequent logging: smoke test")
    p.add_argument("--max-train-clips", type=int, default=None,
                   help="train on the first N clips only (overfitting checks); val samples then come from them")


def pose_stats(ds):
    """Per-dimension mean / std of the normalized keypoints over the training clips."""
    kp = np.concatenate([c["keypoints"].reshape(len(c["keypoints"]), -1) for c in ds._cache])
    return kp.mean(0).astype(np.float32), np.maximum(kp.std(0), 1e-3).astype(np.float32)


def build_model(cfg):
    if cfg["model"] == "diffusion":
        return DanceDenoiser(cfg["n_genres"], AUDIO_DIM, cfg["d_model"], cfg["layers"], cfg["heads"],
                             dropout=cfg["dropout"])
    return GRUBaseline(cfg["n_genres"], AUDIO_DIM, cfg["d_model"], cfg["layers"], dropout=cfg["dropout"])


class Generator:
    """Model + normalization: what `sample` and evaluation need."""

    def __init__(self, model, cfg, mean, std, device):
        self.model, self.cfg, self.device = model.to(device).eval(), cfg, device
        self.mean = torch.as_tensor(mean, device=device)
        self.std = torch.as_tensor(std, device=device)
        self.diffusion = GaussianDiffusion(self.model) if cfg["model"] == "diffusion" else None

    def norm(self, kp):            # (B, L, 17, 2) -> (B, L, 34)
        return (kp.flatten(2) - self.mean) / self.std

    def denorm(self, x):           # (B, L, 34) -> (B, L, 17, 2)
        return (x * self.std + self.mean).view(*x.shape[:2], -1, 2)

    @torch.no_grad()
    def generate(self, audio, genre, first_pose=None, guidance=1.0, steps=50, known=None, known_mask=None):
        audio, genre = audio.to(self.device), genre.to(self.device)
        if self.diffusion:
            x = self.diffusion.sample(audio, genre, steps=steps, guidance=guidance, known=known, known_mask=known_mask)
        else:
            if first_pose is None:
                first_pose = self.mean.expand(len(genre), -1)
            x = gru_generate(self.model, audio, genre, first_pose)
        return self.denorm(x)


def save_bundle(path, model, cfg, mean, std, stats, categories):
    torch.save({"model": model.state_dict(), "config": cfg, "pose_mean": mean, "pose_std": std,
                "audio_stats": stats, "categories": categories}, path)


def load_generator(path, device="cpu"):
    b = torch.load(path, map_location=device, weights_only=False)
    model = build_model(b["config"])
    model.load_state_dict(b["model"])
    return Generator(model, b["config"], b["pose_mean"], b["pose_std"], device), b


def mean_speed(kp):
    return (kp[:, 1:] - kp[:, :-1]).norm(dim=-1).mean().item()


def run(args):
    if args.tiny:
        args.d_model, args.layers, args.heads = 64, 2, 2
        args.log_every, args.val_every, args.sample_every = 100, 500, 1000
        args.ema = min(args.ema, 0.99)
    torch.manual_seed(args.seed)
    dev = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device
    if args.download:
        ensure_data(args.tool)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "metrics.jsonl")

    stats = audio_stats(args.tool)
    categories = aist.load_categories()
    train = DanceClips(args.tool, "train", seq_len=args.seq_len, stats=stats)
    val = DanceClips(args.tool, "val", seq_len=args.seq_len, stats=stats)
    if args.max_train_clips:
        train.rows, train._cache = train.rows[:args.max_train_clips], train._cache[:args.max_train_clips]
        train.augment = False
        val = train
    mean, std = pose_stats(train)
    cfg = {k: getattr(args, k) for k in ("model", "tool", "seq_len", "d_model", "layers", "heads", "dropout")}
    cfg["n_genres"] = len(categories)
    model = build_model(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    gen = Generator(model, cfg, mean, std, dev)
    ema = copy.deepcopy(model).eval() if args.model == "diffusion" else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, min(args.warmup, args.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    step = 0
    ckpt = os.path.join(args.checkpoint_dir, "last.pt") if args.checkpoint_dir else None
    if ckpt and os.path.exists(ckpt):
        s = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        if ema is not None:
            ema.load_state_dict(s["ema"])
        step = s["step"]
        print(f"[train] resumed from {ckpt} at step {step}")

    def log(rec):
        rec = {"step": step, **rec}
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    log({"event": "start", **{k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
         "device": dev, "params": n_params, "train_clips": len(train), "val_clips": len(val)})
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=False)
    diffusion = GaussianDiffusion(model) if args.model == "diffusion" else None

    def batch_to(b):
        kp = b["keypoints"].to(dev)
        return gen.norm(kp), b["audio"].to(dev), b["category"].to(dev), b["valid"].to(dev), kp

    def loss_fn(net, x0, audio, genre, valid):
        if diffusion is not None:
            d = diffusion if net is model else GaussianDiffusion(net)
            return d.loss(x0, audio, genre, valid, args.vel_weight)
        return gru_loss(net, x0, audio, genre, valid, vel_weight=args.vel_weight)

    def validate():
        net = ema if ema is not None else model
        net.eval()
        torch.manual_seed(1234)                         # same diffusion steps / noise every time
        losses = []
        with torch.no_grad():
            for b in val_loader:
                x0, audio, genre, valid, _ = batch_to(b)
                losses.append(loss_fn(net, x0, audio, genre, valid)[0].item())
        net.train() if net is model else None
        return float(np.mean(losses))

    def sample_val(n=8):
        net = ema if ema is not None else model
        g = Generator(net, cfg, mean, std, dev)
        b = next(iter(torch.utils.data.DataLoader(val, batch_size=n, shuffle=False)))
        torch.manual_seed(0)
        kp = g.generate(b["audio"], b["category"], first_pose=gen.norm(b["keypoints"].to(dev))[:, 0],
                        steps=args.sample_steps)
        real = b["keypoints"].to(dev)
        os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
        for i in range(min(n, 4)):
            np.save(os.path.join(args.out, "samples", f"step{step:06d}_{b['video_id'][i]}_{b['start'][i]:04d}.npy"),
                    kp[i].cpu().numpy())
        model.train()
        return mean_speed(kp) / max(mean_speed(real), 1e-8)

    model.train()
    t0, it = time.time(), iter(loader)
    while step < args.steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        x0, audio, genre, valid, _ = batch_to(b)
        loss, parts = loss_fn(model, x0, audio, genre, valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - args.ema)
        step += 1
        if step % args.log_every == 0:
            log({"loss": loss.item(), **parts, "lr": sched.get_last_lr()[0],
                 "steps_per_s": args.log_every / (time.time() - t0)})
            t0 = time.time()
        if step % args.val_every == 0 or step == args.steps:
            log({"val_loss": validate()})
        if step % args.sample_every == 0 or step == args.steps:
            log({"sample_speed_ratio": sample_val()})
        if ckpt and (step % args.val_every == 0 or step == args.steps):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "ema": ema.state_dict() if ema is not None else None, "step": step}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)

    save_bundle(os.path.join(args.out, "model.pt"), ema if ema is not None else model, cfg, mean, std,
                stats, categories)
    log({"event": "done", "model": os.path.join(args.out, "model.pt")})


# ---------------------------------------------------------------------------
# sample: generate a dance for a dataset video's music
# ---------------------------------------------------------------------------
SAMPLE_DOC = """Generate a dance for the music of a dataset video (any split) with a trained model.

    python -m beat2body sample --checkpoint runs/latest/model.pt --video-id gLH_sFM_c01_d16_mLH4_ch05
    python -m beat2body sample ... --genre krump --seconds 20 --guidance 2
    python -m beat2body render --video-id <id> --keypoints runs/latest/generated/<id>.npy

Long pieces are generated window by window; each window repeats the last --overlap frames of the
previous one as fixed frames (diffusion inpainting), so the motion continues smoothly.
"""


def add_sample_args(p):
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video-id", required=True, help="music (audio features) of this dataset video")
    p.add_argument("--genre", default=None, help="genre to dance (default: the video's genre)")
    p.add_argument("--seconds", type=float, default=None, help="length (default: the whole video)")
    p.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance weight (diffusion)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--overlap", type=int, default=30, help="frames shared by consecutive windows")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("-o", "--out", default=None)


def run_sample(args):
    from beat2body.data import normalize_audio
    gen, bundle = load_generator(args.checkpoint, args.device)
    cfg, cats = bundle["config"], bundle["categories"]
    vid = args.video_id
    mel = np.load(paths.feature_path("audio_mel", vid)).astype(np.float32)
    onset = np.load(paths.feature_path("audio_onset", vid)).astype(np.float32)
    audio = torch.from_numpy(normalize_audio(mel, onset, bundle["audio_stats"]))[None]
    T = len(audio[0]) if args.seconds is None else min(len(audio[0]), int(args.seconds * 30))
    genre_name = args.genre or aist.GENRES[vid[1:3]]
    genre = torch.tensor([cats[genre_name]])
    torch.manual_seed(args.seed)
    W, ov = cfg["seq_len"], args.overlap
    out = None
    s = 0
    while s < T:
        e = min(s + W, T)
        a = audio[:, s:e]
        if out is None or gen.diffusion is None:
            first = None if out is None else gen.norm(out[:, -1:])[:, 0]
            win = gen.generate(a, genre, first_pose=first, steps=args.steps, guidance=args.guidance)
            out = win if out is None else torch.cat([out, win], 1)
            s = e
        else:
            k = min(ov, out.shape[1])
            a = audio[:, s - k:e]
            known = torch.zeros(1, a.shape[1], POSE_DIM, device=gen.device)
            known[:, :k] = gen.norm(out[:, -k:])
            mask = torch.zeros(1, a.shape[1], dtype=torch.bool, device=gen.device)
            mask[:, :k] = True
            win = gen.generate(a, genre, steps=args.steps, guidance=args.guidance, known=known, known_mask=mask)
            out = torch.cat([out, win[:, k:]], 1)
            s = e
    path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "generated", f"{vid}.npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, out[0, :T].cpu().numpy())
    print(f"wrote {path}: {T} frames ({T / 30:.1f} s), genre {genre_name}, music {aist.music_id(vid)}")
