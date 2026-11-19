"""Train a dance generator (Task 3), locally or as a SageMaker training job.

    python -m beat2body train                                   # EDGE configuration (see below)
    python -m beat2body train --model gru --steps 20000         # GRU baseline
    python -m beat2body train --tiny --steps 300                # smoke test (small model, CPU ok)

Defaults follow EDGE (github.com/Stanford-TML/EDGE): 5 s windows (150 frames), batch 64, Adan
lr 4e-4 weight decay 0.02, constant learning rate, no gradient clipping, EMA 0.9999 every step,
d 512 / ff 1024 / 8 layers / 8 heads / dropout 0.1, music dropped 25% (+ genre dropped 25%),
guidance 2, DDIM 50 steps eta 1, loss weights 0.636 / 2.964 / 0.646 / 10.942. EDGE trains 2000
epochs; with 2155 train clips and batch 64 that is ~66k steps (the default).

Outputs in --out (on SageMaker: $SM_MODEL_DIR, uploaded to S3 at the end):
    model.pt        EMA weights, config, motion normalizer, music stats, categories
    model_raw.pt    the same with the plain (non-averaged) weights (diffusion only)
    metrics.jsonl   one JSON line per log / validation step (also printed for CloudWatch)
    samples/        generated val windows (.npz) at every --sample-every
Resumable checkpoints go to --checkpoint-dir (SageMaker: /opt/ml/checkpoints, kept across spot restarts).

Validation: loss on val windows, and for generated val windows the joint speed relative to real
motion (`speed_ratio`, near 0 = frozen) and the share of frames with predicted foot contact.
"""
import contextlib
import copy
import json
import os
import time

import numpy as np
import torch

from beat2body import aist, paths
from beat2body.data import (
    CONTACT,
    FEATURE_SETS,
    MOTION_DIM,
    DanceClips,
    audio_dim,
    audio_stats,
    ensure_data,
    music_input,
)
from beat2body.model.adan import Adan
from beat2body.model.diffusion import EDGE_WEIGHTS, EDGEDiffusion, Normalizer, absolute_joints, gru_generate, gru_loss
from beat2body.model.edge import EDGEDecoder
from beat2body.model.nets import GRUBaseline
from beat2body.paths import TOOLS

DOC = __doc__


def add_args(p):
    p.add_argument("--model", choices=["diffusion", "gru"], default="diffusion")
    p.add_argument("--tool", default="dwpose", choices=TOOLS)
    p.add_argument("--feature-set", default="edge", choices=list(FEATURE_SETS))
    p.add_argument("--seq-len", type=int, default=150, help="frames per window (EDGE: 5 s at 30 fps)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=66000, help="EDGE: 2000 epochs = ~66k steps here")
    g = p.add_argument_group("optimizer")
    g.add_argument("--optimizer", choices=["adan", "adamw"], default="adan")
    g.add_argument("--lr", type=float, default=4e-4)
    g.add_argument("--weight-decay", type=float, default=0.02)
    g.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    g.add_argument("--warmup", type=int, default=0)
    g.add_argument("--grad-clip", type=float, default=0.0, help="0 = off (EDGE)")
    g.add_argument("--ema", type=float, default=0.9999,
                   help="max EMA decay (EDGE: 0.9999); warmed up as min(ema, (1+step)/(10+step))")
    g = p.add_argument_group("model")
    g.add_argument("--d-model", type=int, default=512)
    g.add_argument("--ff", type=int, default=1024)
    g.add_argument("--layers", type=int, default=8)
    g.add_argument("--heads", type=int, default=8)
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--gru-hidden", type=int, default=256)
    g.add_argument("--gru-layers", type=int, default=2)
    g = p.add_argument_group("diffusion")
    g.add_argument("--diffusion-steps", type=int, default=1000)
    g.add_argument("--music-drop", type=float, default=0.25)
    g.add_argument("--genre-drop", type=float, default=0.25)
    g.add_argument("--guidance", type=float, default=2.0)
    g.add_argument("--loss-weights", type=float, nargs=4, default=list(EDGE_WEIGHTS),
                   metavar=("RECON", "VEL", "FK", "FOOT"))
    g.add_argument("--sample-steps", type=int, default=50)
    g = p.add_argument_group("run")
    g.add_argument("--log-every", type=int, default=100)
    g.add_argument("--val-every", type=int, default=2000)
    g.add_argument("--sample-every", type=int, default=10000)
    g.add_argument("--out", default=os.environ.get("SM_MODEL_DIR", os.path.join(paths.REPO_ROOT, "runs", "latest")))
    g.add_argument("--checkpoint-dir", default=os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
                   if os.environ.get("SM_MODEL_DIR") else None)
    g.add_argument("--download", action="store_true", help="download the dataset from Hugging Face if missing")
    g.add_argument("--device", default="auto")
    g.add_argument("--amp", action="store_true", help="bfloat16 mixed precision on CUDA (~1.5-2x faster)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--tiny", action="store_true", help="small model, frequent logging: smoke test")
    g.add_argument("--max-train-clips", type=int, default=None,
                   help="train on the first N clips only (overfitting checks); validation uses them too")


CFG_KEYS = ("model", "tool", "feature_set", "seq_len", "d_model", "ff", "layers", "heads", "dropout",
            "gru_hidden", "gru_layers", "diffusion_steps", "music_drop", "genre_drop", "guidance", "loss_weights")


def build_model(cfg):
    cond_dim = audio_dim(cfg["feature_set"])
    if cfg["model"] == "diffusion":
        return EDGEDecoder(MOTION_DIM, cond_dim, cfg["n_genres"], cfg["seq_len"], cfg["d_model"], cfg["ff"],
                           cfg["layers"], cfg["heads"], cfg["dropout"])
    return GRUBaseline(MOTION_DIM, cond_dim, cfg["n_genres"], cfg["gru_hidden"], cfg["gru_layers"],
                       dropout=cfg["dropout"])


class Generator:
    """A trained model + its normalizer: generates denormalized motion (B, L, 38)."""

    def __init__(self, model, cfg, lo, hi, device):
        self.model, self.cfg, self.device = model.to(device).eval(), cfg, device
        self.normalizer = Normalizer(lo, hi, device)
        self.diffusion = EDGEDiffusion(self.model, self.normalizer, cfg["diffusion_steps"], cfg["music_drop"],
                                       cfg["genre_drop"], cfg["guidance"], tuple(cfg["loss_weights"])) \
            if cfg["model"] == "diffusion" else None

    @torch.no_grad()
    def generate(self, cond, genre, first=None, guidance=None, steps=50, link_windows=False, generator=None):
        cond, genre = cond.to(self.device), genre.to(self.device)
        if self.diffusion:
            x = self.diffusion.sample(cond, genre, steps=steps, guidance=guidance, link_windows=link_windows,
                                      generator=generator)
        else:
            if first is None:
                first = self.normalizer.norm((self.normalizer.lo + self.normalizer.hi) / 2).expand(len(genre), -1)
            x = gru_generate(self.model, cond, genre, first.to(self.device))
        return self.normalizer.denorm(x)


def save_bundle(path, model, cfg, lo, hi, stats, categories):
    torch.save({"model": model.state_dict(), "config": cfg, "motion_lo": lo, "motion_hi": hi,
                "audio_stats": stats, "categories": categories}, path)


def load_generator(path, device="cpu"):
    b = torch.load(path, map_location=device, weights_only=False)
    model = build_model(b["config"])
    model.load_state_dict(b["model"])
    return Generator(model, b["config"], b["motion_lo"], b["motion_hi"], device), b


def joint_speed(motion):
    """Mean absolute-joint speed (torso lengths / frame) of denormalized motion (B, L, 38)."""
    j = absolute_joints(motion)
    return (j[:, 1:] - j[:, :-1]).norm(dim=-1).mean().item()


def run(args):
    if args.tiny:
        args.d_model, args.ff, args.layers, args.heads = 64, 128, 2, 2
        args.log_every, args.val_every, args.sample_every = 100, 500, 1000
        args.ema = min(args.ema, 0.99)
    torch.manual_seed(args.seed)
    dev = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device
    if args.download:
        ensure_data(args.tool)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "metrics.jsonl")

    stats = audio_stats(args.tool, args.feature_set)
    categories = aist.load_categories()
    train = DanceClips(args.tool, "train", args.seq_len, args.feature_set, stats=stats)
    val = DanceClips(args.tool, "val", args.seq_len, args.feature_set, stats=stats)
    if args.max_train_clips:
        train.rows = train.rows[:args.max_train_clips]
        train.augment = False
        val = train
    lo, hi = train.motion_range()
    cfg = {k: getattr(args, k) for k in CFG_KEYS}
    cfg["n_genres"] = len(categories)
    model = build_model(cfg).to(dev)
    gen = Generator(model, cfg, lo, hi, dev)
    model.train()
    norm = gen.normalizer
    ema = copy.deepcopy(model).eval() if args.model == "diffusion" else None
    for p_ in (ema.parameters() if ema is not None else []):
        p_.requires_grad_(False)
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = Adan(params, lr=args.lr, weight_decay=args.weight_decay) if args.optimizer == "adan" \
        else torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def lr_factor(s):
        w = min(1.0, (s + 1) / args.warmup) if args.warmup else 1.0
        return w if args.schedule == "constant" else w * 0.5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

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

    gpu = {"gpu": torch.cuda.get_device_name(0)} if dev == "cuda" else {}
    log({"event": "start", **{k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
         **gpu, "device": dev, "params": sum(p_.numel() for p_ in model.parameters()),
         "train_clips": len(train), "val_clips": len(val)})
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=False)

    amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp and dev == "cuda" else contextlib.nullcontext()

    def loss_on(net, b):
        with amp:
            return _loss_on(net, b)

    def _loss_on(net, b):
        x0 = norm.norm(b["motion"].to(dev))
        cond, genre = b["audio"].to(dev), b["category"].to(dev)
        valid, root_valid = b["valid"].to(dev), b["root_valid"].to(dev)
        if args.model == "diffusion":
            d = gen.diffusion if net is model else EDGEDiffusion(net, norm, args.diffusion_steps, args.music_drop,
                                                               args.genre_drop, args.guidance,
                                                               tuple(args.loss_weights))
            return d.loss(x0, cond, genre, valid, root_valid)
        return gru_loss(net, x0, cond, genre, valid, root_valid, norm)

    def validate():
        net = ema if ema is not None else model
        net.eval()
        state = torch.random.get_rng_state()
        torch.manual_seed(1234)                       # same diffusion steps and noise every time
        with torch.no_grad():
            losses = [loss_on(net, b)[0].item() for b in val_loader]
        torch.random.set_rng_state(state)
        model.train()
        return float(np.mean(losses))

    def sample_val(n=8):
        net = ema if ema is not None else model
        g = Generator(net, cfg, lo, hi, dev)
        b = next(iter(torch.utils.data.DataLoader(val, batch_size=n, shuffle=False)))
        real = b["motion"].to(dev)
        state = torch.random.get_rng_state()
        torch.manual_seed(0)
        m = g.generate(b["audio"], b["category"], first=norm.norm(real[:, 0]), steps=args.sample_steps)
        torch.random.set_rng_state(state)
        os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
        for i in range(min(n, 4)):
            name = f"step{step:06d}_{b['video_id'][i]}_{int(b['start'][i]):04d}.npz"
            np.savez(os.path.join(args.out, "samples", name), motion=m[i].cpu().numpy())
        model.train()
        return {"speed_ratio": joint_speed(m) / max(joint_speed(real), 1e-8),
                "contact_rate": (m[..., CONTACT] > 0.5).float().mean().item()}

    t0, it = time.time(), iter(loader)
    while step < args.steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        loss, parts = loss_on(model, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        sched.step()
        if ema is not None:
            # EMA warm-up: early on the average follows the model, so short runs and early
            # validation are not dominated by the random initial weights (EDGE: fixed 0.9999)
            decay = min(args.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - decay)
        step += 1
        if step % args.log_every == 0:
            mem = {"gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)} if dev == "cuda" else {}
            log({"loss": loss.item(), **parts, "lr": sched.get_last_lr()[0],
                 "steps_per_s": args.log_every / (time.time() - t0), **mem})
            t0 = time.time()
        if step % args.val_every == 0 or step == args.steps:
            log({"val_loss": validate()})
        if step % args.sample_every == 0 or step == args.steps:
            log({"sample": sample_val()})
        if ckpt and (step % args.val_every == 0 or step == args.steps):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "ema": ema.state_dict() if ema is not None else None, "step": step}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)

    save_bundle(os.path.join(args.out, "model.pt"), ema if ema is not None else model, cfg, lo, hi, stats, categories)
    if ema is not None:          # the plain (non-averaged) weights too, to compare
        save_bundle(os.path.join(args.out, "model_raw.pt"), model, cfg, lo, hi, stats, categories)
    log({"event": "done", "model": os.path.join(args.out, "model.pt")})


# ---------------------------------------------------------------------------
# sample: generate a dance for a dataset video's music (EDGE long-form sampling)
# ---------------------------------------------------------------------------
SAMPLE_DOC = """Generate a dance for the music of a dataset video (any split) with a trained model.

    python -m beat2body sample --checkpoint runs/<job>/model.pt --video-id gLH_sFM_c01_d16_mLH4_ch05
    python -m beat2body sample ... --genre krump --seconds 20 --guidance 3
    python -m beat2body render --video-id <id> --keypoints runs/<job>/generated/<id>.npz

Long pieces use EDGE's long-form sampling: windows of seq_len frames that share half their frames,
denoised together and cross-faded. Output .npz: motion (T, 38), keypoints (T, 17, 2) pelvis-centred,
root (T, 2), contact (T, 2), absolute (T, 17, 2) = keypoints + root (torso lengths, image centre at 0).
"""


def add_sample_args(p):
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video-id", required=True, help="music of this dataset video")
    p.add_argument("--genre", default=None, help="genre to dance (default: the video's genre)")
    p.add_argument("--seconds", type=float, default=None, help="length (default: the whole video)")
    p.add_argument("--guidance", type=float, default=None, help="guidance weight (default: the trained one)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("-o", "--out", default=None)


def run_sample(args):
    gen, bundle = load_generator(args.checkpoint, args.device)
    cfg, cats = bundle["config"], bundle["categories"]
    vid = args.video_id
    cond = torch.from_numpy(music_input(vid, cfg["feature_set"], bundle["audio_stats"]))
    T = len(cond) if args.seconds is None else min(len(cond), int(args.seconds * 30))
    L, half = cfg["seq_len"], cfg["seq_len"] // 2
    n = max(1, int(np.ceil((T - L) / half)) + 1) if T > L else 1
    need = half * (n + 1) if n > 1 else L
    cond = torch.cat([cond, cond[-1:].expand(max(0, need - len(cond)), -1)])[:need]
    windows = torch.stack([cond[i * half:i * half + L] for i in range(n)])
    genre_name = args.genre or aist.GENRES[vid[1:3]]
    genre = torch.full((n,), cats[genre_name], dtype=torch.long)
    g = torch.Generator(device=args.device).manual_seed(args.seed)
    m = gen.generate(windows, genre, guidance=args.guidance, steps=args.steps, link_windows=n > 1, generator=g)
    motion = (EDGEDiffusion.stitch(m) if n > 1 else m[0])[:T].cpu()
    seam = (m[1:, :half] - m[:-1, half:]).abs().mean().item() if n > 1 else 0.0
    kp = motion[:, 4:].reshape(T, -1, 2)
    path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "generated", f"{vid}.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, motion=motion.numpy(), keypoints=kp.numpy(), root=motion[:, 2:4].numpy(),
             contact=motion[:, :2].numpy(), absolute=absolute_joints(motion).numpy())
    print(f"wrote {path}: {T} frames ({T / 30:.1f} s), {n} windows, genre {genre_name}, music {aist.music_id(vid)}"
          + (f"; overlap disagreement before cross-fade {seam:.3f} torso lengths (mean)" if n > 1 else ""))
