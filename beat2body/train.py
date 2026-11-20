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
    samples/        generated val windows (.npz: generated + real motion) at every --sample-every
    snapshots/      model at every --snapshot-every steps (EMA weights, same format as model.pt)
With --checkpoint-dir (always on SageMaker), metrics.jsonl, samples/ and snapshots/ are also written
there as training runs; SageMaker syncs that folder to S3 continuously, so progress can be
downloaded and shared before the job ends (`python -m beat2body aws download JOB`).
Resumable checkpoints go to --checkpoint-dir (SageMaker: /opt/ml/checkpoints, kept across spot restarts).

Validation: loss on val windows, and for generated val windows the joint speed relative to real
motion (`speed_ratio`, near 0 = frozen), the share of frames with predicted foot contact, and the
drift detector `bone_cv` (bone-length variation; compare with `bone_cv_real`).
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
    SCALE_NORMS,
    DanceClips,
    audio_dim,
    audio_stats,
    ensure_data,
    music_input,
)
from beat2body.model.adan import Adan
from beat2body.model.diffusion import (
    EDGE_WEIGHTS,
    EDGEDiffusion,
    Normalizer,
    absolute_joints,
    bone_lengths,
    gru_generate,
    gru_loss,
)
from beat2body.model.edge import EDGEDecoder
from beat2body.model.nets import GRUBaseline
from beat2body.paths import TOOLS

DOC = __doc__


# Training recipes. "edge" = EDGE's settings; "regularized" (default) adds regularization and augmentation
# against overfitting (a 2,155-clip dataset). Any flag given explicitly overrides the recipe.
RECIPES = {
    "edge": dict(dropout=0.1, weight_decay=0.02, schedule="constant", warmup=0, grad_clip=0.0, extra_tools=[],
                 stretch=0.0, jitter_scale=0.0, jitter_rotate=0.0, jitter_shift=0.0, min_snr=0.0, select="last"),
    "regularized": dict(dropout=0.2, weight_decay=0.1, schedule="cosine", warmup=1000, grad_clip=1.0,
                        extra_tools=["openpose", "mediapipe"], stretch=0.1, jitter_scale=0.05, jitter_rotate=5.0,
                        jitter_shift=0.2, min_snr=5.0, select="fid_k"),
}


def add_args(p):
    p.add_argument("--recipe", choices=list(RECIPES), default="regularized",
                   help="regularized (default): dropout 0.2, weight decay 0.1 (not on norms / biases / embeddings), "
                        "cosine LR + warmup, grad clip 1, 3 pose tools, time-stretch + jitter augmentation, Min-SNR, "
                        "best-checkpoint selection by val FID_k; edge: EDGE's original settings")
    p.add_argument("--model", choices=["diffusion", "gru"], default="diffusion")
    p.add_argument("--tool", default="dwpose", choices=TOOLS)
    p.add_argument("--feature-set", default="edge", choices=list(FEATURE_SETS))
    p.add_argument("--scale-norm", default="dancer", choices=list(SCALE_NORMS),
                   help="dancer (default): every dancer the same body size, measured over all their videos; "
                        "video: per video (inflates floor-work videos); dataset: one scale for all (EDGE-like)")
    p.add_argument("--seq-len", type=int, default=150, help="frames per window (EDGE: 5 s at 30 fps)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=66000, help="EDGE: 2000 epochs = ~66k steps here")
    g = p.add_argument_group("optimizer")
    g.add_argument("--optimizer", choices=["adan", "adamw"], default="adan")
    g.add_argument("--lr", type=float, default=4e-4)
    g.add_argument("--weight-decay", type=float, default=None, help="recipe: edge 0.02, regularized 0.1")
    g.add_argument("--schedule", choices=["constant", "cosine"], default=None, help="recipe: edge constant")
    g.add_argument("--warmup", type=int, default=None)
    g.add_argument("--grad-clip", type=float, default=None, help="0 = off (EDGE)")
    g.add_argument("--ema", type=float, default=0.9999,
                   help="max EMA decay (EDGE: 0.9999); warmed up as min(ema, (1+step)/(10+step))")
    g = p.add_argument_group("model")
    g.add_argument("--d-model", type=int, default=512)
    g.add_argument("--ff", type=int, default=1024)
    g.add_argument("--layers", type=int, default=8)
    g.add_argument("--heads", type=int, default=8)
    g.add_argument("--dropout", type=float, default=None, help="recipe: edge 0.1, regularized 0.2")
    g.add_argument("--beat-attn", action="store_true",
                   help="beat-relative cross-attention: learned per-head bias on the music frames by their offset "
                        "from the motion frame in beats (anticipation / delay / offbeat) and by their beat phase")
    g.add_argument("--beat-bins", type=int, default=8, help="beat-relative bias bins per beat")
    g.add_argument("--beat-range", type=float, default=2.0, help="beat-relative offsets clamped to +/- this, beats")
    g.add_argument("--gru-hidden", type=int, default=256)
    g.add_argument("--gru-layers", type=int, default=2)
    g = p.add_argument_group("diffusion")
    g.add_argument("--diffusion-steps", type=int, default=1000)
    g.add_argument("--music-drop", type=float, default=0.25)
    g.add_argument("--genre-drop", type=float, default=0.25)
    g.add_argument("--guidance", type=float, default=2.0)
    g.add_argument("--loss-weights", type=float, nargs=4, default=list(EDGE_WEIGHTS),
                   metavar=("RECON", "VEL", "FK", "FOOT"))
    g.add_argument("--bone-weight", type=float, default=1.0,
                   help="projected bone-length loss (on-screen bone lengths vs the real frame); 0 = off (EDGE)")
    g.add_argument("--sample-steps", type=int, default=50)
    g.add_argument("--min-snr", type=float, default=None, help="Min-SNR loss weighting gamma; 0 = off (EDGE)")
    g.add_argument("--normalizer", choices=["minmax", "zscore"], default="minmax",
                   help="minmax (EDGE): features to [-1, 1]; zscore: standardized, so fine motion stands out "
                        "from the diffusion noise (recon / velocity losses rescaled to equal the MinMax ones)")
    g.add_argument("--param", choices=["x0", "v"], default="x0",
                   help="what the network predicts: x0 (EDGE) or v (x0 derived from it; skip path at low noise)")
    g.add_argument("--energy-weight", type=float, default=0.0,
                   help="phase-invariant loss on each joint's speed and amplitude per window; 0 = off (EDGE)")
    g.add_argument("--energy-smooth", type=float, default=2.0,
                   help="energy loss: speed measured on motion blurred over this many frames (jitter does not "
                        "count); 0 = raw speed (the first version, which the model met partly with jitter)")
    g.add_argument("--energy-amplitude", type=float, default=1.0,
                   help="energy loss: weight of the movement-amplitude term (0 = speed only; the amplitude term "
                        "pushed limbs to ~1.3x the real range)")
    g.add_argument("--energy-shape", choices=["square", "gaussian"], default="square",
                   help="energy loss per window and joint: square = d^2 of the log ratio to real (every window pulled "
                        "to its real energy); gaussian = 2 sigma^2 (1 - exp(-d^2 / 2 sigma^2)): same for small d, "
                        "bounded for large d")
    g.add_argument("--energy-sigma", type=float, default=0.3, help="gaussian energy loss width (log ratio)")
    g.add_argument("--energy-size-norm", action="store_true",
                   help="energy loss on motion with frame-to-frame body-size changes removed (size pulses and depth "
                        "changes do not count as movement)")
    g.add_argument("--energy-max-t", type=int, default=None,
                   help="energy loss only at diffusion steps t < this (500 = noise below half the signal); "
                        "default: all steps")
    g.add_argument("--foot-contact", choices=["pred", "label", "label-prev"], default="pred",
                   help="foot-skating loss on frames where the ankle is planted according to the model's own contact "
                        "output (pred, EDGE; averaged over all frames) or the real dancer (label; averaged over the "
                        "planted frames; label-prev = the first, one-step-late version used by runs 4.1 / 4.2). "
                        "Its weight is the 4th --loss-weights value, or --foot-weight")
    g.add_argument("--foot-weight", type=float, default=None,
                   help="foot-skating loss weight (overrides the 4th --loss-weights value; one value, so it passes "
                        "through `aws submit`)")
    g.add_argument("--root-accel-weight", type=float, default=0.0,
                   help="loss on the root's acceleration vs the real one (against root jitter); 0 = off")
    g.add_argument("--foot-max-t", type=int, default=None,
                   help="label foot loss and root-acceleration loss only at diffusion steps t < this (500 = noise "
                        "below half the signal); default: all steps")
    g = p.add_argument_group("augmentation (train split only)")
    g.add_argument("--extra-tools", nargs="*", choices=TOOLS, default=None,
                   help="other pose tools' keypoints of the same videos as extra training views")
    g.add_argument("--stretch", type=float, default=None, help="time-stretch motion + music by up to +/- this")
    g.add_argument("--jitter-scale", type=float, default=None)
    g.add_argument("--jitter-rotate", type=float, default=None, help="degrees")
    g.add_argument("--jitter-shift", type=float, default=None, help="root shift, torso lengths")
    g = p.add_argument_group("run")
    g.add_argument("--select", choices=["val_loss", "fid_k", "last"], default=None,
                   help="model.pt = the checkpoint with the lowest validation loss (checked every --val-every), "
                        "the best val FID_k (on 128 windows, every --sample-every) or the last one")
    g.add_argument("--log-every", type=int, default=100)
    g.add_argument("--val-every", type=int, default=2000)
    g.add_argument("--sample-every", type=int, default=10000)
    g.add_argument("--snapshot-every", type=int, default=None,
                   help="save a model snapshot every N steps (default: --sample-every; 0 = off)")
    g.add_argument("--out", default=os.environ.get("SM_MODEL_DIR", os.path.join(paths.REPO_ROOT, "runs", "latest")))
    g.add_argument("--checkpoint-dir", default=os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
                   if os.environ.get("SM_MODEL_DIR") else None)
    g.add_argument("--download", action="store_true", help="download the dataset from Hugging Face if missing")
    g.add_argument("--device", default="auto")
    g.add_argument("--amp", action="store_true",
                   help="mixed precision on CUDA: bfloat16 where supported (A100, L4, ...), else float16 with "
                        "loss scaling (T4)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--tiny", action="store_true", help="small model, frequent logging: smoke test")
    g.add_argument("--max-train-clips", type=int, default=None,
                   help="train on the first N clips only (overfitting checks); validation uses them too")


CFG_KEYS = ("model", "tool", "feature_set", "scale_norm", "bone_weight", "min_snr", "seq_len", "d_model", "ff",
            "layers", "heads", "dropout", "gru_hidden", "gru_layers", "diffusion_steps", "music_drop", "genre_drop",
            "guidance", "loss_weights", "normalizer", "param", "energy_weight", "energy_smooth", "beat_attn",
            "beat_bins", "beat_range", "energy_amplitude", "energy_shape", "energy_sigma", "energy_size_norm",
            "energy_max_t", "foot_contact", "root_accel_weight",
            "foot_max_t")


def build_model(cfg):
    cond_dim = audio_dim(cfg["feature_set"])
    if cfg["model"] == "diffusion":
        return EDGEDecoder(MOTION_DIM, cond_dim, cfg["n_genres"], cfg["seq_len"], cfg["d_model"], cfg["ff"],
                           cfg["layers"], cfg["heads"], cfg["dropout"], cfg.get("beat_attn", False),
                           cfg.get("beat_bins", 8), cfg.get("beat_range", 2.0))
    return GRUBaseline(MOTION_DIM, cond_dim, cfg["n_genres"], cfg["gru_hidden"], cfg["gru_layers"],
                       dropout=cfg["dropout"])


def make_diffusion(net, normalizer, cfg):
    return EDGEDiffusion(net, normalizer, cfg["diffusion_steps"], cfg["music_drop"], cfg["genre_drop"],
                         cfg["guidance"], tuple(cfg["loss_weights"]), cfg.get("bone_weight", 0.0),
                         cfg.get("min_snr", 0.0), cfg.get("param", "x0"), cfg.get("energy_weight", 0.0),
                         cfg.get("energy_smooth", 0.0), cfg.get("energy_amplitude", 1.0),
                         cfg.get("energy_shape", "square"), cfg.get("energy_sigma", 0.3),
                         cfg.get("energy_size_norm", False), cfg.get("energy_max_t"),
                         cfg.get("foot_contact", "pred"), cfg.get("root_accel_weight", 0.0),
                         cfg.get("foot_max_t"))


class Generator:
    """A trained model + its normalizer: generates denormalized motion (B, L, 38)."""

    def __init__(self, model, cfg, lo, hi, device):
        self.model, self.cfg, self.device = model.to(device).eval(), cfg, device
        self.normalizer = Normalizer(lo, hi, device)
        self.diffusion = make_diffusion(self.model, self.normalizer, cfg) if cfg["model"] == "diffusion" else None

    @torch.no_grad()
    def generate(self, cond, genre, first=None, guidance=None, steps=50, link_windows=False, generator=None,
                 eta=1.0, music_guidance=None):
        """guidance: one weight for music + genre; with music_guidance: guidance = the genre weight."""
        cond, genre = cond.to(self.device), genre.to(self.device)
        if self.diffusion:
            x = self.diffusion.sample(cond, genre, steps=steps, eta=eta, guidance=guidance,
                                      link_windows=link_windows, generator=generator, music_guidance=music_guidance)
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


def bone_cv(motion):
    """Drift detector: variation over time (std / mean) of each body bone's on-screen length,
    averaged over bones and windows. Real dance is not 0 (foreshortening); match the real value."""
    lengths = bone_lengths(absolute_joints(motion))                                    # (B, L, 12)
    return (lengths.std(1) / lengths.mean(1).clamp(min=1e-6)).mean().item()


def joint_speed(motion):
    """Mean absolute-joint speed (torso lengths / frame) of denormalized motion (B, L, 38)."""
    j = absolute_joints(motion)
    return (j[:, 1:] - j[:, :-1]).norm(dim=-1).mean().item()


def run(args):
    for k, v in RECIPES[args.recipe].items():            # recipe values for flags not given explicitly
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    if args.foot_weight is not None:
        args.loss_weights = [*args.loss_weights[:3], args.foot_weight]
    if args.tiny:                    # small model; logging defaults only where not set explicitly
        args.d_model, args.ff, args.layers, args.heads = 64, 128, 2, 2
        for k, default, tiny in (("log_every", 100, 100), ("val_every", 2000, 500), ("sample_every", 10000, 1000)):
            if getattr(args, k) == default:
                setattr(args, k, tiny)
        args.ema = min(args.ema, 0.99)
    torch.manual_seed(args.seed)
    dev = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device
    if args.download:
        ensure_data(args.tool)
    os.makedirs(args.out, exist_ok=True)

    stats = audio_stats(args.tool, args.feature_set)
    categories = aist.load_categories()
    train = DanceClips(args.tool, "train", args.seq_len, args.feature_set, stats=stats, scale_norm=args.scale_norm,
                       extra_tools=args.extra_tools, stretch=args.stretch, scale=args.jitter_scale,
                       rotate=args.jitter_rotate, shift=args.jitter_shift)
    tools_used = sorted({t for ts in train.tools_of.values() for t in ts})
    if set(args.extra_tools) - set(tools_used):
        print(f"[train] note: no keypoints found for {sorted(set(args.extra_tools) - set(tools_used))}; "
              f"download them with `python -m beat2body hf download --tool <tool> --parts keypoints`")
    val = DanceClips(args.tool, "val", args.seq_len, args.feature_set, stats=stats, scale_norm=args.scale_norm)
    if args.max_train_clips:
        train.rows = train.rows[:args.max_train_clips]
        train.augment = False
        val = train
    lo, hi = train.motion_range(args.normalizer)
    cfg = {k: getattr(args, k) for k in CFG_KEYS}
    cfg["n_genres"] = len(categories)
    model = build_model(cfg).to(dev)
    gen = Generator(model, cfg, lo, hi, dev)
    model.train()
    norm = gen.normalizer
    if args.normalizer != "minmax":        # recon / velocity losses in MinMax units: same loss, more signal
        mlo, mhi = train.motion_range("minmax")
        norm.dim_weight = torch.as_tensor(((hi - lo) / (mhi - mlo).clip(1e-6)) ** 2, device=dev)
        print(f"[train] {args.normalizer} normalizer: data std in the diffusion space x"
              f"{float(np.median((mhi - mlo)[2:] / (hi - lo)[2:])):.1f} vs MinMax (median feature)")
    ema = copy.deepcopy(model).eval() if args.model == "diffusion" else None
    for p_ in (ema.parameters() if ema is not None else []):
        p_.requires_grad_(False)
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    # no weight decay on biases, norms, embeddings and learned null / position vectors
    no_decay_keys = ("null_", "genre", "embed", "pos", "beat_bias")
    no_decay = [p_ for n_, p_ in model.named_parameters()
                if p_.requires_grad and (p_.ndim < 2 or any(k in n_ for k in no_decay_keys))]
    ids = {id(p_) for p_ in no_decay}
    groups = [{"params": [p_ for p_ in params if id(p_) not in ids], "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    opt = Adan(groups, lr=args.lr) if args.optimizer == "adan" else torch.optim.AdamW(groups, lr=args.lr)

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

    live = args.checkpoint_dir or args.out          # synced to S3 during SageMaker jobs
    os.makedirs(live, exist_ok=True)
    snapshot_every = args.sample_every if args.snapshot_every is None else args.snapshot_every

    def log(rec):
        rec = {"step": step, **rec}
        print(json.dumps(rec), flush=True)
        for d in {args.out, live}:
            with open(os.path.join(d, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")

    amp_dtype = None
    if args.amp and dev == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # float16 needs loss scaling against underflow; bfloat16 does not
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    gpu = {"gpu": torch.cuda.get_device_name(0)} if dev == "cuda" else {}
    log({"event": "start", **{k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
         **gpu, "device": dev, "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else None,
         "params": sum(p_.numel() for p_ in model.parameters()),
         "train_clips": len(train), "val_clips": len(val)})
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=False)

    def loss_on(net, b):
        ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
        with ctx:
            return _loss_on(net, b)

    def _loss_on(net, b):
        x0 = norm.norm(b["motion"].to(dev))
        cond, genre = b["audio"].to(dev), b["category"].to(dev)
        valid, root_valid = b["valid"].to(dev), b["root_valid"].to(dev)
        if args.model == "diffusion":
            d = gen.diffusion if net is model else make_diffusion(net, norm, cfg)
            return d.loss(x0, cond, genre, valid, root_valid)
        return gru_loss(net, x0, cond, genre, valid, root_valid, norm, bone_weight=args.bone_weight)

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

    # fixed val windows spread over the split (different videos / genres), the same at every check
    sample_idx = np.linspace(0, len(val) - 1, 8).round().astype(int).tolist()

    def sample_val(n=8):
        net = ema if ema is not None else model
        g = Generator(net, cfg, lo, hi, dev)
        b = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(val, sample_idx[:n]), batch_size=n)))
        real = b["motion"].to(dev)
        state = torch.random.get_rng_state()
        torch.manual_seed(0)
        m = g.generate(b["audio"], b["category"], first=norm.norm(real[:, 0]), steps=args.sample_steps)
        torch.random.set_rng_state(state)
        for d in {args.out, live}:
            os.makedirs(os.path.join(d, "samples"), exist_ok=True)
            for i in range(min(n, 4)):
                name = f"step{step:06d}_{b['video_id'][i]}_{int(b['start'][i]):04d}.npz"
                np.savez(os.path.join(d, "samples", name), motion=m[i].cpu().numpy(), real=real[i].cpu().numpy(),
                         video_id=b["video_id"][i], start=int(b["start"][i]), step=step)
        model.train()
        bp = beat_power(absolute_joints(m).cpu().numpy()) / max(beat_power(absolute_joints(real).cpu().numpy()), 1e-12)
        return {"speed_ratio": joint_speed(m) / max(joint_speed(real), 1e-8), "beat_power": bp,
                "contact_rate": (m[..., CONTACT] > 0.5).float().mean().item(),
                "bone_cv": bone_cv(m), "bone_cv_real": bone_cv(real)}

    # model selection: FID_k of generated vs real motion on fixed val windows, at every sample check
    from beat2body.evaluate import beat_power, frechet, kinetic_features
    sel_idx = np.linspace(0, len(val) - 1, min(128, len(val))).round().astype(int).tolist()
    sel_real = torch.stack([val[i]["motion"] for i in sel_idx])
    real_k = kinetic_features(absolute_joints(sel_real).numpy())
    k_mu, k_sd = real_k.mean(0), real_k.std(0) + 1e-8
    best = {"fid_k": float("inf"), "val_loss": float("inf"), "step": None}

    def save_best():
        for d in {args.out, live}:
            save_bundle(os.path.join(d, "model_best.pt"), ema if ema is not None else model, cfg, lo, hi, stats,
                        categories)

    def val_fid_k():
        net = ema if ema is not None else model
        g = Generator(net, cfg, lo, hi, dev)
        state = torch.random.get_rng_state()
        torch.manual_seed(0)
        out = []
        for i in range(0, len(sel_idx), 64):
            items = [val[j] for j in sel_idx[i:i + 64]]
            out.append(g.generate(torch.stack([x["audio"] for x in items]), torch.stack([x["category"] for x in items]),
                                  first=norm.norm(torch.stack([x["motion"][0] for x in items]).to(dev)),
                                  steps=min(args.sample_steps, 25)).cpu())
        torch.random.set_rng_state(state)
        model.train()
        k = kinetic_features(absolute_joints(torch.cat(out)).numpy())
        return frechet((k - k_mu) / k_sd, (real_k - k_mu) / k_sd)

    t0, it = time.time(), iter(loader)
    while step < args.steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        loss, parts = loss_on(model, b)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        scaler.step(opt)              # skips the step if float16 gradients overflowed
        scaler.update()
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
            vl = validate()
            log({"val_loss": vl})
            if args.select == "val_loss" and vl < best["val_loss"]:
                best.update(val_loss=vl, step=step)
                save_best()
                log({"select": {"val_loss": vl, "best_step": step}})
        if step % args.sample_every == 0 or step == args.steps:
            log({"sample": sample_val()})
            if args.select == "fid_k":
                fid = val_fid_k()
                if fid < best["fid_k"]:
                    best.update(fid_k=fid, step=step)
                    save_best()
                log({"select": {"val_fid_k": fid, "best_fid_k": best["fid_k"], "best_step": best["step"]}})
        if snapshot_every and (step % snapshot_every == 0 or step == args.steps):
            os.makedirs(os.path.join(live, "snapshots"), exist_ok=True)
            save_bundle(os.path.join(live, "snapshots", f"step{step:06d}.pt"), ema if ema is not None else model,
                        cfg, lo, hi, stats, categories)
        if ckpt and (step % args.val_every == 0 or step == args.steps):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "ema": ema.state_dict() if ema is not None else None, "step": step}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)

    final = ema if ema is not None else model
    save_bundle(os.path.join(args.out, "model_last.pt"), final, cfg, lo, hi, stats, categories)
    if ema is not None:          # the plain (non-averaged) weights too, to compare
        save_bundle(os.path.join(args.out, "model_raw.pt"), model, cfg, lo, hi, stats, categories)
    best_path = os.path.join(args.out, "model_best.pt")
    if args.select in ("fid_k", "val_loss") and os.path.exists(best_path):
        import shutil
        shutil.copy(best_path, os.path.join(args.out, "model.pt"))
    else:
        save_bundle(os.path.join(args.out, "model.pt"), final, cfg, lo, hi, stats, categories)
    log({"event": "done", "model": os.path.join(args.out, "model.pt"), "selected": args.select,
         "best_step": best["step"], "best_val_fid_k": best["fid_k"] if best["fid_k"] < float("inf") else None,
         "best_val_loss": best["val_loss"] if best["val_loss"] < float("inf") else None})


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
    add_guidance_args(p)
    add_postprocess_args(p)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=1.0, help="DDIM noise: 1 = EDGE (stochastic), 0 = deterministic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("-o", "--out", default=None)


def add_guidance_args(p):
    p.add_argument("--music-guidance", type=float, default=None,
                   help="separate guidance: weight of music given the genre; --guidance (or --genre-guidance) is then "
                        "the genre weight. Default: one weight for both (EDGE)")
    p.add_argument("--genre-guidance", type=float, default=None,
                   help="separate guidance: genre weight (overrides --guidance; needs --music-guidance, else = it)")


def add_postprocess_args(p):
    p.add_argument("--foot-lock", action="store_true",
                   help="post-process: pin feet the model marks as planted (root absorbs their motion, then leg IK)")
    p.add_argument("--root-smooth", type=float, default=0.0,
                   help="with --foot-lock: Gaussian smoothing of the root first (frames; 0 = off)")


def postprocess(motion, args):
    """Apply --foot-lock / --root-smooth to denormalized motion (T, 38) or (B, T, 38) torch -> torch."""
    if not getattr(args, "foot_lock", False):
        return motion
    from beat2body.model.footlock import foot_lock
    m = motion.cpu().numpy()
    out = np.stack([foot_lock(x, root_smooth=args.root_smooth) for x in (m if m.ndim == 3 else m[None])])
    return torch.from_numpy(out if m.ndim == 3 else out[0])


def guidance_weights(args, default):
    """(guidance, music_guidance) for Generator.generate from --guidance / --genre-guidance / --music-guidance."""
    g = args.guidance if getattr(args, "genre_guidance", None) is None else args.genre_guidance
    m = getattr(args, "music_guidance", None)
    if m is not None and g is None:
        g = default
    if m is None and getattr(args, "genre_guidance", None) is not None:
        m = default if args.guidance is None else args.guidance
    return g, m


def generate_long(gen, bundle, video_id, seconds=None, genre_name=None, guidance=None, steps=50, seed=0, eta=1.0,
                  music_guidance=None, cond=None):
    """EDGE long-form generation for a dataset video's music -> (motion (T, 38) denormalized, n windows,
    mean overlap disagreement before the cross-fade)."""
    cfg, cats = bundle["config"], bundle["categories"]
    if cond is None:
        cond = torch.from_numpy(music_input(video_id, cfg["feature_set"], bundle["audio_stats"]))
    T = len(cond) if seconds is None else min(len(cond), int(seconds * 30))
    L, half = cfg["seq_len"], cfg["seq_len"] // 2
    n = max(1, int(np.ceil((T - L) / half)) + 1) if T > L else 1
    need = half * (n + 1) if n > 1 else L
    cond = torch.cat([cond, cond[-1:].expand(max(0, need - len(cond)), -1)])[:need]
    windows = torch.stack([cond[i * half:i * half + L] for i in range(n)])
    genre = torch.full((n,), cats[genre_name or aist.GENRES[video_id[1:3]]], dtype=torch.long)
    g = torch.Generator(device=gen.device).manual_seed(seed)
    m = gen.generate(windows, genre, guidance=guidance, steps=steps, link_windows=n > 1, generator=g, eta=eta,
                     music_guidance=music_guidance)
    seam = (m[1:, :half] - m[:-1, half:]).abs().mean().item() if n > 1 else 0.0
    return (EDGEDiffusion.stitch(m) if n > 1 else m[0])[:T].cpu(), n, seam


def run_sample(args):
    gen, bundle = load_generator(args.checkpoint, args.device)
    vid = args.video_id
    genre_name = args.genre or aist.GENRES[vid[1:3]]
    gw, mw = guidance_weights(args, bundle["config"]["guidance"])
    motion, n, seam = generate_long(gen, bundle, vid, args.seconds, genre_name, gw, args.steps, args.seed,
                                    args.eta, mw)
    motion = postprocess(motion, args)
    T = len(motion)
    kp = motion[:, 4:].reshape(T, -1, 2)
    path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "generated", f"{vid}.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, motion=motion.numpy(), keypoints=kp.numpy(), root=motion[:, 2:4].numpy(),
             contact=motion[:, :2].numpy(), absolute=absolute_joints(motion).numpy())
    print(f"wrote {path}: {T} frames ({T / 30:.1f} s), {n} windows, genre {genre_name}, music {aist.music_id(vid)}"
          + (f"; overlap disagreement before cross-fade {seam:.3f} torso lengths (mean)" if n > 1 else ""))
