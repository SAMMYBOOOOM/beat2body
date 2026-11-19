"""EDGE's diffusion (github.com/Stanford-TML/EDGE, model/diffusion.py) for 2D motion.

Same as EDGE: 1000 steps, cosine schedule, the network predicts x0, L2 loss, and the total loss
    0.636 * reconstruction + 2.964 * velocity + 0.646 * joint positions + 10.942 * foot skating
where velocity is on the non-contact features, joint positions are the absolute joints
("forward kinematics": root + pelvis-centred joints, in torso lengths), and foot skating is the
predicted ankle velocity on frames where the model's own contact output (normalized) > 0.95.
Classifier-free guidance: music dropped with p = 0.25, guidance weight 2. Sampling: DDIM, 50 steps,
eta = 1. Long pieces: consecutive windows that share half their frames, kept equal at every
denoising step, with the guidance weight ramped up over the first half of the steps; the overlaps
are cross-faded linearly.
Additions: masks for unobserved joints / root in the losses; genre condition dropped independently
(p = 0.25); inpainting of known frames in DDIM (editing).
"""
import math

import numpy as np
import torch

from beat2body.data import CONTACT, JOINTS, MOTION_DIM, N_JOINTS, ROOT

ANKLE_IDX = (15, 16)
EDGE_WEIGHTS = (0.636, 2.964, 0.646, 10.942)
# body bones of the 2D skeleton (COCO-17 edges without the face)
BODY_BONES = torch.tensor([(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
                           (11, 13), (13, 15), (12, 14), (14, 16)])


def bone_lengths(j):
    """(..., 17, 2) joints -> (..., 12) on-screen lengths of the body bones."""
    return (j[..., BODY_BONES[:, 0], :] - j[..., BODY_BONES[:, 1], :]).norm(dim=-1)


def cosine_alphas_cumprod(T, s=0.008):
    x = torch.linspace(0, T, T + 1, dtype=torch.float64)
    f = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    betas = (1 - (f[1:] / f[0]) / (f[:-1] / f[0])).clamp(0, 0.999)
    return torch.cumprod(1 - betas, 0).float()


class Normalizer:
    """MinMax to [-1, 1] per feature (EDGE's normalizer)."""

    def __init__(self, lo, hi, device="cpu"):
        self.lo = torch.as_tensor(lo, device=device, dtype=torch.float32)
        self.hi = torch.as_tensor(hi, device=device, dtype=torch.float32)
        self.span = (self.hi - self.lo).clamp(min=1e-6)

    def norm(self, x):
        return 2 * (x - self.lo) / self.span - 1

    def denorm(self, x):
        return (x + 1) / 2 * self.span + self.lo


def absolute_joints(motion):
    """(..., 38) denormalized motion -> (..., 17, 2) joints in the image frame (root + pelvis-centred)."""
    joints = motion[..., JOINTS].unflatten(-1, (N_JOINTS, 2))
    return joints + motion[..., None, ROOT]


def feature_mask(valid, root_valid):
    """(B, L, 38) loss mask: contacts always, root where both hips were seen, joints where observed."""
    B, L, _ = valid.shape
    m = torch.ones(B, L, MOTION_DIM, device=valid.device)
    m[..., ROOT] = root_valid[..., None].float()
    m[..., JOINTS] = valid.repeat_interleave(2, dim=-1).float()
    return m


def masked_mse(a, b, mask):
    return ((a - b) ** 2 * mask).sum() / mask.sum().clamp(min=1)


def motion_losses(pred_n, target_n, valid, root_valid, normalizer, weights=EDGE_WEIGHTS, bone_weight=0.0,
                  sample_weight=None):
    """EDGE's four losses on normalized predictions / targets (B, L, 38), plus the projected bone-length
    loss: each body bone's on-screen length vs the same bone in the real frame (relative error), so
    foreshortening in the real data is kept and no constant length is assumed. sample_weight (B,) weights
    each sequence (Min-SNR); the foot loss is not weighted (it uses the model's own contact output)."""
    w = torch.ones(pred_n.shape[0], device=pred_n.device) if sample_weight is None else sample_weight
    m = feature_mask(valid, root_valid) * w[:, None, None]
    recon = masked_mse(pred_n, target_n, m)
    nc = slice(CONTACT.stop, MOTION_DIM)
    mv = m[:, 1:, nc] * m[:, :-1, nc]
    vel = masked_mse(pred_n[:, 1:, nc] - pred_n[:, :-1, nc], target_n[:, 1:, nc] - target_n[:, :-1, nc], mv)
    pa, ta = absolute_joints(normalizer.denorm(pred_n)), absolute_joints(normalizer.denorm(target_n))
    mj = (valid & root_valid[..., None]).float()[..., None].expand_as(pa) * w[:, None, None, None]
    fk = masked_mse(pa, ta, mj)
    static = pred_n[..., CONTACT] > 0.95                                    # (B, L, 2), model's own contact
    feet = pa[..., ANKLE_IDX, :]                                            # (B, L, 2, 2)
    foot_v = torch.zeros_like(feet)
    foot_v[:, :-1] = feet[:, 1:] - feet[:, :-1]
    foot_v = foot_v * static[..., None]
    foot = (foot_v ** 2).mean()
    lp, lt = bone_lengths(pa), bone_lengths(ta)
    mb = (valid[..., BODY_BONES[:, 0]] & valid[..., BODY_BONES[:, 1]]).float() * w[:, None, None]
    bone = (((lp - lt) / lt.clamp(min=0.05)) ** 2 * mb).sum() / mb.sum().clamp(min=1)
    parts = {"recon": recon.item(), "vel": vel.item(), "fk": fk.item(), "foot": foot.item(), "bone": bone.item()}
    total = weights[0] * recon + weights[1] * vel + weights[2] * fk + weights[3] * foot + bone_weight * bone
    return total, parts


class EDGEDiffusion:
    def __init__(self, model, normalizer, steps=1000, music_drop=0.25, genre_drop=0.25, guidance=2.0,
                 weights=EDGE_WEIGHTS, bone_weight=0.0, min_snr=0.0):
        self.model, self.normalizer, self.T = model, normalizer, steps
        self.min_snr = min_snr
        self.ac = cosine_alphas_cumprod(steps)
        self.music_drop, self.genre_drop, self.guidance, self.weights = music_drop, genre_drop, guidance, weights
        self.bone_weight = bone_weight

    def _ac(self, t, device):
        return self.ac.to(device)[t].view(-1, 1, 1)

    def q_sample(self, x0, t, noise):
        ac = self._ac(t, x0.device)
        return ac.sqrt() * x0 + (1 - ac).sqrt() * noise

    def loss(self, x0, cond, genre, valid, root_valid):
        """x0 (B, L, 38) normalized motion; cond (B, L, A); genre (B,); masks (B, L, 17), (B, L)."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t = self.q_sample(x0, t, torch.randn_like(x0))
        keep_music = torch.rand(B, device=x0.device) >= self.music_drop
        g = torch.where(torch.rand(B, device=x0.device) < self.genre_drop,
                        torch.full_like(genre, self.model.n_genres), genre)
        pred = self.model(x_t, cond, t, g, keep_music).float()      # losses in float32 under autocast
        sw = None
        if self.min_snr > 0:    # Min-SNR-gamma (Hang et al. 2023), x0 form: weight = min(SNR, gamma)
            ac = self.ac.to(x0.device)[t]
            sw = torch.clamp(ac / (1 - ac), max=self.min_snr)
            sw = sw / sw.mean().clamp(min=1e-8)
        return motion_losses(pred, x0, valid, root_valid, self.normalizer, self.weights, self.bone_weight, sw)

    def predict(self, x, cond, t, genre, guidance):
        keep = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        cond_out = self.model(x, cond, t, genre, keep)
        if isinstance(guidance, float) and guidance == 1.0:
            return cond_out
        unc = self.model(x, cond, t, torch.full_like(genre, self.model.n_genres), ~keep)
        return unc + (cond_out - unc) * guidance

    @torch.no_grad()
    def sample(self, cond, genre, steps=50, eta=1.0, guidance=None, known=None, known_mask=None,
               link_windows=False, generator=None):
        """DDIM. cond (B, L, A). link_windows: the B windows are consecutive with half overlap
        (EDGE's long-form sampling). known / known_mask (B, L, 38) / (B, L): inpainting."""
        guidance = self.guidance if guidance is None else guidance
        B, L, _ = cond.shape
        dev = cond.device
        x = torch.randn(B, L, MOTION_DIM, device=dev, generator=generator)
        times = torch.linspace(-1, self.T - 1, steps + 1).long().flip(0).tolist()
        ramp = np.clip(np.linspace(0, guidance * 2, steps), None, guidance) if link_windows else [guidance] * steps
        half = L // 2
        for i, (time, time_next) in enumerate(zip(times[:-1], times[1:])):
            tt = torch.full((B,), time, device=dev, dtype=torch.long)
            if known is not None:
                x = torch.where(known_mask[..., None], self.q_sample(known, tt, torch.randn_like(known)), x)
            x0 = self.predict(x, cond, tt, genre, float(ramp[i]))
            if known is not None:
                x0 = torch.where(known_mask[..., None], known, x0)
            ac = self._ac(tt, dev)
            eps = (x - ac.sqrt() * x0) / (1 - ac).sqrt().clamp(min=1e-8)
            if time_next < 0:              # last step: keep the prediction (EDGE: `x = x_start; continue`)
                x = x0
                continue
            ac_next = self._ac(torch.full((B,), time_next, device=dev, dtype=torch.long), dev)
            sigma = eta * ((1 - ac / ac_next) * (1 - ac_next) / (1 - ac)).clamp(min=0).sqrt()
            c = (1 - ac_next - sigma ** 2).clamp(min=0).sqrt()
            x = x0 * ac_next.sqrt() + c * eps + sigma * torch.randn(x.shape, device=dev, generator=generator)
            if link_windows and time > 0 and B > 1:
                x[1:, :half] = x[:-1, half:]   # first half of each window = second half of the previous one
        return x

    @staticmethod
    def stitch(windows):
        """(N, L, F) consecutive half-overlapping windows -> (L * (N + 1) / 2, F), linear cross-fade."""
        N, L, _ = windows.shape
        half = L // 2
        out = [windows[0, :half]]
        fade = torch.linspace(1, 0, half, device=windows.device)[:, None]
        for i in range(1, N):
            out.append(windows[i - 1, half:] * fade + windows[i, :half] * (1 - fade))
        out.append(windows[-1, half:])
        return torch.cat(out, 0)


def gru_loss(model, x0, cond, genre, valid, root_valid, normalizer, input_noise=0.02, bone_weight=0.0):
    """Teacher forcing with noise on the previous-frame input; same losses as the diffusion model."""
    prev = torch.cat([x0[:, :1], x0[:, :-1]], 1)
    pred, _ = model(prev + input_noise * torch.randn_like(prev), cond, genre)
    return motion_losses(pred.float(), x0, valid, root_valid, normalizer, bone_weight=bone_weight)


@torch.no_grad()
def gru_generate(model, cond, genre, first):
    """Autoregressive rollout from the first frame (B, 38), normalized."""
    a = model.audio(cond)
    g = model.genre(genre)
    x, h, out = first, None, []
    for t in range(cond.shape[1]):
        y, h = model.gru(torch.cat([x, a[:, t], g], -1)[:, None], h)
        x = x + model.out(y[:, 0])
        out.append(x)
    return torch.stack(out, 1)


__all__ = ["EDGEDiffusion", "Normalizer", "absolute_joints", "motion_losses", "gru_loss", "gru_generate"]
