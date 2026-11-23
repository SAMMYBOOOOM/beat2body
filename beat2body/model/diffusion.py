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

Options beyond EDGE (off by default, see docs/pipeline_2d.md "Energy"):
    param="v"        the network predicts v = sqrt(a) eps - sqrt(1-a) x0 and x0 is derived from it, so at low
                     noise x0 = x_t minus a small correction (a built-in skip path that keeps fine motion)
    z-score scaling  Normalizer(..., dim_weight): data standardized per feature (more signal vs the noise);
                     dim_weight rescales the recon / velocity terms so the loss equals the MinMax loss
    energy_weight    phase-invariant loss on each joint's mean speed and movement amplitude per window
                     (options: speed only, a bounded "gaussian" shape, size-normalized motion, low noise levels only)
                     (log ratio to the real window), against "swaying instead of dancing"
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
    """Affine map of [lo, hi] to [-1, 1] per feature. EDGE: lo / hi = min / max (MinMax). z-score is the
    same map with lo / hi = mean -/+ std. dim_weight (38,) weights each feature in the recon / velocity
    losses (z-score: (std / half the min-max span)^2, so those losses equal the MinMax ones)."""

    def __init__(self, lo, hi, device="cpu", dim_weight=None):
        self.lo = torch.as_tensor(lo, device=device, dtype=torch.float32)
        self.hi = torch.as_tensor(hi, device=device, dtype=torch.float32)
        self.span = (self.hi - self.lo).clamp(min=1e-6)
        self.dim_weight = None if dim_weight is None else torch.as_tensor(dim_weight, device=device,
                                                                          dtype=torch.float32)

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


def _masked_mean(x, m, dim):
    return (x * m).sum(dim) / m.sum(dim).clamp(min=1)


def smooth_time(x, sigma):
    """Gaussian blur of (B, L, ...) along time (replicate padding); sigma in frames, 0 = unchanged."""
    if sigma <= 0:
        return x
    r = int(3 * sigma)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    B, L = x.shape[:2]
    y = x.reshape(B, L, -1).permute(0, 2, 1).reshape(-1, 1, L)
    y = torch.nn.functional.conv1d(torch.nn.functional.pad(y, (r, r), mode="replicate"), k)
    return y.reshape(B, -1, L).permute(0, 2, 1).reshape(x.shape)


def size_normalized(j, valid):
    """Absolute joints with frame-to-frame body-size changes removed: root + pelvis-centred joints x (window size /
    frame size), size = mean on-screen length of the observed body bones. A dancer who grows or shrinks on screen
    (moving in depth, or a model 'pulsing' its body) no longer counts as moving joints."""
    pel = j[..., 11:13, :].mean(-2, keepdim=True)
    lb = bone_lengths(j)                                                       # (B, L, 12)
    vb = (valid[..., BODY_BONES[:, 0]] & valid[..., BODY_BONES[:, 1]]).float()
    s_t = (lb * vb).sum(-1) / vb.sum(-1).clamp(min=1)                          # (B, L) frame size
    has = (vb.sum(-1) > 0).float()
    s_w = (s_t * has).sum(1, keepdim=True) / has.sum(1, keepdim=True).clamp(min=1)
    s_t = torch.where(has > 0, s_t, s_w.expand_as(s_t)).clamp(min=1e-3)
    return pel + (j - pel) * (s_w / s_t)[..., None, None]


def energy_loss(pa, ta, valid, w, smooth=2.0, amplitude=1.0, shape="square", sigma=0.3, size_norm=False):
    """Phase-invariant: per window and joint, log ratio of the mean speed (absolute joints) and of the
    movement amplitude (std over time of the pelvis-centred joint) to the real window's. (B,) weights w.
    The speed is measured on motion blurred over `smooth` frames, so frame-to-frame jitter does not count
    as energy (without it, the cheapest way to raise speed is jitter).
    amplitude   weight of the amplitude term (0 = speed only)
    shape       "square": d^2 (each window pulled to its real energy);
                "gaussian": 2 sigma^2 (1 - exp(-d^2 / 2 sigma^2)), the same for small d but bounded for large d,
                so energy is a tendency, not a target every window must hit
    size_norm   measure on size_normalized() joints (size changes are not movement)"""
    if size_norm:
        pa, ta = size_normalized(pa, valid), size_normalized(ta, valid)
    mv = (valid[:, 1:] & valid[:, :-1]).float()                               # (B, L-1, 17)
    ps, ts = smooth_time(pa, smooth), smooth_time(ta, smooth)
    sp = _masked_mean((ps[:, 1:] - ps[:, :-1]).norm(dim=-1), mv, 1)            # (B, 17)
    st = _masked_mean((ts[:, 1:] - ts[:, :-1]).norm(dim=-1), mv, 1)
    m = valid.float()[..., None]
    pelvis_p, pelvis_t = pa[..., 11:13, :].mean(-2, keepdim=True), ta[..., 11:13, :].mean(-2, keepdim=True)
    rp, rt = pa - pelvis_p, ta - pelvis_t

    def amp(r):
        mu = _masked_mean(r, m, 1)                                            # (B, 17, 2)
        return (_masked_mean((r - mu[:, None]) ** 2, m, 1).sum(-1) + 1e-8).sqrt()   # (B, 17)

    def shaped(d2):
        return d2 if shape == "square" else 2 * sigma ** 2 * (1 - torch.exp(-d2 / (2 * sigma ** 2)))
    e = shaped((torch.log(sp + 5e-3) - torch.log(st + 5e-3)) ** 2)
    if amplitude:
        e = e + amplitude * shaped((torch.log(amp(rp) + 2e-2) - torch.log(amp(rt) + 2e-2)) ** 2)
    present = (valid.float().sum(1) >= 10).float() * w[:, None]                # joints seen in >= 10 frames
    return (e * present).sum() / present.sum().clamp(min=1e-8)


RELOC_POINTS = ((15,), (16,), (11, 12))          # left ankle, right ankle, pelvis (mid-hip)


def relocation_loss(pa, ta, valid, root_valid, w, span=15, shape="gaussian", sigma=0.3, size_norm=True, eps=0.02):
    """Phase-invariant "do the feet and the body change place": per window and point (each ankle, the pelvis), the
    log ratio of the mean displacement over `span` frames, |x[t + span] - x[t]|, to the real window's. Tapping in
    place (the foot comes back down where it left) scores ~0; stepping to a new place scores high. Like the energy
    loss, it does not ask for the real timing, only for as much relocation. shape / sigma / size_norm: as
    energy_loss. (B,) weights w."""
    if size_norm:
        pa, ta = size_normalized(pa, valid), size_normalized(ta, valid)
    L = pa.shape[1]
    if L <= span:
        return pa.new_zeros(())
    terms, present = [], []
    for pts in RELOC_POINTS:
        idx = list(pts)
        p, t = pa[:, :, idx].mean(2), ta[:, :, idx].mean(2)                     # (B, L, 2)
        ok = valid[:, :, idx].all(-1) & root_valid                                # (B, L)
        m = (ok[:, span:] & ok[:, :-span]).float()
        dp = _masked_mean((p[:, span:] - p[:, :-span]).norm(dim=-1), m, 1)        # (B,)
        dt = _masked_mean((t[:, span:] - t[:, :-span]).norm(dim=-1), m, 1)
        d2 = (torch.log(dp + eps) - torch.log(dt + eps)) ** 2
        terms.append(d2 if shape == "square" else 2 * sigma ** 2 * (1 - torch.exp(-d2 / (2 * sigma ** 2))))
        present.append((m.sum(1) >= 10).float())
    e, pr = torch.stack(terms, 1), torch.stack(present, 1) * w[:, None]          # (B, 3)
    return (e * pr).sum() / pr.sum().clamp(min=1e-8)


def motion_losses(pred_n, target_n, valid, root_valid, normalizer, weights=EDGE_WEIGHTS, bone_weight=0.0,
                  sample_weight=None, energy_weight=0.0, energy_smooth=2.0, energy_mask=None, energy_opts=None,
                  foot_contact="pred", root_accel_weight=0.0, foot_mask=None, reloc_weight=0.0, reloc_span=15,
                  reloc_mask=None, gravity_weight=0.0, gravity_g=0.02, gravity_mask=None, gravity_down_weight=0.0,
                  foot_stance=None):
    """EDGE's four losses on normalized predictions / targets (B, L, 38), plus the projected bone-length
    loss: each body bone's on-screen length vs the same bone in the real frame (relative error), so
    foreshortening in the real data is kept and no constant length is assumed. sample_weight (B,) weights
    each sequence (Min-SNR); the foot loss is not weighted (it uses the model's own contact output).
    energy_mask (B,) selects the sequences the energy loss applies to (e.g. low noise levels); energy_opts: see
    energy_loss.
    foot_contact: "pred" (EDGE) = foot loss on frames the model's own contact output marks as planted, averaged over
    all frames; "label" = on the steps t -> t+1 where the real dancer's ankle is planted (contact[t+1]; observed at
    both ends), averaged over those steps only; "label-prev" = the first version (contact[t], one step late).
    root_accel_weight: squared error of the root's acceleration vs the real one (torso lengths / frame^2), against
    root jitter. reloc_weight / reloc_span: relocation_loss (feet and body change place as much as the real
    window), on the sequences reloc_mask (B,) selects (e.g. low noise). foot_mask (B,) selects the sequences the
    label foot loss and the root-acceleration loss apply to
    (e.g. low noise levels: at high noise the x0 estimate is a blurry mean, and keeping its feet still means keeping
    everything still)."""
    w = torch.ones(pred_n.shape[0], device=pred_n.device) if sample_weight is None else sample_weight
    m = feature_mask(valid, root_valid) * w[:, None, None]
    dw = normalizer.dim_weight
    recon = masked_mse(pred_n, target_n, m) if dw is None else (((pred_n - target_n) ** 2 * m * dw).sum()
                                                              / m.sum().clamp(min=1))
    nc = slice(CONTACT.stop, MOTION_DIM)
    mv = m[:, 1:, nc] * m[:, :-1, nc] * (1 if dw is None else dw[nc])
    vel = masked_mse(pred_n[:, 1:, nc] - pred_n[:, :-1, nc], target_n[:, 1:, nc] - target_n[:, :-1, nc], mv)
    pa, ta = absolute_joints(normalizer.denorm(pred_n)), absolute_joints(normalizer.denorm(target_n))
    mj = (valid & root_valid[..., None]).float()[..., None].expand_as(pa) * w[:, None, None, None]
    fk = masked_mse(pa, ta, mj)
    feet = pa[..., ANKLE_IDX, :]                                            # (B, L, 2, 2)
    foot_v = torch.zeros_like(feet)
    foot_v[:, :-1] = feet[:, 1:] - feet[:, :-1]
    if foot_contact == "stance" and foot_stance is not None:
        # heel / toe stance labels (prep/contact.py), aligned like "label": the step t -> t+1 uses stance[t+1]. The
        # ankle is held still where the real heel is planted (flat or heel stance); toe-only steps (pivots on the
        # ball of the foot) and unknown steps are not penalized. The model's own contact predictions never gate this.
        ok = valid[..., list(ANKLE_IDX)] & root_valid[..., None]
        heel = foot_stance[..., 0] == 1                                     # (B, L, 2)
        static = torch.zeros_like(heel)
        static[:, :-1] = heel[:, 1:] & ok[:, :-1] & ok[:, 1:]
        fw = w if foot_mask is None else w * foot_mask.float()
        sf = static.float() * fw[:, None, None]
        foot = ((foot_v ** 2).sum(-1) * sf).sum() / sf.sum().clamp(min=1)
    elif foot_contact in ("label", "label-prev", "stance"):
        ok = valid[..., list(ANKLE_IDX)] & root_valid[..., None]            # (B, L, 2) ankle seen, root known
        lab = normalizer.denorm(target_n)[..., CONTACT] > 0.5               # lab[t]: still over t-1 -> t
        static = torch.zeros_like(lab)                                      # static[t]: step t -> t+1 planted
        if foot_contact == "label":
            static[:, :-1] = lab[:, 1:] & ok[:, :-1] & ok[:, 1:]
        else:     # runs 4.1 / 4.2 (commit 0825e9d): one step late, penalizes lift-off, misses touchdown
            static[:, :-1] = lab[:, :-1] & ok[:, :-1] & ok[:, 1:]
        fw = w if foot_mask is None else w * foot_mask.float()
        sf = static.float() * fw[:, None, None]
        foot = ((foot_v ** 2).sum(-1) * sf).sum() / sf.sum().clamp(min=1)
    else:
        static = pred_n[..., CONTACT] > 0.95                                # (B, L, 2), model's own contact
        foot = ((foot_v * static[..., None]) ** 2).mean()
    lp, lt = bone_lengths(pa), bone_lengths(ta)
    mb = (valid[..., BODY_BONES[:, 0]] & valid[..., BODY_BONES[:, 1]]).float() * w[:, None, None]
    bone = (((lp - lt) / lt.clamp(min=0.05)) ** 2 * mb).sum() / mb.sum().clamp(min=1)
    parts = {"recon": recon.item(), "vel": vel.item(), "fk": fk.item(), "foot": foot.item(), "bone": bone.item()}
    total = weights[0] * recon + weights[1] * vel + weights[2] * fk + weights[3] * foot + bone_weight * bone
    if root_accel_weight > 0:
        rp, rt = normalizer.denorm(pred_n)[..., ROOT], normalizer.denorm(target_n)[..., ROOT]
        ap, at = rp[:, 2:] - 2 * rp[:, 1:-1] + rp[:, :-2], rt[:, 2:] - 2 * rt[:, 1:-1] + rt[:, :-2]
        rw = w if foot_mask is None else w * foot_mask.float()
        mr = (root_valid[:, 2:] & root_valid[:, 1:-1] & root_valid[:, :-2]).float() * rw[:, None]
        racc = (((ap - at) ** 2).sum(-1) * mr).sum() / mr.sum().clamp(min=1)
        parts["root_acc"] = racc.item()
        total = total + root_accel_weight * racc
    if gravity_weight > 0 or gravity_down_weight > 0:   # physics: physics.gravity_loss on the generated motion
        from beat2body.physics import gravity_loss
        gw = w if gravity_mask is None else w * gravity_mask.float()
        g_fl, g_dn = gravity_loss(pa, valid & root_valid[..., None], gw, gravity_g)
        parts["gravity_flight"], parts["gravity_down"] = g_fl.item(), g_dn.item()
        total = total + gravity_weight * g_fl + gravity_down_weight * g_dn
    if reloc_weight > 0:
        rw = w if reloc_mask is None else w * reloc_mask.float()
        rel = relocation_loss(pa, ta, valid, root_valid, rw, reloc_span)
        parts["reloc"] = rel.item()
        total = total + reloc_weight * rel
    if energy_weight > 0:
        ew = w if energy_mask is None else w * energy_mask.float()
        energy = energy_loss(pa, ta, valid & root_valid[..., None], ew, energy_smooth, **(energy_opts or {}))
        parts["energy"] = energy.item()
        total = total + energy_weight * energy
    return total, parts


class EDGEDiffusion:
    def __init__(self, model, normalizer, steps=1000, music_drop=0.25, genre_drop=0.25, guidance=2.0,
                 weights=EDGE_WEIGHTS, bone_weight=0.0, min_snr=0.0, param="x0", energy_weight=0.0,
                 energy_smooth=2.0, energy_amplitude=1.0, energy_shape="square", energy_sigma=0.3,
                 energy_size_norm=False, energy_max_t=None, foot_contact="pred", root_accel_weight=0.0,
                 foot_max_t=None, reloc_weight=0.0, reloc_span=15, reloc_max_t=None, style_drop=0.25,
                 style_stats=None, gravity_weight=0.0, gravity_g=0.02, gravity_max_t=None, gravity_down_weight=0.0,
                 contact_weight=0.0):
        assert param in ("x0", "v"), param
        self.model, self.normalizer, self.T = model, normalizer, steps
        self.param, self.energy_weight, self.energy_smooth = param, energy_weight, energy_smooth
        self.energy_opts = dict(amplitude=energy_amplitude, shape=energy_shape, sigma=energy_sigma,
                                size_norm=energy_size_norm)
        self.energy_max_t = energy_max_t      # energy loss only for diffusion steps t < energy_max_t (None = all)
        self.min_snr = min_snr
        self.ac = cosine_alphas_cumprod(steps)
        self.music_drop, self.genre_drop, self.guidance, self.weights = music_drop, genre_drop, guidance, weights
        self.bone_weight = bone_weight
        self.foot_contact, self.root_accel_weight = foot_contact, root_accel_weight
        self.foot_max_t = foot_max_t      # label foot / root-acceleration losses only for t < foot_max_t
        self.reloc_weight, self.reloc_span, self.reloc_max_t = reloc_weight, reloc_span, reloc_max_t
        self.style_drop, self.style_stats = style_drop, style_stats   # manner condition (model.style_dim > 0)
        self.gravity_weight, self.gravity_g, self.gravity_max_t = gravity_weight, gravity_g, gravity_max_t
        self.gravity_down_weight = gravity_down_weight
        self.contact_weight = contact_weight          # BCE of the contact head (model.contact_head) on stance labels
        self.last_contact = None                      # (B, L, 4) contact probabilities of the last sample()

    def _ac(self, t, device):
        return self.ac.to(device)[t].view(-1, 1, 1)

    def q_sample(self, x0, t, noise):
        ac = self._ac(t, x0.device)
        return ac.sqrt() * x0 + (1 - ac).sqrt() * noise

    def to_x0(self, x_t, out, t):
        """Network output -> x0 (x0-prediction: unchanged; v-prediction: x0 = sqrt(a) x_t - sqrt(1-a) v)."""
        if self.param == "x0":
            return out
        ac = self._ac(t, x_t.device)
        return ac.sqrt() * x_t - (1 - ac).sqrt() * out

    def loss(self, x0, cond, genre, valid, root_valid, foot_stance=None):
        """x0 (B, L, 38) normalized motion; cond (B, L, A); genre (B,); masks (B, L, 17), (B, L)."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t = self.q_sample(x0, t, torch.randn_like(x0))
        keep_music = torch.rand(B, device=x0.device) >= self.music_drop
        g = torch.where(torch.rand(B, device=x0.device) < self.genre_drop,
                        torch.full_like(genre, self.model.n_genres), genre)
        style_kw = {}
        if getattr(self.model, "style_dim", 0):
            from beat2body.model.style import descriptors, standardize
            with torch.no_grad():
                desc = standardize(descriptors(self.normalizer.denorm(x0.float()), valid, root_valid),
                                   self.style_stats)
            style_kw = {"style": desc, "keep_style": torch.rand(B, device=x0.device) >= self.style_drop}
        pred = self.to_x0(x_t.float(), self.model(x_t, cond, t, g, keep_music, **style_kw).float(), t)
        sw = None
        if self.min_snr > 0:    # Min-SNR-gamma (Hang et al. 2023), x0 form: weight = min(SNR, gamma)
            ac = self.ac.to(x0.device)[t]
            sw = torch.clamp(ac / (1 - ac), max=self.min_snr)
            sw = sw / sw.mean().clamp(min=1e-8)
        total, parts = motion_losses(pred, x0, valid, root_valid, self.normalizer, self.weights, self.bone_weight, sw,
                             self.energy_weight, self.energy_smooth,
                             None if self.energy_max_t is None else t < self.energy_max_t, self.energy_opts,
                             self.foot_contact, self.root_accel_weight,
                             None if self.foot_max_t is None else t < self.foot_max_t,
                             self.reloc_weight, self.reloc_span,
                             None if self.reloc_max_t is None else t < self.reloc_max_t,
                             self.gravity_weight, self.gravity_g,
                             None if self.gravity_max_t is None else t < self.gravity_max_t, self.gravity_down_weight,
                             foot_stance)
        head = getattr(self.model, "contact_head", None)
        if self.contact_weight > 0 and head is not None and foot_stance is not None:
            logits = self.model.contact_logits().float()                       # (B, L, 4)
            target = foot_stance.reshape(B, foot_stance.shape[1], 4)
            m = (target >= 0).float()
            if self.foot_max_t is not None:                                     # low noise, like the foot losses
                m = m * (t < self.foot_max_t).float()[:, None, None]
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target.clamp(min=0).float(),
                                                                       reduction="none")
            c = (bce * m).sum() / m.sum().clamp(min=1)
            parts["contact_bce"] = c.item()
            total = total + self.contact_weight * c
        return total, parts

    def _model(self, x, cond, t, genre, keep, style):
        """The network with an optional style (B0, D) repeated over the batched guidance branches."""
        if style is None or not getattr(self.model, "style_dim", 0):
            return self.model(x, cond, t, genre, keep)
        reps = x.shape[0] // style.shape[0]
        return self.model(x, cond, t, genre, keep, style.repeat(reps, 1),
                          torch.ones(x.shape[0], dtype=torch.bool, device=x.device))

    def predict(self, x, cond, t, genre, guidance, music_guidance=None, style=None):
        """Guided x0. One weight (music_guidance None, EDGE): uncond + w (joint - uncond), where uncond drops
        music and genre. Separate weights: uncond + w_genre (genre_only - uncond) + w_music (joint - genre_only)
        with w_genre = guidance, w_music = music_guidance (equal weights = the one-weight formula)."""
        B = x.shape[0]
        keep = torch.ones(B, dtype=torch.bool, device=x.device)
        no_genre = torch.full_like(genre, self.model.n_genres)
        if music_guidance is None:
            if guidance == 1.0:
                return self.to_x0(x, self._model(x, cond, t, genre, keep, style), t)
            out = self._model(torch.cat([x, x]), torch.cat([cond, cond]), torch.cat([t, t]),
                              torch.cat([genre, no_genre]), torch.cat([keep, ~keep]), style)
            cond_out, unc = self.to_x0(torch.cat([x, x]), out, torch.cat([t, t])).split(B)
            return unc + (cond_out - unc) * guidance
        out = self._model(torch.cat([x, x, x]), torch.cat([cond, cond, cond]), torch.cat([t, t, t]),
                          torch.cat([genre, genre, no_genre]), torch.cat([keep, ~keep, ~keep]), style)
        joint, genre_only, unc = self.to_x0(torch.cat([x, x, x]), out, torch.cat([t, t, t])).split(B)
        return unc + guidance * (genre_only - unc) + music_guidance * (joint - genre_only)

    @torch.no_grad()
    def sample(self, cond, genre, steps=50, eta=1.0, guidance=None, known=None, known_mask=None,
               link_windows=False, generator=None, music_guidance=None, style=None):
        """DDIM. style (B, D) standardized manner descriptors or None (the model chooses).
        cond (B, L, A). link_windows: the B windows are consecutive with half overlap
        (EDGE's long-form sampling). known / known_mask (B, L, 38) / (B, L): inpainting.
        guidance: the one guidance weight, or the genre weight when music_guidance is given (see predict)."""
        guidance = self.guidance if guidance is None else guidance
        B, L, _ = cond.shape
        dev = cond.device
        x = torch.randn(B, L, MOTION_DIM, device=dev, generator=generator)
        times = torch.linspace(-1, self.T - 1, steps + 1).long().flip(0).tolist()
        # long-form: guidance ramped up over the first half of the steps (both weights alike)
        ramp = np.clip(np.linspace(0, 2, steps), None, 1) if link_windows else np.ones(steps)
        half = L // 2
        for i, (time, time_next) in enumerate(zip(times[:-1], times[1:])):
            tt = torch.full((B,), time, device=dev, dtype=torch.long)
            if known is not None:
                x = torch.where(known_mask[..., None], self.q_sample(known, tt, torch.randn_like(known)), x)
            x0 = self.predict(x, cond, tt, genre, float(guidance * ramp[i]),
                              None if music_guidance is None else float(music_guidance * ramp[i]), style)
            if known is not None:
                x0 = torch.where(known_mask[..., None], known, x0)
            ac = self._ac(tt, dev)
            eps = (x - ac.sqrt() * x0) / (1 - ac).sqrt().clamp(min=1e-8)
            if time_next < 0:              # last step: keep the prediction (EDGE: `x = x_start; continue`)
                if getattr(self.model, "contact_head", None) is not None and self.model.last_hidden is not None:
                    self.last_contact = torch.sigmoid(self.model.contact_logits(self.model.last_hidden[:B]).float())
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
