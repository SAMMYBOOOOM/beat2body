"""Gaussian diffusion over pose sequences: x0-prediction, cosine schedule, DDIM sampling with
classifier-free guidance, and inpainting of fixed frames (for stitching long sequences)."""
import math

import torch


def cosine_alphas_cumprod(T, s=0.008):
    x = torch.linspace(0, T, T + 1, dtype=torch.float64)
    f = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0).float()


def masked_mse(a, b, mask):
    """Mean squared error over (B, L, 17, 2) where mask (B, L, 17) is True."""
    m = mask.unsqueeze(-1).float()
    return ((a - b) ** 2 * m).sum() / (m.sum() * a.shape[-1]).clamp(min=1)


def pose_losses(pred, target, valid, vel_weight=1.0):
    """Position + velocity loss on (B, L, 17, 2) with the observed-joint mask."""
    pos = masked_mse(pred, target, valid)
    vel = masked_mse(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1], valid[:, 1:] & valid[:, :-1])
    return pos + vel_weight * vel, {"pos": pos.item(), "vel": vel.item()}


class GaussianDiffusion:
    def __init__(self, model, steps=1000, p_drop_genre=0.1, p_drop_audio=0.1):
        self.model, self.T = model, steps
        self.ac = cosine_alphas_cumprod(steps)
        self.p_drop_genre, self.p_drop_audio = p_drop_genre, p_drop_audio

    def _ac(self, t, device):
        return self.ac.to(device)[t].view(-1, 1, 1)

    def q_sample(self, x0, t, noise):
        ac = self._ac(t, x0.device)
        return ac.sqrt() * x0 + (1 - ac).sqrt() * noise

    def loss(self, x0, audio, genre, valid, vel_weight=1.0):
        """x0 (B, L, 34) normalized poses; valid (B, L, 17)."""
        B, L, _ = x0.shape
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t = self.q_sample(x0, t, torch.randn_like(x0))
        genre = torch.where(torch.rand(B, device=x0.device) < self.p_drop_genre,
                            torch.full_like(genre, self.model.n_genres), genre)
        keep = torch.rand(B, device=x0.device) >= self.p_drop_audio
        pred = self.model(x_t, t, audio, genre, keep)
        return pose_losses(pred.view(B, L, -1, 2), x0.view(B, L, -1, 2), valid, vel_weight)

    def _predict(self, x_t, t, audio, genre, guidance):
        B = x_t.shape[0]
        keep = torch.ones(B, dtype=torch.bool, device=x_t.device)
        cond = self.model(x_t, t, audio, genre, keep)
        if guidance == 1.0:
            return cond
        uncond = self.model(x_t, t, audio, torch.full_like(genre, self.model.n_genres), ~keep)
        return uncond + guidance * (cond - uncond)

    @torch.no_grad()
    def sample(self, audio, genre, steps=50, guidance=1.0, known=None, known_mask=None, generator=None):
        """DDIM (eta = 0). audio (B, L, A), genre (B,). Optional inpainting: frames where
        known_mask (B, L) is True are kept equal to `known` (B, L, 34)."""
        B, L, _ = audio.shape
        dev = audio.device
        x = torch.randn(B, L, self.model.out.out_features, device=dev, generator=generator)
        ts = torch.linspace(self.T - 1, 0, steps, device=dev).long()
        for i, t in enumerate(ts):
            tt = t.repeat(B)
            if known is not None:
                noisy_known = self.q_sample(known, tt, torch.randn(known.shape, device=dev, generator=generator))
                x = torch.where(known_mask[..., None], noisy_known, x)
            x0 = self._predict(x, tt, audio, genre, guidance)
            if known is not None:
                x0 = torch.where(known_mask[..., None], known, x0)
            ac = self._ac(tt, dev)
            eps = (x - ac.sqrt() * x0) / (1 - ac).sqrt().clamp(min=1e-8)
            if i + 1 < len(ts):
                ac_next = self._ac(ts[i + 1].repeat(B), dev)
                x = ac_next.sqrt() * x0 + (1 - ac_next).sqrt() * eps
            else:
                x = x0
        return x


def gru_loss(model, x0, audio, genre, valid, input_noise=0.02, vel_weight=1.0):
    """Teacher forcing with noise on the previous-pose input (reduces exposure bias)."""
    B, L, D = x0.shape
    prev = torch.cat([x0[:, :1], x0[:, :-1]], 1)
    prev = prev + input_noise * torch.randn_like(prev)
    pred, _ = model(prev, audio, genre)
    return pose_losses(pred.view(B, L, -1, 2), x0.view(B, L, -1, 2), valid, vel_weight)


@torch.no_grad()
def gru_generate(model, audio, genre, first_pose):
    """Autoregressive rollout from first_pose (B, 34)."""
    B, L, _ = audio.shape
    a = model.audio(audio)
    g = model.genre(genre)
    pose, h, out = first_pose, None, []
    for t in range(L):
        inp = torch.cat([pose, a[:, t], g], -1)[:, None]
        y, h = model.gru(inp, h)
        pose = pose + model.out(y[:, 0])
        out.append(pose)
    return torch.stack(out, 1)

