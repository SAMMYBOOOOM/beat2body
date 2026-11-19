"""Network modules shared by the diffusion model and the GRU baseline."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_JOINTS = 17
POSE_DIM = N_JOINTS * 2


class AudioEncoder(nn.Module):
    """Per-frame audio features (B, L, A) -> (B, L, d) with a little temporal context (2 conv layers)."""

    def __init__(self, audio_dim, d, kernel=5):
        super().__init__()
        self.inp = nn.Linear(audio_dim, d)
        self.convs = nn.ModuleList([nn.Conv1d(d, d, kernel, padding=kernel // 2) for _ in range(2)])
        self.norm = nn.LayerNorm(d)

    def forward(self, a):
        h = self.inp(a)
        x = h.transpose(1, 2)
        for conv in self.convs:
            x = x + F.gelu(conv(x))
        return self.norm(x.transpose(1, 2))


def timestep_embedding(t, dim):
    """Sinusoidal embedding of diffusion steps (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class AdaLNBlock(nn.Module):
    """Transformer block whose LayerNorms are modulated by a conditioning vector (DiT adaLN-Zero)."""

    def __init__(self, d, heads, dropout=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.mod = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, c):
        s1, b1, g1, s2, b2, g2 = self.mod(F.silu(c))[:, None].chunk(6, dim=-1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class DanceDenoiser(nn.Module):
    """Predicts the clean sequence x0 (B, L, 34) from a noisy one, the diffusion step, audio and genre.

    Genre index `n_genres` and an all-zero audio mask mean "no condition" (classifier-free guidance)."""

    def __init__(self, n_genres, audio_dim=65, d=256, layers=6, heads=4, max_len=512, dropout=0.1):
        super().__init__()
        self.n_genres, self.d = n_genres, d
        self.pose_in = nn.Linear(POSE_DIM, d)
        self.audio = AudioEncoder(audio_dim, d)
        self.null_audio = nn.Parameter(torch.zeros(d))
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        self.genre = nn.Embedding(n_genres + 1, d)
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([AdaLNBlock(d, heads, dropout) for _ in range(layers)])
        self.n_out = nn.LayerNorm(d, elementwise_affine=False)
        self.mod_out = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, POSE_DIM)
        for m in (self.mod_out, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x_t, t, audio, genre, audio_keep):
        """x_t (B, L, 34), t (B,), audio (B, L, A), genre (B,) long, audio_keep (B,) bool."""
        B, L, _ = x_t.shape
        a = self.audio(audio)
        a = torch.where(audio_keep[:, None, None], a, self.null_audio.expand(B, L, -1))
        h = self.pose_in(x_t) + a + self.pos[:L]
        c = self.time(timestep_embedding(t, self.d)) + self.genre(genre)
        for blk in self.blocks:
            h = blk(h, c)
        s, b = self.mod_out(F.silu(c))[:, None].chunk(2, dim=-1)
        return self.out(self.n_out(h) * (1 + s) + b)


class GRUBaseline(nn.Module):
    """Autoregressive baseline (README section 2): next pose from previous pose + audio + genre."""

    def __init__(self, n_genres, audio_dim=65, d=256, layers=2, genre_dim=16, dropout=0.1):
        super().__init__()
        self.audio = AudioEncoder(audio_dim, d)
        self.genre = nn.Embedding(n_genres, genre_dim)
        self.gru = nn.GRU(POSE_DIM + d + genre_dim, d, num_layers=layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(d, POSE_DIM)

    def forward(self, prev_pose, audio, genre, h0=None):
        """prev_pose (B, L, 34): pose at t-1 for each step t. Returns (pose (B, L, 34), hidden)."""
        L = prev_pose.shape[1]
        g = self.genre(genre)[:, None].expand(-1, L, -1)
        y, h = self.gru(torch.cat([prev_pose, self.audio(audio), g], -1), h0)
        return prev_pose + self.out(y), h      # predict the change from the previous pose
