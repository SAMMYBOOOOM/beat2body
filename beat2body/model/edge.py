"""EDGE's DanceDecoder (Tseng et al., CVPR 2023; github.com/Stanford-TML/EDGE, model/model.py),
re-implemented for 2D motion, plus a genre condition (EDGE conditions on music only).

Same as EDGE: input projection; music features -> Linear -> 2 rotary transformer encoder layers;
learned null music tokens / null pooled vector for classifier-free guidance; diffusion step ->
sinusoidal -> Linear(D, 4D) -> Mish, giving a FiLM vector (plus the pooled music) and 2 extra
cross-attention tokens; N decoder layers (pre-norm) of rotary self-attention, cross-attention to
[music tokens, time tokens], feed-forward, each followed by FiLM; no causal mask; linear output.
Rotary embeddings are applied to the layer inputs before the attention projections, as in EDGE.
Added: a genre embedding (with a null genre) added to the FiLM vector and appended as a
cross-attention token.
"""
import math

import torch
import torch.nn as nn


class Rotary(nn.Module):
    """Rotary position embedding ("lang" frequencies, theta 10000, interleaved pairs)."""

    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.register_buffer("freqs", 1.0 / theta ** (torch.arange(0, dim, 2).float() / dim), persistent=False)

    def forward(self, x):                       # x (B, N, D)
        n = x.shape[-2]
        ang = torch.arange(n, device=x.device, dtype=self.freqs.dtype)[:, None] * self.freqs[None]
        ang = ang.repeat_interleave(2, dim=-1)                       # (N, D)
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x * ang.cos().to(x.dtype) + rot * ang.sin().to(x.dtype)


def sinusoidal(t, dim):
    half = dim // 2
    f = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


class DenseFiLM(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Mish(), nn.Linear(d, 2 * d))

    def forward(self, c):                        # (B, D) -> scale, shift (B, 1, D)
        return self.net(c)[:, None].chunk(2, dim=-1)


def film(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class EncoderLayer(nn.Module):
    """nn.TransformerEncoderLayer (post-norm) with rotary applied to the attention input."""

    def __init__(self, d, heads, ff, dropout, rotary):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.d1, self.d2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.rotary = rotary

    def forward(self, x):
        qk = self.rotary(x)
        x = self.n1(x + self.d1(self.attn(qk, qk, x, need_weights=False)[0]))
        return self.n2(x + self.d2(self.ff(x)))


class FiLMDecoderLayer(nn.Module):
    """EDGE's FiLMTransformerDecoderLayer with norm_first=True."""

    def __init__(self, d, heads, ff, dropout, rotary):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.d1, self.d2, self.d3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)
        self.film1, self.film2, self.film3 = DenseFiLM(d), DenseFiLM(d), DenseFiLM(d)
        self.rotary = rotary

    def forward(self, x, memory, t):
        h = self.n1(x)
        qk = self.rotary(h)
        x = x + film(self.d1(self.self_attn(qk, qk, h, need_weights=False)[0]), self.film1(t))
        h = self.n2(x)
        x = x + film(self.d2(self.cross_attn(self.rotary(h), self.rotary(memory), memory,
                                             need_weights=False)[0]), self.film2(t))
        return x + film(self.d3(self.ff(self.n3(x))), self.film3(t))


class EDGEDecoder(nn.Module):
    """Predicts the clean motion x0 (B, L, nfeats) from noisy motion, music, diffusion step and genre."""

    def __init__(self, nfeats, cond_dim, n_genres, seq_len=150, d=512, ff=1024, layers=8, heads=8, dropout=0.1):
        super().__init__()
        self.n_genres, self.seq_len, self.d = n_genres, seq_len, d
        self.rotary = Rotary(d)
        self.input_projection = nn.Linear(nfeats, d)
        self.time_mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.Mish())
        self.to_time_cond = nn.Linear(4 * d, d)
        self.to_time_tokens = nn.Linear(4 * d, 2 * d)
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, d))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, d))
        self.norm_cond = nn.LayerNorm(d)
        self.cond_projection = nn.Linear(cond_dim, d)
        self.cond_encoder = nn.ModuleList([EncoderLayer(d, heads, ff, dropout, self.rotary) for _ in range(2)])
        self.non_attn_cond_projection = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.genre = nn.Embedding(n_genres + 1, d)                  # index n_genres = no genre
        self.layers = nn.ModuleList([FiLMDecoderLayer(d, heads, ff, dropout, self.rotary) for _ in range(layers)])
        self.final_layer = nn.Linear(d, nfeats)

    def forward(self, x, cond, t, genre, keep_music):
        """x (B, L, nfeats), cond (B, L, cond_dim), t (B,), genre (B,) (n_genres = none), keep_music (B,) bool."""
        B, L, _ = x.shape
        x = self.input_projection(x)
        c = self.cond_projection(cond)
        for layer in self.cond_encoder:
            c = layer(c)
        c = torch.where(keep_music[:, None, None], c, self.null_cond_embed[:, :L].to(c.dtype))
        cond_hidden = self.non_attn_cond_projection(c.mean(dim=-2))
        cond_hidden = torch.where(keep_music[:, None], cond_hidden, self.null_cond_hidden.to(c.dtype))
        t_hidden = self.time_mlp(sinusoidal(t, self.d))
        g = self.genre(genre)
        t_vec = self.to_time_cond(t_hidden) + cond_hidden + g
        t_tokens = self.to_time_tokens(t_hidden).view(B, 2, self.d)
        memory = self.norm_cond(torch.cat([c, t_tokens, g[:, None]], dim=-2))
        for layer in self.layers:
            x = layer(x, memory, t_vec)
        return self.final_layer(x)
