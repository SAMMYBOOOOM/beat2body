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

Optional (beat_attn=True): beat-relative cross-attention. Each frame's position in beats (continuous
beat count, unwrapped from the beat-phase sin / cos that are the last two music features) gives every
cross-attention score from motion frame i to music frame j a learned per-layer, per-head bias
    rel[h, bin(beat_pos[j] - beat_pos[i])] + key_phase[h, bin(phase[j])]
with bins of 1 / beat_bins of a beat (offsets clamped to +/- beat_range beats). A head can learn to look
ahead (anticipation), behind (delay) or half a beat away (offbeat accents), and to prefer music frames on
or off the beat. Both tables start at 0 (= the plain model). The bias is 0 where the music is dropped
(classifier-free guidance), outside the beat grid, and for the time / genre tokens.
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

    def forward(self, x, memory, t, cross_bias=None):
        """cross_bias: None or (B * heads, L, S) float, added to the cross-attention scores."""
        h = self.n1(x)
        qk = self.rotary(h)
        x = x + film(self.d1(self.self_attn(qk, qk, h, need_weights=False)[0]), self.film1(t))
        h = self.n2(x)
        q, k = self.rotary(h), self.rotary(memory)
        if cross_bias is not None:
            cross_bias = cross_bias.to(q.dtype)
        x = x + film(self.d2(self.cross_attn(q, k, memory, attn_mask=cross_bias,
                                             need_weights=False)[0]), self.film2(t))
        return x + film(self.d3(self.ff(self.n3(x))), self.film3(t))


def beat_position(cond, eps=0.25):
    """Continuous beat count (B, L) and in-grid mask (B, L) from the beat-phase sin / cos in the last two
    columns of cond. The count is the phase unwrapped over time (+1 at every wrap); beat periods are
    many frames long, so a wrap is a drop of more than half a beat between frames."""
    s, c = cond[..., -2].float(), cond[..., -1].float()
    ok = s * s + c * c > eps
    ph = torch.remainder(torch.atan2(s, c) / (2 * math.pi), 1.0)
    d = ph[:, 1:] - ph[:, :-1]
    wraps = (d < -0.5).float() - (d > 0.5).float()
    pos = ph + torch.cat([torch.zeros_like(ph[:, :1]), wraps.cumsum(1)], 1)
    return pos, ph, ok


class BeatBias(nn.Module):
    """Learned beat-relative cross-attention bias for every decoder layer (see the module docstring)."""

    def __init__(self, layers, heads, bins=8, beat_range=2.0):
        super().__init__()
        self.heads, self.bins, self.R = heads, bins, int(round(beat_range * bins))
        self.rel = nn.Parameter(torch.zeros(layers, heads, 2 * self.R + 1))
        self.key_phase = nn.Parameter(torch.zeros(layers, heads, bins))

    def forward(self, cond, keep_music, extra_tokens):
        """-> list over layers of (B * heads, L, L + extra_tokens) biases, built once for all layers and heads."""
        pos, ph, ok = beat_position(cond)
        B, L = pos.shape
        n_layers, H = self.rel.shape[:2]
        off = pos[:, None, :] - pos[:, :, None]                                   # (B, Lq, Lk): key - query
        rel_bin = (off * self.bins).round().clamp(-self.R, self.R).long() + self.R
        ph_bin = (ph * self.bins).floor().long().clamp(0, self.bins - 1)          # (B, Lk)
        # one-hot @ table: a dense matmul in both directions (an index gather's backward would accumulate
        # millions of gradients into a few dozen bins)
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else self.rel.dtype
        rel = nn.functional.one_hot(rel_bin, 2 * self.R + 1).to(dt) @ self.rel.flatten(0, 1).T.to(dt)
        kp = nn.functional.one_hot(ph_bin, self.bins).to(dt) @ self.key_phase.flatten(0, 1).T.to(dt)
        bias = rel + kp[:, None]                                                  # (B, Lq, Lk, layers * H)
        mask = ok[:, :, None] & ok[:, None, :] & keep_music[:, None, None]
        bias = bias * mask[..., None].to(bias.dtype)
        bias = nn.functional.pad(bias.permute(3, 0, 1, 2), (0, extra_tokens))     # (layers * H, B, Lq, S)
        bias = bias.view(n_layers, H, B, L, L + extra_tokens).transpose(1, 2)
        return [b.reshape(B * H, L, L + extra_tokens) for b in bias]


class EDGEDecoder(nn.Module):
    """Predicts the clean motion x0 (B, L, nfeats) from noisy motion, music, diffusion step and genre."""

    def __init__(self, nfeats, cond_dim, n_genres, seq_len=150, d=512, ff=1024, layers=8, heads=8, dropout=0.1,
                 beat_attn=False, beat_bins=8, beat_range=2.0, style_dim=0, contact_head=False):
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
        self.beat_bias = BeatBias(layers, heads, beat_bins, beat_range) if beat_attn else None
        self.contact_head = nn.Linear(d, 4) if contact_head else None   # heel / toe planted per foot (logits)
        self.last_hidden = None
        self.style_dim = style_dim                    # manner descriptors (model/style.py) as a condition
        if style_dim:
            self.style_mlp = nn.Sequential(nn.Linear(style_dim, d), nn.SiLU(), nn.Linear(d, d))
            self.null_style = nn.Parameter(torch.zeros(1, d))

    def forward(self, x, cond, t, genre, keep_music, style=None, keep_style=None):
        """x (B, L, nfeats), cond (B, L, cond_dim), t (B,), genre (B,) (n_genres = none), keep_music (B,) bool.
        style (B, style_dim) standardized manner descriptors and keep_style (B,) bool, or None (no style)."""
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
        if self.style_dim:                            # added to the genre embedding: FiLM and the genre token
            null = self.null_style.to(g.dtype).expand(B, -1)
            if style is None:
                s = null
            else:
                s = self.style_mlp(style.to(g.dtype))
                s = torch.where(keep_style[:, None], s, null) if keep_style is not None else s
            g = g + s
        t_vec = self.to_time_cond(t_hidden) + cond_hidden + g
        t_tokens = self.to_time_tokens(t_hidden).view(B, 2, self.d)
        memory = self.norm_cond(torch.cat([c, t_tokens, g[:, None]], dim=-2))
        biases = self.beat_bias(cond, keep_music, 3) if self.beat_bias is not None else [None] * len(self.layers)
        for layer, bias in zip(self.layers, biases):
            x = layer(x, memory, t_vec, bias)
        if self.contact_head is not None:
            self.last_hidden = x                      # for contact_logits(): the diffusion loss and sampling read it
        return self.final_layer(x)

    def contact_logits(self, hidden=None):
        """(B, L, 4) logits [left heel, left toe, right heel, right toe] planted, from the last forward pass."""
        return self.contact_head(self.last_hidden if hidden is None else hidden)
