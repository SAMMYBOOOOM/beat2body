"""Network modules shared by the diffusion model and the GRU baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class GRUBaseline(nn.Module):
    """Autoregressive baseline (docs/pipeline_2d.md section 2): next frame from previous frame + music + genre."""

    def __init__(self, nfeats, cond_dim, n_genres, d=256, layers=2, genre_dim=16, dropout=0.1):
        super().__init__()
        self.audio = AudioEncoder(cond_dim, d)
        self.genre = nn.Embedding(n_genres, genre_dim)
        self.gru = nn.GRU(nfeats + d + genre_dim, d, num_layers=layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(d, nfeats)

    def forward(self, prev, cond, genre, h0=None):
        """prev (B, L, nfeats): frame t-1 for each step t. Returns (frames (B, L, nfeats), hidden)."""
        L = prev.shape[1]
        g = self.genre(genre)[:, None].expand(-1, L, -1)
        y, h = self.gru(torch.cat([prev, self.audio(cond), g], -1), h0)
        return prev + self.out(y), h      # predict the change from the previous frame
