"""Models: a diffusion transformer (main) and a GRU regression baseline.

    diffusion  DanceDenoiser + GaussianDiffusion: denoises a whole (L, 17, 2) sequence, conditioned
               on per-frame audio features and the genre (EDGE-style; classifier-free guidance).
    gru        GRUBaseline: predicts the next frame from the previous pose, audio and genre
               (the README baseline; prone to collapsing to an average pose).
"""
