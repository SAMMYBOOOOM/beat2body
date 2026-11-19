"""Models: EDGE's diffusion transformer for 2D motion (main) and a GRU regression baseline.

    edge.py       EDGEDecoder: EDGE's DanceDecoder (rotary, music cross-attention, FiLM) + genre
    diffusion.py  EDGEDiffusion: EDGE's losses, guidance, DDIM and long-form sampling
    adan.py       Adan optimizer (EDGE's)
    nets.py       GRUBaseline (the docs' baseline; prone to collapsing to an average pose)
"""
