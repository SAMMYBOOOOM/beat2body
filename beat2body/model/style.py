"""Manner descriptors: a few numbers per window that describe *how* someone dances (not whether the motion is
correct), used as a condition instead of losses that force every window toward its real clip's statistics.

    speed       log mean joint speed (size-normalized absolute joints, blurred over 2 frames)
    beat_power  log mean power of the pelvis-centred joints between 1 and 4 Hz (the beat band)
    relocation  log mean displacement over 0.5 s of each ankle and the pelvis (stepping to a new place vs in place)
    contact     share of planted steps (the motion's contact channels)

Training computes them from the real window and feeds them, standardized, to the model (dropped with p =
style_drop, like music and genre); at generation they are a control: none (the model chooses), the genre's typical
values, a real reference, or edited ("more travelling").
"""
import torch

from beat2body.data import CONTACT

NAMES = ("speed", "beat_power", "relocation", "contact")
DIM = len(NAMES)


def descriptors(motion, valid, root_valid, fps=30, span=15):
    """motion (B, L, 38) denormalized, valid (B, L, 17), root_valid (B, L) -> (B, 4) raw descriptors."""
    from beat2body.model.diffusion import RELOC_POINTS, _masked_mean, absolute_joints, size_normalized, smooth_time
    j = size_normalized(absolute_joints(motion), valid)
    mv = (valid[:, 1:] & valid[:, :-1]).float()
    js = smooth_time(j, 2.0)
    sp = _masked_mean((js[:, 1:] - js[:, :-1]).norm(dim=-1), mv, 1)            # (B, 17)
    speed = torch.log(_masked_mean(sp, (mv.sum(1) > 0).float(), 1) + 5e-3)
    rel = j - j[..., 11:13, :].mean(-2, keepdim=True)
    rel = rel - rel.mean(1, keepdim=True)
    L = j.shape[1]
    f = torch.fft.rfft(rel.float(), dim=1).abs() ** 2                          # (B, F, 17, 2)
    fr = torch.fft.rfftfreq(L, 1 / fps).to(j.device)
    band = ((fr >= 1) & (fr < 4)).float()[None, :, None, None]
    beat = torch.log((f * band).sum(1).sum(-1).mean(-1) / L + 1e-6)
    d = []
    for pts in RELOC_POINTS:
        p = j[:, :, list(pts)].mean(2)
        ok = valid[:, :, list(pts)].all(-1) & root_valid
        m = (ok[:, span:] & ok[:, :-span]).float()
        d.append(_masked_mean((p[:, span:] - p[:, :-span]).norm(dim=-1), m, 1))
    reloc = torch.log(torch.stack(d, 1).mean(1) + 0.02)
    contact = (motion[..., CONTACT] > 0.5).float().mean((1, 2))
    return torch.stack([speed, beat, reloc, contact], 1)


def standardize(desc, stats):
    mean = torch.as_tensor(stats["mean"], device=desc.device, dtype=desc.dtype)
    std = torch.as_tensor(stats["std"], device=desc.device, dtype=desc.dtype)
    return (desc - mean) / std


def fit_stats(descs):
    """(N, 4) raw descriptors of training windows -> {"mean", "std"} lists."""
    return {"mean": descs.mean(0).tolist(), "std": descs.std(0).clamp(min=1e-3).tolist(), "names": list(NAMES)}
