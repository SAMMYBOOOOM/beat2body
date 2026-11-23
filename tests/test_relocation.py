import numpy as np
import torch

from beat2body.model.diffusion import EDGEDiffusion, Normalizer, motion_losses, relocation_loss

BASE = torch.tensor([[0, -1.5], [-.1, -1.6], [.1, -1.6], [-.2, -1.55], [.2, -1.55], [-.4, -1], [.4, -1],
                     [-.6, -.5], [.6, -.5], [-.7, 0], [.7, 0], [-.2, 0], [.2, 0], [-.25, .8], [.25, .8],
                     [-.25, 1.6], [.25, 1.6]])


def _walker(T=120, step=0.3, period=30, shift=0, tap=False):
    """A body stepping sideways: every `period` frames each foot lifts and lands `step` further (tap: lands where it
    left), the pelvis follows the feet."""
    j = BASE.expand(1, T, 17, 2).clone()
    t = np.arange(T) + shift
    for k, off in ((15, 0), (16, period // 2)):
        phase = ((t + off) % period) / period
        n_steps = (t + off) // period
        x = (0.0 if tap else step) * (n_steps + np.clip((phase - 0.5) * 4, 0, 1))
        lift = 0.15 * np.sin(np.pi * np.clip((phase - 0.5) * 4, 0, 1))
        j[0, :, k, 0] += torch.tensor(x, dtype=torch.float32)
        j[0, :, k, 1] -= torch.tensor(lift, dtype=torch.float32)
    feet_x = j[0, :, [15, 16], 0].mean(1) - BASE[[15, 16], 0].mean()
    j[0, :, :15, 0] += feet_x[:, None]                                         # the body follows the feet
    return j


def _loss(p, t):
    valid = torch.ones(1, p.shape[1], 17, dtype=torch.bool)
    return relocation_loss(p, t, valid, torch.ones(1, p.shape[1], dtype=torch.bool), torch.ones(1)).item()


def test_identical_motion_has_no_relocation_loss():
    w = _walker()
    assert _loss(w, w) < 1e-10


def test_tapping_in_place_is_penalized_against_real_stepping():
    assert _loss(_walker(tap=True), _walker()) > 0.05


def test_relocation_loss_ignores_timing():
    real, shifted = _walker(), _walker(shift=7)                                # the same steps, a quarter period late
    assert _loss(shifted, real) < 0.01 * _loss(_walker(tap=True), real)


def test_motion_losses_report_relocation_and_diffusion_keeps_options():
    norm = Normalizer(-3 * torch.ones(38), 3 * torch.ones(38))
    j = _walker()
    m = torch.zeros(1, j.shape[1], 38)
    pel = j[..., 11:13, :].mean(-2)
    m[..., 2:4] = pel
    m[..., 4:] = (j - pel[..., None, :]).reshape(1, j.shape[1], -1)
    valid = torch.ones(1, j.shape[1], 17, dtype=torch.bool)
    _, parts = motion_losses(norm.norm(m), norm.norm(m), valid, torch.ones(1, j.shape[1], dtype=torch.bool), norm,
                             reloc_weight=1.0)
    assert parts["reloc"] < 1e-10

    class _Net(torch.nn.Module):
        n_genres = 10
    d = EDGEDiffusion(_Net(), norm, reloc_weight=0.5, reloc_span=10, reloc_max_t=500)
    assert (d.reloc_weight, d.reloc_span, d.reloc_max_t) == (0.5, 10, 500)
