import torch

from beat2body.model.diffusion import EDGEDiffusion, Normalizer, energy_loss


class _Net(torch.nn.Module):
    n_genres = 10


def test_v_param_recovers_x0():
    d = EDGEDiffusion(_Net(), Normalizer(-torch.ones(38), torch.ones(38)), param="v")
    x0, eps = torch.randn(4, 150, 38), torch.randn(4, 150, 38)
    t = torch.tensor([0, 10, 500, 999])
    ac = d._ac(t, x0.device)
    x_t = d.q_sample(x0, t, eps)
    v = ac.sqrt() * eps - (1 - ac).sqrt() * x0
    assert torch.allclose(d.to_x0(x_t, v, t), x0, atol=1e-4)


def test_energy_loss_detects_damped_motion():
    t = torch.linspace(0, 6.28 * 5, 150)
    ta = torch.zeros(2, 150, 17, 2)
    ta[..., 0] = torch.sin(3 * t)[None, :, None] * torch.linspace(0.1, 0.5, 17)
    valid, w = torch.ones(2, 150, 17, dtype=torch.bool), torch.ones(2)
    assert energy_loss(ta, ta, valid, w) < 1e-8
    assert energy_loss(0.4 * ta, ta, valid, w) > 0.1          # same phase, too small
    shifted = torch.roll(ta, 7, dims=1)                         # different phase, same energy
    assert energy_loss(shifted, ta, valid, w) < 0.02


def test_smoothed_energy_ignores_jitter():
    torch.manual_seed(0)
    t = torch.linspace(0, 6.28 * 5, 150)
    ta = torch.zeros(1, 150, 17, 2)
    ta[..., 0] = torch.sin(3 * t)[None, :, None] * 0.3
    jitter = 0.05 * torch.randn_like(ta)                 # pure jitter, raw speed close to the real speed
    valid, w = torch.ones(1, 150, 17, dtype=torch.bool), torch.ones(1)
    raw = energy_loss(jitter, ta, valid, w, smooth=0)
    smooth = energy_loss(jitter, ta, valid, w, smooth=2.0)
    assert smooth > raw + 1.0                            # once smoothed, jitter no longer passes for energy
