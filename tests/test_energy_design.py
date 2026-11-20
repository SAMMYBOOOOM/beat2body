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


def _dancer(n=2, amp=0.3):
    """A body with real proportions (COCO-17, torso ~1) whose limbs swing, plus a moving root."""
    base = torch.tensor([[0, -1.5], [-.1, -1.6], [.1, -1.6], [-.2, -1.55], [.2, -1.55], [-.4, -1], [.4, -1],
                         [-.6, -.5], [.6, -.5], [-.7, 0], [.7, 0], [-.2, 0], [.2, 0], [-.25, .8], [.25, .8],
                         [-.25, 1.6], [.25, 1.6]])
    t = torch.linspace(0, 6.28 * 5, 150)
    j = base.expand(n, 150, 17, 2).clone()
    j[..., 7:11, 1] += amp * torch.sin(3 * t)[None, :, None]                 # arms
    j[..., 0] += 0.2 * torch.sin(t)[None, :, None]                            # root sway
    return j


def test_gaussian_energy_matches_square_for_small_and_is_bounded_for_large():
    ta, valid, w = _dancer(), torch.ones(2, 150, 17, dtype=torch.bool), torch.ones(2)
    small = _dancer(amp=0.315)
    sq, ga = energy_loss(small, ta, valid, w), energy_loss(small, ta, valid, w, shape="gaussian", sigma=0.3)
    assert abs(sq - ga) / sq < 0.05
    big = _dancer(amp=3.0)
    assert energy_loss(big, ta, valid, w, shape="gaussian", sigma=0.3) < 2 * 2 * 0.3 ** 2 + 1e-6   # <= 2 terms x bound
    assert energy_loss(big, ta, valid, w) > energy_loss(big, ta, valid, w, shape="gaussian", sigma=0.3)


def test_size_norm_ignores_body_pulsing():
    ta, valid, w = _dancer(), torch.ones(2, 150, 17, dtype=torch.bool), torch.ones(2)
    t = torch.linspace(0, 6.28 * 5, 150)
    pel = ta[..., 11:13, :].mean(-2, keepdim=True)
    pulse = pel + (ta - pel) * (1 + 0.25 * torch.sin(6 * t))[None, :, None, None]   # body grows / shrinks on beat
    plain = energy_loss(pulse, ta, valid, w, amplitude=0)
    normed = energy_loss(pulse, ta, valid, w, amplitude=0, size_norm=True)
    assert plain > 0.005 and normed < 0.01 * plain


def test_energy_only_at_low_noise_levels():
    d = EDGEDiffusion(_Net(), Normalizer(-3 * torch.ones(38), 3 * torch.ones(38)), energy_weight=1.0,
                      energy_max_t=500)

    class Out(torch.nn.Module):          # predicts a damped copy of x0 whatever the input
        n_genres = 10

        def forward(self, x, cond, t, genre, keep):
            return self.x0 * 0.3
    net = Out()
    d.model = net
    from beat2body.data import MOTION_DIM
    from beat2body.model.diffusion import motion_losses
    j = _dancer()
    x0 = torch.zeros(2, 150, MOTION_DIM)
    x0[..., 2:4] = j[..., 11:13, :].mean(-2)
    x0[..., 4:] = (j - j[..., 11:13, :].mean(-2, keepdim=True)).flatten(-2)
    x0n = d.normalizer.norm(x0)
    net.x0 = x0n
    valid, rv = torch.ones(2, 150, 17, dtype=torch.bool), torch.ones(2, 150, dtype=torch.bool)
    on = motion_losses(0.3 * x0n, x0n, valid, rv, d.normalizer, energy_weight=1.0, energy_mask=torch.tensor([1, 1]))
    off = motion_losses(0.3 * x0n, x0n, valid, rv, d.normalizer, energy_weight=1.0, energy_mask=torch.tensor([0, 0]))
    assert on[1]["energy"] > 0.01 and off[1]["energy"] == 0.0
