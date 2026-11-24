import torch
from test_relocation import _walker

from beat2body.model.diffusion import EDGEDiffusion, Normalizer
from beat2body.model.edge import EDGEDecoder
from beat2body.model.style import DIM, descriptors, fit_stats, standardize


def _motion(j):
    """Absolute joints (1, T, 17, 2) -> motion (1, T, 38) with zero contacts."""
    T = j.shape[1]
    m = torch.zeros(1, T, 38)
    pel = j[..., 11:13, :].mean(-2)
    m[..., 2:4] = pel
    m[..., 4:] = (j - pel[..., None, :]).reshape(1, T, -1)
    return m


def _desc(j):
    T = j.shape[1]
    return descriptors(_motion(j), torch.ones(1, T, 17, dtype=torch.bool), torch.ones(1, T, dtype=torch.bool))[0]


def test_descriptors_separate_still_tapping_and_stepping():
    still = _walker(step=0.0) * 0 + _walker(step=0.0)[:, :1]                   # frozen first frame
    tap, step = _walker(tap=True), _walker()
    d_still, d_tap, d_step = _desc(still), _desc(tap), _desc(step)
    assert d_still[0] < d_tap[0] and d_still[0] < d_step[0]                    # speed
    assert d_step[2] > d_tap[2] + 0.5                                          # relocation: stepping > tapping
    assert d_still[2] < d_tap[2]


def test_standardize_with_fitted_stats_is_zero_mean():
    d = torch.randn(64, DIM) * 3 + 1
    z = standardize(d, fit_stats(d))
    assert z.mean(0).abs().max() < 1e-5 and (z.std(0) - 1).abs().max() < 1e-4


def _net():
    torch.manual_seed(0)
    return EDGEDecoder(38, 37, 10, seq_len=30, d=32, ff=64, layers=1, heads=2, style_dim=DIM).eval()


def test_model_uses_the_style_and_null_style_when_missing_or_dropped():
    net = _net()
    with torch.no_grad():
        torch.nn.init.normal_(net.style_mlp[2].weight)                         # make the style matter at init
    x, c = torch.randn(2, 30, 38), torch.randn(2, 30, 37)
    t, g, keep = torch.tensor([10, 10]), torch.tensor([1, 1]), torch.ones(2, dtype=torch.bool)
    s = torch.randn(2, DIM)
    with torch.no_grad():
        a = net(x, c, t, g, keep, s, keep)
        b = net(x, c, t, g, keep, s + 1, keep)
        none = net(x, c, t, g, keep)
        dropped = net(x, c, t, g, keep, s, ~keep)
    assert (a - b).abs().max() > 1e-4                                          # the style changes the output
    assert torch.allclose(none, dropped, atol=1e-6)                            # no style == dropped style


def test_diffusion_loss_runs_with_style_conditioning():
    net = _net().train()
    stats = {"mean": [0.0] * DIM, "std": [1.0] * DIM}
    d = EDGEDiffusion(net, Normalizer(-3 * torch.ones(38), 3 * torch.ones(38)), style_stats=stats)
    x0 = d.normalizer.norm(_motion(_walker(T=30)))
    loss, parts = d.loss(x0, torch.randn(1, 30, 37), torch.tensor([1]), torch.ones(1, 30, 17, dtype=torch.bool),
                         torch.ones(1, 30, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss) and net.style_mlp[0].weight.grad is not None
