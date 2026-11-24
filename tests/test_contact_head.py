import torch
from test_feet import _losses, _step_motion

from beat2body.model.diffusion import EDGEDiffusion, Normalizer
from beat2body.model.edge import EDGEDecoder


def _stance(m, heel_left=1, toe_left=1):
    """(1, L, 2, 2) stance from the motion's ankle labels: left foot planted where its label is, with the given
    heel / toe values; right foot unknown."""
    L = m.shape[0]
    st = torch.full((1, L, 2, 2), -1, dtype=torch.int8)
    planted = m[:, 0] > 0.5
    st[0, :, 0, 0] = torch.where(planted, torch.tensor(heel_left, dtype=torch.int8), torch.tensor(0, dtype=torch.int8))
    st[0, :, 0, 1] = torch.where(planted, torch.tensor(toe_left, dtype=torch.int8), torch.tensor(0, dtype=torch.int8))
    return st


def _slide_while_planted(m):
    pred = m.clone()
    rel = pred[:, 4:].reshape(len(m), 17, 2)
    rel[12:20, 15, 0] += torch.linspace(0, 0.2, 8)                           # the left ankle slides mid-stance
    pred[:, 4:] = rel.reshape(len(m), -1)
    return pred


def test_stance_gating_holds_the_heel_and_frees_toe_pivots_and_unknowns():
    m = _step_motion()
    pred = _slide_while_planted(m)
    heel = _losses(pred, m, foot_contact="stance", foot_stance=_stance(m, 1, 1))["foot"]
    toe_only = _losses(pred, m, foot_contact="stance", foot_stance=_stance(m, 0, 1))["foot"]
    unknown = _losses(pred, m, foot_contact="stance", foot_stance=_stance(m, -1, -1))["foot"]
    assert heel > 1e-4 and toe_only == 0 and unknown == 0
    assert _losses(m, m, foot_contact="stance", foot_stance=_stance(m))["foot"] < 1e-12   # real motion: no penalty


def _net():
    torch.manual_seed(0)
    return EDGEDecoder(38, 37, 10, seq_len=60, d=32, ff=64, layers=1, heads=2, contact_head=True)


def test_contact_head_is_trained_by_the_diffusion_loss():
    net = _net().train()
    d = EDGEDiffusion(net, Normalizer(-3 * torch.ones(38), 3 * torch.ones(38)), contact_weight=0.1)
    m = _step_motion()
    st = _stance(m)
    loss, parts = d.loss(d.normalizer.norm(m)[None], torch.randn(1, 60, 37), torch.tensor([1]),
                         torch.ones(1, 60, 17, dtype=torch.bool), torch.ones(1, 60, dtype=torch.bool), st)
    loss.backward()
    assert "contact_bce" in parts and net.contact_head.weight.grad.abs().sum() > 0


def test_sampling_keeps_the_contact_probabilities():
    net = _net().eval()
    d = EDGEDiffusion(net, Normalizer(-3 * torch.ones(38), 3 * torch.ones(38)))
    d.sample(torch.randn(2, 60, 37), torch.tensor([1, 2]), steps=3)
    assert d.last_contact is not None and d.last_contact.shape == (2, 60, 4)
    assert (d.last_contact >= 0).all() and (d.last_contact <= 1).all()


def test_binary_stats_perfect_and_calibrated():
    from beat2body.evaluate import binary_stats
    y = torch.tensor([1, 1, 0, 0, 1, 0])
    s = binary_stats(y.float(), y)
    assert s["accuracy"] == 1 and s["balanced_accuracy"] == 1 and s["brier"] == 0 and s["ece"] == 0
    s = binary_stats(torch.full((6,), 0.5), y)                       # always 0.5: calibrated on a 50 % share
    assert abs(s["ece"]) < 1e-9 and s["balanced_accuracy"] == 0.5


def test_contact_head_stats_on_real_and_generated_windows():
    from types import SimpleNamespace

    import numpy as np

    from beat2body.evaluate import contact_head_stats
    net = _net().eval()
    d = EDGEDiffusion(net, Normalizer(-3 * torch.ones(38), 3 * torch.ones(38)))
    gen = SimpleNamespace(diffusion=d, device="cpu", normalizer=d.normalizer, model=net)
    m = _step_motion()[None].repeat(2, 1, 1)
    st = _stance(m[0]).repeat(2, 1, 1, 1)
    probs = torch.rand(2, 60, 4)
    out = contact_head_stats(gen, m, torch.randn(2, 60, 37), torch.tensor([1, 2]), st,
                             np.random.default_rng(0).standard_normal((2, 60, 17, 2)), probs, t_eval=10, batch=1)
    assert set(out["real"]) == {"t", "all", "heel", "toe"}
    assert out["real"]["heel"]["n"] == int((st[..., 0] >= 0).sum())       # unknown frames are masked
    assert 0 <= out["generated"]["accuracy"] <= 1 and out["generated"]["n"] == 2 * 60 * 2
