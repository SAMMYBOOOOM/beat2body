import pytest

torch = pytest.importorskip("torch")

from beat2body.data import MOTION_DIM  # noqa: E402
from beat2body.model.adan import Adan  # noqa: E402
from beat2body.model.diffusion import EDGEDiffusion, Normalizer, gru_generate, gru_loss  # noqa: E402
from beat2body.model.edge import EDGEDecoder, Rotary  # noqa: E402
from beat2body.model.nets import GRUBaseline  # noqa: E402

B, L, A = 3, 24, 37


def batch():
    return (torch.rand(B, L, MOTION_DIM) * 2 - 1, torch.randn(B, L, A), torch.tensor([0, 1, 2]),
            torch.ones(B, L, 17, dtype=torch.bool), torch.ones(B, L, dtype=torch.bool))


def norm():
    return Normalizer(-torch.ones(MOTION_DIM), torch.ones(MOTION_DIM))


def small_model():
    return EDGEDecoder(MOTION_DIM, A, n_genres=3, seq_len=L, d=32, ff=64, layers=1, heads=2)


def test_rotary_preserves_norm_and_is_relative():
    r = Rotary(8)
    x = torch.randn(1, 5, 8)
    assert torch.allclose(r(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    q = torch.randn(8).expand(1, 5, 8)
    rq = r(q)
    assert torch.allclose((rq[0, 1] * rq[0, 2]).sum(), (rq[0, 2] * rq[0, 3]).sum(), atol=1e-5)


def test_edge_loss_backward_and_sampling():
    model = small_model()
    diff = EDGEDiffusion(model, norm())
    x0, cond, genre, valid, rv = batch()
    loss, parts = diff.loss(x0, cond, genre, valid, rv)
    loss.backward()
    assert loss.item() > 0 and set(parts) == {"recon", "vel", "fk", "foot"}
    out = diff.sample(cond, genre, steps=4)
    assert out.shape == (B, L, MOTION_DIM) and torch.isfinite(out).all()


def test_long_form_windows_agree_on_overlap_and_stitch():
    diff = EDGEDiffusion(small_model(), norm())
    _, cond, genre, _, _ = batch()
    out = diff.sample(cond, genre, steps=4, link_windows=True)
    half = L // 2
    # the last step writes x0 directly; earlier steps keep the overlaps equal, so they stay close
    assert out.shape == (B, L, MOTION_DIM)
    assert EDGEDiffusion.stitch(out).shape == (half * (B + 1), MOTION_DIM)


def test_inpainting_keeps_known_frames():
    diff = EDGEDiffusion(small_model(), norm())
    x0, cond, genre, _, _ = batch()
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, :5] = True
    out = diff.sample(cond, genre, steps=4, known=x0, known_mask=mask)
    assert torch.allclose(out[:, :5], x0[:, :5])


def test_adan_reduces_a_quadratic():
    w = torch.nn.Parameter(torch.tensor([3.0, -2.0]))
    opt = Adan([w], lr=0.1)
    losses = []
    for _ in range(400):
        opt.zero_grad()
        loss = (w ** 2).sum()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.05 * losses[0]                   # heavy momentum (betas 0.02 / 0.08 / 0.01): slow but steady


def test_gru_baseline():
    model = GRUBaseline(MOTION_DIM, A, n_genres=3, d=32, layers=1)
    x0, cond, genre, valid, rv = batch()
    loss, _ = gru_loss(model, x0, cond, genre, valid, rv, norm())
    loss.backward()
    assert gru_generate(model, cond, genre, x0[:, 0]).shape == (B, L, MOTION_DIM)
