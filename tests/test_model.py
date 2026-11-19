import pytest

torch = pytest.importorskip("torch")

from beat2body.model.diffusion import GaussianDiffusion, gru_generate, gru_loss  # noqa: E402
from beat2body.model.nets import POSE_DIM, DanceDenoiser, GRUBaseline  # noqa: E402

B, L = 3, 24


def batch():
    return (torch.randn(B, L, POSE_DIM), torch.randn(B, L, 65), torch.tensor([0, 1, 2]),
            torch.ones(B, L, 17, dtype=torch.bool))


def test_diffusion_loss_backward_and_sample_shapes():
    model = DanceDenoiser(n_genres=3, d=32, layers=1, heads=2)
    diff = GaussianDiffusion(model)
    x0, audio, genre, valid = batch()
    loss, parts = diff.loss(x0, audio, genre, valid)
    loss.backward()
    assert loss.item() > 0 and set(parts) == {"pos", "vel"}
    out = diff.sample(audio, genre, steps=4, guidance=2.0)
    assert out.shape == (B, L, POSE_DIM) and torch.isfinite(out).all()


def test_inpainting_keeps_known_frames():
    model = DanceDenoiser(n_genres=3, d=32, layers=1, heads=2)
    diff = GaussianDiffusion(model)
    x0, audio, genre, _ = batch()
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, :5] = True
    out = diff.sample(audio, genre, steps=4, known=x0, known_mask=mask)
    assert torch.allclose(out[:, :5], x0[:, :5])


def test_gru_baseline():
    model = GRUBaseline(n_genres=3, d=32, layers=1)
    x0, audio, genre, valid = batch()
    loss, _ = gru_loss(model, x0, audio, genre, valid)
    loss.backward()
    assert gru_generate(model, audio, genre, x0[:, 0]).shape == (B, L, POSE_DIM)
