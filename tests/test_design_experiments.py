import numpy as np
import pytest

torch = pytest.importorskip("torch")

from beat2body.data import MOTION_DIM, beat_phase  # noqa: E402
from beat2body.model.diffusion import EDGEDiffusion, Normalizer  # noqa: E402
from beat2body.model.edge import EDGEDecoder, beat_position  # noqa: E402
from beat2body.probe import accent_phase, beat_phases, wrap  # noqa: E402

B, L, A = 3, 40, 37


def cond_with_beats(period=12.5, offset=3.0):
    c = torch.randn(B, L, A)
    beats = np.arange(offset, L + 2 * period, period)
    c[..., -2:] = torch.from_numpy(beat_phase(beats, L))
    return c, beats


def model(beat_attn):
    torch.manual_seed(0)
    return EDGEDecoder(MOTION_DIM, A, n_genres=3, seq_len=L, d=32, ff=64, layers=2, heads=2,
                       beat_attn=beat_attn).eval()


def test_beat_position_unwraps_phase():
    c, beats = cond_with_beats()
    pos, ph, ok = beat_position(c)
    t = np.arange(L)
    inside = t >= beats[0]
    assert (ok[0].numpy() == inside).all()
    expected = (t - beats[0]) / 12.5
    got = pos[0].numpy()[inside]
    assert np.allclose(got - got[0], expected[inside] - expected[inside][0], atol=1e-4)


def test_equal_weights_reproduce_single_guidance():
    m = model(False)
    diff = EDGEDiffusion(m, Normalizer(-torch.ones(MOTION_DIM), torch.ones(MOTION_DIM)))
    x = torch.randn(B, L, MOTION_DIM)
    c, _ = cond_with_beats()
    t = torch.full((B,), 500)
    g = torch.tensor([0, 1, 2])
    one = diff.predict(x, c, t, g, 2.5)
    sep = diff.predict(x, c, t, g, 2.5, music_guidance=2.5)
    assert torch.allclose(one, sep, atol=1e-5)
    # music weight 0: only the genre-conditioned prediction counts -> independent of the music
    a = diff.predict(x, c, t, g, 1.0, music_guidance=0.0)
    b = diff.predict(x, torch.randn_like(c), t, g, 1.0, music_guidance=0.0)
    assert torch.allclose(a, b, atol=1e-5)


def test_beat_bias_starts_as_plain_model_and_learns():
    plain, beat = model(False), model(True)
    beat.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(B, L, MOTION_DIM)
    c, _ = cond_with_beats()
    t, g, keep = torch.full((B,), 100), torch.tensor([0, 1, 2]), torch.tensor([True, False, True])
    assert torch.allclose(plain(x, c, t, g, keep), beat(x, c, t, g, keep), atol=1e-5)
    with torch.no_grad():
        beat.beat_bias.rel.normal_()
    out = beat(x, c, t, g, keep)
    assert not torch.allclose(out[0], plain(x, c, t, g, keep)[0], atol=1e-4)
    assert torch.allclose(out[1], plain(x, c, t, g, keep)[1], atol=1e-5)       # music dropped: no beat bias
    beat.train()
    diff = EDGEDiffusion(beat, Normalizer(-torch.ones(MOTION_DIM), torch.ones(MOTION_DIM)))
    loss, _ = diff.loss(torch.rand(B, L, MOTION_DIM) * 2 - 1, c, g, torch.ones(B, L, 17, dtype=torch.bool),
                        torch.ones(B, L, dtype=torch.bool))
    loss.backward()
    assert beat.beat_bias.rel.grad.abs().sum() > 0 and beat.beat_bias.key_phase.grad.abs().sum() > 0


def test_beat_bias_under_autocast():
    m = model(True)
    c, _ = cond_with_beats()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(torch.randn(B, L, MOTION_DIM), c, torch.full((B,), 10), torch.tensor([0, 1, 2]),
                torch.ones(B, dtype=torch.bool))
    assert torch.isfinite(out.float()).all()


def test_accent_phase_measures_shift():
    beats = np.arange(0.0, 300, 15.0)
    t = np.arange(300)

    def dancer(shift_frames):          # speed minima a fixed time after each beat
        speed = 1.2 - np.cos(2 * np.pi * (t - 3 - shift_frames) / 15)
        pos = np.cumsum(speed)
        return np.repeat(np.stack([pos, 0 * pos], -1)[:, None], 17, 1)[None]
    a0, r0 = accent_phase(dancer(0), [beats])
    a1, _ = accent_phase(dancer(7.5), [beats])
    assert r0 > 0.9 and abs(a0 - 0.2) < 0.1      # (speed from np.diff: up to a frame early)
    assert abs(wrap(a1 - a0) - 0.5) < 0.05 or abs(wrap(a1 - a0) + 0.5) < 0.05
    assert np.isnan(beat_phases([-5.0], beats)[0])
