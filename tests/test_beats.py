import numpy as np

from beat2body import aist
from beat2body.prep.audio import beat_grid


def test_published_tempos():
    assert aist.music_bpm("gLH_sFM_c01_d16_mLH0_ch01") == 80
    assert aist.music_bpm("gBR_sFM_c01_d04_mBR5_ch06") == 130
    assert aist.music_bpm("gHO_sFM_c01_d19_mHO5_ch21") == 135
    assert aist.music_bpm("my_song") is None


def test_grid_tempo_and_phase_from_onsets():
    period, phase = 22.5, 7.3                     # 80 BPM at 30 fps, beat at 7.3 + k * 22.5
    t = np.arange(900)
    onset = np.zeros(900)
    for b in np.arange(phase, 900, period):       # strong on the beat, weaker off the beat
        onset += np.exp(-0.5 * ((t - b) / 1.0) ** 2) + 0.4 * np.exp(-0.5 * ((t - b - period / 2) / 1.0) ** 2)
    beats, source, margin = beat_grid(onset, 80)
    assert source == "onset" and margin > 1.5
    assert np.allclose(np.diff(beats), period, atol=1e-4)
    assert abs(beats[0] - phase) < 0.3          # onset is sampled per frame: ~0.2 frame (7 ms) bias


def test_grid_phase_from_reference():
    ref = np.arange(5.0, 900, 22.5) + np.random.default_rng(0).normal(0, 0.5, 40)
    beats, source, _ = beat_grid(np.ones(900), 80, reference=ref)
    assert source == "reference" and abs(beats[0] - 5.0) < 0.3
