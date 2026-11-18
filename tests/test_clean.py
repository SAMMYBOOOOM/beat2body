import numpy as np

from beat2body.prep.clean import clean_video, interpolate_gaps, remove_spikes


def test_interpolate_short_gaps_only():
    x = np.array([0, np.nan, 2, np.nan, np.nan, np.nan, 6, np.nan], float)
    y = interpolate_gaps(x, max_gap=1)
    assert y[1] == 1 and np.isnan(y[3:6]).all() and np.isnan(y[7])
    z = interpolate_gaps(x, None)
    assert np.allclose(z[:7], np.arange(7)) and z[7] == 6


def test_remove_single_frame_spike_keeps_fast_motion():
    xy = np.stack([np.arange(20.0), np.zeros(20)], 1)
    xy[10] = [10, 100]                       # spike: away and straight back
    out, n = remove_spikes(xy, factor=3)
    assert n == 1 and np.isnan(out[10]).all()
    fast = np.stack([np.arange(20.0) ** 2, np.zeros(20)], 1)
    assert remove_spikes(fast, factor=3)[1] == 0


def test_clean_video_thresholds_confidence():
    k = np.zeros((30, 17, 3), np.float32)
    k[..., 0] = np.arange(30)[:, None]
    k[..., 2] = 1.0
    k[5:15, 3, 2] = 0.0                      # 10 low-confidence frames: longer than max_gap
    xy, _ = clean_video(k, conf_thr=0.3, max_gap=5, outlier_factor=3, sg_window=7)
    assert np.isnan(xy[5:15, 3]).all() and np.isfinite(xy[:, 0]).all()
