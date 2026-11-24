import numpy as np

from beat2body.physics import centre_of_mass, local_floor, physics_stats, summarize

BASE = np.array([[0, -1.5], [-.1, -1.6], [.1, -1.6], [-.2, -1.55], [.2, -1.55], [-.4, -1], [.4, -1],
                 [-.6, -.5], [.6, -.5], [-.7, 0], [.7, 0], [-.2, 0], [.2, 0], [-.25, .8], [.25, .8],
                 [-.25, 1.6], [.25, 1.6]])


def test_com_of_a_symmetric_standing_body_is_central_and_near_the_hips():
    c = centre_of_mass(BASE)
    assert abs(c[0]) < 1e-9 and -0.6 < c[1] < 0.2


def test_jump_recovers_gravity_and_no_horizontal_acceleration():
    g, T = 0.02, 60
    y = np.zeros(T)
    y[20:40] = -(0.5 * g * (np.arange(20) - 0) * (20 - np.arange(20)))   # a parabola: up and down in 20 frames
    j = np.repeat(BASE[None], T, 0).copy()
    j[..., 1] += y[:, None]
    j[..., 0] += np.linspace(0, 0.3, T)[:, None]                          # steady drift, no horizontal acceleration
    sc = np.zeros(T)
    sc[20:40] = 0.08 * np.sin(2 * np.pi * np.arange(20) / 19)            # legs scissor in the air (ankles never slow),
    j[:, 15, 0] += sc                                                     # opposite, so the CoM does not shift sideways
    j[:, 16, 0] -= sc
    s = summarize([physics_stats(j)])
    assert s["flights"] == 1
    assert abs(s["flight_acc_y_median"] - g) < 1e-3                       # image y down: falling = +g
    assert s["flight_acc_x_abs_median"] < 1e-6


def test_standing_still_has_no_flight_no_penetration_and_balance():
    j = np.repeat(BASE[None], 40, 0)
    s = summarize([physics_stats(j)])
    assert s["flights"] == 0 and s["balance_out_share"] == 0
    assert np.isnan(s["penetration_mean"])                                # no swinging foot to measure


def test_local_floor_interpolates_planted_heights():
    ay = np.array([[1.0, 1.0], [0.5, 1.2], [0.4, 0.4], [1.4, 0.3]])
    planted = np.array([[True, False], [False, False], [False, False], [True, True]])
    f = local_floor(ay, planted)
    assert np.allclose(f, [1.0, 1.1333, 1.2667, 1.4], atol=1e-3)


def test_genre_matched_reference_covers_every_genre():
    from beat2body.evaluate import genre_matched
    rows = [{"category": c} for c in ["a"] * 10 + ["b"] * 4 + ["c"] * 6]
    ref = [{"category": c} for c in ["a"] * 2 + ["b"] * 3 + ["c"] * 3]
    idx = genre_matched(rows, ref)
    got = [rows[i]["category"] for i in idx]
    assert got.count("a") == 2 and got.count("b") == 3 and got.count("c") == 3
    assert idx[:2] == [0, 5]                                    # spread over the genre, not the first windows


def test_by_genre_scores_each_genre():
    import numpy as np
    import torch

    from beat2body.evaluate import by_genre, kinetic_features, pose_features
    rng = np.random.default_rng(0)
    j = rng.standard_normal((8, 40, 17, 2)) * 0.1 + np.linspace(0, 1, 17)[None, None, :, None]
    genre = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    m = torch.zeros(8, 40, 38)
    z_k = (kinetic_features(j).mean(0), kinetic_features(j).std(0) + 1e-8)
    z_p = (pose_features(j).mean(0), pose_features(j).std(0) + 1e-8)
    beats = [np.array([10.0, 30.0])] * 8
    out = by_genre(["x", "y"], genre, j, j, j, genre, m, m, beats, beats, z_k, z_p,
                   {"x": ({"torso": 0.1}, {"torso": 0.2})})
    assert set(out) == {"x", "y"} and out["x"]["windows"] == 4
    assert abs(out["x"]["real_train_FID_k"]) < 1e-3 and out["x"]["drift_torso"] == 0.1
