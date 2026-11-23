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
