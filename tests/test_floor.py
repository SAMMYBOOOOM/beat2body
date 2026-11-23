import numpy as np

from beat2body.physics import model_floor
from beat2body.prep.floor import fit_horizon, tripods


def test_horizon_from_identical_objects_at_two_depths():
    # a level camera: base = horizon + c * height; horizon 270, c = 0.9
    tri = [(200.0, 369 - 110, 369), (480.0, 354 - 93, 354), (780.0, 370 - 111, 370)]
    tri = [(x, int(round(b - (b - 270) / 0.9)), b) for x, _, b in tri]
    h, c = fit_horizon(tri)
    assert abs(h - 270) < 2 and abs(c - 0.9) < 0.05


def test_horizon_needs_two_depths():
    assert fit_horizon([(100.0, 200, 300), (500.0, 201, 301)]) is None        # same height = same depth


def test_tripod_detection_on_a_drawn_background():
    bg = np.full((300, 600), 240, np.uint8)
    for x, top, base in ((150, 100, 220), (420, 120, 200)):
        bg[top:top + 8, x - 6:x + 6] = 20                                      # camera head
        for dx in (-25, 0, 25):                                                # three legs
            for y in range(top + 8, base + 1):
                xx = int(round(x + dx * (y - top - 8) / (base - top - 8)))
                bg[y, xx - 1:xx + 1] = 20
    found = sorted(tripods(bg))
    assert len(found) == 2
    assert all(abs(f[1] - t) <= 1 and abs(f[2] - b) <= 1 for f, (t, b) in zip(found, ((100, 220), (120, 200))))


def test_model_floor_follows_body_size_and_snaps_to_contacts():
    T = 40
    j = np.zeros((T, 17, 2))
    j[:, [5, 6], 1], j[:, [11, 12], 1] = -1.0, 0.0                              # torso length 1
    j[:, [15, 16], 1] = 2.0
    f = model_floor(j, horizon=-0.5, k=2.0)                                     # no contacts: horizon + k * size
    assert np.allclose(f, 1.5)
    planted = np.ones((T, 2), bool)
    f2 = model_floor(j, horizon=-0.5, k=2.0, planted=planted)                   # corrected to the planted ankles
    assert np.allclose(f2, 2.0)
