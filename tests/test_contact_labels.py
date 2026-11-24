import numpy as np

from beat2body.data import CONTACT_THRESHOLD
from beat2body.prep.contact import UNKNOWN, hysteresis, stance, step_labels


def test_hysteresis_fills_single_gaps_and_drops_short_runs():
    lab = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0], np.int8)
    out = hysteresis(lab, min_run=2)
    assert out.tolist() == [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


def test_hysteresis_keeps_unknown_and_breaks_runs_there():
    lab = np.array([1, 1, UNKNOWN, 1, 0, 0], np.int8)
    out = hysteresis(lab, min_run=2)
    assert out.tolist() == [1, 1, UNKNOWN, 0, 0, 0]


def test_step_labels_use_the_step_into_each_frame_and_mask_unobserved():
    T = 12
    pos = np.zeros((T, 2))
    pos[:5, 0] = np.arange(5) * 0.05                       # moving into frame 4, still from 4 on
    pos[5:, 0] = 0.2
    ok = np.ones(T, bool)
    ok[9] = False                                          # point not observed at frame 9
    lab, sp = step_labels(pos, ok, CONTACT_THRESHOLD, min_run=2)
    assert lab[4] == 0 and lab[5] == 1 and lab[6] == 1     # step 4 -> 5 is the first still one
    assert lab[9] == UNKNOWN and lab[10] == UNKNOWN        # steps 8 -> 9 and 9 -> 10 touch the unobserved frame
    assert np.isnan(sp[9]) and lab[0] == lab[1]


def test_stance_classes():
    heel = np.array([[1, 1, 0, 0, UNKNOWN]]).T.repeat(2, 1).astype(np.int8)
    toe = np.array([[1, 0, 1, 0, 1]]).T.repeat(2, 1).astype(np.int8)
    assert stance(heel, toe)[:, 0].tolist() == [0, 1, 2, 3, 4]       # flat, heel, toe, swing, unknown
