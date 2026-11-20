import numpy as np
import torch

from beat2body.data import CONTACT_THRESHOLD
from beat2body.model.diffusion import EDGEDiffusion, Normalizer, absolute_joints, motion_losses
from beat2body.model.footlock import contact_weights, feet_stats, foot_lock, two_bone_ik

BASE = np.array([[0, -1.5], [-.1, -1.6], [.1, -1.6], [-.2, -1.55], [.2, -1.55], [-.4, -1], [.4, -1],
                 [-.6, -.5], [.6, -.5], [-.7, 0], [.7, 0], [-.2, 0], [.2, 0], [-.25, .8], [.25, .8],
                 [-.25, 1.6], [.25, 1.6]], np.float32)


def _sliding(T=60):
    """A standing body (pelvis-centred joints) whose root slides right while the left foot is 'planted'."""
    m = np.zeros((T, 38), np.float32)
    rel = BASE - BASE[[11, 12]].mean(0)
    m[:, 4:] = rel.reshape(-1)
    m[:, 2] = np.linspace(0, 0.6, T)                       # root slides 0.01 / frame
    m[:, 0] = 1.0                                           # left ankle planted throughout
    return m


def _ankle_speed(m, k):
    j = absolute_joints(torch.from_numpy(m)).numpy()
    return np.linalg.norm(np.diff(j[:, 15 + k], axis=0), axis=-1)


def test_foot_lock_pins_a_planted_foot_and_keeps_legs():
    m = _sliding()
    assert _ankle_speed(m, 0).mean() > 0.009
    out = foot_lock(m)
    assert _ankle_speed(out, 0).max() < 1e-5
    j0, j1 = absolute_joints(torch.from_numpy(m)).numpy(), absolute_joints(torch.from_numpy(out)).numpy()
    for a, b in ((11, 13), (13, 15), (12, 14), (14, 16)):
        assert np.allclose(np.linalg.norm(j0[:, a] - j0[:, b], axis=-1),
                           np.linalg.norm(j1[:, a] - j1[:, b], axis=-1), atol=1e-4)


def test_foot_lock_both_feet_planted_uses_ik():
    m = _sliding()
    m[:, 1] = 1.0                                           # both planted while the body drifts
    rel = m[:, 4:].reshape(len(m), 17, 2)
    rel[:, 15, 0] += np.linspace(0, 0.1, len(m))            # the left ankle also drifts relative to the body
    m[:, 4:] = rel.reshape(len(m), -1)
    out = foot_lock(m)
    assert _ankle_speed(out, 0).max() < 1e-3 and _ankle_speed(out, 1).max() < 1e-3


def test_foot_lock_leaves_motion_without_contact():
    m = _sliding()
    m[:, :2] = 0
    assert np.allclose(foot_lock(m), m)


def test_contact_weights_ignore_short_contacts_and_fade():
    c = np.zeros((30, 2), bool)
    c[10:20, 0] = True
    c[5:6, 1] = True                                        # 1 frame: ignored
    w, start = contact_weights(c, min_frames=3, blend=4)
    assert w[10:20, 0].min() == 1 and w[:, 1].max() == 0
    assert 0 < w[8, 0] < 1 and w[3, 0] == 0 and start[12, 0] == 10


def test_two_bone_ik_reaches_target_with_same_lengths():
    hip, knee, ankle = np.array([0., 0.]), np.array([0.1, 0.8]), np.array([0., 1.6])
    k, a = two_bone_ik(hip, knee, ankle, np.array([0.3, 1.4]))
    assert np.allclose(a, [0.3, 1.4], atol=1e-4)
    assert np.isclose(np.linalg.norm(k - hip), np.linalg.norm(knee - hip), atol=1e-4)
    assert np.isclose(np.linalg.norm(a - k), np.linalg.norm(ankle - knee), atol=1e-4)
    def cross(p, q):
        return p[0] * q[1] - p[1] * q[0]
    assert cross(a - hip, k - hip) * cross(ankle - hip, knee - hip) > 0     # knee bends to the same side


def test_feet_stats_sees_sliding():
    j = absolute_joints(torch.from_numpy(_sliding())).numpy()
    s = feet_stats(j, CONTACT_THRESHOLD)
    assert s["no_foot_planted"] == 0 and s["planted_foot_speed"] > 0.009
    s2 = feet_stats(absolute_joints(torch.from_numpy(foot_lock(_sliding()))).numpy(), CONTACT_THRESHOLD)
    assert s2["planted_foot_speed"] < 1e-5


def _losses(pred, target, **kw):
    norm = Normalizer(-2 * torch.ones(38), 2 * torch.ones(38))
    valid, rv = torch.ones(1, 60, 17, dtype=torch.bool), torch.ones(1, 60, dtype=torch.bool)
    return motion_losses(norm.norm(pred)[None], norm.norm(target)[None], valid, rv, norm, **kw)[1]


def test_label_foot_loss_counts_only_real_contacts():
    target = torch.from_numpy(_sliding())
    target[:, 2] = 0                                        # real: root still, left foot planted
    pred = target.clone()
    pred[:, 2] = torch.linspace(0, 0.6, 60)                 # generated: slides
    assert _losses(target, target, foot_contact="label")["foot"] < 1e-10
    slide = _losses(pred, target, foot_contact="label")["foot"]
    assert abs(slide - 0.01 ** 2) < 2e-5                    # mean over planted frames: (0.01 / frame)^2
    target[:, 0] = 0                                        # real dancer not planted: no foot loss
    assert _losses(pred, target, foot_contact="label")["foot"] < 1e-10


def test_root_accel_loss_penalizes_jitter_not_travel():
    target = torch.from_numpy(_sliding())                   # constant root velocity: no acceleration
    jit = target.clone()
    jit[:, 2] += 0.01 * torch.tensor([(-1) ** i for i in range(60)], dtype=torch.float32)
    assert _losses(target, target, root_accel_weight=1.0)["root_acc"] < 1e-10
    assert _losses(jit, target, root_accel_weight=1.0)["root_acc"] > 1e-4


class _Net(torch.nn.Module):
    n_genres = 10


def test_diffusion_keeps_foot_options():
    d = EDGEDiffusion(_Net(), Normalizer(-torch.ones(38), torch.ones(38)), foot_contact="label", root_accel_weight=2.0)
    assert d.foot_contact == "label" and d.root_accel_weight == 2.0


def test_foot_mask_limits_foot_and_root_losses_to_selected_sequences():
    target = torch.from_numpy(_sliding())
    target[:, 2] = 0
    pred = target.clone()
    pred[:, 2] = torch.linspace(0, 0.6, 60)
    off = _losses(pred, target, foot_contact="label", root_accel_weight=1.0, foot_mask=torch.tensor([False]))
    on = _losses(pred, target, foot_contact="label", root_accel_weight=1.0, foot_mask=torch.tensor([True]))
    assert off["foot"] == 0 and on["foot"] > 1e-5


def _step_motion(T=60, land=10, lift=20):
    """Left ankle moves until frame `land`, stands still through frame `lift`, then moves again; labels from the
    loader's rule (contact_labels)."""
    from beat2body.data import contact_labels
    m = torch.from_numpy(_sliding(T))
    m[:, 2] = 0                                             # root still
    x = torch.zeros(T)
    for t in range(1, T):
        x[t] = x[t - 1] + (0.0 if land < t <= lift else 0.05)
    rel = m[:, 4:].reshape(T, 17, 2)
    rel[:, 15, 0] += x
    m[:, 4:] = rel.reshape(T, -1)
    ank = absolute_joints(m)[:, [15, 16]].numpy()
    m[:, :2] = torch.from_numpy(contact_labels(ank))
    return m


def _static_steps(m, mode):
    """Which steps t -> t+1 the label foot loss penalizes for the left ankle (probe: move only that step)."""
    out, base = [], _losses(m, m, foot_contact=mode)["foot"]
    for t in range(len(m) - 1):
        pred = m.clone()
        rel = pred[:, 4:].reshape(len(m), 17, 2)
        rel[t + 1:, 15, 1] += 0.1                           # a jump at step t -> t+1 only
        pred[:, 4:] = rel.reshape(len(m), -1)
        if _losses(pred, m, foot_contact=mode)["foot"] > base + 1e-6:
            out.append(t)
    return out


def test_aligned_labels_cover_touchdown_and_lift_off():
    m = _step_motion()
    lab = m[:, 0]
    assert lab[10] == 0 and lab[11] == 1 and lab[20] == 1 and lab[21] == 0     # contact[t]: step t-1 -> t
    # aligned: exactly the still steps 10 -> 11 ... 19 -> 20 (touchdown step included, lift-off step 20 -> 21 not)
    assert _static_steps(m, "label") == list(range(10, 20))
    # the first version: one step late (misses touchdown 10 -> 11, penalizes lift-off 20 -> 21)
    assert _static_steps(m, "label-prev") == list(range(11, 21))


def test_real_motion_has_no_foot_loss_with_aligned_labels():
    m = _step_motion()
    assert _losses(m, m, foot_contact="label")["foot"] < 1e-12
    assert _losses(m, m, foot_contact="label-prev")["foot"] > 1e-5                  # penalizes the real lift-off
