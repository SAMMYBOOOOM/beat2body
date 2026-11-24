import numpy as np
import torch

from beat2body.physics import flight_mask, gravity_loss

BASE = torch.tensor([[0, -1.5], [-.1, -1.6], [.1, -1.6], [-.2, -1.55], [.2, -1.55], [-.4, -1], [.4, -1],
                     [-.6, -.5], [.6, -.5], [-.7, 0], [.7, 0], [-.2, 0], [.2, 0], [-.25, .8], [.25, .8],
                     [-.25, 1.6], [.25, 1.6]])
G = 0.02


def _jump(acc=G, T=70, start=25, frames=24):
    """Stand, jump (the body follows a parabola with downward acceleration `acc`, legs scissoring so the ankles keep
    moving), land, stand."""
    j = BASE.expand(1, T, 17, 2).clone()
    t = torch.arange(frames, dtype=torch.float32)
    v0 = acc * (frames - 1) / 2                                                  # up, then back to the ground
    y = -(v0 * t - 0.5 * acc * t ** 2)
    j[0, start:start + frames, :, 1] += y[:, None]
    sc = 0.08 * torch.sin(2 * np.pi * t / (frames - 1))
    j[0, start:start + frames, 15, 0] += sc
    j[0, start:start + frames, 16, 0] -= sc
    return j


def _losses(j):
    valid = torch.ones(j.shape[:3], dtype=torch.bool)
    return [x.item() for x in gravity_loss(j, valid, torch.ones(j.shape[0]), G)]


def test_flight_is_detected_in_a_jump_and_not_when_standing():
    assert flight_mask(_jump(), 0.012)[0, 30:44].all()
    assert not flight_mask(BASE.expand(1, 50, 17, 2).clone(), 0.012).any()


def test_a_jump_at_g_has_little_penalty_and_a_floaty_one_more():
    real, floaty = _losses(_jump(G)), _losses(_jump(0.4 * G, frames=30))
    assert real[0] < 0.1 * floaty[0] and floaty[0] > 0.2
    assert real[1] == 0 and floaty[1] == 0                                      # never faster than g downward


def test_crouching_is_free_and_dropping_faster_than_g_is_penalized():
    T = 60
    crouch = BASE.expand(1, T, 17, 2).clone()
    s = torch.zeros(T)
    s[20:40] = 0.4 * (1 - torch.cos(np.pi * torch.arange(20) / 19)) / 2        # smooth descent, feet planted
    s[40:] = 0.4
    crouch[0, :, :15, 1] += s[:, None]
    assert _losses(crouch) == [0.0, 0.0]
    drop = BASE.expand(1, T, 17, 2).clone()
    d = torch.zeros(T)
    d[20:30] = 0.5 * 3 * G * torch.arange(10, dtype=torch.float32) ** 2          # accelerating down at 3 g
    d[30:] = d[29]
    drop[0, :, :15, 1] += d[:, None]
    assert _losses(drop)[1] > 0.1


def test_gravity_loss_has_gradients():
    j = _jump(0.4 * G, frames=30).requires_grad_(True)
    fl, dn = gravity_loss(j, torch.ones(j.shape[:3], dtype=torch.bool), torch.ones(1), G)
    (fl + dn).backward()
    assert j.grad is not None and j.grad.abs().sum() > 0
