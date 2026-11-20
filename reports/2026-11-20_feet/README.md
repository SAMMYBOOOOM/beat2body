# Foot sliding and whole-body shifting: fixes

Background: `reports/2026-11-19_2.1b_variants/README.md`, "Whole-body shifting and foot sliding". The models rarely
keep a foot planted, and a foot the model itself marks as planted keeps moving. EDGE's foot loss, keyed on the model's
own contact output, measured at about 0.1% of the total loss.

## 1. Foot locking after sampling (no retraining)

`beat2body/model/footlock.py`; `python -m beat2body sample|evaluate ... --foot-lock [--root-smooth 1]`.

- Where the model's contact output marks an ankle as planted, the root's motion absorbs that ankle's motion, so the
  body moves over a still foot.
- If both feet are planted, a 2-bone leg IK pins the second one, keeping thigh and shin lengths and the side the knee
  bends to.
- Contacts shorter than 3 frames are ignored, and the correction fades in and out over 4 frames.
- `--root-smooth` applies a Gaussian of that many frames to the root first, against jitter.

Held-out test songs, 769 windows (same protocol as `reports/2026-11-19_eval_2x`):

| | Kinetic FID | Pose FID | BeatPow | PFC (real 0.032) | Planted foot (real 0.008) | No foot planted (real 40%) | Root jitter (real 0.0125) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2.1b | 28.6 | 42.2 | 0.98 | 0.050 | 0.026 | 74% | 0.021 |
| 2.1b + lock | 31.6 | 42.3 | 0.98 | **0.030** | **0.009** | **47%** | 0.021 |
| 2.1b.2 | 22.0 | 22.5 | 0.44 | 0.046 | 0.019 | 65% | 0.016 |
| 2.1b.2 + lock | 23.5 | 22.5 | 0.45 | 0.025 | 0.002 | 37% | 0.015 |
| 2.1b.2 + lock + root smoothing 1 | 23.6 | 22.5 | 0.45 | **0.023** | 0.002 | 37% | **0.012** |

Measures: planted foot = median speed of the slower ankle; no foot planted = share of frames where both ankles move
faster than the contact threshold; root jitter = mean |root acceleration|. All are in torso lengths per frame.
The metrics are in `metrics/`.

- Locking brings foot sliding to real levels (PFC, planted-foot speed, share of frames with a planted foot). Pose
  FID and beat-band power are unchanged.
- Kinetic FID rises a little (+1.5 to +3). The kinetic features include the root's velocity, and locking replaces
  the sliding root motion with motion over the planted foot.
- On 2.1b.2, locking slightly over-plants (planted-foot speed 0.002 vs 0.008 real, 37% vs 40% of frames with no foot
  planted). The model's contact output marks more frames as planted than a real dancer has.
- Root smoothing over 1 frame brings root jitter to real (0.012) at no other cost.
- Over 20 s the body does not wander. The median root range is 2.0 torso lengths for 2.1b.2 with locking, the same
  as real, against 1.6 without locking.

Videos: [`videos/lock_<model>_<genre>_<video>.mp4`](https://huggingface.co/SizheZ03/beat2body-runs/tree/main/reports/2026-11-20_feet/videos)
(on Hugging Face). Each shows real | model | + foot lock | + root smoothing, for house, krump, lock and break, for
2.1b and 2.1b.2. `make_lock_videos.py` makes them and `lock_stats.json`.

## 2. Foot loss on the real contact labels (training)

`train --foot-contact label --foot-weight 50 --root-accel-weight 5`:

- The foot-skating loss applies on frames where the real dancer's ankle is planted (and observed), averaged over
  those frames. At weight 50 it is about 15% of the total loss.
- A new loss matches the root's acceleration to the real one (about 5% of the total), against jitter.

Models trained with this fix are the **4.x series**. They are evaluated with `--foot-lock --root-smooth 1` as well
as without it.

| Run | = | Size | SageMaker job (tag `run`) |
|---|---|---|---|
| 4.1 | 2.1b.1 + foot fix (old energy loss 0.25) | 46.9M | `r41-large-6b34a851` (g6) |
| 4.2 | 2.1b.2 + foot fix (new energy loss: speed only, size-normalized, t < 500, Gaussian) | 7.1M | `f-gauss-4c4e192e` (g5 spot) |

A run of 2.1b + the foot fix (`f-base-0691b3bd`) was stopped before it started, to free the g6 quota for 4.1.
