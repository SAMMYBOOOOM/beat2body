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

`train --foot-contact label --foot-weight 50 --root-accel-weight 5 --foot-max-t 500`:

- The foot-skating loss applies on frames where the real dancer's ankle is planted (and observed), averaged over
  those frames. At weight 50 it is about 15% of the total loss.
- A new loss matches the root's acceleration to the real one (about 5% of the total), against jitter.
- Both apply only at low noise (t < 500). Applied at every noise level (run `4.2-allt`, job `r42-od-ddd05822`), they
  froze the dancer early in training: at 4k steps, speed 0.13× real vs 0.50 for 2.1b.2, and on-beat movement 0.01 vs
  0.09. At high noise the model's x0 estimate is a blurry mean, and keeping its feet still means keeping it still.

Models trained with this fix are the **4.x series**. They are evaluated with `--foot-lock --root-smooth 1` as well
as without it.

| Run | = | Size | SageMaker job (tag `run`) |
|---|---|---|---|
| 4.1 | 2.1b.1 + foot fix (old energy loss 0.25) | 46.9M | `v41-*` (first copy to start; us-east-1 or us-west-2) |
| 4.2 | 2.1b.2 + foot fix (new energy loss: speed only, size-normalized, t < 500, Gaussian) | 7.1M | `v42-*` (same) |
| 4.2-allt | 4.2 with the foot losses at every noise level (comparison) | 7.1M | `r42-od-ddd05822` |

Each run was submitted as several identical copies across instance types, purchase options and the two regions
(AWS GPU capacity was short); the first copy to start training was kept and the others stopped. For a us-west-2 job,
run the `aws` commands with `AWS_REGION=us-west-2`.

## Results of the foot-fix runs (2026-11-20)

Test songs, 769 windows, seed 0. "Lock" is the corrected foot lock (stage A conventions) with root smoothing 1.
The full table, including contact duration and share, is in `metrics/feet2_summary.csv`
(`python reports/2026-11-20_feet/summarize.py`).

| | Kinetic FID | Pose FID | Speed | On-beat | BoneCV | PFC | No foot planted | Root jitter | Stance drift | Swing speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Real | 0 | 0 | 1.00 | 1.00 | 0.167 | 0.033 | 40% | 0.0125 | 0.022 | 0.089 |
| 2.1b.2 raw | 21.9 | 22.5 | 0.84 | 0.44 | 0.227 | 0.046 | 65% | 0.016 | 0.199 | 0.054 |
| 2.1b.2 + lock | 23.4 | 22.5 | 0.79 | 0.45 | 0.227 | 0.021 | 37% | 0.012 | 0.007 | 0.050 |
| **4.1 raw** (46.9M) | 23.5 | 25.1 | 1.16 | 1.15 | 0.226 | 0.020 | 33% | 0.0122 | 0.081 | 0.102 |
| 4.1 + lock | 57.0 | 25.0 | 1.24 | 1.15 | 0.226 | 0.007 | 29% | 0.024 | 0.002 | 0.097 |
| 4.2 raw (7.1M) | 32.7 | 109.5 | 0.72 | 0.66 | 0.199 | 0.065 | 1.6% | 0.009 | 0.013 | **0.004** |
| 4.2-allt raw | 28.8 | 77.9 | 0.70 | 0.58 | 0.185 | 0.069 | 1.5% | 0.008 | 0.021 | 0.005 |
| 4.3 raw (aligned labels) | 31.5 | 118.8 | 0.74 | 0.71 | 0.192 | 0.068 | 1.4% | 0.009 | 0.012 | 0.004 |

- **The small models froze their feet.** 4.2, 4.2-allt and 4.3 keep a foot planted in 98% of frames and barely
  move it between contacts (swing speed 0.004, real 0.089). Pose FID collapses (78–119). At weight 50, a small model
  that cannot place steps precisely minimizes the penalty by never stepping. Aligning the labels (4.3) does not
  change this. The training checks missed it because they measure whole-body speed, which stayed reasonable.
- **The large model learned to step.** 4.1 improves on its parent 2.1b.1 everywhere: pose FID 33.5 → 25.1,
  on-beat movement 1.39 → 1.15, BoneCV 0.272 → 0.226, PFC 0.039 → 0.020. Its contact share, contact length, swing
  speed and root jitter are close to real. What remains is stance drift (0.081 vs 0.022) and slight over-energy.
- **Foot locking hurts 4.1.** Kinetic FID goes 23.5 → 57.0 and root jitter doubles. With many real-like contacts,
  the lock switches which foot drives the root at every hand-over, which puts jumps in the root's velocity. Use 4.1
  raw. The soft sequence lock (stage D) is meant for its residual stance drift.
- **Next:** the foot weight must be much lower for 7.1M models (about 5). For the large model: 4.1 + aligned labels +
  2.1b.2's energy loss. Evaluations must always include swing speed.
