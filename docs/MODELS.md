# Model registry

Every training run so far: what it is, how it differs from its parent, and how it scored. The settings come from
each run's saved config (`model.pt` → `config`, and the `start` line of `metrics.jsonl`). Checkpoints and videos
are on Hugging Face, in [SizheZ03/beat2body-runs](https://huggingface.co/SizheZ03/beat2body-runs) (private), under
`runs/<job>/`.

**Current base model: run 4.1** (decided 2026-11-20). New variants = 4.1's settings + one change.

## Shared by every run

| Part | Setting |
|---|---|
| Data | AIST Dance DB, front camera (c01), 1,410 videos, 10 genres; DWPose keypoints → COCO-17, cleaned (`docs/pipeline_2d.md`). Train / val split by video; one song per genre is held out as the test set (60 songs, 10 for test). |
| Motion | 38 features per frame, 30 fps: 2 foot-contact flags (ankle speed < 0.012 torso lengths per frame), root (pelvis position in the image), 17 joints × (x, y) relative to the pelvis. |
| Music | EDGE's 35 audio features + beat phase (sin, cos); `feature_set edge`. |
| Network | EDGE-style transformer decoder: the noisy motion attends to the music (cross-attention); FiLM conditioning on the diffusion step and genre. |
| Diffusion | 1,000 steps, cosine schedule, predicts x0; loss weights 0.636 reconstruction, 2.964 velocity, 0.646 joint positions, 10.942 foot skating (EDGE), unless noted. |
| Guidance | Classifier-free: music and genre dropped independently with p = 0.25; guidance weight 2. |
| Sampling | DDIM, 50 steps, eta 1. Long dances: 150-frame (5 s) windows overlapping by 75 frames, denoised together and cross-faded. |
| Optimizer | Adan, lr 4e-4, batch 64, EMA 0.9999. |
| Model sizes | **7.1M** = d 256, ff 512, 4 layers, 4 heads. **46.9M** = d 512, ff 1024, 8 layers, 8 heads (EDGE's size). |

## Design options

| Option | Values | Meaning |
|---|---|---|
| Recipe | `edge` | Weight decay 0.02, dropout 0.1, no augmentation, cosine LR with 1,000 warm-up steps, grad clip 1.0, 30k steps, model.pt = lowest val loss. |
| | `regularized` | Weight decay 0.1, dropout 0.2, Min-SNR 5, augmentation (time-stretch ±10%, scale ±5%, rotate ±5°, root shift ±0.2), model.pt = best val FID_k. |
| Body scale | `dataset` / `video` / `dancer` | One scale for all videos / per-video torso / per-dancer median torso (robust to floor work). |
| Bone loss | weight | Each body bone's on-screen length matches the same bone in the real frame. |
| Energy loss (old) | weight, amplitude 1 | Per joint and window: log ratio of speed and of movement amplitude to the real window; squared; every noise level. |
| Energy loss (new) | `--energy-amplitude 0 --energy-size-norm --energy-max-t 500 --energy-shape gaussian` | Speed only, on motion with body-size changes removed, only at low noise, bounded (Gaussian) penalty. |
| Foot fix | `--foot-contact label --foot-weight W --root-accel-weight 5 --foot-max-t 500` | Foot-skating loss on the real dancer's planted steps (not the model's own contact guess), plus root-acceleration matching; both at low noise only. |
| Contact labels | `label-prev` / `label` | First version: one step late (penalizes lift-off, misses touchdown). Fixed 2026-11-20: aligned with the step the loss penalizes. |
| Beat attention | `--beat-attn` | Learned cross-attention bias by beat-relative offset and beat phase. |
| z-score / v-pred | `--normalizer zscore --param v` | Standardized features; the network predicts v instead of x0. |
| Relocation loss | `--reloc-weight W --reloc-max-t 500` | Each ankle's and the pelvis's 0.5 s displacement vs the real window's (log ratio). Implemented, not trained: a style target, replaced by the style condition. |
| Gravity loss | `--gravity-weight 0.1 --gravity-down-weight 10 --gravity-max-t 500` | Physics: per jump (flight detected on the generated motion), the centre of mass falls at g with no sideways acceleration (parabola over the jump); everywhere it never accelerates down faster than 1.25 g. |
| Style condition | `--style-cond` | Manner descriptors (speed, beat-band power, relocation, contact share) of the real window as a condition, dropped with p 0.25; at generation `--style none \| genre \| real`. Replaces style-forcing losses. |
| Foot lock | `--foot-lock --root-smooth 1` (sampling only) | Pins a foot the model marks as planted (the root absorbs its motion, leg IK) and smooths the root. Not a training change. |

## All runs

Test = held-out test songs, 769 windows, seed 0. Real: Div_k 7.8, BeatPow 1, BoneCV 0.167, PFC 0.033, 40% of frames
with no foot planted, swing speed 0.089. "–" = not measured.

| Run | Job (`beat2body-diffusion-…`) | Size | Design (vs parent) | Test FID_k / FID_p | Speed / BeatPow | PFC | Status |
|---|---|---|---|---|---|---|---|
| First small | `small-3067b97d` | 7.1M | EDGE losses, dataset scale, 20k steps, constant LR | – | – | – | superseded |
| Model A | `small-bone-1fa7ddee` | 7.1M | + per-video scale, bone loss 1.0 | 43.7 / 34.6 | 0.52 / 0.20 | 0.042 | superseded (too little energy) |
| reg-small | `reg-small-c78c44fb` | 7.1M | Model A + `regularized` recipe | val only | – | – | worse poses and drift |
| fix-small | `fix-small-0822b90b` | 7.1M | reg-small + z-score + v-prediction | val only | – | – | failed (val FID_k 974) |
| fix-energy | `fix-energy-ec38b1da` | 7.1M | fix-small + energy loss 0.25 | val only | – | – | dropped |
| 2.1 | `b-small-co-6c8ccdc6` | 7.1M | `regularized`, bone 1.0, energy 0.25, dancer scale, 30k, select FID_k | 22.5 / 41.8 | 1.19 / 1.31 | 0.050 | superseded |
| 2.1b | `c-small-co-best-so-far` (local name of `c-small-co-f9ed5641`) | 7.1M | `edge` recipe, bone 1.0, old energy 0.25, dancer scale | 28.6 / 42.2 | 1.13 / 0.98 | 0.050 | base 2026-11-19 → 11-20 |
| 2.2 | `c-edge-pur-0fc7c51e` | 46.9M | pure EDGE (no bone, no energy, dataset scale) | 25.0 / 15.6 | 0.72 / 0.45 | – | reference |
| 2.3 | `c-small-pu-eb0ffe3f` | 7.1M | pure EDGE | 33.9 / 22.1 | 0.56 / 0.21 | – | reference |
| 3.1 | `ba-pure-8a92bce1` | 7.1M | 2.3 + beat attention | 30.9 / 35.1 | 0.65 / 0.31 | – | dropped (no timing gain) |
| 3.2 | `ba-co-48ba8a4d` | 7.1M | 2.1b + beat attention | 41.8 / 33.4 | 1.27 / 1.20 | – | dropped |
| 2.1b.1 | `2.1b.1-large-2f9d37ca` (local name of `base-large-2f9d37ca`) | 46.9M | 2.1b at 46.9M | 23.7 / 33.5 | 1.23 / 1.39 | 0.039 | superseded by 4.1 |
| 2.1b.2 | `e-gauss-b9934c0c` | 7.1M | 2.1b, new energy loss (weight 0.1, Gaussian σ 0.3) | **21.9 / 22.5** | 0.84 / 0.44 | 0.046 | best FIDs; too little energy |
| 2.1b.3 | `e-square-5f731354` | 7.1M | 2.1b.2 with the squared shape | 22.7 / 23.2 | 0.87 / 0.45 | 0.051 | control |
| **4.1** | `v41-863026e8` | 46.9M | 2.1b.1 + foot fix (`label-prev`, weight 50) | 23.5 / 25.1 | 1.16 / 1.15 | **0.020** | **base model** |
| 4.2-allt | `r42-od-ddd05822` | 7.1M | 2.1b.2 + foot fix (`label-prev`, weight 50) at every noise level | 28.8 / 77.9 | 0.70 / 0.58 | 0.069 | feet frozen |
| 4.2 | `v42-98ca09dc` | 7.1M | 2.1b.2 + foot fix (`label-prev`, weight 50) | 32.7 / 109.5 | 0.72 / 0.66 | 0.065 | feet frozen |
| 4.3 | `v43-c03b2cc0` | 7.1M | 4.2 with aligned labels (`label`) | 31.5 / 118.8 | 0.74 / 0.71 | 0.068 | feet frozen |
| 4.4 | `v44-5df86e13` | 46.9M | 4.1 + aligned labels + the new energy loss | 28.0 / 29.7 | **0.98 / 0.90** | 0.028 | real energy and contacts; slow swing (0.025) |
| 4.4a | `v46-1585b336` (us-west-2) | 46.9M | 4.4 without the root-acceleration loss | trained | – | – | diagnostic (was named 4.6) |
| 4.8 | `v48-9756f3d1` | 7.1M | 4.5 without the root-acceleration loss | trained | – | – | diagnostic (small models) |
| **5.1** | `v51-da74960e` | 46.9M | 4.4 with the energy loss replaced by the style condition | trained | – | – | first of the 5.x series |
| 4.5 | `v45-4c119151` | 7.1M | 4.3 with foot weight 5 | 26.0 / 113.3 | 0.72 / 0.58 | 0.055 | poses still broken (not only the weight) |

2.1b.2 / 2.1b.3 / 4.2–4.5 use the new energy loss (weight 0.1). The old energy loss is weight 0.25 with
amplitude 1, squared, at every noise level. All 2.x–4.x runs use bone loss 1.0 and dancer scale, except the
pure-EDGE runs 2.2, 2.3 and 3.1.

Series: **4.x** = foot fix; **5.x** = manner (style) as a condition instead of style-forcing losses.

## Lineage and what each step taught us

- **First small → Model A:** fixing body scale and adding the bone loss made poses realistic, but the dancer
  barely moved (BeatPow 0.20).
- **Model A → reg / fix runs:** regularization, z-score and v-prediction did not help (worse poses; fix-small
  diverged).
- **2.1 / 2.1b:** the old energy loss restored energy (BeatPow ~1). Later analysis showed most of it came from
  oversized limbs (1.4× real) and body-size pulsing, not rhythm.
- **2.2 / 2.3 (pure EDGE):** good poses, too little energy. The 46.9M size helps on every measure.
- **3.1 / 3.2:** beat-relative attention gave no measurable timing gain; dropped.
- **2.1b.1:** size helps on the test songs (FIDs, drift), but energy overshoots (BeatPow 1.39) and it overfits
  after 14k steps.
- **2.1b.2 / 2.1b.3:** the new energy loss gives real limb size and body size and the best FIDs, but the dancer
  sways (only 9% of motion in the 1–4 Hz beat band, real 22%).
- **4.x (foot fix):** at 46.9M (4.1) the model learned real-like stepping (contact share, swing speed and root
  jitter close to real; PFC halved). At 7.1M and foot weight 50 (4.2, 4.2-allt, 4.3) the feet froze (swing speed
  0.004 vs 0.089). Aligning the labels did not change that, nor did foot weight 5 (4.5: feet move 5× more but
  pose FID stays ~113). The small models' failure is not only the foot weight: the root-acceleration loss or the
  label-gated foot loss itself is the next suspect.
- **4.4 (46.9M):** the new energy loss + aligned labels give real energy (speed 0.98, BeatPow 0.90), real contact
  length, share and stance drift, and the best body-size stability, but slow swinging feet (0.025 vs 0.089) and
  FIDs above 4.1's. 4.1 stays the base.
- **Physics (2D, `reports/2026-11-22_physics/`):** real jumps fall at g; every model's jumps are floaty (42–64% of
  g). A gravity-in-flight loss is the physics-based candidate.

## How to evaluate a new run

```bash
python -m beat2body evaluate --checkpoint runs/<job>/model.pt --split test --subset all --seed 0 --steps 50 \
    --eta 1 --guidance 2 --batch-size 64 --long-seconds 20 --tag feet2_raw
python -m beat2body evaluate ... --tag feet2_lock --foot-lock --root-smooth 1
python reports/2026-11-20_feet/summarize.py          # add the run to RUNS first
```

Always check swing speed (feet must still move between contacts) and contact share, not only FID and BeatPow.

Reports: [`reports/2026-11-19_eval_2x/`](../reports/2026-11-19_eval_2x/README.md) (2.x / 3.x),
[`reports/2026-11-19_2.1b_variants/`](../reports/2026-11-19_2.1b_variants/README.md) (2.1b family, sliding analysis),
[`reports/2026-11-20_feet/`](../reports/2026-11-20_feet/README.md) (foot fixes, 4.x, contact labels).
