# beat2body — Status and Next Steps (handoff)

Snapshot taken **2026-11-18** from the GitHub repo (branch `main` @ `c145365`) and the
Hugging Face datasets under `SAMBOOM`. Use this file as the starting brief for the next working
session: read it first, then the files it points to.

---

## 1. Where things stand

### 1.1 Done: 2D data pipeline (README §4.1–4.7), all 10 genres × 3 pose tools

The pipeline ran on Colab (the commands are now in `docs/pipeline_2d.md`, "Running the full pose step
on Colab"). Its outputs are on Hugging Face, one dataset per pose tool. Clip counts **currently on HF**
(old per-video split; the new split is built locally, see Task 1):

| HF dataset | Clips train / val / test | Valid joints (overall / worst video) |
| --- | --- | --- |
| `SAMBOOM/AISTDance_dwpose` | 2200 / 454 / 430 | **99.7 % / 92.1 %** |
| `SAMBOOM/AISTDance_openpose` | 2200 / 454 / 430 | 96.8 % / 88.7 % |
| `SAMBOOM/AISTDance_mediapipe` | 2198 / 454 / 430 | 97.1 % / 81.2 % |

- The data covers 210 videos (10 genres × 21), 230,484 frames at 30 fps, about 2.1 h.
- Old split on HF: by video, 150 / 30 / 30 videos (15 / 3 / 3 per genre). New default: one song per
  genre held out as test, train / val by video (Task 1).
- MediaPipe has 2 fewer training clips because two clips of `gJB_sFM_c01_d08_mJB0_ch08` fell below
  `--min-valid 0.5`. Val and test are the same for all three tools.
- The `SAMBOOM/Space2`–`Space5` Spaces on the account are old 2024 demos and have nothing to do with
  this project.

**HF layout** (identical to the local data root `2d/`):

```text
2d/categories.json                                  {"ballet_jazz":0,"break":1,"house":2,"krump":3,"la_hiphop":4,
                                                     "lock":5,"middle_hiphop":6,"pop":7,"street_jazz":8,"waack":9}
2d/keypoints/<tool>/<category>/<video_id>.npz      keypoints (T,17,2) f32, valid (T,17) bool, pelvis (T,2), scale,
                                                    fps, width, height, tool, joint_names   (COCO-17, pelvis-centred,
                                                    divided by dataset median torso length)
2d/keypoints/<tool>/normalization.json
2d/features/audio_mel/<video_id>.npy               (T,64) log-mel, hop 735 @ 22.05 kHz  (1 frame = 1 motion frame)
2d/features/audio_onset/<video_id>.npy             (T,)
2d/features/beats/<video_id>.npy                   beat frame indices (motion-frame units)
2d/clips/<tool>/meta.csv                            clip_path, audio_feat_path, category, start_frame, end_frame,
                                                    split, video_id, dancer
2d/clips/<tool>/<split>/<category>__<video_id>__<start:06d>.npz
                                                    keypoints (240,17,2), valid (240,17), mel (240,64), onset (240,),
                                                    beats (window-relative), category, category_index, video_id,
                                                    start, end, fps
```

Download one tool with:

```python
from huggingface_hub import snapshot_download
root = snapshot_download("SizheZ03/AISTDance_dwpose", repo_type="dataset")  # private: needs HF_TOKEN
```

### 1.2 Pose-tool comparison (computed on all 210 videos)

The table gives the median per-frame L2 distance between tools, in torso-length units:

| Pair | Median L2 | Worst joints |
| --- | --- | --- |
| DWPose vs OpenPose | 0.034 | wrists, elbows (~0.04) |
| DWPose vs MediaPipe | 0.052 | ankles (~0.10), ears (~0.07) |
| OpenPose vs MediaPipe | 0.058 | ankles (~0.10), ears (~0.08) |

**Decision: train on DWPose.** Keep OpenPose and MediaPipe for robustness and ablation runs.

### 1.3 Not started

- A PyTorch `Dataset` / `DataLoader`
- The model (README §2 baseline)
- A training loop, checkpoints, and config
- Evaluation metrics (success criteria in `docs/pipeline_2d.md` §1)
- The 3D phase

---

## 2. Issues found (and what was done)

1. ✅ **Test music was also in training.** 29 of the 30 old test videos used a song from train.
   Decision (2026-11-18): this project is **not zero-shot**, so train and val share songs (split by
   video), but **one song per genre is held out completely as the test set** for the final evaluation
   (`--test-songs 1`, the new default). Seed 0: 2155 / 440 / 489 clips, 145 / 29 / 36 videos.
2. ✅ **Wrong beat labels (fixed 2026-11-18).** librosa's beat tracker picked the wrong tempo for 10 of
   60 songs (all 80 BPM songs except mWA0, plus mHO5 and mJB1), affecting 389 of 2155 train clips. Beats are
   now a grid at the published AIST tempo (`python -m beat2body beats`); `python -m beat2body audit` checks
   labels, split, clips and beat tempo. Phase is uncertain for mHO4, mJS3, mKR1, mLH4.
3. **The planned beat-alignment metric barely tells real from random.** On the ground-truth DWPose
   keypoints, the Beat Align Score (BAS; σ = 3 frames, kinematic beats = local minima of the
   Gaussian-smoothed (σ = 5) mean joint speed) is **0.269** with matched music and **0.253** with
   shuffled music. BAS alone cannot support success criterion 1 (see Task 4).
4. ✅ **Code and docs disagreed with the data; logic lived in notebooks.** The code is now the
   `beat2body` package (`python -m beat2body <command>`); notebooks were replaced by commands
   (`pose ... --backup-dir`, `all`, `hf`, `render`). Docs moved to `docs/pipeline_2d.md`.
5. ✅ **Dataset cards.** Published with the data at `SizheZ03/AISTDance_*` (public since 2026-11-18).

---

## 3. Next steps (in order)

Each task has an acceptance check. New code goes in the `beat2body` package (no notebooks);
run `ruff check beat2body` and `pytest`.

### Task 1 — Housekeeping

- [x] Per-tool layout in the pipeline; `clips` works from cleaned keypoints (no videos needed).
- [x] Held-out-song test split (`--test-songs`, `--by`, `--val-ratio`), recorded in `split.json`.
- [x] Repo reorganized into the `beat2body` package, `pyproject.toml`, `tests/`, `docs/`.
- [x] Renderer: `python -m beat2body render` (was Task 5).
- [x] **Published to HF** (2026-11-18) as private datasets in the user's own account:
  `SizheZ03/AISTDance_{dwpose,mediapipe,openpose}` with the new split and dataset cards.
  `SAMBOOM/AISTDance_*` (public, old split, no card) were left untouched.

### Task 2 — Data loading ✅ (`beat2body/data.py`)

- [x] `DanceClips(tool, split, seq_len)` / `make_loader(...)`: `keypoints (L,17,2)`, `valid (L,17)`,
  `audio (L,65)` = mel + onset, `beats (L,)` 0/1 mask (instead of padded indices), `category`, `video_id`, `start`.
- [x] Audio normalized per channel with **train-split** stats, cached in `clips/<tool>/audio_stats.json` with a
  fingerprint of the train videos (recomputed after a re-split). Save them in the model checkpoint.
- [x] Train augmentation: random temporal crop (`seq_len` < 240) and left/right flip (x negated, joints swapped).
- [x] `ensure_data(tool)` / `data --download` fetch the dataset on a fresh machine (needs `HF_TOKEN`).
- [x] Check: `python -m beat2body data --tool dwpose` → all splits load, no NaNs, 10 genres, train audio
  mean 0 / std 1; 2155 train clips in memory, 0.15 s per epoch with `num_workers=0`. Tests in `tests/test_data.py`.

### Task 3 — Model and training on AWS (EDGE replica; code ready, not launched)

The model replicates EDGE (Tseng et al., CVPR 2023) as closely as 2D allows; values below were read
from EDGE's code (github.com/Stanford-TML/EDGE). `python -m beat2body train` uses them by default.

| | EDGE | ours |
| --- | --- | --- |
| Motion per frame | 4 foot contacts + 3 root + 24 x 6D rotations = 151 | 2 ankle contacts + 2 root (pelvis in image) + 17 x 2D joints = 38 |
| Window | 150 frames (5 s, 30 fps) | same |
| Music input | Jukebox (4800) or "baseline" (35: envelope, 20 MFCC, 12 chroma, onset, beat) | baseline 35 (beat one-hot from the published-tempo grid) + 2 beat-phase (sin, cos) |
| Network | DanceDecoder: d 512, ff 1024, 8 layers, 8 heads, dropout 0.1, rotary, 2-layer music encoder, cross-attention, FiLM (step + pooled music) | same (46.9M params) + genre embedding (FiLM + cross-attn token) |
| Diffusion | 1000 steps, cosine, predicts x0, L2 | same |
| Losses | 0.636 recon + 2.964 velocity + 0.646 FK + 10.942 foot skating (contact > 0.95) | same; FK = absolute 2D joints; masks for unobserved joints |
| Guidance | music dropped 25%, weight 2 | same + genre dropped 25% |
| Optimizer | Adan lr 4e-4 wd 0.02, constant LR, no clipping, EMA 0.9999, batch 64, 2000 epochs | same (66k steps) |
| Sampling | DDIM 50 steps eta 1; long form: half-overlapping windows + guidance ramp + cross-fade | same |

- [x] Implemented (`beat2body/model/edge.py`, `diffusion.py`, `adan.py`, `data.py`, `train.py`); 22 tests.
- [x] Checks: a tiny model memorizing 32 clips reproduces their motion (speed ratio 1.07, root travel
  0.62 vs 0.55, contact 25% vs 28%); a simulated SageMaker job produces `model.pt`.
- [x] EDGE features computed for all 210 videos (`python -m beat2body edge-features`); S3 archive
  `data/dwpose/2d.tar` (125 MB) holds keypoints, features, meta.csv.
- [ ] Watch in the first real run: windows of long pieces disagree on their overlaps before the
  cross-fade (the tiny memorizing model: up to 3.9 torso lengths); `sample` prints this.
- [ ] Launch (billed per instance hour; see AWS section below for setup):
  ```bash
  pip install -e ".[train,aws]"
  python -m beat2body aws submit --instance ml.g5.2xlarge --max-hours 0.5 --name smoke -- --tiny --steps 300
  python -m beat2body aws submit --instance ml.g5.2xlarge --max-hours 8 --name edge
  python -m beat2body aws submit --instance ml.g5.xlarge --spot --max-hours 2 --name gru -- --model gru --steps 20000
  python -m beat2body aws status ; python -m beat2body aws download <job>
  python -m beat2body sample --checkpoint runs/<job>/model.pt --video-id gLH_sFM_c01_d16_mLH4_ch05
  python -m beat2body render --video-id gLH_sFM_c01_d16_mLH4_ch05 --keypoints runs/<job>/generated/gLH_sFM_c01_d16_mLH4_ch05.npz
  ```
- AWS (us-east-1, account 670682107687): bucket `beat2body-670682107687-us-east-1`; role
  `SageMakerTraining-beat2body`; image `pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker` (digest-pinned);
  quotas 1 x ml.g5.2xlarge / ml.g6e.xlarge on demand, 1 x ml.g5.xlarge / ml.g6.xlarge spot.

### Task 4 — Evaluation (`beat2body/evaluate.py`, new)

Write one metric per success criterion in `docs/pipeline_2d.md` §1. Report val during development
and the held-out-song test set only for final numbers. Always report three reference rows:
**GT (matched music)**, **GT (shuffled music)**, and a **static mean pose**.

1. **Beat alignment.** Compute both:
   - BAS, as defined in §2.2 above: $\mathrm{BAS} = \frac{1}{|B^m|}\sum_{b\in B^m} \exp\!\big(-\min_{d\in B^d}(d-b)^2 / 2\sigma^2\big)$.
   - The correlation between the onset envelope and the mean joint speed, at lags of ±5 frames.

   The reference numbers are BAS 0.269 for GT with matched music and 0.253 with shuffled music.
2. **Category consistency.** Train a small genre classifier (1D CNN over keypoints) on real train
   clips, then report its accuracy on generated clips. Also compare per-genre joint-speed
   distributions.
3. **Diversity.** Compute the mean pairwise distance between generated clips for the same genre,
   and FID on movement features (velocity and acceleration statistics).
4. *Check:* GT scores better than the static and shuffled baselines on each metric.

### Task 5 — Rendering

- [x] `python -m beat2body render --video-id ID [--overlay] [--keypoints generated.npy ...]`:
  one stick-figure panel per sequence, beat flashes, audio track, optional original video.

### Task 6 — Later

- Add the beat loss: pull local minima of joint speed toward `beats`.
- Compare against a small diffusion model in the style of EDGE.
- The 3D phase (`3d/`).

---

## 4. Useful facts

- Song ID: `gLH_sFM_c01_d16_mLH0_ch01` → genre `gLH`, dancer `d16`, song `mLH0`.
- Tempo: beats per minute are about 85 (5th percentile), 113 (median), and 158 (95th percentile).
  The beat count per video ranges from 47 to 128.
- `scale` is a single value per tool, shared by all videos. Each video's `pelvis` track is stored
  so pixel coordinates can be recovered.
- Do not mix outputs from runs on fewer categories with these: the scale and the category indices
  differ.
- The OpenPose and AIST licenses allow academic, non-commercial research use only.
