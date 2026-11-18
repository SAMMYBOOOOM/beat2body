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
root = snapshot_download("SAMBOOM/AISTDance_dwpose", repo_type="dataset")
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
2. **The planned beat-alignment metric barely tells real from random.** On the ground-truth DWPose
   keypoints, the Beat Align Score (BAS; σ = 3 frames, kinematic beats = local minima of the
   Gaussian-smoothed (σ = 5) mean joint speed) is **0.269** with matched music and **0.253** with
   shuffled music. BAS alone cannot support success criterion 1 (see Task 4).
3. ✅ **Code and docs disagreed with the data; logic lived in notebooks.** The code is now the
   `beat2body` package (`python -m beat2body <command>`); notebooks were replaced by commands
   (`pose ... --backup-dir`, `all`, `hf`, `render`). Docs moved to `docs/pipeline_2d.md`.
4. **The HF datasets are public with no dataset card.** Cards are generated by `hf card`
   (AIST terms + citation, format, split) but **not uploaded yet**.

---

## 3. Next steps (in order)

Each task has an acceptance check. New code goes in the `beat2body` package (no notebooks);
run `ruff check beat2body` and `pytest`.

### Task 1 — Housekeeping

- [x] Per-tool layout in the pipeline; `clips` works from cleaned keypoints (no videos needed).
- [x] Held-out-song test split (`--test-songs`, `--by`, `--val-ratio`), recorded in `split.json`.
- [x] Repo reorganized into the `beat2body` package, `pyproject.toml`, `tests/`, `docs/`.
- [x] Renderer: `python -m beat2body render` (was Task 5).
- [ ] **Publish to HF:** review `python -m beat2body hf card --tool dwpose mediapipe openpose`
  (writes `2d/hf_cards/`), then `python -m beat2body hf upload --tool dwpose mediapipe openpose --prune`.
  This replaces the clips with the new split and adds the cards. Needs a write token for `SAMBOOM`.
  Decide whether the datasets stay public or become gated.

### Task 2 — Data loading (`beat2body/data.py`, new)

- [ ] **Write `DanceClips(root, tool="dwpose", split="train")`.** It reads `clips/<tool>/meta.csv`
  and returns a dict with these fields:
  `keypoints (W,17,2)`, `valid (W,17)`, `mel (W,64)`, `onset (W,)`, `beats` (padded, plus a
  length), and `category_index`.
- [ ] **Normalize mel per channel.** Compute the mean and std on the **train split only** and save
  them to `mel_stats.json`.
- [ ] **Augment training data with a horizontal flip.** Negate x and swap the left/right COCO joints.
- [ ] *Check:* one batch loads, the shapes match, and no NaNs are present.

### Task 3 — Baseline model and training on AWS (`beat2body/model.py`, `beat2body/train.py`, new)

This follows `docs/pipeline_2d.md` §2 and §6. On AWS the machine only needs
`pip install -e ".[data]"` and `python -m beat2body hf download --tool dwpose`.

- **Audio encoder:** a 2-layer 1D CNN or a 2-layer Transformer (d = 128, 4 heads) over the mel
  bands plus onset, giving 65 channels per frame.
- **Category conditioning:** an embedding with 10 categories and dimension 16, injected at every
  decoder step. The number of categories is read from `categories.json` and stored in the
  checkpoint.
- **Decoder:** a GRU (1–2 layers, hidden 256) that predicts frame $t$ from the audio features at
  $t$ and the pose at $t-1$. The output head is linear, giving $17 \times 2$ values per frame.
- **Losses**, all masked by `valid`:

$$
\mathcal{L} = \frac{\sum_{t,j} v_{t,j}\,\lVert \hat{x}_{t,j}-x_{t,j}\rVert^2}{\sum_{t,j} v_{t,j}}
+ \lambda_v \frac{\sum_{t,j} v_{t,j}v_{t-1,j}\,\lVert \Delta\hat{x}_{t,j}-\Delta x_{t,j}\rVert^2}{\sum_{t,j} v_{t,j}v_{t-1,j}}
\;(+\;\lambda_b \mathcal{L}_{\text{beat}} \text{ later})
$$

- **Avoid collapse to an average pose.** Use scheduled sampling, feeding the model's own
  predictions with a probability that ramps from 0 to about 0.5, or add Gaussian noise to the
  previous-pose input. Also seed generation with a few real frames, or with a learned start pose
  per category.
- **Hardware:** it must fit in 8 GB. Write a `train.py` with a config, checkpoints to `runs/`, and
  logging of validation loss.
- *Check:* the model overfits one batch, and the validation loss goes down. The generated
  sequence's mean joint speed is not ≈ 0.

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
