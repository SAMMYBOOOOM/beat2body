# beat2body — Status and Next Steps (handoff)

Snapshot taken **2026-11-18** from the GitHub repo (branch `main` @ `c145365`) and the
Hugging Face datasets under `SAMBOOM`. Use this file as the starting brief for the next working
session: read it first, then the files it points to.

---

## 1. Where things stand

### 1.1 Done: 2D data pipeline (README §4.1–4.7), all 10 genres × 3 pose tools

The pipeline ran on Colab with `2d/run_pose_tools_colab.ipynb`. Its outputs are on Hugging Face, one
dataset per pose tool:

| HF dataset | Clips train / val / test | Valid joints (overall / worst video) |
| --- | --- | --- |
| `SAMBOOM/AISTDance_dwpose` | 2200 / 454 / 430 | **99.7 % / 92.1 %** |
| `SAMBOOM/AISTDance_openpose` | 2200 / 454 / 430 | 96.8 % / 88.7 % |
| `SAMBOOM/AISTDance_mediapipe` | 2198 / 454 / 430 | 97.1 % / 81.2 % |

- The data covers 210 videos (10 genres × 21), 230,484 frames at 30 fps, about 2.1 h.
- The split is by video: 150 / 30 / 30 videos, which is 15 / 3 / 3 per genre (`--seed 0`).
- MediaPipe has 2 fewer training clips because two clips of `gJB_sFM_c01_d08_mJB0_ch08` fell below
  `--min-valid 0.5`. Val and test are the same for all three tools.
- The `SAMBOOM/Space2`–`Space5` Spaces on the account are old 2024 demos and have nothing to do with
  this project.

**HF layout.** This differs from what `pipeline.py` writes. Colab cell 4 moves the files around
afterwards:

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
- Evaluation metrics (success criteria in `2d/README.md` §1)
- A rendering script. The only renderer is `2d/test_code/combine_test.ipynb`, which has hard-coded
  Windows paths.
- The 3D phase

---

## 2. Issues found

> **Update 2026-11-18:** issues 1 and 3 are fixed in the repo (PR #1). The music-disjoint clips and
> the dataset cards (issue 4) are built locally but **not uploaded to Hugging Face yet**; see Task 1.

1. ✅ **Music leakage across splits** (fixed: `--by music` is now the default split). 29 of the 30 test videos use a song that also appears in train.
   There are 60 songs, 6 per genre (`mXX0`–`mXX5`, parsed from `video_id`). Every dancer also
   appears in more than one split. For music-to-dance generation, this test set measures
   memorization rather than generalization.
2. **The planned beat-alignment metric barely tells real from random.** On the ground-truth DWPose
   keypoints, the Beat Align Score (BAS; σ = 3 frames, kinematic beats = local minima of the
   Gaussian-smoothed (σ = 5) mean joint speed) is **0.269** with matched music and **0.253** with
   shuffled music. BAS alone cannot support success criterion 1.
3. ✅ **Code and docs disagree with the data** (fixed).
   - `pipeline.py` writes `keypoints/<cat>/`, `clips/<split>/` and `2d/meta.csv`. The per-tool
     layout that is on HF exists only in Colab cell 4.
   - In the root `README.md`, the status checklist still shows the data pipeline as not done, and
     the layout section lists only `hiphop/` and `house/`.
   - `2d/README.md` §6 still says "2 categories"; the data now has 10.
   - `2d/README.md` §4.3 lists the OpenPose tool as the PyTorch port, but the Colab run used
     `--backend official` (CMU build, BODY_25).
4. **The HF datasets are public with no dataset card.** They have no license, no citation, and no
   statement of the AIST terms. AIST Dance DB forbids redistributing the videos and requires citing
   Tsuchida et al., ISMIR 2019. The mel features come from copyrighted music.

---

## 3. Next steps (in order)

Each task has an acceptance check. Do not commit generated data (it is already in `.gitignore`).

### Task 1 — Housekeeping

- [x] **`pipeline.py`: add a `--tool-dirs` flag, or make the per-tool layout the default.** Done: the
  per-tool layout is the default, `clips` now reads the cleaned keypoints (no raw videos needed), and
  Colab cell 4 no longer moves files. It should
  write `keypoints/<tool>/<cat>/`, `keypoints/<tool>/normalization.json`, `clips/<tool>/<split>/`
  and `clips/<tool>/meta.csv` directly, the way Colab cell 4 does now.
  *Check:* rerunning `clean` → `clips` on one category reproduces the HF file names and the
  `meta.csv` columns.
- [x] **Add a music-disjoint split: `--by music`.** Done and made the default. Seed 0 gives
  2129 / 523 / 432 clips (143 / 34 / 33 videos), 0 shared songs; `meta.csv` gains a `music` column
  and `clips/<tool>/split.json` records the options. Group by the song ID (regex `_(m[A-Z]{2}\d)_`).
  Per genre, put 1 song in test, 1 in val, and 4 in train.
  *Check:* no song appears in two splits, and every split contains every genre.
- [x] **Fix the docs.** In the root README, update the checklist and layout. In `2d/README.md`,
  change §6 to 10 categories, note the HF layout in §5, and note in §4.3 that the official OpenPose
  backend was used.
- [ ] **Publish to HF:** `python 2d/scripts/hf_sync.py upload --tool dwpose mediapipe openpose --prune`
  replaces the clips with the music split and adds the card. Review `2d/hf_cards/*.md` first
  (`hf_sync.py card`). Needs a write token for `SAMBOOM`.
- [ ] **Add a dataset card (`README.md`) to each HF dataset.** Generated by `hf_sync.py card`. Include the source, the AIST
  citation and terms, the file format (copy §1.1 above), the split method, and the tool. Decide
  whether the datasets should stay public, or become gated or private.

### Task 2 — Data loading (`2d/model/data.py`, new)

- [ ] **Write `DanceClips(root, tool="dwpose", split="train")`.** It reads `clips/<tool>/meta.csv`
  and returns a dict with these fields:
  `keypoints (W,17,2)`, `valid (W,17)`, `mel (W,64)`, `onset (W,)`, `beats` (padded, plus a
  length), and `category_index`.
- [ ] **Normalize mel per channel.** Compute the mean and std on the **train split only** and save
  them to `mel_stats.json`.
- [ ] **Augment training data with a horizontal flip.** Negate x and swap the left/right COCO joints.
- [ ] *Check:* one batch loads, the shapes match, and no NaNs are present.

### Task 3 — Baseline model and training (`2d/model/`, new)

This follows `2d/README.md` §2 and §6.

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

### Task 4 — Evaluation (`2d/model/evaluate.py`, new)

Write one metric per success criterion in `2d/README.md` §1. Always report three reference rows:
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

### Task 5 — Rendering (`2d/scripts/render.py`, new)

- [ ] Port the renderer from `combine_test.ipynb`. It should load `(T,17,2)`, undo the
  normalization (multiply by `scale` and add `pelvis`, or centre the figure on the canvas), draw
  the COCO-17 skeleton, and write a 30 fps `.mp4`. The audio track (`ffmpeg -i video -i wav`) is
  optional.
- [ ] Add an option to show generated and ground-truth sequences side by side.

### Task 6 — Later

- Add the beat loss: pull local minima of joint speed toward `beats`.
- Compare against a small diffusion model in the style of EDGE.
- Re-split with `--by music` and retrain; report results on both splits.
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
