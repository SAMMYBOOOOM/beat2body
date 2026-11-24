# beat2body — handoff (2026-11-19)

State of the project for whoever continues it (teammate or a new session): what exists, what the
experiments showed, what is broken or open, and what to try next. Supersedes `NEXT_STEPS.md` for the
model / training part. Branch: `claude/relaxed-knuth-fc6tqf` (fork `novaz03/beat2body`, PR
[SAMMYBOOOOM/beat2body#5](https://github.com/SAMMYBOOOOM/beat2body/pull/5) to the original repo).

---

## 1. Goal and approach

Generate a 2D dance skeleton (COCO-17, 30 fps) from **music + genre**. Data: AIST Dance Video Database
(front camera). Model: a 2D re-implementation of **EDGE** (Tseng et al., CVPR 2023), a transformer
diffusion model with music cross-attention and classifier-free guidance; a GRU baseline exists.
Later stage: the same model on AIST++ 3D.

## 2. Code (Python package, no notebooks)

`python -m beat2body <command>`; details in `docs/pipeline_2d.md`, `docs/colab.md`.

| Area | Module | Commands |
|---|---|---|
| Pose, cleaning, audio, clips, split | `beat2body/pose/`, `beat2body/prep/` | `pose`, `clean`, `audio`, `beats`, `edge-features`, `clips`, `all`, `audit` |
| Dataset / loader | `beat2body/data.py` | `data` |
| Model, diffusion, losses | `beat2body/model/` (`edge.py`, `diffusion.py`, `adan.py`, `nets.py`) | |
| Training, sampling | `beat2body/train.py` | `train`, `sample` |
| Evaluation, videos | `beat2body/evaluate.py`, `beat2body/report.py`, `beat2body/render.py` | `evaluate`, `report`, `render` |
| Hugging Face | `beat2body/prep/hf.py` | `hf download / upload` |
| AWS SageMaker | `beat2body/aws.py`, `beat2body/sagemaker_entry.py` | `aws upload-data / submit / status / progress / download / pose / pose-download` |
| Colab (one cell) | `scripts/colab_train.py` | |

Tests: `pytest` (29 passing). Lint: `ruff check beat2body` (line length 120).

## 3. Data (complete, audited)

**Source:** AIST Dance DB, front camera `c01`, all single-dancer videos of 10 genres:

| Set | Videos | Length | Total |
|---|---|---|---|
| Advanced `sFM` (dancer's own choreography) | 210 | ~35 s | 2.1 h |
| Basic `sBM` (10 genre-basic routines per dancer, each on 4 songs) | 1,200 | 7–12 s | 3.1 h |
| **All** | **1,410** | | **5.2 h** (= AIST++ size) |

**Processing:** 30 fps; DWPose (all), OpenPose + MediaPipe (advanced only) → COCO-17; outlier removal,
gap filling, Savitzky–Golay; pelvis-centred, divided by the dataset median torso. Music: EDGE's 35
features + beat phase (37). Beats on a grid at the **published AIST tempo** (librosa had the wrong tempo
for 10/60 songs).

**Split** (`prep/clips.py: split_videos`): one song per genre held out entirely as **test**; advanced
dances split train/val by video exactly as before; basic dances follow it (test if on a held-out song,
else train/val **by routine**, so tempo versions of a routine never cross splits).

| Split | Videos (adv + basic) | Clips (8 s, stride 2 s) | Hours | Songs |
|---|---|---|---|---|
| train | 145 + 868 = 1,013 | 3,423 | 3.7 | 50 |
| val | 29 + 132 = 161 | 632 | 0.6 | 50 (shared with train) |
| test | 36 + 200 = 236 | 769 | 0.8 | 10 (never seen) |

Test songs: mBR4 mHO4 mJB3 mJS0 mKR5 mLH4 mLO2 mMH4 mPO0 mWA5. `python -m beat2body audit --tool dwpose`
passes (16 break floor-work videos reported as info, see §6).

**Where the data lives**
- Hugging Face (public): `SizheZ03/AISTDance_dwpose` (all 1,410), `SizheZ03/AISTDance_openpose`,
  `SizheZ03/AISTDance_mediapipe` (advanced only). `python -m beat2body hf download --tool dwpose --parts keypoints features meta`.
- S3 training archive: `s3://beat2body-670682107687-us-east-1/data/dwpose/2d.tar` (376 MB, all tools).
- Raw videos: `2d/raw_video/<genre>/` locally. The 210 advanced videos are tracked in git (the repo is
  public; AIST's terms forbid redistribution; the team decided to keep them for now). The 1,200 basic
  videos are git-ignored; re-download with the AIST URL list (`aist.VIDEO_URL`).

## 4. Model and training

Per frame, **motion** = 38 values (2 ankle contacts, 2 root = pelvis position in torso lengths, 34
pelvis-centred joint coordinates); **music** = 37. Windows of 150 frames (5 s).

- Denoiser: 8-layer transformer decoder (d 512, ff 1024, 8 heads, 46.9M) or small (d 256, ff 512,
  4 layers, 4 heads, 7.1M); rotary positions, 2-layer music encoder, cross-attention, FiLM on (t, genre).
- Diffusion: 1000 steps, cosine schedule, predicts x0; music and genre dropped 25% (CFG, guidance 2);
  DDIM 50 steps; long dances by half-overlapping windows.
- Losses (EDGE): 0.636 recon + 2.964 velocity + 0.646 joint positions (FK) + 10.942 foot contact.
  Ours (flags): `--bone-weight` projected bone-length loss, `--energy-weight` per-joint speed / amplitude
  log-ratio loss (`--energy-smooth 2` = measured on blurred motion, jitter does not count),
  `--normalizer zscore`, `--param v`, `--min-snr`.
- Body scale (`--scale-norm`): `dancer` (default; per dancer over all their videos), `video`,
  `dataset` (EDGE-like).
- Recipes (`--recipe`): `edge` (dropout 0.1, wd 0.02, constant LR, no aug); `regularized` (dropout 0.2,
  wd 0.1, cosine LR, warm-up 1000, clip 1.0, 3 pose tools, time-stretch ±10% + scale/rotate/shift
  jitter, Min-SNR 5). Flags override the recipe.
- Checkpoint selection (`--select`): `val_loss` (current choice), `fid_k`, `last`. Snapshots every
  `--sample-every` steps go to the checkpoint folder (S3) and can be evaluated afterwards.
- Training checks every 2,000 steps on 8 val clips: `speed_ratio`, `beat_power`, `bone_cv`.

## 5. Results

### Metrics (`python -m beat2body evaluate`)

FID_k / FID_p: Fréchet distance of per-clip kinetic (joint speed², acceleration; 51) / pose
(pelvis-centred joint means and spreads; 68) features vs real, lower better; Div_k: variety; BAS: beat
alignment (unreliable in 2D); PFC: foot sliding, lower better; speed and **BeatPow** (motion power at
1–4 Hz) relative to real (1 = real); BoneCV and the **20 s drift** (torso variation over a long dance):
should match real. These are 2D adaptations, **not comparable to EDGE's published numbers** (EDGE:
FID_k 42.16, FID_g 22.12 in 3D; ground truth 17.10 / 10.60).

### Validation, advanced dances (440 windows; `--subset advanced`, same set for every model)

| Model | FID_k | FID_p | Div_k | BAS | PFC | Speed | BeatPow | BoneCV | Drift (torso) |
|---|---|---|---|---|---|---|---|---|---|
| Real | 0 | 0 | 9.0 | 0.228 | 0.029 | 1.00 | 1.00 | 0.171 | 0.103 |
| First small model (no size fix) | 52.8 | 55.5 | 7.3 | 0.220 | 0.048 | 0.58 | – | 0.193 | 0.134 |
| **Model A** (small, EDGE recipe, video scale, bone 1.0) | 43.7 | **23.1** | 5.1 | 0.206 | **0.038** | 0.53 | 0.24 | 0.204 | **0.111** |
| reg-small (regularized recipe) | 46.5 | 46.6 | 4.0 | 0.209 | 0.063 | 0.57 | 0.20 | 0.224 | 0.157 |
| fix-small (+ z-score + v-pred) | 974 | 69.1 | 19.3 | 0.244 | 0.075 | 1.06 | 0.26 | 0.238 | 0.047 |
| fix-energy (+ energy 0.25, raw) | 327 | 37.0 | 13.2 | 0.217 | 0.076 | 0.80 | 0.33 | 0.257 | 0.077 |
| **Run 2.1** (step 26k, best val loss) | **36.9** | 67.3 | **10.9** | **0.223** | 0.054 | 1.19 | 1.71 | 0.283 | 0.186 |

Model A, reg/fix runs: advanced dances only. Run 2.1: all 1,410 videos.

### Held-out test songs (769 windows, all dances)

| Model | FID_k | FID_p | Div_k | BAS | PFC | Speed | BeatPow | BoneCV | Drift (torso) |
|---|---|---|---|---|---|---|---|---|---|
| Real | 0 | 0 | 7.4–7.8 | 0.229 | 0.032 | 1.00 | 1.00 | 0.167 | 0.105 |
| Model A | 43.7 | **34.6** | 2.5 | 0.225 | **0.042** | 0.52 | 0.20 | 0.157 | 0.067 (too rigid) |
| Run 2.1 | **22.5** | 41.8 | **7.1** | 0.220 | 0.050 | 1.19 | 1.31 | 0.296 | 0.223 |

### Current runs on the full dataset (30k steps, warm-up 1000 + clip 1.0 + cosine LR)

Training checks (8 val clips; indicative only). Check live with
`python -m beat2body aws progress 6c8ccdc6 f9ed5641 0fc7c51e eb0ffe3f --last 0`.

| Run | Job | Settings | Status | Val loss best | Speed | BeatPow | BoneCV (real 0.140) |
|---|---|---|---|---|---|---|---|
| 2.1 | `b-small-co-6c8ccdc6` | small, regularized recipe, bone 1.0, energy 0.25, dancer scale, select fid_k | done, evaluated | 0.165 @ 24–30k | 1.18 | 2.18 | 0.247 |
| 2.1b | `c-small-co-f9ed5641` | small, EDGE recipe, bone 1.0, energy 0.25, dancer scale | finishing | 0.376 @ 16k | 1.12 | 1.60 | 0.255 |
| 2.2 | `c-edge-pur-0fc7c51e` | **EDGE size (46.9M)**, pure EDGE (no bone/energy, dataset scale) | finishing | 0.160 @ 16–20k | 0.64 | 0.75 (peak 0.82) | 0.187 |
| 2.3 | `c-small-pu-eb0ffe3f` | small, pure EDGE | done, not evaluated | 0.171 @ 14–16k | 0.64 | 0.57 | 0.198 |

Job names are `beat2body-diffusion-<prefix>-<id>-<date>`; results go to
`s3://beat2body-670682107687-us-east-1/output/<job>/`, snapshots to `.../checkpoints/<prefix>/`.

**Videos and checkpoints** are on Hugging Face, [SizheZ03/beat2body-runs](https://huggingface.co/SizheZ03/beat2body-runs)
(private): `runs/<job>/` for every training run, and every generated `.mp4` / `.npz` at its `reports/...` path
(git ignores them). Get one with `hf download SizheZ03/beat2body-runs --include "<path>" --local-dir .`.
Proposal PDF: `reports/proposal/Beat2Body_Proposal.pdf`.

## 6. What we learned / current issues

1. **Low energy ("swaying") was the main failure of the first models.** Model A had 0.25× real power in
   the 1–2 Hz band and 0.17× at 2–4 Hz while slow sway was intact; it was the same on training music
   (not overfitting). The penalty terms (energy loss) fixed it; a bigger model and longer training help too.
2. **z-score scaling (×7.9) breaks the diffusion** (equivalent to far less noise): shrunken poses, root
   jitter, FID_k 974. Do not use `--normalizer zscore` as is.
3. **An energy loss on raw speed is satisfied by jitter** (8–15 Hz at 1.7× real). Fixed: `--energy-smooth 2`.
4. **The regularized recipe made poses and drift worse** (reg-small vs model A: FID_p 46.6 vs 23.1,
   drift 0.157 vs 0.111) and **made 2.1 overshoot** the energy (BeatPow 2.2 vs 1.6 for 2.1b with the
   same penalties). Suspects: scale jitter augmentation (teaches size variation), mixing three pose tools
   (MediaPipe hips 8% narrower, shins 8% shorter than DWPose), Min-SNR. Not yet isolated.
5. **Body size varies too much in every new run** (BoneCV 0.19–0.26 vs real 0.14), most with the
   penalties, although the bone-length loss is 40–50% of their loss. Likely the energy loss's
   **amplitude** term (rewards bigger movement ranges) fighting the bone loss.
6. **Energy keeps rising after validation loss stops improving**, while the EDGE-recipe runs start to
   overfit (train-val gap +0.04 to +0.05 by 30k). Selecting by val loss picks checkpoints at 14–20k with
   less movement on the beat; the late gains may be partly memorization. Val loss is also not comparable
   across runs with different loss terms. Undecided: compare val-loss vs final checkpoints on test songs.
7. **The velocity loss is ~1% of the total loss** (weight 2.964) because frame-to-frame changes are tiny in
   MinMax space; joint positions (FK) dominate (~90% in the pure runs), which favours averaged motion.
8. **EDGE recipe at full size diverged** (loss 1.3M at step 500): no warm-up, no clipping, batch 64 vs
   EDGE's 512. Fixed with warm-up 1000 + grad clip 1.0 + cosine LR (used by all current runs).
9. **Floor work** (break basic routines ch08 / ch10): the torso points at the camera, per-video scaling
   inflated those dancers 2–3×. Fixed: `--scale-norm dancer`.
10. Metrics: BAS is unreliable in 2D (a frozen pose scores high); FID on 440–769 windows is noisy (a few
    points are noise); no genre-accuracy metric yet.

## 7. Next steps (suggested order)

1. **Evaluate 2.1b, 2.2, 2.3** on val (`--subset advanced`) and test, both the best-val-loss and the final
   checkpoints (snapshots in `runs/<job>/live/snapshots/` after `aws download <job> --live`); make
   real | A | 2.x comparison videos (`report --checkpoint ... --split val`, then the ffmpeg hstack used in
   `reports/2026-11-19_run21`). Decide the selection rule from the test results.
2. **Loss balance (small model, ~45 min per run):**
   - rescale the velocity term so it contributes (EDGE's balance in 2D),
   - energy loss on speed only (drop the amplitude term) or weight 0.1,
   - bone weight 1.0 → 0.3.
3. **Isolate the regularized recipe:** EDGE recipe + one component at a time (3 pose tools; scale jitter;
   Min-SNR), or drop scale jitter and extra tools by default.
4. **Train longer** where beat power is still rising (resume from the checkpoint to 60k; EDGE ≈ 2000 epochs
   ≈ 110k steps here) with the chosen regularization against overfitting.
5. **Full-size model with the corrections** once the small-model settings are fixed (2.2 shows size helps).
6. Evaluation: genre classifier (genre accuracy), a rhythm metric to replace BAS, eta 0.5 sampling
   (better than EDGE's eta 1 in a sweep on model A).
7. 3D stage on AIST++ (official features, comparable to EDGE's numbers); other cameras c02–c09 as 2D
   augmentation; OpenPose / MediaPipe on the basic dances for the pose-tool comparison.

### Model design experiments

Prototype separate guidance with an existing checkpoint, then train a beat-relative attention variant
at 7.1M. If timing improves and torso variation remains high, try the kinematic decoder. Add phrase
planning if the 20-second samples still repeat short movements or lack clear transitions.

1. **Separate music and genre guidance.** Training already drops music and genre independently, while
   sampling applies one weight to the joint conditional prediction. In `EDGEDiffusion.predict`, combine
   three predictions as `uncond + w_genre * (genre_only - uncond) + w_music * (joint - genre_only)`.
   Setting both weights equal reproduces the current guidance formula. With a fixed seed, vary one
   weight at a time and check whether motion responds to track changes and requested genre changes.
   [Diffusion guidance](beat2body/model/diffusion.py#L187)
2. **Add beat-relative cross-attention.** Keep the existing beat-phase features and give motion queries
   a learned attention bias based on their signed distance to nearby audio beats in fractions of a beat.
   Allow different heads to learn anticipation, delay, and offbeat accents. Mask the beat information
   whenever the music condition is dropped. Compare against the same 7.1M backbone and test whether
   accents follow a deliberately shifted beat grid at fixed tempo, genre, and seed. Report BeatPow
   alongside timing so an increase in movement alone does not count as better alignment.
   [Denoiser](beat2body/model/edge.py#L73) · [ChoreDiffusion](https://doi.org/10.3390/info17080808)
3. **Decode a connected skeleton.** Predict pelvis translation and bone directions through a kinematic
   tree, with a body scale shared across the sequence and per-frame projected-length changes to preserve
   foreshortening. Check torso and limb variation on long clips and inspect break floor work at matched
   motion energy. As a follow-up, give ankle contact its own supervised head and use confidence-masked
   target contacts to weight the skating loss. The current skating gate depends on the model's own
   thresholded contact output, and ankle speed is an image-plane proxy for contact.
   [Motion representation](beat2body/data.py#L121) · [Bone and contact losses](beat2body/model/diffusion.py#L140)
4. **Plan motion across phrases.** Add a lightweight module that reads the full audio excerpt and predicts
   a coarse root trajectory and pose anchors every few beats. Condition the existing five-second denoising
   windows on that shared plan. Compare 20-second clips for phrase transitions and repetition, and measure
   overlap disagreement before cross-fading.
   [Long-form sampler](beat2body/train.py#L499) · [Lodge](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Lodge_A_Coarse_to_Fine_Diffusion_Network_for_Long_Dance_CVPR_2024_paper.html)

#### Implemented (2026-11-19)

- **Separate guidance** (item 1): `EDGEDiffusion.predict(..., music_guidance=w_music)` with `guidance` as the genre
  weight; the conditional / unconditional passes are batched into one forward call. Flags `--music-guidance`,
  `--genre-guidance` on `sample` and `evaluate` (omitted = the single weight as before; equal weights give the same
  output, tested).
- **Beat-relative cross-attention** (item 2): `train --beat-attn` (`--beat-bins 8`, `--beat-range 2`). Per layer and
  head, a learned bias on each music frame by its offset from the motion frame in beats (1/8-beat bins, ±2 beats)
  plus a bias by the music frame's beat phase. The beat count is unwrapped from the beat-phase sin / cos already in
  the music input, so no data change. Zero-initialized (= the plain model at step 0); 0 where music is dropped,
  outside the beat grid and for the time / genre tokens. Old checkpoints load unchanged.
- **`python -m beat2body probe`**: `guidance` (sweeps one weight at a time with fixed noise: change in motion when
  the track is swapped / another genre is requested, nearest-centroid genre accuracy, beat locking, FID_k, BeatPow)
  and `beat-shift` (moves the beat phase + EDGE beat one-hot by 0.25 / 0.5 beat at fixed tempo, genre, seed and
  reports how far the motion accents move: follow = 1 means they follow the grid; BeatPow alongside).
- **Runs** (7.1M, matched to 2.3 and 2.1b except `--beat-attn`; same seed):
  3.1 `ba-pure-8a92bce1` (= 2.3 + beat attention; 5 h cap, faster bias code), 3.2 `ba-co-48ba8a4d` (= 2.1b + beat attention; 5 h cap, faster bias code; the first submissions with the per-layer bias ran at ~3 steps/s vs 10.5 without and were stopped). The bias is now built once per batch for all layers as one-hot @ table
  matmuls (the per-layer gather's backward accumulated millions of gradients into 33 bins).

**Results of 3.1 / 3.2** (`reports/2026-11-19_eval_2x/metrics/run_3.*`): beat attention made no measurable timing
difference. 3.2 vs 2.1b: test FID_k 41.8 vs 28.6, FID_p 33.4 vs 42.2, speed 1.27, BeatPow 1.20; beat lock 0.024
(chance 0.026), shifted-grid response within noise. Dropped from the base model.

**Run 2.1b.1** = 2.1b's settings at 46.9M (d 512, ff 1024, 8 layers, 8 heads), job
`beat2body-diffusion-base-large-2f9d37ca-2026-11-19-20-32-57-631` (SageMaker tag `run=2.1b.1`; job names cannot be
changed). Tests whether size keeps 2.1b's energy and adds 2.2's poses and body stability.

**Energy overshoot investigation (2026-11-19)**: 128 advanced val windows, generated / real:
- Generated bodies are not larger (torso 0.97-1.00, bones 0.95-0.97 for the 2.1b family), so size does not explain
  the extra speed. The movement range (limbs about the pelvis) is 1.30-1.37x real with the energy loss and 0.73-0.78x
  without it: the energy loss (amplitude term, applied at every noise level) overshoots.
- Real data has depth motion: in 43 of 174 advanced videos all bones grow together as the pelvis moves down in the
  image (toward the camera); slow size change median 25%, up to 85% (ballet jazz). The models learned size changes
  as free motion: torso length pulses at 1-4 Hz at 1.6x real (2.1b) to 2.7x (2.1b.1 at 14k), and the energy loss
  counts these pulses as movement.
- New energy-loss options (`train`): `--energy-amplitude` (0 = speed only), `--energy-shape gaussian`
  (`--energy-sigma`; bounded per window: energy as a tendency, not a target), `--energy-size-norm` (size changes are
  not movement), `--energy-max-t` (low noise levels only). Defaults = the previous loss.
- Runs: 2.1b.2 `e-gauss-b9934c0c` = 2.1b with energy 0.1, speed only, size-normalized, t < 500, gaussian sigma 0.3;
  2.1b.3 `e-square-5f731354` = the same with the square shape (control).
- Run 2.1b.1 (46.9M) finished: lowest val loss at 14k, then overfitting; local folder
  `runs/beat2body-diffusion-2.1b.1-large-2f9d37ca/`.

**Results of the 2.1b family** (`reports/2026-11-19_2.1b_variants/`; test songs, 769 windows):

| Run | FID_k | FID_p | Speed | BeatPow | BoneCV | Torso drift 20 s |
|---|---|---|---|---|---|---|
| 2.1b | 28.6 | 42.2 | 1.13 | 0.98 | 0.27 | 0.178 |
| 2.1b.1 (46.9M) | 23.7 | 33.5 | 1.23 | 1.39 | 0.27 | 0.124 |
| 2.1b.2 (new energy, gaussian) | **22.0** | **22.5** | 0.84 | 0.44 | 0.23 | 0.127 |
| 2.1b.3 (new energy, square) | 22.7 | 23.2 | 0.87 | 0.45 | 0.23 | 0.141 |

- 2.1b's real-looking BeatPow came from oversized limbs (1.4x real) and body-size pulsing, not rhythm: only 11% of
  its motion power is at 1-4 Hz (real 22%). The new loss gives real limb amplitude and body size (hence the FIDs)
  but slow motion (91% below 1 Hz, BeatPow 0.4). 2.1b.1 has the most realistic rhythm (18% at 1-4 Hz) and the fewest
  body collapses.
- **Shifting and foot sliding** (`sliding.py`): root speed is real, root jitter 1.2-1.5x real. The feet are the
  problem: no foot planted in 70-77% of frames (real 44%), and a foot the model marks as planted moves 3-4x faster
  than a real planted foot, so EDGE's foot loss (keyed on the model's own contact output) is nearly inactive. Window
  cross-fading is not the cause (windows disagree by ~0.1 torso lengths, spread over 75 frames).
- Next: (1) contact-driven foot locking after sampling (pin a planted ankle, move the root to compensate; no
  retraining); (2) foot loss on the real contact labels + slower ankle, higher weight, root-acceleration penalty;
  (3) an energy term on the size-normalized limbs' 1-4 Hz power; then the winning loss at 46.9M. 3D (AIST++) would
  make contact physical but EDGE itself still slides in 3D, and it would drop the 2D-from-video premise.

**Foot fixes (2026-11-20, `reports/2026-11-20_feet/`)**: (1) `--foot-lock` (+ `--root-smooth 1`) after sampling brings
foot sliding to real on the test songs (PFC 0.050 -> 0.030 for 2.1b, 0.046 -> 0.023 for 2.1b.2; pose FID unchanged,
kinetic FID +1.5 to +3). (2) Training: `--foot-contact label --foot-weight 50 --root-accel-weight 5 --foot-max-t 500` (at every noise level
the foot losses froze the dancer: run 4.2-allt).
**Naming: the 4.x series = models trained with the foot fix**: 4.1 = 2.1b.1 + foot fix (46.9M, job `v41-*`),
4.2 = 2.1b.2 + foot fix (7.1M, job `v42-*`); jobs may run in us-east-1 or us-west-2 (`AWS_REGION=us-west-2`; bucket,
data and role are set up in both). Evaluate them with and without `--foot-lock --root-smooth 1`.

**Model registry: [docs/MODELS.md](docs/MODELS.md)** (every run's design, parent, test results and status).

**Base model (decided 2026-11-20): run 4.1** (`v41-863026e8`, 46.9M, 2.1b.1 + foot fix; it replaces 2.1b). New
variants = 4.1's settings + one change; use 4.1 without foot locking.

**4.x results (test songs; `reports/2026-11-20_feet/README.md`)**: 4.1 (46.9M) = most balanced model so far, raw:
FID_k 23.5, FID_p 25.1, speed 1.16, BeatPow 1.15, BoneCV 0.226, PFC 0.020, no-foot-planted 33% (real 40%), root
jitter at real, swing speed 0.102 (real 0.089); locking hurts it (FID_k 57). The 7.1M runs 4.2 / 4.2-allt / 4.3
froze their feet (swing 0.004, FID_p 78-119): foot weight 50 is too high for the small model; aligned labels (4.3,
`--foot-contact label`, the fixed version; 4.1 / 4.2 used `label-prev`) do not change that. Always check swing speed.
Contact labels from heel / toe points (stage B) and the plan for a contact head and soft lock:
`reports/2026-11-20_feet/CONTACT_PLAN.md`.

**Guidance sweep results** (64 advanced val windows, fixed noise; nearest-centroid genre accuracy on the real
windows 0.53):

| Model | w_genre | w_music | music_resp | genre_resp | genre_acc | FID_k | BeatPow |
|---|---|---|---|---|---|---|---|
| 2.3 | 2 | 2 (= current) | 0.74 | 0.82 | 0.34 | 38.2 | 0.20 |
| 2.3 | 2 | 1 | 0.46 | 0.80 | 0.45 | 37.5 | 0.20 |
| 2.3 | 4 | 2 | 0.67 | 1.40 | 0.48 | 71.2 | 0.35 |
| 2.1b | 2 | 2 (= current) | 0.96 | 0.94 | 0.34 | 20.5 | 1.07 |
| 2.1b | 2 | 1 | 0.60 | 0.94 | 0.45 | 20.5 | 1.21 |
| 2.1b | 0 | 2 | 1.51 | 1.16 | 0.08 | 19.6 | 1.07 |
| 2.1b | 2 | 4 | 1.53 | 1.31 | 0.22 | 50.2 | 0.88 |

- Both weights work as intended: the motion follows a track change in proportion to w_music (0 = identical motion)
  and a requested genre in proportion to w_genre. They compete: a higher music weight lowers genre accuracy.
- **w_genre 2, w_music 1** keeps FID_k and gives genre accuracy 0.45 vs 0.34 (real 0.53) on both models; weights
  above 3 raise FID_k sharply. Next: confirm (2, 1) with `evaluate --genre-guidance 2 --music-guidance 1` on val and
  test before changing the default.
- No guidance setting increases beat locking: the music weight changes *what* moves, not *when*.

**Beat-shift calibration**: on 64 windows the accent detector barely separates real dances from chance (beat lock
0.088 on their own grid vs 0.053 against random grids; jittered frozen pose 0.037), and the shift's bootstrap
s.e. is ~0.26 beats, so a 64-window follow score is noise. Use all val windows (`--subset all --windows 1000`).

**Beat-grid phase is uncertain for 7 of 60 songs** (checked 2026-11-19): tempo = published BPM for all 60, but the
grid's phase is estimated from the audio, and for mHO2, mHO4, mKR1, mLH3, mLH4, mMH4, mWA5 the onsets are no
stronger on the grid than half a beat away (mHO2, mLH3, mLH4, mWA5 prefer the half-beat grid). Four are test songs
(mHO4, mLH4, mMH4, mWA5). A half-beat error flips the beat-phase input and corrupts BAS / the beat-shift probe for
those songs. Not fixed (decided to wait for the test results); possible check: the real dancers' accent phase per
song vs the other songs.

## 8. Operations

**AWS** (account 670682107687, us-east-1; role `SageMakerTraining-beat2body`; bucket
`beat2body-670682107687-us-east-1`; PyTorch 2.9 DLC pinned by digest in `aws.py`):

```bash
aws login                                                     # credentials expire; re-run when calls fail
python -m beat2body aws upload-data                            # after changing data
python -m beat2body aws submit --instance ml.g6.xlarge [--spot] --max-hours 4 --name <tag> -- <train options>
python -m beat2body aws progress [ids...] [--last 0]           # per-checkpoint training checks
python -m beat2body aws status <job>
python -m beat2body aws download <job> [--live]                # -> runs/<job>/
python -m beat2body aws pose --videos ids.txt --shards 4 --instances ml.g6.xlarge ml.g6.xlarge:spot ...
```

- Quotas: SageMaker training 1 instance per type per market (on-demand and spot separately) for
  ml.g6.xlarge / ml.g5.xlarge; ml.g4dn = 0; **no SageMaker P quota**. EC2: P spot 8 vCPU (no current P
  instance fits; p5.4xlarge needs 16), G 8 vCPU.
- Speeds: small model ~11–14 steps/s on g6 / g5 with `--amp` (30k steps ≈ 45 min); EDGE size ~5 steps/s
  on g5 (30k ≈ 1 h 45 min). Spot jobs can wait 10–30 min for capacity.
- `--select fid_k` / `val_loss` copy the best checkpoint to `model.pt`; `model_last.pt`, `model_raw.pt`
  (no EMA) are kept.

**Local:** `.venv` (Python 3.11); `ffmpeg` comes from imageio-ffmpeg (`.venv/bin/ffmpeg`; put `.venv/bin`
on PATH for `audio`, `clean`, `render`). In zsh, expand option variables with `${=VAR}`.

**Typical evaluation:**
```bash
python -m beat2body evaluate --checkpoint runs/<job>/model.pt --subset advanced --tag adv --device mps   # val
python -m beat2body evaluate --checkpoint runs/<job>/model.pt --split test --device mps
python -m beat2body report runs/<job> --checkpoint model.pt --split val --songs 4 --device mps
```
Do not rebuild clips (`clips`) while an evaluation runs: it rewrites `meta.csv`.

**Colab:** `scripts/colab_train.py` (paste into one cell; runs several trainings in parallel on one GPU;
results in Google Drive). Its `RUNS` still hold the older sweep; update before use.
