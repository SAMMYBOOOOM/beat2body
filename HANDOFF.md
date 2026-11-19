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

**Videos:** `reports/2026-11-19_progress/` (model A, first model; committed) and
`reports/2026-11-19_run21/compare_*.mp4` (real | model A | run 2.1, 20 s with music; not committed).
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
