# Beat2Body evaluation of runs 2.1–2.3

November 19, 2026. All four training jobs completed 30,000 updates. This evaluation uses each
run's lowest-validation-loss checkpoint, which keeps checkpoint selection consistent. Run 2.1's
26,000-step snapshot is stored as `model_vl.pt` because its original training job selected by kinetic FID.
Both 2.1 rows come from its previously saved full evaluation. Runs 2.1b, 2.2, and 2.3 were evaluated afresh.

| Run | SageMaker job ID | Checkpoint | Recipe | Body scaling |
|---|---|---|---|---|
| 2.1 | `6c8ccdc6` | `model_vl.pt`, 26,000 | 7.1M regularized, bone 1.0, energy 0.25 | Dancer |
| 2.1b | `f9ed5641` | `model.pt`, 18,000 | 7.1M EDGE recipe, bone 1.0, energy 0.25 | Dancer |
| 2.2 | `0fc7c51e` | `model.pt`, 20,000 | 46.9M EDGE recipe, bone and energy losses off | Dataset |
| 2.3 | `eb0ffe3f` | `model.pt`, 16,000 | 7.1M EDGE recipe, bone and energy losses off | Dataset |

The fresh evaluations used commit `403c883d2b9db0abd214c9b71e221a8aebb9ab58`, seed 0, 50 DDIM steps,
eta 1, joint classifier-free guidance 2, batch size 64, and local Apple MPS in float32.
Run 2.1's archived files use the checkpoint's default guidance and omit the other sampling flags.
All four runs use the DWPose split already present in the repo. The advanced
validation split has 440 five-second windows. The test split has 769 five-second windows covering
advanced and basic dances on 10 held-out songs. The basic subset can contain routines also
performed to training tracks. The 20-second body-variation check uses one
eligible video per genre on each split. The evaluator uses each checkpoint's motion scaling
for its reference. Comparisons between dancer-scale and dataset-scale runs include that difference.

Lower FID and PFC are preferable. Diversity and BoneCV should be read against each run's real
reference. A speed or BeatPow ratio of 1 matches real motion. BeatPow measures 1–4 Hz motion
power. The legacy BAS has known limitations for estimating synchronization.

[CSV with all metrics, checkpoint hashes, and real references](summary.csv)

## Advanced validation (440 windows)

| Run | Kinetic FID | Pose FID | Kinetic div. | Pose div. | BAS | PFC | Speed | BeatPow | BoneCV | Torso CV (20 s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [2.1](metrics/run_2.1_val.json) | 36.95 | 67.33 | 10.91 | 15.61 | 0.223 | 0.054 | 1.194 | 1.712 | 0.283 | 0.186 |
| [2.1b](metrics/run_2.1b_val.json) | 27.40 | 68.42 | 11.18 | 17.35 | 0.233 | 0.055 | 1.039 | 1.150 | 0.254 | 0.186 |
| [2.2](metrics/run_2.2_val.json) | 27.04 | 26.82 | 7.71 | 13.05 | 0.219 | 0.042 | 0.640 | 0.412 | 0.189 | 0.079 |
| [2.3](metrics/run_2.3_val.json) | 45.50 | 30.23 | 7.30 | 13.75 | 0.207 | 0.053 | 0.533 | 0.210 | 0.185 | 0.077 |

## Held-out-song test (769 windows)

| Run | Kinetic FID | Pose FID | Kinetic div. | Pose div. | BAS | PFC | Speed | BeatPow | BoneCV | Torso CV (20 s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [2.1](metrics/run_2.1_test.json) | 22.52 | 41.85 | 7.09 | 12.41 | 0.220 | 0.050 | 1.192 | 1.314 | 0.296 | 0.223 |
| [2.1b](metrics/run_2.1b_test.json) | 28.61 | 42.24 | 6.56 | 12.97 | 0.221 | 0.050 | 1.127 | 0.977 | 0.273 | 0.178 |
| [2.2](metrics/run_2.2_test.json) | 25.01 | 15.57 | 5.28 | 9.95 | 0.226 | 0.037 | 0.719 | 0.446 | 0.202 | 0.101 |
| [2.3](metrics/run_2.3_test.json) | 33.86 | 22.07 | 6.29 | 10.88 | 0.213 | 0.056 | 0.555 | 0.205 | 0.178 | 0.078 |

## Observations

- The two runs with the bone and energy losses, 2.1 and 2.1b, reached 1.31 and 0.98 times real beat-band power on the test split. Their kinetic distances were 22.52 and 28.61, and their 20-second torso variation was 0.223 and 0.178. The matched real torso reference is 0.105.
- Within the dataset-scaled EDGE recipe, the 46.9M model (2.2) had test kinetic and pose distances of 25.01 and 15.57. The 7.1M model (2.3) scored 33.86 and 22.07. Their respective power ratios were 0.45 and 0.21.

## Reproduce

From the repository root, set `checkpoint` using the file name and `local_run_directory` in the CSV.
The 2.1b run was evaluated under its SageMaker job name and is now stored at
`runs/beat2body-diffusion-c-small-co-best-so-far/` with the same checkpoint hash.

```bash
python -m beat2body evaluate --checkpoint "$checkpoint" --split val --subset advanced \
  --seed 0 --steps 50 --eta 1 --guidance 2 --batch-size 64 --long-seconds 20 --tag four_run_full --device mps
python -m beat2body evaluate --checkpoint "$checkpoint" --split test --subset all \
  --seed 0 --steps 50 --eta 1 --guidance 2 --batch-size 64 --long-seconds 20 --tag four_run_full --device mps
```

The SHA-256 of `2d/clips/dwpose/meta.csv` is
`f5250202bef543f813bee70ba8de9d82a7e870f5b353164ef0b756b78f6c72ad`.
