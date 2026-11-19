# Training on Google Colab

Colab is used only as a GPU machine: every step is a `beat2body` command, no notebook code.
Results and resumable checkpoints go to Google Drive, so a disconnected session continues where it
stopped when the same cells are run again.

**Runtime → Change runtime type → GPU** (L4 or A100 with Colab Pro; T4 on the free tier works but is
slower). Then run these cells in order.

```python
# 1. Code (without the 2.1 GB of raw videos) and packages
!git clone -q --depth 1 --branch claude/relaxed-knuth-fc6tqf --filter=blob:none --sparse \
    https://github.com/novaz03/beat2body /content/beat2body
%cd /content/beat2body
!git sparse-checkout set --no-cone '/*' '!/2d/raw_video/'
!pip -q install -e ".[train]"
```

```python
# 2. Data from Hugging Face (public, no login): per-video keypoints + features + clip list (~150 MB)
!python -m beat2body hf download --tool dwpose --parts keypoints features meta
!python -m beat2body data          # check: shapes, no NaNs, 10 genres
```

```python
# 3. Google Drive for results and checkpoints
from google.colab import drive
drive.mount("/content/drive")
RUN = "/content/drive/MyDrive/beat2body_runs/small"
```

```python
# 4. Train. Small model (7.1M parameters, 20k steps):
!python -m beat2body train --amp --d-model 256 --ff 512 --layers 4 --heads 4 --steps 20000 \
    --val-every 2000 --sample-every 5000 --out {RUN} --checkpoint-dir {RUN}/ckpt
# Full EDGE size (46.9M, ~66k steps), in its own folder:
# !python -m beat2body train --amp --out /content/drive/MyDrive/beat2body_runs/edge \
#     --checkpoint-dir /content/drive/MyDrive/beat2body_runs/edge/ckpt
```

Smoke test of the full EDGE size first (300 steps, a few minutes): reports the GPU, speed
(`steps_per_s`) and peak memory (`gpu_mem_gb`):

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
!python -m beat2body train --amp --steps 300 --log-every 50 --val-every 300 --sample-every 300 \
    --sample-steps 10 --out /content/drive/MyDrive/beat2body_runs/edge_smoke
```

```python
# 5. Score it on the val songs, and generate + render one dance
!python -m beat2body evaluate --checkpoint {RUN}/model.pt
!python -m beat2body sample --checkpoint {RUN}/model.pt --video-id gLH_sFM_c01_d16_mLH4_ch05 --device cuda
!python -m beat2body render --video-id gLH_sFM_c01_d16_mLH4_ch05 \
    --keypoints {RUN}/generated/gLH_sFM_c01_d16_mLH4_ch05.npz -o {RUN}/gLH_mLH4.mp4
```

Notes
- Training prints one JSON line per 100 steps (`loss`, `steps_per_s`), validation every `--val-every`
  steps (`val_loss`), and generated-motion checks every `--sample-every` steps (`speed_ratio` near 1,
  `contact_rate` near 0.4). Everything is also written to `{RUN}/metrics.jsonl`.
- If the session disconnects, rerun cells 1-4 unchanged: training resumes from `{RUN}/ckpt/last.pt`
  (saved every `--val-every` steps).
- `--amp` (bfloat16) needs an L4 / A100 / newer GPU; drop it on a T4.
- `render` writes a stick-figure video (`{RUN}/gLH_mLH4.mp4`; no audio, since the raw videos are not
  cloned). `gLH_sFM_c01_d16_mLH4_ch05` uses song mLH4, which is in the held-out test set.
