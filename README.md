# beat2body (Music2Dance)

Music-conditioned dance motion generation: given an audio clip and a dance
category, generate a plausible skeleton sequence.

| Phase  | Output                                   | Status     | Docs                                       |
| ------ | ---------------------------------------- | ---------- | ------------------------------------------ |
| **2D** | `(T, J, 2)` joints + stick-figure `.mp4` | 🚧 Active  | **[docs/pipeline_2d.md](docs/pipeline_2d.md)** |
| **3D** | `(T, J, 3)` joints                       | ⏳ Planned | [docs/3d.md](docs/3d.md)                   |

👉 **Read [docs/pipeline_2d.md](docs/pipeline_2d.md)**: data pipeline, baseline architecture,
training losses, data format and split. The task list is in [NEXT_STEPS.md](NEXT_STEPS.md).

## Setup

```bash
pip install -e ".[train]"         # training machine: torch, huggingface_hub, opencv
pip install -e ".[train,audio,dev]"  # + audio features (librosa) and tests
python -m beat2body --help
```

Videos (pose, audio, render) also need `ffmpeg` on the PATH.

## Data

Processed data is on Hugging Face, one public dataset per pose tool, with the same layout as `2d/`:

| Dataset | Pose tool |
| --- | --- |
| [`SizheZ03/AISTDance_dwpose`](https://huggingface.co/datasets/SizheZ03/AISTDance_dwpose) | DWPose (used for training) |
| [`SizheZ03/AISTDance_openpose`](https://huggingface.co/datasets/SizheZ03/AISTDance_openpose) | OpenPose |
| [`SizheZ03/AISTDance_mediapipe`](https://huggingface.co/datasets/SizheZ03/AISTDance_mediapipe) | MediaPipe |

The account is taken from `--org`, else `BEAT2BODY_HF_ORG`, else the logged-in account
(`hf auth login` or `HF_TOKEN`); The original public
copies (old per-video split) are under `SAMBOOM/`.

```bash
export HF_TOKEN=...  BEAT2BODY_HF_ORG=SizheZ03     # e.g. on AWS
python -m beat2body hf download --tool dwpose      # into 2d/, no videos or GPU needed
python -m beat2body data --tool dwpose             # check the PyTorch loader
python -m beat2body clips --tool dwpose            # optional: rebuild clips / split locally
python -m beat2body render --video-id gLH_sFM_c01_d16_mLH0_ch01
```

Split: train / val by video (shared songs); **one song per genre is held out as the test set**
for the final evaluation on unseen music.

The videos come from the [AIST Dance Video Database](https://aistdancedb.ongaaccel.jp/)
([data formats](https://aistdancedb.ongaaccel.jp/data_formats/)), which is free for academic
research only and must be cited (Tsuchida et al., ISMIR 2019); see its
[terms of use](https://aistdancedb.ongaaccel.jp/terms_of_use/).

## Repository layout

```text
.
├── beat2body/              # Python package; run as `python -m beat2body <command>`
│   ├── cli.py              # command line
│   ├── paths.py            # data folder layout (default data root: 2d/)
│   ├── aist.py             # AIST naming: genres, dancers, songs
│   ├── skeleton.py         # COCO-17 joints, tool joint mappings
│   ├── video.py            # fixed-fps frame reading, drawing, mp4 writing
│   ├── data.py             # PyTorch Dataset / DataLoader for training
│   ├── model/              # EDGE-style diffusion (edge.py, diffusion.py), GRU baseline, Adan
│   ├── train.py            # training (local, Colab, SageMaker), `sample`
│   ├── evaluate.py         # FID_k/p, diversity, BAS, PFC, speed, BeatPow, drift detector
│   ├── report.py           # shareable videos of a run
│   ├── aws.py              # SageMaker: upload-data, submit, status, download
│   ├── pose/               # 4.3 pose tools: dwpose.py, mediapipe.py, openpose.py
│   ├── prep/               # 4.2/4.4-4.7: frames, clean, audio, clips (+ split), all, hf
│   └── render.py           # stick-figure videos of real or generated keypoints
├── tests/                  # pytest
├── scripts/                # colab_train.py (one-cell Colab trainer), colab_build_openpose.sh
├── docs/                   # pipeline_2d.md, colab.md, 3d.md
├── reports/                # progress reports with metrics, charts and generated videos
├── 2d/                     # data root
│   ├── raw_video/<genre>/  # AIST source videos (10 genres x 21, tracked in git)
│   ├── categories.json     # genre -> embedding index (tracked)
│   └── keypoints/ features/ clips/ audio/ frames/   # generated, ignored by git
├── pyproject.toml
└── NEXT_STEPS.md           # status and task list
```

Development: code goes in the package as modules with a CLI entry point, not in notebooks.
Run `ruff check beat2body` and `pytest` before committing.

## Status
- [x] 2D data pipeline (pose → cleaning → audio features → clips), 10 genres × 3 pose tools, on Hugging Face
- [x] Code organized as the `beat2body` package; held-out-song test split
- [x] New split + dataset cards published (public, SizheZ03); beat labels fixed (published AIST tempos)
- [x] PyTorch data loader (`beat2body.data`)
- [x] 2D model: EDGE replica (diffusion transformer) + GRU baseline
- [x] 2D training (AWS SageMaker, Colab) + evaluation (FID, diversity, BAS, PFC, speed, BeatPow, drift)
- [x] Body-size drift fixed (per-video scale + projected bone-length loss)
- [ ] Energy: generated dances miss movement on the beat; fixes being tested
  ([latest report](reports/2026-11-19_progress/README.md))
- [ ] More data: AIST basic dances (1,200 videos); genre-accuracy metric; full-size model
- [ ] 3D phase
