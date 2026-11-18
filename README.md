# Music2Dance

Music-conditioned dance motion generation: given an audio clip and a dance
category, generate a plausible skeleton sequence.

The project is organized in two phases. **Only the 2D phase is actively being
developed right now.**

| Phase  | Output                                   | Status     | Docs                                 |
| ------ | ---------------------------------------- | ---------- | ------------------------------------ |
| **2D** | `(T, J, 2)` joints + stick-figure `.mp4` | 🚧 Active  | **[2d →](2d/README.md)**             |
| **3D** | `(T, J, 3)` joints                       | ⏳ Planned | [3d →](3d/README.md) *(placeholder)* |

## Start here

👉 **Read [2d/README.md](2d/README.md)** — data pipeline, baseline
architecture, training losses, and output format.

## Repository layout

```text
.
├── 2d/                       # 2D skeleton pipeline, model, training (current focus)
│   ├── README.md             # pipeline, model and data format: start here
│   ├── run_pose_tools_colab.ipynb   # full pose + data run on Colab
│   ├── scripts/              # pose_*.py, pipeline.py, hf_sync.py
│   ├── categories.json       # genre -> embedding index (10 genres)
│   ├── raw_video/<genre>/    # AIST source videos (10 genres x 21)
│   ├── keypoints/<tool>/     # cleaned (T, 17, 2) joints per video   } generated, ignored by git,
│   ├── features/             # audio_mel/, audio_onset/, beats/      } published on Hugging Face
│   ├── clips/<tool>/         # 240-frame clips, meta.csv, split.json }
│   ├── audio/, frames/       # intermediate files
│   └── test_code/            # exploration notebooks
├── 3d/                       # future 3D extension
└── NEXT_STEPS.md             # current status and task list
```

## Status
- [x] Repo skeleton
- [x] 2D data pipeline (pose → cleaning → audio features → clips), 10 genres × 3 pose tools, on Hugging Face
- [x] Song-disjoint train / val / test split (val and test music unseen in training)
- [ ] 2D baseline model (audio encoder + category embedding + GRU decoder)
- [ ] 2D training + evaluation (beat alignment, motion diversity), on AWS
- [ ] 3D phase

See [NEXT_STEPS.md](NEXT_STEPS.md) for the task list.

## Data

Processed data is on Hugging Face, one dataset per pose tool, with the same layout as `2d/`:

| Dataset | Pose tool |
| --- | --- |
| [`SAMBOOM/AISTDance_dwpose`](https://huggingface.co/datasets/SAMBOOM/AISTDance_dwpose) | DWPose (used for training) |
| [`SAMBOOM/AISTDance_openpose`](https://huggingface.co/datasets/SAMBOOM/AISTDance_openpose) | OpenPose |
| [`SAMBOOM/AISTDance_mediapipe`](https://huggingface.co/datasets/SAMBOOM/AISTDance_mediapipe) | MediaPipe |

```bash
pip install numpy huggingface_hub
python 2d/scripts/hf_sync.py download --tool dwpose
```

The videos come from the [AIST Dance Video Database](https://aistdancedb.ongaaccel.jp/)
([data formats](https://aistdancedb.ongaaccel.jp/data_formats/)), which is free for academic
research only and must be cited (Tsuchida et al., ISMIR 2019); see its
[terms of use](https://aistdancedb.ongaaccel.jp/terms_of_use/).
