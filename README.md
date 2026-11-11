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
├── 2d/ # 2D skeleton pipeline, model, training (current focus)  
│ ├── README.md  
│ ├── audio/ # extracted .wav per video (hiphop/, house/)  
│ ├── clips/ # sliding-window clips (train/, val/, test/)  
│ ├── features/ # audio_mel/, audio_onset/, beats/  
│ ├── frames/ # extracted video frames (can be deleted after keypoints)  
│ ├── keypoints/ # (T, J, 3) joints per video (hiphop/, house/)  
│ └── raw_video/ # source videos (hiphop/, house/)  
└── 3d/ # future 3D extension
```

## Status
- [x] Repo skeleton
- [ ] 2D data pipeline (frame extraction → pose → cleaning → audio features)
- [ ] 2D baseline model (audio encoder + category embedding + GRU decoder)
- [ ] 2D training + evaluation (beat alignment, motion diversity)
- [ ] 3D phase

## Resources

[Data Formats - AIST Dance Video Database (AIST Dance DB)](https://aistdancedb.ongaaccel.jp/data_formats/)


### Datasets

SAMBOOM/AISTDance_dwpose
SAMBOOM/AISTDance_mediapipe
SAMBOOM/AISTDance_openpose