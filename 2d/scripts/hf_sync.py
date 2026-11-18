"""Download / upload the processed 2D data on Hugging Face (one dataset per pose tool).

The datasets mirror the repository layout under 2d/, so a download drops the files
exactly where pipeline.py and the training code expect them:

    2d/categories.json
    2d/keypoints/<tool>/<category>/<video_id>.npz, 2d/keypoints/<tool>/normalization.json
    2d/features/{audio_mel,audio_onset,beats}/<video_id>.npy
    2d/clips/<tool>/<split>/*.npz, 2d/clips/<tool>/meta.csv, 2d/clips/<tool>/split.json

Commands (run from the repository root):

    # e.g. on a training machine: get everything for one tool
    python 2d/scripts/hf_sync.py download --tool dwpose
    # only what is needed to rebuild the clips with pipeline.py clips
    python 2d/scripts/hf_sync.py download --tool dwpose --parts keypoints features

    # write the dataset card (README.md) locally to review it
    python 2d/scripts/hf_sync.py card --tool dwpose
    # upload keypoints, features, clips and the card; --prune deletes remote files
    # under those folders that no longer exist locally (e.g. clips that moved split)
    python 2d/scripts/hf_sync.py upload --tool dwpose --prune

Uploading needs a token with write access: `huggingface-cli login` or HF_TOKEN.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

sys.dont_write_bytecode = True

ROOT_2D = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REPO_ROOT = os.path.dirname(ROOT_2D)
TOOLS = ["dwpose", "mediapipe", "openpose"]
DEFAULT_ORG = "SAMBOOM"
CARD_DIR = os.path.join(ROOT_2D, "hf_cards")


def repo_id(org, tool):
    return f"{org}/AISTDance_{tool}"


def patterns(tool, parts):
    """Repo-relative glob patterns for the chosen parts of one tool's dataset."""
    p = ["2d/categories.json"]
    if "keypoints" in parts:
        p.append(f"2d/keypoints/{tool}/**")
    if "features" in parts:
        p.append("2d/features/**")
    if "clips" in parts:
        p.append(f"2d/clips/{tool}/**")
    return p


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
def run_download(args):
    from huggingface_hub import snapshot_download
    for tool in args.tool:
        rid = repo_id(args.org, tool)
        print(f"[{tool}] downloading {rid} ({', '.join(args.parts)}) into {REPO_ROOT}/2d")
        snapshot_download(rid, repo_type="dataset", local_dir=REPO_ROOT,
                          allow_patterns=patterns(tool, args.parts), revision=args.revision)
        n = len(glob.glob(os.path.join(ROOT_2D, "keypoints", tool, "*", "*.npz")))
        c = len(glob.glob(os.path.join(ROOT_2D, "clips", tool, "*", "*.npz")))
        print(f"[{tool}] {n} cleaned keypoint files, {c} clips on disk")


# ---------------------------------------------------------------------------
# dataset card
# ---------------------------------------------------------------------------
TOOL_INFO = {
    "dwpose": ("DWPose-L whole-body (ONNX), body joints mapped to COCO-17", "Apache-2.0", "0.3"),
    "openpose": ("official CMU OpenPose (BODY_25), mapped to COCO-17",
                 "academic / non-commercial research only", "0.1"),
    "mediapipe": ("MediaPipe Pose Landmarker (heavy, 33 landmarks), mapped to COCO-17", "Apache-2.0", "0.5"),
}

CITATION = """@inproceedings{tsuchida2019aist,
  author    = {Shuhei Tsuchida and Satoru Fukayama and Masahiro Hamasaki and Masataka Goto},
  title     = {AIST Dance Video Database: Multi-genre, Multi-dancer, and Multi-camera Database
               for Dance Information Processing},
  booktitle = {Proceedings of the 20th International Society for Music Information Retrieval
               Conference (ISMIR 2019)},
  year      = {2019}
}"""


def split_summary(tool):
    """Clip / video / song counts per split from the local meta.csv and split.json."""
    meta = os.path.join(ROOT_2D, "clips", tool, "meta.csv")
    if not os.path.exists(meta):
        sys.exit(f"{meta} not found; run pipeline.py clips --tool {tool} (or download the clips) first")
    with open(meta) as f:
        rows = list(csv.DictReader(f))
    info = {}
    sj = os.path.join(ROOT_2D, "clips", tool, "split.json")
    if os.path.exists(sj):
        with open(sj) as f:
            info = json.load(f)
    clips = Counter(r["split"] for r in rows)
    videos = {s: len({r["video_id"] for r in rows if r["split"] == s}) for s in clips}
    songs = {s: len({r.get("music") for r in rows if r["split"] == s}) for s in clips}
    return rows, info, clips, videos, songs


def make_card(tool):
    rows, info, clips, videos, songs = split_summary(tool)
    model, tool_license, conf = TOOL_INFO[tool]
    by = info.get("by", "video")
    window, stride = info.get("window", 240), info.get("stride", 60)
    cats = sorted({r["category"] for r in rows})
    split_text = {
        "music": "by **song**: within each genre, all videos danced to one song are in the same split "
                 "(4 / 1 / 1 songs per genre for train / val / test). Val and test music never "
                 "appears in training. Dancers do appear in more than one split.",
        "video": "by **video**, stratified per genre (15 / 3 / 3 videos per genre). Songs and dancers "
                 "are shared between splits, so test scores partly measure memorized music.",
        "dancer": "by **dancer**: each dancer's videos are in a single split. Songs are shared between splits.",
    }[by]
    table = "\n".join(f"| {s} | {clips.get(s, 0)} | {videos.get(s, 0)} | {songs.get(s, 0)} |"
                      for s in ["train", "val", "test"])
    return f"""---
pretty_name: AIST Dance 2D keypoints ({tool})
license: other
license_name: aist-dance-db-terms-of-use
license_link: https://aistdancedb.ongaaccel.jp/terms_of_use/
tags:
- dance
- pose-estimation
- music-to-dance
- motion-generation
- keypoints
size_categories:
- 1K<n<10K
---

# AIST Dance 2D keypoints ({tool})

2D skeletons and aligned music features extracted from the
[AIST Dance Video Database](https://aistdancedb.ongaaccel.jp/), for music-conditioned dance
generation (project: beat2body). **No video or audio is included**: only joint coordinates
and audio features (log-mel, onset strength, beat positions).

- Source videos: advanced choreography (`sFM`), front camera (`c01`), 10 genres x 21 videos
  ({", ".join(cats)}), resampled to 30 fps.
- Pose tool: {model}. Confidence threshold {conf}.
- Companion datasets with the same videos, split and format:
  {", ".join(f"[{t}](https://huggingface.co/datasets/{repo_id(DEFAULT_ORG, t)})" for t in TOOLS if t != tool)}.

## Files

```text
2d/categories.json                              {{"<genre>": index}} for the genre embedding
2d/keypoints/{tool}/<genre>/<video_id>.npz      keypoints (T,17,2) float32, valid (T,17) bool,
                                                pelvis (T,2) px, scale, fps, width, height, joint_names
2d/keypoints/{tool}/normalization.json          cleaning + normalization parameters
2d/features/audio_mel/<video_id>.npy            (T,64) log-mel (dB), 22.05 kHz, hop 735 (1 frame = 1 motion frame)
2d/features/audio_onset/<video_id>.npy          (T,) onset strength
2d/features/beats/<video_id>.npy                beat frame indices (librosa beat_track)
2d/clips/{tool}/meta.csv                        one row per clip: clip_path, audio_feat_path, category,
                                                start_frame, end_frame, split, video_id, dancer, music
2d/clips/{tool}/split.json                      split options and songs per split
2d/clips/{tool}/<split>/<genre>__<video_id>__<start>.npz
                                                {window}-frame windows (stride {stride}): keypoints, valid, mel,
                                                onset, beats (window-relative), category, category_index, ...
```

Keypoints are in COCO-17 order, centred on the mid-hip and divided by the median torso length
over the whole dataset. Undo with `xy_px = keypoints * scale + pelvis[:, None]`. `valid` is
False where a joint was interpolated rather than observed; use it as a loss mask.

Video ids follow AIST naming: `gLH_sFM_c01_d16_mLH0_ch01` = genre `LH`, dancer `d16`, song `mLH0`.

## Split

Split {split_text}

| split | clips | videos | songs |
| --- | --- | --- | --- |
{table}

## Loading

```python
from huggingface_hub import snapshot_download
import pandas as pd, numpy as np
root = snapshot_download("{repo_id(DEFAULT_ORG, tool)}", repo_type="dataset")
meta = pd.read_csv(f"{{root}}/2d/clips/{tool}/meta.csv")
clip = np.load(f"{{root}}/" + meta.clip_path[0])
```

## Terms and citation

The AIST Dance Video Database is free for **academic research only**; see its
[terms of use](https://aistdancedb.ongaaccel.jp/terms_of_use/). The videos themselves are not
redistributed here. The music features are derived from the database's copyrighted music.
Pose tool license: {tool_license}. If you use this data, cite the AIST Dance Video Database:

```bibtex
{CITATION}
```
"""


def card_path(tool):
    return os.path.join(CARD_DIR, f"{tool}.md")


def run_card(args):
    os.makedirs(CARD_DIR, exist_ok=True)
    for tool in args.tool:
        with open(card_path(tool), "w") as f:
            f.write(make_card(tool))
        print(f"[{tool}] wrote {os.path.relpath(card_path(tool), REPO_ROOT)}")


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------
def run_upload(args):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for tool in args.tool:
        rid = repo_id(args.org, tool)
        local = [p for p in patterns(tool, args.parts)]
        n_clips = len(glob.glob(os.path.join(ROOT_2D, "clips", tool, "*", "*.npz")))
        if "clips" in args.parts and n_clips < args.min_clips:
            sys.exit(f"[{tool}] only {n_clips} local clips (< --min-clips {args.min_clips}); refusing to upload")
        msg = args.message or f"Update {tool} data ({', '.join(args.parts)})"
        print(f"[{tool}] uploading to {rid}: {local}")
        api.upload_folder(repo_id=rid, repo_type="dataset", folder_path=REPO_ROOT,
                          allow_patterns=local, commit_message=msg)
        api.create_commit(rid, repo_type="dataset", commit_message="Update dataset card",
                          operations=[CommitOperationAdd("README.md", make_card(tool).encode())])
        if args.prune:
            prefixes = tuple(p[:-2] for p in local if p.endswith("/**"))
            remote = api.list_repo_files(rid, repo_type="dataset")
            stale = [f for f in remote if f.startswith(prefixes) and not os.path.exists(os.path.join(REPO_ROOT, f))]
            print(f"[{tool}] {len(stale)} stale remote files, e.g. {stale[:3]}")
            if stale:
                api.create_commit(rid, repo_type="dataset", commit_message="Remove files from earlier uploads",
                                  operations=[CommitOperationDelete(f) for f in stale])
        print(f"[{tool}] done: https://huggingface.co/datasets/{rid}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("download", run_download), ("card", run_card), ("upload", run_upload)]:
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        sp.add_argument("--tool", nargs="+", choices=TOOLS, default=["dwpose"])
        sp.add_argument("--org", default=DEFAULT_ORG)
        if name != "card":
            sp.add_argument("--parts", nargs="+", choices=["keypoints", "features", "clips"],
                            default=["keypoints", "features", "clips"])
        if name == "download":
            sp.add_argument("--revision", default=None, help="branch, tag or commit of the dataset")
        if name == "upload":
            sp.add_argument("--prune", action="store_true",
                            help="delete remote files under the uploaded folders that are not present locally")
            sp.add_argument("--min-clips", type=int, default=3000,
                            help="safety check: refuse to upload clips if fewer exist locally")
            sp.add_argument("--message", default=None)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
