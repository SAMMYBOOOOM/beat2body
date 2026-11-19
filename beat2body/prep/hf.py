"""Download / upload the processed data on Hugging Face (one dataset per pose tool).

The datasets have the same layout as the data root, under "2d/":

    2d/categories.json
    2d/keypoints/<tool>/<category>/<video_id>.npz, 2d/keypoints/<tool>/normalization.json
    2d/features/{audio_mel,audio_onset,beats}/<video_id>.npy
    2d/clips/<tool>/<split>/*.npz, 2d/clips/<tool>/meta.csv, 2d/clips/<tool>/split.json

    python -m beat2body hf download --tool dwpose                          # e.g. on a training machine
    python -m beat2body hf download --tool dwpose --parts keypoints features   # to rebuild clips
    python -m beat2body hf card --tool dwpose                              # write the card to review it
    python -m beat2body hf upload --tool dwpose --prune                    # data + card

Account (namespace) of the datasets, first match wins:
    --org NAME  >  BEAT2BODY_HF_ORG  >  the logged-in account (`hf auth login` or HF_TOKEN)  >  SizheZ03
SizheZ03 holds the current public datasets, so downloads work without logging in. (SAMBOOM has
the original datasets with the old per-video split.) `upload` creates
the dataset in your account if needed, private unless --public. Private datasets need a token
to download as well (e.g. HF_TOKEN on the training machine).
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

from beat2body import paths
from beat2body.paths import SPLITS, TOOLS

DOC = __doc__
FALLBACK_ORG = "SizheZ03"  # current public datasets (read-only fallback when not logged in)
ORG_ENV = "BEAT2BODY_HF_ORG"
PARTS = ("keypoints", "features", "clips")
DOWNLOAD_PARTS = PARTS + ("meta",)   # meta: only clips/<tool>/meta.csv + split.json (what training needs)


def repo_id(org, tool):
    return f"{org}/AISTDance_{tool}"


def logged_in_user():
    """Name of the account of the current token, or None."""
    try:
        from huggingface_hub import HfApi, get_token
        return HfApi().whoami()["name"] if get_token() else None
    except Exception:  # noqa: BLE001  (offline, invalid token, ...)
        return None


def resolve_org(org=None, for_upload=False):
    """--org > BEAT2BODY_HF_ORG > logged-in account > SAMBOOM (download only)."""
    org = org or os.environ.get(ORG_ENV) or logged_in_user()
    if org:
        return org
    if for_upload:
        sys.exit(f"no Hugging Face account: run `hf auth login` (or set HF_TOKEN), or pass --org / {ORG_ENV}")
    return FALLBACK_ORG


def add_args(p):
    sub = p.add_subparsers(dest="hf_cmd", required=True)
    for name, fn in [("download", download), ("card", card), ("upload", upload)]:
        sp = sub.add_parser(name)
        sp.set_defaults(hf_fn=fn)
        sp.add_argument("--tool", nargs="+", choices=TOOLS, default=["dwpose"])
        sp.add_argument("--org", default=None,
                        help=f"dataset namespace (default: ${ORG_ENV}, else your logged-in account, "
                             f"else {FALLBACK_ORG} for downloads)")
        if name != "card":
            sp.add_argument("--parts", nargs="+", choices=DOWNLOAD_PARTS if name == "download" else PARTS,
                            default=list(PARTS), help="for training: keypoints features meta")
        if name == "download":
            sp.add_argument("--revision", default=None, help="branch, tag or commit of the dataset")
        if name == "upload":
            sp.add_argument("--prune", action="store_true",
                            help="delete remote files under the uploaded folders that are not present locally")
            sp.add_argument("--min-clips", type=int, default=3000,
                            help="safety check: refuse to upload clips if fewer exist locally")
            sp.add_argument("--message", default=None)
            sp.add_argument("--public", action="store_true",
                            help="make a newly created dataset public (default: private); "
                                 "an existing dataset keeps its visibility")


def run(args):
    args.hf_fn(args)


def _prefix():
    name = os.path.basename(paths.data_root())
    if name != "2d":
        sys.exit(f"the data root must be a folder named '2d' to match the datasets (got {paths.data_root()})")
    return name


def patterns(tool, parts):
    d = _prefix()
    p = [f"{d}/categories.json"]
    if "keypoints" in parts:
        p.append(f"{d}/keypoints/{tool}/**")
    if "features" in parts:
        p.append(f"{d}/features/**")
    if "clips" in parts:
        p.append(f"{d}/clips/{tool}/**")
    elif "meta" in parts:
        p += [f"{d}/clips/{tool}/meta.csv", f"{d}/clips/{tool}/split.json"]
    return p


def download(args):
    from huggingface_hub import snapshot_download
    org = resolve_org(args.org)
    for tool in args.tool:
        rid = repo_id(org, tool)
        print(f"[{tool}] downloading {rid} ({', '.join(args.parts)}) into {paths.data_root()}")
        snapshot_download(rid, repo_type="dataset", local_dir=paths.base_dir(),
                          allow_patterns=patterns(tool, args.parts), revision=args.revision)
        n = len(glob.glob(os.path.join(paths.keypoints_dir(), tool, "*", "*.npz")))
        c = len(glob.glob(os.path.join(paths.tool_clips_dir(tool), "*", "*.npz")))
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


def _split_text(info):
    by = info.get("by", "video")
    unit = {"video": "video", "dancer": "dancer", "music": "song"}[by]
    n = info.get("test_songs", 0)
    if n:
        return (f"**Test** holds out {n} song(s) per genre completely (all their videos), so test music is "
                f"never seen in training; use it for the final evaluation. The other videos are split into "
                f"**train / val by {unit}** ({int(round(100 * info.get('val_ratio', 0.15)))}% val), so train and "
                "val share songs and dancers.")
    return f"Split by {unit}, stratified per genre."


def make_card(tool, org):
    meta = paths.meta_csv_path(tool)
    if not os.path.exists(meta):
        sys.exit(f"{meta} not found; run `python -m beat2body clips --tool {tool}` (or download the clips) first")
    with open(meta) as f:
        rows = list(csv.DictReader(f))
    info = {}
    if os.path.exists(paths.split_json_path(tool)):
        with open(paths.split_json_path(tool)) as f:
            info = json.load(f)
    clips = Counter(r["split"] for r in rows)
    videos = {s: len({r["video_id"] for r in rows if r["split"] == s}) for s in SPLITS}
    songs = {s: len({r.get("music") for r in rows if r["split"] == s}) for s in SPLITS}
    model, tool_license, conf = TOOL_INFO[tool]
    window, stride = info.get("window", 240), info.get("stride", 60)
    cats = sorted({r["category"] for r in rows})
    table = "\n".join(f"| {s} | {clips.get(s, 0)} | {videos[s]} | {songs[s]} |" for s in SPLITS)
    others = ", ".join(f"[{t}](https://huggingface.co/datasets/{repo_id(org, t)})" for t in TOOLS if t != tool)
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
- Companion datasets with the same videos, split and format: {others}.

## Files

```text
2d/categories.json                              {{"<genre>": index}} for the genre embedding
2d/keypoints/{tool}/<genre>/<video_id>.npz      keypoints (T,17,2) float32, valid (T,17) bool,
                                                pelvis (T,2) px, scale, fps, width, height, joint_names
2d/keypoints/{tool}/normalization.json          cleaning + normalization parameters
2d/features/audio_mel/<video_id>.npy            (T,64) log-mel (dB), 22.05 kHz, hop 735 (1 frame = 1 motion frame)
2d/features/audio_onset/<video_id>.npy          (T,) onset strength
2d/features/beats/<video_id>.npy                beat times in frames (float): grid at the published AIST tempo,
                                                phase fitted to the onset envelope
2d/features/audio_edge/<video_id>.npy           (T,35) EDGE's "baseline" music features at 30 fps: onset envelope (1),
                                                MFCC (20), chroma CENS (12), onset peaks one-hot (1), beats one-hot (1)
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

## How the training code uses it

The beat2body model is a 2D re-implementation of EDGE (Tseng et al., CVPR 2023). It does not read
the clip files; it reads the per-video files above and uses `meta.csv` only for the windows and the
split. Per frame it builds:

- **motion (38):** 2 ankle contacts + 2 root + 34 joints. Root = `(pelvis - image centre) / scale`
  (the dancer's position in the 1920x1080 frame, in torso lengths). Joints = `keypoints` (pelvis-centred).
  An ankle is in contact when its image speed, root motion included, is below 0.012 torso lengths
  per frame (about 1.5 px; about 40% of frames).
- **music (37):** `audio_edge` (35), normalized with train-split mean / std, + sin / cos of the
  position within the beat from `beats` (2).
- **windows:** 150 frames (5 s) cropped from the clips of each split.

Code: `beat2body.data` in the project repository; `python -m beat2body data` checks it.

Video ids follow AIST naming: `gLH_sFM_c01_d16_mLH0_ch01` = genre `LH`, dancer `d16`, song `mLH0`.

## Split

{_split_text(info)}

| split | clips | videos | songs |
| --- | --- | --- | --- |
{table}

## Loading

```python
from huggingface_hub import snapshot_download
import pandas as pd, numpy as np
root = snapshot_download("{repo_id(org, tool)}", repo_type="dataset")
meta = pd.read_csv(f"{{root}}/2d/clips/{tool}/meta.csv")
clip = np.load(f"{{root}}/" + meta.clip_path[0])
```

Or, with the project code: `python -m beat2body hf download --tool {tool}`.

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
    return os.path.join(paths.data_root(), "hf_cards", f"{tool}.md")


def card(args):
    org = resolve_org(args.org)
    for tool in args.tool:
        os.makedirs(os.path.dirname(card_path(tool)), exist_ok=True)
        with open(card_path(tool), "w") as f:
            f.write(make_card(tool, org))
        print(f"[{tool}] wrote {paths.rel(card_path(tool))}")


def upload(args):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    org = resolve_org(args.org, for_upload=True)
    for tool in args.tool:
        rid = repo_id(org, tool)
        pats = patterns(tool, args.parts)
        n_clips = len(glob.glob(os.path.join(paths.tool_clips_dir(tool), "*", "*.npz")))
        if "clips" in args.parts and n_clips < args.min_clips:
            sys.exit(f"[{tool}] only {n_clips} local clips (< --min-clips {args.min_clips}); refusing to upload")
        url = api.create_repo(rid, repo_type="dataset", private=not args.public, exist_ok=True)
        private = api.repo_info(rid, repo_type="dataset").private
        print(f"[{tool}] uploading to {url} ({'private' if private else 'public'}): {pats}")
        api.upload_folder(repo_id=rid, repo_type="dataset", folder_path=paths.base_dir(), allow_patterns=pats,
                          commit_message=args.message or f"Update {tool} data ({', '.join(args.parts)})")
        api.create_commit(rid, repo_type="dataset", commit_message="Update dataset card",
                          operations=[CommitOperationAdd("README.md", make_card(tool, org).encode())])
        if args.prune:
            prefixes = tuple(p[:-2] for p in pats if p.endswith("/**"))
            remote = api.list_repo_files(rid, repo_type="dataset")
            stale = [f for f in remote
                     if f.startswith(prefixes) and not os.path.exists(os.path.join(paths.base_dir(), f))]
            print(f"[{tool}] {len(stale)} stale remote files, e.g. {stale[:3]}")
            if stale:
                api.create_commit(rid, repo_type="dataset", commit_message="Remove files from earlier uploads",
                                  operations=[CommitOperationDelete(f) for f in stale])
        print(f"[{tool}] done: https://huggingface.co/datasets/{rid}")
