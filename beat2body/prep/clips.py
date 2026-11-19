"""4.6 Alignment + 4.7 Dataset split: sliding-window clips and meta.csv.

Input: the cleaned keypoints of one tool (4.4) and the audio features (4.5). The raw
videos are NOT needed, so this can be rerun on data downloaded from Hugging Face.

Split (4.7), stratified per category so every split contains every category:
  test        --test-songs N (default 1) songs per genre are held out completely:
              all their videos go to test and nothing of that music is seen in training.
              This is the final evaluation on new music.
  train / val the remaining videos, grouped by --by and split with --val-ratio:
                video  (default) each video on its own; songs and dancers are shared
                       between train and val, so val checks fit, not new-music generalization
                dancer all videos of one dancer together
                music  all videos of one song together
With --test-songs 0 the test split is taken from the same groups with --test-ratio.
The split is fixed by --seed and recorded in meta.csv and split.json.

Alignment (4.6): motion frame t and audio feature frame t refer to the same time (both
use the same fixed fps), so both are trimmed to the shorter length, then each video is
cut into overlapping windows.

Output:
    <data>/clips/<tool>/<split>/<category>__<video_id>__<start>.npz
        keypoints (W, 17, 2), valid (W, 17), mel (W, 64), onset (W,),
        beats (beat times in frames inside the window, float), category, category_index,
        video_id, start, end, fps
    <data>/clips/<tool>/meta.csv
        clip_path, audio_feat_path, category, start_frame, end_frame, split, video_id, dancer, music
    <data>/clips/<tool>/split.json   options used and the songs in each split
    <data>/categories.json           {"<category>": index}, fixed mapping for the category embedding

    python -m beat2body clips --tool dwpose
    python -m beat2body clips --tool dwpose --test-songs 0 --by video   # plain per-video split
"""
import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

from beat2body import aist, paths
from beat2body.paths import SPLITS, TOOLS

DOC = __doc__
SPLIT_KEYS = {"video": lambda v: v, "dancer": aist.dancer_id, "music": aist.music_id}
META_FIELDS = ["clip_path", "audio_feat_path", "category", "start_frame", "end_frame",
               "split", "video_id", "dancer", "music"]


def add_args(p, with_tool=True):
    if with_tool:
        p.add_argument("--tool", required=True, choices=TOOLS)
        p.add_argument("--categories", nargs="*", default=None)
        p.add_argument("--limit", type=int, default=None, help="at most N videos per category")
    p.add_argument("--window", type=int, default=240, help="clip length in frames (README: 120-300)")
    p.add_argument("--stride", type=int, default=60)
    p.add_argument("--min-length", type=int, default=120,
                   help="videos shorter than --window but at least this long become one shorter clip")
    p.add_argument("--min-valid", type=float, default=0.5,
                   help="drop clips where less than this fraction of joints was observed")
    p.add_argument("--test-songs", type=int, default=1,
                   help="songs per genre held out completely as the test set (0: use --test-ratio)")
    p.add_argument("--by", choices=list(SPLIT_KEYS), default="video",
                   help="unit kept within one split for train / val (default: video)")
    p.add_argument("--val-ratio", type=float, default=0.15, help="share of the non-test units for val")
    p.add_argument("--test-ratio", type=float, default=0.15, help="only with --test-songs 0")
    p.add_argument("--seed", type=int, default=0)


def allocate(n, ratios):
    """Counts per split that sum to n; a split with a positive ratio gets >= 1 unit when n >= 3."""
    counts = [int(round(n * r)) for r in ratios]
    counts[0] = n - sum(counts[1:])
    if n >= 3:
        for i in (1, 2):
            if ratios[i] > 0 and counts[i] == 0:
                counts[i], counts[0] = 1, counts[0] - 1
    return counts


def split_videos(videos, by="video", seed=0, test_songs=1, val_ratio=0.15, test_ratio=0.15):
    """Advanced dances (and non-AIST videos) are split exactly as before (`_split_main`), so adding
    basic dances never changes their split. Basic dances (sBM) follow it: a basic dance on a held-out
    test song goes to test; the rest are split into train / val by routine (all tempo versions of one
    dancer's routine stay together), with the same val ratio."""
    main = [(c, v) for c, v in videos if aist.situation(v) != "sBM"]
    basic = [(c, v) for c, v in videos if aist.situation(v) == "sBM"]
    splits = _split_main(main, by, seed, test_songs, val_ratio, test_ratio) if main else {}
    if not basic:
        return splits
    rng = random.Random(seed + 1)
    by_cat = defaultdict(list)
    for cat, vid in basic:
        by_cat[cat].append(vid)
    for cat, vids in sorted(by_cat.items()):
        held = {aist.music_id(v) for (c, v), x in splits.items() if c == cat and x == "test"}
        rest = []
        for v in sorted(vids):
            if aist.music_id(v) in held:
                splits[(cat, v)] = "test"
            else:
                rest.append(v)
        groups = defaultdict(list)
        for v in rest:
            groups[aist.routine_id(v)].append(v)
        units = [g for _, g in sorted(groups.items())]
        rng.shuffle(units)
        i = 0
        for split, c in zip(SPLITS, allocate(len(units), [1 - val_ratio, val_ratio, 0.0])):
            for unit in units[i:i + c]:
                for v in unit:
                    splits[(cat, v)] = split
            i += c
    return splits


def _split_main(videos, by="video", seed=0, test_songs=1, val_ratio=0.15, test_ratio=0.15):
    """videos: [(category, video_id)] -> {(category, video_id): split}.

    Per category: `test_songs` whole songs go to test; the remaining videos are grouped by
    SPLIT_KEYS[by] and whole groups are assigned to train / val (and test if test_songs == 0)."""
    key = SPLIT_KEYS[by]
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for cat, vid in videos:
        by_cat[cat].append(vid)
    splits = {}
    for cat, vids in sorted(by_cat.items()):
        vids = sorted(vids)
        if test_songs > 0:
            songs = sorted({aist.music_id(v) or v for v in vids})
            if len(songs) <= test_songs:
                sys.exit(f"{cat}: only {len(songs)} songs, cannot hold out {test_songs} for test")
            held = set(rng.sample(songs, test_songs))
            for v in vids:
                if (aist.music_id(v) or v) in held:
                    splits[(cat, v)] = "test"
            vids = [v for v in vids if (cat, v) not in splits]
            ratios = [1 - val_ratio, val_ratio, 0.0]
        else:
            ratios = [1 - val_ratio - test_ratio, val_ratio, test_ratio]
        groups = defaultdict(list)
        for v in vids:
            groups[key(v) or v].append(v)
        units = [g for _, g in sorted(groups.items())]
        rng.shuffle(units)
        i = 0
        for split, c in zip(SPLITS, allocate(len(units), ratios)):
            for unit in units[i:i + c]:
                for v in unit:
                    splits[(cat, v)] = split
            i += c
    return splits


def list_clean_videos(tool, categories=None, limit=None):
    """[(category, video_id)] that have cleaned keypoints for `tool` (4.4)."""
    base = os.path.join(paths.keypoints_dir(), tool)
    if not os.path.isdir(base):
        return []
    cats = categories or sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    out = []
    for c in cats:
        vids = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(base, c, "*.npz")))
        out += [(c, v) for v in vids[:limit]]
    return out


def write_clips(tool, cat, vid, split, cat_index, args):
    """Cut one video into windows; returns meta.csv rows."""
    kp = np.load(paths.clean_keypoint_path(tool, cat, vid))
    mel_path = paths.feature_path("audio_mel", vid)
    mel = np.load(mel_path)
    onset = np.load(paths.feature_path("audio_onset", vid))
    beats = np.load(paths.feature_path("beats", vid))
    T = min(len(kp["keypoints"]), len(mel), len(onset))
    if abs(len(kp["keypoints"]) - len(mel)) > 3:
        print(f"warning {cat}/{vid}: motion {len(kp['keypoints'])} vs audio {len(mel)} frames")
    if T < args.min_length:
        return None
    rows = []
    win = min(args.window, T)
    for s in range(0, T - win + 1, args.stride):
        e = s + win
        valid = kp["valid"][s:e]
        if valid.mean() < args.min_valid:
            continue
        out = paths.clip_path(tool, split, cat, vid, s)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        b = beats[(beats >= s) & (beats < e)] - s
        np.savez_compressed(out, keypoints=kp["keypoints"][s:e], valid=valid, mel=mel[s:e],
                            onset=onset[s:e], beats=b.astype(np.float32), category=cat,
                            category_index=cat_index[cat], video_id=vid, start=s, end=e, fps=kp["fps"])
        rows.append(dict(clip_path=paths.rel(out), audio_feat_path=paths.rel(mel_path), category=cat,
                         start_frame=s, end_frame=e, split=split, video_id=vid,
                         dancer=aist.dancer_id(vid) or "", music=aist.music_id(vid) or ""))
    return rows


def run(args):
    tool = args.tool
    videos, skipped = [], []
    for cat, vid in list_clean_videos(tool, args.categories, args.limit):
        missing = [paths.rel(paths.feature_path(k, vid)) for k in paths.FEATURE_KINDS
                   if not os.path.exists(paths.feature_path(k, vid))]
        if missing:
            skipped.append((f"{cat}/{vid}", "missing " + ", ".join(missing)))
        else:
            videos.append((cat, vid))
    if not videos:
        sys.exit(f"no video has both cleaned keypoints ({paths.rel(os.path.join(paths.keypoints_dir(), tool))}) "
                 "and audio features; run clean (4.4) and audio (4.5), or `python -m beat2body hf download`")

    splits = split_videos(videos, args.by, args.seed, args.test_songs, args.val_ratio, args.test_ratio)
    cat_index = aist.write_categories_json({c for c, _ in videos})

    for old in glob.glob(os.path.join(paths.tool_clips_dir(tool), "*", "*.npz")):
        os.remove(old)
    rows = []
    for (cat, vid), split in sorted(splits.items()):
        r = write_clips(tool, cat, vid, split, cat_index, args)
        if r is None:
            skipped.append((f"{cat}/{vid}", "too short"))
        else:
            rows += r

    with open(paths.meta_csv_path(tool), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        w.writerows(rows)

    songs = {s: sorted({aist.music_id(v) or v for (_, v), x in splits.items() if x == s}) for s in SPLITS}
    info = dict(tool=tool, test_songs=args.test_songs, by=args.by, val_ratio=args.val_ratio,
                test_ratio=args.test_ratio if args.test_songs == 0 else None, seed=args.seed,
                window=args.window, stride=args.stride, min_length=args.min_length, min_valid=args.min_valid,
                videos={s: sum(1 for x in splits.values() if x == s) for s in SPLITS},
                clips={s: sum(r["split"] == s for r in rows) for s in SPLITS},
                music=songs)
    with open(paths.split_json_path(tool), "w") as f:
        json.dump(info, f, indent=2)

    for cat in sorted({c for c, _ in videos}):
        n = {s: sum(1 for (c, _), v in splits.items() if c == cat and v == s) for s in SPLITS}
        print(f"{cat}: videos {n}")
    for s in SPLITS:
        print(f"{s}: {info['clips'][s]} clips from {info['videos'][s]} videos, {len(songs[s])} songs")
    unseen = set(songs["test"]) - set(songs["train"]) - set(songs["val"])
    print(f"test songs never seen in train / val: {len(unseen)} of {len(songs['test'])}")
    if skipped:
        print(f"skipped {len(skipped)} videos, e.g.:")
        for k, why in skipped[:5]:
            print(f"  {k}: {why}")
    print(f"wrote {paths.rel(paths.meta_csv_path(tool))}")
