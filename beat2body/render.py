"""Stick-figure videos of real or generated keypoints (README 1, "can be rendered").

Each panel shows one normalized (T, 17, 2) sequence on a blank canvas; the panel border
flashes on detected music beats. Sources:

    --video-id ID        the cleaned keypoints of a dataset video (+ its beats and audio)
    --keypoints FILE     a .npy (T, 17, 2) or .npz with `keypoints`, e.g. a generated sequence
    --overlay            add the original video with the skeleton drawn on it (needs raw_video/)

    python -m beat2body render --video-id gLH_sFM_c01_d16_mLH0_ch01 --overlay
    python -m beat2body render --video-id gLH_sFM_c01_d16_mLH0_ch01 --keypoints runs/x/sample.npy -o cmp.mp4

The audio track is taken from <data>/audio/, or from the raw video, when available.
"""
import glob
import os
import sys

import numpy as np

from beat2body import paths
from beat2body.paths import TOOLS
from beat2body.video import DEFAULT_FPS, VideoWriter, draw_skeleton, iter_frames

DOC = __doc__
PANEL_COLORS = [(80, 220, 80), (80, 160, 255), (255, 120, 80), (220, 80, 220)]


def add_args(p):
    p.add_argument("--video-id", default=None, help="dataset video to render (ground truth)")
    p.add_argument("--tool", default="dwpose", choices=TOOLS, help="which cleaned keypoints to use")
    p.add_argument("--keypoints", nargs="*", default=[], help="extra .npy / .npz sequences, one panel each")
    p.add_argument("--labels", nargs="*", default=None, help="panel titles (default: file names)")
    p.add_argument("--overlay", action="store_true", help="add the original video with the skeleton")
    p.add_argument("--start", type=int, default=0, help="first frame")
    p.add_argument("--frames", type=int, default=None, help="number of frames (default: all)")
    p.add_argument("--size", type=int, default=540, help="panel height in pixels")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("-o", "--out", default=None, help="output .mp4 (default: renders/<name>.mp4)")


def find_clean(tool, video_id):
    hits = glob.glob(paths.clean_keypoint_path(tool, "*", video_id))
    if not hits:
        sys.exit(f"no cleaned keypoints for {video_id} ({tool}); run clean or `python -m beat2body hf download`")
    return hits[0]


def load_sequence(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    with np.load(path) as d:
        return d["keypoints"].astype(np.float32)


class StickPanel:
    """Maps normalized coordinates (pelvis at 0, torso length 1) to a square canvas.
    The view is fixed per sequence, so the figure's own motion is visible."""

    def __init__(self, seq, size, title, color):
        self.size, self.title, self.color = size, title, color
        lo, hi = np.percentile(seq.reshape(-1, 2), [0.5, 99.5], axis=0)
        center = (lo + hi) / 2
        span = max(float((hi - lo).max()) * 1.15, 2.5)
        self.lo = center - span / 2
        self.px = (size - 40) / span

    def draw(self, kp, beat):
        import cv2
        img = np.full((self.size, self.size, 3), 18, np.uint8)
        xy = (kp - self.lo) * self.px + 20
        draw_skeleton(img, xy, color=self.color, thickness=4, radius=5)
        if beat:
            cv2.rectangle(img, (2, 2), (self.size - 3, self.size - 3), (0, 220, 255), 6)
        cv2.putText(img, self.title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
        return img


def run(args):
    import cv2

    seqs, titles, beats, audio, raw_video, gt = [], [], None, None, None, None
    if args.video_id:
        f = find_clean(args.tool, args.video_id)
        gt = np.load(f)
        cat = os.path.basename(os.path.dirname(f))
        seqs.append(gt["keypoints"])
        titles.append(f"{cat} ({args.tool})")
        if os.path.exists(paths.feature_path("beats", args.video_id)):
            beats = np.load(paths.feature_path("beats", args.video_id))
        candidates = glob.glob(os.path.join(paths.raw_video_dir(), cat, args.video_id + ".*"))
        raw_video = candidates[0] if candidates else None
        wav = paths.audio_path(cat, args.video_id)
        audio = wav if os.path.exists(wav) else raw_video
    for k in args.keypoints:
        seqs.append(load_sequence(k))
        titles.append(os.path.splitext(os.path.basename(k))[0])
    if args.labels:
        titles[:len(args.labels)] = args.labels
    if not seqs:
        sys.exit("nothing to render: give --video-id and/or --keypoints")
    if args.overlay and not raw_video:
        sys.exit("--overlay needs --video-id and the raw video in <data>/raw_video/")

    T = min(len(s) for s in seqs)
    start = max(0, args.start)
    stop = T if args.frames is None else min(T, start + args.frames)
    panels = [StickPanel(s[start:stop], args.size, t, PANEL_COLORS[i % len(PANEL_COLORS)])
              for i, (s, t) in enumerate(zip(seqs, titles))]
    beat_frames = set()
    if beats is not None:
        for b in beats:
            beat_frames.update(range(int(b), int(b) + 3))   # flash for 3 frames (0.1 s)

    frames = None
    ov_w = 0
    if args.overlay:
        fps_frames = iter_frames(raw_video, DEFAULT_FPS)
        frames = (fr for i, fr in enumerate(fps_frames) if i >= start)
        ov_w = int(round(gt["width"] * args.size / gt["height"])) // 2 * 2
    width = ov_w + args.size * len(panels)
    name = args.video_id or titles[0]
    out = args.out or os.path.join(paths.REPO_ROOT, "renders", f"{name}.mp4")
    use_audio = audio if (audio and not args.no_audio and start == 0) else None
    if audio and start > 0 and not args.no_audio:
        print("note: audio is only added when rendering from --start 0")

    with VideoWriter(out, DEFAULT_FPS, width, args.size, audio=use_audio) as w:
        for t in range(start, stop):
            row = []
            if frames is not None:
                fr = next(frames, None)
                if fr is None:
                    break
                px = gt["keypoints"][t] * float(gt["scale"]) + gt["pelvis"][t]
                fr = draw_skeleton(fr.copy(), px, color=PANEL_COLORS[0])
                row.append(cv2.resize(fr, (ov_w, args.size), interpolation=cv2.INTER_AREA))
            row += [p.draw(s[t], t in beat_frames) for p, s in zip(panels, seqs)]
            w.write(np.concatenate(row, axis=1))
    print(f"wrote {out} ({stop - start} frames, {len(panels)} panel(s){', overlay' if frames is not None else ''})")
