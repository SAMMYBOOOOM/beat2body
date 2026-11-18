"""4.5 Audio extraction and features (shared by all pose tools).

For every video:
    <data>/audio/<category>/<video_id>.wav               mono, 22.05 kHz
    <data>/features/audio_mel/<video_id>.npy    (T, 64)  log-mel spectrogram (dB)
    <data>/features/audio_onset/<video_id>.npy  (T,)     onset strength
    <data>/features/beats/<video_id>.npy        (B,)     beat frame indices

hop_length = sample_rate / fps (22050 / 30 = 735), so feature frame t lines up with
motion frame t (both use the same fixed fps). Needs: pip install librosa

    python -m beat2body audio --categories la_hiphop house
"""
import os
import subprocess
import sys

import numpy as np

from beat2body import aist, paths
from beat2body.video import DEFAULT_FPS

DOC = __doc__


def add_args(p, with_categories=True):
    if with_categories:
        p.add_argument("--categories", nargs="*", default=None)
        p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--overwrite", action="store_true")


def extract_wav(video, wav, sr):
    os.makedirs(os.path.dirname(wav), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vn", "-ac", "1", "-ar", str(sr), wav], check=True)


def _save(kind, vid, arr):
    path = paths.feature_path(kind, vid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def run(args):
    import librosa

    hop = args.sr / args.fps
    if abs(hop - round(hop)) > 1e-6:
        sys.exit(f"sample rate {args.sr} / fps {args.fps} is not an integer hop length")
    hop = int(round(hop))

    for cat, vid, path in aist.list_videos(args.categories, limit=args.limit):
        wav = paths.audio_path(cat, vid)
        if os.path.exists(paths.feature_path("beats", vid)) and not args.overwrite:
            print("skip", f"{cat}/{vid}")
            continue
        extract_wav(path, wav, args.sr)
        y, sr = librosa.load(wav, sr=args.sr, mono=True)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=args.n_fft, hop_length=hop, n_mels=args.n_mels)
        log_mel = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)                     # (T, n_mels)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop).astype(np.float32)  # (T,)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)

        _save("audio_mel", vid, log_mel)
        _save("audio_onset", vid, onset_env)
        _save("beats", vid, np.asarray(beats, dtype=np.int32))                                  # last = done marker
        bpm = float(np.atleast_1d(tempo)[0])
        print(f"{cat}/{vid}: {len(log_mel)} frames, {len(beats)} beats (~{bpm:.0f} BPM)")
