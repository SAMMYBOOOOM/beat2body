"""4.5 Audio extraction and features (shared by all pose tools).

For every video:
    <data>/audio/<category>/<video_id>.wav               mono, 22.05 kHz
    <data>/features/audio_mel/<video_id>.npy    (T, 64)  log-mel spectrogram (dB)
    <data>/features/audio_onset/<video_id>.npy  (T,)     onset strength
    <data>/features/beats/<video_id>.npy        (B,)     beat times in motion frames (float32)

hop_length = sample_rate / fps (22050 / 30 = 735), so feature frame t lines up with
motion frame t (both use the same fixed fps). Needs: pip install librosa

Beats: AIST songs have a published constant tempo (aist.music_bpm), so their beats are a grid
at exactly that tempo, kept as fractional frames. The grid's phase comes from librosa's beats
when librosa found the right tempo, else from the onset envelope. librosa alone picked the wrong
tempo for 10 of 60 AIST songs (e.g. 80 BPM tracked as 164 or 106): it prefers tempos near 120.

    python -m beat2body audio --categories la_hiphop house
    python -m beat2body beats          # recompute only the beats from existing onset features
"""
import glob
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


def _grid_score(onset, phase, period):
    t = np.arange(len(onset))
    return np.interp(np.arange(phase, len(onset) - 1, period), t, onset).mean()


def beat_grid(onset, bpm, fps=DEFAULT_FPS, reference=None, step=0.05):
    """Beat times (fractional frames) at a known constant tempo.

    The phase is the circular mean of `reference` beats (e.g. librosa's, when their tempo is right),
    else the phase that maximizes the mean onset strength on the grid. Returns (beats, phase_source,
    margin), where margin = onset score at the chosen phase / score half a beat away (> 1: clear)."""
    period = 60.0 * fps / bpm
    if reference is not None and len(reference) > 2:
        ang = np.angle(np.exp(2j * np.pi * np.asarray(reference, np.float64) / period).mean())
        phase, source = (ang / (2 * np.pi) % 1.0) * period, "reference"
    else:
        phases = np.arange(0.0, period, step)
        phase = phases[int(np.argmax([_grid_score(onset, p, period) for p in phases]))]
        source = "onset"
    margin = _grid_score(onset, phase, period) / max(_grid_score(onset, (phase + period / 2) % period, period), 1e-9)
    return np.arange(phase, len(onset) - 1, period).astype(np.float32), source, float(margin)


def compute_beats(video_id, onset_env, sr, hop, fps):
    """Beats for one video -> (beats, bpm, method).

    AIST songs: grid at the published tempo; phase from librosa's beats when librosa found that
    tempo (within 5%), otherwise from the onset envelope. Other music: librosa's beats."""
    import librosa
    tempo, lb = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
    lb = np.asarray(lb, np.float64)
    bpm = aist.music_bpm(video_id)
    if not bpm:
        return lb.astype(np.float32), float(np.atleast_1d(tempo)[0]), "librosa"
    lb_bpm = 60 * fps / np.median(np.diff(lb)) if len(lb) > 2 else 0
    ref = lb if abs(lb_bpm - bpm) / bpm < 0.05 else None
    beats, source, margin = beat_grid(onset_env, bpm, fps, reference=ref)
    return beats, float(bpm), f"aist-tempo/{source}" + ("" if margin > 1.05 else "/ambiguous")


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
        beats, bpm, how = compute_beats(vid, onset_env, sr, hop, args.fps)

        _save("audio_mel", vid, log_mel)
        _save("audio_onset", vid, onset_env)
        _save("beats", vid, beats)                                                              # last = done marker
        print(f"{cat}/{vid}: {len(log_mel)} frames, {len(beats)} beats ({bpm:.0f} BPM, {how})")


EDGE_DOC = """EDGE's "baseline" music features (35 per frame), from the raw videos.

Same as EDGE (data/audio_extraction/baseline_features.py): audio at 15360 Hz, hop 512 = 30 fps;
    onset envelope (1), MFCC (20), chroma CENS (12), onset peaks one-hot (1), beats one-hot (1).
Difference: the beat one-hot comes from our beat grid (published AIST tempo; `beats`), where EDGE
runs librosa's tracker with the published tempo as start_bpm. Output (T, 35) float32:
    <data>/features/audio_edge/<video_id>.npy

    python -m beat2body edge-features            # needs the raw videos, ffmpeg and librosa
"""
EDGE_SR, EDGE_HOP = 15360, 512


def edge_features(audio_file, beats):
    """(T, 35) EDGE baseline features; beats = beat times in frames (float)."""
    import librosa
    y, sr = librosa.load(audio_file, sr=EDGE_SR, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=EDGE_HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=EDGE_HOP).T
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=EDGE_HOP, n_chroma=12).T
    T = min(len(env), len(mfcc), len(chroma))
    peaks = np.zeros(T, np.float32)
    idx = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=EDGE_HOP)
    peaks[idx[idx < T]] = 1
    beat = np.zeros(T, np.float32)
    b = np.round(beats).astype(int)
    beat[b[(b >= 0) & (b < T)]] = 1
    return np.concatenate([env[:T, None], mfcc[:T], chroma[:T], peaks[:, None], beat[:, None]], 1).astype(np.float32)


def add_edge_args(p):
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")


def run_edge(args):
    import tempfile
    n = 0
    for cat, vid, path in aist.list_videos(args.categories):
        out = paths.feature_path("audio_edge", vid)
        if os.path.exists(out) and not args.overwrite:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "a.wav")
            extract_wav(path, wav, EDGE_SR)
            feats = edge_features(wav, np.load(paths.feature_path("beats", vid)))
        _save("audio_edge", vid, feats)
        n += 1
        if n % 20 == 0:
            print(f"  {n} videos ({cat}/{vid}: {feats.shape})")
    print(f"wrote EDGE features for {n} videos")


BEATS_DOC = """Recompute the beats (features/beats/) from the existing onset features (no audio needed).

    python -m beat2body beats            # all videos with onset features
Then rebuild the clips (python -m beat2body clips --tool <tool>), which carry the beats.
"""


def add_beats_args(p):
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)


def run_beats(args):
    files = sorted(glob.glob(paths.feature_path("audio_onset", "*")))
    if not files:
        sys.exit(f"no onset features in {paths.features_dir()}; run `audio` or `hf download` first")
    counts = {}
    for f in files:
        vid = os.path.splitext(os.path.basename(f))[0]
        beats, bpm, how = compute_beats(vid, np.load(f), 22050, int(round(22050 / args.fps)), args.fps)
        _save("beats", vid, beats)
        counts[how] = counts.get(how, 0) + 1
    print(f"recomputed beats for {len(files)} videos: {counts}")
