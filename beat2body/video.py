"""Reading videos at a fixed frame rate and drawing skeletons (needs ffmpeg on the PATH)."""
import json
import os
import shutil
import subprocess

import numpy as np

from beat2body.skeleton import COCO17_EDGES

DEFAULT_FPS = 30


def probe_video(path):
    """(width, height, fps) of the first video stream (ffprobe, or OpenCV when ffprobe is missing)."""
    if shutil.which("ffprobe"):
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", path]
        s = json.loads(subprocess.check_output(cmd))["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        return int(s["width"]), int(s["height"]), float(num) / float(den)
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return w, h, fps


def scaled_size(width, height, target_height):
    if not target_height or target_height == height:
        return width, height
    return int(round(width * target_height / height / 2)) * 2, target_height


def iter_frames(path, fps=DEFAULT_FPS, height=None):
    """Yield BGR uint8 frames resampled to a constant `fps` (AIST videos are 59.94 fps).

    The same constant fps is used for the audio features, so motion frame t lines up
    with audio feature frame t.
    """
    w, h, _ = probe_video(path)
    vf = f"fps={fps}"
    if height and height != h:
        w, h = scaled_size(w, h, height)
        vf += f",scale={w}:{h}"
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        if proc.poll() is None:   # stopped early: end ffmpeg before closing its pipe
            proc.kill()
        proc.stdout.close()
        proc.wait()


def draw_skeleton(img, kpts, thr=None, color=(0, 255, 0), thickness=3, radius=5):
    """Draw COCO-17 joints on img (in place). kpts: (17, 2) or (17, 3) with a confidence column;
    joints that are NaN, or below `thr` when a confidence is given, are skipped."""
    import cv2

    def ok(j):
        if not np.isfinite(kpts[j, :2]).all():
            return False
        return thr is None or kpts.shape[1] < 3 or kpts[j, 2] >= thr

    pt = lambda j: (int(round(kpts[j, 0])), int(round(kpts[j, 1])))
    for a, b in COCO17_EDGES:
        if ok(a) and ok(b):
            cv2.line(img, pt(a), pt(b), color, thickness, cv2.LINE_AA)
    for j in range(len(kpts)):
        if ok(j):
            cv2.circle(img, pt(j), radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pt(j), radius, color, 2, cv2.LINE_AA)
    return img


class VideoWriter:
    """Writes BGR frames to an H.264 mp4 through ffmpeg, optionally muxing an audio file."""

    def __init__(self, path, fps, width, height, audio=None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.size = (width, height)
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
        if audio:  # a .wav or a video file; its first audio stream is used
            cmd += ["-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, img):
        assert img.shape[1] == self.size[0] and img.shape[0] == self.size[1], (img.shape, self.size)
        self.proc.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class PreviewWriter(VideoWriter):
    """Video frames with the skeleton drawn on top and a title bar (pose tool previews)."""

    def __init__(self, path, fps, width, height, title, max_height=720):
        scale = min(1.0, max_height / height)
        super().__init__(path, fps, int(width * scale) // 2 * 2, int(height * scale) // 2 * 2)
        self.title = title

    def write_frame(self, frame, kpts, thr, color=(0, 255, 0)):
        import cv2
        img = draw_skeleton(frame.copy(), kpts, thr, color)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img, self.title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        self.write(img)
