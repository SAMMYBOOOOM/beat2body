import csv
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from beat2body import paths  # noqa: E402
from beat2body.data import CONTACT, JOINTS, ROOT, DanceClips, audio_stats, beat_phase, flip_motion, make_loader  # noqa: E402
from beat2body.prep.clips import META_FIELDS  # noqa: E402

T = 60
VIDEOS = [("train", "gLH_sFM_c01_d01_mLH0_ch01", "la_hiphop", 10.0),
          ("train", "gHO_sFM_c01_d02_mHO0_ch02", "house", 10.0),
          ("val", "gLH_sFM_c01_d01_mLH1_ch03", "la_hiphop", 50.0)]


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "2d"
    old = paths.data_root()
    paths.set_data_root(str(root))
    root.mkdir()
    json.dump({"house": 0, "la_hiphop": 1}, open(root / "categories.json", "w"))
    rng = np.random.default_rng(0)
    rows = []
    for split, vid, cat, shift in VIDEOS:
        kp = 0.3 * rng.standard_normal((T, 17, 2)).astype(np.float32)
        kp[:, 15:17] = 0.0                                   # ankles still relative to the pelvis ...
        pelvis = np.tile([960.0, 700.0], (T, 1)).astype(np.float32)
        pelvis[30:, 0] += 10 * np.arange(T - 30)             # ... the body travels right from frame 30
        out = paths.clean_keypoint_path("dwpose", cat, vid)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, keypoints=kp, valid=np.ones((T, 17), bool), pelvis=pelvis, scale=100.0, width=1920,
                 height=1080, fps=30)
        for kind, arr in [("audio_edge", (shift + rng.standard_normal((T, 35))).astype(np.float32)),
                          ("beats", np.arange(2.0, T, 15.0, dtype=np.float32))]:
            os.makedirs(os.path.dirname(paths.feature_path(kind, vid)), exist_ok=True)
            np.save(paths.feature_path(kind, vid), arr)
        rows.append(dict(clip_path="", audio_feat_path="", category=cat, start_frame=0, end_frame=48, split=split,
                         video_id=vid, dancer="", music=""))
    os.makedirs(paths.tool_clips_dir("dwpose"), exist_ok=True)
    with open(paths.meta_csv_path("dwpose"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        w.writerows(rows)
    yield root
    paths.set_data_root(old)


def test_item_layout(data_root):
    x = DanceClips("dwpose", "val", seq_len=48, scale_norm="dataset")[0]
    m = x["motion"]
    assert m.shape == (48, 38) and x["audio"].shape == (48, 37) and x["category"].item() == 1
    assert torch.allclose(m[0, ROOT], torch.tensor([0.0, 1.6]))          # (960, 700) - centre, / scale 100
    assert (m[:29, CONTACT] == 1).all() and (m[31:, CONTACT] == 0).all()   # ankles stop being still at frame 30
    assert torch.nonzero(x["beats"]).flatten().tolist() == [2, 17, 32, 47]


def test_video_scale_norm_makes_torso_one(data_root):
    from beat2body.data import torso_length
    m = DanceClips("dwpose", "val", seq_len=48)[0]["motion"].numpy()
    kp = m[:, JOINTS].reshape(48, 17, 2)
    assert abs(torso_length(kp, np.ones((48, 17), bool)) - 1.0) < 1e-5


def test_beat_phase():
    ph = beat_phase([10.0, 20.0, 30.0], 40)
    assert np.allclose(ph[10], [0, 1]) and np.allclose(ph[15], [0, -1], atol=1e-6) and (ph[:10] == 0).all()


def test_stats_from_train_only(data_root):
    s = audio_stats("dwpose", "edge")
    assert s["videos"] == 2 and abs(np.mean(s["mean"]) - 10) < 0.5
    assert DanceClips("dwpose", "val", 48)[0]["audio"][:, :35].mean() > 5


def test_flip_and_range(data_root):
    ds = DanceClips("dwpose", "train", seq_len=48)
    m = ds.videos[("dwpose", "gLH_sFM_c01_d01_mLH0_ch01")]["motion"][:48]
    f, v, _ = flip_motion(m, np.ones((48, 17), bool), np.ones(48, bool))
    assert np.allclose(f[:, ROOT.start], -m[:, ROOT.start])
    assert np.allclose(f[:, JOINTS.start + 2 * 10], -m[:, JOINTS.start + 2 * 9])   # right wrist x = -left wrist x
    lo, hi = ds.motion_range()
    assert np.allclose(lo[ROOT.start], -hi[ROOT.start]) and lo[0] == 0 and hi[0] == 1


def test_train_crops_and_loader(data_root):
    ds = DanceClips("dwpose", "train", seq_len=16, flip_prob=0.0)
    assert len({ds[0]["start"] for _ in range(30)}) > 1
    b = next(iter(make_loader("dwpose", "train", batch_size=2, seq_len=32)))
    assert b["motion"].shape == (2, 32, 38) and b["root_valid"].shape == (2, 32)


def test_jitter_keeps_shape_and_rotates_root():
    from beat2body.data import jitter_motion
    m = np.zeros((4, 38), np.float32)
    m[:, ROOT] = [1.0, 0.0]
    m[:, JOINTS.start:JOINTS.start + 2] = [0.0, 1.0]
    out = jitter_motion(m, 2.0, np.pi / 2, (0.5, 0.0))
    assert np.allclose(out[0, ROOT], [0.5, 2.0], atol=1e-5)                 # (1,0) * 2 rotated 90 deg + shift
    assert np.allclose(out[0, JOINTS.start:JOINTS.start + 2], [-2.0, 0.0], atol=1e-5)
    assert np.allclose(out[:, CONTACT], m[:, CONTACT])


def test_stretch_and_tools(data_root):
    import shutil
    # a second pose tool for one video: same layout, different numbers
    vid, cat = "gLH_sFM_c01_d01_mLH0_ch01", "la_hiphop"
    dst = paths.clean_keypoint_path("openpose", cat, vid)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(paths.clean_keypoint_path("dwpose", cat, vid), dst)
    ds = DanceClips("dwpose", "train", seq_len=32, extra_tools=("openpose",), stretch=0.1, scale=0.05,
                    rotate=5, shift=0.1)
    assert ds.tools_of[vid] == ["dwpose", "openpose"] and ds.tools_of["gHO_sFM_c01_d02_mHO0_ch02"] == ["dwpose"]
    srcs = {ds.window(0)[3] for _ in range(40)}
    assert min(srcs) < 32 < max(srcs) and max(srcs) <= 36                 # stretched source lengths
    x = ds[0]
    assert x["motion"].shape == (32, 38) and x["audio"].shape == (32, 37) and torch.isfinite(x["motion"]).all()
    val = DanceClips("dwpose", "val", seq_len=32, extra_tools=("openpose",), stretch=0.1)
    assert val.window(0)[1:] == ("dwpose", 0, 32)                            # val stays deterministic
