import csv
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from beat2body import paths  # noqa: E402
from beat2body.data import DanceClips, audio_stats, flip_keypoints, make_loader  # noqa: E402
from beat2body.prep.clips import META_FIELDS  # noqa: E402

W = 40


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "2d"
    old = paths.data_root()
    paths.set_data_root(str(root))
    (root).mkdir()
    json.dump({"house": 0, "la_hiphop": 1}, open(root / "categories.json", "w"))
    rows = []
    rng = np.random.default_rng(0)
    # train audio ~ N(10, 2); val audio is shifted, so stats must come from train only
    for split, vid, cat, shift in [("train", "gLH_sFM_c01_d01_mLH0_ch01", "la_hiphop", 10.0),
                                   ("train", "gHO_sFM_c01_d02_mHO0_ch02", "house", 10.0),
                                   ("val", "gLH_sFM_c01_d01_mLH1_ch03", "la_hiphop", 50.0)]:
        mel = (shift + 2 * rng.standard_normal((W, 64))).astype(np.float32)
        onset = (shift + rng.standard_normal(W)).astype(np.float32)
        for kind, arr in [("audio_mel", mel), ("audio_onset", onset), ("beats", np.array([3, 20], np.int32))]:
            os.makedirs(os.path.dirname(paths.feature_path(kind, vid)), exist_ok=True)
            np.save(paths.feature_path(kind, vid), arr)
        kp = rng.standard_normal((W, 17, 2)).astype(np.float32)
        out = paths.clip_path("dwpose", split, cat, vid, 0)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, keypoints=kp, valid=np.ones((W, 17), bool), mel=mel, onset=onset,
                            beats=np.array([3, 20], np.int32), category=cat,
                            category_index={"house": 0, "la_hiphop": 1}[cat], video_id=vid, start=0, end=W, fps=30)
        rows.append(dict(clip_path=paths.rel(out), audio_feat_path=paths.rel(paths.feature_path("audio_mel", vid)),
                         category=cat, start_frame=0, end_frame=W, split=split, video_id=vid, dancer="", music=""))
    with open(paths.meta_csv_path("dwpose"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        w.writerows(rows)
    yield root
    paths.set_data_root(old)


def test_item_shapes_and_beats(data_root):
    ds = DanceClips("dwpose", "val", seq_len=W)
    x = ds[0]
    assert x["keypoints"].shape == (W, 17, 2) and x["valid"].shape == (W, 17)
    assert x["audio"].shape == (W, 65) and x["audio"].dtype == torch.float32
    assert torch.nonzero(x["beats"]).flatten().tolist() == [3, 20]
    assert x["category"].item() == 1


def test_stats_from_train_only_and_cached(data_root):
    s = audio_stats("dwpose")
    assert s["videos"] == 2 and abs(np.mean(s["mel_mean"]) - 10) < 0.5
    assert os.path.exists(paths.audio_stats_path("dwpose"))
    train = torch.stack([DanceClips("dwpose", "train")[i]["audio"] for i in range(2)])
    assert abs(train.mean().item()) < 0.1 and abs(train.std().item() - 1) < 0.1
    assert DanceClips("dwpose", "val")[0]["audio"].mean() > 5        # val is not re-normalized on itself


def test_eval_is_deterministic_and_train_crops(data_root):
    val = DanceClips("dwpose", "val", seq_len=16)
    a, b = val[0], val[0]
    assert a["start"] == 0 and torch.equal(a["keypoints"], b["keypoints"])
    train = DanceClips("dwpose", "train", seq_len=16, flip_prob=0.0)
    starts = {train[0]["start"] for _ in range(30)}
    assert len(starts) > 1 and max(starts) <= W - 16


def test_flip():
    kp = np.zeros((1, 17, 2), np.float32)
    kp[0, 9] = [0.5, 1.0]                        # left wrist
    valid = np.zeros((1, 17), bool)
    valid[0, 9] = True
    f, v = flip_keypoints(kp, valid)
    assert np.allclose(f[0, 10], [-0.5, 1.0]) and v[0, 10] and not v[0, 9]   # became the right wrist
    ff, vv = flip_keypoints(f, v)
    assert np.allclose(ff, kp) and (vv == valid).all()


def test_loader_batches(data_root):
    batch = next(iter(make_loader("dwpose", "train", batch_size=2, seq_len=32)))
    assert batch["keypoints"].shape == (2, 32, 17, 2) and batch["category"].shape == (2,)
