from collections import Counter

from beat2body import aist
from beat2body.prep.clips import allocate, split_videos

# 2 genres x 3 dancers x 6 songs, plus a 7th video per dancer for some songs (like AIST: 21 per genre)
VIDEOS = [(cat, f"g{code}_sFM_c01_d{d:02d}_m{code}{s}_ch{d * 10 + s:02d}")
          for cat, code in [("la_hiphop", "LH"), ("house", "HO")]
          for d in (1, 2, 3) for s in range(6)]
VIDEOS += [(cat, f"g{code}_sFM_c01_d{d:02d}_m{code}{d}_ch{90 + d:02d}")
           for cat, code in [("la_hiphop", "LH"), ("house", "HO")] for d in (1, 2, 3)]


def songs(splits, split):
    return {aist.music_id(v) for (_, v), s in splits.items() if s == split}


def test_ids():
    v = "gLH_sFM_c01_d16_mLH0_ch01"
    assert aist.dancer_id(v) == "d16" and aist.music_id(v) == "mLH0"
    assert aist.dancer_id("my_video") is None and aist.music_id("my_video") is None


def test_allocate():
    assert allocate(18, [0.85, 0.15, 0.0]) == [15, 3, 0]
    assert allocate(21, [0.7, 0.15, 0.15]) == [15, 3, 3]
    assert sum(allocate(3, [0.9, 0.05, 0.05])) == 3 and min(allocate(3, [0.9, 0.05, 0.05])) == 1


def test_default_holds_out_one_song_per_genre():
    sp = split_videos(VIDEOS)
    assert len(sp) == len(VIDEOS)
    test = songs(sp, "test")
    assert len(test) == 2 and len({s[1:3] for s in test}) == 2       # one song in each genre
    assert not test & (songs(sp, "train") | songs(sp, "val"))         # test music never in training
    # train / val are split by video, so they share songs
    assert songs(sp, "train") & songs(sp, "val")
    for cat in ("la_hiphop", "house"):
        c = Counter(s for (k, _), s in sp.items() if k == cat)
        assert c["val"] >= 1 and c["train"] > c["val"]


def test_deterministic_and_seeded():
    assert split_videos(VIDEOS, seed=3) == split_videos(VIDEOS, seed=3)
    assert any(split_videos(VIDEOS, seed=s) != split_videos(VIDEOS, seed=0) for s in range(1, 6))


def test_group_never_spans_splits():
    for by, key in [("dancer", aist.dancer_id), ("music", aist.music_id)]:
        sp = split_videos(VIDEOS, by=by, test_songs=0)
        seen = {}
        for (cat, v), s in sp.items():
            assert seen.setdefault((cat, key(v)), s) == s
        assert set(sp.values()) == {"train", "val", "test"}
