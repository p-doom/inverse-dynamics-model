import pickle

import numpy as np
import pytest

grain = pytest.importorskip("grain")

from idm.utils.data import (  # noqa: E402
    ActionDensityFilter,
    BuildSFTExampleFromFrames,
    EpisodeLengthFilter,
    ProcessEpisodeAndSlice,
)


def _make_record_bytes(
    T: int = 6, H: int = 2, W: int = 2, C: int = 3, with_actions: bool = True
) -> bytes:
    frames_THWC = np.zeros((T, H, W, C), dtype=np.uint8)
    for t_i in range(T):
        frames_THWC[t_i] = t_i
    rec_d = {
        "raw_video": frames_THWC.tobytes(),
        "sequence_length": T,
        "path": "foo/bar.mp4",
    }
    if with_actions:
        rec_d["actions"] = [f"a{t_i}" for t_i in range(T)]
    return pickle.dumps(rec_d)


def test_episode_length_filter_short_false():
    filt = EpisodeLengthFilter(seq_len=8, image_h=2, image_w=2, image_c=3)
    assert not filt.filter(_make_record_bytes(T=6))


def test_episode_length_filter_requires_in_record_actions():
    filt = EpisodeLengthFilter(seq_len=4, image_h=2, image_w=2, image_c=3)
    assert not filt.filter(_make_record_bytes(T=6, with_actions=False))


def test_process_episode_and_slice_contiguous():
    tr = ProcessEpisodeAndSlice(seq_len=4, image_h=2, image_w=2, image_c=3)
    out_d = tr.random_map(_make_record_bytes(T=6), np.random.default_rng(0))
    frames_SHWC = out_d["frames"]
    actions_S = out_d["actions"]
    assert frames_SHWC.shape == (4, 2, 2, 3)
    start_i = int(frames_SHWC[0, 0, 0, 0])
    assert actions_S == [f"a{start_i + i}" for i in range(4)]
    assert int(frames_SHWC[-1, 0, 0, 0]) == start_i + 3


def test_process_episode_raises_when_actions_missing():
    tr = ProcessEpisodeAndSlice(seq_len=4, image_h=2, image_w=2, image_c=3)
    with pytest.raises(ValueError):
        tr.random_map(
            _make_record_bytes(T=6, with_actions=False), np.random.default_rng(1)
        )


def test_build_sft_example_outputs_all_actions_text():
    tr = BuildSFTExampleFromFrames()
    ex_d = {
        "frames": np.zeros((3, 2, 2, 3), dtype=np.uint8),
        "actions": ["left", "jump", "shoot"],
        "meta": {"k": "v"},
    }
    out_d = tr.map(ex_d)
    assert "Frame 0: left" in out_d["target_text"]
    assert "Frame 1: jump" in out_d["target_text"]
    assert "Frame 2: shoot" in out_d["target_text"]


def test_action_density_filter_rejects_too_sparse_sequence():
    filt = ActionDensityFilter(min_action_density=0.5)
    element_d = {
        "frames": np.zeros((4, 2, 2, 3), dtype=np.uint8),
        "actions": ["NO_OP", "NO_OP", "NO_OP", "MOUSE:1,0,0"],
    }
    assert not filt.filter(element_d)


def test_action_density_filter_accepts_sufficiently_active_sequence():
    filt = ActionDensityFilter(min_action_density=0.5)
    element_d = {
        "frames": np.zeros((4, 2, 2, 3), dtype=np.uint8),
        "actions": ["NO_OP", "MOUSE:1,0,0", "MOUSE:0,0,0 ; W", "NO_OP"],
    }
    assert filt.filter(element_d)
