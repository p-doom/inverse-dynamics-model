import pickle

import numpy as np
import pytest

grain = pytest.importorskip("grain")

from idm.utils.data import (  # noqa: E402
    ActionDensityRejectionSampler,
    ActionDensityFilter,
    BuildSFTExampleFromFrames,
    EpisodeLengthFilter,
    NotNoneFilter,
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


def test_action_density_rejection_sampler_keeps_all_when_random_fraction_is_one():
    tr = ActionDensityRejectionSampler(random_sample_fraction=1.0)
    element_d = {
        "frames": np.zeros((4, 2, 2, 3), dtype=np.uint8),
        "actions": ["NO_OP", "NO_OP", "NO_OP", "NO_OP"],
    }
    out_d = tr.random_map(element_d, np.random.default_rng(0))
    assert out_d == element_d


def test_action_density_rejection_sampler_drops_all_noop_when_random_fraction_is_zero():
    tr = ActionDensityRejectionSampler(random_sample_fraction=0.0)
    element_d = {
        "frames": np.zeros((4, 2, 2, 3), dtype=np.uint8),
        "actions": ["NO_OP", "NO_OP", "NO_OP", "NO_OP"],
    }
    out_d = tr.random_map(element_d, np.random.default_rng(0))
    assert out_d is None


def test_action_density_rejection_sampler_keeps_all_active_when_random_fraction_is_zero():
    tr = ActionDensityRejectionSampler(random_sample_fraction=0.0)
    element_d = {
        "frames": np.zeros((4, 2, 2, 3), dtype=np.uint8),
        "actions": ["MOUSE:1,0,0", "MOUSE:1,0,0", "MOUSE:1,0,0", "MOUSE:1,0,0"],
    }
    out_d = tr.random_map(element_d, np.random.default_rng(0))
    assert out_d == element_d


def test_action_density_rejection_sampler_matches_expected_keep_rate():
    tr = ActionDensityRejectionSampler(random_sample_fraction=0.5)
    element_d = {
        "frames": np.zeros((4, 2, 2, 3), dtype=np.uint8),
        "actions": ["NO_OP", "NO_OP", "MOUSE:1,0,0", "NO_OP"],
    }
    # density = 0.25 -> expected keep probability = 0.5 + 0.5 * 0.25 = 0.625
    trials_n = 4000
    kept_n = 0
    rng = np.random.default_rng(123)
    for _ in range(trials_n):
        kept_n += int(tr.random_map(element_d, rng) is not None)
    keep_rate_f = float(kept_n) / float(trials_n)
    assert keep_rate_f == pytest.approx(0.625, abs=0.03)


def test_not_none_filter_accepts_only_non_none():
    filt = NotNoneFilter()
    assert filt.filter({"x": 1})
    assert not filt.filter(None)
