from __future__ import annotations

import dataclasses
import importlib.util
import pickle
from pathlib import Path

from array_record.python.array_record_module import ArrayRecordReader
import msgpack
import numpy as np
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "video_to_array_records.py"
)
_SPEC = importlib.util.spec_from_file_location("video_to_array_records", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_actions_from_keylog_entries = _MODULE._actions_from_keylog_entries
_actions_from_keylog_file = _MODULE._actions_from_keylog_file
_chunk_video_records = _MODULE._chunk_video_records
_drop_no_op_frames = _MODULE._drop_no_op_frames
_filter_black_frames = _MODULE._filter_black_frames
_get_keylog_path = _MODULE._get_keylog_path
_write_chunk_records = _MODULE._write_chunk_records


def test_args_expose_new_noop_drop_flag() -> None:
    field_names = {field.name for field in dataclasses.fields(_MODULE.Args)}
    assert "drop_no_op_prob_train" in field_names
    assert "mouse_delta_clip" in field_names
    assert "mouse_scroll_clip" in field_names
    assert "filter_pure_noop_chunks" not in field_names
    assert "filter_identical_noop_frames" not in field_names
    assert "identical_noop_mad_threshold" not in field_names


def test_get_keylog_path() -> None:
    filename_s = "/tmp/uploads/0.1.0/u123/recordings/recording_abc-def_seg0007.mp4"
    out_p = _get_keylog_path(filename_s)
    assert out_p == Path(
        "/tmp/uploads/0.1.0/u123/keylogs/input_abc-def_seg0007.msgpack"
    )


def test_get_keylog_path_accepts_filtered_suffix() -> None:
    filename_s = (
        "/tmp/uploads/0.1.0/u123/recordings/recording_abc-def_seg0007_filtered.mp4"
    )
    out_p = _get_keylog_path(filename_s)
    assert out_p == Path(
        "/tmp/uploads/0.1.0/u123/keylogs/input_abc-def_seg0007_filtered.msgpack"
    )


def test_actions_from_keylog_file_aligns_to_target_fps(tmp_path: Path) -> None:
    entries_L = [
        [0, ["KeyPress", [0, "KeyW"]]],
        [100_000, ["MouseMove", [5.0, 0.0]]],
        [700_000, ["MouseScroll", [0.0, -1.0]]],
        [850_000, ["KeyRelease", [0, "KeyW"]]],
        [900_000, ["MouseMove", [0.0, 0.0]]],
    ]
    keylog_p = tmp_path / "k.msgpack"
    keylog_p.write_bytes(msgpack.packb(entries_L, use_bin_type=True))

    actions_L = _actions_from_keylog_file(keylog_p, n_frames=10, target_fps=10)

    assert len(actions_L) == 10
    assert actions_L[0] == "MOUSE:0,0,0 ; W"
    assert actions_L[1] == "MOUSE:1,0,0 ; W"
    assert actions_L[7] == "MOUSE:0,0,-1 ; W"
    assert actions_L[8] == "NO_OP"
    assert actions_L[9] == "NO_OP"


def test_actions_from_empty_keylog_file_is_noop(tmp_path: Path) -> None:
    keylog_p = tmp_path / "empty.msgpack"
    keylog_p.write_bytes(msgpack.packb([], use_bin_type=True))

    actions_L = _actions_from_keylog_file(keylog_p, n_frames=6, target_fps=10)

    assert actions_L == ["NO_OP"] * 6


def test_actions_quantize_and_sum_mouse_motion_per_frame() -> None:
    entries_L = [
        [0, ["MouseMove", [2.0, 2.0]]],
        [1_000, ["MouseMove", [3.0, 2.0]]],
    ]

    actions_L = _actions_from_keylog_entries(entries_L, n_frames=1, target_fps=10)

    assert actions_L == ["MOUSE:1,1,0"]


def test_actions_clamp_mouse_motion_and_scroll_ranges() -> None:
    entries_L = [
        [0, ["MouseMove", [6_000.0, -6_000.0]]],
        [0, ["MouseScroll", [0.0, 80.0]]],
        [0, ["MouseScroll", [0.0, -180.0]]],
    ]

    actions_L = _actions_from_keylog_entries(entries_L, n_frames=1, target_fps=10)

    assert actions_L == ["MOUSE:64,-64,-5"]


def test_actions_clamp_respects_custom_clip_args() -> None:
    entries_L = [
        [0, ["MouseMove", [10_000.0, -10_000.0]]],
        [0, ["MouseScroll", [0.0, 100.0]]],
    ]

    actions_L = _actions_from_keylog_entries(
        entries_L,
        n_frames=1,
        target_fps=10,
        mouse_delta_clip=12,
        mouse_scroll_clip=2,
    )

    assert actions_L == ["MOUSE:12,-12,2"]


def test_actions_clamp_rejects_negative_clip_args() -> None:
    entries_L = [[0, ["MouseMove", [1.0, 1.0]]]]
    with pytest.raises(ValueError, match="mouse_delta_clip must be non-negative"):
        _actions_from_keylog_entries(
            entries_L,
            n_frames=1,
            target_fps=10,
            mouse_delta_clip=-1,
        )
    with pytest.raises(ValueError, match="mouse_scroll_clip must be non-negative"):
        _actions_from_keylog_entries(
            entries_L,
            n_frames=1,
            target_fps=10,
            mouse_scroll_clip=-1,
        )


def test_actions_track_pressed_key_state_across_frames() -> None:
    entries_L = [
        [0, ["KeyPress", [0, "KeyW"]]],
        [250_000, ["KeyRelease", [0, "KeyW"]]],
    ]

    actions_L = _actions_from_keylog_entries(entries_L, n_frames=4, target_fps=10)

    assert actions_L[0] == "MOUSE:0,0,0 ; W"
    assert actions_L[1] == "MOUSE:0,0,0 ; W"
    assert actions_L[2] == "NO_OP"
    assert actions_L[3] == "NO_OP"


def test_actions_track_mouse_button_state_as_keys() -> None:
    entries_L = [
        [0, ["MousePress", ["Left", 0.0, 0.0]]],
        [100_000, ["MouseRelease", ["Left", 0.0, 0.0]]],
    ]

    actions_L = _actions_from_keylog_entries(entries_L, n_frames=3, target_fps=10)

    assert actions_L[0] == "MOUSE:0,0,0 ; LMB"
    assert actions_L[1] == "NO_OP"
    assert actions_L[2] == "NO_OP"


def test_chunk_video_records_embeds_action_slices() -> None:
    frames_THWC = np.arange(8 * 2 * 2 * 3, dtype=np.uint8).reshape(8, 2, 2, 3)
    actions_L = [f"a{idx_i}" for idx_i in range(8)]
    video_info_d = {
        "filename": "/tmp/v.mp4",
        "path": "x/v.mp4",
    }

    chunks_L = _chunk_video_records(
        video_tensor=frames_THWC,
        video_info=video_info_d,
        chunk_size=4,
        actions=actions_L,
    )
    assert len(chunks_L) == 2
    assert chunks_L[0]["actions"] == ["a0", "a1", "a2", "a3"]
    assert chunks_L[1]["actions"] == ["a4", "a5", "a6", "a7"]


def test_chunk_video_records_keeps_pure_noop_chunks() -> None:
    frames_THWC = np.arange(8 * 2 * 2 * 3, dtype=np.uint8).reshape(8, 2, 2, 3)
    actions_L = ["NO_OP"] * 8
    video_info_d = {
        "filename": "/tmp/v.mp4",
        "path": "x/v.mp4",
    }

    chunks_L = _chunk_video_records(
        video_tensor=frames_THWC,
        video_info=video_info_d,
        chunk_size=4,
        actions=actions_L,
    )

    assert len(chunks_L) == 2
    assert chunks_L[0]["actions"] == ["NO_OP"] * 4
    assert chunks_L[1]["actions"] == ["NO_OP"] * 4


def test_write_chunk_records_can_mix_multiple_videos(tmp_path: Path) -> None:
    chunk0_d = {
        "raw_video": np.zeros((4, 2, 2, 3), dtype=np.uint8).tobytes(),
        "sequence_length": 4,
        "actions": ["a0", "a1", "a2", "a3"],
        "path": "/abs/v0.mp4",
    }
    chunk1_d = {
        "raw_video": np.ones((4, 2, 2, 3), dtype=np.uint8).tobytes(),
        "sequence_length": 4,
        "actions": ["b0", "b1", "b2", "b3"],
        "path": "/abs/v1.mp4",
    }
    out_rows_L = _write_chunk_records(
        chunk_records=[chunk0_d, chunk1_d],
        output_folder=str(tmp_path),
        chunks_per_file=4,
        worker_idx=0,
        start_file_index=0,
    )
    assert len(out_rows_L) == 1

    rec_path_p = Path(out_rows_L[0]["filename"])
    reader = ArrayRecordReader(str(rec_path_p))
    assert reader.num_records() == 2
    rec0_d = pickle.loads(reader.read())
    rec1_d = pickle.loads(reader.read())
    reader.close()
    assert {rec0_d["path"], rec1_d["path"]} == {"/abs/v0.mp4", "/abs/v1.mp4"}


def test_filter_black_frames_ignores_top_bar_fraction() -> None:
    frames_THWC = np.full((3, 4, 4, 3), 255, dtype=np.uint8)
    frames_THWC[1, 1:, :, :] = 0
    actions_L = ["a0", "a1", "a2"]

    no_crop_segments = _filter_black_frames(
        frames=frames_THWC,
        actions=actions_L,
        threshold=10.0,
        black_ratio=0.95,
        top_bar_fraction=0.0,
    )
    with_crop_segments = _filter_black_frames(
        frames=frames_THWC,
        actions=actions_L,
        threshold=10.0,
        black_ratio=0.95,
        top_bar_fraction=0.25,
    )

    assert len(no_crop_segments) == 1
    assert len(with_crop_segments) == 2
    assert with_crop_segments[0][1] == ["a0"]
    assert with_crop_segments[1][1] == ["a2"]


def test_drop_no_op_frames_keeps_all_when_prob_zero() -> None:
    frames_THWC = np.arange(6 * 2 * 2 * 3, dtype=np.uint8).reshape(6, 2, 2, 3)
    actions_L = ["NO_OP", "MOUSE:1,0,0", "NO_OP", "MOUSE:0,0,0 ; W", "NO_OP", "NO_OP"]

    out_frames, out_actions = _drop_no_op_frames(
        frames=frames_THWC,
        actions=actions_L,
        drop_prob_f=0.0,
        rng=np.random.default_rng(123),
    )

    assert np.array_equal(out_frames, frames_THWC)
    assert out_actions == actions_L


def test_drop_no_op_frames_drops_only_noop_when_prob_one() -> None:
    frames_THWC = np.arange(6 * 2 * 2 * 3, dtype=np.uint8).reshape(6, 2, 2, 3)
    actions_L = ["NO_OP", "MOUSE:1,0,0", "NO_OP", "MOUSE:0,0,0 ; W", "NO_OP", "NO_OP"]

    out_frames, out_actions = _drop_no_op_frames(
        frames=frames_THWC,
        actions=actions_L,
        drop_prob_f=1.0,
        rng=np.random.default_rng(123),
    )

    assert out_actions == ["MOUSE:1,0,0", "MOUSE:0,0,0 ; W"]
    assert out_frames.shape[0] == len(out_actions)
    assert np.array_equal(out_frames[0], frames_THWC[1])
    assert np.array_equal(out_frames[1], frames_THWC[3])


def test_drop_no_op_frames_is_deterministic_for_seed() -> None:
    frames_THWC = np.arange(100 * 2 * 2 * 3, dtype=np.uint8).reshape(100, 2, 2, 3)
    actions_L = ["NO_OP" if i % 2 == 0 else "MOUSE:1,0,0" for i in range(100)]

    out_a_frames, out_a_actions = _drop_no_op_frames(
        frames=frames_THWC,
        actions=actions_L,
        drop_prob_f=0.95,
        rng=np.random.default_rng(11),
    )
    out_b_frames, out_b_actions = _drop_no_op_frames(
        frames=frames_THWC,
        actions=actions_L,
        drop_prob_f=0.95,
        rng=np.random.default_rng(11),
    )

    assert np.array_equal(out_a_frames, out_b_frames)
    assert out_a_actions == out_b_actions


def test_drop_no_op_frames_rejects_invalid_probability() -> None:
    frames_THWC = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    actions_L = ["NO_OP", "NO_OP", "NO_OP"]

    with pytest.raises(ValueError, match="drop_prob_f must be in \\[0, 1\\]"):
        _drop_no_op_frames(
            frames=frames_THWC,
            actions=actions_L,
            drop_prob_f=-0.1,
            rng=np.random.default_rng(0),
        )

    with pytest.raises(ValueError, match="drop_prob_f must be in \\[0, 1\\]"):
        _drop_no_op_frames(
            frames=frames_THWC,
            actions=actions_L,
            drop_prob_f=1.1,
            rng=np.random.default_rng(0),
        )


def test_process_video_shard_mixes_chunks_across_videos(tmp_path: Path) -> None:
    original_preprocess = _MODULE.preprocess_video
    seen_drop_probs: list[float] = []
    seen_drop_seeds: list[int] = []
    seen_mouse_delta_clips: list[int] = []
    seen_mouse_scroll_clips: list[int] = []

    def _fake_preprocess(
        idx: int,
        video_info: dict[str, object],
        target_width: int,
        target_height: int,
        target_fps: int,
        chunk_size: int,
        top_bar_fraction: float,
        black_ratio: float,
        drop_no_op_prob: float = 0.0,
        drop_no_op_seed: int = 0,
        mouse_delta_clip: int = 64,
        mouse_scroll_clip: int = 5,
        decode_timeout_sec: int = 0,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        del (
            idx,
            target_width,
            target_height,
            target_fps,
            chunk_size,
            top_bar_fraction,
            black_ratio,
            decode_timeout_sec,
        )
        seen_drop_probs.append(float(drop_no_op_prob))
        seen_drop_seeds.append(int(drop_no_op_seed))
        seen_mouse_delta_clips.append(int(mouse_delta_clip))
        seen_mouse_scroll_clips.append(int(mouse_scroll_clip))
        chunk_d = {
            "raw_video": np.zeros((4, 2, 2, 3), dtype=np.uint8).tobytes(),
            "sequence_length": 4,
            "actions": ["x0", "x1", "x2", "x3"],
            "path": str(video_info["path"]),
        }
        return [chunk_d], []

    try:
        _MODULE.preprocess_video = _fake_preprocess
        shard_args = [
            (
                0,
                {"filename": "/a.mp4", "path": "/abs/a.mp4"},
                str(tmp_path),
                160,
                90,
                10,
                4,
                4,
                0.15,
                0.95,
                0.95,
                10,
                12,
                2,
                0,
            ),
            (
                1,
                {"filename": "/b.mp4", "path": "/abs/b.mp4"},
                str(tmp_path),
                160,
                90,
                10,
                4,
                4,
                0.15,
                0.95,
                0.95,
                11,
                12,
                2,
                0,
            ),
        ]
        out_rows_L = _MODULE._process_video_shard(worker_idx=0, shard_args=shard_args)
    finally:
        _MODULE.preprocess_video = original_preprocess

    assert len(out_rows_L) == 1
    rec_path_p = Path(out_rows_L[0]["filename"])
    reader = ArrayRecordReader(str(rec_path_p))
    assert reader.num_records() == 2
    rec0_d = pickle.loads(reader.read())
    rec1_d = pickle.loads(reader.read())
    reader.close()
    assert {rec0_d["path"], rec1_d["path"]} == {"/abs/a.mp4", "/abs/b.mp4"}
    assert seen_drop_probs == [0.95, 0.95]
    assert seen_drop_seeds == [10, 11]
    assert seen_mouse_delta_clips == [12, 12]
    assert seen_mouse_scroll_clips == [2, 2]


def test_process_video_shard_forwards_decode_timeout(tmp_path: Path) -> None:
    original_preprocess = _MODULE.preprocess_video
    seen_decode_timeouts: list[int] = []

    def _fake_preprocess(
        idx: int,
        video_info: dict[str, object],
        target_width: int,
        target_height: int,
        target_fps: int,
        chunk_size: int,
        top_bar_fraction: float,
        black_ratio: float,
        drop_no_op_prob: float = 0.0,
        drop_no_op_seed: int = 0,
        mouse_delta_clip: int = 64,
        mouse_scroll_clip: int = 5,
        decode_timeout_sec: int = 0,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        del (
            idx,
            video_info,
            target_width,
            target_height,
            target_fps,
            chunk_size,
            top_bar_fraction,
            black_ratio,
            drop_no_op_prob,
            drop_no_op_seed,
            mouse_delta_clip,
            mouse_scroll_clip,
        )
        seen_decode_timeouts.append(int(decode_timeout_sec))
        return [], []

    try:
        _MODULE.preprocess_video = _fake_preprocess
        _MODULE._process_video_shard(
            worker_idx=0,
            shard_args=[
                (
                    0,
                    {"filename": "/a.mp4", "path": "/abs/a.mp4"},
                    str(tmp_path),
                    160,
                    90,
                    10,
                    4,
                    4,
                    0.15,
                    0.95,
                    0.95,
                    10,
                    12,
                    2,
                    123,
                )
            ],
        )
    finally:
        _MODULE.preprocess_video = original_preprocess

    assert seen_decode_timeouts == [123]
