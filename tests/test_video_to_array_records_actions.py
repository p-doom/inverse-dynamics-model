from __future__ import annotations

import pickle
from pathlib import Path
import importlib.util

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
_write_chunk_records = _MODULE._write_chunk_records
_get_keylog_path = _MODULE._get_keylog_path
_filter_black_frames = _MODULE._filter_black_frames
_filter_identical_noop_frames = _MODULE._filter_identical_noop_frames
_clip_abs_from_percentile = _MODULE._clip_abs_from_percentile


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

    assert actions_L == ["MOUSE:1000,-1000,-5"]


def test_clip_abs_from_percentile_reduces_outlier_clipping_window() -> None:
    clip_abs_i = _clip_abs_from_percentile(
        values_f=[0.0, 5.0, 5.0, 5.0, 5_000.0],
        quant_unit_f=5.0,
        max_clip_abs_i=1000,
        percentile_f=99.5,
    )

    assert clip_abs_i < 1000
    assert clip_abs_i >= 1


def test_actions_percentile_clipping_downweights_extreme_outlier_frame() -> None:
    entries_L = [
        [0, ["MouseMove", [5.0, 0.0]]],
        [100_000, ["MouseMove", [5.0, 0.0]]],
        [200_000, ["MouseMove", [5_000.0, 0.0]]],
    ]

    actions_L = _actions_from_keylog_entries(
        entries_L,
        n_frames=3,
        target_fps=10,
        mouse_delta_clip_percentile_f=99.5,
    )

    assert actions_L[0] == "MOUSE:1,0,0"
    assert actions_L[1] == "MOUSE:1,0,0"
    assert actions_L[2].startswith("MOUSE:")
    assert actions_L[2] != "MOUSE:1000,0,0"


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
    rec0_d = chunks_L[0]
    rec1_d = chunks_L[1]

    assert rec0_d["actions"] == ["a0", "a1", "a2", "a3"]
    assert rec1_d["actions"] == ["a4", "a5", "a6", "a7"]


def test_chunk_video_records_filters_pure_noop_chunks_when_enabled() -> None:
    frames_THWC = np.arange(8 * 2 * 2 * 3, dtype=np.uint8).reshape(8, 2, 2, 3)
    actions_L = [
        "NO_OP",
        "NO_OP",
        "NO_OP",
        "NO_OP",
        "NO_OP",
        "KEY_DOWN:W",
        "NO_OP",
        "NO_OP",
    ]
    video_info_d = {
        "filename": "/tmp/v.mp4",
        "path": "x/v.mp4",
    }

    chunks_L = _chunk_video_records(
        video_tensor=frames_THWC,
        video_info=video_info_d,
        chunk_size=4,
        actions=actions_L,
        filter_pure_noop_chunks=True,
    )

    assert len(chunks_L) == 1
    assert chunks_L[0]["actions"] == ["NO_OP", "KEY_DOWN:W", "NO_OP", "NO_OP"]


def test_chunk_video_records_keeps_pure_noop_chunks_by_default() -> None:
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


def test_filter_identical_noop_frames_keeps_run_boundaries() -> None:
    frames_THWC = np.zeros((7, 2, 2, 3), dtype=np.uint8)
    frames_THWC[0] = 10
    frames_THWC[1] = 20
    frames_THWC[2] = 20
    frames_THWC[3] = 20
    frames_THWC[4] = 30
    frames_THWC[5] = 30
    frames_THWC[6] = 30
    actions_L = [
        "MOUSE:0,0,0 ; W",
        "NO_OP",
        "NO_OP",
        "NO_OP",
        "NO_OP",
        "NO_OP",
        "NO_OP",
    ]

    out_frames, out_actions = _filter_identical_noop_frames(frames_THWC, actions_L)

    assert out_actions == [
        "MOUSE:0,0,0 ; W",
        "NO_OP",
        "NO_OP",
        "NO_OP",
        "NO_OP",
    ]
    assert out_frames.shape[0] == len(out_actions)
    assert np.array_equal(out_frames[1], out_frames[2])
    assert np.array_equal(out_frames[3], out_frames[4])


def test_filter_identical_noop_frames_keeps_non_noop_identical_frames() -> None:
    frames_THWC = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    actions_L = [
        "MOUSE:0,0,0 ; Ctrl",
        "MOUSE:0,0,0 ; Ctrl",
        "NO_OP",
        "NO_OP",
    ]

    out_frames, out_actions = _filter_identical_noop_frames(frames_THWC, actions_L)

    assert out_actions == actions_L
    assert out_frames.shape[0] == 4


def test_filter_identical_noop_frames_uses_mad_tolerance() -> None:
    frames_THWC = np.zeros((5, 10, 10, 3), dtype=np.uint8)
    frames_THWC[0] = 10
    frames_THWC[1] = 20
    frames_THWC[2] = frames_THWC[1]
    frames_THWC[2, 0, 0, 0] = 21  # MAD ~= 0.0033
    frames_THWC[3] = frames_THWC[2]
    frames_THWC[4] = 30
    actions_L = ["MOUSE:0,0,0 ; W", "NO_OP", "NO_OP", "NO_OP", "MOUSE:0,0,0 ; Ctrl"]

    out_frames, out_actions = _filter_identical_noop_frames(
        frames_THWC, actions_L, mad_threshold_f=0.01
    )

    assert out_actions == [
        "MOUSE:0,0,0 ; W",
        "NO_OP",
        "NO_OP",
        "MOUSE:0,0,0 ; Ctrl",
    ]
    assert out_frames.shape[0] == len(out_actions)


def test_filter_identical_noop_frames_tolerance_is_configurable() -> None:
    frames_THWC = np.zeros((4, 10, 10, 3), dtype=np.uint8)
    frames_THWC[1] = 20
    frames_THWC[2] = frames_THWC[1]
    frames_THWC[2, 0, 0, 0] = 21  # MAD ~= 0.0033
    actions_L = ["NO_OP", "NO_OP", "NO_OP", "NO_OP"]

    out_frames, out_actions = _filter_identical_noop_frames(
        frames_THWC, actions_L, mad_threshold_f=0.001
    )

    assert out_actions == actions_L
    assert out_frames.shape[0] == len(actions_L)


def test_filter_identical_noop_frames_does_not_drop_meaningful_changes() -> None:
    frames_THWC = np.zeros((4, 10, 10, 3), dtype=np.uint8)
    frames_THWC[2, :, :, 0] = 40  # MAD ~= 13.33
    actions_L = ["NO_OP", "NO_OP", "NO_OP", "NO_OP"]

    out_frames, out_actions = _filter_identical_noop_frames(
        frames_THWC, actions_L, mad_threshold_f=0.01
    )

    assert out_actions == actions_L
    assert out_frames.shape[0] == len(actions_L)


def test_filter_identical_noop_frames_rejects_negative_tolerance() -> None:
    frames_THWC = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    actions_L = ["NO_OP", "NO_OP", "NO_OP"]

    with pytest.raises(ValueError, match="mad_threshold_f must be non-negative"):
        _filter_identical_noop_frames(frames_THWC, actions_L, mad_threshold_f=-0.1)


def test_chunk_video_records_after_identical_noop_filter_keeps_fixed_sequence_length() -> (
    None
):
    frames_THWC = np.zeros((10, 2, 2, 3), dtype=np.uint8)
    frames_THWC[0] = 10
    frames_THWC[9] = 11
    actions_L = ["MOUSE:0,0,0 ; W"] + ["NO_OP"] * 8 + ["MOUSE:0,0,0 ; Ctrl"]
    video_info_d = {"filename": "/tmp/v.mp4", "path": "x/v.mp4"}

    filtered_frames, filtered_actions = _filter_identical_noop_frames(
        frames_THWC, actions_L
    )
    chunks_L = _chunk_video_records(
        video_tensor=filtered_frames,
        video_info=video_info_d,
        chunk_size=4,
        actions=filtered_actions,
    )

    assert len(chunks_L) == 1
    assert int(chunks_L[0]["sequence_length"]) == 4
    assert len(chunks_L[0]["actions"]) == 4


def test_process_video_shard_mixes_chunks_across_videos(tmp_path: Path) -> None:
    original_preprocess = _MODULE.preprocess_video
    seen_filter_flags: list[bool] = []

    def _fake_preprocess(
        idx: int,
        video_info: dict[str, object],
        target_width: int,
        target_height: int,
        target_fps: int,
        chunk_size: int,
        top_bar_fraction: float,
        black_ratio: float,
        filter_pure_noop_chunks: bool = False,
        filter_identical_noop_frames: bool = False,
        identical_noop_mad_threshold: float = 0.01,
        mouse_delta_clip_percentile: float = 99.5,
        mouse_scroll_clip_percentile: float = 99.5,
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
            filter_identical_noop_frames,
            identical_noop_mad_threshold,
            mouse_delta_clip_percentile,
            mouse_scroll_clip_percentile,
            decode_timeout_sec,
        )
        seen_filter_flags.append(filter_pure_noop_chunks)
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
                False,
                False,
                0.01,
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
                False,
                False,
                0.01,
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
    assert seen_filter_flags == [False, False]


def test_process_video_shard_forwards_filter_noop_flag(tmp_path: Path) -> None:
    original_preprocess = _MODULE.preprocess_video
    seen_filter_flags: list[bool] = []

    def _fake_preprocess(
        idx: int,
        video_info: dict[str, object],
        target_width: int,
        target_height: int,
        target_fps: int,
        chunk_size: int,
        top_bar_fraction: float,
        black_ratio: float,
        filter_pure_noop_chunks: bool = False,
        filter_identical_noop_frames: bool = False,
        identical_noop_mad_threshold: float = 0.01,
        mouse_delta_clip_percentile: float = 99.5,
        mouse_scroll_clip_percentile: float = 99.5,
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
            filter_identical_noop_frames,
            identical_noop_mad_threshold,
            mouse_delta_clip_percentile,
            mouse_scroll_clip_percentile,
            decode_timeout_sec,
        )
        seen_filter_flags.append(filter_pure_noop_chunks)
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
                    True,
                    False,
                    0.01,
                    0,
                )
            ],
        )
    finally:
        _MODULE.preprocess_video = original_preprocess

    assert seen_filter_flags == [True]


def test_process_video_shard_forwards_filter_identical_noop_flag(
    tmp_path: Path,
) -> None:
    original_preprocess = _MODULE.preprocess_video
    seen_filter_flags: list[bool] = []

    def _fake_preprocess(
        idx: int,
        video_info: dict[str, object],
        target_width: int,
        target_height: int,
        target_fps: int,
        chunk_size: int,
        top_bar_fraction: float,
        black_ratio: float,
        filter_pure_noop_chunks: bool = False,
        filter_identical_noop_frames: bool = False,
        identical_noop_mad_threshold: float = 0.01,
        mouse_delta_clip_percentile: float = 99.5,
        mouse_scroll_clip_percentile: float = 99.5,
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
            filter_pure_noop_chunks,
            identical_noop_mad_threshold,
            mouse_delta_clip_percentile,
            mouse_scroll_clip_percentile,
            decode_timeout_sec,
        )
        seen_filter_flags.append(filter_identical_noop_frames)
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
                    False,
                    True,
                    0.01,
                    0,
                )
            ],
        )
    finally:
        _MODULE.preprocess_video = original_preprocess

    assert seen_filter_flags == [True]


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
        filter_pure_noop_chunks: bool = False,
        filter_identical_noop_frames: bool = False,
        identical_noop_mad_threshold: float = 0.01,
        mouse_delta_clip_percentile: float = 99.5,
        mouse_scroll_clip_percentile: float = 99.5,
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
            filter_pure_noop_chunks,
            filter_identical_noop_frames,
            identical_noop_mad_threshold,
            mouse_delta_clip_percentile,
            mouse_scroll_clip_percentile,
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
                    False,
                    False,
                    0.01,
                    123,
                )
            ],
        )
    finally:
        _MODULE.preprocess_video = original_preprocess

    assert seen_decode_timeouts == [123]


def test_process_video_shard_forwards_identical_noop_mad_threshold(
    tmp_path: Path,
) -> None:
    original_preprocess = _MODULE.preprocess_video
    seen_thresholds: list[float] = []

    def _fake_preprocess(
        idx: int,
        video_info: dict[str, object],
        target_width: int,
        target_height: int,
        target_fps: int,
        chunk_size: int,
        top_bar_fraction: float,
        black_ratio: float,
        filter_pure_noop_chunks: bool = False,
        filter_identical_noop_frames: bool = False,
        identical_noop_mad_threshold: float = 0.01,
        mouse_delta_clip_percentile: float = 99.5,
        mouse_scroll_clip_percentile: float = 99.5,
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
            filter_pure_noop_chunks,
            filter_identical_noop_frames,
            mouse_delta_clip_percentile,
            mouse_scroll_clip_percentile,
            decode_timeout_sec,
        )
        seen_thresholds.append(float(identical_noop_mad_threshold))
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
                    False,
                    True,
                    0.0125,
                    0,
                )
            ],
        )
    finally:
        _MODULE.preprocess_video = original_preprocess

    assert seen_thresholds == [0.0125]
