from __future__ import annotations

import pickle
from pathlib import Path
import importlib.util

from array_record.python.array_record_module import ArrayRecordReader
import msgpack
import numpy as np

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "idm_data"
    / "video_to_array_records.py"
)
_SPEC = importlib.util.spec_from_file_location("video_to_array_records", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_actions_from_keylog_file = _MODULE._actions_from_keylog_file
_chunk_video_records = _MODULE._chunk_video_records
_write_chunk_records = _MODULE._write_chunk_records
_get_keylog_path = _MODULE._get_keylog_path


def test_get_keylog_path() -> None:
    filename_s = "/tmp/uploads/0.1.0/u123/recordings/recording_abc-def_seg0007.mp4"
    out_p = _get_keylog_path(filename_s)
    assert out_p == Path("/tmp/uploads/0.1.0/u123/keylogs/input_abc-def_seg0007.msgpack")


def test_actions_from_keylog_file_aligns_to_target_fps(tmp_path: Path) -> None:
    entries_L = [
        [0, ["KeyPress", [0, "KeyW"]]],
        [100_000, ["MouseMove", [5.0, 0.0]]],
        [700_000, ["MouseScroll", [0.0, -1.0]]],
        [900_000, ["MouseMove", [0.0, 0.0]]],
    ]
    keylog_p = tmp_path / "k.msgpack"
    keylog_p.write_bytes(msgpack.packb(entries_L, use_bin_type=True))

    actions_L = _actions_from_keylog_file(keylog_p, n_frames=10, target_fps=10)

    assert len(actions_L) == 10
    assert actions_L[0] == "KEY_DOWN:W"
    assert actions_L[1] == "MOUSE_MOVE"
    assert actions_L[7] == "MOUSE_SCROLL"
    assert actions_L[9] == "NO_OP"


def test_actions_from_empty_keylog_file_is_noop(tmp_path: Path) -> None:
    keylog_p = tmp_path / "empty.msgpack"
    keylog_p.write_bytes(msgpack.packb([], use_bin_type=True))

    actions_L = _actions_from_keylog_file(keylog_p, n_frames=6, target_fps=10)

    assert actions_L == ["NO_OP"] * 6


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


def test_process_video_shard_mixes_chunks_across_videos(tmp_path: Path) -> None:
    original_preprocess = _MODULE.preprocess_video

    def _fake_preprocess(
        idx: int,
        video_info: dict[str, object],
        target_width: int,
        target_height: int,
        target_fps: int,
        chunk_size: int,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        del idx, target_width, target_height, target_fps, chunk_size
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
            (0, {"filename": "/a.mp4", "path": "/abs/a.mp4"}, str(tmp_path), 160, 90, 10, 4, 4),
            (1, {"filename": "/b.mp4", "path": "/abs/b.mp4"}, str(tmp_path), 160, 90, 10, 4, 4),
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
