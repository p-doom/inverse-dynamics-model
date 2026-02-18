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
_chunk_and_save_video = _MODULE._chunk_and_save_video
_keylog_path_for_video_info = _MODULE._keylog_path_for_video_info


def test_keylog_path_for_video_info() -> None:
    video_info_d = {
        "path": "/tmp/uploads/0.1.0/u123/recordings/recording_abc-def_seg0007.mp4",
        "session_id": "abc-def",
        "seg_idx": 7,
    }
    out_p = _keylog_path_for_video_info(video_info_d)
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


def test_chunk_and_save_video_embeds_action_slices(tmp_path: Path) -> None:
    frames_THWC = np.arange(8 * 2 * 2 * 3, dtype=np.uint8).reshape(8, 2, 2, 3)
    actions_L = [f"a{idx_i}" for idx_i in range(8)]
    video_info_d = {
        "path": "/tmp/v.mp4",
        "relative_path": "x/v.mp4",
        "user_id": "u",
        "session_id": "s",
        "seg_idx": 0,
        "split": "train",
    }

    out_rows_L = _chunk_and_save_video(
        video_tensor=frames_THWC,
        video_info=video_info_d,
        output_folder=str(tmp_path),
        chunk_size=4,
        chunks_per_file=100,
        file_index=0,
        actions=actions_L,
    )
    assert len(out_rows_L) == 1

    rec_path_p = Path(out_rows_L[0]["path"])
    reader = ArrayRecordReader(str(rec_path_p))
    assert reader.num_records() == 2
    rec0_d = pickle.loads(reader.read())
    rec1_d = pickle.loads(reader.read())
    reader.close()

    assert rec0_d["actions"] == ["a0", "a1", "a2", "a3"]
    assert rec1_d["actions"] == ["a4", "a5", "a6", "a7"]
