import pytest

from idm.data import derive_record_key


def test_derive_record_key_prefers_relative_path():
    rec_d = {
        "relative_path": "a/b/c.mp4",
        "user_id": "u",
        "session_id": "s",
        "seg_idx": 1,
    }
    assert derive_record_key(rec_d) == "a/b/c.mp4"


def test_derive_record_key_user_session_seg_fallback():
    rec_d = {"user_id": "u", "session_id": "s", "seg_idx": 7}
    assert derive_record_key(rec_d) == "u|s|7"


def test_derive_record_key_video_name_seg_fallback():
    rec_d = {"video_file_name": "/tmp/v.mp4", "seg_idx": 3}
    assert derive_record_key(rec_d) == "/tmp/v.mp4|3"


def test_derive_record_key_errors_when_insufficient():
    with pytest.raises(ValueError):
        derive_record_key({"sequence_length": 10})
