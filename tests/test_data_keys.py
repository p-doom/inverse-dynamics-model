import pytest

from idm.data import derive_record_key


def test_derive_record_key_prefers_path():
    rec_d = {
        "path": "a/b/c.mp4",
    }
    assert derive_record_key(rec_d) == "a/b/c.mp4"


def test_derive_record_key_errors_when_insufficient():
    with pytest.raises(ValueError):
        derive_record_key({"sequence_length": 10})
