from __future__ import annotations

import pytest

from idm.utils.collator import CollatorPrefetchIterator


class _FakeRawIterator:
    def __init__(self, batch_L):
        self._batch_L = list(batch_L)
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self._batch_L):
            raise StopIteration
        batch_d = self._batch_L[self._idx]
        self._idx += 1
        return batch_d

    def get_state(self) -> bytes:
        return f"idx={self._idx}".encode("utf-8")


class _FakeCollator:
    def __call__(self, batch_d):
        return {"value": int(batch_d["value"]) + 1}


def test_collator_prefetch_iterator_collates_and_tracks_state():
    raw_it = _FakeRawIterator([{"value": 1}, {"value": 2}])
    prefetch_it = CollatorPrefetchIterator(raw_it=raw_it, collator=_FakeCollator())
    try:
        assert next(prefetch_it) == {"value": 2}
        assert next(prefetch_it) == {"value": 3}
        assert prefetch_it.last_state() == b"idx=2"
        with pytest.raises(StopIteration):
            next(prefetch_it)
    finally:
        prefetch_it.close()
