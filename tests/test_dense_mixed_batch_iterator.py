from __future__ import annotations

import pickle

import numpy as np
import pytest

from idm.utils.data import DenseMixedBatchIterator, concat_batches


class _StatefulIter:
    def __init__(self, batches_L: list[dict[str, object]]):
        self._batches_L = batches_L
        self._idx_i = 0

    def __iter__(self) -> "_StatefulIter":
        return self

    def __next__(self) -> dict[str, object]:
        if self._idx_i >= len(self._batches_L):
            raise StopIteration
        out_d = self._batches_L[self._idx_i]
        self._idx_i += 1
        return out_d

    def get_state(self) -> bytes:
        return pickle.dumps(self._idx_i)

    def set_state(self, state_b: bytes) -> None:
        self._idx_i = int(pickle.loads(state_b))


def _make_batch(start_i: int, batch_n: int) -> dict[str, object]:
    return {
        "frames": np.arange(start_i, start_i + batch_n, dtype=np.int64).reshape(
            batch_n, 1
        ),
        "target_text": [f"t{idx_i}" for idx_i in range(start_i, start_i + batch_n)],
    }


def test_concat_batches_concatenates_all_supported_fields():
    base_batch_d = _make_batch(start_i=0, batch_n=3)
    dense_batch_d = _make_batch(start_i=100, batch_n=1)
    out_d = concat_batches(base_batch_d=base_batch_d, dense_batch_d=dense_batch_d)

    assert out_d["frames"].shape == (4, 1)
    assert out_d["target_text"] == ["t0", "t1", "t2", "t100"]


def test_concat_batches_rejects_mismatched_keys():
    with pytest.raises(ValueError):
        concat_batches(
            base_batch_d={"frames": np.zeros((1, 1), dtype=np.int64)},
            dense_batch_d={"target_text": ["x"]},
        )


def test_dense_mixed_batch_iterator_stops_when_any_stream_is_exhausted():
    base_it = _StatefulIter([_make_batch(0, 3), _make_batch(3, 3)])
    dense_it = _StatefulIter([_make_batch(100, 1)])
    mixed_it = DenseMixedBatchIterator(base_it=base_it, dense_it=dense_it)

    first_batch_d = next(mixed_it)
    assert first_batch_d["target_text"] == ["t0", "t1", "t2", "t100"]

    with pytest.raises(StopIteration):
        next(mixed_it)


def test_dense_mixed_batch_iterator_state_roundtrip_restores_position():
    base_batches_L = [_make_batch(0, 3), _make_batch(3, 3), _make_batch(6, 3)]
    dense_batches_L = [_make_batch(100, 1), _make_batch(101, 1), _make_batch(102, 1)]

    it_a = DenseMixedBatchIterator(
        base_it=_StatefulIter(base_batches_L),
        dense_it=_StatefulIter(dense_batches_L),
    )
    _ = next(it_a)
    state_b = it_a.get_state()
    second_a_d = next(it_a)

    it_b = DenseMixedBatchIterator(
        base_it=_StatefulIter(base_batches_L),
        dense_it=_StatefulIter(dense_batches_L),
    )
    it_b.set_state(state_b)
    second_b_d = next(it_b)

    assert second_a_d["target_text"] == second_b_d["target_text"]
    assert np.array_equal(second_a_d["frames"], second_b_d["frames"])
