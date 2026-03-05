import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

_TRAIN_PATH = Path(__file__).resolve().parents[1] / "idm" / "train.py"
_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "train_module_for_meta", _TRAIN_PATH
)
assert _TRAIN_SPEC is not None and _TRAIN_SPEC.loader is not None
_TRAIN_MOD = importlib.util.module_from_spec(_TRAIN_SPEC)
sys.modules["train_module_for_meta"] = _TRAIN_MOD
_TRAIN_SPEC.loader.exec_module(_TRAIN_MOD)


def test_metadata_shape_check_passes_when_matching(tmp_path: Path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"target_height": 90, "target_width": 160, "target_channels": 3}),
        encoding="utf-8",
    )
    args = _TRAIN_MOD.Args(data_root=str(tmp_path), image_h=90, image_w=160, image_c=3)
    _TRAIN_MOD._assert_image_hwc_matches_metadata(args)


def test_metadata_shape_check_raises_on_mismatch(tmp_path: Path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"target_height": 512, "target_width": 512, "target_channels": 3}),
        encoding="utf-8",
    )
    args = _TRAIN_MOD.Args(data_root=str(tmp_path), image_h=90, image_w=160, image_c=3)
    with pytest.raises(ValueError, match="Image shape mismatch"):
        _TRAIN_MOD._assert_image_hwc_matches_metadata(args)


def test_metadata_shape_check_raises_when_metadata_missing(tmp_path: Path):
    args = _TRAIN_MOD.Args(data_root=str(tmp_path), image_h=90, image_w=160, image_c=3)
    with pytest.raises(ValueError, match="metadata.json not found"):
        _TRAIN_MOD._assert_image_hwc_matches_metadata(args)


class _OneBatchIterator:
    def __init__(self):
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        self._done = True
        return {"x": 1}


def test_next_synced_batch_single_rank_iterates_then_stops():
    it = _OneBatchIterator()
    batch_d, stop_b = _TRAIN_MOD._next_synced_batch(
        batch_it=it,
        world_i=1,
        device=torch.device("cpu"),
    )
    assert not stop_b
    assert batch_d == {"x": 1}

    batch_d, stop_b = _TRAIN_MOD._next_synced_batch(
        batch_it=it,
        world_i=1,
        device=torch.device("cpu"),
    )
    assert stop_b
    assert batch_d is None


def test_next_synced_batch_multi_rank_truncates_when_any_rank_exhausted(monkeypatch):
    # Simulate this rank having a batch while another rank is exhausted.
    class _AlwaysBatchIterator:
        def __iter__(self):
            return self

        def __next__(self):
            return {"x": 1}

    def _fake_all_reduce(tensor_t, op=None):
        del op
        tensor_t.fill_(0.0)

    monkeypatch.setattr(_TRAIN_MOD.dist, "all_reduce", _fake_all_reduce)

    batch_d, stop_b = _TRAIN_MOD._next_synced_batch(
        batch_it=_AlwaysBatchIterator(),
        world_i=2,
        device=torch.device("cpu"),
    )
    assert stop_b
    assert batch_d is None


def test_train_batch_split_sizes_returns_expected_sizes():
    base_batch_i, dense_batch_i = _TRAIN_MOD._train_batch_split_sizes(
        global_batch_size=16,
        world_i=4,
        dense_mix_fraction=0.25,
    )
    assert base_batch_i == 12
    assert dense_batch_i == 4


def test_train_batch_split_sizes_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        _TRAIN_MOD._train_batch_split_sizes(
            global_batch_size=16,
            world_i=4,
            dense_mix_fraction=1.0,
        )


def test_train_batch_split_sizes_rejects_unrealizable_per_rank_split():
    with pytest.raises(ValueError):
        _TRAIN_MOD._train_batch_split_sizes(
            global_batch_size=4,
            world_i=4,
            dense_mix_fraction=0.25,
        )
