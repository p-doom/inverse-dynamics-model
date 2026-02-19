import importlib.util
import json
from pathlib import Path
import sys

import pytest

_TRAIN_PATH = Path(__file__).resolve().parents[1] / "train.py"
_TRAIN_SPEC = importlib.util.spec_from_file_location("train_module_for_meta", _TRAIN_PATH)
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
