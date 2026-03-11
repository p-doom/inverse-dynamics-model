import inspect
import importlib.util
from pathlib import Path
import sys

from idm.utils import data

_TRAIN_PATH = Path(__file__).resolve().parents[1] / "idm" / "train.py"
_TRAIN_SPEC = importlib.util.spec_from_file_location("train_module", _TRAIN_PATH)
assert _TRAIN_SPEC is not None and _TRAIN_SPEC.loader is not None
_TRAIN_MOD = importlib.util.module_from_spec(_TRAIN_SPEC)
sys.modules["train_module"] = _TRAIN_MOD
_TRAIN_SPEC.loader.exec_module(_TRAIN_MOD)


def test_train_args_does_not_expose_actions_path():
    assert "actions_path" not in _TRAIN_MOD.Args.__annotations__


def test_train_args_does_not_expose_split():
    assert "split" not in _TRAIN_MOD.Args.__annotations__


def test_train_args_exposes_loss_weighting_flags_with_defaults():
    anns = _TRAIN_MOD.Args.__annotations__
    assert "no_op_loss_weight" in anns
    assert "mouse_loss_weight" in anns
    args = _TRAIN_MOD.Args()
    assert args.no_op_loss_weight == 1.0
    assert args.mouse_loss_weight == 1.0


def test_train_args_exposes_action_upsample_random_fraction_with_default():
    anns = _TRAIN_MOD.Args.__annotations__
    assert "train_action_upsample_random_fraction" in anns
    args = _TRAIN_MOD.Args()
    assert args.train_action_upsample_random_fraction == 1.0


def test_get_dataloader_does_not_accept_actions_map():
    sig = inspect.signature(data.get_dataloader)
    assert "actions_map_d" not in sig.parameters


def test_count_valid_records_does_not_accept_actions_map():
    sig = inspect.signature(data.count_valid_records)
    assert "actions_map_d" not in sig.parameters
