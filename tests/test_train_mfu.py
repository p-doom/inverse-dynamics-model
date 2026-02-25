from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_TRAIN_PATH = Path(__file__).resolve().parents[1] / "idm" / "train.py"
_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "train_module_for_mfu", _TRAIN_PATH
)
assert _TRAIN_SPEC is not None and _TRAIN_SPEC.loader is not None
_TRAIN_MOD = importlib.util.module_from_spec(_TRAIN_SPEC)
sys.modules["train_module_for_mfu"] = _TRAIN_MOD
_TRAIN_SPEC.loader.exec_module(_TRAIN_MOD)


class _Obj:
    def __init__(self, **kwargs):
        for key_s, val in kwargs.items():
            setattr(self, key_s, val)


def test_flops_per_token_estimate_matches_standard_formula():
    got = _TRAIN_MOD._flops_per_token_estimate(
        n_params_i=1_000,
        n_layers_i=2,
        n_heads_i=4,
        head_dim_i=8,
        seq_len_f=16.0,
    )
    expected = (6 * 1_000) + (12 * 2 * 4 * 8 * 16.0)
    assert got == pytest.approx(expected)


def test_mfu_from_throughput_matches_expected_ratio():
    mfu_f = _TRAIN_MOD._mfu_from_throughput(
        n_params_i=1_000,
        n_layers_i=2,
        n_heads_i=4,
        head_dim_i=8,
        seq_len_f=16.0,
        tokens_per_s_f=2_000.0,
        peak_flops_f=1e9,
    )
    expected = _TRAIN_MOD._flops_per_token_estimate(
        n_params_i=1_000,
        n_layers_i=2,
        n_heads_i=4,
        head_dim_i=8,
        seq_len_f=16.0,
    )
    expected = (expected * 2_000.0) / 1e9
    assert mfu_f == pytest.approx(expected)


def test_transformer_dims_for_mfu_prefers_text_config():
    model = _Obj(
        config=_Obj(
            text_config=_Obj(
                num_hidden_layers=28,
                num_attention_heads=16,
                head_dim=128,
            ),
        )
    )
    assert _TRAIN_MOD._transformer_dims_for_mfu(model) == (28, 16, 128)


def test_transformer_dims_for_mfu_raises_on_invalid_dims():
    model = _Obj(
        config=_Obj(
            text_config=_Obj(
                num_hidden_layers=0,
                num_attention_heads=16,
                head_dim=128,
            ),
        )
    )
    with pytest.raises(ValueError, match="Invalid Qwen3-VL text_config dimensions"):
        _TRAIN_MOD._transformer_dims_for_mfu(model)


def test_peak_device_flops_known_and_unknown_cases():
    assert _TRAIN_MOD._peak_device_flops(
        "bf16", "NVIDIA H100 80GB HBM3"
    ) == pytest.approx(989e12)
    assert _TRAIN_MOD._peak_device_flops("bf16", "NVIDIA H100 PCIe") == pytest.approx(
        756e12
    )
    assert _TRAIN_MOD._peak_device_flops("bf16", "NVIDIA GeForce RTX 4090") is None
    assert _TRAIN_MOD._peak_device_flops("fp32", "NVIDIA H100 80GB HBM3") is None
