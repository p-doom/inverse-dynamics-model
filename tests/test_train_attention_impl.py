import importlib.util
from pathlib import Path
import sys

import pytest
import torch


_TRAIN_PATH = Path(__file__).resolve().parents[1] / "train.py"
_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "train_module_for_attn_impl", _TRAIN_PATH
)
assert _TRAIN_SPEC is not None and _TRAIN_SPEC.loader is not None
_TRAIN_MOD = importlib.util.module_from_spec(_TRAIN_SPEC)
sys.modules["train_module_for_attn_impl"] = _TRAIN_MOD
_TRAIN_SPEC.loader.exec_module(_TRAIN_MOD)


def test_default_attn_implementation_is_flash_attention_2():
    args = _TRAIN_MOD.Args()
    assert args.attn_implementation == "flash_attention_2"


def test_build_model_rejects_unsupported_attn_implementation():
    args = _TRAIN_MOD.Args(
        attn_implementation="eager",
        use_lora=False,
    )
    with pytest.raises(ValueError, match="Unsupported --attn-implementation"):
        _TRAIN_MOD._build_model(
            args=args,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )


def test_build_model_auto_does_not_pass_attn_implementation(monkeypatch):
    captured_kwargs = {}

    class _FakeModel:
        def __init__(self):
            self.config = type("Cfg", (), {"use_cache": True})()

        def to(self, _: torch.device):
            return self

    def _fake_from_pretrained(model_id: str, **kwargs):
        captured_kwargs["model_id"] = model_id
        captured_kwargs.update(kwargs)
        return _FakeModel()

    monkeypatch.setattr(
        _TRAIN_MOD.Qwen3VLForConditionalGeneration,
        "from_pretrained",
        _fake_from_pretrained,
    )
    args = _TRAIN_MOD.Args(
        attn_implementation="auto",
        use_lora=False,
        grad_checkpointing=False,
    )
    _TRAIN_MOD._build_model(
        args=args,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert captured_kwargs["model_id"] == args.model_id
    assert captured_kwargs["trust_remote_code"] is True
    assert "attn_implementation" not in captured_kwargs
