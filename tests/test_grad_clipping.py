from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


_TRAIN_PATH = Path(__file__).resolve().parents[1] / "train.py"
_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "train_module_for_grad_clip", _TRAIN_PATH
)
assert _TRAIN_SPEC is not None and _TRAIN_SPEC.loader is not None
_TRAIN_MOD = importlib.util.module_from_spec(_TRAIN_SPEC)
sys.modules["train_module_for_grad_clip"] = _TRAIN_MOD
_TRAIN_SPEC.loader.exec_module(_TRAIN_MOD)


def _total_grad_norm(params: list[torch.nn.Parameter]) -> float:
    sq_sum = 0.0
    for param in params:
        assert param.grad is not None
        sq_sum += float(param.grad.detach().pow(2).sum().item())
    return sq_sum**0.5


def test_grad_norm_and_clip_clips_and_returns_pre_clip_norm():
    p0 = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
    p1 = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
    p0.grad = torch.tensor([3.0], dtype=torch.float32)
    p1.grad = torch.tensor([4.0], dtype=torch.float32)
    params = [p0, p1]

    grad_norm_f = _TRAIN_MOD._grad_norm_and_clip(params, max_grad_norm=2.0)

    assert grad_norm_f == pytest.approx(5.0)
    assert _total_grad_norm(params) == pytest.approx(2.0, rel=1e-5)


def test_grad_norm_and_clip_with_zero_max_norm_keeps_grads_unchanged():
    p0 = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
    p1 = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
    p0.grad = torch.tensor([3.0], dtype=torch.float32)
    p1.grad = torch.tensor([4.0], dtype=torch.float32)
    params = [p0, p1]

    grad_norm_f = _TRAIN_MOD._grad_norm_and_clip(params, max_grad_norm=0.0)

    assert grad_norm_f == pytest.approx(5.0)
    assert float(p0.grad.item()) == pytest.approx(3.0)
    assert float(p1.grad.item()) == pytest.approx(4.0)


def test_grad_norm_and_clip_returns_zero_without_gradients():
    p0 = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    p1 = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float32))

    grad_norm_f = _TRAIN_MOD._grad_norm_and_clip([p0, p1], max_grad_norm=1.0)

    assert grad_norm_f == 0.0


def test_train_args_exposes_max_grad_norm_with_default():
    args = _TRAIN_MOD.Args()
    assert args.max_grad_norm == pytest.approx(1.0)
