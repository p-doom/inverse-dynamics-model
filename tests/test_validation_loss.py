from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


_TRAIN_PATH = Path(__file__).resolve().parents[1] / "train.py"
_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "train_module_for_val_loss", _TRAIN_PATH
)
assert _TRAIN_SPEC is not None and _TRAIN_SPEC.loader is not None
_TRAIN_MOD = importlib.util.module_from_spec(_TRAIN_SPEC)
sys.modules["train_module_for_val_loss"] = _TRAIN_MOD
_TRAIN_SPEC.loader.exec_module(_TRAIN_MOD)


class _LossOut:
    def __init__(self, loss_f: float, logits: torch.Tensor | None = None):
        self.loss = torch.tensor(loss_f, dtype=torch.float32)
        self.logits = (
            logits
            if logits is not None
            else torch.zeros((1, 2, 5), dtype=torch.float32)
        )


class _FakeDDPModel:
    def __init__(
        self,
        outputs_L: list[tuple[float, torch.Tensor]],
        training: bool = True,
    ):
        self._out_it = iter(outputs_L)
        self.training = training
        self.eval_calls = 0
        self.train_calls = 0

    def eval(self):
        self.training = False
        self.eval_calls += 1
        return self

    def train(self, mode: bool = True):
        self.training = bool(mode)
        self.train_calls += 1
        return self

    def __call__(self, **_: object):
        loss_f, logits = next(self._out_it)
        return _LossOut(loss_f, logits=logits)


class _FakeTokenizer:
    _TOK_D = {
        1: "Frame 0: a",
        2: "Frame 0: b",
        3: "Frame 0: c",
        4: "Frame 0: z",
    }

    def batch_decode(self, ids_B, skip_special_tokens: bool = True):
        del skip_special_tokens
        return [
            "".join(self._TOK_D.get(int(tok_i), "") for tok_i in ids_L)
            for ids_L in ids_B
        ]


class _IdentityCollator:
    tokenizer = _FakeTokenizer()

    def __call__(self, batch_d):
        return batch_d


def _logits_for_pred(pred_tok_i: int, vocab_n: int = 8) -> torch.Tensor:
    logits = torch.zeros((1, 2, vocab_n), dtype=torch.float32)
    logits[0, 0, pred_tok_i] = 10.0
    return logits


def test_should_run_validation_uses_val_every_interval():
    assert not _TRAIN_MOD._should_run_validation(global_step=10, val_every=0)
    assert not _TRAIN_MOD._should_run_validation(global_step=10, val_every=-1)
    assert not _TRAIN_MOD._should_run_validation(global_step=9, val_every=5)
    assert _TRAIN_MOD._should_run_validation(global_step=10, val_every=5)


def test_run_validation_steps_averages_loss_and_rolls_iterator():
    batch0 = {
        "labels": torch.tensor([[-100, 1]], dtype=torch.long),
        "target_text": ["Frame 0: a"],
    }
    batch1 = {
        "labels": torch.tensor([[-100, 2]], dtype=torch.long),
        "target_text": ["Frame 0: b"],
    }
    batch2 = {
        "labels": torch.tensor([[-100, 3]], dtype=torch.long),
        "target_text": ["Frame 0: c"],
    }
    batches_by_epoch = [[batch0], [batch1, batch2]]
    make_calls = []

    def _make_val_iter(epoch_i: int):
        make_calls.append(epoch_i)
        return iter(batches_by_epoch[epoch_i])

    val_it = _make_val_iter(0)
    model = _FakeDDPModel(
        outputs_L=[
            (1.0, _logits_for_pred(1)),
            (2.0, _logits_for_pred(2)),
            (3.0, _logits_for_pred(4)),
        ],
        training=True,
    )

    (
        mean_loss_f,
        val_tok_n,
        val_action_correct_n,
        val_action_total_n,
        val_it,
        val_epoch_i,
    ) = _TRAIN_MOD._run_validation_steps(
        ddp_model=model,
        collator=_IdentityCollator(),
        val_it=val_it,
        make_val_iter=_make_val_iter,
        val_epoch_i=0,
        val_steps=3,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert mean_loss_f == pytest.approx(2.0)
    assert val_tok_n == 3
    assert val_action_correct_n == 2
    assert val_action_total_n == 3
    assert val_epoch_i == 1
    assert make_calls == [0, 1]
    assert model.eval_calls == 1
    assert model.train_calls == 1
    assert model.training is True
    with pytest.raises(StopIteration):
        next(val_it)


def test_run_validation_steps_preserves_eval_mode():
    batch0 = {
        "labels": torch.tensor([[-100, 1]], dtype=torch.long),
        "target_text": ["Frame 0: a"],
    }
    model = _FakeDDPModel(outputs_L=[(4.0, _logits_for_pred(1))], training=False)

    mean_loss_f, _, val_action_correct_n, val_action_total_n, _, _ = (
        _TRAIN_MOD._run_validation_steps(
            ddp_model=model,
            collator=_IdentityCollator(),
            val_it=iter([batch0]),
            make_val_iter=lambda _: iter([batch0]),
            val_epoch_i=0,
            val_steps=1,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
    )

    assert mean_loss_f == pytest.approx(4.0)
    assert val_action_correct_n == 1
    assert val_action_total_n == 1
    assert model.eval_calls == 1
    assert model.train_calls == 0
    assert model.training is False


def test_run_validation_steps_raises_when_val_loader_is_empty():
    with pytest.raises(ValueError, match="yielded no batches"):
        _TRAIN_MOD._run_validation_steps(
            ddp_model=_FakeDDPModel(outputs_L=[(1.0, _logits_for_pred(1))]),
            collator=_IdentityCollator(),
            val_it=iter([]),
            make_val_iter=lambda _: iter([]),
            val_epoch_i=0,
            val_steps=1,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )


def test_action_accuracy_counts_from_logits():
    logits = torch.zeros((2, 2, 8), dtype=torch.float32)
    logits[0, 0, 1] = 10.0
    logits[1, 0, 4] = 10.0
    labels = torch.tensor([[-100, 1], [-100, 2]], dtype=torch.long)
    target_text = ["Frame 0: a", "Frame 0: b"]
    correct_n, total_n = _TRAIN_MOD._action_accuracy_counts_from_logits(
        logits_BSV=logits,
        labels_BS=labels,
        target_text_B=target_text,
        tokenizer=_FakeTokenizer(),
    )
    assert correct_n == 1
    assert total_n == 2
