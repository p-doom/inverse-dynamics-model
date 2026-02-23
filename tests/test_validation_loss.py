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
        generated_ids_L: list[torch.Tensor] | None = None,
        training: bool = True,
    ):
        self._out_it = iter(outputs_L)
        self._gen_it = iter(generated_ids_L or [])
        self.training = training
        self.eval_calls = 0
        self.train_calls = 0
        self.generate_calls = 0

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

    def generate(self, **_: object):
        self.generate_calls += 1
        return next(self._gen_it)


class _FakeTokenizer:
    _TOK_D = {
        1: "Frame 0: a",
        2: "Frame 0: b",
        3: "Frame 0: c",
        4: "Frame 0: z",
        5: "Frame 0: NO_OP\nFrame 1: b",
        6: "Frame 0: x\nFrame 1: b",
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
        return {"labels": batch_d["labels"]}

    def prompt_model_inputs(self, batch_d):
        batch_n = int(batch_d["labels"].shape[0])
        return {
            "input_ids": torch.full((batch_n, 1), 99, dtype=torch.long),
            "attention_mask": torch.ones((batch_n, 1), dtype=torch.long),
            "prompt_lens": [1 for _ in range(batch_n)],
        }


class _MaskingIdentityCollator(_IdentityCollator):
    def __init__(
        self,
        mask_no_op_actions: bool = False,
        mask_mouse_actions: bool = False,
    ):
        self.mask_no_op_actions = mask_no_op_actions
        self.mask_mouse_actions = mask_mouse_actions

    def _should_mask_action(self, action_s: str) -> bool:
        action_s = action_s.strip()
        if self.mask_no_op_actions and action_s == "NO_OP":
            return True
        if self.mask_mouse_actions and "MOUSE_" in action_s:
            return True
        return False


def _logits_for_pred(pred_tok_i: int, vocab_n: int = 8) -> torch.Tensor:
    logits = torch.zeros((1, 2, vocab_n), dtype=torch.float32)
    logits[0, 0, pred_tok_i] = 10.0
    return logits


def test_run_validation_steps_averages_loss_and_counts_action_accuracy():
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
    val_it = iter([batch0, batch1, batch2])
    model = _FakeDDPModel(
        outputs_L=[
            (1.0, _logits_for_pred(1)),
            (2.0, _logits_for_pred(2)),
            (3.0, _logits_for_pred(4)),
        ],
        generated_ids_L=[
            torch.tensor([[99, 1]], dtype=torch.long),
            torch.tensor([[99, 2]], dtype=torch.long),
            torch.tensor([[99, 4]], dtype=torch.long),
        ],
        training=True,
    )

    mean_loss_f, val_tok_n, val_action_correct_n, val_action_total_n = (
        _TRAIN_MOD._run_validation_steps(
            ddp_model=model,
            collator=_IdentityCollator(),
            val_it=val_it,
            val_steps=3,
            val_generate_max_new_tokens=4,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
    )

    assert mean_loss_f == pytest.approx(2.0)
    assert val_tok_n == 3
    assert val_action_correct_n == 2
    assert val_action_total_n == 3
    assert model.eval_calls == 1
    assert model.train_calls == 1
    assert model.generate_calls == 3
    assert model.training is True


def test_run_validation_steps_weights_loss_by_supervised_tokens():
    batch0 = {
        "labels": torch.tensor([[-100, 1, -100, -100]], dtype=torch.long),
        "target_text": ["Frame 0: a"],
    }
    batch1 = {
        "labels": torch.tensor([[-100, 2, 3, 4]], dtype=torch.long),
        "target_text": ["Frame 0: b"],
    }
    model = _FakeDDPModel(
        outputs_L=[
            (1.0, _logits_for_pred(1)),
            (3.0, _logits_for_pred(2)),
        ],
        generated_ids_L=[
            torch.tensor([[99, 1]], dtype=torch.long),
            torch.tensor([[99, 2]], dtype=torch.long),
        ],
        training=True,
    )

    mean_loss_f, val_tok_n, _, _ = _TRAIN_MOD._run_validation_steps(
        ddp_model=model,
        collator=_IdentityCollator(),
        val_it=iter([batch0, batch1]),
        val_steps=2,
        val_generate_max_new_tokens=4,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert val_tok_n == 4
    assert mean_loss_f == pytest.approx((1.0 * 1 + 3.0 * 3) / 4.0)


def test_run_validation_steps_collects_debug_examples_when_requested():
    batch0 = {
        "labels": torch.tensor([[-100, 1], [-100, 2]], dtype=torch.long),
        "target_text": ["Frame 0: a", "Frame 0: b"],
    }
    model = _FakeDDPModel(
        outputs_L=[(1.0, torch.zeros((2, 2, 8), dtype=torch.float32))],
        generated_ids_L=[torch.tensor([[99, 1], [99, 2]], dtype=torch.long)],
        training=True,
    )
    debug_examples: list[tuple[str, str]] = []

    _TRAIN_MOD._run_validation_steps(
        ddp_model=model,
        collator=_IdentityCollator(),
        val_it=iter([batch0]),
        val_steps=1,
        val_generate_max_new_tokens=4,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        debug_examples_n=1,
        debug_examples_out_L=debug_examples,
    )

    assert debug_examples == [("Frame 0: a", "Frame 0: a")]


def test_run_validation_steps_raises_when_val_loader_is_empty():
    with pytest.raises(StopIteration):
        _TRAIN_MOD._run_validation_steps(
            ddp_model=_FakeDDPModel(outputs_L=[(1.0, _logits_for_pred(1))]),
            collator=_IdentityCollator(),
            val_it=iter([]),
            val_steps=1,
            val_generate_max_new_tokens=4,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )


def test_action_accuracy_counts_from_texts():
    pred_text = ["Frame 0: a", "Frame 0: z"]
    target_text = ["Frame 0: a", "Frame 0: b"]
    correct_n, total_n = _TRAIN_MOD._action_accuracy_counts_from_texts(
        pred_text_B=pred_text,
        target_text_B=target_text,
    )
    assert correct_n == 1
    assert total_n == 2


def test_action_accuracy_counts_from_texts_filters_no_op_and_mouse_actions():
    pred_text = [
        "Frame 0: NO_OP\nFrame 1: b\nFrame 2: MOUSE_MOVE\nFrame 3: z",
    ]
    target_text = [
        "Frame 0: NO_OP\nFrame 1: b\nFrame 2: MOUSE_MOVE\nFrame 3: a",
    ]

    def action_is_counted(action_s: str) -> bool:
        action_s = action_s.strip()
        return action_s != "NO_OP" and "MOUSE_" not in action_s

    correct_n, total_n = _TRAIN_MOD._action_accuracy_counts_from_texts(
        pred_text_B=pred_text,
        target_text_B=target_text,
        action_is_counted_fn=action_is_counted,
    )
    assert correct_n == 1
    assert total_n == 2


def test_action_accuracy_filter_keeps_original_frame_alignment():
    pred_text = ["Frame 0: x\nFrame 1: b"]
    target_text = ["Frame 0: NO_OP\nFrame 1: b"]

    correct_n, total_n = _TRAIN_MOD._action_accuracy_counts_from_texts(
        pred_text_B=pred_text,
        target_text_B=target_text,
        action_is_counted_fn=lambda action_s: action_s.strip() != "NO_OP",
    )
    assert correct_n == 1
    assert total_n == 1


def test_run_validation_steps_filters_masked_actions_from_action_accuracy():
    batch0 = {
        "labels": torch.tensor([[-100, 2]], dtype=torch.long),
        "target_text": ["Frame 0: NO_OP\nFrame 1: b"],
    }
    model = _FakeDDPModel(
        outputs_L=[(1.0, _logits_for_pred(2))],
        generated_ids_L=[torch.tensor([[99, 6]], dtype=torch.long)],
        training=True,
    )
    _, _, val_action_correct_n, val_action_total_n = _TRAIN_MOD._run_validation_steps(
        ddp_model=model,
        collator=_MaskingIdentityCollator(mask_no_op_actions=True),
        val_it=iter([batch0]),
        val_steps=1,
        val_generate_max_new_tokens=4,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert val_action_correct_n == 1
    assert val_action_total_n == 1


def test_run_validation_steps_keeps_action_accuracy_unfiltered_when_masks_disabled():
    batch0 = {
        "labels": torch.tensor([[-100, 2]], dtype=torch.long),
        "target_text": ["Frame 0: NO_OP\nFrame 1: b"],
    }
    model = _FakeDDPModel(
        outputs_L=[(1.0, _logits_for_pred(2))],
        generated_ids_L=[torch.tensor([[99, 6]], dtype=torch.long)],
        training=True,
    )
    _, _, val_action_correct_n, val_action_total_n = _TRAIN_MOD._run_validation_steps(
        ddp_model=model,
        collator=_MaskingIdentityCollator(mask_no_op_actions=False),
        val_it=iter([batch0]),
        val_steps=1,
        val_generate_max_new_tokens=4,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert val_action_correct_n == 1
    assert val_action_total_n == 2
