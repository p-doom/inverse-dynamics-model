from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


_TRAIN_PATH = Path(__file__).resolve().parents[1] / "idm" / "train.py"
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
        7: "Frame 0: MOUSE_MOVE\nFrame 1: b",
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


def _logits_for_pred(pred_tok_i: int, vocab_n: int = 8) -> torch.Tensor:
    logits = torch.zeros((1, 2, vocab_n), dtype=torch.float32)
    logits[0, 0, pred_tok_i] = 10.0
    return logits


def _logits_for_pred_with_seq(
    pred_tok_i: int, seq_len: int, vocab_n: int = 8
) -> torch.Tensor:
    logits = torch.zeros((1, seq_len, vocab_n), dtype=torch.float32)
    if seq_len > 1:
        logits[0, : seq_len - 1, pred_tok_i] = 10.0
    return logits


def test_weighted_causal_lm_loss_matches_manual_weighted_average():
    logits = torch.tensor(
        [
            [
                [1.0, 2.0, 0.5],
                [0.3, 0.7, 1.1],
                [2.0, 0.1, 0.2],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[-100, 1, 2, 0]], dtype=torch.long)
    label_weights = torch.tensor([[0.0, 0.25, 1.0, 2.0]], dtype=torch.float32)

    loss_t = _TRAIN_MOD._weighted_causal_lm_loss(
        logits_BSV=logits,
        labels_BS=labels,
        label_weights_BS=label_weights,
    )

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(shift_labels)
    valid = shift_labels != -100
    shift_weights = label_weights[:, 1:]
    expected_loss = (token_loss * shift_weights * valid.to(torch.float32)).sum() / (
        shift_weights * valid.to(torch.float32)
    ).sum()
    assert loss_t.item() == pytest.approx(expected_loss.item())


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

    expected_losses = [
        _TRAIN_MOD._weighted_causal_lm_loss(
            logits_BSV=_logits_for_pred(1),
            labels_BS=batch0["labels"],
        ).item(),
        _TRAIN_MOD._weighted_causal_lm_loss(
            logits_BSV=_logits_for_pred(2),
            labels_BS=batch1["labels"],
        ).item(),
        _TRAIN_MOD._weighted_causal_lm_loss(
            logits_BSV=_logits_for_pred(4),
            labels_BS=batch2["labels"],
        ).item(),
    ]
    assert mean_loss_f == pytest.approx(sum(expected_losses) / len(expected_losses))
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
            (1.0, _logits_for_pred_with_seq(1, seq_len=4)),
            (3.0, _logits_for_pred_with_seq(2, seq_len=4)),
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
    loss0 = _TRAIN_MOD._weighted_causal_lm_loss(
        logits_BSV=_logits_for_pred_with_seq(1, seq_len=4),
        labels_BS=batch0["labels"],
    ).item()
    loss1 = _TRAIN_MOD._weighted_causal_lm_loss(
        logits_BSV=_logits_for_pred_with_seq(2, seq_len=4),
        labels_BS=batch1["labels"],
    ).item()
    assert mean_loss_f == pytest.approx((loss0 * 1 + loss1 * 3) / 4.0)


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


def test_action_accuracy_counts_from_texts_populates_per_class_counts():
    pred_text = [
        "Frame 0: NO_OP\nFrame 1: MOUSE_MOVE\nFrame 2: KEY_DOWN:W",
    ]
    target_text = [
        "Frame 0: NO_OP\nFrame 1: MOUSE_MOVE\nFrame 2: KEY_UP:W",
    ]
    class_counts = {}

    correct_n, total_n = _TRAIN_MOD._action_accuracy_counts_from_texts(
        pred_text_B=pred_text,
        target_text_B=target_text,
        class_counts_out_d=class_counts,
    )

    assert correct_n == 2
    assert total_n == 3
    assert class_counts["no_op_correct_n"] == 1
    assert class_counts["no_op_total_n"] == 1
    assert class_counts["mouse_correct_n"] == 1
    assert class_counts["mouse_total_n"] == 1
    assert class_counts["keyboard_correct_n"] == 0
    assert class_counts["keyboard_total_n"] == 1


def test_run_validation_steps_populates_action_type_stats():
    batch0 = {
        "labels": torch.tensor([[-100, 7]], dtype=torch.long),
        "target_text": ["Frame 0: NO_OP\nFrame 1: MOUSE_MOVE"],
    }
    model = _FakeDDPModel(
        outputs_L=[(1.0, _logits_for_pred(7))],
        generated_ids_L=[torch.tensor([[99, 7]], dtype=torch.long)],
        training=True,
    )
    action_stats = {}

    _TRAIN_MOD._run_validation_steps(
        ddp_model=model,
        collator=_IdentityCollator(),
        val_it=iter([batch0]),
        val_steps=1,
        val_generate_max_new_tokens=4,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        action_stats_out_d=action_stats,
    )

    assert action_stats["pred_no_op_n"] == 0
    assert action_stats["pred_mouse_n"] == 1
    assert action_stats["pred_action_total_n"] == 2
    assert action_stats["target_no_op_n"] == 1
    assert action_stats["target_mouse_n"] == 1
    assert action_stats["target_action_total_n"] == 2
    assert action_stats["class_no_op_correct_n"] == 0
    assert action_stats["class_no_op_total_n"] == 1
    assert action_stats["class_mouse_correct_n"] == 0
    assert action_stats["class_mouse_total_n"] == 1
    assert action_stats["class_keyboard_correct_n"] == 0
    assert action_stats["class_keyboard_total_n"] == 0
