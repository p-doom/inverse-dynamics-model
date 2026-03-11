from __future__ import annotations

import pytest

from idm.utils.action_metrics import (
    action_event_match_counts,
    precision_recall_f1_from_counts,
)


def _target_text(actions_L: list[str]) -> str:
    return "\n".join(f"Frame {i}: {action_s}" for i, action_s in enumerate(actions_L))


def test_action_event_match_counts_perfect_alignment():
    target_s = _target_text(["A", "NO_OP", "B"])
    pred_s = _target_text(["A", "NO_OP", "B"])

    strict_tp_i, strict_fp_i, strict_fn_i = action_event_match_counts(
        pred_text_B=[pred_s],
        target_text_B=[target_s],
        tolerance_frames=0,
        ignore_no_op=True,
    )
    tolerant_tp_i, tolerant_fp_i, tolerant_fn_i = action_event_match_counts(
        pred_text_B=[pred_s],
        target_text_B=[target_s],
        tolerance_frames=5,
        ignore_no_op=True,
    )

    assert (strict_tp_i, strict_fp_i, strict_fn_i) == (2, 0, 0)
    assert (tolerant_tp_i, tolerant_fp_i, tolerant_fn_i) == (2, 0, 0)


def test_action_event_match_counts_lagged_actions_improve_with_tolerance():
    target_s = _target_text(["A", "B", "C", "NO_OP", "NO_OP"])
    pred_s = _target_text(["NO_OP", "NO_OP", "A", "B", "C"])

    strict_tp_i, strict_fp_i, strict_fn_i = action_event_match_counts(
        pred_text_B=[pred_s],
        target_text_B=[target_s],
        tolerance_frames=0,
        ignore_no_op=True,
    )
    tolerant_tp_i, tolerant_fp_i, tolerant_fn_i = action_event_match_counts(
        pred_text_B=[pred_s],
        target_text_B=[target_s],
        tolerance_frames=2,
        ignore_no_op=True,
    )

    assert (strict_tp_i, strict_fp_i, strict_fn_i) == (0, 3, 3)
    assert (tolerant_tp_i, tolerant_fp_i, tolerant_fn_i) == (3, 0, 0)


def test_action_event_match_counts_prevents_duplicate_credit():
    target_s = _target_text(["L", "NO_OP", "NO_OP"])
    pred_s = _target_text(["L", "L", "L"])

    tp_i, fp_i, fn_i = action_event_match_counts(
        pred_text_B=[pred_s],
        target_text_B=[target_s],
        tolerance_frames=2,
        ignore_no_op=True,
    )

    assert (tp_i, fp_i, fn_i) == (1, 2, 0)


def test_action_event_match_counts_handles_repeated_same_action_instances():
    target_s = _target_text(["NO_OP", "L", "NO_OP", "L", "NO_OP"])
    pred_s = _target_text(["NO_OP", "NO_OP", "L", "NO_OP", "L"])

    tp_i, fp_i, fn_i = action_event_match_counts(
        pred_text_B=[pred_s],
        target_text_B=[target_s],
        tolerance_frames=1,
        ignore_no_op=True,
    )

    assert (tp_i, fp_i, fn_i) == (2, 0, 0)


def test_action_event_match_counts_tolerance_must_be_non_negative():
    with pytest.raises(ValueError):
        action_event_match_counts(
            pred_text_B=["Frame 0: A"],
            target_text_B=["Frame 0: A"],
            tolerance_frames=-1,
        )


def test_precision_recall_f1_from_counts():
    precision_f, recall_f, f1_f = precision_recall_f1_from_counts(
        tp_n=3,
        fp_n=1,
        fn_n=2,
    )

    assert precision_f == pytest.approx(0.75)
    assert recall_f == pytest.approx(0.6)
    assert f1_f == pytest.approx(2.0 * 0.75 * 0.6 / (0.75 + 0.6))
