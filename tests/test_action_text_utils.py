from __future__ import annotations

import pytest

from idm.utils.action_text import (
    action_accuracy_counts_from_texts,
    action_type_counts_from_texts,
    parse_frame_actions,
    per_action_stats_from_actions,
)


def test_parse_frame_actions_pads_to_expected_length():
    parsed = parse_frame_actions(
        "Frame 0: KEY_DOWN:W\nFrame 2: KEY_UP:W",
        expected_n=4,
    )
    assert parsed == ["KEY_DOWN:W", "", "KEY_UP:W", ""]


def test_parse_frame_actions_strips_action_prefix_and_maps_no_action():
    parsed = parse_frame_actions("Frame 0: Action: NO_OP\nFrame 1: Action: no action")
    assert parsed == ["NO_OP", "NO_OP"]


def test_parse_frame_actions_keeps_first_non_empty_duplicate_frame():
    parsed = parse_frame_actions("Frame 0: KEY_DOWN:W\nFrame 0: Action:")
    assert parsed == ["KEY_DOWN:W"]


def test_parse_frame_actions_supports_timestamp_style_frame_lines():
    parsed = parse_frame_actions(
        "Frame 0:00 - 0:01: No visible action.\nFrame 0:01 - 0:02: KEY_DOWN:W",
        expected_n=2,
    )
    assert parsed == ["NO_OP", "KEY_DOWN:W"]


def test_parse_frame_actions_supports_minsec_style_frame_lines():
    parsed = parse_frame_actions(
        "Frame 0:0: No visible action.\nFrame 0:1: KEY_DOWN:W",
        expected_n=2,
    )
    assert parsed == ["NO_OP", "KEY_DOWN:W"]


def test_parse_frame_actions_supports_inline_repeated_frame_chunks():
    parsed = parse_frame_actions(
        "Frame 0:00 - 0:01: No visible action.  Frame 0:01 - 0:02: KEY_DOWN:W",
        expected_n=2,
    )
    assert parsed == ["NO_OP", "KEY_DOWN:W"]


def test_parse_frame_actions_prefers_sequential_when_indexed_is_sparse():
    parsed = parse_frame_actions(
        "Frame 0:00 - 0:01: No visible action.\n"
        "Frame 0:01 - 0:02: No visible action.\n"
        "Frame 1:46 - 1:47",
        expected_n=2,
    )
    assert parsed == ["NO_OP", "NO_OP"]


def test_parse_frame_actions_pads_no_op_when_only_no_op_aliases_are_present():
    parsed = parse_frame_actions(
        "Frame 0:00 - 0:01: No visible action.",
        expected_n=3,
    )
    assert parsed == ["NO_OP", "NO_OP", "NO_OP"]


def test_action_accuracy_counts_use_frame_indices_not_only_line_order():
    pred_text = ["Frame 1: a\nFrame 2: b"]
    target_text = ["Frame 0: a\nFrame 1: b"]
    correct_n, total_n = action_accuracy_counts_from_texts(
        pred_text_B=pred_text,
        target_text_B=target_text,
    )
    assert correct_n == 0
    assert total_n == 2


def test_action_accuracy_counts_support_filters_and_class_counts():
    pred_text = ["Frame 0: NO_OP\nFrame 1: MOUSE_MOVE\nFrame 2: KEY_DOWN:W"]
    target_text = ["Frame 0: NO_OP\nFrame 1: MOUSE_MOVE\nFrame 2: KEY_DOWN:W"]
    class_counts: dict[str, int] = {}
    correct_n, total_n = action_accuracy_counts_from_texts(
        pred_text_B=pred_text,
        target_text_B=target_text,
        action_is_counted_fn=lambda action_s: "MOUSE_" not in action_s,
        class_counts_out_d=class_counts,
    )
    assert correct_n == 2
    assert total_n == 2
    assert class_counts["no_op_correct_n"] == 1
    assert class_counts["keyboard_correct_n"] == 1
    assert class_counts["mouse_total_n"] == 0


def test_action_type_counts_ignore_unparsed_or_empty_actions():
    no_op_n, mouse_n, total_n = action_type_counts_from_texts(
        ["Frame 1: NO_OP\nFrame 3: MOUSE_DOWN:Left"]
    )
    assert no_op_n == 1
    assert mouse_n == 1
    assert total_n == 2


def test_per_action_stats_from_actions_normalizes_case():
    stats = per_action_stats_from_actions(
        pred_actions_L=["NO_OP", "key_down:w", "x"],
        gt_actions_L=["no_op", "KEY_DOWN:W", "y"],
    )
    assert stats["no_op"]["f1"] == pytest.approx(1.0)
    assert stats["key_down:w"]["f1"] == pytest.approx(1.0)
    assert stats["y"]["recall"] == pytest.approx(0.0)
