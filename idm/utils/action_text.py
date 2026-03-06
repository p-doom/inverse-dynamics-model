from __future__ import annotations

from collections import Counter
import re
from typing import Callable

from idm.utils.actions import action_class_s


_FRAME_ACTION_RE = re.compile(r"^frame\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)
_FRAME_TIME_ACTION_RE = re.compile(
    r"^frame\s+\d+\s*:\s*\d+\s*-\s*\d+\s*:\s*\d+\s*:\s*(.*)$",
    re.IGNORECASE,
)
_FRAME_MINSEC_ACTION_RE = re.compile(
    r"^frame\s+\d+\s*:\s*\d+\s*:\s*(.*)$",
    re.IGNORECASE,
)


def _canonicalize_parsed_action(action_s: str) -> str:
    action_s = action_s.strip()
    while action_s.lower().startswith("action:"):
        action_s = action_s.split(":", 1)[1].strip()

    action_lower_s = action_s.lower()
    action_key_s = action_lower_s.rstrip(".")
    if action_key_s in {
        "no action",
        "no visible action",
        "no visible text",
        "none",
        "no-op",
        "noop",
    }:
        return "NO_OP"
    return action_s


def normalize_action(action_s: str) -> str:
    return action_s.strip().lower()


def parse_frame_actions(text_s: str, expected_n: int | None = None) -> list[str]:
    indexed_actions_d: dict[int, str] = {}
    sequential_actions_L: list[str] = []
    max_idx_i = -1
    candidate_lines_L: list[str] = []
    for raw_line_s in text_s.splitlines():
        raw_line_s = raw_line_s.strip()
        if not raw_line_s:
            continue
        inline_parts_L = re.split(
            r"(?=frame\s+\d+\s*:)",
            raw_line_s,
            flags=re.IGNORECASE,
        )
        for part_s in inline_parts_L:
            part_s = part_s.strip()
            if part_s:
                candidate_lines_L.append(part_s)

    for line_s in candidate_lines_L:
        if not line_s:
            continue

        match_time = _FRAME_TIME_ACTION_RE.match(line_s)
        if match_time is not None:
            sequential_actions_L.append(
                _canonicalize_parsed_action(match_time.group(1))
            )
            continue
        match_minsec = _FRAME_MINSEC_ACTION_RE.match(line_s)
        if match_minsec is not None:
            sequential_actions_L.append(
                _canonicalize_parsed_action(match_minsec.group(1))
            )
            continue

        match = _FRAME_ACTION_RE.match(line_s)
        if match is None:
            continue
        frame_idx_i = int(match.group(1))
        action_s = _canonicalize_parsed_action(match.group(2))
        prev_action_s = indexed_actions_d.get(frame_idx_i)
        if prev_action_s is None or (
            (not prev_action_s.strip()) and bool(action_s.strip())
        ):
            indexed_actions_d[frame_idx_i] = action_s
        max_idx_i = max(max_idx_i, frame_idx_i)

    if indexed_actions_d and sequential_actions_L:
        if len(sequential_actions_L) > len(indexed_actions_d):
            actions_L = sequential_actions_L
        else:
            actions_L = [""] * (max_idx_i + 1) if max_idx_i >= 0 else []
            for frame_idx_i, action_s in indexed_actions_d.items():
                if frame_idx_i >= 0:
                    actions_L[frame_idx_i] = action_s
    elif indexed_actions_d:
        actions_L = [""] * (max_idx_i + 1) if max_idx_i >= 0 else []
        for frame_idx_i, action_s in indexed_actions_d.items():
            if frame_idx_i >= 0:
                actions_L[frame_idx_i] = action_s
    else:
        actions_L = sequential_actions_L

    if expected_n is None:
        return actions_L
    if expected_n < 0:
        raise ValueError(f"expected_n must be >= 0, got {expected_n}.")
    pad_action_s = ""
    non_empty_norm_L = [normalize_action(x) for x in actions_L if x.strip()]
    if non_empty_norm_L and set(non_empty_norm_L) == {"no_op"}:
        pad_action_s = "NO_OP"
    return (actions_L + [pad_action_s] * expected_n)[:expected_n]


def action_class(action_s: str) -> str:
    return action_class_s(action_s)


def action_accuracy_counts_from_texts(
    pred_text_B: list[str],
    target_text_B: list[str],
    action_is_counted_fn: Callable[[str], bool] | None = None,
    class_counts_out_d: dict[str, int] | None = None,
) -> tuple[int, int]:
    correct_n = 0
    total_n = 0
    if action_is_counted_fn is None:
        action_is_counted_fn = lambda _: True

    if class_counts_out_d is not None:
        class_counts_out_d.setdefault("no_op_correct_n", 0)
        class_counts_out_d.setdefault("no_op_total_n", 0)
        class_counts_out_d.setdefault("mouse_correct_n", 0)
        class_counts_out_d.setdefault("mouse_total_n", 0)
        class_counts_out_d.setdefault("keyboard_correct_n", 0)
        class_counts_out_d.setdefault("keyboard_total_n", 0)

    for pred_s, target_s in zip(pred_text_B, target_text_B):
        target_actions_L = parse_frame_actions(str(target_s))
        pred_actions_L = parse_frame_actions(
            str(pred_s), expected_n=len(target_actions_L)
        )
        for idx_i, target_action_s in enumerate(target_actions_L):
            if not action_is_counted_fn(target_action_s):
                continue
            total_n += 1
            action_class_s = action_class(target_action_s)
            if class_counts_out_d is not None:
                class_counts_out_d[f"{action_class_s}_total_n"] += 1
            if pred_actions_L[idx_i] == target_action_s:
                correct_n += 1
                if class_counts_out_d is not None:
                    class_counts_out_d[f"{action_class_s}_correct_n"] += 1
    return correct_n, total_n


def action_type_counts_from_texts(action_text_B: list[str]) -> tuple[int, int, int]:
    no_op_n = 0
    mouse_n = 0
    total_n = 0
    for action_text_s in action_text_B:
        for action_s in parse_frame_actions(action_text_s):
            action_s = action_s.strip()
            if not action_s:
                continue
            total_n += 1
            action_class_s = action_class(action_s)
            if action_class_s == "no_op":
                no_op_n += 1
            if action_class_s == "mouse":
                mouse_n += 1
    return no_op_n, mouse_n, total_n


def per_action_stats_from_actions(
    pred_actions_L: list[str],
    gt_actions_L: list[str],
) -> dict[str, dict[str, float]]:
    gt_norm_L = [normalize_action(x) for x in gt_actions_L]
    pred_norm_L = [normalize_action(x) for x in pred_actions_L]

    gt_counts = Counter(gt_norm_L)
    pred_counts = Counter(pred_norm_L)
    correct_counts = Counter(
        gt for gt, pred in zip(gt_norm_L, pred_norm_L) if gt == pred
    )

    out_d: dict[str, dict[str, float]] = {}
    for action_s in sorted(set(gt_norm_L) | set(pred_norm_L)):
        tp_i = correct_counts[action_s]
        gt_total_i = gt_counts[action_s]
        pred_total_i = pred_counts[action_s]
        recall_f = tp_i / gt_total_i if gt_total_i else 0.0
        precision_f = tp_i / pred_total_i if pred_total_i else 0.0
        f1_f = (
            2.0 * precision_f * recall_f / (precision_f + recall_f)
            if (precision_f + recall_f) > 0.0
            else 0.0
        )
        out_d[action_s] = {
            "recall": recall_f,
            "precision": precision_f,
            "f1": f1_f,
            "support": gt_total_i,
            "predicted_count": pred_total_i,
        }
    return out_d
