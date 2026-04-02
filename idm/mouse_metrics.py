from __future__ import annotations

from typing import Callable

import numpy as np
import wandb

from mouse_actions import (
    _ACTION_CLASS_NAMES,
    _ACTION_CONFUSION_PRED_CLASS_NAMES,
    _action_class_s,
    _actions_from_pred_text,
    _actions_from_target_text,
    _deltas_to_pixel_offsets,
    _parse_mouse_delta,
)


# ---------------------------------------------------------------------------
# Action accuracy / confusion -- mouse-only (no keyboard)
# ---------------------------------------------------------------------------

def _action_confusion_count_key(target_class_s: str, pred_class_s: str) -> str:
    return f"confusion_{target_class_s}_as_{pred_class_s}_n"


def _action_accuracy_counts_from_texts(
    pred_text_B: list[str],
    target_text_B: list[str],
    action_is_counted_fn: Callable[[str], bool] | None = None,
    class_counts_out_d: dict[str, int] | None = None,
    confusion_counts_out_d: dict[str, int] | None = None,
) -> tuple[int, int]:
    correct_n = 0
    total_n = 0
    if action_is_counted_fn is None:
        action_is_counted_fn = lambda _: True
    if class_counts_out_d is not None:
        for cls_s in _ACTION_CLASS_NAMES:
            class_counts_out_d.setdefault(f"{cls_s}_correct_n", 0)
            class_counts_out_d.setdefault(f"{cls_s}_total_n", 0)
    if confusion_counts_out_d is not None:
        for t_s in _ACTION_CLASS_NAMES:
            for p_s in _ACTION_CONFUSION_PRED_CLASS_NAMES:
                confusion_counts_out_d.setdefault(
                    _action_confusion_count_key(t_s, p_s), 0
                )
    for pred_s, target_s in zip(pred_text_B, target_text_B):
        pred_actions_L = _actions_from_pred_text(pred_s)
        target_actions_L = _actions_from_target_text(target_s)
        for idx_i, target_action_s in enumerate(target_actions_L):
            if not action_is_counted_fn(target_action_s):
                continue
            target_cls = _action_class_s(target_action_s)
            has_pred = idx_i < len(pred_actions_L)
            pred_action_s = pred_actions_L[idx_i] if has_pred else ""
            pred_cls = _action_class_s(pred_action_s) if has_pred else "missing"
            total_n += 1
            if class_counts_out_d is not None:
                class_counts_out_d[f"{target_cls}_total_n"] += 1
            if confusion_counts_out_d is not None:
                confusion_counts_out_d[
                    _action_confusion_count_key(target_cls, pred_cls)
                ] += 1
            if has_pred and pred_action_s == target_action_s:
                correct_n += 1
                if class_counts_out_d is not None:
                    class_counts_out_d[f"{target_cls}_correct_n"] += 1
    return correct_n, total_n


def _action_type_counts_from_texts(
    action_text_B: list[str],
) -> tuple[int, int, int]:
    no_op_n = 0
    mouse_n = 0
    total_n = 0
    for text_s in action_text_B:
        for action_s in _actions_from_pred_text(text_s):
            cls = _action_class_s(action_s)
            total_n += 1
            if cls == "no_op":
                no_op_n += 1
            elif cls == "mouse":
                mouse_n += 1
    return no_op_n, mouse_n, total_n


def _action_confusion_matrix_counts(
    stats_d: dict[str, int],
) -> list[list[int]]:
    counts: list[list[int]] = []
    for t_s in _ACTION_CLASS_NAMES:
        row: list[int] = []
        for p_s in _ACTION_CONFUSION_PRED_CLASS_NAMES:
            row.append(int(stats_d.get(_action_confusion_count_key(t_s, p_s), 0)))
        counts.append(row)
    return counts


def _action_f1_from_counts(
    correct_n_f: float,
    pred_total_n_f: float,
    target_total_n_f: float,
) -> float:
    precision_f = float(correct_n_f) / max(float(pred_total_n_f), 1.0)
    recall_f = float(correct_n_f) / max(float(target_total_n_f), 1.0)
    denom_f = precision_f + recall_f
    if denom_f <= 0.0:
        return 0.0
    return 2.0 * precision_f * recall_f / denom_f


def _wandb_confusion_chart(counts_NM: list[list[int]]) -> wandb.viz.CustomChart | None:
    y_true: list[int] = []
    preds: list[int] = []
    for t_i in range(len(_ACTION_CLASS_NAMES)):
        for p_i in range(len(_ACTION_CONFUSION_PRED_CLASS_NAMES)):
            c = int(counts_NM[t_i][p_i])
            if c > 0:
                y_true.extend([t_i] * c)
                preds.extend([p_i] * c)
    if not y_true:
        return None
    return wandb.plot.confusion_matrix(
        y_true=y_true,
        preds=preds,
        class_names=list(_ACTION_CONFUSION_PRED_CLASS_NAMES),
        title="Val Mouse Action Confusion Matrix",
    )


def _mouse_proximity_counts_from_texts(
    pred_text_B: list[str],
    target_text_B: list[str],
    threshold_px: float = 50.0,
) -> tuple[int, int]:
    """Count MOUSE predictions whose (dx, dy) euclidean distance to the
    ground-truth is within *threshold_px* pixels.

    Returns (correct_n, total_n) where total_n is the number of ground-truth
    MOUSE actions and correct_n counts those where a matching-class prediction
    was found within the pixel threshold.  A prediction of the wrong action
    class (e.g. NO_OP) counts as incorrect regardless of distance.
    """
    correct_n = 0
    total_n = 0
    for pred_s, target_s in zip(pred_text_B, target_text_B):
        pred_actions = _actions_from_pred_text(pred_s)
        target_actions = _actions_from_target_text(target_s)
        for idx, target_a in enumerate(target_actions):
            if _action_class_s(target_a) != "mouse":
                continue
            total_n += 1
            if idx >= len(pred_actions):
                continue
            pred_a = pred_actions[idx]
            if _action_class_s(pred_a) != "mouse":
                continue
            t_parsed = _parse_mouse_delta(target_a)
            p_parsed = _parse_mouse_delta(pred_a)
            if t_parsed is None or p_parsed is None:
                continue
            t_dx, t_dy = _deltas_to_pixel_offsets(t_parsed[0], t_parsed[1])
            p_dx, p_dy = _deltas_to_pixel_offsets(p_parsed[0], p_parsed[1])
            dist_px = float(np.hypot(p_dx - t_dx, p_dy - t_dy))
            if dist_px <= threshold_px:
                correct_n += 1
    return correct_n, total_n


def _mouse_vector_metrics_from_texts(
    pred_text_B: list[str],
    target_text_B: list[str],
) -> tuple[float, float, int]:
    """Return (cosine_sim_sum, euclidean_dist_sum, count) over matched mouse actions."""
    cos_sum = 0.0
    euc_sum = 0.0
    count = 0
    for pred_s, target_s in zip(pred_text_B, target_text_B):
        pred_actions = _actions_from_pred_text(pred_s)
        target_actions = _actions_from_target_text(target_s)
        for idx, target_a in enumerate(target_actions):
            if _action_class_s(target_a) != "mouse":
                continue
            if idx >= len(pred_actions):
                continue
            pred_a = pred_actions[idx]
            if _action_class_s(pred_a) != "mouse":
                continue
            t_parsed = _parse_mouse_delta(target_a)
            p_parsed = _parse_mouse_delta(pred_a)
            if t_parsed is None or p_parsed is None:
                continue
            t_dx, t_dy = _deltas_to_pixel_offsets(t_parsed[0], t_parsed[1])
            p_dx, p_dy = _deltas_to_pixel_offsets(p_parsed[0], p_parsed[1])
            t_vec = np.array([t_dx, t_dy, float(t_parsed[2])], dtype=np.float64)
            p_vec = np.array([p_dx, p_dy, float(p_parsed[2])], dtype=np.float64)
            # Euclidean distance
            euc_sum += float(np.linalg.norm(p_vec - t_vec))
            # Cosine similarity
            t_norm = np.linalg.norm(t_vec)
            p_norm = np.linalg.norm(p_vec)
            if t_norm > 1e-9 and p_norm > 1e-9:
                cos_sum += float(np.dot(t_vec, p_vec) / (t_norm * p_norm))
            else:
                cos_sum += 1.0 if (t_norm <= 1e-9 and p_norm <= 1e-9) else 0.0
            count += 1
    return cos_sum, euc_sum, count
