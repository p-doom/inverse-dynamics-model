from __future__ import annotations

from collections import defaultdict

from idm.utils.actions import action_is_no_op_b


def actions_from_target_text(target_s: str) -> list[str]:
    actions_L: list[str] = []
    for line_s in str(target_s).splitlines():
        line_s = line_s.strip()
        if not line_s.startswith("Frame "):
            continue
        parts_L = line_s.split(":", 1)
        if len(parts_L) != 2:
            continue
        actions_L.append(parts_L[1].strip())
    return actions_L


def action_event_match_counts(
    pred_text_B: list[str],
    target_text_B: list[str],
    tolerance_frames: int = 0,
    ignore_no_op: bool = True,
) -> tuple[int, int, int]:
    if tolerance_frames < 0:
        raise ValueError("tolerance_frames must be >= 0.")

    tp_n = 0
    fp_n = 0
    fn_n = 0
    for pred_s, target_s in zip(pred_text_B, target_text_B):
        pred_actions_L = actions_from_target_text(pred_s)
        target_actions_L = actions_from_target_text(target_s)
        pred_events_by_action_d: dict[str, list[tuple[int, bool]]] = defaultdict(list)
        for frame_i, action_s in enumerate(pred_actions_L):
            if ignore_no_op and action_is_no_op_b(action_s):
                continue
            pred_events_by_action_d[action_s].append((frame_i, False))

        for target_frame_i, target_action_s in enumerate(target_actions_L):
            if ignore_no_op and action_is_no_op_b(target_action_s):
                continue
            pred_events_L = pred_events_by_action_d.get(target_action_s)
            if not pred_events_L:
                fn_n += 1
                continue
            lo_i = target_frame_i
            hi_i = target_frame_i + tolerance_frames
            match_idx_i: int | None = None
            for idx_i, (pred_frame_i, used_b) in enumerate(pred_events_L):
                if used_b:
                    continue
                if pred_frame_i < lo_i:
                    continue
                if pred_frame_i > hi_i:
                    break
                match_idx_i = idx_i
                break
            if match_idx_i is None:
                fn_n += 1
                continue
            pred_frame_i, _ = pred_events_L[match_idx_i]
            pred_events_L[match_idx_i] = (pred_frame_i, True)
            tp_n += 1

        for pred_events_L in pred_events_by_action_d.values():
            for _, used_b in pred_events_L:
                if not used_b:
                    fp_n += 1

    return tp_n, fp_n, fn_n


def precision_recall_f1_from_counts(
    tp_n: float,
    fp_n: float,
    fn_n: float,
) -> tuple[float, float, float]:
    precision_f = float(tp_n) / max(float(tp_n) + float(fp_n), 1.0)
    recall_f = float(tp_n) / max(float(tp_n) + float(fn_n), 1.0)
    denom_f = precision_f + recall_f
    if denom_f <= 0.0:
        return precision_f, recall_f, 0.0
    return precision_f, recall_f, (2.0 * precision_f * recall_f / denom_f)
