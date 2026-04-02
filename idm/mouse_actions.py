from __future__ import annotations

import re
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Constants from the data-gen pipeline (must match generate_simulated_dataset)
# ---------------------------------------------------------------------------

MOUSE_X_QUANT_UNIT_F = 5.0
MOUSE_Y_QUANT_UNIT_F = 4.0
MOUSE_DELTA_CLIP_I = 64
MOUSE_DELTA_EXP_CURVATURE_F = 1.0

# ---------------------------------------------------------------------------
# Action helpers -- mouse-only
# ---------------------------------------------------------------------------

_ACTION_CLASS_NAMES = ("no_op", "mouse")
_ACTION_CONFUSION_PRED_CLASS_NAMES = ("no_op", "mouse", "missing")

_MOUSE_RE = re.compile(r"^MOUSE:\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*$")
_BARE_TRIPLET_RE = re.compile(r"^(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*$")


def _action_class_s(action_s: str) -> str:
    action_s = action_s.strip()
    if action_s in ("NO_OP", ""):
        return "no_op"
    if action_s.startswith("MOUSE:"):
        return "mouse"
    m = _BARE_TRIPLET_RE.match(action_s)
    if m is not None:
        return "no_op" if all(int(m.group(i)) == 0 for i in (1, 2, 3)) else "mouse"
    return "no_op"


def _action_is_no_op(action_s: str) -> bool:
    return _action_class_s(action_s) == "no_op"


def _parse_mouse_delta(action_s: str) -> tuple[int, int, int] | None:
    """Return (dx_q, dy_q, scroll_q) from a MOUSE:dx,dy,scroll string, or None."""
    m = _MOUSE_RE.match(action_s.strip())
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# ---------------------------------------------------------------------------
# Inverse-quantize: map quantized delta back to pixel delta
# ---------------------------------------------------------------------------

def _inv_quantize_exp(
    q_i: int,
    quant_unit_f: float,
    clip_abs_i: int,
    curvature_f: float = MOUSE_DELTA_EXP_CURVATURE_F,
) -> float:
    """Approximate inverse of _quantize_mouse_delta_exponential."""
    if clip_abs_i <= 0 or q_i == 0:
        return 0.0
    sign = -1.0 if q_i < 0 else 1.0
    abs_q = min(abs(q_i), clip_abs_i)
    curved = abs_q / float(clip_abs_i)  # in [0, 1]
    # curved = log1p(c*n) / log1p(c)  ->  n = (exp(curved*log1p(c)) - 1) / c
    log1p_c = float(np.log1p(curvature_f))
    normalized = (np.expm1(curved * log1p_c)) / curvature_f
    max_val = quant_unit_f * float(clip_abs_i)
    return sign * normalized * max_val


def _deltas_to_pixel_offsets(
    dx_q: int, dy_q: int
) -> tuple[float, float]:
    """Convert quantized deltas to pixel-space deltas."""
    dx_px = _inv_quantize_exp(dx_q, MOUSE_X_QUANT_UNIT_F, MOUSE_DELTA_CLIP_I)
    dy_px = _inv_quantize_exp(dy_q, MOUSE_Y_QUANT_UNIT_F, MOUSE_DELTA_CLIP_I)
    return dx_px, dy_px


def _accumulate_positions(
    actions: list[str],
    img_w: int,
    img_h: int,
    start_x: float | None = None,
    start_y: float | None = None,
) -> list[tuple[int, int]]:
    """Walk through a list of action strings and accumulate absolute cursor
    positions (clamped to image bounds)."""
    cx = img_w / 2.0 if start_x is None else start_x
    cy = img_h / 2.0 if start_y is None else start_y
    positions: list[tuple[int, int]] = []
    for a in actions:
        parsed = _parse_mouse_delta(a)
        if parsed is not None:
            dx_q, dy_q, _ = parsed
            dx_px, dy_px = _deltas_to_pixel_offsets(dx_q, dy_q)
            cx = float(np.clip(cx + dx_px, 0, img_w - 1))
            cy = float(np.clip(cy + dy_px, 0, img_h - 1))
        positions.append((int(round(cx)), int(round(cy))))
    return positions


# ---------------------------------------------------------------------------
# Action parsing from target text
# ---------------------------------------------------------------------------

def _actions_from_target_text(target_s: str) -> list[str]:
    actions_L = []
    for line_s in target_s.splitlines():
        line_s = line_s.strip()
        if not line_s.startswith("Frame "):
            continue
        parts_L = line_s.split(":", 1)
        if len(parts_L) != 2:
            continue
        actions_L.append(parts_L[1].strip())
    return actions_L


def _decode_pred_text_B_from_generated_ids(
    generated_ids_BS,
    prompt_lens_B: list[int],
    tokenizer: Any,
) -> list[str]:
    pred_ids_B: list[list[int]] = []
    for b_i, prompt_len_i in enumerate(prompt_lens_B):
        row_S = generated_ids_BS[b_i]
        start_i = max(int(prompt_len_i), 0)
        if start_i >= int(row_S.shape[0]):
            pred_ids_B.append([])
            continue
        pred_ids_B.append([int(x) for x in row_S[start_i:].detach().cpu().tolist()])
    return [
        str(x) for x in tokenizer.batch_decode(pred_ids_B, skip_special_tokens=True)
    ]


def _actions_from_pred_text(pred_s: str) -> list[str]:
    """Parse actions from prediction text.

    Handles two formats:
    1. Structured: "Frame N: ACTION\\n..." (same as target format)
    2. Unstructured: space/newline-separated action tokens (e.g. "NO_OP MOUSE:1,-1,0 NO_OP ...")

    Falls back to unstructured parsing when no "Frame N:" lines are found.
    """
    structured = _actions_from_target_text(pred_s)
    if structured:
        return structured
    # Unstructured fallback: split by whitespace and identify valid action tokens
    actions: list[str] = []
    for token in pred_s.split():
        token = token.strip()
        if token == "NO_OP":
            actions.append("NO_OP")
        elif _MOUSE_RE.match(token):
            actions.append(token)
        elif _BARE_TRIPLET_RE.match(token):
            actions.append(token)
    return actions


def _truncate_for_log(text_s: str, max_chars: int = 1200) -> str:
    text_s = str(text_s).strip()
    if len(text_s) <= max_chars:
        return text_s
    return f"{text_s[:max_chars]} ...[truncated]"
