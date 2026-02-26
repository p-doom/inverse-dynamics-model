from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionInfo:
    is_no_op_b: bool
    has_mouse_b: bool
    has_nonzero_mouse_b: bool
    has_keyboard_b: bool


def _parse_mouse_triplet(component_s: str) -> tuple[int, int, int] | None:
    component_s = component_s.strip()
    if not component_s.startswith("MOUSE:"):
        return None
    payload_s = component_s.split(":", 1)[1].strip()
    parts_L = [part_s.strip() for part_s in payload_s.split(",")]
    if len(parts_L) != 3:
        return None
    try:
        return tuple(int(float(part_s)) for part_s in parts_L)
    except (TypeError, ValueError):
        return None


def action_info(action_s: str) -> ActionInfo:
    action_s = str(action_s).strip()
    if not action_s or action_s == "NO_OP":
        return ActionInfo(
            is_no_op_b=True,
            has_mouse_b=False,
            has_nonzero_mouse_b=False,
            has_keyboard_b=False,
        )

    components_L = [component_s.strip() for component_s in action_s.split(";")]
    mouse_triplet_t = _parse_mouse_triplet(components_L[0]) if components_L else None
    if mouse_triplet_t is not None:
        has_keyboard_b = any(
            bool(component_s.strip()) for component_s in components_L[1:]
        )
        has_nonzero_mouse_b = any(v_i != 0 for v_i in mouse_triplet_t)
        is_no_op_b = (not has_nonzero_mouse_b) and (not has_keyboard_b)
        return ActionInfo(
            is_no_op_b=is_no_op_b,
            has_mouse_b=True,
            has_nonzero_mouse_b=has_nonzero_mouse_b,
            has_keyboard_b=has_keyboard_b,
        )

    tokens_L = [token_s.strip() for token_s in action_s.split("+")]
    has_mouse_b = False
    has_nonzero_mouse_b = False
    has_keyboard_b = False
    for token_s in tokens_L:
        if not token_s or token_s == "NO_OP":
            continue
        mouse_triplet_t = _parse_mouse_triplet(token_s)
        if mouse_triplet_t is not None:
            has_mouse_b = True
            has_nonzero_mouse_b = has_nonzero_mouse_b or any(
                v_i != 0 for v_i in mouse_triplet_t
            )
            continue
        if token_s.startswith("MOUSE_"):
            has_mouse_b = True
            has_nonzero_mouse_b = True
            continue
        has_keyboard_b = True

    is_no_op_b = (not has_nonzero_mouse_b) and (not has_keyboard_b)
    return ActionInfo(
        is_no_op_b=is_no_op_b,
        has_mouse_b=has_mouse_b,
        has_nonzero_mouse_b=has_nonzero_mouse_b,
        has_keyboard_b=has_keyboard_b,
    )


def action_is_no_op_b(action_s: str) -> bool:
    return action_info(action_s).is_no_op_b


def action_has_nonzero_mouse_b(action_s: str) -> bool:
    return action_info(action_s).has_nonzero_mouse_b


def action_class_s(action_s: str) -> str:
    info = action_info(action_s)
    if info.is_no_op_b:
        return "no_op"
    if info.has_nonzero_mouse_b:
        return "mouse"
    return "keyboard"
