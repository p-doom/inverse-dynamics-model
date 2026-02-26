from idm.utils.actions import (
    action_class_s,
    action_has_nonzero_mouse_b,
    action_is_no_op_b,
)


def test_action_parser_handles_lumine_mouse_and_keyboard_format() -> None:
    action_s = "MOUSE:12,-3,0 ; Shift W"
    assert not action_is_no_op_b(action_s)
    assert action_has_nonzero_mouse_b(action_s)
    assert action_class_s(action_s) == "mouse"


def test_action_parser_treats_zero_mouse_with_keys_as_keyboard_only() -> None:
    action_s = "MOUSE:0,0,0 ; W"
    assert not action_is_no_op_b(action_s)
    assert not action_has_nonzero_mouse_b(action_s)
    assert action_class_s(action_s) == "keyboard"


def test_action_parser_treats_zero_mouse_without_keys_as_no_op() -> None:
    action_s = "MOUSE:0,0,0"
    assert action_is_no_op_b(action_s)
    assert not action_has_nonzero_mouse_b(action_s)
    assert action_class_s(action_s) == "no_op"


def test_action_parser_is_backward_compatible_with_legacy_mouse_tokens() -> None:
    action_s = "KEY_DOWN:W + MOUSE_MOVE"
    assert not action_is_no_op_b(action_s)
    assert action_has_nonzero_mouse_b(action_s)
    assert action_class_s(action_s) == "mouse"
