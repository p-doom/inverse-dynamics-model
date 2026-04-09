"""Tests for key name normalization and modifier combo formatting."""

from prepare_data import normalize_key_name, format_key_with_modifiers


class TestNormalizeKeyName:
    def test_letter_keys(self):
        assert normalize_key_name("KeyA") == "A"
        assert normalize_key_name("KeyZ") == "Z"

    def test_digit_keys(self):
        assert normalize_key_name("Digit0") == "0"
        assert normalize_key_name("Digit9") == "9"

    def test_modifiers(self):
        assert normalize_key_name("MetaLeft") == "Cmd"
        assert normalize_key_name("MetaRight") == "Cmd"
        assert normalize_key_name("ShiftLeft") == "Shift"
        assert normalize_key_name("ShiftRight") == "Shift"
        assert normalize_key_name("ControlLeft") == "Ctrl"
        assert normalize_key_name("ControlRight") == "Ctrl"
        assert normalize_key_name("AltGr") == "AltGr"
        assert normalize_key_name("Alt") == "Alt"

    def test_special_keys(self):
        assert normalize_key_name("Return") == "Return"
        assert normalize_key_name("Space") == "Space"
        assert normalize_key_name("Backspace") == "Backspace"
        assert normalize_key_name("Tab") == "Tab"
        assert normalize_key_name("Escape") == "Escape"

    def test_arrow_keys(self):
        assert normalize_key_name("UpArrow") == "UpArrow"
        assert normalize_key_name("DownArrow") == "DownArrow"
        assert normalize_key_name("LeftArrow") == "LeftArrow"
        assert normalize_key_name("RightArrow") == "RightArrow"

    def test_function_keys(self):
        assert normalize_key_name("F1") == "F1"
        assert normalize_key_name("F12") == "F12"

    def test_unknown_passthrough(self):
        assert normalize_key_name("SomeWeirdKey") == "SomeWeirdKey"


class TestFormatKeyWithModifiers:
    def test_no_modifiers(self):
        assert format_key_with_modifiers("A", set()) == "A"

    def test_single_modifier(self):
        assert format_key_with_modifiers("C", {"Cmd"}) == "Cmd+C"
        assert format_key_with_modifiers("V", {"Ctrl"}) == "Ctrl+V"

    def test_multiple_modifiers(self):
        result = format_key_with_modifiers("Z", {"Cmd", "Shift"})
        assert result == "Cmd+Shift+Z"

    def test_modifier_key_alone(self):
        # Pressing Shift alone should just return "Shift"
        assert format_key_with_modifiers("Shift", {"Shift"}) == "Shift"
        assert format_key_with_modifiers("Cmd", set()) == "Cmd"

    def test_modifier_order(self):
        # Order should always be Cmd, Ctrl, Alt, Shift
        result = format_key_with_modifiers("A", {"Shift", "Cmd", "Ctrl", "Alt"})
        assert result == "Cmd+Ctrl+Alt+Shift+A"
