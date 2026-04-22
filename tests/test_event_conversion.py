"""Tests for raw keylog → sparse event conversion."""

from prepare_data import (
    parse_keylog_events,
    coalesce_scroll_events,
    filter_event_types,
    events_to_eval_format,
)


def _make_entry(ts_us: int, event_type: str, payload=None):
    """Helper to build a raw keylog entry."""
    ev = [event_type]
    if payload is not None:
        ev.append(payload)
    return [ts_us, ev]


class TestParseKeylogEvents:
    def test_simple_keypress(self):
        entries = [
            _make_entry(500_000, "KeyPress", [1, "KeyA"]),
            _make_entry(600_000, "KeyRelease", [1, "KeyA"]),
        ]
        events = parse_keylog_events(entries, fps=10, num_frames=10)
        assert len(events) == 1
        assert events[0]["type"] == "KeyPress"
        assert events[0]["details"] == "A"
        assert events[0]["frame_idx"] == 5  # 500_000 * 10 / 1_000_000

    def test_modifier_combo(self):
        entries = [
            _make_entry(1_000_000, "KeyPress", [1, "MetaLeft"]),
            _make_entry(1_100_000, "KeyPress", [1, "KeyC"]),
            _make_entry(1_200_000, "KeyRelease", [1, "KeyC"]),
            _make_entry(1_300_000, "KeyRelease", [1, "MetaLeft"]),
        ]
        events = parse_keylog_events(entries, fps=10, num_frames=20)
        # Only the C press should produce an event (Cmd is tracked as modifier)
        assert len(events) == 1
        assert events[0]["details"] == "Cmd+C"

    def test_modifier_alone_skipped(self):
        entries = [
            _make_entry(1_000_000, "KeyPress", [1, "ShiftLeft"]),
            _make_entry(1_200_000, "KeyRelease", [1, "ShiftLeft"]),
        ]
        events = parse_keylog_events(entries, fps=10, num_frames=20)
        # Modifier pressed and released alone — no event emitted
        assert len(events) == 0

    def test_mouse_click(self):
        entries = [
            _make_entry(2_000_000, "MousePress", ["Left", 0.0, 0.0]),
            _make_entry(2_100_000, "MouseRelease", ["Left", 0.0, 0.0]),
        ]
        events = parse_keylog_events(entries, fps=10, num_frames=30)
        assert len(events) == 1
        assert events[0]["type"] == "MouseClick"
        assert events[0]["details"] == "Left"
        assert events[0]["frame_idx"] == 20

    def test_right_click(self):
        # Press + release in same frame = single event
        entries = [
            _make_entry(500_000, "MousePress", ["Right", 0.0, 0.0]),
            _make_entry(550_000, "MouseRelease", ["Right", 0.0, 0.0]),
        ]
        events = parse_keylog_events(entries, fps=10, num_frames=10)
        assert len(events) == 1
        assert events[0]["details"] == "Right"

    def test_mouse_drag_held(self):
        # Press without release = held until end of clip
        entries = [_make_entry(500_000, "MousePress", ["Right", 0.0, 0.0])]
        events = parse_keylog_events(entries, fps=10, num_frames=10)
        assert len(events) == 4  # F05 through F08 (held to clip end - 1)
        assert all(e["details"] == "Right" for e in events)

    def test_mouse_scroll_horizontal(self):
        entries = [_make_entry(300_000, "MouseScroll", [-1, 0, 0, 0])]
        events = parse_keylog_events(entries, fps=10, num_frames=10)
        assert len(events) == 1
        assert events[0]["type"] == "MouseScroll"
        assert events[0]["details"] == "down"  # dx=-1 → down

    def test_mouse_scroll_vertical(self):
        entries = [_make_entry(300_000, "MouseScroll", [0, -1, 0, 0])]
        events = parse_keylog_events(entries, fps=10, num_frames=10)
        assert len(events) == 1
        assert events[0]["details"] == "down"  # dy=-1 → down

    def test_mouse_move_ignored(self):
        entries = [_make_entry(300_000, "MouseMove", [-5.0, 3.0])]
        events = parse_keylog_events(entries, fps=10, num_frames=10)
        assert len(events) == 0

    def test_out_of_range_frames_skipped(self):
        entries = [_make_entry(9_999_999, "KeyPress", [1, "KeyA"])]
        events = parse_keylog_events(entries, fps=10, num_frames=10)
        # frame_idx = 9_999_999 * 10 / 1_000_000 = 99, which is >= 10
        assert len(events) == 0

    def test_frame_index_calculation(self):
        # At 10fps, frame 0 covers [0, 100ms), frame 1 covers [100ms, 200ms), etc.
        entries = [_make_entry(0, "KeyPress", [1, "KeyA"])]
        events = parse_keylog_events(entries, fps=10, num_frames=10)
        assert events[0]["frame_idx"] == 0

        entries = [_make_entry(150_000, "KeyPress", [1, "KeyB"])]
        events = parse_keylog_events(entries, fps=10, num_frames=10)
        assert events[0]["frame_idx"] == 1

    def test_zero_scroll_ignored(self):
        entries = [_make_entry(300_000, "MouseScroll", [0, 0, 0, 0])]
        events = parse_keylog_events(entries, fps=10, num_frames=10)
        assert len(events) == 0


class TestCoalesceScrollEvents:
    def test_adjacent_same_direction_merged(self):
        events = [
            {"frame_idx": 0, "type": "MouseScroll", "details": "down"},
            {"frame_idx": 1, "type": "MouseScroll", "details": "down"},
            {"frame_idx": 2, "type": "MouseScroll", "details": "down"},
        ]
        result = coalesce_scroll_events(events)
        assert len(result) == 1
        assert result[0]["frame_idx"] == 0
        assert result[0]["details"] == "down"

    def test_direction_reversal_splits(self):
        events = [
            {"frame_idx": 0, "type": "MouseScroll", "details": "down"},
            {"frame_idx": 1, "type": "MouseScroll", "details": "down"},
            {"frame_idx": 2, "type": "MouseScroll", "details": "up"},
            {"frame_idx": 3, "type": "MouseScroll", "details": "up"},
        ]
        result = coalesce_scroll_events(events)
        assert len(result) == 2
        assert result[0]["details"] == "down"
        assert result[1]["details"] == "up"

    def test_gap_splits(self):
        events = [
            {"frame_idx": 0, "type": "MouseScroll", "details": "down"},
            {"frame_idx": 5, "type": "MouseScroll", "details": "down"},
        ]
        result = coalesce_scroll_events(events)
        assert len(result) == 2

    def test_non_scroll_events_preserved(self):
        events = [
            {"frame_idx": 0, "type": "KeyPress", "details": "A"},
            {"frame_idx": 1, "type": "MouseScroll", "details": "down"},
            {"frame_idx": 2, "type": "MouseScroll", "details": "down"},
            {"frame_idx": 5, "type": "MouseClick", "details": "Left"},
        ]
        result = coalesce_scroll_events(events)
        assert len(result) == 3
        assert result[0]["type"] == "KeyPress"
        assert result[1]["type"] == "MouseScroll"
        assert result[2]["type"] == "MouseClick"

    def test_empty(self):
        assert coalesce_scroll_events([]) == []


class TestFilterEventTypes:
    def test_keyboard_only(self):
        events = [
            {"frame_idx": 0, "type": "KeyPress", "details": "A"},
            {"frame_idx": 1, "type": "MouseClick", "details": "Left"},
            {"frame_idx": 2, "type": "MouseScroll", "details": "down"},
        ]
        result = filter_event_types(events, {"KeyPress"})
        assert len(result) == 1
        assert result[0]["type"] == "KeyPress"

    def test_keyboard_and_click(self):
        events = [
            {"frame_idx": 0, "type": "KeyPress", "details": "A"},
            {"frame_idx": 1, "type": "MouseClick", "details": "Left"},
            {"frame_idx": 2, "type": "MouseScroll", "details": "down"},
        ]
        result = filter_event_types(events, {"KeyPress", "MouseClick"})
        assert len(result) == 2

    def test_all(self):
        events = [
            {"frame_idx": 0, "type": "KeyPress", "details": "A"},
            {"frame_idx": 1, "type": "MouseClick", "details": "Left"},
        ]
        result = filter_event_types(events, {"KeyPress", "MouseClick", "MouseScroll"})
        assert len(result) == 2


class TestEventsToEvalFormat:
    def test_basic(self):
        events = [
            {"frame_idx": 3, "type": "KeyPress", "details": "A"},
            {"frame_idx": 12, "type": "MouseClick", "details": "Left"},
        ]
        result = events_to_eval_format(events)
        assert result == [
            {"frame": "F03", "type": "KeyPress", "details": "A"},
            {"frame": "F12", "type": "MouseClick", "details": "Left"},
        ]

    def test_empty(self):
        assert events_to_eval_format([]) == []
