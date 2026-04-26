"""Tests for black frame detection and filtering."""

import numpy as np
from PIL import Image

from prepare_data import is_black_frame, filter_black_frames


def _make_image(rgb=(0, 0, 0), size=(64, 64)):
    """Create a solid-color image."""
    arr = np.full((*size, 3), rgb, dtype=np.uint8)
    return Image.fromarray(arr)


class TestIsBlackFrame:
    def test_all_black(self):
        assert is_black_frame(_make_image((0, 0, 0))) is True

    def test_near_black(self):
        assert is_black_frame(_make_image((10, 10, 10))) is True

    def test_not_black(self):
        assert is_black_frame(_make_image((128, 128, 128))) is False

    def test_mostly_black(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:4, :, :] = 200  # 4% non-black
        img = Image.fromarray(arr)
        assert is_black_frame(img, threshold=0.95) is True

    def test_barely_not_black(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:10, :, :] = 200  # 10% non-black
        img = Image.fromarray(arr)
        assert is_black_frame(img, threshold=0.95) is False


class TestFilterBlackFrames:
    def test_removes_black_and_reindexes(self, tmp_path):
        # Create 5 frames: frames 1 and 3 are black
        paths = []
        for i in range(5):
            rgb = (0, 0, 0) if i in (1, 3) else (128, 128, 128)
            img = _make_image(rgb)
            p = tmp_path / f"frame_{i:03d}.jpg"
            img.save(p)
            paths.append(str(p))

        events = [
            {"frame_idx": 0, "type": "KeyPress", "details": "A"},
            {"frame_idx": 1, "type": "KeyPress", "details": "B"},  # on black frame
            {"frame_idx": 2, "type": "KeyPress", "details": "C"},
            {"frame_idx": 4, "type": "KeyPress", "details": "E"},
        ]

        filtered_paths, filtered_events = filter_black_frames(paths, events, threshold=0.95)

        # Frames 0, 2, 4 survive → new indices 0, 1, 2
        assert len(filtered_paths) == 3
        assert len(filtered_events) == 3  # B was on black frame, dropped
        assert filtered_events[0] == {"frame_idx": 0, "type": "KeyPress", "details": "A"}
        assert filtered_events[1] == {"frame_idx": 1, "type": "KeyPress", "details": "C"}
        assert filtered_events[2] == {"frame_idx": 2, "type": "KeyPress", "details": "E"}

    def test_threshold_1_disables(self, tmp_path):
        img = _make_image((0, 0, 0))
        p = tmp_path / "frame.jpg"
        img.save(p)
        paths, events = filter_black_frames([str(p)], [], threshold=1.0)
        assert len(paths) == 1  # nothing removed
