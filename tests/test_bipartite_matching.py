"""Tests for bipartite matching F1 computation."""

from eval import compute_f1


class TestComputeF1:
    def test_perfect_match(self):
        gt = [{"frame": "F03", "type": "KeyPress", "details": "A"}]
        pred = [{"frame": "F03", "type": "KeyPress", "details": "A"}]
        p, r, f1 = compute_f1(pred, gt)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_within_tolerance(self):
        gt = [{"frame": "F03", "type": "KeyPress", "details": "A"}]
        pred = [{"frame": "F05", "type": "KeyPress", "details": "A"}]
        p, r, f1 = compute_f1(pred, gt, tolerance=2)
        assert f1 == 1.0

    def test_beyond_tolerance(self):
        gt = [{"frame": "F03", "type": "KeyPress", "details": "A"}]
        pred = [{"frame": "F06", "type": "KeyPress", "details": "A"}]
        p, r, f1 = compute_f1(pred, gt, tolerance=2)
        assert f1 == 0.0

    def test_type_mismatch(self):
        gt = [{"frame": "F03", "type": "KeyPress", "details": "A"}]
        pred = [{"frame": "F03", "type": "MouseClick", "details": "Left"}]
        p, r, f1 = compute_f1(pred, gt)
        assert f1 == 0.0

    def test_empty_both(self):
        p, r, f1 = compute_f1([], [])
        assert f1 == 1.0

    def test_empty_pred(self):
        gt = [{"frame": "F03", "type": "KeyPress", "details": "A"}]
        p, r, f1 = compute_f1([], gt)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_empty_gt(self):
        pred = [{"frame": "F03", "type": "KeyPress", "details": "A"}]
        p, r, f1 = compute_f1(pred, [])
        assert f1 == 0.0

    def test_partial_match(self):
        gt = [
            {"frame": "F03", "type": "KeyPress", "details": "A"},
            {"frame": "F10", "type": "MouseClick", "details": "Left"},
        ]
        pred = [
            {"frame": "F03", "type": "KeyPress", "details": "A"},
            {"frame": "F20", "type": "KeyPress", "details": "B"},
        ]
        p, r, f1 = compute_f1(pred, gt, tolerance=2)
        assert p == 0.5  # 1/2 predictions correct
        assert r == 0.5  # 1/2 GT found
        assert abs(f1 - 0.5) < 1e-6

    def test_multiple_matches(self):
        gt = [
            {"frame": "F03", "type": "KeyPress", "details": "A"},
            {"frame": "F05", "type": "KeyPress", "details": "B"},
            {"frame": "F10", "type": "MouseClick", "details": "Left"},
        ]
        pred = [
            {"frame": "F03", "type": "KeyPress", "details": "A"},
            {"frame": "F05", "type": "KeyPress", "details": "B"},
            {"frame": "F10", "type": "MouseClick", "details": "Left"},
        ]
        p, r, f1 = compute_f1(pred, gt)
        assert f1 == 1.0

    def test_greedy_no_double_match(self):
        """Each GT can only be matched once."""
        gt = [{"frame": "F05", "type": "KeyPress", "details": "A"}]
        pred = [
            {"frame": "F05", "type": "KeyPress", "details": "A"},
            {"frame": "F06", "type": "KeyPress", "details": "B"},
        ]
        p, r, f1 = compute_f1(pred, gt, tolerance=2)
        assert p == 0.5  # 1/2 predictions matched
        assert r == 1.0  # 1/1 GT matched
