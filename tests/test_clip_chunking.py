"""Tests for sliding-window clip chunking."""

from prepare_data import chunk_into_clips


def _paths(n):
    return [f"/tmp/frame_{i:03d}.jpg" for i in range(n)]


class TestChunkIntoClips:
    def test_basic_chunking(self):
        frames = _paths(60)
        events = [
            {"frame_idx": 5, "type": "KeyPress", "details": "A"},
            {"frame_idx": 35, "type": "KeyPress", "details": "B"},
        ]
        clips = chunk_into_clips(frames, events, clip_length=30, clip_stride=15)
        # Starts: 0, 15, 30
        assert len(clips) == 3

        # First clip [0..29]: event at frame 5
        clip_frames, clip_events = clips[0]
        assert len(clip_frames) == 30
        assert len(clip_events) == 1
        assert clip_events[0]["frame_idx"] == 5

        # Second clip [15..44]: events at local 5-15=-10 (out), 35-15=20 (in)
        clip_frames, clip_events = clips[1]
        assert len(clip_events) == 1
        assert clip_events[0]["frame_idx"] == 20  # 35 - 15

        # Third clip [30..59]: event at local 35-30=5
        clip_frames, clip_events = clips[2]
        assert len(clip_events) == 1
        assert clip_events[0]["frame_idx"] == 5

    def test_short_sequence(self):
        frames = _paths(10)
        events = [{"frame_idx": 3, "type": "KeyPress", "details": "X"}]
        clips = chunk_into_clips(frames, events, clip_length=30, clip_stride=15)
        # Only one clip since sequence < clip_length
        assert len(clips) == 1
        assert len(clips[0][0]) == 10
        assert clips[0][1][0]["frame_idx"] == 3

    def test_no_events_clip_still_created(self):
        """Chunking creates clips even with no events (filtering is separate)."""
        frames = _paths(30)
        clips = chunk_into_clips(frames, [], clip_length=30, clip_stride=15)
        assert len(clips) == 1
        assert len(clips[0][1]) == 0

    def test_stride_equals_length(self):
        frames = _paths(90)
        events = [{"frame_idx": 45, "type": "KeyPress", "details": "A"}]
        clips = chunk_into_clips(frames, events, clip_length=30, clip_stride=30)
        # Starts: 0, 30, 60
        assert len(clips) == 3
        # Event at 45 falls in clip [30..59], local idx 15
        assert len(clips[0][1]) == 0
        assert len(clips[1][1]) == 1
        assert clips[1][1][0]["frame_idx"] == 15
        assert len(clips[2][1]) == 0

    def test_events_at_boundaries(self):
        frames = _paths(30)
        events = [
            {"frame_idx": 0, "type": "KeyPress", "details": "A"},
            {"frame_idx": 29, "type": "KeyPress", "details": "Z"},
        ]
        clips = chunk_into_clips(frames, events, clip_length=30, clip_stride=30)
        assert len(clips) == 1
        assert len(clips[0][1]) == 2
        assert clips[0][1][0]["frame_idx"] == 0
        assert clips[0][1][1]["frame_idx"] == 29
