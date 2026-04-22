"""Tests for prompt construction and collation logic."""

import json
import re

from data import build_prompt, build_sft_messages


class TestBuildPrompt:
    def test_contains_fps_info(self):
        prompt = build_prompt(fps=10, num_frames=30)
        assert "10fps" in prompt
        assert "100ms" in prompt

    def test_contains_frame_range(self):
        prompt = build_prompt(fps=5, num_frames=25)
        assert "F00 to F24" in prompt
        assert "25 frames total" in prompt

    def test_contains_action_types(self):
        prompt = build_prompt(fps=10, num_frames=10)
        assert "KeyPress" in prompt
        assert "MouseClick" in prompt
        assert "MouseScroll" in prompt

    def test_json_example(self):
        prompt = build_prompt(fps=10, num_frames=10)
        # Should contain a valid JSON example
        assert '"frame": "F03"' in prompt
        assert '"type": "KeyPress"' in prompt

    def test_matches_eval_format(self):
        """Verify our prompt is semantically identical to eval pipeline's."""
        prompt = build_prompt(fps=5, num_frames=30)
        # Key phrases from eval's build_prompt
        assert "SINGLE KEY" in prompt
        assert "Do NOT list mouse movements" in prompt
        assert "Output ONLY a valid JSON array" in prompt
        assert "no markdown fences" in prompt


class TestBuildSftMessages:
    def test_structure(self):
        from PIL import Image

        frames = [Image.new("RGB", (64, 64), color="red")]
        msgs = build_sft_messages(frames, "test prompt", "test target", fps=10.0)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_user_message_has_video_and_text(self):
        from PIL import Image

        frames = [Image.new("RGB", (64, 64)) for _ in range(3)]
        msgs = build_sft_messages(frames, "hello", "world", fps=10.0)
        user_content = msgs[0]["content"]
        # 1 video + 1 text
        assert len(user_content) == 2
        assert user_content[0]["type"] == "video"
        assert user_content[1]["type"] == "text"
        assert user_content[1]["text"] == "hello"

    def test_assistant_message_has_target(self):
        from PIL import Image

        frames = [Image.new("RGB", (64, 64))]
        target = json.dumps([{"frame": "F03", "type": "KeyPress", "details": "A"}])
        msgs = build_sft_messages(frames, "prompt", target, fps=10.0)
        assert msgs[1]["content"][0]["text"] == target
