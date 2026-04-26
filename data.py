#!/usr/bin/env python3
"""Dataset classes, collation, and prompt/message building for IDM training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from transformers import AutoProcessor


# ---------------------------------------------------------------------------
# Prompt (shared by train and eval — must stay identical)
# ---------------------------------------------------------------------------


def build_prompt(fps: int, num_frames: int) -> str:
    interval_ms = 1000 / fps
    return (
        f"Analyze these consecutive screenshots from a macOS screen recording "
        f"(browser and IDE workflows).\n"
        f"The frames are labeled F00, F01, F02, etc. in the top-left corner, "
        f"sampled at {fps}fps (each frame is {interval_ms:.0f}ms apart). "
        f"There are {num_frames} frames total (F00 to F{num_frames-1:02d}).\n\n"
        f"Detect individual user INPUT ACTIONS by comparing consecutive frames. "
        f"For each action provide:\n"
        f'- frame: the frame label where the effect first becomes visible (e.g. "F07")\n'
        f"- type: one of KeyPress, MouseClick, MouseScroll\n"
        f'- details: for KeyPress give the SINGLE KEY name (e.g. "A", "Return", '
        f'"Space", "Backspace", "Cmd+C", "LeftArrow"). For MouseClick give the '
        f'button ("Left" or "Right"). For MouseScroll give the direction ("up" or "down").\n\n'
        f"IMPORTANT rules:\n"
        f'- Each KeyPress is ONE key. If "hello" is typed, that is 5 separate '
        f"KeyPress events: H, E, L, L, O.\n"
        f"- Do NOT list mouse movements or key releases.\n"
        f"- Ignore screen changes not caused by user input (loading content, "
        f"streaming LLM output, animations, cursor blinks).\n"
        f"- If you see a character appear in a text field, that is a KeyPress of that character.\n"
        f"- If you see a UI element change state (button highlight, tab switch, "
        f"menu open), that is likely a MouseClick.\n"
        f"- If you see content scroll, that is a MouseScroll.\n\n"
        f"Output ONLY a valid JSON array, no markdown fences, no explanation. Example:\n"
        f'[{{"frame": "F03", "type": "KeyPress", "details": "A"}}, '
        f'{{"frame": "F12", "type": "MouseClick", "details": "Left"}}]\n'
        f"If no user actions occurred, output: []"
    )


# Normalize key names from prepare_data.py to eval-format conventions.
_KEY_NORMALIZE = {
    "SemiColon": "Semicolon",
    "BackSlash": "Backslash",
    "BackQuote": "Backtick",
}


def normalize_actions(actions: list[dict]) -> list[dict]:
    """Normalize action key names and strip unpredictable keys.

    - SemiColon -> Semicolon, BackSlash -> Backslash, BackQuote -> Backtick
    - Strips Unknown(...) keys since they have no visual cue.
    """
    result = []
    for a in actions:
        a = dict(a)
        detail = a.get("details", "")
        if "Unknown(" in detail:
            continue
        parts = detail.split("+")
        a["details"] = "+".join(_KEY_NORMALIZE.get(p, p) for p in parts)
        result.append(a)
    return result


def build_sft_messages(
    frames: list[Image.Image],
    prompt: str,
    target: str,
    fps: float,
    video_mode: str = "video",
    interleave_labels: bool = False,
) -> list[dict]:
    """Build Qwen3-VL chat messages with video/images + prompt -> target."""
    if video_mode == "image":
        if interleave_labels:
            content = []
            for i, f in enumerate(frames):
                content.append({"type": "text", "text": f"Frame F{i:02d}:"})
                content.append({"type": "image", "image": f})
        else:
            content = [{"type": "image", "image": f} for f in frames]
    else:
        content = [{"type": "video", "video": frames, "fps": fps}]
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class ClipDataset(Dataset):
    """Validation dataset: does all heavy processing (JPEG decode + processor) in workers."""

    def __init__(
        self, jsonl_path: str, data_root: str, processor: Any, max_length: int = 8192
    ):
        self.data_root = Path(data_root)
        self.processor = processor
        self.max_length = max_length
        self._assistant_marker = processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self.clips = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.clips.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.clips)

    def _find_prompt_len(self, input_ids: list[int]) -> int:
        """Find where the assistant response starts by searching for the marker."""
        marker = self._assistant_marker
        for i in range(len(input_ids) - len(marker), -1, -1):
            if input_ids[i : i + len(marker)] == marker:
                return i + len(marker)
        return 0

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        clip_dir = self.data_root / clip["clip_dir"]
        num_frames = clip["num_frames"]
        fps = clip["fps"]
        actions = normalize_actions(clip["actions"])

        frames = []
        for i in range(num_frames):
            img_path = clip_dir / f"F{i:02d}.jpg"
            frames.append(Image.open(img_path).convert("RGB"))

        prompt = build_prompt(fps, num_frames)
        target = json.dumps(actions, separators=(",", ":"))
        video_meta = [{"fps": float(fps), "total_num_frames": len(frames)}]

        full_msgs = build_sft_messages(frames, prompt, target, fps)
        full_text = self.processor.apply_chat_template(full_msgs, tokenize=False)
        full_inputs = self.processor(
            text=[full_text],
            videos=[frames],
            video_metadata=video_meta,
            return_tensors="pt",
            padding=False,
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]

        prompt_len = self._find_prompt_len(input_ids.tolist())

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        mm_token_type_ids = full_inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.squeeze(0)
            if mm_token_type_ids.shape[0] > self.max_length:
                mm_token_type_ids = mm_token_type_ids[: self.max_length]

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "actions": actions,
            "fps": fps,
            "num_frames": num_frames,
        }
        if mm_token_type_ids is not None:
            result["mm_token_type_ids"] = mm_token_type_ids
        if "pixel_values_videos" in full_inputs:
            result["pixel_values_videos"] = full_inputs["pixel_values_videos"].squeeze(
                0
            )
        if "video_grid_thw" in full_inputs:
            result["video_grid_thw"] = full_inputs["video_grid_thw"]
        return result


class ProcessedClipDataset(Dataset):
    """Training dataset: does FULL processing in DataLoader workers (parallel processes).

    Each worker holds its own processor instance. Returns model-ready tensors.
    This avoids the single-threaded collator bottleneck.
    """

    def __init__(
        self,
        jsonl_path: str,
        data_root: str,
        model_id: str,
        max_pixels: int,
        max_length: int,
        video_mode: str,
        interleave_labels: bool,
    ):
        self.data_root = Path(data_root)
        self.model_id = model_id
        self.max_pixels = max_pixels
        self.max_length = max_length
        self.video_mode = video_mode
        self.interleave_labels = interleave_labels
        self._processor = None  # lazy init per worker
        self.clips = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.clips.append(json.loads(line))

    def _get_processor(self):
        """Lazy-init processor in each DataLoader worker process."""
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                min_pixels=3136,
                max_pixels=self.max_pixels,
            )
        return self._processor

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        for _attempt in range(5):
            try:
                return self._process_clip(idx)
            except (OSError, SyntaxError, ValueError) as e:
                print(f"[processed-dataset] skipping corrupt clip idx={idx}: {e}")
                idx = random.randint(0, len(self.clips) - 1)
        return self._process_clip(idx)

    def _process_clip(self, idx: int) -> dict:
        clip = self.clips[idx]
        clip_dir = self.data_root / clip["clip_dir"]
        num_frames = clip["num_frames"]
        fps = clip["fps"]
        actions = normalize_actions(clip["actions"])

        processor = self._get_processor()

        frame_list = [
            Image.open(clip_dir / f"F{i:02d}.jpg").convert("RGB")
            for i in range(num_frames)
        ]

        target = json.dumps(actions, separators=(",", ":"))
        prompt = build_prompt(fps, num_frames)
        msgs = build_sft_messages(
            frame_list,
            prompt,
            target,
            fps,
            video_mode=self.video_mode,
            interleave_labels=self.interleave_labels,
        )
        text = processor.apply_chat_template(msgs, tokenize=False)

        enc = processor(
            text=[text], images=frame_list, return_tensors="pt", padding=False
        )
        input_ids = enc["input_ids"].squeeze(0)
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]

        # Build labels: mask prompt tokens
        assistant_marker = processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        prompt_len = 0
        ids_list = input_ids.tolist()
        for i in range(len(ids_list) - len(assistant_marker), -1, -1):
            if ids_list[i : i + len(assistant_marker)] == assistant_marker:
                prompt_len = i + len(assistant_marker)
                break
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        result = {"input_ids": input_ids, "labels": labels}

        for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
        ):
            if key in enc:
                val = enc[key]
                if val.dim() > 0:
                    result[key] = val.squeeze(0) if val.shape[0] == 1 else val
        if "mm_token_type_ids" in enc:
            mm = enc["mm_token_type_ids"].squeeze(0)
            if mm.shape[0] > self.max_length:
                mm = mm[: self.max_length]
            result["mm_token_type_ids"] = mm

        return result


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_processed(batch: list[dict], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Pad and stack fully-processed items from ProcessedClipDataset."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    bs = len(batch)

    padded_ids = torch.full((bs, max_len), pad_id, dtype=torch.long)
    padded_labels = torch.full((bs, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
    has_mm = "mm_token_type_ids" in batch[0]
    padded_mm = torch.zeros((bs, max_len), dtype=torch.long) if has_mm else None

    all_pv = []
    all_grid = []

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        padded_ids[i, :seq_len] = item["input_ids"]
        padded_labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = 1
        if has_mm:
            padded_mm[i, :seq_len] = item["mm_token_type_ids"]
        for key in ("pixel_values", "pixel_values_videos"):
            if key in item:
                all_pv.append(item[key])
        for key in ("image_grid_thw", "video_grid_thw"):
            if key in item:
                all_grid.append(item[key])

    result = {
        "input_ids": padded_ids,
        "labels": padded_labels,
        "attention_mask": attention_mask,
    }
    if padded_mm is not None:
        result["mm_token_type_ids"] = padded_mm
    if all_pv:
        result[
            "pixel_values" if "pixel_values" in batch[0] else "pixel_values_videos"
        ] = torch.cat(all_pv, dim=0)
    if all_grid:
        result[
            "image_grid_thw" if "image_grid_thw" in batch[0] else "video_grid_thw"
        ] = torch.cat(all_grid, dim=0)
    return result
