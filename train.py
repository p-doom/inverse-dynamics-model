#!/usr/bin/env python3
"""Single-file SFT training for the IDM sparse-event format.

Trains Qwen3-VL (or compatible VLMs) to predict user input actions from
sequences of screenshot frames, using the same JSON format as the eval pipeline.
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
import tyro
import wandb

# Scoring functions from the eval pipeline (for inline eval)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval-models"))
from score_eval import (
    filter_gt_actions,
    coalesce_gt_events,
    filter_predictions,
    match_clip,
)


# ---------------------------------------------------------------------------
# Prompt (identical to eval pipeline)
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


# ---------------------------------------------------------------------------
# Response parsing (from eval pipeline)
# ---------------------------------------------------------------------------


def parse_response(text: str) -> list[dict]:
    """Extract JSON action array from model output."""
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    if "<think>" in cleaned:
        cleaned = cleaned[: cleaned.index("<think>")]
    cleaned = re.sub(r"<\|begin_of_box\|>", "", cleaned)
    cleaned = re.sub(r"<\|end_of_box\|>", "", cleaned)
    cleaned = cleaned.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return []


# ---------------------------------------------------------------------------
# Bipartite matching for F1 (from eval/score_eval.py)
# ---------------------------------------------------------------------------


def compute_f1(
    predictions: list[dict], ground_truth: list[dict], tolerance: int = 2
) -> tuple[float, float, float]:
    """Greedy bipartite matching between predicted and GT actions.

    Returns (precision, recall, f1).
    """
    if not predictions and not ground_truth:
        return 1.0, 1.0, 1.0
    if not predictions or not ground_truth:
        return 0.0, 0.0, 0.0

    def _frame_idx(action: dict) -> int:
        f = action.get("frame", "F00")
        return int(re.sub(r"[^0-9]", "", f))

    matched_gt = set()
    tp = 0
    for pred in predictions:
        best_dist = tolerance + 1
        best_j = -1
        for j, gt in enumerate(ground_truth):
            if j in matched_gt:
                continue
            if pred.get("type") != gt.get("type"):
                continue
            dist_val = abs(_frame_idx(pred) - _frame_idx(gt))
            if dist_val <= tolerance and dist_val < best_dist:
                best_dist = dist_val
                best_j = j
        if best_j >= 0:
            tp += 1
            matched_gt.add(best_j)

    precision = tp / len(predictions) if predictions else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ClipDataset(Dataset):
    """Dataset that does all heavy processing (JPEG decode + processor) in workers."""

    def __init__(
        self, jsonl_path: str, data_root: str, processor: Any, max_length: int = 8192
    ):
        self.data_root = Path(data_root)
        self.processor = processor
        self.max_length = max_length
        # Pre-compute the assistant marker token ids for fast prompt_len detection
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

        # Load frames (CPU-heavy: JPEG decode)
        frames = []
        for i in range(num_frames):
            img_path = clip_dir / f"F{i:02d}.jpg"
            frames.append(Image.open(img_path).convert("RGB"))

        prompt = build_prompt(fps, num_frames)
        target = json.dumps(actions, separators=(",", ":"))
        video_meta = [{"fps": float(fps), "total_num_frames": len(frames)}]

        # Single processor call for full message (prompt + target)
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

        # Find prompt length by locating assistant marker (avoids second processor call)
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
    """Dataset that does FULL processing in DataLoader workers (parallel processes).

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
        actions = clip["actions"]
        actions = normalize_actions(actions)

        processor = self._get_processor()

        # Load frames
        frame_list = [
            Image.open(clip_dir / f"F{i:02d}.jpg").convert("RGB")
            for i in range(num_frames)
        ]

        # Build messages + target
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

        # Process
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

        # Vision tensors
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


def build_sft_messages(
    frames: list[Image.Image],
    prompt: str,
    target: str,
    fps: float,
    video_mode: str = "video",
    interleave_labels: bool = False,
) -> list[dict]:
    """Build Qwen3-VL chat messages with video/images + prompt → target."""
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
# Training utilities
# ---------------------------------------------------------------------------


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    valid = (shift_labels != -100).float()
    return (loss * valid).sum() / valid.sum().clamp(min=1)


def lr_at_step(
    step: int, warmup: int, max_steps: int, decay_steps: int, peak_lr: float
) -> float:
    """WSD schedule: warmup → stable → decay."""
    if step < warmup:
        return peak_lr * step / max(warmup, 1)
    stable_end = max_steps - decay_steps
    if step < stable_end:
        return peak_lr
    progress = (step - stable_end) / max(decay_steps, 1)
    return peak_lr * (1 - progress) * 0.9 + peak_lr * 0.1  # decay to 10%


def seed_all(seed: int, rank: int = 0) -> None:
    s = seed + 100_003 * rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


@dataclass
class Args:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    attn_implementation: str = "flash_attention_2"
    data_dir: str = ""
    val_dir: str = ""
    max_length: int = 8192
    max_pixels: int = (
        200704  # 256 * 28 * 28 — tokens per image, controls resolution sent to model
    )
    video_mode: str = (
        "image"  # "image" = frames as separate images (fast, per-frame ViT); "video" = single video (slow, global ViT attention)
    )
    interleave_labels: bool = (
        False  # Insert "Frame F00:", "Frame F01:", ... text before each image
    )
    batch_size: int = 1
    grad_accum: int = 4
    max_grad_norm: float = 1.0
    max_steps: int = 5000
    lr: float = 2e-5
    warmup_steps: int = 100
    wsd_decay_steps: int = 500
    weight_decay: float = 0.0
    precision: str = "bf16"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    train_vision: bool = (
        False  # Apply LoRA to ViT layers too (qkv, attn.proj, linear_fc1/fc2)
    )
    vision_lr_scale: float = (
        0.1  # LR multiplier for ViT params when doing full fine-tune
    )
    save_every: int = 500
    val_every: int = 200
    val_steps: int = 20
    eval_clips_dir: str = (
        ""  # Path to eval clips (mp4+json pairs). If set, runs real eval with proper scoring.
    )
    eval_coalesce: bool = True  # Match training data format in eval scoring
    eval_tolerance: int = 2
    log_every: int = 10
    out_dir: str = "./runs/default"
    resume_from: str = ""
    run_id: str = ""  # Unique run ID for wandb (e.g. SLURM_JOB_ID)
    wandb_enable: bool = True
    wandb_project: str = "idm"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_mode: str = "offline"
    seed: int = 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = tyro.cli(Args)
    if not args.data_dir:
        raise ValueError("--data-dir is required")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    seed_all(args.seed, rank)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    # Model
    if rank == 0:
        print(f"Loading model {args.model_id}...")
    model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    # Workaround: cuDNN 9.1 has a pathological bf16 Conv3d kernel for the ViT's patch
    # embedding shape (large batch, tiny spatial). torch.compile bypasses it (3.6s → 22ms).
    model.model.visual.patch_embed = torch.compile(model.model.visual.patch_embed)

    if args.use_lora:
        # LLM targets (always)
        lora_targets = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
        if args.train_vision:
            # Add ViT attention + MLP layers (Qwen3-VL uses fused QKV, not separate q/k/v)
            lora_targets.extend(["qkv", "attn.proj", "linear_fc1", "linear_fc2"])
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                target_modules=lora_targets,
            ),
        )
    else:
        # Full fine-tune: ensure all params are trainable
        for p in model.parameters():
            p.requires_grad = True
    model = model.to(device)

    trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_n = sum(p.numel() for p in model.parameters())

    # MFU setup: detect H100 peak flops
    device_name = torch.cuda.get_device_name(local_rank).lower()
    if ("h100" in device_name or "gh200" in device_name) and "pcie" not in device_name:
        peak_flops = 989e12
    elif "h100" in device_name:
        peak_flops = 756e12
    elif "a100" in device_name:
        peak_flops = 312e12
    else:
        peak_flops = 0.0

    if rank == 0:
        print(
            f"Params: {trainable_n:,} trainable / {total_n:,} total ({trainable_n/total_n:.4%})"
        )
        if peak_flops > 0:
            print(f"MFU enabled: peak_flops={peak_flops:.0e} ({device_name})")

    if world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    else:
        ddp_model = model
    raw_model = ddp_model.module if hasattr(ddp_model, "module") else ddp_model

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        min_pixels=3136,
        max_pixels=args.max_pixels,  # min=4*28*28, let processor choose grid
    )
    if args.video_mode == "video" and hasattr(processor, "video_processor"):
        # Disable frame sampling so all frames are processed by the ViT.
        # WARNING: this triggers O(N²) global attention over all patches — very slow for many frames.
        processor.video_processor.do_sample_frames = False

    # Data
    data_root = Path(args.data_dir)
    train_jsonl = data_root / "train" / "train.jsonl"
    if not train_jsonl.exists():
        raise FileNotFoundError(f"Train JSONL not found: {train_jsonl}")

    train_dataset = ProcessedClipDataset(
        str(train_jsonl),
        str(data_root),
        args.model_id,
        args.max_pixels,
        args.max_length,
        args.video_mode,
        args.interleave_labels,
    )
    if rank == 0:
        print(f"Train: {len(train_dataset)} clips")

    val_dataset = None
    if args.val_dir:
        val_jsonl = Path(args.val_dir) / "val" / "val.jsonl"
        if not val_jsonl.exists():
            raise FileNotFoundError(f"Val JSONL not found: {val_jsonl}")
        val_dataset = ClipDataset(
            str(val_jsonl), str(Path(args.val_dir)), processor, args.max_length
        )
        if rank == 0:
            print(f"Val: {len(val_dataset)} clips")

    # Load real eval clips (mp4+json) if configured
    eval_clips = []
    if args.eval_clips_dir:
        eval_clips = discover_eval_clips(args.eval_clips_dir)
        if rank == 0:
            print(f"Eval clips: {len(eval_clips)} (from {args.eval_clips_dir})")

    pad_id = processor.tokenizer.pad_token_id or 0

    num_dl_workers = min(8, max(1, os.cpu_count() or 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_dl_workers,
        collate_fn=lambda batch: collate_processed(batch, pad_id),
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    if rank == 0:
        print(f"DataLoader: {num_dl_workers} workers (pin_memory)")

    # Optimizer + scheduler — use separate param group for ViT with lower LR in full-FT mode
    if not args.use_lora and args.vision_lr_scale != 1.0:
        vision_params = []
        other_params = []
        for name, p in ddp_model.named_parameters():
            if not p.requires_grad:
                continue
            if "visual" in name:
                vision_params.append(p)
            else:
                other_params.append(p)
        param_groups = [
            {"params": other_params, "lr": args.lr},
            {"params": vision_params, "lr": args.lr * args.vision_lr_scale},
        ]
        if rank == 0:
            print(
                f"Param groups: LLM={len(other_params)} params @ lr={args.lr}, "
                f"ViT={len(vision_params)} params @ lr={args.lr * args.vision_lr_scale}"
            )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            [p for p in ddp_model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_at_step(
            step, args.warmup_steps, args.max_steps, args.wsd_decay_steps, args.lr
        )
        / max(args.lr, 1e-12),
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Wandb — use run_id so evals can log to the same run
    wandb_run = None
    wandb_id = f"train-{args.run_id}" if args.run_id else None
    if rank == 0 and args.wandb_enable:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            id=wandb_id,
            resume="allow",
            mode=args.wandb_mode,
            dir=args.out_dir,
            config=asdict(args),
        )

    # Resume
    global_step = 0
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if ckpt_path.exists():
            ckpt = torch.load(
                ckpt_path / "checkpoint.pt", map_location=device, weights_only=False
            )
            raw_model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            global_step = ckpt["global_step"]
            if rank == 0:
                print(f"Resumed from step {global_step}")

    # Training loop
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    log_loss_sum = 0.0
    log_loss_n = 0
    log_tok_n = 0
    t0 = time.time()

    # Data iteration setup
    data_iter = iter(train_loader)

    if rank == 0:
        print(
            f"Starting training: {args.max_steps} steps, batch={args.batch_size}, grad_accum={args.grad_accum}"
        )

    while global_step < args.max_steps:
        ddp_model.train()

        # Get next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Move to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for key in ("pixel_values", "pixel_values_videos"):
            if key in batch:
                model_inputs[key] = batch[key].to(device, dtype=dtype)
        for key in ("image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
            if key in batch:
                model_inputs[key] = batch[key].to(device)

        with torch.autocast("cuda", dtype=dtype):
            outputs = ddp_model(**model_inputs)
            loss = outputs.loss / args.grad_accum

        loss.backward()
        micro_step += 1

        tok_n = int(attention_mask.sum().item())
        log_loss_sum += float(loss.detach().item()) * args.grad_accum
        log_loss_n += 1
        log_tok_n += tok_n

        if micro_step < args.grad_accum:
            continue

        # Optimizer step
        micro_step = 0
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in ddp_model.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        global_step += 1

        # Logging
        if global_step % args.log_every == 0 and rank == 0:
            dt = max(time.time() - t0, 1e-9)
            avg_loss = log_loss_sum / max(log_loss_n, 1)
            tps = log_tok_n / dt
            samples_per_s = (args.log_every * args.batch_size * world_size) / dt
            lr_val = optimizer.param_groups[0]["lr"]
            # MFU: approx 6 * N * tokens_per_s / peak_flops (forward + backward)
            mfu = (6 * total_n * tps / peak_flops) if peak_flops > 0 else 0.0
            mfu_str = f" mfu={mfu:.4f}" if peak_flops > 0 else ""
            print(
                f"step={global_step} loss={avg_loss:.4f} lr={lr_val:.2e} tok/s={tps:.0f} samples/s={samples_per_s:.2f}{mfu_str}"
            )
            if wandb_run:
                log_d = {
                    "step": global_step,
                    "train/loss": avg_loss,
                    "train/lr": lr_val,
                    "train/tok_per_s": tps,
                    "train/samples_per_s": samples_per_s,
                }
                if peak_flops > 0:
                    log_d["train/mfu"] = mfu
                wandb_run.log(log_d)
            log_loss_sum = 0.0
            log_loss_n = 0
            log_tok_n = 0
            t0 = time.time()

        # Validation + eval + save — all rank-0-only, so barrier first to prevent NCCL timeout
        is_val_step = args.val_every > 0 and global_step % args.val_every == 0
        is_save_step = args.save_every > 0 and global_step % args.save_every == 0
        if (is_val_step or is_save_step) and world_size > 1:
            dist.barrier()  # all ranks wait here before rank 0 does slow eval/save

        # Validation (on training val split)
        if is_val_step and val_dataset and rank == 0:
            val_metrics = run_validation(
                raw_model, processor, val_dataset, args, device, dtype, pad_id
            )
            print(
                f"  val: loss={val_metrics['loss']:.4f} f1={val_metrics['f1']:.3f} "
                f"p={val_metrics['precision']:.3f} r={val_metrics['recall']:.3f}"
            )
            if wandb_run:
                val_log = {f"val/{k}": v for k, v in val_metrics.items()}
                val_log["step"] = global_step
                wandb_run.log(val_log)

        # Real eval on mp4 clips — distributed across all ranks for speed
        if is_val_step and eval_clips:
            # Each rank handles a shard of clips
            my_clips = eval_clips[rank::world_size] if world_size > 1 else eval_clips
            local_metrics = run_real_eval(
                raw_model, processor, my_clips, args, device, dtype
            )

            # All-reduce TP/FP/FN across ranks
            if world_size > 1:
                counts = torch.tensor(
                    [
                        local_metrics["eval/tp"],
                        local_metrics["eval/fp"],
                        local_metrics["eval/fn"],
                    ],
                    device=device,
                    dtype=torch.long,
                )
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                tp, fp, fn = counts.tolist()
            else:
                tp = local_metrics["eval/tp"]
                fp = local_metrics["eval/fp"]
                fn = local_metrics["eval/fn"]

            if rank == 0:
                p = tp / max(tp + fp, 1)
                r_val = tp / max(tp + fn, 1)
                f1 = 2 * p * r_val / max(p + r_val, 1e-9)
                print(
                    f"  EVAL: f1={f1:.3f} p={p:.3f} r={r_val:.3f} (TP={tp} FP={fp} FN={fn})"
                )
                if wandb_run:
                    wandb_run.log(
                        {
                            "step": global_step,
                            "eval/f1": f1,
                            "eval/precision": p,
                            "eval/recall": r_val,
                            "eval/tp": tp,
                            "eval/fp": fp,
                            "eval/fn": fn,
                        }
                    )

        # Wait for rank 0 to finish eval before all ranks proceed
        if (is_val_step or is_save_step) and world_size > 1:
            dist.barrier()

        # Save checkpoint
        if is_save_step and rank == 0:
            save_dir = Path(args.out_dir) / f"step_{global_step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "global_step": global_step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                save_dir / "checkpoint.pt",
            )
            print(f"  saved checkpoint to {save_dir}")

    if rank == 0:
        print("Training complete.")
    if wandb_run:
        wandb_run.finish()
    if world_size > 1:
        dist.destroy_process_group()


def discover_eval_clips(clips_dir: str) -> list[dict]:
    """Find all clip mp4/json pairs under clips_dir (same as eval pipeline)."""
    clips = []
    for json_path in sorted(Path(clips_dir).rglob("clip_*.json")):
        mp4_path = json_path.with_suffix(".mp4")
        if not mp4_path.exists():
            continue
        with open(json_path) as f:
            meta = json.load(f)
        clips.append(
            {
                "mp4_path": str(mp4_path),
                "clip_name": json_path.stem,
                "start_s": meta["start_s"],
                "end_s": meta["end_s"],
                "tag": meta["tag"],
                "actions": meta["actions"],
            }
        )
    return clips


def extract_frames_for_eval(mp4_path: str, fps: int) -> list[Image.Image]:
    """Extract frames from mp4 at given fps, return as PIL Images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pattern = os.path.join(tmpdir, "frame_%04d.png")
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                mp4_path,
                "-vf",
                f"fps={fps}",
                pattern,
                "-y",
                "-loglevel",
                "error",
            ],
            check=True,
        )
        frame_paths = sorted(glob.glob(os.path.join(tmpdir, "frame_*.png")))
        return [Image.open(p).convert("RGB") for p in frame_paths]


def run_real_eval(
    model: torch.nn.Module,
    processor: Any,
    eval_clips: list[dict],
    args: Args,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Run eval on real mp4 clips with proper scoring (coalesce, detail matching).

    Uses the same frame processing and prompt as training, ensuring config match.
    """
    model.eval()
    # Use the same FPS as training data (read from first training clip)
    data_root = Path(args.data_dir)
    train_jsonl = data_root / "train" / "train.jsonl"
    with open(train_jsonl) as f:
        first_clip = json.loads(f.readline())
    fps = first_clip.get("fps", 5)
    tp_total, fp_total, fn_total = 0, 0, 0

    with torch.no_grad():
        for i, clip in enumerate(eval_clips):
            try:
                frames = extract_frames_for_eval(clip["mp4_path"], fps)
                if not frames:
                    continue

                prompt = build_prompt(fps, len(frames))

                # Build messages matching training format exactly
                if args.interleave_labels:
                    content = []
                    for j, f in enumerate(frames):
                        content.append({"type": "text", "text": f"Frame F{j:02d}:"})
                        content.append({"type": "image", "image": f})
                else:
                    content = [{"type": "image", "image": f} for f in frames]
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[text], images=frames, return_tensors="pt", padding=True
                )
                inputs = {
                    k: v.to(device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
                if "pixel_values_videos" in inputs:
                    inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(
                        dtype=dtype
                    )

                with torch.autocast("cuda", dtype=dtype):
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        use_cache=True,
                    )
                prompt_len = inputs["input_ids"].shape[1]
                pred_text = processor.tokenizer.decode(
                    gen_ids[0][prompt_len:], skip_special_tokens=True
                )
                predictions = parse_response(pred_text)

                # Score with proper matching — filter_predictions handles format normalization
                gt = filter_gt_actions(clip["actions"], clip["start_s"], fps)
                if args.eval_coalesce:
                    gt = coalesce_gt_events(gt)
                preds_f = filter_predictions(predictions)
                result = match_clip(gt, preds_f, tolerance=args.eval_tolerance)
                tp_total += len(result["matches"])
                fp_total += len(result["unmatched_preds"])
                fn_total += len(result["unmatched_gt"])

            except Exception as e:
                print(f"  eval clip {i} error: {e}")
                continue

            torch.cuda.empty_cache()

    model.train()

    p = tp_total / max(tp_total + fp_total, 1)
    r = tp_total / max(tp_total + fn_total, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {
        "eval/f1": f1,
        "eval/precision": p,
        "eval/recall": r,
        "eval/tp": tp_total,
        "eval/fp": fp_total,
        "eval/fn": fn_total,
    }


def run_validation(
    model: torch.nn.Module,
    processor: Any,
    val_dataset: ClipDataset,
    args: Args,
    device: torch.device,
    dtype: torch.dtype,
    pad_id: int = 0,
) -> dict[str, float]:
    """Run validation: compute loss + generate predictions for F1."""
    model.eval()
    val_loss_sum = 0.0
    val_loss_n = 0
    all_p, all_r, all_f1 = [], [], []

    indices = list(range(len(val_dataset)))
    random.shuffle(indices)
    indices = indices[: args.val_steps]

    with torch.no_grad():
        for idx in indices:
            item = val_dataset[idx]
            batch = collate_processed([item], pad_id)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            if "pixel_values_videos" in batch:
                model_inputs["pixel_values_videos"] = batch["pixel_values_videos"].to(
                    device, dtype=dtype
                )
            if "video_grid_thw" in batch:
                model_inputs["video_grid_thw"] = batch["video_grid_thw"].to(device)
            if "mm_token_type_ids" in batch:
                model_inputs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(
                    device
                )

            with torch.autocast("cuda", dtype=dtype):
                outputs = model(**model_inputs)
            val_loss_sum += float(outputs.loss.item())
            val_loss_n += 1

            # Generate for F1 — use prompt portion of pre-processed inputs
            prompt_mask = (labels[0] == -100).cpu()
            gen_input_ids = batch["input_ids"][:, prompt_mask].contiguous().to(device)
            gen_attn = batch["attention_mask"][:, prompt_mask].contiguous().to(device)
            gen_inputs = {"input_ids": gen_input_ids, "attention_mask": gen_attn}
            if "pixel_values_videos" in batch:
                gen_inputs["pixel_values_videos"] = batch["pixel_values_videos"].to(
                    device, dtype=dtype
                )
            if "video_grid_thw" in batch:
                gen_inputs["video_grid_thw"] = batch["video_grid_thw"].to(device)
            if "mm_token_type_ids" in batch:
                gen_inputs["mm_token_type_ids"] = (
                    batch["mm_token_type_ids"][:, prompt_mask].contiguous().to(device)
                )

            with torch.autocast("cuda", dtype=dtype):
                gen_ids = model.generate(
                    **gen_inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True,
                )
            # Decode only generated tokens
            prompt_len = gen_inputs["input_ids"].shape[1]
            pred_text = processor.tokenizer.decode(
                gen_ids[0][prompt_len:], skip_special_tokens=True
            )
            predictions = parse_response(pred_text)
            p, r, f1 = compute_f1(predictions, item["actions"])
            all_p.append(p)
            all_r.append(r)
            all_f1.append(f1)

    model.train()
    return {
        "loss": val_loss_sum / max(val_loss_n, 1),
        "precision": sum(all_p) / max(len(all_p), 1),
        "recall": sum(all_r) / max(len(all_r), 1),
        "f1": sum(all_f1) / max(len(all_f1), 1),
    }


if __name__ == "__main__":
    main()
