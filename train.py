#!/usr/bin/env python3
"""Single-file SFT training for the IDM sparse-event format.

Trains Qwen3-VL (or compatible VLMs) to predict user input actions from
sequences of screenshot frames, using the same JSON format as the eval pipeline.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
import tyro
import wandb


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
        f"- Each KeyPress is ONE key. If \"hello\" is typed, that is 5 separate "
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


# ---------------------------------------------------------------------------
# Response parsing (from eval pipeline)
# ---------------------------------------------------------------------------

def parse_response(text: str) -> list[dict]:
    """Extract JSON action array from model output."""
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    if "<think>" in cleaned:
        cleaned = cleaned[:cleaned.index("<think>")]
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

def compute_f1(predictions: list[dict], ground_truth: list[dict], tolerance: int = 2) -> tuple[float, float, float]:
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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClipDataset(Dataset):
    """Dataset that does all heavy processing (JPEG decode + processor) in workers."""

    def __init__(self, jsonl_path: str, data_root: str, processor: Any, max_length: int = 8192):
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
            if input_ids[i:i + len(marker)] == marker:
                return i + len(marker)
        return 0

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        clip_dir = self.data_root / clip["clip_dir"]
        num_frames = clip["num_frames"]
        fps = clip["fps"]
        actions = clip["actions"]

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
            text=[full_text], videos=[frames], video_metadata=video_meta,
            return_tensors="pt", padding=False,
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[:self.max_length]

        # Find prompt length by locating assistant marker (avoids second processor call)
        prompt_len = self._find_prompt_len(input_ids.tolist())

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        mm_token_type_ids = full_inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.squeeze(0)
            if mm_token_type_ids.shape[0] > self.max_length:
                mm_token_type_ids = mm_token_type_ids[:self.max_length]

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
            result["pixel_values_videos"] = full_inputs["pixel_values_videos"].squeeze(0)
        if "video_grid_thw" in full_inputs:
            result["video_grid_thw"] = full_inputs["video_grid_thw"]
        return result


class RawClipDataset(Dataset):
    """Lightweight dataset: loads JPEG frames only, defers processing to collator."""

    def __init__(self, jsonl_path: str, data_root: str):
        self.data_root = Path(data_root)
        self.clips = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.clips.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        clip_dir = self.data_root / clip["clip_dir"]
        num_frames = clip["num_frames"]

        # JPEG decode in workers (parallel across processes, GIL-free in C)
        frames = np.stack([
            np.array(Image.open(clip_dir / f"F{i:02d}.jpg").convert("RGB"))
            for i in range(num_frames)
        ])  # (T, H, W, 3) uint8

        return {
            "frames": torch.from_numpy(frames),  # uint8 tensor for efficient shared-memory IPC
            "actions": clip["actions"],
            "fps": clip["fps"],
            "num_frames": clip["num_frames"],
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def build_sft_messages(frames: list[Image.Image], prompt: str, target: str, fps: float, video_mode: str = "video") -> list[dict]:
    """Build Qwen3-VL chat messages with video/images + prompt → target."""
    if video_mode == "image":
        content = [{"type": "image", "image": f} for f in frames]
    else:
        content = [{"type": "video", "video": frames, "fps": fps}]
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]


def build_prompt_only_messages(frames: list[Image.Image], prompt: str, fps: float, video_mode: str = "video") -> list[dict]:
    """Build messages for generation (no assistant response)."""
    if video_mode == "image":
        content = [{"type": "image", "image": f} for f in frames]
    else:
        content = [{"type": "video", "video": frames, "fps": fps}]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def collate_for_training(batch: list[dict], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Pad and stack pre-processed items from ClipDataset into a batch."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    padded_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    has_mm = "mm_token_type_ids" in batch[0]
    padded_mm = torch.zeros((len(batch), max_len), dtype=torch.long) if has_mm else None

    all_pixel_values = []
    all_grid_thw = []

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        padded_ids[i, :seq_len] = item["input_ids"]
        padded_labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = 1
        if has_mm:
            padded_mm[i, :seq_len] = item["mm_token_type_ids"]
        if "pixel_values_videos" in item:
            all_pixel_values.append(item["pixel_values_videos"])
        if "video_grid_thw" in item:
            all_grid_thw.append(item["video_grid_thw"])

    result = {
        "input_ids": padded_ids,
        "labels": padded_labels,
        "attention_mask": attention_mask,
    }
    if padded_mm is not None:
        result["mm_token_type_ids"] = padded_mm
    if all_pixel_values:
        result["pixel_values_videos"] = torch.cat(all_pixel_values, dim=0)
    if all_grid_thw:
        result["video_grid_thw"] = torch.cat(all_grid_thw, dim=0)
    return result


class VideoSFTCollator:
    """Batched collator: one processor call for the entire batch + label construction."""

    def __init__(self, processor: Any, max_length: int = 8192, video_mode: str = "video"):
        self.processor = processor
        self.max_length = max_length
        self.video_mode = video_mode
        self._assistant_marker = processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    def _find_prompt_len(self, input_ids: list[int]) -> int:
        marker = self._assistant_marker
        for i in range(len(input_ids) - len(marker), -1, -1):
            if input_ids[i : i + len(marker)] == marker:
                return i + len(marker)
        return 0

    def __call__(self, raw_batch: list[dict]) -> dict:
        full_text_B: list[str] = []
        # For video mode: list of frame lists; for image mode: flat list of all images
        videos_B: list[list] = []
        images_B: list[Image.Image] = []
        video_meta_B: list[dict] = []

        for item in raw_batch:
            frames_np = item["frames"].numpy()  # (T, H, W, 3) uint8
            frame_list = [Image.fromarray(frames_np[i]) for i in range(frames_np.shape[0])]
            fps = item["fps"]
            num_frames = item["num_frames"]
            target = json.dumps(item["actions"], separators=(",", ":"))

            prompt = build_prompt(fps, num_frames)
            msgs = build_sft_messages(frame_list, prompt, target, fps, video_mode=self.video_mode)
            text = self.processor.apply_chat_template(msgs, tokenize=False)
            full_text_B.append(text)

            if self.video_mode == "image":
                images_B.extend(frame_list)
            else:
                videos_B.append(frame_list)
                video_meta_B.append({
                    "fps": float(fps),
                    "total_num_frames": num_frames,
                    "frames_indices": list(range(num_frames)),
                })

        # Single batched processor call for the entire batch
        proc_kwargs: dict[str, Any] = {
            "text": full_text_B,
            "return_tensors": "pt",
            "padding": True,
        }
        if self.video_mode == "image":
            proc_kwargs["images"] = images_B
        else:
            proc_kwargs["videos"] = videos_B
            proc_kwargs["video_metadata"] = video_meta_B

        enc = self.processor(**proc_kwargs)

        input_ids = enc["input_ids"]
        if input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, : self.max_length]

        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            if attn_mask.shape[1] > self.max_length:
                attn_mask = attn_mask[:, : self.max_length]
        else:
            attn_mask = torch.ones_like(input_ids)

        # Build labels: mask prompt tokens and padding
        labels = input_ids.clone()
        for b in range(len(raw_batch)):
            prompt_len = self._find_prompt_len(input_ids[b].tolist())
            labels[b, :prompt_len] = -100
        labels[attn_mask == 0] = -100

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn_mask,
        }
        if "mm_token_type_ids" in enc:
            mm = enc["mm_token_type_ids"]
            if mm.shape[1] > self.max_length:
                mm = mm[:, : self.max_length]
            result["mm_token_type_ids"] = mm
        # Image mode uses pixel_values/image_grid_thw; video mode uses pixel_values_videos/video_grid_thw
        for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
            if key in enc:
                result[key] = enc[key]
        return result


class CollatorPrefetchIterator:
    """Overlaps collator CPU work with GPU compute via a background thread.

    While the GPU runs forward/backward on the current batch, the background
    thread prepares the next batch (apply_chat_template + processor + labels).
    CUDA ops release the GIL, so the collator thread runs during GPU work.
    """

    def __init__(self, dataloader_iter: Any, collator: VideoSFTCollator):
        self._dl_iter = dataloader_iter
        self._collator = collator
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._next_fut: Future | None = None
        self._closed = False
        self._submit_next()

    def _process_one(self) -> dict:
        raw_batch = next(self._dl_iter)
        return self._collator(raw_batch)

    def _submit_next(self) -> None:
        if self._closed:
            return
        self._next_fut = self._pool.submit(self._process_one)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=False)
        self._next_fut = None

    def __iter__(self) -> CollatorPrefetchIterator:
        return self

    def __next__(self) -> dict:
        if self._next_fut is None:
            raise StopIteration
        try:
            batch = self._next_fut.result()
        except StopIteration:
            self.close()
            raise
        except Exception:
            self.close()
            raise
        self._submit_next()
        return batch


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


def lr_at_step(step: int, warmup: int, max_steps: int, decay_steps: int, peak_lr: float) -> float:
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
    max_pixels: int = 200704  # 256 * 28 * 28 — tokens per image, controls resolution sent to model
    video_mode: str = "image"  # "image" = frames as separate images (fast, per-frame ViT); "video" = single video (slow, global ViT attention)
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
    save_every: int = 500
    val_every: int = 200
    val_steps: int = 20
    log_every: int = 10
    out_dir: str = "./runs/default"
    resume_from: str = ""
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
    assert args.data_dir, "--data-dir is required"

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")
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
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            ),
        )
    model = model.to(device)

    trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_n = sum(p.numel() for p in model.parameters())

    # MFU setup: detect H100 peak flops
    device_name = torch.cuda.get_device_name(local_rank).lower()
    peak_flops = 989e12 if "h100" in device_name and "pcie" not in device_name else (
        756e12 if "h100" in device_name else 312e12 if "a100" in device_name else 0.0)

    if rank == 0:
        print(f"Params: {trainable_n:,} trainable / {total_n:,} total ({trainable_n/total_n:.4%})")
        if peak_flops > 0:
            print(f"MFU enabled: peak_flops={peak_flops:.0e} ({device_name})")

    if world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    else:
        ddp_model = model
    raw_model = ddp_model.module if hasattr(ddp_model, "module") else ddp_model

    processor = AutoProcessor.from_pretrained(
        args.model_id, trust_remote_code=True,
        min_pixels=args.max_pixels, max_pixels=args.max_pixels,
    )
    if args.video_mode == "video" and hasattr(processor, "video_processor"):
        # Disable frame sampling so all frames are processed by the ViT.
        # WARNING: this triggers O(N²) global attention over all patches — very slow for many frames.
        processor.video_processor.do_sample_frames = False

    # Data — workers only load JPEGs; processing is batched in collator thread
    data_root = Path(args.data_dir)
    train_jsonl = data_root / "train" / "train.jsonl"
    assert train_jsonl.exists(), f"Missing {train_jsonl}"
    train_dataset = RawClipDataset(str(train_jsonl), str(data_root))
    collator = VideoSFTCollator(processor, args.max_length, video_mode=args.video_mode)
    if rank == 0:
        print(f"Train: {len(train_dataset)} clips")

    val_dataset = None
    if args.val_dir:
        val_root = Path(args.val_dir)
        val_jsonl = val_root / "val" / "val.jsonl"
        if val_jsonl.exists():
            val_dataset = ClipDataset(str(val_jsonl), str(val_root), processor, args.max_length)
            if rank == 0:
                print(f"Val: {len(val_dataset)} clips")

    pad_id = processor.tokenizer.pad_token_id or 0

    num_dl_workers = min(8, max(1, os.cpu_count() or 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_dl_workers,
        collate_fn=lambda batch: batch,  # identity: return list of raw items
        drop_last=True,
        pin_memory=False,  # raw uint8 frames, not model tensors
        prefetch_factor=2,
        persistent_workers=True,
    )
    if rank == 0:
        print(f"DataLoader: {num_dl_workers} workers + prefetch collator thread")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        [p for p in ddp_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_at_step(step, args.warmup_steps, args.max_steps, args.wsd_decay_steps, args.lr) / max(args.lr, 1e-12),
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Wandb
    wandb_run = None
    if rank == 0 and args.wandb_enable:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            mode=args.wandb_mode,
            dir=args.out_dir,
            config=asdict(args),
        )

    # Resume
    global_step = 0
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path / "checkpoint.pt", map_location=device, weights_only=False)
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

    # Prefetch iterator: collator runs on background thread while GPU computes
    dl_iter = iter(train_loader)
    prefetch_iter = CollatorPrefetchIterator(dl_iter, collator)

    if rank == 0:
        print(f"Starting training: {args.max_steps} steps, batch={args.batch_size}, grad_accum={args.grad_accum}")

    while global_step < args.max_steps:
        ddp_model.train()

        # Get next batch (already processed by prefetch thread)
        try:
            batch = next(prefetch_iter)
        except StopIteration:
            dl_iter = iter(train_loader)
            prefetch_iter = CollatorPrefetchIterator(dl_iter, collator)
            batch = next(prefetch_iter)

        # Move to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
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

        tok_n = int((labels != -100).sum().item())
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
            print(f"step={global_step} loss={avg_loss:.4f} lr={lr_val:.2e} tok/s={tps:.0f} samples/s={samples_per_s:.2f}{mfu_str}")
            if wandb_run:
                log_d = {"train/loss": avg_loss, "train/lr": lr_val, "train/tok_per_s": tps, "train/samples_per_s": samples_per_s}
                if peak_flops > 0:
                    log_d["train/mfu"] = mfu
                wandb_run.log(log_d, step=global_step)
            log_loss_sum = 0.0
            log_loss_n = 0
            log_tok_n = 0
            t0 = time.time()

        # Validation
        if args.val_every > 0 and global_step % args.val_every == 0 and val_dataset and rank == 0:
            val_metrics = run_validation(raw_model, processor, val_dataset, args, device, dtype, pad_id)
            print(f"  val: loss={val_metrics['loss']:.4f} f1={val_metrics['f1']:.3f} "
                  f"p={val_metrics['precision']:.3f} r={val_metrics['recall']:.3f}")
            if wandb_run:
                wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

        # Save checkpoint
        if args.save_every > 0 and global_step % args.save_every == 0 and rank == 0:
            save_dir = Path(args.out_dir) / f"step_{global_step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "global_step": global_step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, save_dir / "checkpoint.pt")
            print(f"  saved checkpoint to {save_dir}")

    prefetch_iter.close()
    if rank == 0:
        print("Training complete.")
    if wandb_run:
        wandb_run.finish()
    if world_size > 1:
        dist.destroy_process_group()


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
    indices = indices[:args.val_steps]

    with torch.no_grad():
        for idx in indices:
            item = val_dataset[idx]
            batch = collate_for_training([item], pad_id)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            if "pixel_values_videos" in batch:
                model_inputs["pixel_values_videos"] = batch["pixel_values_videos"].to(device, dtype=dtype)
            if "video_grid_thw" in batch:
                model_inputs["video_grid_thw"] = batch["video_grid_thw"].to(device)
            if "mm_token_type_ids" in batch:
                model_inputs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(device)

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
                gen_inputs["pixel_values_videos"] = batch["pixel_values_videos"].to(device, dtype=dtype)
            if "video_grid_thw" in batch:
                gen_inputs["video_grid_thw"] = batch["video_grid_thw"].to(device)
            if "mm_token_type_ids" in batch:
                gen_inputs["mm_token_type_ids"] = batch["mm_token_type_ids"][:, prompt_mask].contiguous().to(device)

            with torch.autocast("cuda", dtype=dtype):
                gen_ids = model.generate(
                    **gen_inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True,
                )
            # Decode only generated tokens
            prompt_len = gen_inputs["input_ids"].shape[1]
            pred_text = processor.tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)
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
