from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import json
import os
import random
import re
import time
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw
from peft import LoraConfig, get_peft_model
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import tyro
import wandb

from utils.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from utils.collator import CollatorPrefetchIterator, VideoSFTCollator
from utils.data_jpeg import (
    find_array_record_paths,
    get_dataloader,
)
from utils.lr_schedules import LRScheduleArgs, lr_at_step


# ---------------------------------------------------------------------------
# Constants from the data-gen pipeline (must match generate_simulated_dataset)
# ---------------------------------------------------------------------------

MOUSE_X_QUANT_UNIT_F = 5.0
MOUSE_Y_QUANT_UNIT_F = 4.0
MOUSE_DELTA_CLIP_I = 64
MOUSE_DELTA_EXP_CURVATURE_F = 1.0

# ---------------------------------------------------------------------------
# Action helpers — mouse-only
# ---------------------------------------------------------------------------

_ACTION_CLASS_NAMES = ("no_op", "mouse")
_ACTION_CONFUSION_PRED_CLASS_NAMES = ("no_op", "mouse", "missing")

_MOUSE_RE = re.compile(r"^MOUSE:\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*$")


def _action_class_s(action_s: str) -> str:
    action_s = action_s.strip()
    if action_s in ("NO_OP", ""):
        return "no_op"
    if action_s.startswith("MOUSE:"):
        return "mouse"
    return "no_op"


def _action_is_no_op(action_s: str) -> bool:
    return _action_class_s(action_s) == "no_op"


def _parse_mouse_delta(action_s: str) -> tuple[int, int, int] | None:
    """Return (dx_q, dy_q, scroll_q) from a MOUSE:dx,dy,scroll string, or None."""
    m = _MOUSE_RE.match(action_s.strip())
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# ---------------------------------------------------------------------------
# Inverse-quantize: map quantized delta back to pixel delta
# ---------------------------------------------------------------------------

def _inv_quantize_exp(
    q_i: int,
    quant_unit_f: float,
    clip_abs_i: int,
    curvature_f: float = MOUSE_DELTA_EXP_CURVATURE_F,
) -> float:
    """Approximate inverse of _quantize_mouse_delta_exponential."""
    if clip_abs_i <= 0 or q_i == 0:
        return 0.0
    sign = -1.0 if q_i < 0 else 1.0
    abs_q = min(abs(q_i), clip_abs_i)
    curved = abs_q / float(clip_abs_i)  # in [0, 1]
    # curved = log1p(c*n) / log1p(c)  →  n = (exp(curved*log1p(c)) - 1) / c
    log1p_c = float(np.log1p(curvature_f))
    normalized = (np.expm1(curved * log1p_c)) / curvature_f
    max_val = quant_unit_f * float(clip_abs_i)
    return sign * normalized * max_val


def _deltas_to_pixel_offsets(
    dx_q: int, dy_q: int
) -> tuple[float, float]:
    """Convert quantized deltas to pixel-space deltas."""
    dx_px = _inv_quantize_exp(dx_q, MOUSE_X_QUANT_UNIT_F, MOUSE_DELTA_CLIP_I)
    dy_px = _inv_quantize_exp(dy_q, MOUSE_Y_QUANT_UNIT_F, MOUSE_DELTA_CLIP_I)
    return dx_px, dy_px


# ---------------------------------------------------------------------------
# Visual cursor overlay
# ---------------------------------------------------------------------------

def _accumulate_positions(
    actions: list[str],
    img_w: int,
    img_h: int,
    start_x: float | None = None,
    start_y: float | None = None,
) -> list[tuple[int, int]]:
    """Walk through a list of action strings and accumulate absolute cursor
    positions (clamped to image bounds)."""
    cx = img_w / 2.0 if start_x is None else start_x
    cy = img_h / 2.0 if start_y is None else start_y
    positions: list[tuple[int, int]] = []
    for a in actions:
        parsed = _parse_mouse_delta(a)
        if parsed is not None:
            dx_q, dy_q, _ = parsed
            dx_px, dy_px = _deltas_to_pixel_offsets(dx_q, dy_q)
            cx = float(np.clip(cx + dx_px, 0, img_w - 1))
            cy = float(np.clip(cy + dy_px, 0, img_h - 1))
        positions.append((int(round(cx)), int(round(cy))))
    return positions


def _draw_cursor(
    img: Image.Image,
    x: int,
    y: int,
    color: tuple[int, int, int],
    radius: int = 4,
    outline_color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    draw = ImageDraw.Draw(img)
    bbox = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(bbox, fill=color, outline=outline_color, width=1)


def _render_cursor_frames(
    jpeg_frames: list[bytes],
    gt_actions: list[str],
    pred_actions: list[str],
    img_w: int,
    img_h: int,
    max_frames: int = 16,
    upscale: int = 4,
) -> list[Image.Image]:
    """Render up to `max_frames` frames with GT (green) and pred (red) cursors.

    Each frame is upscaled for visibility.
    """
    n = min(len(jpeg_frames), len(gt_actions), max_frames)
    gt_pos = _accumulate_positions(gt_actions[:n], img_w, img_h)
    pred_pos = _accumulate_positions(pred_actions[:n], img_w, img_h)

    rendered: list[Image.Image] = []
    radius = max(2, 3 * upscale // 4)
    for i in range(n):
        frame_img = Image.open(io.BytesIO(jpeg_frames[i])).convert("RGB")
        if upscale > 1:
            frame_img = frame_img.resize(
                (img_w * upscale, img_h * upscale), Image.NEAREST,
            )
        gx, gy = gt_pos[i][0] * upscale, gt_pos[i][1] * upscale
        try:
            px, py = pred_pos[i][0] * upscale, pred_pos[i][1] * upscale
        except IndexError:
            px, py = 0,0
        _draw_cursor(frame_img, gx, gy, color=(0, 220, 0), radius=radius)   # GT green
        _draw_cursor(frame_img, px, py, color=(220, 0, 0), radius=radius)   # pred red
        rendered.append(frame_img)
    return rendered


def _make_grid(
    images: list[Image.Image],
    cols: int = 8,
) -> Image.Image:
    """Tile a list of PIL images into a single grid image."""
    if not images:
        return Image.new("RGB", (1, 1))
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * h), (30, 30, 30))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * w, r * h))
    return grid


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


@dataclass
class Args:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    attn_implementation: str = "sdpa"
    data_root: str = ""
    image_h: int = 540
    image_w: int = 960
    image_c: int = 3
    video_fps: float = 10.0
    seq_len: int = 64
    train_min_action_density: float = 0.0
    global_batch_size: int = 4
    grad_accum: int = 8
    max_grad_norm: float = 1.0
    max_steps: int = 5000
    lr: float = 2e-5
    init_lr: float = 0.0
    decay_end: float = 0.0
    lr_schedule: str = "wsd"
    warmup_steps: int = 100
    wsd_decay_steps: int = 200
    weight_decay: float = 0.0
    precision: str = "bf16"
    grad_checkpointing: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 0
    num_workers: int = 4
    prefetch_buffer_size: int = 8
    read_num_threads: int = 4
    worker_buffer_size: int = 4
    collator_prefetch: bool = True
    log_every: int = 1
    val_every: int = 1
    val_steps: int = 4
    val_generate_max_new_tokens: int = 512
    val_log_examples: int = 2
    val_visual_every: int = 1
    """Log visual cursor overlay frames to WandB every N optimizer steps (0=off)."""
    val_visual_max_frames: int = 16
    """Max frames per visual sample to render."""
    val_visual_upscale: int = 4
    """Upscale factor for rendered cursor frames."""
    save_every: int = 1000
    out_dir: str = "./runs/mouse_sim"
    resume_from: str = ""
    instruction_text: str = (
        "Given the video frames, output the mouse action for each frame in order."
    )
    no_op_loss_weight: float = 1.0
    mouse_loss_weight: float = 1.0
    wandb_enable: bool = True
    wandb_project: str = "idm-mouse"
    wandb_entity: str = "instant-uv"
    wandb_run_name: str = "idm_mouse_run"
    wandb_mode: str = "offline"
    mfu_peak_flops: float = 0.0


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------

def _rng_state_d() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _set_rng_state(state_d: dict[str, Any]) -> None:
    random.setstate(state_d["python"])
    np.random.set_state(state_d["numpy"])
    torch.set_rng_state(state_d["torch_cpu"])
    if torch.cuda.is_available() and state_d.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state_d["torch_cuda"])


def _seed_all(seed_i: int, rank_i: int) -> None:
    seed_i = seed_i + 100_003 * rank_i
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    torch.cuda.manual_seed_all(seed_i)


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def _build_scheduler(optimizer: torch.optim.Optimizer, args: Args):
    cfg = LRScheduleArgs(
        schedule=args.lr_schedule,
        init_lr=args.init_lr,
        max_lr=args.lr,
        decay_end=args.decay_end,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        wsd_decay_steps=args.wsd_decay_steps,
    )
    base_lr = max(args.lr, 1e-12)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step_i: lr_at_step(cfg, int(step_i)) / base_lr,
    )


# ---------------------------------------------------------------------------
# Grad helpers
# ---------------------------------------------------------------------------

def _grad_norm_and_clip(
    params: list[torch.nn.Parameter],
    max_grad_norm: float,
) -> float:
    grad_params = [p for p in params if p.grad is not None]
    if not grad_params:
        return 0.0
    clip_max_norm = float(max_grad_norm) if max_grad_norm > 0 else float("inf")
    total_norm_t = torch.nn.utils.clip_grad_norm_(grad_params, clip_max_norm)
    if isinstance(total_norm_t, torch.Tensor):
        return float(total_norm_t.detach().item())
    return float(total_norm_t)


def _to_device(
    batch_d: dict[str, Any],
    device: torch.device,
    skip_keys: set[str] | None = None,
) -> dict[str, Any]:
    skip_keys = (
        {"videos", "prompt_lens", "meta", "label_weights"}
        if skip_keys is None
        else set(skip_keys)
    )
    out_d: dict[str, Any] = {}
    for key_s, val in batch_d.items():
        if key_s in skip_keys:
            continue
        out_d[key_s] = val.to(device, non_blocking=True)
    return out_d


# ---------------------------------------------------------------------------
# Action parsing from target text
# ---------------------------------------------------------------------------

def _actions_from_target_text(target_s: str) -> list[str]:
    actions_L = []
    for line_s in target_s.splitlines():
        line_s = line_s.strip()
        if not line_s.startswith("Frame "):
            continue
        parts_L = line_s.split(":", 1)
        if len(parts_L) != 2:
            continue
        actions_L.append(parts_L[1].strip())
    return actions_L


def _decode_pred_text_B_from_generated_ids(
    generated_ids_BS: torch.Tensor,
    prompt_lens_B: list[int],
    tokenizer: Any,
) -> list[str]:
    pred_ids_B: list[list[int]] = []
    for b_i, prompt_len_i in enumerate(prompt_lens_B):
        row_S = generated_ids_BS[b_i]
        start_i = max(int(prompt_len_i), 0)
        if start_i >= int(row_S.shape[0]):
            pred_ids_B.append([])
            continue
        pred_ids_B.append([int(x) for x in row_S[start_i:].detach().cpu().tolist()])
    return [
        str(x) for x in tokenizer.batch_decode(pred_ids_B, skip_special_tokens=True)
    ]


def _truncate_for_log(text_s: str, max_chars: int = 1200) -> str:
    text_s = str(text_s).strip()
    if len(text_s) <= max_chars:
        return text_s
    return f"{text_s[:max_chars]} ...[truncated]"


# ---------------------------------------------------------------------------
# Action accuracy / confusion — mouse-only (no keyboard)
# ---------------------------------------------------------------------------

def _action_confusion_count_key(target_class_s: str, pred_class_s: str) -> str:
    return f"confusion_{target_class_s}_as_{pred_class_s}_n"


def _action_accuracy_counts_from_texts(
    pred_text_B: list[str],
    target_text_B: list[str],
    action_is_counted_fn: Callable[[str], bool] | None = None,
    class_counts_out_d: dict[str, int] | None = None,
    confusion_counts_out_d: dict[str, int] | None = None,
) -> tuple[int, int]:
    correct_n = 0
    total_n = 0
    if action_is_counted_fn is None:
        action_is_counted_fn = lambda _: True
    if class_counts_out_d is not None:
        for cls_s in _ACTION_CLASS_NAMES:
            class_counts_out_d.setdefault(f"{cls_s}_correct_n", 0)
            class_counts_out_d.setdefault(f"{cls_s}_total_n", 0)
    if confusion_counts_out_d is not None:
        for t_s in _ACTION_CLASS_NAMES:
            for p_s in _ACTION_CONFUSION_PRED_CLASS_NAMES:
                confusion_counts_out_d.setdefault(
                    _action_confusion_count_key(t_s, p_s), 0
                )
    for pred_s, target_s in zip(pred_text_B, target_text_B):
        pred_actions_L = _actions_from_target_text(pred_s)
        target_actions_L = _actions_from_target_text(target_s)
        for idx_i, target_action_s in enumerate(target_actions_L):
            if not action_is_counted_fn(target_action_s):
                continue
            target_cls = _action_class_s(target_action_s)
            has_pred = idx_i < len(pred_actions_L)
            pred_action_s = pred_actions_L[idx_i] if has_pred else ""
            pred_cls = _action_class_s(pred_action_s) if has_pred else "missing"
            total_n += 1
            if class_counts_out_d is not None:
                class_counts_out_d[f"{target_cls}_total_n"] += 1
            if confusion_counts_out_d is not None:
                confusion_counts_out_d[
                    _action_confusion_count_key(target_cls, pred_cls)
                ] += 1
            if has_pred and pred_action_s == target_action_s:
                correct_n += 1
                if class_counts_out_d is not None:
                    class_counts_out_d[f"{target_cls}_correct_n"] += 1
    return correct_n, total_n


def _action_type_counts_from_texts(
    action_text_B: list[str],
) -> tuple[int, int, int]:
    no_op_n = 0
    mouse_n = 0
    total_n = 0
    for text_s in action_text_B:
        for action_s in _actions_from_target_text(text_s):
            cls = _action_class_s(action_s)
            total_n += 1
            if cls == "no_op":
                no_op_n += 1
            elif cls == "mouse":
                mouse_n += 1
    return no_op_n, mouse_n, total_n


def _action_confusion_matrix_counts(
    stats_d: dict[str, int],
) -> list[list[int]]:
    counts: list[list[int]] = []
    for t_s in _ACTION_CLASS_NAMES:
        row: list[int] = []
        for p_s in _ACTION_CONFUSION_PRED_CLASS_NAMES:
            row.append(int(stats_d.get(_action_confusion_count_key(t_s, p_s), 0)))
        counts.append(row)
    return counts


def _action_f1_from_counts(
    correct_n_f: float,
    pred_total_n_f: float,
    target_total_n_f: float,
) -> float:
    precision_f = float(correct_n_f) / max(float(pred_total_n_f), 1.0)
    recall_f = float(correct_n_f) / max(float(target_total_n_f), 1.0)
    denom_f = precision_f + recall_f
    if denom_f <= 0.0:
        return 0.0
    return 2.0 * precision_f * recall_f / denom_f


def _wandb_confusion_chart(counts_NM: list[list[int]]) -> Any | None:
    y_true: list[int] = []
    preds: list[int] = []
    for t_i in range(len(_ACTION_CLASS_NAMES)):
        for p_i in range(len(_ACTION_CONFUSION_PRED_CLASS_NAMES)):
            c = int(counts_NM[t_i][p_i])
            if c > 0:
                y_true.extend([t_i] * c)
                preds.extend([p_i] * c)
    if not y_true:
        return None
    return wandb.plot.confusion_matrix(
        y_true=y_true,
        preds=preds,
        class_names=list(_ACTION_CONFUSION_PRED_CLASS_NAMES),
        title="Val Mouse Action Confusion Matrix",
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _weighted_causal_lm_loss(
    logits_BSV: torch.Tensor,
    labels_BS: torch.Tensor,
    label_weights_BS: torch.Tensor | None = None,
) -> torch.Tensor:
    shift_logits = logits_BSV[:, :-1, :].contiguous()
    shift_labels = labels_BS[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        input=shift_logits.view(-1, shift_logits.shape[-1]),
        target=shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    valid_mask = shift_labels != -100
    if label_weights_BS is None:
        denom = torch.clamp(valid_mask.sum(), min=1).to(token_loss.dtype)
        return (token_loss * valid_mask.to(token_loss.dtype)).sum() / denom
    shift_weights = label_weights_BS[:, 1:].to(token_loss.dtype).contiguous()
    valid_weights = shift_weights * valid_mask.to(token_loss.dtype)
    denom = torch.clamp(valid_weights.sum(), min=1.0)
    return (token_loss * valid_weights).sum() / denom


# ---------------------------------------------------------------------------
# Validation (with optional visual logging)
# ---------------------------------------------------------------------------

def _run_validation_steps(
    ddp_model: torch.nn.Module,
    collator: VideoSFTCollator,
    val_it: Any,
    val_steps: int,
    val_generate_max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    debug_examples_n: int = 0,
    debug_examples_out_L: list[tuple[str, str]] | None = None,
    action_stats_out_d: dict[str, int] | None = None,
    visual_samples_out_L: list[dict[str, Any]] | None = None,
    visual_max_frames: int = 16,
) -> tuple[float, int, int, int]:
    """Run val steps.  When `visual_samples_out_L` is not None, the first
    batch's raw data (jpeg_frames, gt actions, pred actions) is saved for
    cursor-overlay rendering on rank 0."""
    ddp_model.eval()
    val_loss_num = 0.0
    val_tok_n = 0
    val_correct_n = 0
    val_total_n = 0
    val_pred_no_op_n = 0
    val_pred_mouse_n = 0
    val_pred_total_n = 0
    val_target_no_op_n = 0
    val_target_mouse_n = 0
    val_target_total_n = 0
    class_counts_d: dict[str, int] = {}
    confusion_d: dict[str, int] = {}

    with torch.no_grad():
        for step_idx in range(val_steps):
            raw_batch = next(val_it)
            collated = collator(raw_batch)
            label_weights = collated.get("label_weights")
            model_batch = _to_device(collated, device)
            batch_tok = int((model_batch["labels"] != -100).sum().item())
            val_tok_n += batch_tok
            if label_weights is not None:
                label_weights = label_weights.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=(device.type == "cuda")):
                outputs = ddp_model(**model_batch)
                loss = _weighted_causal_lm_loss(
                    outputs.logits, model_batch["labels"], label_weights,
                )
            val_loss_num += float(loss.detach().item()) * float(batch_tok)

            # Generate predictions
            prompt_batch = collator.prompt_model_inputs(raw_batch)
            prompt_lens = [int(x) for x in prompt_batch.pop("prompt_lens")]
            prompt_model = _to_device(prompt_batch, device, skip_keys={"videos", "meta"})
            gen_model = getattr(ddp_model, "module", ddp_model)
            pad_id = getattr(collator.tokenizer, "pad_token_id", None)
            eos_id = getattr(collator.tokenizer, "eos_token_id", None)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": int(val_generate_max_new_tokens),
                "do_sample": False,
                "use_cache": True,
            }
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = int(pad_id)
            if eos_id is not None:
                gen_kwargs["eos_token_id"] = int(eos_id)
            with torch.autocast(device.type, dtype=dtype, enabled=(device.type == "cuda")):
                gen_ids = gen_model.generate(**prompt_model, **gen_kwargs)
            pred_text_B = _decode_pred_text_B_from_generated_ids(
                gen_ids, prompt_lens, collator.tokenizer,
            )
            target_text_B = [str(x) for x in raw_batch["target_text"]]

            # ── Collect visual sample from 1st batch, 1st example ────
            if (
                visual_samples_out_L is not None
                and step_idx == 0
                and len(visual_samples_out_L) == 0
            ):
                # Encode frames back to JPEG bytes from the numpy array
                frames_np = raw_batch["frames"]  # (B, T, H, W, C) numpy
                if frames_np is not None and len(frames_np) > 0:
                    sample_frames = frames_np[0][:visual_max_frames]  # (T, H, W, C)
                    jpeg_bytes = []
                    for f in sample_frames:
                        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                        if ok:
                            jpeg_bytes.append(enc.tobytes())
                    if jpeg_bytes:
                        visual_samples_out_L.append({
                            "jpeg_frames": jpeg_bytes,
                            "gt_text": target_text_B[0],
                            "pred_text": pred_text_B[0],
                        })

            if debug_examples_out_L is not None and debug_examples_n > 0:
                remaining = max(debug_examples_n - len(debug_examples_out_L), 0)
                if remaining > 0:
                    for p, t in zip(pred_text_B, target_text_B):
                        debug_examples_out_L.append((str(p), str(t)))
                        remaining -= 1
                        if remaining <= 0:
                            break

            c, t = _action_accuracy_counts_from_texts(
                pred_text_B, target_text_B,
                class_counts_out_d=class_counts_d,
                confusion_counts_out_d=confusion_d,
            )
            pno, pmo, pto = _action_type_counts_from_texts(pred_text_B)
            tno, tmo, tto = _action_type_counts_from_texts(target_text_B)
            val_correct_n += c
            val_total_n += t
            val_pred_no_op_n += pno
            val_pred_mouse_n += pmo
            val_pred_total_n += pto
            val_target_no_op_n += tno
            val_target_mouse_n += tmo
            val_target_total_n += tto

    ddp_model.train()
    val_loss_f = val_loss_num / max(float(val_tok_n), 1.0)
    if action_stats_out_d is not None:
        action_stats_out_d["pred_no_op_n"] = val_pred_no_op_n
        action_stats_out_d["pred_mouse_n"] = val_pred_mouse_n
        action_stats_out_d["pred_action_total_n"] = val_pred_total_n
        action_stats_out_d["target_no_op_n"] = val_target_no_op_n
        action_stats_out_d["target_mouse_n"] = val_target_mouse_n
        action_stats_out_d["target_action_total_n"] = val_target_total_n
        for cls_s in _ACTION_CLASS_NAMES:
            action_stats_out_d[f"class_{cls_s}_correct_n"] = class_counts_d.get(
                f"{cls_s}_correct_n", 0
            )
            action_stats_out_d[f"class_{cls_s}_total_n"] = class_counts_d.get(
                f"{cls_s}_total_n", 0
            )
        for t_s in _ACTION_CLASS_NAMES:
            for p_s in _ACTION_CONFUSION_PRED_CLASS_NAMES:
                k = _action_confusion_count_key(t_s, p_s)
                action_stats_out_d[k] = confusion_d.get(k, 0)
    return val_loss_f, val_tok_n, val_correct_n, val_total_n


# ---------------------------------------------------------------------------
# Model / MFU helpers
# ---------------------------------------------------------------------------

def _resolve_resume_dir(args: Args) -> str:
    if not args.resume_from:
        return ""
    if args.resume_from == "latest":
        return find_latest_checkpoint(args.out_dir) or ""
    return args.resume_from


def _load_metadata_json(data_root: str) -> dict[str, Any]:
    meta_path = os.path.join(data_root, "metadata.json")
    if not os.path.exists(meta_path):
        raise ValueError(f"metadata.json not found in {data_root}.")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _assert_image_hwc_matches_metadata(args: Args) -> None:
    meta_d = _load_metadata_json(args.data_root)
    meta_h = int(meta_d["target_height"])
    meta_w = int(meta_d["target_width"])
    meta_c = int(meta_d.get("target_channels", 3))
    if (args.image_h, args.image_w, args.image_c) != (meta_h, meta_w, meta_c):
        raise ValueError(
            f"Image shape mismatch: metadata={meta_h}x{meta_w}x{meta_c}, "
            f"args={args.image_h}x{args.image_w}x{args.image_c}"
        )


def _build_model(
    args: Args, dtype: torch.dtype, device: torch.device
) -> torch.nn.Module:
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, **model_kwargs,
    )
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    if args.use_lora:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "up_proj", "down_proj", "gate_proj",
                ],
            ),
        )
    return model.to(device)


def _transformer_dims_for_mfu(model: torch.nn.Module) -> tuple[int, int, int]:
    cfg = model.config.text_config
    return int(cfg.num_hidden_layers), int(cfg.num_attention_heads), int(cfg.head_dim)


def _peak_device_flops(precision_s: str, device_name_s: str) -> float | None:
    if precision_s.lower() not in {"bf16", "fp16"}:
        return None
    name = device_name_s.lower()
    if "h100" in name:
        return 756e12 if "pcie" in name else 989e12
    if "a100" in name:
        return 312e12
    return None


def _mfu_from_throughput(
    n_params: int, n_layers: int, n_heads: int, head_dim: int,
    seq_len_f: float, tokens_per_s_f: float, peak_flops_f: float,
) -> float | None:
    if peak_flops_f <= 0 or tokens_per_s_f <= 0:
        return None
    flops_per_tok = (
        6.0 * float(n_params)
        + 12.0 * float(n_layers * n_heads * head_dim) * seq_len_f
    )
    return (flops_per_tok * tokens_per_s_f) / peak_flops_f


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = tyro.cli(Args)
    assert args.data_root, "--data_root is required."

    rank_i = int(os.environ.get("RANK", 0))
    world_i = int(os.environ.get("WORLD_SIZE", 1))
    local_rank_i = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.device_count() == 0:
        raise RuntimeError("No CUDA devices found.")
    if args.global_batch_size % world_i != 0:
        raise ValueError(
            f"global_batch_size ({args.global_batch_size}) must divide "
            f"world_size ({world_i})."
        )

    print(f"Rank {rank_i} initializing process group...")
    if rank_i == 0:
        print(
            f"World size: {world_i}\n"
            f"CUDA device: {torch.cuda.get_device_name(local_rank_i)}"
        )

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank_i)
    device = torch.device(f"cuda:{local_rank_i}")
    _seed_all(args.seed, rank_i)
    _assert_image_hwc_matches_metadata(args)

    array_paths = find_array_record_paths(args.data_root, "train")
    val_array_paths: list[str] = []
    if args.val_every > 0:
        val_array_paths = find_array_record_paths(args.data_root, "val")

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = _build_model(args, dtype, device)
    train_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_n = sum(p.numel() for p in model.parameters())
    mfu_nl, mfu_nh, mfu_hd = _transformer_dims_for_mfu(model)
    mfu_peak = (
        float(args.mfu_peak_flops) if args.mfu_peak_flops > 0.0
        else (_peak_device_flops(args.precision, torch.cuda.get_device_name(local_rank_i)) or 0.0)
    )
    mfu_enabled = mfu_peak > 0.0 and total_n > 0

    if rank_i == 0:
        print(
            f"trainable={train_n} total={total_n} "
            f"ratio={train_n / max(total_n, 1):.6f}"
        )
        print(f"loss_weights no_op={args.no_op_loss_weight:.4f} mouse={args.mouse_loss_weight:.4f}")
        if args.val_visual_every > 0:
            print(
                f"Visual cursor logging enabled every {args.val_visual_every} steps "
                f"(max_frames={args.val_visual_max_frames}, upscale={args.val_visual_upscale}x)"
            )

    ddp_model = DDP(
        model, device_ids=[local_rank_i], output_device=local_rank_i,
        find_unused_parameters=False,
    )
    collator = VideoSFTCollator(
        processor=processor,
        instruction_text=args.instruction_text,
        video_fps=args.video_fps,
        no_op_loss_weight=args.no_op_loss_weight,
        mouse_loss_weight=args.mouse_loss_weight,
    )
    val_collator = collator
    val_it = None
    if args.val_every > 0:
        val_processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        val_collator = VideoSFTCollator(
            processor=val_processor,
            instruction_text=args.instruction_text,
            video_fps=args.video_fps,
            no_op_loss_weight=args.no_op_loss_weight,
            mouse_loss_weight=args.mouse_loss_weight,
        )
        val_loader = get_dataloader(
            array_record_paths=val_array_paths,
            seq_len=args.seq_len,
            global_batch_size=args.global_batch_size,
            image_h=args.image_h, image_w=args.image_w, image_c=args.image_c,
            rank=rank_i, world_size=world_i, seed=args.seed, epoch_i=0,
            num_epochs=None,
            num_workers=args.num_workers,
            prefetch_buffer_size=args.prefetch_buffer_size,
            read_num_threads=args.read_num_threads,
            worker_buffer_size=args.worker_buffer_size,
            min_action_density=0.0,
        )
        val_it = iter(val_loader)

    trainable_params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler(enabled=args.precision == "fp16")
    os.makedirs(args.out_dir, exist_ok=True)

    wandb_run = None
    if rank_i == 0 and args.wandb_enable:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            mode=args.wandb_mode,
            dir=args.out_dir,
            config=asdict(args),
        )

    global_step = 0
    epoch_i = 0
    resume_epoch_i = -1
    pending_state_b: bytes | None = None
    resume_dir = _resolve_resume_dir(args)
    if resume_dir:
        payload_d = load_checkpoint(
            ckpt_dir=resume_dir, model=ddp_model.module,
            use_lora=args.use_lora, optimizer=optimizer,
            scheduler=scheduler, rank_i=rank_i,
        )
        if args.precision == "fp16" and payload_d.get("scaler_state_d") is not None:
            scaler.load_state_dict(payload_d["scaler_state_d"])
        state_d = payload_d["train_state_d"]
        global_step = int(state_d["global_step"])
        epoch_i = int(state_d["epoch_i"])
        resume_epoch_i = epoch_i
        pending_state_b = payload_d.get("grain_state_b")
        if state_d.get("rng_state_d") is not None:
            _set_rng_state(state_d["rng_state_d"])
        if rank_i == 0:
            print(f"Resumed from {resume_dir} at step={global_step}, epoch={epoch_i}")

    optimizer.zero_grad(set_to_none=True)
    micro_in_accum = 0
    log_loss_sum = 0.0
    log_micro_n = 0
    log_tok_n = 0
    log_input_tok_n = 0
    log_sample_n = 0
    log_step_n = 0
    log_grad_norm_sum = 0.0
    log_grad_norm_n = 0
    t0 = time.time()

    while global_step < args.max_steps:
        loader = get_dataloader(
            array_record_paths=array_paths,
            seq_len=args.seq_len,
            global_batch_size=args.global_batch_size,
            image_h=args.image_h, image_w=args.image_w, image_c=args.image_c,
            rank=rank_i, world_size=world_i, seed=args.seed, epoch_i=epoch_i,
            num_workers=args.num_workers,
            prefetch_buffer_size=args.prefetch_buffer_size,
            read_num_threads=args.read_num_threads,
            worker_buffer_size=args.worker_buffer_size,
            min_action_density=args.train_min_action_density,
        )
        raw_it = iter(loader)
        if pending_state_b is not None and epoch_i == resume_epoch_i:
            raw_it.set_state(pending_state_b)
            pending_state_b = None

        prefetched_it = (
            CollatorPrefetchIterator(raw_it=raw_it, collator=collator)
            if args.collator_prefetch else None
        )
        batch_it = prefetched_it if prefetched_it is not None else raw_it
        epoch_step_start = global_step

        try:
            for batch_d in batch_it:
                ddp_model.train()
                collated = batch_d if prefetched_it is not None else collator(batch_d)
                label_weights = collated.get("label_weights")
                model_batch = _to_device(collated, device)
                if label_weights is not None:
                    label_weights = label_weights.to(device, non_blocking=True)
                tok_n = int((model_batch["labels"] != -100).sum().item())

                with torch.autocast("cuda", dtype=dtype):
                    outputs = ddp_model(**model_batch)
                    loss = _weighted_causal_lm_loss(
                        outputs.logits, model_batch["labels"], label_weights,
                    )

                log_loss_sum += float(loss.detach().item())
                log_micro_n += 1
                log_tok_n += tok_n
                log_input_tok_n += int(model_batch["input_ids"].numel())
                log_sample_n += int(model_batch["input_ids"].shape[0])

                micro_in_accum += 1
                loss = loss / args.grad_accum
                should_step = micro_in_accum >= args.grad_accum
                if not should_step and world_i > 1:
                    with ddp_model.no_sync():
                        if args.precision == "fp16":
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                else:
                    if args.precision == "fp16":
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                if not should_step:
                    continue

                micro_in_accum = 0
                if args.precision == "fp16":
                    scaler.unscale_(optimizer)
                grad_norm = _grad_norm_and_clip(trainable_params, args.max_grad_norm)
                if args.precision == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                log_step_n += 1
                log_grad_norm_sum += grad_norm
                log_grad_norm_n += 1

                # ── Logging ──────────────────────────────────────────
                if global_step % args.log_every == 0:
                    mean_loss_t = torch.tensor(
                        log_loss_sum / max(log_micro_n, 1), device=device
                    )
                    mean_gn_t = torch.tensor(
                        log_grad_norm_sum / max(log_grad_norm_n, 1), device=device
                    )
                    tok_t = torch.tensor(float(log_tok_n), device=device)
                    itok_t = torch.tensor(float(log_input_tok_n), device=device)
                    samp_t = torch.tensor(float(log_sample_n), device=device)
                    if world_i > 1:
                        for t in (mean_loss_t, mean_gn_t, tok_t, itok_t, samp_t):
                            dist.all_reduce(t, op=dist.ReduceOp.SUM)
                        mean_loss_t /= world_i
                        mean_gn_t /= world_i

                    dt = max(time.time() - t0, 1e-9)
                    steps_s = log_step_n / dt
                    toks_s = tok_t.item() / dt
                    itoks_s = itok_t.item() / dt
                    lr_f = optimizer.param_groups[0]["lr"]
                    mean_seq = itok_t.item() / max(samp_t.item(), 1.0)
                    mfu_f = (
                        _mfu_from_throughput(
                            total_n, mfu_nl, mfu_nh, mfu_hd,
                            mean_seq, itoks_s, mfu_peak,
                        ) if mfu_enabled else None
                    )
                    if rank_i == 0:
                        mfu_s = f" mfu={mfu_f:.4f}" if mfu_f is not None else ""
                        print(
                            f"step={global_step} loss={mean_loss_t.item():.6f} "
                            f"grad_norm={mean_gn_t.item():.6f} lr={lr_f:.3e} "
                            f"steps/s={steps_s:.3f} tok/s={toks_s:.1f}{mfu_s}"
                        )
                        if wandb_run is not None:
                            log_d: dict[str, Any] = {
                                "train/loss": mean_loss_t.item(),
                                "train/grad_norm": mean_gn_t.item(),
                                "train/lr": lr_f,
                                "train/steps_per_s": steps_s,
                                "train/tokens_per_s": toks_s,
                                "train/epoch_estimate": epoch_i,
                            }
                            if mfu_f is not None:
                                log_d["train/mfu"] = mfu_f
                            wandb_run.log(log_d, step=global_step)
                    log_loss_sum = 0.0
                    log_micro_n = 0
                    log_tok_n = 0
                    log_input_tok_n = 0
                    log_sample_n = 0
                    log_step_n = 0
                    log_grad_norm_sum = 0.0
                    log_grad_norm_n = 0
                    t0 = time.time()

                # ── Validation ───────────────────────────────────────
                if args.val_every > 0 and global_step % args.val_every == 0:
                    assert val_it is not None
                    val_t0 = time.time()
                    val_examples: list[tuple[str, str]] = []
                    val_stats: dict[str, int] = {}

                    # Decide whether to collect visual samples this step
                    do_visual = (
                        args.val_visual_every > 0
                        and global_step % args.val_visual_every == 0
                        and rank_i == 0
                        and wandb_run is not None
                    )
                    visual_samples: list[dict[str, Any]] = [] if do_visual else None

                    vl, vtok, vc, vt = _run_validation_steps(
                        ddp_model, val_collator, val_it,
                        args.val_steps, args.val_generate_max_new_tokens,
                        device, dtype,
                        debug_examples_n=args.val_log_examples,
                        debug_examples_out_L=val_examples,
                        action_stats_out_d=val_stats,
                        visual_samples_out_L=visual_samples,
                        visual_max_frames=args.val_visual_max_frames,
                    )
                    # All-reduce val metrics
                    vl_num_t = torch.tensor(vl * max(float(vtok), 1.0), device=device)
                    vtok_t = torch.tensor(float(vtok), device=device)
                    vc_t = torch.tensor(float(vc), device=device)
                    vt_t = torch.tensor(float(vt), device=device)
                    vpno_t = torch.tensor(float(val_stats.get("pred_no_op_n", 0)), device=device)
                    vpmo_t = torch.tensor(float(val_stats.get("pred_mouse_n", 0)), device=device)
                    vpto_t = torch.tensor(float(val_stats.get("pred_action_total_n", 0)), device=device)
                    vtno_t = torch.tensor(float(val_stats.get("target_no_op_n", 0)), device=device)
                    vtmo_t = torch.tensor(float(val_stats.get("target_mouse_n", 0)), device=device)
                    vtto_t = torch.tensor(float(val_stats.get("target_action_total_n", 0)), device=device)
                    cnoc_t = torch.tensor(float(val_stats.get("class_no_op_correct_n", 0)), device=device)
                    cnot_t = torch.tensor(float(val_stats.get("class_no_op_total_n", 0)), device=device)
                    cmoc_t = torch.tensor(float(val_stats.get("class_mouse_correct_n", 0)), device=device)
                    cmot_t = torch.tensor(float(val_stats.get("class_mouse_total_n", 0)), device=device)
                    conf_t = torch.tensor(
                        _action_confusion_matrix_counts(val_stats),
                        device=device, dtype=torch.float32,
                    )
                    if world_i > 1:
                        for t in (vl_num_t, vtok_t, vc_t, vt_t, vpno_t, vpmo_t,
                                  vpto_t, vtno_t, vtmo_t, vtto_t, cnoc_t, cnot_t,
                                  cmoc_t, cmot_t, conf_t):
                            dist.all_reduce(t, op=dist.ReduceOp.SUM)

                    val_loss_f = (vl_num_t / torch.clamp(vtok_t, min=1.0)).item()
                    val_dt = max(time.time() - val_t0, 1e-9)
                    val_tps = vtok_t.item() / val_dt
                    val_acc = vc_t.item() / max(vt_t.item(), 1.0)
                    val_f1 = _action_f1_from_counts(vc_t.item(), vpto_t.item(), vtto_t.item())
                    pnor = vpno_t.item() / max(vpto_t.item(), 1.0)
                    tnor = vtno_t.item() / max(vtto_t.item(), 1.0)
                    pmor = vpmo_t.item() / max(vpto_t.item(), 1.0)
                    tmor = vtmo_t.item() / max(vtto_t.item(), 1.0)
                    acc_no = cnoc_t.item() / max(cnot_t.item(), 1.0)
                    acc_mo = cmoc_t.item() / max(cmot_t.item(), 1.0)
                    conf_NM = [[int(x) for x in row] for row in conf_t.detach().cpu().tolist()]

                    if rank_i == 0:
                        print(
                            f"step={global_step} val_loss={val_loss_f:.6f} "
                            f"val_acc={val_acc:.6f} val_f1={val_f1:.6f} "
                            f"acc_no_op={acc_no:.4f} acc_mouse={acc_mo:.4f} "
                            f"pred_no_op_rate={pnor:.4f} target_no_op_rate={tnor:.4f} "
                            f"pred_mouse_rate={pmor:.4f} target_mouse_rate={tmor:.4f}"
                        )
                        if val_examples:
                            for ex_i, (p, t) in enumerate(val_examples, 1):
                                print(f"[val {ex_i}] pred:\n{p}")
                                print(f"[val {ex_i}] target:\n{_truncate_for_log(t)}")
                        if wandb_run is not None:
                            wlog: dict[str, Any] = {
                                "val/loss": val_loss_f,
                                "val/tokens_per_s": val_tps,
                                "val_action/acc": val_acc,
                                "val_action/f1": val_f1,
                                "val_action/pred_no_op_rate": pnor,
                                "val_action/target_no_op_rate": tnor,
                                "val_action/pred_mouse_rate": pmor,
                                "val_action/target_mouse_rate": tmor,
                                "val_action/acc_no_op": acc_no,
                                "val_action/acc_mouse": acc_mo,
                                "val_action/total_no_op": cnot_t.item(),
                                "val_action/total_mouse": cmot_t.item(),
                                "val_action/pred_total": vpto_t.item(),
                                "val_action/target_total": vtto_t.item(),
                            }
                            chart = _wandb_confusion_chart(conf_NM)
                            if chart is not None:
                                wlog["val_action/confusion_matrix"] = chart

                            # ── Visual cursor overlay ────────────────
                            if visual_samples and len(visual_samples) > 0:
                                sample = visual_samples[0]
                                gt_actions = _actions_from_target_text(sample["gt_text"])
                                pred_actions = _actions_from_target_text(sample["pred_text"])
                                jpeg_frames = sample["jpeg_frames"]
                                if gt_actions and jpeg_frames:
                                    rendered = _render_cursor_frames(
                                        jpeg_frames=jpeg_frames,
                                        gt_actions=gt_actions,
                                        pred_actions=pred_actions,
                                        img_w=args.image_w,
                                        img_h=args.image_h,
                                        max_frames=args.val_visual_max_frames,
                                        upscale=args.val_visual_upscale,
                                    )
                                    if rendered:
                                        grid = _make_grid(rendered, cols=min(8, len(rendered)))
                                        wlog["val_visual/cursor_overlay"] = wandb.Image(
                                            grid,
                                            caption=(
                                                f"step={global_step} | "
                                                f"green=GT  red=pred | "
                                                f"{len(rendered)} frames"
                                            ),
                                        )
                                        # Also log individual frames
                                        frame_images = [
                                            wandb.Image(
                                                img,
                                                caption=f"frame {i} gt={gt_actions[i] if i < len(gt_actions) else 'N/A'} pred={pred_actions[i] if i < len(pred_actions) else 'N/A'}",
                                            )
                                            for i, img in enumerate(rendered[:8])
                                        ]
                                        wlog["val_visual/cursor_frames"] = frame_images

                            wandb_run.log(wlog, step=global_step)

                # ── Save ─────────────────────────────────────────────
                if global_step % args.save_every == 0 or global_step == args.max_steps:
                    save_checkpoint(
                        out_dir=args.out_dir,
                        step_i=global_step,
                        model=ddp_model.module,
                        use_lora=args.use_lora,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler_state=(
                            scaler.state_dict() if args.precision == "fp16" else None
                        ),
                        train_state_d={
                            "global_step": global_step,
                            "epoch_i": epoch_i,
                            "rng_state_d": _rng_state_d(),
                        },
                        grain_state_b=(
                            prefetched_it.last_state()
                            if prefetched_it is not None
                            else raw_it.get_state()
                        ),
                        args_d=asdict(args),
                        rank_i=rank_i,
                        save_model=(rank_i == 0),
                    )
                    if world_i > 1:
                        dist.barrier()

                if global_step >= args.max_steps:
                    break
        finally:
            if prefetched_it is not None:
                prefetched_it.close()

        if global_step == epoch_step_start:
            print(f"[rank {rank_i}] epoch={epoch_i} zero steps. "
                f"array_paths={len(array_paths)} seq_len={args.seq_len} "
                f"batch_size={args.global_batch_size} world={world_i}")
            raise RuntimeError(
                "No training steps in epoch; lower --train-min-action-density."
            )
        epoch_i += 1

    if rank_i == 0:
        print(f"Training complete. global_step={global_step}")
        if wandb_run is not None:
            wandb_run.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
