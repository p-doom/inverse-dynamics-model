from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
from peft import LoraConfig, get_peft_model
import torch
from transformers import Qwen3VLForConditionalGeneration

from mouse_args import Args
from utils.checkpoint import find_latest_checkpoint
from utils.lr_schedules import LRScheduleArgs, lr_at_step


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
# EMA model helper
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Apply EMA weights and return backup of original weights."""
        backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])


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
# Curriculum helper
# ---------------------------------------------------------------------------

def _current_min_action_density(args: Args, global_step: int) -> float:
    """Compute the current min_action_density, optionally ramped."""
    target = args.train_min_action_density
    if target <= 0.0:
        return 0.0
    ramp = args.train_min_action_density_ramp_steps
    if ramp <= 0 or global_step >= ramp:
        return target
    return target * (float(global_step) / float(ramp))
