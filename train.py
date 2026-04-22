#!/usr/bin/env python3
"""SFT training for the IDM sparse-event format.

Trains Qwen3-VL (or compatible VLMs) to predict user input actions from
sequences of screenshot frames, using the same JSON format as the eval pipeline.
"""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor
import tyro
import wandb

from data import (
    ClipDataset,
    ProcessedClipDataset,
    build_prompt,
    build_sft_messages,
    collate_processed,
    normalize_actions,
)
from eval import (
    compute_f1,
    compute_prf,
    discover_eval_clips,
    parse_response,
    run_real_eval,
)


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
    """WSD schedule: warmup -> stable -> decay."""
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
# Validation (on training val split)
# ---------------------------------------------------------------------------


def run_validation(
    model: torch.nn.Module,
    processor: Any,
    val_dataset: ClipDataset,
    args: "Args",
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
        "image"  # "image" = frames as separate images; "video" = single video
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
        ""  # Path to eval clips (mp4+json pairs). If set, runs real eval.
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

    # Reduce CUDA fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    # Selective gradient checkpointing for LoRA at low resolution:
    # Disable on even-numbered LLM layers + ViT blocks to save ~50% recomputation.
    # Only safe when max_pixels <= 200704 (51 GB peak). At 518k+ pixels, full
    # checkpointing is needed to avoid OOM (75-77 GB without it).
    if args.use_lora and args.max_pixels <= 200704:
        for i, layer in enumerate(model.model.language_model.layers):
            if i % 2 == 0:
                layer.gradient_checkpointing = False
        for blk in model.model.visual.blocks:
            blk.gradient_checkpointing = False

    # Workaround: cuDNN 9.1 has a pathological bf16 Conv3d kernel for the ViT's patch
    # embedding shape (large batch, tiny spatial). torch.compile bypasses it (3.6s -> 22ms).
    model.model.visual.patch_embed = torch.compile(model.model.visual.patch_embed)

    if args.use_lora:
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
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    else:
        ddp_model = model
    raw_model = ddp_model.module if hasattr(ddp_model, "module") else ddp_model

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        min_pixels=3136,
        max_pixels=args.max_pixels,
    )
    if args.video_mode == "video" and hasattr(processor, "video_processor"):
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

    # Read FPS from training data for eval (ensures train/eval FPS match)
    with open(train_jsonl) as f:
        train_fps = json.loads(f.readline()).get("fps", 5)

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

    # Optimizer — separate param group for ViT with lower LR in full-FT mode
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
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=args.weight_decay, fused=True
        )
    else:
        optimizer = torch.optim.AdamW(
            [p for p in ddp_model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=True,
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_at_step(
            step, args.warmup_steps, args.max_steps, args.wsd_decay_steps, args.lr
        )
        / max(args.lr, 1e-12),
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Wandb
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
    data_iter = iter(train_loader)

    if rank == 0:
        print(
            f"Starting training: {args.max_steps} steps, batch={args.batch_size}, "
            f"grad_accum={args.grad_accum}"
        )

    # no_sync context: skip DDP all-reduce during gradient accumulation steps
    _no_sync = ddp_model.no_sync if world_size > 1 else nullcontext

    while global_step < args.max_steps:
        ddp_model.train()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

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

        # Only all-reduce on the last micro-step of each grad accumulation
        is_last_accum = (micro_step + 1) >= args.grad_accum
        sync_ctx = nullcontext() if is_last_accum else _no_sync()
        with sync_ctx:
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
            mfu = (6 * total_n * tps / peak_flops) if peak_flops > 0 else 0.0
            mfu_str = f" mfu={mfu:.4f}" if peak_flops > 0 else ""
            print(
                f"step={global_step} loss={avg_loss:.4f} lr={lr_val:.2e} "
                f"tok/s={tps:.0f} samples/s={samples_per_s:.2f}{mfu_str}"
            )
            peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
            if wandb_run:
                log_d = {
                    "step": global_step,
                    "train/loss": avg_loss,
                    "train/lr": lr_val,
                    "train/tok_per_s": tps,
                    "train/samples_per_s": samples_per_s,
                    "train/peak_mem_gb": peak_mem_gb,
                }
                if peak_flops > 0:
                    log_d["train/mfu"] = mfu
                wandb_run.log(log_d)
            log_loss_sum = 0.0
            log_loss_n = 0
            log_tok_n = 0
            t0 = time.time()

        # Validation + eval + save
        is_val_step = args.val_every > 0 and global_step % args.val_every == 0
        is_save_step = args.save_every > 0 and global_step % args.save_every == 0
        if (is_val_step or is_save_step) and world_size > 1:
            dist.barrier()

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
            my_clips = eval_clips[rank::world_size] if world_size > 1 else eval_clips
            local_metrics = run_real_eval(
                raw_model,
                processor,
                my_clips,
                fps=train_fps,
                tolerance=args.eval_tolerance,
                coalesce=args.eval_coalesce,
                interleave_labels=args.interleave_labels,
                device=device,
                dtype=dtype,
            )

            # All-reduce TP/FP/FN across ranks
            if world_size > 1:
                counts = torch.tensor(
                    [local_metrics["tp"], local_metrics["fp"], local_metrics["fn"]],
                    device=device,
                    dtype=torch.long,
                )
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                tp, fp, fn = counts.tolist()
            else:
                tp = local_metrics["tp"]
                fp = local_metrics["fp"]
                fn = local_metrics["fn"]

            if rank == 0:
                p, r_val, f1 = compute_prf(tp, fp, fn)
                print(
                    f"  EVAL: f1={f1:.3f} p={p:.3f} r={r_val:.3f} "
                    f"(TP={tp} FP={fp} FN={fn})"
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

        # Wait for eval to finish before all ranks proceed
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


if __name__ == "__main__":
    main()
