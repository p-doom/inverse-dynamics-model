from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import random
import time
from typing import Any

import numpy as np
from peft import LoraConfig, get_peft_model
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import tyro
import wandb

from idm.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from idm.collator import VideoSFTCollator
from idm.data import (
    count_valid_records,
    find_array_record_paths,
    get_dataloader,
)
from idm.lr_schedules import LRScheduleArgs, lr_at_step


@dataclass
class Args:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    data_root: str = ""
    image_h: int = 90
    image_w: int = 160
    image_c: int = 3
    video_fps: float = 10.0
    seq_len: int = 32
    global_batch_size: int = 8
    grad_accum: int = 1
    max_steps: int = 1000
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
    num_workers: int = 0
    prefetch_buffer_size: int = 1
    log_every: int = 10
    save_every: int = 100
    out_dir: str = "./runs/default"
    resume_from: str = ""
    instruction_text: str = (
        "Given the video frames, output the action text for each frame in order."
    )
    wandb_enable: bool = True
    wandb_project: str = "idm"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_mode: str = "online"


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


def _trainable_count(model: torch.nn.Module) -> tuple[int, int]:
    train_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_n = sum(p.numel() for p in model.parameters())
    return train_n, total_n


def _to_device(batch_d: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out_d: dict[str, Any] = {}
    for key_s, val in batch_d.items():
        if key_s in {"videos", "prompt_lens", "meta"}:
            continue
        out_d[key_s] = val.to(device, non_blocking=True) if hasattr(val, "to") else val
    return out_d


def _resolve_resume_dir(args: Args) -> str:
    if not args.resume_from:
        return ""
    if args.resume_from == "latest":
        return find_latest_checkpoint(args.out_dir) or ""
    return args.resume_from


def _load_metadata_json(data_root: str) -> dict[str, Any]:
    meta_path = os.path.join(data_root, "metadata.json")
    if not os.path.exists(meta_path):
        raise ValueError(
            f"metadata.json not found in {data_root}. "
            "Run preprocessing first and keep metadata.json in --data-root."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_d = json.load(f)
    if not isinstance(meta_d, dict):
        raise ValueError(f"metadata.json must contain a JSON object: {meta_path}")
    return meta_d


def _assert_image_hwc_matches_metadata(args: Args) -> None:
    meta_d = _load_metadata_json(args.data_root)
    missing_L = [k for k in ("target_height", "target_width") if meta_d.get(k) is None]
    if missing_L:
        raise ValueError(
            f"metadata.json is missing required keys: {', '.join(missing_L)}. "
            "Expected target_height/target_width and optional target_channels."
        )

    meta_h = int(meta_d["target_height"])
    meta_w = int(meta_d["target_width"])
    meta_c = int(meta_d.get("target_channels", 3))
    if (args.image_h, args.image_w, args.image_c) != (meta_h, meta_w, meta_c):
        raise ValueError(
            "Image shape mismatch with metadata.json. "
            f"metadata HxWxC={meta_h}x{meta_w}x{meta_c}, "
            f"args HxWxC={args.image_h}x{args.image_w}x{args.image_c}. "
            "Pass matching --image-h/--image-w/--image-c."
        )


def _build_model(
    args: Args, dtype: torch.dtype, device: torch.device
) -> torch.nn.Module:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
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
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "up_proj",
                    "down_proj",
                    "gate_proj",
                ],
            ),
        )
    return model.to(device)


def main() -> None:
    args = tyro.cli(Args)
    assert args.data_root, "--data_root is required."

    rank_i = int(os.environ.get("RANK", 0))
    world_i = int(os.environ.get("WORLD_SIZE", 1))
    local_rank_i = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.device_count() == 0:
        raise RuntimeError("No CUDA devices found.")
    if world_i <= 0 or rank_i < 0 or rank_i >= world_i:
        raise ValueError(
            f"Invalid distributed env rank={rank_i}, world_size={world_i}."
        )
    if args.global_batch_size % world_i != 0:
        raise ValueError(
            f"global_batch_size ({args.global_batch_size}) must divide world_size ({world_i})."
        )
    if args.grad_accum <= 0:
        raise ValueError("--grad_accum must be >= 1.")
    if args.log_every <= 0:
        raise ValueError("--log_every must be >= 1.")
    if args.save_every <= 0:
        raise ValueError("--save_every must be >= 1.")
    if args.video_fps <= 0:
        raise ValueError("--video_fps must be > 0.")

    print(f"Rank {rank_i} initializing process group...")
    print(f"Local rank: {local_rank_i}")
    if rank_i == 0:
        print(
            f"World size: {world_i}\n"
            f"CUDA devices: {torch.cuda.device_count()}\n"
            f"CUDA device: {torch.cuda.get_device_name(local_rank_i)}"
        )

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank_i)
    device = torch.device(f"cuda:{local_rank_i}")
    _seed_all(args.seed, rank_i)
    _assert_image_hwc_matches_metadata(args)

    array_paths = find_array_record_paths(args.data_root, "train")
    valid_n = count_valid_records(
        array_paths, args.seq_len, args.image_h, args.image_w, args.image_c
    )
    if valid_n <= 0:
        raise ValueError(
            "No valid records after checks. "
            "This usually means your explicit --image-h/--image-w/--image-c or --seq-len "
            "does not match the preprocessed data, or records are missing in-record `actions`."
        )

    per_rank_bs = args.global_batch_size // world_i
    micro_per_epoch = (valid_n // world_i) // per_rank_bs
    opt_per_epoch = max(micro_per_epoch // max(args.grad_accum, 1), 1)
    if rank_i == 0:
        print(
            f"valid_records={valid_n} optimizer_steps_per_epoch~{opt_per_epoch} "
            f"estimated_epochs~{args.max_steps / opt_per_epoch:.3f}"
        )

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = _build_model(args, dtype, device)
    if rank_i == 0:
        train_n, total_n = _trainable_count(model)
        print(
            f"trainable_params={train_n} total_params={total_n} ratio={train_n/max(total_n,1):.6f}"
        )

    ddp_model = DDP(
        model,
        device_ids=[local_rank_i],
        output_device=local_rank_i,
        find_unused_parameters=False,
    )
    collator = VideoSFTCollator(
        processor=processor,
        instruction_text=args.instruction_text,
        video_fps=args.video_fps,
    )
    optimizer = torch.optim.AdamW(
        (p for p in ddp_model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
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
            ckpt_dir=resume_dir,
            model=ddp_model.module,
            use_lora=args.use_lora,
            optimizer=optimizer,
            scheduler=scheduler,
            rank_i=rank_i,
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
            print(
                f"Resumed from {resume_dir} at global_step={global_step}, epoch_i={epoch_i}"
            )

    optimizer.zero_grad(set_to_none=True)
    micro_in_accum_i = 0
    log_loss_sum = 0.0
    log_micro_n = 0
    log_tok_n = 0
    log_step_n = 0
    t0 = time.time()

    while global_step < args.max_steps:
        # Rebuild each epoch for deterministic seed+epoch reshuffle.
        loader = get_dataloader(
            array_record_paths=array_paths,
            seq_len=args.seq_len,
            global_batch_size=args.global_batch_size,
            image_h=args.image_h,
            image_w=args.image_w,
            image_c=args.image_c,
            rank=rank_i,
            world_size=world_i,
            seed=args.seed,
            epoch_i=epoch_i,
            num_workers=args.num_workers,
            prefetch_buffer_size=args.prefetch_buffer_size,
        )
        it = iter(loader)
        if pending_state_b is not None and epoch_i == resume_epoch_i:
            it.set_state(pending_state_b)
            pending_state_b = None

        for raw_batch_d in it:
            ddp_model.train()
            model_batch_d = _to_device(collator(raw_batch_d), device)
            tok_n = int((model_batch_d["labels"] != -100).sum().item())

            with torch.autocast("cuda", dtype=dtype):
                loss = ddp_model(**model_batch_d).loss

            log_loss_sum += float(loss.detach().item())
            log_micro_n += 1
            log_tok_n += tok_n

            micro_in_accum_i += 1
            loss = loss / args.grad_accum
            should_step_b = micro_in_accum_i >= args.grad_accum
            if not should_step_b and world_i > 1:
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
            if not should_step_b:
                continue

            micro_in_accum_i = 0
            if args.precision == "fp16":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            log_step_n += 1

            if global_step % args.log_every == 0:
                mean_loss_t = torch.tensor(
                    log_loss_sum / max(log_micro_n, 1), device=device
                )
                tok_t = torch.tensor(float(log_tok_n), device=device)
                if world_i > 1:
                    dist.all_reduce(mean_loss_t, op=dist.ReduceOp.SUM)
                    mean_loss_t /= world_i
                    dist.all_reduce(tok_t, op=dist.ReduceOp.SUM)

                dt = max(time.time() - t0, 1e-9)
                steps_per_s = log_step_n / dt
                toks_per_s = tok_t.item() / dt
                lr_f = optimizer.param_groups[0]["lr"]
                if rank_i == 0:
                    print(
                        f"step={global_step} loss={mean_loss_t.item():.6f} "
                        f"lr={lr_f:.3e} steps_per_s={steps_per_s:.3f} "
                        f"tokens_per_s={toks_per_s:.1f}"
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": mean_loss_t.item(),
                                "train/lr": lr_f,
                                "train/steps_per_s": steps_per_s,
                                "train/tokens_per_s": toks_per_s,
                                "train/epoch_estimate": epoch_i,
                            },
                            step=global_step,
                        )
                log_loss_sum = 0.0
                log_micro_n = 0
                log_tok_n = 0
                log_step_n = 0
                t0 = time.time()

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
                    grain_state_b=it.get_state(),
                    args_d=asdict(args),
                    rank_i=rank_i,
                    save_model=(rank_i == 0),
                )
                if world_i > 1:
                    dist.barrier()

            if global_step >= args.max_steps:
                break
        epoch_i += 1

    if rank_i == 0:
        print(f"Training complete. global_step={global_step}")
        if wandb_run is not None:
            wandb_run.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
