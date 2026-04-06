from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
import random
import time
from typing import Any

import numpy as np
from peft import LoraConfig, get_peft_model
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import tyro
import wandb

from idm.utils.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from idm.utils.collator import ChatSFTCollator
from idm.utils.data import (
    find_jsonl_path,
    get_chat_dataloader,
    load_chat_dataset,
)
from idm.utils.lr_schedules import LRScheduleArgs, lr_at_step

"""
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    idm/train.py \
      --data-root /path/to/chat_jsonl_dir \
      --model-id /path/to/Qwen3-VL-2B-Instruct \
      --attn-implementation sdpa
"""


@dataclass
class Args:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    attn_implementation: str = "flash_attention_2"
    data_root: str = ""
    max_turns: int = 32
    global_batch_size: int = 8
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    max_steps: int = 1000
    lr: float = 2e-5
    init_lr: float = 0.0
    decay_end: float = 0.0
    lr_schedule: str = "wsd"
    warmup_steps: int = 100
    wsd_decay_steps: int = 200
    weight_decay: float = 0.0
    precision: str = "bf16"
    master_weights_fp32: bool = False
    grad_checkpointing: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 0
    num_workers: int = 4
    log_every: int = 10
    val_every: int = 0
    val_steps: int = 1
    save_every: int = 100
    out_dir: str = "./runs/default"
    resume_from: str = ""
    wandb_enable: bool = True
    wandb_tags: list[str] = field(default_factory=list)
    wandb_project: str = "idm"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_mode: str = "offline"
    mfu_peak_flops: float = 0.0


# ── RNG helpers ──────────────────────────────────────────────────────


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


# ── Scheduler / grad helpers ────────────────────────────────────────


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


# ── Device transfer ─────────────────────────────────────────────────


def _to_device(
    batch_d: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Move every :class:`torch.Tensor` in *batch_d* to *device*."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch_d.items()
    }


# ── Model construction ──────────────────────────────────────────────


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
        args.model_id,
        **model_kwargs,
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


# ── MFU estimation ──────────────────────────────────────────────────


def _transformer_dims_for_mfu(model: torch.nn.Module) -> tuple[int, int, int]:
    text_cfg = model.config.text_config
    n_layers_i = int(text_cfg.num_hidden_layers)
    n_heads_i = int(text_cfg.num_attention_heads)
    head_dim_i = int(text_cfg.head_dim)
    if n_layers_i <= 0 or n_heads_i <= 0 or head_dim_i <= 0:
        raise ValueError("Invalid Qwen3-VL text_config dimensions for MFU.")
    return n_layers_i, n_heads_i, head_dim_i


def _peak_device_flops(precision_s: str, device_name_s: str) -> float | None:
    precision_s = str(precision_s).lower()
    if precision_s not in {"bf16", "fp16"}:
        return None
    name_s = str(device_name_s).lower()
    if "h100" in name_s:
        return 756e12 if "pcie" in name_s else 989e12
    if "a100" in name_s:
        return 312e12
    return None


def _flops_per_token_estimate(
    n_params_i: int,
    n_layers_i: int,
    n_heads_i: int,
    head_dim_i: int,
    seq_len_f: float,
) -> float:
    return float(
        (6.0 * float(n_params_i))
        + (12.0 * float(n_layers_i * n_heads_i * head_dim_i) * float(seq_len_f))
    )


def _mfu_from_throughput(
    n_params_i: int,
    n_layers_i: int,
    n_heads_i: int,
    head_dim_i: int,
    seq_len_f: float,
    tokens_per_s_f: float,
    peak_flops_f: float,
) -> float | None:
    if peak_flops_f <= 0 or tokens_per_s_f <= 0:
        return None
    flops_per_token_f = _flops_per_token_estimate(
        n_params_i=n_params_i,
        n_layers_i=n_layers_i,
        n_heads_i=n_heads_i,
        head_dim_i=head_dim_i,
        seq_len_f=seq_len_f,
    )
    return (flops_per_token_f * float(tokens_per_s_f)) / float(peak_flops_f)


# ── Validation ──────────────────────────────────────────────────────


def _run_validation_steps(
    ddp_model: torch.nn.Module,
    val_loader: Any,
    val_steps: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    use_amp: bool,
) -> tuple[float, int]:
    """Run *val_steps* forward passes and return ``(mean_loss, total_tokens)``."""
    ddp_model.eval()
    val_loss_sum = 0.0
    val_tok_n = 0
    val_it = iter(val_loader)

    with torch.no_grad():
        for _ in range(val_steps):
            try:
                batch_d = next(val_it)
            except StopIteration:
                break
            model_batch_d = _to_device(batch_d, device)
            with torch.autocast(
                device_type=device.type,
                dtype=compute_dtype,
                enabled=use_amp,
            ):
                outputs = ddp_model(**model_batch_d)
                loss = outputs.loss
            batch_tok_n = int((model_batch_d["labels"] != -100).sum().item())
            val_loss_sum += float(loss.detach().item()) * batch_tok_n
            val_tok_n += batch_tok_n

    ddp_model.train()
    return val_loss_sum / max(float(val_tok_n), 1.0), val_tok_n


# ── Epoch-sync helper ───────────────────────────────────────────────


def _next_synced_batch(
    batch_it: Any,
    *,
    world_i: int,
    device: torch.device,
) -> tuple[dict[str, Any] | None, bool]:
    try:
        batch_d = next(batch_it)
        local_has_batch_i = 1
    except StopIteration:
        batch_d = None
        local_has_batch_i = 0

    if world_i <= 1:
        return batch_d, (local_has_batch_i == 0)

    has_batch_t = torch.tensor(float(local_has_batch_i), device=device)
    dist.all_reduce(has_batch_t, op=dist.ReduceOp.MIN)
    all_have_batch_b = bool(has_batch_t.item() >= 0.5)
    if not all_have_batch_b:
        return None, True
    return batch_d, False


def _resolve_resume_dir(args: Args) -> str:
    if not args.resume_from:
        return ""
    if args.resume_from == "latest":
        return find_latest_checkpoint(args.out_dir) or ""
    return args.resume_from


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    args = tyro.cli(Args)
    assert args.data_root, "--data-root is required."

    # ── distributed setup ──
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
            f"global_batch_size ({args.global_batch_size}) must divide "
            f"world_size ({world_i})."
        )
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be >= 1.")
    if args.attn_implementation not in {"flash_attention_2", "sdpa", "auto"}:
        raise ValueError(
            "Unsupported --attn-implementation. "
            "Expected one of: flash_attention_2, sdpa, auto."
        )

    print(f"Rank {rank_i} initializing process group...")
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

    # ── data ──
    per_device_batch = args.global_batch_size // world_i
    train_jsonl = find_jsonl_path(args.data_root, "train")
    train_dataset = load_chat_dataset(train_jsonl, max_turns=args.max_turns)
    if rank_i == 0:
        print(f"train_dataset: {len(train_dataset)} conversations")

    val_dataset = None
    val_loader = None
    val_jsonl_path = os.path.join(args.data_root, "val.jsonl")
    if args.val_every > 0 and os.path.isfile(val_jsonl_path):
        val_dataset = load_chat_dataset(val_jsonl_path, max_turns=args.max_turns)
        if rank_i == 0:
            print(f"val_dataset: {len(val_dataset)} conversations")

    # ── model ──
    if args.precision == "bf16":
        compute_dtype = torch.bfloat16
    elif args.precision == "fp16":
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    param_dtype = (
        torch.float32
        if args.master_weights_fp32 or args.precision == "fp32"
        else compute_dtype
    )
    use_amp = compute_dtype != torch.float32

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = _build_model(args, param_dtype, device)
    train_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_n = sum(p.numel() for p in model.parameters())
    mfu_n_layers_i, mfu_n_heads_i, mfu_head_dim_i = _transformer_dims_for_mfu(model)
    mfu_peak_flops_f = (
        float(args.mfu_peak_flops)
        if args.mfu_peak_flops > 0.0
        else (
            _peak_device_flops(args.precision, torch.cuda.get_device_name(local_rank_i))
            or 0.0
        )
    )
    mfu_enabled_b = bool(mfu_peak_flops_f > 0.0 and total_n > 0)
    if rank_i == 0:
        print(
            f"trainable_params={train_n} total_params={total_n} "
            f"ratio={train_n / max(total_n, 1):.6f}"
        )
        if mfu_enabled_b:
            print(
                f"mfu_enabled=True peak_flops={mfu_peak_flops_f:.3e} "
                f"n_layers={mfu_n_layers_i} n_heads={mfu_n_heads_i} "
                f"head_dim={mfu_head_dim_i}"
            )
        else:
            print("mfu_enabled=False (set --mfu-peak-flops to enable)")

    ddp_model = DDP(
        model,
        device_ids=[local_rank_i],
        output_device=local_rank_i,
        find_unused_parameters=False,
    )

    # ── collator ──
    collator = ChatSFTCollator(processor=processor)

    # ── validation loader (created once, re-iterated each validation) ──
    if val_dataset is not None:
        val_processor = AutoProcessor.from_pretrained(
            args.model_id, trust_remote_code=True
        )
        val_collator = ChatSFTCollator(processor=val_processor)
        val_loader = get_chat_dataloader(
            dataset=val_dataset,
            collate_fn=val_collator,
            batch_size=per_device_batch,
            num_workers=args.num_workers,
            shuffle=False,
            rank=rank_i,
            world_size=world_i,
            seed=args.seed,
            drop_last=True,
        )

    # ── train loader ──
    train_loader = get_chat_dataloader(
        dataset=train_dataset,
        collate_fn=collator,
        batch_size=per_device_batch,
        num_workers=args.num_workers,
        shuffle=True,
        rank=rank_i,
        world_size=world_i,
        seed=args.seed,
        drop_last=True,
    )

    # ── optimizer / scheduler / scaler ──
    trainable_params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
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
            tags=args.wandb_tags,
            dir=args.out_dir,
            config=asdict(args),
        )

    # ── resume ──
    global_step = 0
    epoch_i = 0
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
        if state_d.get("rng_state_d") is not None:
            _set_rng_state(state_d["rng_state_d"])
        if rank_i == 0:
            print(
                f"Resumed from {resume_dir} at "
                f"global_step={global_step}, epoch_i={epoch_i}"
            )

    # ── training loop ──
    optimizer.zero_grad(set_to_none=True)
    micro_in_accum_i = 0
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
        train_sampler = train_loader.sampler
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch_i)

        batch_it = iter(train_loader)
        epoch_step_start = global_step

        while True:
            batch_d, should_stop_b = _next_synced_batch(
                batch_it=batch_it,
                world_i=world_i,
                device=device,
            )
            if should_stop_b:
                break
            assert batch_d is not None

            ddp_model.train()
            model_batch_d = _to_device(batch_d, device)
            tok_n = int((model_batch_d["labels"] != -100).sum().item())

            with torch.autocast("cuda", dtype=compute_dtype, enabled=use_amp):
                outputs = ddp_model(**model_batch_d)
                loss = outputs.loss

            log_loss_sum += float(loss.detach().item())
            log_micro_n += 1
            log_tok_n += tok_n
            log_input_tok_n += int(model_batch_d["input_ids"].numel())
            log_sample_n += int(model_batch_d["input_ids"].shape[0])

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
                scaler.unscale_(optimizer)
            grad_norm_f = _grad_norm_and_clip(trainable_params, args.max_grad_norm)
            if args.precision == "fp16":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            log_step_n += 1
            log_grad_norm_sum += grad_norm_f
            log_grad_norm_n += 1

            # ── train logging ──
            if global_step % args.log_every == 0:
                mean_loss_t = torch.tensor(
                    log_loss_sum / max(log_micro_n, 1), device=device
                )
                mean_grad_norm_t = torch.tensor(
                    log_grad_norm_sum / max(log_grad_norm_n, 1), device=device
                )
                tok_t = torch.tensor(float(log_tok_n), device=device)
                input_tok_t = torch.tensor(float(log_input_tok_n), device=device)
                sample_t = torch.tensor(float(log_sample_n), device=device)
                if world_i > 1:
                    dist.all_reduce(mean_loss_t, op=dist.ReduceOp.SUM)
                    mean_loss_t /= world_i
                    dist.all_reduce(mean_grad_norm_t, op=dist.ReduceOp.SUM)
                    mean_grad_norm_t /= world_i
                    dist.all_reduce(tok_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(input_tok_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(sample_t, op=dist.ReduceOp.SUM)

                dt = max(time.time() - t0, 1e-9)
                steps_per_s = log_step_n / dt
                toks_per_s = tok_t.item() / dt
                input_toks_per_s = input_tok_t.item() / dt
                lr_f = optimizer.param_groups[0]["lr"]
                mean_input_seq_len_f = input_tok_t.item() / max(
                    sample_t.item(), 1.0
                )

                mfu_f = None
                if mfu_enabled_b:
                    mfu_f = _mfu_from_throughput(
                        n_params_i=total_n,
                        n_layers_i=mfu_n_layers_i,
                        n_heads_i=mfu_n_heads_i,
                        head_dim_i=mfu_head_dim_i,
                        seq_len_f=mean_input_seq_len_f,
                        tokens_per_s_f=input_toks_per_s,
                        peak_flops_f=mfu_peak_flops_f,
                    )

                if rank_i == 0:
                    mfu_s = f" mfu={mfu_f:.4f}" if mfu_f is not None else ""
                    print(
                        f"step={global_step} "
                        f"loss={mean_loss_t.item():.6f} "
                        f"grad_norm={mean_grad_norm_t.item():.6f} "
                        f"lr={lr_f:.3e} "
                        f"steps/s={steps_per_s:.3f} "
                        f"tok/s={toks_per_s:.1f}{mfu_s}"
                    )
                    if wandb_run is not None:
                        log_d = {
                            "train/loss": mean_loss_t.item(),
                            "train/grad_norm": mean_grad_norm_t.item(),
                            "train/lr": lr_f,
                            "train/steps_per_s": steps_per_s,
                            "train/tokens_per_s": toks_per_s,
                            "train/supervised_tokens": tok_t.item() / max(log_step_n, 1),
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

            # ── validation ──
            if (
                args.val_every > 0
                and global_step % args.val_every == 0
                and val_loader is not None
            ):
                val_t0 = time.time()
                val_loss_f, val_tok_n = _run_validation_steps(
                    ddp_model=ddp_model,
                    val_loader=val_loader,
                    val_steps=args.val_steps,
                    device=device,
                    compute_dtype=compute_dtype,
                    use_amp=use_amp,
                )
                val_loss_num_t = torch.tensor(
                    val_loss_f * max(float(val_tok_n), 1.0), device=device
                )
                val_tok_t = torch.tensor(float(val_tok_n), device=device)
                if world_i > 1:
                    dist.all_reduce(val_loss_num_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_tok_t, op=dist.ReduceOp.SUM)
                val_loss_t = val_loss_num_t / torch.clamp(val_tok_t, min=1.0)
                val_dt = max(time.time() - val_t0, 1e-9)
                val_toks_per_s = val_tok_t.item() / val_dt

                if rank_i == 0:
                    print(
                        f"step={global_step} "
                        f"val_loss={val_loss_t.item():.6f} "
                        f"val_tok/s={val_toks_per_s:.1f}"
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "val/loss": val_loss_t.item(),
                                "val/tokens_per_s": val_toks_per_s,
                            },
                            step=global_step,
                        )

            # ── checkpoint ──
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
                    grain_state_b=None,
                    args_d=asdict(args),
                    rank_i=rank_i,
                    save_model=(rank_i == 0),
                )
                if world_i > 1:
                    dist.barrier()

            if global_step >= args.max_steps:
                break

        if global_step == epoch_step_start:
            raise RuntimeError(
                "No training steps completed in this epoch; check data."
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
