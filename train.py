from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import random
import time
from typing import Any, Callable

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
    video_fps: float = 30.0
    seq_len: int = 128
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
    val_every: int = 0
    val_steps: int = 1
    val_generate_max_new_tokens: int = 512
    val_log_examples: int = 0
    save_every: int = 100
    out_dir: str = "./runs/default"
    resume_from: str = ""
    instruction_text: str = (
        "Given the video frames, output the action text for each frame in order."
    )
    mask_no_op_actions: bool = False
    mask_mouse_actions: bool = False
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


def _to_device(
    batch_d: dict[str, Any],
    device: torch.device,
    skip_keys: set[str] | None = None,
) -> dict[str, Any]:
    skip_keys = (
        {"videos", "prompt_lens", "meta"} if skip_keys is None else set(skip_keys)
    )
    out_d: dict[str, Any] = {}
    for key_s, val in batch_d.items():
        if key_s in skip_keys:
            continue
        out_d[key_s] = val.to(device, non_blocking=True)
    return out_d


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


def _action_accuracy_counts_from_texts(
    pred_text_B: list[str],
    target_text_B: list[str],
    action_is_counted_fn: Callable[[str], bool] | None = None,
) -> tuple[int, int]:
    correct_n = 0
    total_n = 0
    if action_is_counted_fn is None:
        action_is_counted_fn = lambda _: True
    for pred_s, target_s in zip(pred_text_B, target_text_B):
        pred_actions_L = _actions_from_target_text(pred_s)
        target_actions_L = _actions_from_target_text(target_s)
        for idx_i, target_action_s in enumerate(target_actions_L):
            if not action_is_counted_fn(target_action_s):
                continue
            total_n += 1
            if idx_i < len(pred_actions_L) and pred_actions_L[idx_i] == target_action_s:
                correct_n += 1
    return correct_n, total_n


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
) -> tuple[float, int, int, int]:
    ddp_model.eval()
    val_loss_num = 0.0
    val_tok_n = 0
    val_action_correct_n = 0
    val_action_total_n = 0
    action_is_counted_fn: Callable[[str], bool] | None = None
    mask_no_op_actions = bool(getattr(collator, "mask_no_op_actions", False))
    mask_mouse_actions = bool(getattr(collator, "mask_mouse_actions", False))
    if mask_no_op_actions or mask_mouse_actions:
        should_mask_action_fn = getattr(collator, "_should_mask_action", None)
        if callable(should_mask_action_fn):
            action_is_counted_fn = lambda action_s: not bool(
                should_mask_action_fn(action_s)
            )
        else:
            # Keep validation action-accuracy masking aligned with label masking.
            def _fallback_action_is_counted(action_s: str) -> bool:
                action_s = action_s.strip()
                if mask_no_op_actions and action_s == "NO_OP":
                    return False
                if mask_mouse_actions and "MOUSE_" in action_s:
                    return False
                return True

            action_is_counted_fn = _fallback_action_is_counted

    with torch.no_grad():
        for _ in range(val_steps):
            raw_batch_d = next(val_it)
            model_batch_d = _to_device(collator(raw_batch_d), device)
            batch_tok_n = int((model_batch_d["labels"] != -100).sum().item())
            val_tok_n += batch_tok_n
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=(device.type == "cuda"),
            ):
                outputs = ddp_model(**model_batch_d)
                loss = outputs.loss
            val_loss_num += float(loss.detach().item()) * float(batch_tok_n)
            prompt_batch_d = collator.prompt_model_inputs(raw_batch_d)
            prompt_lens_B = [int(x) for x in prompt_batch_d.pop("prompt_lens")]
            prompt_model_d = _to_device(
                prompt_batch_d,
                device,
                skip_keys={"videos", "meta"},
            )
            generate_model = getattr(ddp_model, "module", ddp_model)
            pad_id = getattr(collator.tokenizer, "pad_token_id", None)
            eos_id = getattr(collator.tokenizer, "eos_token_id", None)
            gen_kwargs = {
                "max_new_tokens": int(val_generate_max_new_tokens),
                "do_sample": False,
                "use_cache": True,
            }
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = int(pad_id)
            if eos_id is not None:
                gen_kwargs["eos_token_id"] = int(eos_id)

            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=(device.type == "cuda"),
            ):
                generated_ids_BS = generate_model.generate(
                    **prompt_model_d,
                    **gen_kwargs,
                )
            pred_text_B = _decode_pred_text_B_from_generated_ids(
                generated_ids_BS=generated_ids_BS,
                prompt_lens_B=prompt_lens_B,
                tokenizer=collator.tokenizer,
            )
            target_text_L = [str(x) for x in raw_batch_d["target_text"]]
            if debug_examples_out_L is not None and debug_examples_n > 0:
                remaining_i = max(int(debug_examples_n) - len(debug_examples_out_L), 0)
                if remaining_i > 0:
                    for pred_s, target_s in zip(pred_text_B, target_text_L):
                        debug_examples_out_L.append((str(pred_s), str(target_s)))
                        remaining_i -= 1
                        if remaining_i <= 0:
                            break
            correct_i, total_i = _action_accuracy_counts_from_texts(
                pred_text_B=pred_text_B,
                target_text_B=target_text_L,
                action_is_counted_fn=action_is_counted_fn,
            )
            val_action_correct_n += correct_i
            val_action_total_n += total_i

    ddp_model.train()
    val_loss_f = val_loss_num / max(float(val_tok_n), 1.0)
    return (
        val_loss_f,
        val_tok_n,
        val_action_correct_n,
        val_action_total_n,
    )


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
        return json.load(f)


def _assert_image_hwc_matches_metadata(args: Args) -> None:
    meta_d = _load_metadata_json(args.data_root)
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
    if args.val_every < 0:
        raise ValueError("--val_every must be >= 0.")
    if args.val_steps <= 0:
        raise ValueError("--val_steps must be >= 1.")
    if args.val_generate_max_new_tokens <= 0:
        raise ValueError("--val_generate_max_new_tokens must be >= 1.")
    if args.val_log_examples < 0:
        raise ValueError("--val_log_examples must be >= 0.")
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

    val_array_paths: list[str] = []
    if args.val_every > 0:
        val_array_paths = find_array_record_paths(args.data_root, "val")

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = _build_model(args, dtype, device)
    if rank_i == 0:
        train_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_n = sum(p.numel() for p in model.parameters())
        print(
            f"trainable_params={train_n} total_params={total_n} ratio={train_n/max(total_n,1):.6f}"
        )
        print(f"model_id={args.model_id}")

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
        mask_no_op_actions=args.mask_no_op_actions,
        mask_mouse_actions=args.mask_mouse_actions,
    )
    val_it = None
    if args.val_every > 0:
        val_loader = get_dataloader(
            array_record_paths=val_array_paths,
            seq_len=args.seq_len,
            global_batch_size=args.global_batch_size,
            image_h=args.image_h,
            image_w=args.image_w,
            image_c=args.image_c,
            rank=rank_i,
            world_size=world_i,
            seed=args.seed,
            epoch_i=0,
            num_epochs=None,
            num_workers=args.num_workers,
            prefetch_buffer_size=args.prefetch_buffer_size,
        )
        val_it = iter(val_loader)
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

            if args.val_every > 0 and global_step % args.val_every == 0:
                assert val_it is not None
                val_t0 = time.time()
                val_examples_L: list[tuple[str, str]] = []
                (
                    val_loss_f,
                    val_tok_n,
                    val_action_correct_n,
                    val_action_total_n,
                ) = _run_validation_steps(
                    ddp_model=ddp_model,
                    collator=collator,
                    val_it=val_it,
                    val_steps=args.val_steps,
                    val_generate_max_new_tokens=args.val_generate_max_new_tokens,
                    device=device,
                    dtype=dtype,
                    debug_examples_n=args.val_log_examples,
                    debug_examples_out_L=val_examples_L,
                )
                val_loss_num_t = torch.tensor(
                    val_loss_f * max(float(val_tok_n), 1.0), device=device
                )
                val_tok_t = torch.tensor(float(val_tok_n), device=device)
                val_action_correct_t = torch.tensor(
                    float(val_action_correct_n), device=device
                )
                val_action_total_t = torch.tensor(
                    float(val_action_total_n), device=device
                )
                if world_i > 1:
                    dist.all_reduce(val_loss_num_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_tok_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_action_correct_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_action_total_t, op=dist.ReduceOp.SUM)
                val_loss_t = val_loss_num_t / torch.clamp(val_tok_t, min=1.0)
                val_dt = max(time.time() - val_t0, 1e-9)
                val_toks_per_s = val_tok_t.item() / val_dt
                val_action_acc_f = val_action_correct_t.item() / max(
                    val_action_total_t.item(), 1.0
                )
                if rank_i == 0:
                    print(
                        f"step={global_step} val_loss={val_loss_t.item():.6f} "
                        f"val_steps={args.val_steps} val_tokens_per_s={val_toks_per_s:.1f} "
                        f"val_action_acc={val_action_acc_f:.6f}"
                    )
                    if val_examples_L:
                        for ex_i, (pred_s, target_s) in enumerate(
                            val_examples_L, start=1
                        ):
                            pred_actions_L = _actions_from_target_text(pred_s)
                            target_actions_L = _actions_from_target_text(target_s)
                            print(
                                f"[val_example {ex_i}] pred_actions={len(pred_actions_L)} "
                                f"target_actions={len(target_actions_L)}"
                            )
                            print(
                                f"[val_example {ex_i}] pred_raw:\n{_truncate_for_log(pred_s)}"
                            )
                            print(
                                f"[val_example {ex_i}] target_raw:\n{_truncate_for_log(target_s)}"
                            )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "val/loss": val_loss_t.item(),
                                "val/tokens_per_s": val_toks_per_s,
                                "val/action_acc": val_action_acc_f,
                            },
                            step=global_step,
                        )

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
