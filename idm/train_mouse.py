from __future__ import annotations

from dataclasses import asdict
import os
import time
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor
import tyro
import wandb

from mouse_actions import _actions_from_pred_text, _actions_from_target_text, _truncate_for_log
from mouse_args import Args
from mouse_loss import (
    _build_vocab_int_values_tensor,
    _mouse_soft_label_loss,
    _weighted_causal_lm_loss,
)
from mouse_metrics import (
    _action_confusion_matrix_counts,
    _action_f1_from_counts,
    _wandb_confusion_chart,
)
from mouse_train_utils import (
    _assert_image_hwc_matches_metadata,
    _build_model,
    _build_scheduler,
    _current_min_action_density,
    _grad_norm_and_clip,
    _mfu_from_throughput,
    _peak_device_flops,
    _resolve_resume_dir,
    _rng_state_d,
    _seed_all,
    _set_rng_state,
    _to_device,
    _transformer_dims_for_mfu,
    EMAModel,
)
from mouse_validation import _run_validation_steps
from mouse_viz import (
    _make_grid,
    _render_cursor_frames,
)
from utils.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from utils.collator import CollatorPrefetchIterator, VideoSFTCollator
from utils.data_jpeg import (
    find_array_record_paths,
    get_dataloader,
)


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
        if args.label_smoothing > 0:
            print(f"label_smoothing={args.label_smoothing}")
        if args.focal_loss_gamma > 0:
            print(f"focal_loss_gamma={args.focal_loss_gamma}")
        if args.ema_decay > 0:
            print(f"EMA decay={args.ema_decay}")
        if args.diversity_penalty > 0:
            print(f"diversity_penalty={args.diversity_penalty}")
        if args.val_temperature != 1.0:
            print(f"val_temperature={args.val_temperature}")
        if args.val_visual_every > 0:
            print(
                f"Visual cursor logging enabled every {args.val_visual_every} steps "
                f"(max_frames={args.val_visual_max_frames}, upscale={args.val_visual_upscale}x)"
            )

    ddp_model = DDP(
        model, device_ids=[local_rank_i], output_device=local_rank_i,
        find_unused_parameters=False,
    )

    # EMA
    ema: EMAModel | None = None
    if args.ema_decay > 0:
        ema = EMAModel(model, decay=args.ema_decay)
        if rank_i == 0:
            print(f"EMA model initialized with decay={args.ema_decay}")

    # Build vocab integer-value lookup (needed for soft-label loss)
    vocab_int_values_V: torch.Tensor | None = None
    if args.mouse_soft_label_sigma > 0.0 and args.mouse_soft_label_weight > 0.0:
        vocab_size = processor.tokenizer.vocab_size
        vocab_int_values_V = _build_vocab_int_values_tensor(processor.tokenizer, vocab_size)
        n_int_toks = int((~torch.isnan(vocab_int_values_V)).sum().item())
        if rank_i == 0:
            print(
                f"soft_label_sigma={args.mouse_soft_label_sigma} "
                f"soft_label_weight={args.mouse_soft_label_weight} "
                f"integer_vocab_tokens={n_int_toks}"
            )

    collator = VideoSFTCollator(
        processor=processor,
        instruction_text=args.instruction_text,
        video_fps=args.video_fps,
        no_op_loss_weight=args.no_op_loss_weight,
        mouse_loss_weight=args.mouse_loss_weight,
        format_loss_weight=args.format_loss_weight,
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
            format_loss_weight=args.format_loss_weight,
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
            noop_format=args.noop_format,
            skip_noop_frames=args.skip_noop_frames,
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

    # Track best val metrics for logging
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_mouse_cos = -1.0

    while global_step < args.max_steps:
        # Compute current min_action_density (possibly ramped)
        cur_min_density = _current_min_action_density(args, global_step)

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
            min_action_density=cur_min_density,
            noop_format=args.noop_format,
            skip_noop_frames=args.skip_noop_frames,
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
                    if "mm_token_type_ids" in model_batch:
                        has_image_grid = (
                            "image_grid_thw" in model_batch
                            and model_batch["image_grid_thw"] is not None
                        )
                        if not has_image_grid:
                            del model_batch["mm_token_type_ids"]

                    outputs = ddp_model(**model_batch)
                    loss = _weighted_causal_lm_loss(
                        outputs.logits, model_batch["labels"], label_weights,
                        label_smoothing=args.label_smoothing,
                        focal_gamma=args.focal_loss_gamma,
                        class_balanced=args.class_balanced_loss,
                    )
                    if (
                        vocab_int_values_V is not None
                        and args.mouse_soft_label_weight > 0.0
                        and label_weights is not None
                    ):
                        soft_loss = _mouse_soft_label_loss(
                            outputs.logits, model_batch["labels"], label_weights,
                            vocab_int_values_V,
                            mouse_loss_weight=args.mouse_loss_weight,
                            sigma=args.mouse_soft_label_sigma,
                        )
                        loss = loss + args.mouse_soft_label_weight * soft_loss

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
                if ema is not None:
                    ema.update(ddp_model.module)
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

                    do_visual = (
                        args.val_visual_every > 0
                        and global_step % args.val_visual_every == 0
                        and rank_i == 0
                        and wandb_run is not None
                    )
                    visual_samples: list[dict[str, Any]] = [] if do_visual else None

                    ema_backup: dict[str, Any] | None = None
                    if ema is not None:
                        ema_backup = ema.apply_shadow(ddp_model.module)

                    (
                        vl, vtok, vc, vt,
                        v_cos_sum, v_euc_sum, v_vec_n,
                        v_prox_correct, v_prox_total,
                    ) = _run_validation_steps(
                        ddp_model, val_collator, val_it,
                        args.val_steps, args.val_generate_max_new_tokens,
                        device, dtype,
                        debug_examples_n=args.val_log_examples,
                        debug_examples_out_L=val_examples,
                        action_stats_out_d=val_stats,
                        visual_samples_out_L=visual_samples,
                        visual_max_frames=args.val_visual_max_frames,
                        label_smoothing=args.label_smoothing,
                        focal_gamma=args.focal_loss_gamma,
                        class_balanced=args.class_balanced_loss,
                        val_temperature=args.val_temperature,
                        diversity_penalty=args.diversity_penalty,
                        mouse_prox_px_threshold=args.mouse_prox_px_threshold,
                        skip_noop_frames=args.skip_noop_frames,
                    )
                    if ema_backup is not None:
                        ema.restore(ddp_model.module, ema_backup)

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
                    vcos_t = torch.tensor(v_cos_sum, device=device)
                    veuc_t = torch.tensor(v_euc_sum, device=device)
                    vvecn_t = torch.tensor(float(v_vec_n), device=device)
                    vproxc_t = torch.tensor(float(v_prox_correct), device=device)
                    vproxt_t = torch.tensor(float(v_prox_total), device=device)

                    if world_i > 1:
                        for t in (vl_num_t, vtok_t, vc_t, vt_t, vpno_t, vpmo_t,
                                  vpto_t, vtno_t, vtmo_t, vtto_t, cnoc_t, cnot_t,
                                  cmoc_t, cmot_t, conf_t, vcos_t, veuc_t, vvecn_t,
                                  vproxc_t, vproxt_t):
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
                    val_mouse_cos_sim = vcos_t.item() / max(vvecn_t.item(), 1.0)
                    val_mouse_euc_dist = veuc_t.item() / max(vvecn_t.item(), 1.0)
                    val_mouse_prox_acc = vproxc_t.item() / max(vproxt_t.item(), 1.0)
                    conf_NM = [[int(x) for x in row] for row in conf_t.detach().cpu().tolist()]

                    if rank_i == 0:
                        print(
                            f"step={global_step} val_loss={val_loss_f:.6f} "
                            f"val_acc={val_acc:.6f} val_f1={val_f1:.6f} "
                            f"acc_no_op={acc_no:.4f} acc_mouse={acc_mo:.4f} "
                            f"mouse_cos_sim={val_mouse_cos_sim:.4f} "
                            f"mouse_euc_dist={val_mouse_euc_dist:.2f} "
                            f"mouse_prox_acc={val_mouse_prox_acc:.4f} "
                            f"(thr={args.mouse_prox_px_threshold:.0f}px) "
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
                                "val_action/mouse_cosine_sim": val_mouse_cos_sim,
                                "val_action/mouse_euclidean_dist": val_mouse_euc_dist,
                                "val_action/mouse_prox_acc": val_mouse_prox_acc,
                                "val_action/mouse_prox_threshold_px": args.mouse_prox_px_threshold,
                                "val_action/mouse_vector_pairs_n": vvecn_t.item(),
                            }
                            chart = _wandb_confusion_chart(conf_NM)
                            if chart is not None:
                                wlog["val_action/confusion_matrix"] = chart

                            # ── Visual cursor overlay ────────────────
                            if visual_samples and len(visual_samples) > 0:
                                sample = visual_samples[0]
                                gt_actions = _actions_from_target_text(sample["gt_text"])
                                pred_actions = _actions_from_pred_text(sample["pred_text"])
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
