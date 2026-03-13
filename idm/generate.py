from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from idm.utils.action_metrics import (
    action_event_match_counts,
    precision_recall_f1_from_counts,
)
from idm.utils.action_text import (
    action_accuracy_counts_from_texts,
    action_type_counts_from_texts,
    normalize_action,
    parse_frame_actions,
    per_action_stats_from_actions,
)


DEFAULT_INSTRUCTION = (
    "Given the video frames, output the action text for each frame in order."
)
GRID_COLS_DEFAULT = 8
GRID_TEXT_H_DEFAULT = 40


@dataclass
class Args:
    data_root: str = ""
    split: str = "val"
    output_dir: str = "./eval_results"
    hf_model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    hf_model_state_pkl: str = ""
    hf_attn_implementation: str = "auto"
    hf_precision: str = "bf16"
    hf_device: str = "cuda"
    image_h: int = 540
    image_w: int = 960
    image_c: int = 3
    video_fps: float = 30.0
    seq_len: int = 48
    global_batch_size: int = 12
    val_steps: int = 12
    skip_val_batches: int = 0
    val_generate_max_new_tokens: int = 1536
    seed: int = 0
    num_workers: int = 2
    prefetch_buffer_size: int = 2
    read_num_threads: int = 2
    worker_buffer_size: int = 2
    instruction_text: str = DEFAULT_INSTRUCTION
    visualize: bool = False
    max_visualizations: int = 128


def _parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="HF-only IDM evaluation/generation with train-equivalent inference."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--hf-model-path", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--hf-model-state-pkl", default="")
    parser.add_argument("--hf-attn-implementation", default="auto")
    parser.add_argument(
        "--hf-precision", choices=["bf16", "fp16", "fp32"], default="bf16"
    )
    parser.add_argument("--hf-device", default="cuda")
    parser.add_argument("--image-h", type=int, default=540)
    parser.add_argument("--image-w", type=int, default=960)
    parser.add_argument("--image-c", type=int, default=3)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--global-batch-size", type=int, default=12)
    parser.add_argument("--val-steps", type=int, default=12)
    parser.add_argument("--skip-val-batches", type=int, default=0)
    parser.add_argument("--val-generate-max-new-tokens", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-buffer-size", type=int, default=2)
    parser.add_argument("--read-num-threads", type=int, default=2)
    parser.add_argument("--worker-buffer-size", type=int, default=2)
    parser.add_argument("--instruction-text", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--max-visualizations", type=int, default=128)
    ns = parser.parse_args()
    return Args(
        data_root=ns.data_root,
        split=ns.split,
        output_dir=ns.output_dir,
        hf_model_path=ns.hf_model_path,
        hf_model_state_pkl=ns.hf_model_state_pkl,
        hf_attn_implementation=ns.hf_attn_implementation,
        hf_precision=ns.hf_precision,
        hf_device=ns.hf_device,
        image_h=ns.image_h,
        image_w=ns.image_w,
        image_c=ns.image_c,
        video_fps=ns.video_fps,
        seq_len=ns.seq_len,
        global_batch_size=ns.global_batch_size,
        val_steps=ns.val_steps,
        skip_val_batches=ns.skip_val_batches,
        val_generate_max_new_tokens=ns.val_generate_max_new_tokens,
        seed=ns.seed,
        num_workers=ns.num_workers,
        prefetch_buffer_size=ns.prefetch_buffer_size,
        read_num_threads=ns.read_num_threads,
        worker_buffer_size=ns.worker_buffer_size,
        instruction_text=ns.instruction_text,
        visualize=ns.visualize,
        max_visualizations=ns.max_visualizations,
    )


def _validate_args(args: Args) -> None:
    if not Path(args.data_root).exists():
        raise ValueError(f"--data-root does not exist: {args.data_root}")
    if not args.split.strip():
        raise ValueError("--split cannot be empty.")
    if not args.hf_model_path.strip():
        raise ValueError("--hf-model-path cannot be empty.")
    if args.hf_model_state_pkl.strip() and not Path(args.hf_model_state_pkl).exists():
        raise ValueError(f"--hf-model-state-pkl not found: {args.hf_model_state_pkl}")
    if args.hf_attn_implementation not in {"auto", "flash_attention_2", "sdpa"}:
        raise ValueError(
            "--hf-attn-implementation must be one of: auto, flash_attention_2, sdpa."
        )
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be > 0.")
    if args.image_h <= 0 or args.image_w <= 0 or args.image_c <= 0:
        raise ValueError("--image-h/--image-w/--image-c must be >= 1.")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be >= 1.")
    if args.global_batch_size <= 0:
        raise ValueError("--global-batch-size must be >= 1.")
    if args.val_steps <= 0:
        raise ValueError("--val-steps must be >= 1.")
    if args.skip_val_batches < 0:
        raise ValueError("--skip-val-batches must be >= 0.")
    if args.val_generate_max_new_tokens <= 0:
        raise ValueError("--val-generate-max-new-tokens must be >= 1.")
    if args.seed < 0:
        raise ValueError("--seed must be >= 0.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")
    if args.prefetch_buffer_size <= 0:
        raise ValueError("--prefetch-buffer-size must be >= 1.")
    if args.read_num_threads <= 0:
        raise ValueError("--read-num-threads must be >= 1.")
    if args.worker_buffer_size <= 0:
        raise ValueError("--worker-buffer-size must be >= 1.")
    if args.max_visualizations < 0:
        raise ValueError("--max-visualizations must be >= 0.")


def _window_coverage_counts(
    n_actions: int,
    windows: list[tuple[int, int, int, int]],
) -> tuple[list[int], list[int]]:
    in_any_count = [0] * n_actions
    in_used_count = [0] * n_actions
    for start, end, use_start, use_end in windows:
        for idx in range(max(start, 0), min(end, n_actions)):
            in_any_count[idx] += 1
        use_global_start = start + use_start
        use_global_end = start + use_end
        for idx in range(max(use_global_start, 0), min(use_global_end, n_actions)):
            in_used_count[idx] += 1
    return in_any_count, in_used_count


def _create_video_grid(
    frames: np.ndarray,
    gt_actions: list[str],
    pred_actions: list[str],
    cols: int = GRID_COLS_DEFAULT,
    text_h: int = GRID_TEXT_H_DEFAULT,
) -> Any:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "--visualize was requested but Pillow is not installed."
        ) from exc

    n_frames = len(frames)
    h_i = int(frames.shape[1])
    w_i = int(frames.shape[2])
    rows_i = (n_frames + cols - 1) // cols
    cell_h_i = h_i + text_h

    grid = Image.new("RGB", (cols * w_i, rows_i * cell_h_i), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except OSError:
        font = ImageFont.load_default()

    for frame_i, frame_HWC in enumerate(frames):
        row_i = frame_i // cols
        col_i = frame_i % cols
        x_i = col_i * w_i
        y_i = row_i * cell_h_i
        grid.paste(Image.fromarray(frame_HWC), (x_i, y_i))

        outline_color = (120, 120, 120)
        if frame_i < len(gt_actions):
            gt_s = gt_actions[frame_i]
            pred_s = pred_actions[frame_i] if frame_i < len(pred_actions) else ""
            match_b = normalize_action(gt_s) == normalize_action(pred_s)
            outline_color = (18, 145, 90) if match_b else (200, 45, 45)
            text_color = (0, 128, 0) if match_b else (200, 0, 0)
            draw.text(
                (x_i + 2, y_i + h_i + 2),
                f"GT:{gt_s[:12]}\\nP:{pred_s[:12]}",
                fill=text_color,
                font=font,
            )
        draw.rectangle(
            (x_i, y_i, x_i + w_i - 1, y_i + h_i - 1),
            outline=outline_color,
            width=2,
        )
    return grid


def build_visualization_payload(
    *,
    video_id: str,
    frames: np.ndarray,
    gt_actions: list[str],
    pred_actions: list[str],
    correct: int,
    total: int,
    cols: int = GRID_COLS_DEFAULT,
    text_h: int = GRID_TEXT_H_DEFAULT,
) -> dict[str, Any]:
    n_actions = int(len(gt_actions))
    pred_actions = list(pred_actions[:n_actions]) + [""] * max(
        0, n_actions - len(pred_actions)
    )
    windows = [(0, n_actions, 0, n_actions)]
    in_any_count, in_used_count = _window_coverage_counts(n_actions, windows)

    frame_rows = []
    for idx_i, (gt_s, pred_s) in enumerate(zip(gt_actions, pred_actions)):
        frame_rows.append(
            {
                "index": idx_i,
                "gt_action": gt_s,
                "pred_action": pred_s,
                "correct": normalize_action(gt_s) == normalize_action(pred_s),
                "in_any_window": in_any_count[idx_i] > 0,
                "in_window_region": in_used_count[idx_i] > 0,
                "window_coverage_count": in_any_count[idx_i],
                "window_region_count": in_used_count[idx_i],
                "selected_by_stitching": True,
            }
        )

    return {
        "video_id": video_id,
        "frame_count": int(len(frames)),
        "action_count": n_actions,
        "frame_h": int(frames.shape[1]),
        "frame_w": int(frames.shape[2]),
        "grid_cols": cols,
        "grid_text_h": text_h,
        "accuracy": (float(correct) / max(float(total), 1.0)),
        "correct": int(correct),
        "total": int(total),
        "windows": [
            {
                "start": 0,
                "end": n_actions,
                "use_start": 0,
                "use_end": n_actions,
                "use_global_start": 0,
                "use_global_end": n_actions,
            }
        ],
        "frames": frame_rows,
    }


def save_visualization_artifacts(
    *,
    output_stem: Path,
    video_id: str,
    frames: np.ndarray,
    gt_actions: list[str],
    pred_actions: list[str],
    correct: int,
    total: int,
) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    grid = _create_video_grid(frames, gt_actions, pred_actions)
    png_path = output_stem.parent / f"{output_stem.name}.png"
    json_path = output_stem.parent / f"{output_stem.name}.json"
    grid.save(png_path)
    payload_d = build_visualization_payload(
        video_id=video_id,
        frames=frames,
        gt_actions=gt_actions,
        pred_actions=pred_actions,
        correct=correct,
        total=total,
    )
    json_path.write_text(json.dumps(payload_d, indent=2))
    return png_path, json_path


def _to_device(
    batch_d: dict[str, Any],
    device: torch.device,
    skip_keys: set[str],
) -> dict[str, Any]:
    out_d: dict[str, Any] = {}
    for key_s, val in batch_d.items():
        if key_s in skip_keys:
            continue
        out_d[key_s] = (
            val.to(device, non_blocking=True) if isinstance(val, torch.Tensor) else val
        )
    return out_d


def _weighted_causal_lm_loss(
    logits_BSV: torch.Tensor,
    labels_BS: torch.Tensor,
    label_weights_BS: torch.Tensor | None,
) -> torch.Tensor:
    shift_logits_BSV = logits_BSV[:, :-1, :].contiguous()
    shift_labels_BS = labels_BS[:, 1:].contiguous()
    token_loss_BS = torch.nn.functional.cross_entropy(
        input=shift_logits_BSV.view(-1, shift_logits_BSV.shape[-1]),
        target=shift_labels_BS.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels_BS)
    valid_mask_BS = shift_labels_BS != -100

    if label_weights_BS is None:
        denom_t = torch.clamp(valid_mask_BS.sum(), min=1).to(token_loss_BS.dtype)
        return (token_loss_BS * valid_mask_BS.to(token_loss_BS.dtype)).sum() / denom_t

    shift_weights_BS = label_weights_BS[:, 1:].to(token_loss_BS.dtype).contiguous()
    valid_weights_BS = shift_weights_BS * valid_mask_BS.to(token_loss_BS.dtype)
    denom_t = torch.clamp(valid_weights_BS.sum(), min=1.0)
    return (token_loss_BS * valid_weights_BS).sum() / denom_t


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


def _action_f1_from_counts(
    *,
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


def _load_hf_model_and_processor(
    args: Args,
) -> tuple[Qwen3VLForConditionalGeneration, Any, torch.device, torch.dtype, bool]:
    device = torch.device(args.hf_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--hf-device cuda requires visible CUDA devices.")

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model_dtype = dtype_map[args.hf_precision]
    model_kwargs: dict[str, Any] = {
        "torch_dtype": model_dtype,
        "trust_remote_code": True,
    }
    if args.hf_attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.hf_attn_implementation

    processor = AutoProcessor.from_pretrained(
        args.hf_model_path, trust_remote_code=True
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.hf_model_path,
        **model_kwargs,
    ).to(device)

    if args.hf_model_state_pkl.strip():
        with open(args.hf_model_state_pkl, "rb") as f:
            state_d = pickle.load(f)
        missing_L, unexpected_L = model.load_state_dict(state_d, strict=False)
        if missing_L or unexpected_L:
            print(
                "Warning: state dict mismatch on non-strict load: "
                f"missing={len(missing_L)} unexpected={len(unexpected_L)}"
            )
    model.eval()

    amp_enabled = device.type == "cuda" and args.hf_precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.hf_precision == "bf16" else torch.float16
    return model, processor, device, amp_dtype, amp_enabled


def run_eval(args: Args) -> dict[str, Any]:
    from idm.utils.collator import VideoSFTCollator
    from idm.utils.data import find_array_record_paths, get_dataloader

    model, processor, device, amp_dtype, amp_enabled = _load_hf_model_and_processor(
        args
    )

    collator = VideoSFTCollator(
        processor=processor,
        instruction_text=args.instruction_text,
        video_fps=args.video_fps,
        no_op_loss_weight=1.0,
        mouse_loss_weight=1.0,
    )

    val_paths = find_array_record_paths(args.data_root, split=args.split)
    val_loader = get_dataloader(
        array_record_paths=val_paths,
        seq_len=args.seq_len,
        global_batch_size=args.global_batch_size,
        image_h=args.image_h,
        image_w=args.image_w,
        image_c=args.image_c,
        rank=0,
        world_size=1,
        seed=args.seed,
        epoch_i=0,
        num_epochs=None,
        num_workers=args.num_workers,
        prefetch_buffer_size=args.prefetch_buffer_size,
        read_num_threads=args.read_num_threads,
        worker_buffer_size=args.worker_buffer_size,
        min_action_density=0.0,
        action_upsample_random_fraction=1.0,
    )

    val_it = iter(val_loader)
    for _ in range(int(args.skip_val_batches)):
        try:
            next(val_it)
        except StopIteration as exc:
            raise ValueError(
                "Validation loader exhausted while applying --skip-val-batches."
            ) from exc

    val_loss_num_f = 0.0
    val_tok_n = 0
    val_action_correct_n = 0
    val_action_total_n = 0
    val_pred_no_op_n = 0
    val_pred_mouse_n = 0
    val_pred_action_total_n = 0
    val_target_no_op_n = 0
    val_target_mouse_n = 0
    val_target_action_total_n = 0
    class_counts_d: dict[str, int] = {}
    strict_tp_n = 0
    strict_fp_n = 0
    strict_fn_n = 0
    tolerant_tp_n = 0
    tolerant_fp_n = 0
    tolerant_fn_n = 0
    all_predictions_L: list[str] = []
    all_ground_truth_L: list[str] = []

    vis_map_L: list[dict[str, Any]] = []
    vis_saved_n = 0
    vis_dir = Path(args.output_dir) / "visualizations"
    if args.visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = processor.tokenizer

    model.eval()
    with torch.no_grad():
        for step_i in range(int(args.val_steps)):
            try:
                raw_batch_d = next(val_it)
            except StopIteration as exc:
                raise ValueError(
                    "Validation loader exhausted before --val-steps. "
                    "Reduce --val-steps or --skip-val-batches."
                ) from exc

            collated_batch_d = collator(raw_batch_d)
            model_batch_d = _to_device(
                collated_batch_d,
                device,
                skip_keys={
                    "videos",
                    "prompt_lens",
                    "meta",
                    "label_weights",
                    "target_text",
                },
            )
            labels_BS = model_batch_d["labels"]
            batch_tok_n = int((labels_BS != -100).sum().item())
            val_tok_n += batch_tok_n

            label_weights_BS = collated_batch_d.get("label_weights")
            if label_weights_BS is not None:
                label_weights_BS = label_weights_BS.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                outputs = model(**model_batch_d)
                loss_t = _weighted_causal_lm_loss(
                    logits_BSV=outputs.logits,
                    labels_BS=labels_BS,
                    label_weights_BS=label_weights_BS,
                )
            val_loss_num_f += float(loss_t.detach().item()) * float(batch_tok_n)

            prompt_batch_d = collator.prompt_model_inputs(raw_batch_d)
            prompt_lens_B = [int(x) for x in prompt_batch_d.pop("prompt_lens")]
            prompt_model_d = _to_device(
                prompt_batch_d,
                device,
                skip_keys={"videos", "meta"},
            )
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": int(args.val_generate_max_new_tokens),
                "do_sample": False,
                "use_cache": True,
            }
            pad_id = getattr(tokenizer, "pad_token_id", None)
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = int(pad_id)
            if eos_id is not None:
                gen_kwargs["eos_token_id"] = int(eos_id)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                generated_ids_BS = model.generate(
                    **prompt_model_d,
                    **gen_kwargs,
                )

            pred_text_B = _decode_pred_text_B_from_generated_ids(
                generated_ids_BS=generated_ids_BS,
                prompt_lens_B=prompt_lens_B,
                tokenizer=tokenizer,
            )
            target_text_B = [str(x) for x in raw_batch_d["target_text"]]

            for pred_s, target_s in zip(pred_text_B, target_text_B):
                target_actions_L = parse_frame_actions(target_s)
                pred_actions_L = parse_frame_actions(
                    pred_s, expected_n=len(target_actions_L)
                )
                all_predictions_L.extend(pred_actions_L)
                all_ground_truth_L.extend(target_actions_L)

            correct_i, total_i = action_accuracy_counts_from_texts(
                pred_text_B=pred_text_B,
                target_text_B=target_text_B,
                class_counts_out_d=class_counts_d,
            )
            pred_no_op_i, pred_mouse_i, pred_total_i = action_type_counts_from_texts(
                pred_text_B
            )
            target_no_op_i, target_mouse_i, target_total_i = (
                action_type_counts_from_texts(target_text_B)
            )
            strict_tp_i, strict_fp_i, strict_fn_i = action_event_match_counts(
                pred_text_B=pred_text_B,
                target_text_B=target_text_B,
                tolerance_frames=0,
                ignore_no_op=True,
            )
            tolerant_tp_i, tolerant_fp_i, tolerant_fn_i = action_event_match_counts(
                pred_text_B=pred_text_B,
                target_text_B=target_text_B,
                tolerance_frames=5,
                ignore_no_op=True,
            )

            val_action_correct_n += int(correct_i)
            val_action_total_n += int(total_i)
            val_pred_no_op_n += int(pred_no_op_i)
            val_pred_mouse_n += int(pred_mouse_i)
            val_pred_action_total_n += int(pred_total_i)
            val_target_no_op_n += int(target_no_op_i)
            val_target_mouse_n += int(target_mouse_i)
            val_target_action_total_n += int(target_total_i)
            strict_tp_n += int(strict_tp_i)
            strict_fp_n += int(strict_fp_i)
            strict_fn_n += int(strict_fn_i)
            tolerant_tp_n += int(tolerant_tp_i)
            tolerant_fp_n += int(tolerant_fp_i)
            tolerant_fn_n += int(tolerant_fn_i)

            if args.visualize:
                frames_B = raw_batch_d["frames"]
                for sample_i, (frames_SHWC, pred_s, target_s) in enumerate(
                    zip(frames_B, pred_text_B, target_text_B)
                ):
                    if (
                        args.max_visualizations > 0
                        and vis_saved_n >= args.max_visualizations
                    ):
                        break
                    gt_actions_L = parse_frame_actions(target_s)
                    pred_actions_L = parse_frame_actions(
                        pred_s, expected_n=len(gt_actions_L)
                    )
                    sample_correct_n = int(
                        sum(
                            normalize_action(pred_a) == normalize_action(gt_a)
                            for pred_a, gt_a in zip(pred_actions_L, gt_actions_L)
                        )
                    )
                    sample_total_n = int(len(gt_actions_L))
                    video_id = f"sample_{vis_saved_n:06d}"
                    output_stem = (
                        vis_dir
                        / f"{video_id}_acc{(sample_correct_n / max(sample_total_n, 1)):.2f}"
                    )
                    save_visualization_artifacts(
                        output_stem=output_stem,
                        video_id=video_id,
                        frames=np.asarray(frames_SHWC, dtype=np.uint8),
                        gt_actions=gt_actions_L,
                        pred_actions=pred_actions_L,
                        correct=sample_correct_n,
                        total=sample_total_n,
                    )
                    vis_map_L.append(
                        {
                            "sample_id": video_id,
                            "step": int(step_i),
                            "batch_index": int(sample_i),
                        }
                    )
                    vis_saved_n += 1

    val_loss_f = val_loss_num_f / max(float(val_tok_n), 1.0)
    val_action_acc_f = float(val_action_correct_n) / max(float(val_action_total_n), 1.0)
    val_action_f1_f = _action_f1_from_counts(
        correct_n_f=float(val_action_correct_n),
        pred_total_n_f=float(val_pred_action_total_n),
        target_total_n_f=float(val_target_action_total_n),
    )
    val_pred_no_op_rate_f = float(val_pred_no_op_n) / max(
        float(val_pred_action_total_n), 1.0
    )
    val_target_no_op_rate_f = float(val_target_no_op_n) / max(
        float(val_target_action_total_n), 1.0
    )
    val_pred_mouse_rate_f = float(val_pred_mouse_n) / max(
        float(val_pred_action_total_n), 1.0
    )
    val_target_mouse_rate_f = float(val_target_mouse_n) / max(
        float(val_target_action_total_n), 1.0
    )

    val_action_acc_no_op_f = float(class_counts_d.get("no_op_correct_n", 0)) / max(
        float(class_counts_d.get("no_op_total_n", 0)),
        1.0,
    )
    val_action_acc_mouse_f = float(class_counts_d.get("mouse_correct_n", 0)) / max(
        float(class_counts_d.get("mouse_total_n", 0)),
        1.0,
    )
    val_action_acc_keyboard_f = float(
        class_counts_d.get("keyboard_correct_n", 0)
    ) / max(
        float(class_counts_d.get("keyboard_total_n", 0)),
        1.0,
    )

    val_precision_strict_f, val_recall_strict_f, val_f1_strict_f = (
        precision_recall_f1_from_counts(
            tp_n=strict_tp_n,
            fp_n=strict_fp_n,
            fn_n=strict_fn_n,
        )
    )
    val_precision_tolerant_f, val_recall_tolerant_f, val_f1_tolerant_f = (
        precision_recall_f1_from_counts(
            tp_n=tolerant_tp_n,
            fp_n=tolerant_fp_n,
            fn_n=tolerant_fn_n,
        )
    )

    results_d: dict[str, Any] = {
        "metrics": {
            "val_loss": val_loss_f,
            "val_tokens": int(val_tok_n),
            "val_action_correct": int(val_action_correct_n),
            "val_action_total": int(val_action_total_n),
            "val_action_acc": val_action_acc_f,
            "val_action_f1": val_action_f1_f,
            "val_pred_no_op_rate": val_pred_no_op_rate_f,
            "val_target_no_op_rate": val_target_no_op_rate_f,
            "val_pred_mouse_rate": val_pred_mouse_rate_f,
            "val_target_mouse_rate": val_target_mouse_rate_f,
            "val_action_acc_no_op": val_action_acc_no_op_f,
            "val_action_acc_mouse": val_action_acc_mouse_f,
            "val_action_acc_keyboard": val_action_acc_keyboard_f,
            "val_precision_strict": val_precision_strict_f,
            "val_recall_strict": val_recall_strict_f,
            "val_f1_strict": val_f1_strict_f,
            "val_precision_tolerant": val_precision_tolerant_f,
            "val_recall_tolerant": val_recall_tolerant_f,
            "val_f1_tolerant": val_f1_tolerant_f,
        },
        "per_action": per_action_stats_from_actions(
            pred_actions_L=all_predictions_L,
            gt_actions_L=all_ground_truth_L,
        ),
        "config": {
            "data_root": args.data_root,
            "split": args.split,
            "hf_model_path": args.hf_model_path,
            "hf_model_state_pkl": args.hf_model_state_pkl,
            "hf_attn_implementation": args.hf_attn_implementation,
            "hf_precision": args.hf_precision,
            "hf_device": args.hf_device,
            "image_h": args.image_h,
            "image_w": args.image_w,
            "image_c": args.image_c,
            "video_fps": args.video_fps,
            "seq_len": args.seq_len,
            "global_batch_size": args.global_batch_size,
            "val_steps": args.val_steps,
            "skip_val_batches": args.skip_val_batches,
            "val_generate_max_new_tokens": args.val_generate_max_new_tokens,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "prefetch_buffer_size": args.prefetch_buffer_size,
            "read_num_threads": args.read_num_threads,
            "worker_buffer_size": args.worker_buffer_size,
            "visualize": args.visualize,
            "max_visualizations": args.max_visualizations,
        },
    }
    if args.visualize:
        results_d["visualization_count"] = int(vis_saved_n)
        results_d["visualization_dir"] = str(vis_dir)
        (Path(args.output_dir) / "video_id_map.json").write_text(
            json.dumps(vis_map_L, indent=2)
        )
    return results_d


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_d = run_eval(args)
    metrics_d = results_d["metrics"]
    print(
        "val_action_acc="
        f"{float(metrics_d['val_action_acc']):.6f} "
        "val_action_acc_keyboard="
        f"{float(metrics_d['val_action_acc_keyboard']):.6f} "
        "val_f1_tolerant="
        f"{float(metrics_d['val_f1_tolerant']):.6f}"
    )

    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(results_d, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
