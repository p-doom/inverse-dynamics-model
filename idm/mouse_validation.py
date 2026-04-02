from __future__ import annotations

from typing import Any

import cv2
import torch

from mouse_actions import (
    _ACTION_CLASS_NAMES,
    _ACTION_CONFUSION_PRED_CLASS_NAMES,
    _actions_from_target_text,
    _decode_pred_text_B_from_generated_ids,
)
from mouse_metrics import _action_confusion_count_key
from mouse_loss import _weighted_causal_lm_loss
from mouse_metrics import (
    _action_accuracy_counts_from_texts,
    _action_type_counts_from_texts,
    _mouse_proximity_counts_from_texts,
    _mouse_vector_metrics_from_texts,
)
from mouse_train_utils import _to_device


def _run_validation_steps(
    ddp_model: torch.nn.Module,
    collator: Any,
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
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
    class_balanced: bool = False,
    val_temperature: float = 1.0,
    diversity_penalty: float = 0.0,
    mouse_prox_px_threshold: float = 50.0,
) -> tuple[float, int, int, int, float, float, int, int, int]:
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
    val_mouse_cos_sum = 0.0
    val_mouse_euc_sum = 0.0
    val_mouse_vec_n = 0
    val_mouse_prox_correct = 0
    val_mouse_prox_total = 0
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
                    label_smoothing=label_smoothing,
                    focal_gamma=focal_gamma,
                    class_balanced=class_balanced,
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
            if val_temperature != 1.0 and val_temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = val_temperature
                gen_kwargs["top_k"] = 50
            if diversity_penalty > 0.0:
                gen_kwargs["repetition_penalty"] = diversity_penalty
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = int(pad_id)
            if eos_id is not None:
                gen_kwargs["eos_token_id"] = int(eos_id)
            with torch.autocast(device.type, dtype=dtype, enabled=(device.type == "cuda")):
                if "mm_token_type_ids" in prompt_model:
                    if "image_grid_thw" not in prompt_model or prompt_model["image_grid_thw"] is None:
                        del prompt_model["mm_token_type_ids"]
                gen_ids = gen_model.generate(**prompt_model, **gen_kwargs)
            pred_text_B = _decode_pred_text_B_from_generated_ids(
                gen_ids, prompt_lens, collator.tokenizer,
            )
            target_text_B = [str(x) for x in raw_batch["target_text"]]

            # -- Collect visual samples from first few batches --
            if (
                visual_samples_out_L is not None
                and len(visual_samples_out_L) < 3
            ):
                frames_np = raw_batch["frames"]  # (B, T, H, W, C) numpy
                if frames_np is not None and len(frames_np) > 0:
                    for b_idx in range(min(len(frames_np), 3 - len(visual_samples_out_L))):
                        sample_frames = frames_np[b_idx][:visual_max_frames]
                        jpeg_bytes = []
                        for f in sample_frames:
                            ok, enc = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                            if ok:
                                jpeg_bytes.append(enc.tobytes())
                        if jpeg_bytes and b_idx < len(target_text_B) and b_idx < len(pred_text_B):
                            visual_samples_out_L.append({
                                "jpeg_frames": jpeg_bytes,
                                "gt_text": target_text_B[b_idx],
                                "pred_text": pred_text_B[b_idx],
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
            cos_s, euc_s, vec_n = _mouse_vector_metrics_from_texts(
                pred_text_B, target_text_B,
            )
            prox_c, prox_t = _mouse_proximity_counts_from_texts(
                pred_text_B, target_text_B,
                threshold_px=mouse_prox_px_threshold,
            )
            val_mouse_cos_sum += cos_s
            val_mouse_euc_sum += euc_s
            val_mouse_vec_n += vec_n
            val_mouse_prox_correct += prox_c
            val_mouse_prox_total += prox_t
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
    return (
        val_loss_f, val_tok_n, val_correct_n, val_total_n,
        val_mouse_cos_sum, val_mouse_euc_sum, val_mouse_vec_n,
        val_mouse_prox_correct, val_mouse_prox_total,
    )
