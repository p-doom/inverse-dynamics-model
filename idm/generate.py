from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
import glob
import json
import os
import pickle
from pathlib import Path
import tempfile
from typing import Any, Iterator

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
DEFAULT_SGLANG_INSTRUCTION = (
    "Given the video, output exactly one line per frame in this format: "
    "Frame <index>: <ACTION>. Use IDM action strings like NO_OP, "
    "MOUSE:dx,dy,dz, or MOUSE:dx,dy,dz ; <pressed_keys>. "
    "Do not add explanations."
)
GRID_COLS_DEFAULT = 8
GRID_TEXT_H_DEFAULT = 40


@dataclass
class SlidingWindowConfig:
    seq_len: int = 128
    stride: int = 64
    center_start: int = 32
    center_end: int = 96


@dataclass
class Args:
    data_root: str = ""
    split: str = "val"
    eval_mode: str = "sliding_window"
    backend: str = "sglang"
    sglang_url: str = "http://localhost:30000"
    output_dir: str = "./eval_results"
    image_h: int = 90
    image_w: int = 160
    image_c: int = 3
    video_fps: float = 30.0
    seq_len: int = 128
    stride: int = 64
    center_start: int = 32
    center_end: int = 96
    max_videos: int | None = None
    sample_seed: int = 0
    visualize: bool = False
    instruction_text: str = DEFAULT_INSTRUCTION
    model: str = "default"
    hf_model_path: str = ""
    hf_model_state_pkl: str = ""
    hf_attn_implementation: str = "auto"
    hf_precision: str = "bf16"
    hf_device: str = "cuda"
    max_tokens: int = 1024
    max_frames_per_request: int = 16
    request_timeout_s: int = 120
    train_style_val_steps: int = 12
    train_style_skip_val_batches: int = 0
    train_style_val_generate_max_new_tokens: int = 1536
    train_style_global_batch_size: int = 12
    train_style_num_workers: int = 2
    train_style_prefetch_buffer_size: int = 2
    train_style_read_num_threads: int = 2
    train_style_worker_buffer_size: int = 2
    train_style_no_op_loss_weight: float = 1.0
    train_style_mouse_loss_weight: float = 1.0


@dataclass
class EvalMetrics:
    correct: int = 0
    total: int = 0
    per_video: list[float] = field(default_factory=list)
    all_predictions: list[str] = field(default_factory=list)
    all_ground_truth: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def get_per_action_accuracy(self) -> dict[str, dict[str, float]]:
        return per_action_stats_from_actions(
            pred_actions_L=self.all_predictions,
            gt_actions_L=self.all_ground_truth,
        )


@dataclass
class VideoEvalResult:
    correct: int
    total: int
    predictions: list[str]
    windows: list[tuple[int, int, int, int]]
    selected_mask: list[bool]


def _parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Evaluate IDM via SGLang endpoint or local HF generation."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--eval-mode",
        choices=["sliding_window", "apples_to_apples", "train_style"],
        default="sliding_window",
    )
    parser.add_argument("--backend", choices=["sglang", "hf"], default="sglang")
    parser.add_argument("--sglang-url", default="http://localhost:30000")
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--image-h", type=int, default=90)
    parser.add_argument("--image-w", type=int, default=160)
    parser.add_argument("--image-c", type=int, default=3)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--center-start", type=int, default=32)
    parser.add_argument("--center-end", type=int, default=96)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--instruction-text", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--model", default="default")
    parser.add_argument("--hf-model-path", default="")
    parser.add_argument("--hf-model-state-pkl", default="")
    parser.add_argument("--hf-attn-implementation", default="auto")
    parser.add_argument(
        "--hf-precision", choices=["bf16", "fp16", "fp32"], default="bf16"
    )
    parser.add_argument("--hf-device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-frames-per-request", type=int, default=16)
    parser.add_argument("--request-timeout-s", type=int, default=120)
    parser.add_argument("--train-style-val-steps", type=int, default=12)
    parser.add_argument("--train-style-skip-val-batches", type=int, default=0)
    parser.add_argument(
        "--train-style-val-generate-max-new-tokens",
        type=int,
        default=1536,
    )
    parser.add_argument("--train-style-global-batch-size", type=int, default=12)
    parser.add_argument("--train-style-num-workers", type=int, default=2)
    parser.add_argument("--train-style-prefetch-buffer-size", type=int, default=2)
    parser.add_argument("--train-style-read-num-threads", type=int, default=2)
    parser.add_argument("--train-style-worker-buffer-size", type=int, default=2)
    parser.add_argument("--train-style-no-op-loss-weight", type=float, default=1.0)
    parser.add_argument("--train-style-mouse-loss-weight", type=float, default=1.0)
    ns = parser.parse_args()
    eval_mode_s = "apples_to_apples" if ns.eval_mode == "train_style" else ns.eval_mode
    return Args(
        data_root=ns.data_root,
        split=ns.split,
        eval_mode=eval_mode_s,
        backend=ns.backend,
        sglang_url=ns.sglang_url,
        output_dir=ns.output_dir,
        image_h=ns.image_h,
        image_w=ns.image_w,
        image_c=ns.image_c,
        video_fps=ns.video_fps,
        seq_len=ns.seq_len,
        stride=ns.stride,
        center_start=ns.center_start,
        center_end=ns.center_end,
        max_videos=ns.max_videos,
        sample_seed=ns.sample_seed,
        visualize=ns.visualize,
        instruction_text=ns.instruction_text,
        model=ns.model,
        hf_model_path=ns.hf_model_path,
        hf_model_state_pkl=ns.hf_model_state_pkl,
        hf_attn_implementation=ns.hf_attn_implementation,
        hf_precision=ns.hf_precision,
        hf_device=ns.hf_device,
        max_tokens=ns.max_tokens,
        max_frames_per_request=ns.max_frames_per_request,
        request_timeout_s=ns.request_timeout_s,
        train_style_val_steps=ns.train_style_val_steps,
        train_style_skip_val_batches=ns.train_style_skip_val_batches,
        train_style_val_generate_max_new_tokens=ns.train_style_val_generate_max_new_tokens,
        train_style_global_batch_size=ns.train_style_global_batch_size,
        train_style_num_workers=ns.train_style_num_workers,
        train_style_prefetch_buffer_size=ns.train_style_prefetch_buffer_size,
        train_style_read_num_threads=ns.train_style_read_num_threads,
        train_style_worker_buffer_size=ns.train_style_worker_buffer_size,
        train_style_no_op_loss_weight=ns.train_style_no_op_loss_weight,
        train_style_mouse_loss_weight=ns.train_style_mouse_loss_weight,
    )


def _validate_args(args: Args) -> None:
    if not Path(args.data_root).exists():
        raise ValueError(f"--data-root does not exist: {args.data_root}")
    if not args.split.strip():
        raise ValueError("--split cannot be empty.")
    if args.backend == "sglang" and not args.sglang_url.strip():
        raise ValueError("--sglang-url cannot be empty for backend=sglang.")
    if args.backend == "hf" and not args.hf_model_path.strip():
        raise ValueError("--hf-model-path cannot be empty for backend=hf.")
    if args.hf_model_state_pkl.strip() and not Path(args.hf_model_state_pkl).exists():
        raise ValueError(f"--hf-model-state-pkl not found: {args.hf_model_state_pkl}")
    if args.eval_mode == "apples_to_apples":
        if args.backend != "hf":
            raise ValueError("--eval-mode apples_to_apples requires --backend hf.")
        if args.visualize:
            raise ValueError(
                "--visualize is not supported for --eval-mode apples_to_apples."
            )
        if args.max_videos is not None:
            raise ValueError(
                "--max-videos is not supported for --eval-mode apples_to_apples."
            )
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be >= 1.")
    if args.stride <= 0:
        raise ValueError("--stride must be >= 1.")
    if args.center_start < 0:
        raise ValueError("--center-start must be >= 0.")
    if args.center_end <= args.center_start:
        raise ValueError("--center-end must be > --center-start.")
    if args.image_h <= 0 or args.image_w <= 0 or args.image_c <= 0:
        raise ValueError("--image-h/--image-w/--image-c must be >= 1.")
    if args.max_videos is not None and args.max_videos <= 0:
        raise ValueError("--max-videos must be > 0 when provided.")
    if args.sample_seed < 0:
        raise ValueError("--sample-seed must be >= 0.")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0.")
    if args.max_frames_per_request <= 0:
        raise ValueError("--max-frames-per-request must be > 0.")
    if args.request_timeout_s <= 0:
        raise ValueError("--request-timeout-s must be > 0.")
    if args.train_style_val_steps <= 0:
        raise ValueError("--train-style-val-steps must be >= 1.")
    if args.train_style_skip_val_batches < 0:
        raise ValueError("--train-style-skip-val-batches must be >= 0.")
    if args.train_style_val_generate_max_new_tokens <= 0:
        raise ValueError("--train-style-val-generate-max-new-tokens must be >= 1.")
    if args.train_style_global_batch_size <= 0:
        raise ValueError("--train-style-global-batch-size must be >= 1.")
    if args.train_style_num_workers < 0:
        raise ValueError("--train-style-num-workers must be >= 0.")
    if args.train_style_prefetch_buffer_size <= 0:
        raise ValueError("--train-style-prefetch-buffer-size must be >= 1.")
    if args.train_style_read_num_threads <= 0:
        raise ValueError("--train-style-read-num-threads must be >= 1.")
    if args.train_style_worker_buffer_size <= 0:
        raise ValueError("--train-style-worker-buffer-size must be >= 1.")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be > 0.")
    if args.hf_attn_implementation not in {"auto", "flash_attention_2", "sdpa"}:
        raise ValueError(
            "--hf-attn-implementation must be one of: auto, flash_attention_2, sdpa."
        )


def find_array_record_paths(data_root: str, split: str) -> list[str]:
    split_dir = os.path.join(data_root, split)
    paths = sorted(glob.glob(os.path.join(split_dir, "*.array_record")))
    if not paths:
        raise ValueError(f"No .array_record files found in {split_dir}")
    return paths


def compute_windows(
    n_frames: int, cfg: SlidingWindowConfig
) -> list[tuple[int, int, int, int]]:
    if n_frames <= 0:
        return []
    if n_frames <= cfg.seq_len:
        return [(0, n_frames, 0, n_frames)]

    windows: list[tuple[int, int, int, int]] = []
    starts = list(range(0, n_frames - cfg.seq_len, cfg.stride))
    starts.append(n_frames - cfg.seq_len)

    for i, start in enumerate(starts):
        if i == 0:
            use_start, use_end = 0, cfg.center_end
        elif i == len(starts) - 1:
            use_start, use_end = cfg.center_start, cfg.seq_len
        else:
            use_start, use_end = cfg.center_start, cfg.center_end
        use_start = max(0, min(use_start, cfg.seq_len))
        use_end = max(use_start, min(use_end, cfg.seq_len))
        windows.append((start, start + cfg.seq_len, use_start, use_end))
    return windows


def _window_coverage_counts(
    n_actions: int, windows: list[tuple[int, int, int, int]]
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


def _sample_record_indices_by_path(
    paths_L: list[str],
    record_count_L: list[int],
    sample_size: int,
    seed: int,
) -> dict[str, list[int]]:
    if sample_size <= 0:
        return {}
    total_records_i = sum(record_count_L)
    if total_records_i <= 0:
        return {}

    sample_count_i = min(sample_size, total_records_i)
    rng = np.random.default_rng(seed)
    sampled_global_idx_A = np.sort(
        rng.choice(total_records_i, size=sample_count_i, replace=False)
    )

    plan_d: dict[str, list[int]] = {}
    global_start_i = 0
    sampled_ptr_i = 0
    for path_s, n_records_i in zip(paths_L, record_count_L, strict=True):
        global_end_i = global_start_i + n_records_i
        while (
            sampled_ptr_i < sample_count_i
            and sampled_global_idx_A[sampled_ptr_i] < global_end_i
        ):
            local_idx_i = int(sampled_global_idx_A[sampled_ptr_i] - global_start_i)
            plan_d.setdefault(path_s, []).append(local_idx_i)
            sampled_ptr_i += 1
        global_start_i = global_end_i

    return plan_d


def _frames_to_mp4_data_uri(frames: np.ndarray, fps: float) -> str:
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "OpenCV is required for SGLang video payload encoding. Install with `pip install opencv-python`."
        ) from exc

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"Expected frames with shape [T, H, W, 3], got {tuple(frames.shape)}."
        )

    h_i = int(frames.shape[1])
    w_i = int(frames.shape[2])
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w_i, h_i),
    )
    if not writer.isOpened():
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Failed to open OpenCV VideoWriter for MP4 encoding.")

    try:
        for frame in frames:
            if frame.dtype == np.uint8:
                frame_u8 = frame
            else:
                frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    try:
        video_bytes = tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)

    if not video_bytes:
        raise RuntimeError("Encoded MP4 payload is empty.")
    return f"data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}"


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
    h, w = int(frames.shape[1]), int(frames.shape[2])
    rows = (n_frames + cols - 1) // cols
    cell_h = h + text_h
    cell_w = w

    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except OSError:
        font = ImageFont.load_default()

    for i, frame in enumerate(frames):
        row, col = i // cols, i % cols
        x, y = col * cell_w, row * cell_h
        grid.paste(Image.fromarray(frame), (x, y))

        outline_color = (120, 120, 120)
        if i < len(gt_actions):
            gt = gt_actions[i]
            pred = pred_actions[i] if i < len(pred_actions) else ""
            match = normalize_action(gt) == normalize_action(pred)
            color = (0, 128, 0) if match else (200, 0, 0)
            outline_color = (18, 145, 90) if match else (200, 45, 45)
            text = f"GT:{gt[:12]}\nP:{pred[:12]}"
            draw.text((x + 2, y + h + 2), text, fill=color, font=font)
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=outline_color, width=2)

    return grid


def build_visualization_payload(
    video_id: str,
    frames: np.ndarray,
    gt_actions: list[str],
    pred_actions: list[str],
    windows: list[tuple[int, int, int, int]],
    selected_mask: list[bool] | None = None,
    correct: int | None = None,
    total: int | None = None,
    cols: int = GRID_COLS_DEFAULT,
    text_h: int = GRID_TEXT_H_DEFAULT,
) -> dict[str, Any]:
    n_frames = int(len(frames))
    n_actions = int(len(gt_actions))
    pred_actions = list(pred_actions[:n_actions]) + [""] * max(
        0, n_actions - len(pred_actions)
    )
    in_any_count, in_used_count = _window_coverage_counts(n_actions, windows)
    if selected_mask is None or len(selected_mask) != n_actions:
        selected_mask = [count > 0 for count in in_used_count]

    frame_rows = []
    for idx, (gt, pred) in enumerate(zip(gt_actions, pred_actions)):
        frame_rows.append(
            {
                "index": idx,
                "gt_action": gt,
                "pred_action": pred,
                "correct": normalize_action(gt) == normalize_action(pred),
                "in_any_window": in_any_count[idx] > 0,
                "in_window_region": in_used_count[idx] > 0,
                "window_coverage_count": in_any_count[idx],
                "window_region_count": in_used_count[idx],
                "selected_by_stitching": bool(selected_mask[idx]),
            }
        )

    if correct is None:
        correct = int(sum(int(x["correct"]) for x in frame_rows))
    if total is None:
        total = n_actions

    return {
        "video_id": video_id,
        "frame_count": n_frames,
        "action_count": n_actions,
        "frame_h": int(frames.shape[1]),
        "frame_w": int(frames.shape[2]),
        "grid_cols": cols,
        "grid_text_h": text_h,
        "accuracy": (correct / total) if total else 0.0,
        "correct": int(correct),
        "total": int(total),
        "windows": [
            {
                "start": start,
                "end": end,
                "use_start": use_start,
                "use_end": use_end,
                "use_global_start": start + use_start,
                "use_global_end": start + use_end,
            }
            for start, end, use_start, use_end in windows
        ],
        "frames": frame_rows,
    }


def save_visualization_artifacts(
    output_stem: Path,
    video_id: str,
    frames: np.ndarray,
    gt_actions: list[str],
    pred_actions: list[str],
    windows: list[tuple[int, int, int, int]],
    selected_mask: list[bool] | None = None,
    correct: int | None = None,
    total: int | None = None,
    cols: int = GRID_COLS_DEFAULT,
    text_h: int = GRID_TEXT_H_DEFAULT,
) -> tuple[Path, Path]:
    grid = _create_video_grid(
        frames, gt_actions, pred_actions, cols=cols, text_h=text_h
    )
    png_path = output_stem.parent / f"{output_stem.name}.png"
    json_path = output_stem.parent / f"{output_stem.name}.json"
    grid.save(png_path)
    payload = build_visualization_payload(
        video_id=video_id,
        frames=frames,
        gt_actions=gt_actions,
        pred_actions=pred_actions,
        windows=windows,
        selected_mask=selected_mask,
        correct=correct,
        total=total,
        cols=cols,
        text_h=text_h,
    )
    json_path.write_text(json.dumps(payload, indent=2))
    return png_path, json_path


def _coerce_message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                    continue
                if isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
        return "\n".join(x for x in text_parts if x)
    return str(content)


def _effective_instruction_text(backend: str, instruction_text: str) -> str:
    if backend == "sglang" and instruction_text.strip() == DEFAULT_INSTRUCTION:
        return DEFAULT_SGLANG_INSTRUCTION
    return instruction_text


class IDMEvaluator:
    def __init__(
        self,
        backend: str = "sglang",
        sglang_url: str = "http://localhost:30000",
        hf_model_path: str = "",
        hf_model_state_pkl: str = "",
        hf_attn_implementation: str = "auto",
        hf_precision: str = "bf16",
        hf_device: str = "cuda",
        config: SlidingWindowConfig | None = None,
        instruction_text: str = DEFAULT_INSTRUCTION,
        model: str = "default",
        video_fps: float = 30.0,
        max_tokens: int = 1024,
        max_frames_per_request: int = 16,
        request_timeout_s: int = 120,
    ):
        self.backend = backend
        self.config = config or SlidingWindowConfig()
        self.instruction_text = _effective_instruction_text(
            backend=backend,
            instruction_text=instruction_text,
        )
        self.model = model
        self.video_fps = float(video_fps)
        self.max_tokens = max_tokens
        self.max_frames_per_request = max_frames_per_request
        self.request_timeout_s = request_timeout_s
        self.debug_raw = os.getenv("IDM_EVAL_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.client = None
        self.hf_processor = None
        self.hf_model = None
        self.hf_device = torch.device(hf_device)
        self.hf_precision = hf_precision

        if self.backend == "sglang":
            import openai

            base_url = sglang_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            self.client = openai.OpenAI(base_url=base_url, api_key="dummy")
        elif self.backend == "hf":
            if self.hf_device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "backend=hf with --hf-device cuda requires visible CUDA devices."
                )
            hf_dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }[hf_precision]
            model_kwargs: dict[str, Any] = {
                "torch_dtype": hf_dtype,
                "trust_remote_code": True,
            }
            if hf_attn_implementation != "auto":
                model_kwargs["attn_implementation"] = hf_attn_implementation
            self.hf_processor = AutoProcessor.from_pretrained(
                hf_model_path,
                trust_remote_code=True,
            )
            self.hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
                hf_model_path,
                **model_kwargs,
            ).to(self.hf_device)
            if hf_model_state_pkl.strip():
                state_path = Path(hf_model_state_pkl)
                if not state_path.exists():
                    raise FileNotFoundError(
                        f"--hf-model-state-pkl not found: {state_path}"
                    )
                with open(state_path, "rb") as f:
                    state_d = pickle.load(f)
                missing_L, unexpected_L = self.hf_model.load_state_dict(
                    state_d, strict=False
                )
                if missing_L or unexpected_L:
                    print(
                        "Warning: state dict mismatch on non-strict load: "
                        f"missing={len(missing_L)} unexpected={len(unexpected_L)}"
                    )
            self.hf_model.eval()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _video_input(self, frames_SHWC: np.ndarray) -> list[Any]:
        return [frames_SHWC[i] for i in range(frames_SHWC.shape[0])]

    def _video_metadata(self, frames_SHWC: np.ndarray) -> dict[str, Any]:
        return {
            "total_num_frames": int(frames_SHWC.shape[0]),
            "fps": float(self.video_fps),
            "frames_indices": list(range(int(frames_SHWC.shape[0]))),
        }

    def _build_sglang_content(self, frames: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "type": "video_url",
                "video_url": {"url": _frames_to_mp4_data_uri(frames, self.video_fps)},
            },
            {"type": "text", "text": self.instruction_text},
        ]

    def _predict_segment_single_call_sglang(self, frames: np.ndarray) -> list[str]:
        if self.client is None:
            raise RuntimeError("SGLang client is not initialized.")
        content = self._build_sglang_content(frames)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=min(self.max_tokens, max(64, int(len(frames) * 12))),
            temperature=0.0,
            timeout=float(self.request_timeout_s),
        )
        raw_content = response.choices[0].message.content
        content_text = _coerce_message_content_to_text(raw_content)
        parsed = parse_frame_actions(content_text, expected_n=len(frames))

        if self.debug_raw:
            preview = (
                content_text
                if len(content_text) <= 6000
                else content_text[:6000] + "\n...[truncated]"
            )
            non_empty = sum(1 for x in parsed if x.strip())
            print(
                "[IDM_EVAL_DEBUG] content_type="
                f"{type(raw_content).__name__} parsed_non_empty={non_empty}/{len(parsed)}"
            )
            print("[IDM_EVAL_DEBUG] raw_response_begin")
            print(preview)
            print("[IDM_EVAL_DEBUG] raw_response_end")
        return parsed

    def _predict_segment_single_call_hf(self, frames: np.ndarray) -> list[str]:
        if self.hf_model is None or self.hf_processor is None:
            raise RuntimeError("HF backend is not initialized.")

        prompt_msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": self.instruction_text},
                ],
            }
        ]
        prompt_text = self.hf_processor.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_kwargs: dict[str, Any] = {
            "text": [prompt_text],
            "videos": [self._video_input(frames)],
            "padding": True,
            "return_tensors": "pt",
            "video_metadata": [self._video_metadata(frames)],
        }
        enc_d = self.hf_processor(**prompt_kwargs)
        enc_d.pop("token_type_ids", None)
        prompt_len_i = int(enc_d["attention_mask"].sum(dim=1).item())

        model_inputs_d: dict[str, Any] = {}
        for key_s, val in enc_d.items():
            if isinstance(val, torch.Tensor):
                model_inputs_d[key_s] = val.to(self.hf_device)
            else:
                model_inputs_d[key_s] = val

        gen_kwargs = {
            "max_new_tokens": min(self.max_tokens, max(64, int(len(frames) * 12))),
            "do_sample": False,
            "use_cache": True,
        }
        tokenizer = self.hf_processor.tokenizer
        pad_id = getattr(tokenizer, "pad_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = int(pad_id)
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = int(eos_id)

        with torch.no_grad():
            if self.hf_device.type == "cuda" and self.hf_precision in {"bf16", "fp16"}:
                amp_dtype = (
                    torch.bfloat16 if self.hf_precision == "bf16" else torch.float16
                )
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    generated_ids_BS = self.hf_model.generate(
                        **model_inputs_d,
                        **gen_kwargs,
                    )
            else:
                generated_ids_BS = self.hf_model.generate(
                    **model_inputs_d,
                    **gen_kwargs,
                )

        pred_ids_L = [
            int(x) for x in generated_ids_BS[0, prompt_len_i:].detach().cpu().tolist()
        ]
        content_text = str(tokenizer.decode(pred_ids_L, skip_special_tokens=True))
        parsed = parse_frame_actions(content_text, expected_n=len(frames))

        if self.debug_raw:
            preview = (
                content_text
                if len(content_text) <= 6000
                else content_text[:6000] + "\n...[truncated]"
            )
            non_empty = sum(1 for x in parsed if x.strip())
            print(
                "[IDM_EVAL_DEBUG] backend=hf "
                f"parsed_non_empty={non_empty}/{len(parsed)}"
            )
            print("[IDM_EVAL_DEBUG] raw_response_begin")
            print(preview)
            print("[IDM_EVAL_DEBUG] raw_response_end")
        return parsed

    def _predict_segment_single_call(self, frames: np.ndarray) -> list[str]:
        if self.backend == "sglang":
            return self._predict_segment_single_call_sglang(frames)
        if self.backend == "hf":
            return self._predict_segment_single_call_hf(frames)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def predict_segment(self, frames: np.ndarray) -> list[str]:
        n_frames = int(len(frames))
        if n_frames <= 0:
            return []

        if n_frames > self.max_frames_per_request:
            merged: list[str] = []
            for start in range(0, n_frames, self.max_frames_per_request):
                merged.extend(
                    self.predict_segment(
                        frames[start : start + self.max_frames_per_request]
                    )
                )
            return (merged + [""] * n_frames)[:n_frames]

        try:
            return self._predict_segment_single_call(frames)
        except Exception as exc:
            msg = str(exc).lower()
            cache_bust_signals = (
                "embedding cache is full",
                "vlm cache",
                "cache size",
                "multimodal embedding cache",
                "out of memory",
                "timed out",
                "timeout",
                "connection error",
                "connect error",
            )
            if n_frames > 1 and any(sig in msg for sig in cache_bust_signals):
                mid = max(1, n_frames // 2)
                left = self.predict_segment(frames[:mid])
                right = self.predict_segment(frames[mid:])
                merged = left + right
                return (merged + [""] * n_frames)[:n_frames]
            raise

    def _evaluate_video_with_details(
        self, frames: np.ndarray, gt_actions: list[str]
    ) -> VideoEvalResult:
        n_actions = len(gt_actions)
        windows = compute_windows(n_frames=len(frames), cfg=self.config)
        if n_actions <= 0:
            return VideoEvalResult(
                correct=0,
                total=0,
                predictions=[],
                windows=windows,
                selected_mask=[],
            )

        predictions: list[str | None] = [None] * n_actions
        selected_mask = [False] * n_actions
        for start, end, use_start, use_end in windows:
            segment_preds = self.predict_segment(frames[start:end])
            for local_i in range(use_start, min(use_end, len(segment_preds))):
                global_i = start + local_i
                if global_i < n_actions and predictions[global_i] is None:
                    predictions[global_i] = segment_preds[local_i]
                    selected_mask[global_i] = True

        prediction_text = [p or "" for p in predictions]
        correct = sum(
            normalize_action(pred) == normalize_action(gt)
            for pred, gt in zip(prediction_text, gt_actions)
        )
        return VideoEvalResult(
            correct=correct,
            total=n_actions,
            predictions=prediction_text,
            windows=windows,
            selected_mask=selected_mask,
        )

    def evaluate_video(
        self, frames: np.ndarray, gt_actions: list[str]
    ) -> tuple[int, int, list[str]]:
        result = self._evaluate_video_with_details(frames, gt_actions)
        return result.correct, result.total, result.predictions

    def evaluate(
        self,
        data_iter: Iterator[tuple[str, np.ndarray, list[str]]],
        max_videos: int | None = None,
        visualize: bool = False,
        output_dir: Path | None = None,
    ) -> EvalMetrics:
        metrics = EvalMetrics()
        vis_dir = None
        if visualize:
            vis_dir = (output_dir or Path("./eval_results")) / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)

        try:
            from tqdm import tqdm
        except ModuleNotFoundError:
            tqdm = None  # type: ignore[assignment]

        if tqdm is not None:
            it = enumerate(tqdm(data_iter, desc="Evaluating"))
        else:
            it = enumerate(data_iter)

        for i, (video_id, frames, actions) in it:
            if max_videos is not None and i >= max_videos:
                break
            video_result = self._evaluate_video_with_details(frames, actions)
            metrics.correct += video_result.correct
            metrics.total += video_result.total
            metrics.per_video.append(
                video_result.correct / video_result.total if video_result.total else 0.0
            )
            metrics.all_predictions.extend(video_result.predictions)
            metrics.all_ground_truth.extend(actions)

            if vis_dir is not None:
                acc = (
                    video_result.correct / video_result.total
                    if video_result.total
                    else 0.0
                )
                safe_vid = "".join(c if c.isalnum() else "_" for c in video_id)
                save_visualization_artifacts(
                    output_stem=vis_dir / f"{safe_vid}_acc{acc:.2f}",
                    video_id=video_id,
                    frames=frames,
                    gt_actions=actions,
                    pred_actions=video_result.predictions,
                    windows=video_result.windows,
                    selected_mask=video_result.selected_mask,
                    correct=video_result.correct,
                    total=video_result.total,
                )
        return metrics


def load_split_videos(
    data_root: str,
    split: str,
    image_h: int,
    image_w: int,
    image_c: int = 3,
    max_videos: int | None = None,
    sample_seed: int = 0,
) -> Iterator[tuple[str, np.ndarray, list[str]]]:
    from array_record.python.array_record_module import ArrayRecordReader

    paths_L = find_array_record_paths(data_root, split)
    sampled_record_idx_d: dict[str, list[int]] | None = None
    if max_videos is not None:
        record_count_L: list[int] = []
        for path_s in paths_L:
            reader = ArrayRecordReader(path_s)
            record_count_L.append(reader.num_records())
            reader.close()
        sampled_record_idx_d = _sample_record_indices_by_path(
            paths_L=paths_L,
            record_count_L=record_count_L,
            sample_size=max_videos,
            seed=sample_seed,
        )

    frame_size = image_h * image_w * image_c
    for path_s in paths_L:
        reader = ArrayRecordReader(path_s)
        if sampled_record_idx_d is None:
            record_idx_iter = range(reader.num_records())
        else:
            record_idx_iter = sampled_record_idx_d.get(path_s, [])

        for i in record_idx_iter:
            raw = reader.read([i])[0]
            rec = pickle.loads(raw)
            actions = rec.get("actions")
            if not actions:
                continue
            actions_list = list(actions)
            raw_video = rec.get("raw_video", b"")
            if (
                not isinstance(raw_video, (bytes, bytearray))
                or len(raw_video) < frame_size
            ):
                continue
            n_frames = len(raw_video) // frame_size
            if n_frames <= 0:
                continue
            video = np.frombuffer(raw_video[: n_frames * frame_size], dtype=np.uint8)
            try:
                frames = video.reshape(n_frames, image_h, image_w, image_c)
            except ValueError:
                continue
            n_actions = len(actions_list)
            if n_actions <= 0:
                continue
            if n_frames > n_actions:
                # Align dense video frames to action timestep granularity.
                frame_idx = np.linspace(0, n_frames - 1, num=n_actions, dtype=np.int64)
                frames = frames[frame_idx]
            elif n_actions > n_frames:
                actions_list = actions_list[:n_frames]
            video_id = str(rec.get("video_id", rec.get("path", f"{path_s}:{i}")))
            yield video_id, frames, actions_list
        reader.close()


def _to_device(
    batch_d: dict[str, Any],
    device: torch.device,
    skip_keys: set[str],
) -> dict[str, Any]:
    out_d: dict[str, Any] = {}
    for key_s, val in batch_d.items():
        if key_s in skip_keys:
            continue
        if isinstance(val, torch.Tensor):
            out_d[key_s] = val.to(device, non_blocking=True)
        else:
            out_d[key_s] = val
    return out_d


def _weighted_causal_lm_loss(
    logits_BSV: torch.Tensor,
    labels_BS: torch.Tensor,
    label_weights_BS: torch.Tensor | None = None,
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


def evaluate_apples_to_apples(
    *,
    args: Args,
    evaluator: IDMEvaluator,
) -> dict[str, Any]:
    if (
        evaluator.backend != "hf"
        or evaluator.hf_model is None
        or evaluator.hf_processor is None
    ):
        raise RuntimeError("apples_to_apples mode requires an initialized HF backend.")

    from idm.utils.collator import VideoSFTCollator
    from idm.utils.data import get_dataloader

    val_paths = find_array_record_paths(args.data_root, split=args.split)
    val_loader = get_dataloader(
        array_record_paths=val_paths,
        seq_len=args.seq_len,
        global_batch_size=args.train_style_global_batch_size,
        image_h=args.image_h,
        image_w=args.image_w,
        image_c=args.image_c,
        rank=0,
        world_size=1,
        seed=args.sample_seed,
        epoch_i=0,
        num_epochs=None,
        num_workers=args.train_style_num_workers,
        prefetch_buffer_size=args.train_style_prefetch_buffer_size,
        read_num_threads=args.train_style_read_num_threads,
        worker_buffer_size=args.train_style_worker_buffer_size,
        min_action_density=0.0,
        action_upsample_random_fraction=1.0,
    )
    val_it = iter(val_loader)
    for _ in range(int(args.train_style_skip_val_batches)):
        try:
            next(val_it)
        except StopIteration as exc:
            raise ValueError(
                "Validation loader exhausted while applying --train-style-skip-val-batches."
            ) from exc

    collator = VideoSFTCollator(
        processor=evaluator.hf_processor,
        instruction_text=evaluator.instruction_text,
        video_fps=args.video_fps,
        no_op_loss_weight=args.train_style_no_op_loss_weight,
        mouse_loss_weight=args.train_style_mouse_loss_weight,
    )

    model = evaluator.hf_model
    tokenizer = evaluator.hf_processor.tokenizer
    device = evaluator.hf_device
    precision_s = str(evaluator.hf_precision).lower()
    amp_enabled_b = device.type == "cuda" and precision_s in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision_s == "bf16" else torch.float16

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

    model.eval()
    with torch.no_grad():
        for _ in range(int(args.train_style_val_steps)):
            try:
                raw_batch_d = next(val_it)
            except StopIteration as exc:
                raise ValueError(
                    "Validation loader exhausted before --train-style-val-steps. "
                    "Reduce --train-style-val-steps or --train-style-skip-val-batches."
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
                enabled=amp_enabled_b,
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
                "max_new_tokens": int(args.train_style_val_generate_max_new_tokens),
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
                enabled=amp_enabled_b,
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
            (
                target_no_op_i,
                target_mouse_i,
                target_total_i,
            ) = action_type_counts_from_texts(target_text_B)
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
    per_action = per_action_stats_from_actions(
        pred_actions_L=all_predictions_L,
        gt_actions_L=all_ground_truth_L,
    )
    return {
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
        "per_action": per_action,
        "config": {
            "data_root": args.data_root,
            "split": args.split,
            "backend": args.backend,
            "hf_model_path": args.hf_model_path,
            "hf_model_state_pkl": args.hf_model_state_pkl,
            "hf_attn_implementation": args.hf_attn_implementation,
            "hf_precision": args.hf_precision,
            "hf_device": args.hf_device,
            "video_fps": args.video_fps,
            "image_h": args.image_h,
            "image_w": args.image_w,
            "image_c": args.image_c,
            "seq_len": args.seq_len,
            "sample_seed": args.sample_seed,
            "train_style_val_steps": args.train_style_val_steps,
            "train_style_skip_val_batches": args.train_style_skip_val_batches,
            "train_style_val_generate_max_new_tokens": args.train_style_val_generate_max_new_tokens,
            "train_style_global_batch_size": args.train_style_global_batch_size,
            "train_style_num_workers": args.train_style_num_workers,
            "train_style_prefetch_buffer_size": args.train_style_prefetch_buffer_size,
            "train_style_read_num_threads": args.train_style_read_num_threads,
            "train_style_worker_buffer_size": args.train_style_worker_buffer_size,
            "train_style_no_op_loss_weight": args.train_style_no_op_loss_weight,
            "train_style_mouse_loss_weight": args.train_style_mouse_loss_weight,
        },
    }


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = SlidingWindowConfig(
        seq_len=args.seq_len,
        stride=args.stride,
        center_start=args.center_start,
        center_end=args.center_end,
    )
    evaluator = IDMEvaluator(
        backend=args.backend,
        sglang_url=args.sglang_url,
        hf_model_path=args.hf_model_path,
        hf_model_state_pkl=args.hf_model_state_pkl,
        hf_attn_implementation=args.hf_attn_implementation,
        hf_precision=args.hf_precision,
        hf_device=args.hf_device,
        config=cfg,
        instruction_text=args.instruction_text,
        model=args.model,
        video_fps=args.video_fps,
        max_tokens=args.max_tokens,
        max_frames_per_request=args.max_frames_per_request,
        request_timeout_s=args.request_timeout_s,
    )
    if args.eval_mode == "apples_to_apples":
        results = evaluate_apples_to_apples(args=args, evaluator=evaluator)
        val_metrics_d = results["metrics"]
        print(
            "\nApples-to-apples metrics: "
            f"val_action_acc={float(val_metrics_d['val_action_acc']):.4f} "
            f"val_action_acc_keyboard={float(val_metrics_d['val_action_acc_keyboard']):.4f} "
            f"val_f1_tolerant={float(val_metrics_d['val_f1_tolerant']):.4f}"
        )
    else:
        data_iter = load_split_videos(
            data_root=args.data_root,
            split=args.split,
            image_h=args.image_h,
            image_w=args.image_w,
            image_c=args.image_c,
            max_videos=args.max_videos,
            sample_seed=args.sample_seed,
        )
        metrics = evaluator.evaluate(
            data_iter=data_iter,
            max_videos=None,
            visualize=args.visualize,
            output_dir=output_dir,
        )

        print(f"\nAccuracy: {metrics.accuracy:.4f} ({metrics.correct}/{metrics.total})")
        per_action = metrics.get_per_action_accuracy()
        print("\nPer-Action Metrics:")
        print(f"{'Action':<20} {'Support':>8} {'Recall':>8} {'Precision':>8} {'F1':>8}")
        print("-" * 56)
        for action, stats in sorted(per_action.items(), key=lambda x: -x[1]["support"]):
            print(
                f"{action:<20} {stats['support']:>8} {stats['recall']:>8.3f} "
                f"{stats['precision']:>8.3f} {stats['f1']:>8.3f}"
            )

        results = {
            "accuracy": metrics.accuracy,
            "correct": metrics.correct,
            "total": metrics.total,
            "per_video": metrics.per_video,
            "per_action": per_action,
            "config": {
                "data_root": args.data_root,
                "split": args.split,
                "backend": args.backend,
                "sglang_url": args.sglang_url,
                "model": args.model,
                "hf_model_path": args.hf_model_path,
                "hf_model_state_pkl": args.hf_model_state_pkl,
                "hf_attn_implementation": args.hf_attn_implementation,
                "hf_precision": args.hf_precision,
                "hf_device": args.hf_device,
                "video_fps": args.video_fps,
                "seq_len": args.seq_len,
                "stride": args.stride,
                "center_start": args.center_start,
                "center_end": args.center_end,
                "sample_seed": args.sample_seed,
                "max_tokens": args.max_tokens,
                "max_frames_per_request": args.max_frames_per_request,
                "request_timeout_s": args.request_timeout_s,
            },
        }
    results["eval_mode"] = args.eval_mode
    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
