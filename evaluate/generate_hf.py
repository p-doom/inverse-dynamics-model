from __future__ import annotations

import argparse
import base64
from collections import Counter
from dataclasses import dataclass, field
import glob
import io
import json
import os
import pickle
from pathlib import Path
import re
from typing import Any, Iterator
import multiprocessing as mp
from copy import deepcopy

import numpy as np

"""
python evaluate/generate_hf.py   --data-root /p/scratch/envcomp/crowd-cast/crowd-cast-2026-02-18/array_records   --split train   --base-model /p/scratch/envcomp/rieger7/Qwen3-VL-2B-Instruct --adapter /p/home/jusers/rieger7/juwels/idm/inverse-dynamics-model/runs/default/checkpoints/step_00000100/adapter/ --attn-implementation "sdpa"  --max-videos 4
"""

DEFAULT_INSTRUCTION = (
    """Given the video frames, output the action text for each frame in order."""
)
GRID_COLS_DEFAULT = 8
GRID_TEXT_H_DEFAULT = 40
_FRAME_ACTION_RE = re.compile(r"^frame\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)


@dataclass
class SlidingWindowConfig:
    seq_len: int = 128
    stride: int = 64
    center_start: int = 32
    center_end: int = 96


@dataclass
class Args:
    num_gpus: int = 4
    data_root: str = ""
    split: str = "val"
    base_model: str = ""
    adapter: str | None = None
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
    visualize: bool = False
    instruction_text: str = DEFAULT_INSTRUCTION
    max_tokens: int = 128
    max_frames_per_request: int = 16
    dtype: str = "bfloat16"
    device: str = "cuda"
    attn_implementation: str = "flash_attention_2"


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
        gt_norm = [normalize_action(a) for a in self.all_ground_truth]
        pred_norm = [normalize_action(a) for a in self.all_predictions]

        gt_counts = Counter(gt_norm)
        pred_counts = Counter(pred_norm)
        correct_counts = Counter(
            gt for gt, pred in zip(gt_norm, pred_norm) if gt == pred
        )

        per_action: dict[str, dict[str, float]] = {}
        for action in sorted(set(gt_norm) | set(pred_norm)):
            tp = correct_counts[action]
            gt_total = gt_counts[action]
            pred_total = pred_counts[action]
            recall = tp / gt_total if gt_total else 0.0
            precision = tp / pred_total if pred_total else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_action[action] = {
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "support": gt_total,
                "predicted_count": pred_total,
            }
        return per_action


@dataclass
class VideoEvalResult:
    correct: int
    total: int
    predictions: list[str]
    windows: list[tuple[int, int, int, int]]
    selected_mask: list[bool]

def worker_fn(rank: int, args: Args, video_paths: list[str]):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    args.device = "cuda:0" 
    
    print(f"[Worker {rank}] Initializing on GPU {rank} with {len(video_paths)} files...")
    
    cfg = SlidingWindowConfig(
        seq_len=args.seq_len, stride=args.stride,
        center_start=args.center_start, center_end=args.center_end
    )
    evaluator = IDMEvaluatorHF(
        model_name_or_path=args.base_model, adapter_path=args.adapter, config=cfg,
        instruction_text=args.instruction_text, max_tokens=args.max_tokens,
        max_frames_per_request=args.max_frames_per_request,
        video_fps=args.video_fps, dtype=args.dtype,
        device="cuda",
        attn_implementation=args.attn_implementation
    )
    evaluator.batch_size = args.num_gpus

    data_iter = load_videos_from_path_list(
        path_list=video_paths,
        image_h=args.image_h, image_w=args.image_w, image_c=args.image_c
    )

    metrics = evaluator.evaluate(
        data_iter=data_iter,
        max_videos=args.max_videos,
        visualize=args.visualize,
        output_dir=Path(args.output_dir) / f"worker_{rank}",
    )

    worker_results = {
        "correct": metrics.correct,
        "total": metrics.total,
        "per_video": metrics.per_video,
        "all_predictions": metrics.all_predictions,
        "all_ground_truth": metrics.all_ground_truth
    }
    with open(Path(args.output_dir) / f"metrics_shard_{rank}.pkl", "wb") as f:
        pickle.dump(worker_results, f)
    print(f"[Worker {rank}] Finished.")


def normalize_action(action_s: str) -> str:
    return action_s.strip().lower()


def parse_frame_actions(text_s: str, expected_n: int | None = None) -> list[str]:
    indexed_actions: dict[int, str] = {}
    max_idx_i = -1
    for raw_line_s in text_s.splitlines():
        line_s = raw_line_s.strip()
        if not line_s:
            continue
        match = _FRAME_ACTION_RE.match(line_s)
        if match is None:
            continue
        frame_idx_i = int(match.group(1))
        action_s = match.group(2).strip()
        indexed_actions[frame_idx_i] = action_s
        max_idx_i = max(max_idx_i, frame_idx_i)

    actions_L = [""] * (max_idx_i + 1) if max_idx_i >= 0 else []
    for frame_idx_i, action_s in indexed_actions.items():
        if frame_idx_i >= 0:
            actions_L[frame_idx_i] = action_s

    if expected_n is None:
        return actions_L
    if expected_n < 0:
        raise ValueError(f"expected_n must be >= 0, got {expected_n}.")
    return (actions_L + [""] * expected_n)[:expected_n]


def _parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Evaluate IDM via HuggingFace Transformers forward pass."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--base-model", default="", help="Base HF model id/path (e.g. Qwen/Qwen3-VL-2B-Instruct)")
    parser.add_argument("--adapter", default="", help="Path to LoRA adapter dir (e.g. .../checkpoints/step_00000100/adapter)")
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--image-h", type=int, default=270)
    parser.add_argument("--image-w", type=int, default=480)
    parser.add_argument("--image-c", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--center-start", type=int, default=32)
    parser.add_argument("--center-end", type=int, default=96)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--instruction-text", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-frames-per-request", type=int, default=16)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model weight dtype.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation passed to from_pretrained.",
    )
    ns = parser.parse_args()
    return Args(
        data_root=ns.data_root,
        split=ns.split,
        base_model=ns.base_model,
        adapter=ns.adapter,
        output_dir=ns.output_dir,
        image_h=ns.image_h,
        image_w=ns.image_w,
        image_c=ns.image_c,
        seq_len=ns.seq_len,
        stride=ns.stride,
        center_start=ns.center_start,
        center_end=ns.center_end,
        max_videos=ns.max_videos,
        visualize=ns.visualize,
        instruction_text=ns.instruction_text,
        max_tokens=ns.max_tokens,
        max_frames_per_request=ns.max_frames_per_request,
        video_fps=ns.video_fps,
        dtype=ns.dtype,
        device=ns.device,
        attn_implementation=ns.attn_implementation,
    )


def _validate_args(args: Args) -> None:
    if not Path(args.data_root).exists():
        raise ValueError(f"--data-root does not exist: {args.data_root}")
    if not args.split.strip():
        raise ValueError("--split cannot be empty.")
    if not args.base_model.strip():
        raise ValueError("--base-model cannot be empty.")
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
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0.")
    if args.max_frames_per_request <= 0:
        raise ValueError("--max-frames-per-request must be > 0.")


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


class IDMEvaluatorHF:
    """IDM evaluator that runs inference via a local HuggingFace model."""

    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: str | None = None,
        config: SlidingWindowConfig | None = None,
        instruction_text: str = DEFAULT_INSTRUCTION,
        max_tokens: int = 128,
        max_frames_per_request: int = 16,
        video_fps: float = 30.0,
        dtype: str = "bfloat16",
        device: str = "cuda",
        attn_implementation: str = "flash_attention_2",
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel

        _dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = _dtype_map.get(dtype, torch.bfloat16)

        print(f"Loading processor from {model_name_or_path} ...")
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        print(f"Loading model from {model_name_or_path} ...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
            device_map=device,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        if adapter_path:
            print(f"Loading LoRA adapter from {adapter_path} ...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
            self.model = self.model.merge_and_unload()
            self.model.config.use_cache = True  
        else:
            print("No adapter path provided, using base model without adapter. Prompt might be unsuited for that model.")
        self.model.eval()
        self.device = device
        self._torch_dtype = torch_dtype
        self.video_fps = video_fps
        self.config = config or SlidingWindowConfig()
        self.instruction_text = instruction_text
        self.max_tokens = max_tokens
        self.max_frames_per_request = max_frames_per_request
        self.debug_raw = os.getenv("IDM_EVAL_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.batch_size = 4

    def _frames_to_pil(self, frames: np.ndarray):
        from PIL import Image

        return [Image.fromarray(frame) for frame in frames]

    def predict_segments_batched(self, segments_list: list[np.ndarray]) -> list[list[str]]:
        import torch
        
        batch_messages = []
        for frames in segments_list:
            batch_messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": self.instruction_text},
                    ],
                }
            ])

        prompts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) 
                for m in batch_messages]
        
        video_metadata = [
            {
                "total_num_frames": len(seg),
                "fps": float(self.video_fps),
                "frames_indices": list(range(len(seg))),
            }
            for seg in segments_list
        ]

        inputs = self.processor(
            text=prompts,
            videos=segments_list,
            video_metadata=video_metadata,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": self.max_tokens,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=self._torch_dtype):
                generated_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        responses = self.processor.batch_decode(
            generated_ids[:, input_len:], 
            skip_special_tokens=True
        )

        return [parse_frame_actions(res, expected_n=len(seg)) 
                for res, seg in zip(responses, segments_list)]


    def _evaluate_video_with_details(
        self, frames: np.ndarray, gt_actions: list[str]
    ) -> VideoEvalResult:
        n_actions = len(gt_actions)
        windows = compute_windows(n_frames=len(frames), cfg=self.config)
        
        # Prepare all segments for this video
        all_segments = [frames[start:end] for start, end, _, _ in windows]
        predictions: list[str | None] = [None] * n_actions
        selected_mask = [False] * n_actions

        # Process in batches
        for i in range(0, len(all_segments), self.batch_size):
            batch = all_segments[i : i + self.batch_size]
            batch_windows = windows[i : i + self.batch_size]
            batch_results = self.predict_segments_batched(batch)

            for segment_preds, (start, end, use_start, use_end) in zip(batch_results, batch_windows):
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
            tqdm = None  

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


def load_videos_from_path_list(
    path_list: list[str],
    image_h: int,
    image_w: int,
    image_c: int = 3,
) -> Iterator[tuple[str, np.ndarray, list[str]]]:
    from array_record.python.array_record_module import ArrayRecordReader

    frame_size = image_h * image_w * image_c
    for path_s in path_list:
        reader = ArrayRecordReader(path_s)
        n_records = reader.num_records()
        for i in range(n_records):
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
                frame_idx = np.linspace(0, n_frames - 1, num=n_actions, dtype=np.int64)
                frames = frames[frame_idx]
            elif n_actions > n_frames:
                actions_list = actions_list[:n_frames]
            video_id = str(rec.get("video_id", rec.get("path", f"{path_s}:{i}")))
            yield video_id, frames, actions_list
        reader.close()


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_paths = find_array_record_paths(args.data_root, args.split)
    num_gpus = args.num_gpus

    if args.max_videos is not None:
        per_gpu = [args.max_videos // num_gpus] * num_gpus
        for i in range(args.max_videos % num_gpus):
            per_gpu[i] += 1
    else:
        per_gpu = [None] * num_gpus

    shards = [all_paths[i::num_gpus] for i in range(num_gpus)]

    processes = []
    mp.set_start_method('spawn', force=True)

    for rank in range(num_gpus):
        worker_args = deepcopy(args)
        worker_args.max_videos = per_gpu[rank]
        p = mp.Process(target=worker_fn, args=(rank, worker_args, shards[rank]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    metrics = EvalMetrics()
    for rank in range(num_gpus):
        shard_path = output_dir / f"metrics_shard_{rank}.pkl"
        if shard_path.exists():
            with open(shard_path, "rb") as f:
                d = pickle.load(f)
                metrics.correct += d["correct"]
                metrics.total += d["total"]
                metrics.per_video.extend(d["per_video"])
                metrics.all_predictions.extend(d["all_predictions"])
                metrics.all_ground_truth.extend(d["all_ground_truth"])

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
            "base_model": args.base_model,
            "adapter": args.adapter,
            "seq_len": args.seq_len,
            "stride": args.stride,
            "center_start": args.center_start,
            "center_end": args.center_end,
            "max_tokens": args.max_tokens,
            "max_frames_per_request": args.max_frames_per_request,
            "video_fps": args.video_fps,
            "dtype": args.dtype,
            "device": args.device,
        },
    }
    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
