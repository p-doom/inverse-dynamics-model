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

import numpy as np


DEFAULT_INSTRUCTION = (
    "Given the video frames, output the action text for each frame in order."
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
    data_root: str = ""
    split: str = "val"
    sglang_url: str = "http://localhost:30000"
    output_dir: str = "./eval_results"
    image_h: int = 90
    image_w: int = 160
    image_c: int = 3
    seq_len: int = 128
    stride: int = 64
    center_start: int = 32
    center_end: int = 96
    max_videos: int | None = None
    visualize: bool = False
    instruction_text: str = DEFAULT_INSTRUCTION
    model: str = "default"
    max_tokens: int = 1024
    max_frames_per_request: int = 16
    request_timeout_s: int = 120


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
    parser = argparse.ArgumentParser(description="Evaluate IDM via SGLang endpoint.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sglang-url", default="http://localhost:30000")
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--image-h", type=int, default=90)
    parser.add_argument("--image-w", type=int, default=160)
    parser.add_argument("--image-c", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--center-start", type=int, default=32)
    parser.add_argument("--center-end", type=int, default=96)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--instruction-text", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--model", default="default")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-frames-per-request", type=int, default=16)
    parser.add_argument("--request-timeout-s", type=int, default=120)
    ns = parser.parse_args()
    return Args(
        data_root=ns.data_root,
        split=ns.split,
        sglang_url=ns.sglang_url,
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
        model=ns.model,
        max_tokens=ns.max_tokens,
        max_frames_per_request=ns.max_frames_per_request,
        request_timeout_s=ns.request_timeout_s,
    )


def _validate_args(args: Args) -> None:
    if not Path(args.data_root).exists():
        raise ValueError(f"--data-root does not exist: {args.data_root}")
    if not args.split.strip():
        raise ValueError("--split cannot be empty.")
    if not args.sglang_url.strip():
        raise ValueError("--sglang-url cannot be empty.")
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
    if args.request_timeout_s <= 0:
        raise ValueError("--request-timeout-s must be > 0.")


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


def _frames_to_base64(frames: np.ndarray) -> list[str]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Pillow is required for frame encoding. Install with `pip install pillow`."
        ) from exc

    out: list[str] = []
    for frame in frames:
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


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


class IDMEvaluator:
    def __init__(
        self,
        sglang_url: str = "http://localhost:30000",
        config: SlidingWindowConfig | None = None,
        instruction_text: str = DEFAULT_INSTRUCTION,
        model: str = "default",
        max_tokens: int = 1024,
        max_frames_per_request: int = 16,
        request_timeout_s: int = 120,
    ):
        import openai

        base_url = sglang_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        self.client = openai.OpenAI(base_url=base_url, api_key="dummy")
        self.config = config or SlidingWindowConfig()
        self.instruction_text = instruction_text
        self.model = model
        self.max_tokens = max_tokens
        self.max_frames_per_request = max_frames_per_request
        self.request_timeout_s = request_timeout_s
        self.debug_raw = os.getenv("IDM_EVAL_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _predict_segment_single_call(self, frames: np.ndarray) -> list[str]:
        content: list[dict[str, Any]] = []
        for b64 in _frames_to_base64(frames):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        content.append({"type": "text", "text": self.instruction_text})

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
) -> Iterator[tuple[str, np.ndarray, list[str]]]:
    from array_record.python.array_record_module import ArrayRecordReader

    frame_size = image_h * image_w * image_c
    for path_s in find_array_record_paths(data_root, split):
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
                # Align dense video frames to action timestep granularity.
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

    cfg = SlidingWindowConfig(
        seq_len=args.seq_len,
        stride=args.stride,
        center_start=args.center_start,
        center_end=args.center_end,
    )
    evaluator = IDMEvaluator(
        sglang_url=args.sglang_url,
        config=cfg,
        instruction_text=args.instruction_text,
        model=args.model,
        max_tokens=args.max_tokens,
        max_frames_per_request=args.max_frames_per_request,
        request_timeout_s=args.request_timeout_s,
    )
    data_iter = load_split_videos(
        data_root=args.data_root,
        split=args.split,
        image_h=args.image_h,
        image_w=args.image_w,
        image_c=args.image_c,
    )
    metrics = evaluator.evaluate(
        data_iter=data_iter,
        max_videos=args.max_videos,
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
            "sglang_url": args.sglang_url,
            "model": args.model,
            "seq_len": args.seq_len,
            "stride": args.stride,
            "center_start": args.center_start,
            "center_end": args.center_end,
            "max_tokens": args.max_tokens,
            "max_frames_per_request": args.max_frames_per_request,
            "request_timeout_s": args.request_timeout_s,
        },
    }
    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
