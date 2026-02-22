from __future__ import annotations

import base64
import io
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

DEFAULT_INSTRUCTION = (
    "Given the video frames, output the action text for each frame in order."
)


@dataclass
class SlidingWindowConfig:
    seq_len: int = 128
    stride: int = 64
    center_start: int = 32
    center_end: int = 96


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

    def get_confusion_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Returns confusion matrix and list of action labels."""
        gt_norm = [a.lower().strip() for a in self.all_ground_truth]
        pred_norm = [a.lower().strip() for a in self.all_predictions]
        
        labels = sorted(set(gt_norm) | set(pred_norm))
        label_to_idx = {label: i for i, label in enumerate(labels)}
        
        n_labels = len(labels)
        cm = np.zeros((n_labels, n_labels), dtype=np.int32)
        for gt, pred in zip(gt_norm, pred_norm):
            cm[label_to_idx[gt], label_to_idx[pred]] += 1
        
        return cm, labels

    def get_per_action_accuracy(self) -> dict[str, dict[str, float]]:
        """Returns per-action accuracy, precision, recall, and support."""
        gt_norm = [a.lower().strip() for a in self.all_ground_truth]
        pred_norm = [a.lower().strip() for a in self.all_predictions]
        
        gt_counts = Counter(gt_norm)
        pred_counts = Counter(pred_norm)
        correct_counts = Counter(
            gt for gt, pred in zip(gt_norm, pred_norm) if gt == pred
        )
        
        all_actions = sorted(set(gt_norm) | set(pred_norm))
        per_action = {}
        
        for action in all_actions:
            tp = correct_counts[action]
            gt_total = gt_counts[action]
            pred_total = pred_counts[action]
            
            recall = tp / gt_total if gt_total > 0 else 0.0
            precision = tp / pred_total if pred_total > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_action[action] = {
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "support": gt_total,
                "predicted_count": pred_total,
            }
        
        return per_action


def _frames_to_base64(frames: np.ndarray) -> list[str]:
    result = []
    for frame in frames:
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result.append(base64.b64encode(buf.getvalue()).decode())
    return result


def _create_video_grid(
    frames: np.ndarray,
    gt_actions: list[str],
    pred_actions: list[str],
    cols: int = 8,
) -> Image.Image:
    """Create grid showing frames with GT and predicted actions."""
    n_frames = len(frames)
    h, w = frames.shape[1], frames.shape[2]
    
    # Add padding for text
    text_h = 40
    cell_h = h + text_h
    cell_w = w
    
    rows = (n_frames + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except OSError:
        font = ImageFont.load_default()

    for i, frame in enumerate(frames):
        row, col = i // cols, i % cols
        x, y = col * cell_w, row * cell_h
        
        # Paste frame
        grid.paste(Image.fromarray(frame), (x, y))
        
        # Draw action text (GT vs Pred for frame transitions)
        if i < len(gt_actions):
            gt, pred = gt_actions[i], pred_actions[i] if i < len(pred_actions) else ""
            match = gt.lower().strip() == pred.lower().strip()
            color = (0, 128, 0) if match else (200, 0, 0)
            
            text = f"GT:{gt[:12]}\nP:{pred[:12]}"
            draw.text((x + 2, y + h + 2), text, fill=color, font=font)

    return grid


class IDMEvaluator:
    def __init__(
        self,
        sglang_url: str = "http://localhost:30000",
        config: SlidingWindowConfig | None = None,
        instruction_text: str = DEFAULT_INSTRUCTION,
    ):
        import openai
        self.client = openai.OpenAI(base_url=f"{sglang_url}/v1", api_key="dummy")
        self.config = config or SlidingWindowConfig()
        self.instruction_text = instruction_text

    def predict_segment(self, frames: np.ndarray) -> list[str]:
        content = []
        for b64 in _frames_to_base64(frames):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        content.append({"type": "text", "text": self.instruction_text})

        response = self.client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": content}],
            max_tokens=1024,
            temperature=0,
        )
        return self._parse_actions(response.choices[0].message.content, len(frames) - 1)

    def _parse_actions(self, text: str, expected_n: int) -> list[str]:
        actions = []
        for line in text.strip().split("\n"):
            if ":" in line and line.strip().startswith("Frame"):
                actions.append(line.split(":", 1)[1].strip())
        return (actions + [""] * expected_n)[:expected_n]

    def _get_windows(self, n_frames: int) -> list[tuple[int, int, int, int]]:
        cfg = self.config
        if n_frames <= cfg.seq_len:
            return [(0, n_frames, 0, n_frames - 1)]

        windows = []
        starts = list(range(0, n_frames - cfg.seq_len, cfg.stride))
        starts.append(n_frames - cfg.seq_len)

        for i, start in enumerate(starts):
            end = start + cfg.seq_len
            if i == 0:
                use_start, use_end = 0, cfg.center_end
            elif i == len(starts) - 1:
                use_start, use_end = cfg.center_start, cfg.seq_len - 1
            else:
                use_start, use_end = cfg.center_start, cfg.center_end
            windows.append((start, end, use_start, use_end))
        return windows

    def evaluate_video(
        self, frames: np.ndarray, gt_actions: list[str]
    ) -> tuple[int, int, list[str]]:
        """Returns (correct, total, predictions)."""
        n_actions = len(frames) - 1
        predictions = [None] * n_actions

        for start, end, use_start, use_end in self._get_windows(len(frames)):
            segment_preds = self.predict_segment(frames[start:end])
            for local_i in range(use_start, min(use_end, len(segment_preds))):
                global_i = start + local_i
                if global_i < n_actions and predictions[global_i] is None:
                    predictions[global_i] = segment_preds[local_i]

        predictions = [p or "" for p in predictions]
        correct = sum(
            p.lower().strip() == g.lower().strip()
            for p, g in zip(predictions, gt_actions)
        )
        return correct, n_actions, predictions

    def evaluate(
        self,
        data_iter: Iterator[tuple[str, np.ndarray, list[str]]],
        max_videos: int | None = None,
        visualize: bool = False,
        output_dir: Path | None = None,
    ) -> EvalMetrics:
        metrics = EvalMetrics()
        
        if visualize:
            vis_dir = (output_dir or Path("./eval_results")) / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)

        for i, (vid, frames, actions) in enumerate(tqdm(data_iter, desc="Evaluating")):
            if max_videos and i >= max_videos:
                break
            
            correct, total, predictions = self.evaluate_video(frames, actions)
            metrics.correct += correct
            metrics.total += total
            metrics.per_video.append(correct / total if total else 0.0)

            metrics.all_predictions.extend(predictions)
            metrics.all_ground_truth.extend(actions)

            if visualize:
                grid = _create_video_grid(frames, actions, predictions)
                acc = correct / total if total else 0.0
                safe_vid = "".join(c if c.isalnum() else "_" for c in vid)
                grid.save(vis_dir / f"{safe_vid}_acc{acc:.2f}.png")
            
            self._log_to_wandb(metrics)

        return metrics

    def _log_to_wandb(self, metrics: EvalMetrics) -> None:
        """Log metrics, confusion matrix, and per-action stats to W&B."""
        import wandb
        
        wandb.log({
            "eval/accuracy": metrics.accuracy,
            "eval/correct": metrics.correct,
            "eval/total": metrics.total,
        })
        
        per_action = metrics.get_per_action_accuracy()
        for action, stats in per_action.items():
            safe_action = action.replace(" ", "_").replace("/", "_")
            wandb.log({
                f"eval/per_action/{safe_action}/recall": stats["recall"],
                f"eval/per_action/{safe_action}/precision": stats["precision"],
                f"eval/per_action/{safe_action}/f1": stats["f1"],
                f"eval/per_action/{safe_action}/support": stats["support"],
            })
        
        # Ensure equal lengths before logging confusion matrix
        gt = list(metrics.all_ground_truth)
        preds = list(metrics.all_predictions)
        min_len = min(len(gt), len(preds))
        gt = gt[:min_len]
        preds = preds[:min_len]
        
        if min_len == 0:
            return
        
        gt_norm = [a.lower().strip() for a in gt]
        pred_norm = [a.lower().strip() for a in preds]
        labels = sorted(set(gt_norm) | set(pred_norm))
        
        try:
            wandb.log({
                "eval/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=gt_norm,
                    preds=pred_norm,
                    class_names=labels,
                    title="Action Confusion Matrix",
                )
            })
        except Exception:
            pass  # skip if wandb can't render it
        
        cm, cm_labels = metrics.get_confusion_matrix()
        if len(cm_labels) > 5:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(max(10, len(cm_labels) * 0.5), max(8, len(cm_labels) * 0.4)))
            
            cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
            
            im = ax.imshow(cm_normalized, cmap="Blues")
            ax.set_xticks(np.arange(len(cm_labels)))
            ax.set_yticks(np.arange(len(cm_labels)))
            ax.set_xticklabels(cm_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(cm_labels, fontsize=8)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Ground Truth")
            ax.set_title("Normalized Confusion Matrix (Recall)")
            fig.colorbar(im)
            plt.tight_layout()
            
            wandb.log({"eval/confusion_matrix_heatmap": wandb.Image(fig)})
            plt.close(fig)
        
        table_data = []
        for action, stats in sorted(per_action.items(), key=lambda x: -x[1]["support"]):
            table_data.append([
                action,
                stats["support"],
                f"{stats['recall']:.3f}",
                f"{stats['precision']:.3f}",
                f"{stats['f1']:.3f}",
            ])
        
        table = wandb.Table(
            columns=["Action", "Support", "Recall", "Precision", "F1"],
            data=table_data,
        )
        wandb.log({"eval/per_action_table": table})


def load_val_videos(
    data_root: str, image_h: int, image_w: int, image_c: int = 3
) -> Iterator[tuple[str, np.ndarray, list[str]]]:
    import pickle

    from array_record.python.array_record_module import ArrayRecordReader
    from idm.data import find_array_record_paths

    frame_size = image_h * image_w * image_c

    for path in find_array_record_paths(data_root, "train"):
        reader = ArrayRecordReader(path)
        n = reader.num_records()
        for i in range(n):
            raw = reader.read([i])[0]
            rec = pickle.loads(raw)
            actions = rec.get("actions")
            if not actions:
                continue
            raw_video = rec["raw_video"]
            n_frames = len(raw_video) // frame_size
            if n_frames == 0:
                continue
            frames = np.frombuffer(raw_video, dtype=np.uint8).reshape(
                n_frames, image_h, image_w, image_c
            )
            video_id = rec.get("video_id", rec.get("path", f"{path}:{i}"))
            yield video_id, frames, actions
        reader.close()