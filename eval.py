#!/usr/bin/env python3
"""IDM evaluation: ground-truth processing, scoring, and inference.

Standalone CLI for evaluating any model via OpenAI-compatible API, and library
functions used by train.py for inline eval during training.

Usage (standalone):
    # Local model via sglang/vLLM:
    python eval.py --clips-dir /path/to/clips --api-url http://localhost:30000/v1 \\
        --model-id Qwen/Qwen3-VL-8B-Instruct --fps 2 --tolerance 2 --coalesce

    # OpenAI:
    python eval.py --clips-dir /path/to/clips --api-url https://api.openai.com/v1 \\
        --api-key $OPENAI_API_KEY --model-id gpt-5 --fps 2

    # Gemini (via OpenAI-compatible endpoint):
    python eval.py --clips-dir /path/to/clips \\
        --api-url https://generativelanguage.googleapis.com/v1beta/openai \\
        --api-key $GEMINI_API_KEY --model-id gemini-2.5-flash --fps 2

    # Re-score existing results with different settings:
    python eval.py --rescore results.json --tolerance 5 --coalesce
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from data import build_prompt


# ---------------------------------------------------------------------------
# GT processing — key/button normalization (from score_eval.py)
# ---------------------------------------------------------------------------

KEY_NAME_MAP = {
    "Return": "Return",
    "Escape": "Escape",
    "Backspace": "Backspace",
    "Space": "Space",
    "Tab": "Tab",
    "ShiftLeft": "Shift",
    "ShiftRight": "Shift",
    "ControlLeft": "Ctrl",
    "ControlRight": "Ctrl",
    "Alt": "Alt",
    "AltLeft": "Alt",
    "AltRight": "Alt",
    "AltGr": "AltGr",
    "MetaLeft": "Cmd",
    "MetaRight": "Cmd",
    "UpArrow": "UpArrow",
    "DownArrow": "DownArrow",
    "LeftArrow": "LeftArrow",
    "RightArrow": "RightArrow",
    "CapsLock": "CapsLock",
    "Delete": "Delete",
    "Home": "Home",
    "End": "End",
    "PageUp": "PageUp",
    "PageDown": "PageDown",
}

MODIFIER_KEYS = {"Shift", "Ctrl", "Alt", "AltGr", "Cmd"}


def _normalize_key_name(raw: str) -> str:
    """Map raw key names (e.g. 'KeyA', 'MetaLeft') to display names."""
    if raw in KEY_NAME_MAP:
        return KEY_NAME_MAP[raw]
    if raw.startswith("Key") and len(raw) == 4:
        return raw[3].upper()
    if raw.startswith("Digit") and len(raw) == 6:
        return raw[5]
    if raw.startswith("F") and raw[1:].isdigit():
        return raw  # F1, F2, ...
    return raw


def _format_key_with_modifiers(key: str, held_modifiers: set[str]) -> str:
    if key in MODIFIER_KEYS:
        return key
    parts = []
    for mod in ["Cmd", "Ctrl", "Alt", "Shift"]:
        if mod in held_modifiers:
            parts.append(mod)
    parts.append(key)
    return "+".join(parts)


def _normalize_scroll_direction(params) -> str:
    """Extract scroll direction from raw params [dx, dy, ...] or coalesced dict."""
    if isinstance(params, dict):
        return params.get("direction", "")
    if isinstance(params, list) and len(params) >= 2:
        try:
            dy = float(params[1])
        except (TypeError, ValueError):
            return ""
        return "down" if dy < 0 else ("up" if dy > 0 else "")
    return ""


def _normalize_button(params) -> str:
    """Extract button name from raw params ['Left', 0, 0] or similar."""
    if isinstance(params, list) and len(params) >= 1:
        b = str(params[0])
        if b in ("Left", "Right", "Middle"):
            return b
    return ""


# ---------------------------------------------------------------------------
# GT processing — stateful: emit per-frame events for held keys/buttons
# ---------------------------------------------------------------------------


def filter_gt_actions(
    actions: list, start_s: float, fps: int, num_frames: int | None = None
) -> list[dict]:
    """Convert raw GT events to per-frame held-state events.

    Tracks KeyPress/KeyRelease and MousePress/MouseRelease to determine which
    keys/buttons are held on each frame. Emits one event per frame per held
    key/button. Scrolls are emitted as instantaneous events.

    Returns list of {frame: int, type: str, detail: str}.
    """
    start_us = start_s * 1e6
    held_modifiers: set[str] = set()

    sorted_events = []
    for timestamp_us, (action_type, params) in actions:
        sorted_events.append((timestamp_us, action_type, params))
    sorted_events.sort(key=lambda x: x[0])

    held_keys: dict[str, tuple[int, str]] = {}  # raw_key -> (press_frame, detail)
    held_buttons: dict[str, int] = {}  # button_name -> press_frame

    key_spans: list[tuple[int, int, str]] = []  # (start, end, detail)
    button_spans: list[tuple[int, int, str]] = []  # (start, end, button)
    scroll_events: list[dict] = []

    def _release_all(frame: int) -> None:
        for key, (start, detail) in list(held_keys.items()):
            key_spans.append((start, frame, detail))
        held_keys.clear()
        held_modifiers.clear()
        for button, start in list(held_buttons.items()):
            button_spans.append((start, frame, button))
        held_buttons.clear()

    max_frame = 0

    for timestamp_us, action_type, params in sorted_events:
        relative_us = timestamp_us - start_us
        frame = round(relative_us / 1e6 * fps)
        if frame < 0:
            continue
        max_frame = max(max_frame, frame)

        if action_type == "ContextChanged":
            if isinstance(params, list) and "UNCAPTURED" in params:
                _release_all(frame)

        elif action_type == "KeyPress":
            raw_key = (
                str(params[1])
                if isinstance(params, list) and len(params) >= 2
                else "UNKNOWN"
            )
            if raw_key == "UNKNOWN":
                continue
            key = _normalize_key_name(raw_key)
            if key in MODIFIER_KEYS:
                held_modifiers.add(key)
                continue
            detail = _format_key_with_modifiers(key, held_modifiers)
            if key not in held_keys:
                held_keys[key] = (frame, detail)

        elif action_type == "KeyRelease":
            raw_key = (
                str(params[1])
                if isinstance(params, list) and len(params) >= 2
                else "UNKNOWN"
            )
            if raw_key == "UNKNOWN":
                continue
            key = _normalize_key_name(raw_key)
            if key in MODIFIER_KEYS:
                held_modifiers.discard(key)
                continue
            if key in held_keys:
                start, detail = held_keys.pop(key)
                key_spans.append((start, frame, detail))

        elif action_type == "MousePress":
            button = _normalize_button(params)
            if button and button not in held_buttons:
                held_buttons[button] = frame

        elif action_type == "MouseRelease":
            button = _normalize_button(params)
            if button and button in held_buttons:
                button_spans.append((held_buttons.pop(button), frame, button))

        elif action_type == "MouseScroll":
            direction = _normalize_scroll_direction(params)
            if direction:
                scroll_events.append(
                    {"frame": frame, "type": "MouseScroll", "detail": direction}
                )

    # Close any unclosed spans at end of clip
    clip_end = num_frames - 1 if num_frames else max_frame
    for key, (start, detail) in held_keys.items():
        key_spans.append((start, clip_end, detail))
    for button, start in held_buttons.items():
        button_spans.append((start, clip_end, button))

    # Emit per-frame events from spans
    result = []
    for start, end, detail in key_spans:
        for f in range(start, max(end, start + 1)):
            if num_frames and f >= num_frames:
                break
            result.append({"frame": f, "type": "KeyPress", "detail": detail})

    for start, end, button in button_spans:
        for f in range(start, max(end, start + 1)):
            if num_frames and f >= num_frames:
                break
            result.append({"frame": f, "type": "MouseClick", "detail": button})

    result.extend(scroll_events)
    result.sort(key=lambda x: (x["frame"], x["type"]))
    return result


def coalesce_gt_events(gt_actions: list[dict], gap: int = 1) -> list[dict]:
    """Coalesce consecutive MouseScroll events into gesture-level events.

    Groups scrolls on consecutive frames (within `gap` frames) into single events.
    Splits on direction reversal. Leaves KeyPress and MouseClick untouched.
    """
    scrolls = [a for a in gt_actions if a["type"] == "MouseScroll"]
    others = [a for a in gt_actions if a["type"] != "MouseScroll"]

    if not scrolls:
        return gt_actions

    scrolls.sort(key=lambda x: x["frame"])

    gestures = []
    current_gesture = [scrolls[0]]

    for s in scrolls[1:]:
        prev = current_gesture[-1]
        frame_gap = s["frame"] - prev["frame"]
        direction_flip = s.get("detail", "") != prev.get("detail", "")

        if frame_gap <= gap and not direction_flip:
            current_gesture.append(s)
        else:
            gestures.append(current_gesture)
            current_gesture = [s]
    gestures.append(current_gesture)

    coalesced = []
    for gesture in gestures:
        direction = gesture[0].get("detail", "")
        if not direction:
            continue
        coalesced.append(
            {"frame": gesture[0]["frame"], "type": "MouseScroll", "detail": direction}
        )

    result = others + coalesced
    result.sort(key=lambda x: (x["frame"], x["type"]))
    return result


# ---------------------------------------------------------------------------
# Prediction processing
# ---------------------------------------------------------------------------


def parse_response(text: str) -> list[dict]:
    """Extract JSON action array from model output."""
    cleaned = text.strip()
    # Strip thinking tags
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    if "<think>" in cleaned:
        cleaned = cleaned[: cleaned.index("<think>")]
    cleaned = re.sub(r"<\|begin_of_box\|>", "", cleaned)
    cleaned = re.sub(r"<\|end_of_box\|>", "", cleaned)
    cleaned = cleaned.strip()
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return []


def _parse_pred_frame(frame_label: str) -> int:
    """Parse 'F07' -> 7."""
    m = frame_label.strip().upper()
    if m.startswith("F"):
        try:
            return int(m[1:])
        except ValueError:
            pass
    return -1


PRED_KEY_ALIASES = {
    "enter": "Return",
    "return": "Return",
    "esc": "Escape",
    "escape": "Escape",
    "backspace": "Backspace",
    "delete": "Delete",
    "space": "Space",
    "tab": "Tab",
    "capslock": "CapsLock",
    "uparrow": "UpArrow",
    "up": "UpArrow",
    "downarrow": "DownArrow",
    "down": "DownArrow",
    "leftarrow": "LeftArrow",
    "left": "LeftArrow",
    "rightarrow": "RightArrow",
    "right": "RightArrow",
    "cmd": "Cmd",
    "command": "Cmd",
    "meta": "Cmd",
    "ctrl": "Ctrl",
    "control": "Ctrl",
    "alt": "Alt",
    "option": "Alt",
    "shift": "Shift",
    "home": "Home",
    "end": "End",
    "pageup": "PageUp",
    "pagedown": "PageDown",
}


def _normalize_pred_key(detail: str) -> str:
    """Normalize a predicted key name to match GT conventions."""
    if not detail:
        return detail
    parts = detail.split("+")
    normalized = []
    for part in parts:
        p = part.strip()
        low = p.lower()
        if low in PRED_KEY_ALIASES:
            normalized.append(PRED_KEY_ALIASES[low])
        elif len(p) == 1 and p.isalpha():
            normalized.append(p.upper())
        else:
            normalized.append(p)
    return "+".join(normalized)


def filter_predictions(predictions: list[dict]) -> list[dict]:
    """Parse and normalize predictions to scoring format {frame: int, type: str, detail: str}."""
    valid_types = {"KeyPress", "MouseClick", "MouseScroll"}
    result = []
    for pred in predictions:
        ptype = pred.get("type", "")
        if ptype not in valid_types:
            continue
        frame_idx = _parse_pred_frame(pred.get("frame", ""))
        if frame_idx < 0:
            continue
        detail = str(pred.get("details", pred.get("detail", ""))).strip()
        if ptype == "MouseScroll":
            detail = detail.lower()
            if detail not in ("up", "down"):
                detail = ""
        elif ptype == "KeyPress":
            detail = _normalize_pred_key(detail)
        result.append({"frame": frame_idx, "type": ptype, "detail": detail})
    result.sort(key=lambda x: x["frame"])
    return result


# ---------------------------------------------------------------------------
# Scoring — greedy bipartite matching
# ---------------------------------------------------------------------------


def match_clip(gt: list[dict], preds: list[dict], tolerance: int) -> dict[str, list]:
    """Greedy bipartite matching by type, detail, and frame proximity.

    For each GT action (chronological), find closest unmatched prediction
    of the same type AND detail within +/-tolerance frames.

    Both gt and preds must be in scoring format: {frame: int, type: str, detail: str}.
    """
    used_preds = set()
    matches = []
    unmatched_gt = []

    for gt_action in gt:
        best_idx = None
        best_dist = tolerance + 1
        gt_detail = gt_action.get("detail", "")
        for j, pred in enumerate(preds):
            if j in used_preds:
                continue
            if pred["type"] != gt_action["type"]:
                continue
            if pred.get("detail", "") != gt_detail:
                continue
            dist = abs(pred["frame"] - gt_action["frame"])
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_idx = j
        if best_idx is not None:
            used_preds.add(best_idx)
            matches.append(
                {"gt": gt_action, "pred": preds[best_idx], "frame_dist": best_dist}
            )
        else:
            unmatched_gt.append(gt_action)

    unmatched_preds = [p for j, p in enumerate(preds) if j not in used_preds]
    return {
        "matches": matches,
        "unmatched_gt": unmatched_gt,
        "unmatched_preds": unmatched_preds,
    }


def compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_f1(
    predictions: list[dict], ground_truth: list[dict], tolerance: int = 2
) -> tuple[float, float, float]:
    """Convenience wrapper: compute F1 from training-format actions.

    Accepts actions with string frames ("F05") and "details" (plural) key,
    as used in JSONL training data and model output from parse_response().
    Converts to scoring format internally and delegates to match_clip.

    Returns (precision, recall, f1).
    """
    if not predictions and not ground_truth:
        return 1.0, 1.0, 1.0
    if not predictions or not ground_truth:
        return 0.0, 0.0, 0.0

    def _to_scoring_fmt(actions: list[dict]) -> list[dict]:
        result = []
        for a in actions:
            frame_str = str(a.get("frame", "F00"))
            frame_idx = int(re.sub(r"[^0-9]", "", frame_str)) if frame_str else 0
            detail = str(a.get("details", a.get("detail", "")))
            result.append({"frame": frame_idx, "type": a["type"], "detail": detail})
        return result

    gt = _to_scoring_fmt(ground_truth)
    preds = _to_scoring_fmt(predictions)
    result = match_clip(gt, preds, tolerance)
    tp = len(result["matches"])
    fp = len(result["unmatched_preds"])
    fn = len(result["unmatched_gt"])
    return compute_prf(tp, fp, fn)


# ---------------------------------------------------------------------------
# Frame extraction and labeling
# ---------------------------------------------------------------------------


def discover_eval_clips(clips_dir: str) -> list[dict]:
    """Find all clip mp4/json pairs under clips_dir."""
    clips = []
    for json_path in sorted(Path(clips_dir).rglob("clip_*.json")):
        mp4_path = json_path.with_suffix(".mp4")
        if not mp4_path.exists():
            continue
        with open(json_path) as f:
            meta = json.load(f)
        clips.append(
            {
                "mp4_path": str(mp4_path),
                "clip_name": json_path.stem,
                "start_s": meta["start_s"],
                "end_s": meta["end_s"],
                "tag": meta["tag"],
                "actions": meta["actions"],
            }
        )
    return clips


def extract_frames(mp4_path: str, fps: int) -> list[Image.Image]:
    """Extract frames from mp4 at given fps, return as PIL Images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pattern = os.path.join(tmpdir, "frame_%04d.png")
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                mp4_path,
                "-vf",
                f"fps={fps}",
                pattern,
                "-y",
                "-loglevel",
                "error",
            ],
            check=True,
        )
        frame_paths = sorted(glob.glob(os.path.join(tmpdir, "frame_*.png")))
        return [Image.open(p).convert("RGB") for p in frame_paths]


def label_frames(frames: list[Image.Image]) -> list[Image.Image]:
    """Burn F00, F01, ... labels into top-left corner. Returns new images."""
    labeled = []
    for i, img in enumerate(frames):
        img = img.copy()
        draw = ImageDraw.Draw(img)
        label = f"F{i:02d}"
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14
            )
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
            except (OSError, IOError):
                font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 3
        draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill="black")
        draw.text((pad, pad), label, fill="white", font=font)
        labeled.append(img)
    return labeled


def encode_frames_base64(frames: list[Image.Image]) -> list[str]:
    """Encode PIL Images to base64 PNG strings."""
    import io

    encoded = []
    for img in frames:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded.append(base64.standard_b64encode(buf.getvalue()).decode("ascii"))
    return encoded


# ---------------------------------------------------------------------------
# API inference (OpenAI-compatible — works for sglang, vLLM, OpenAI, Gemini)
# ---------------------------------------------------------------------------


def call_api(
    messages: list[dict],
    api_url: str,
    model_id: str,
    api_key: str = "",
    max_tokens: int = 4096,
    temperature: float = 0,
    extra_body: dict | None = None,
    max_retries: int = 3,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint."""
    from openai import OpenAI

    client = OpenAI(base_url=api_url, api_key=api_key or "dummy")

    for attempt in range(max_retries):
        try:
            kwargs = {}
            if extra_body:
                kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "PerDay" in err or "per day" in err.lower():
                raise
            is_retryable = any(
                code in err
                for code in [
                    "429",
                    "503",
                    "RESOURCE_EXHAUSTED",
                    "UNAVAILABLE",
                    "overloaded",
                ]
            )
            if is_retryable and attempt < max_retries - 1:
                wait = min(2**attempt * 15, 300)
                print(
                    f"  Retry {attempt + 1}/{max_retries} after {wait}s: {err[:80]}..."
                )
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# HF-model inference for inline eval during training
# ---------------------------------------------------------------------------


def run_real_eval(
    model: Any,
    processor: Any,
    eval_clips: list[dict],
    fps: int,
    tolerance: int = 2,
    coalesce: bool = True,
    interleave_labels: bool = False,
    device: Any = None,
    dtype: Any = None,
) -> dict[str, int]:
    """Run eval on real mp4 clips with HF model.generate().

    Returns {tp, fp, fn} counts (caller computes P/R/F1 and handles DDP all-reduce).
    """
    import torch

    model.eval()
    tp_total, fp_total, fn_total = 0, 0, 0

    with torch.no_grad():
        for i, clip in enumerate(eval_clips):
            try:
                frames = extract_frames(clip["mp4_path"], fps)
                if not frames:
                    continue

                prompt = build_prompt(fps, len(frames))

                # Build messages matching training format
                if interleave_labels:
                    content = []
                    for j, f in enumerate(frames):
                        content.append({"type": "text", "text": f"Frame F{j:02d}:"})
                        content.append({"type": "image", "image": f})
                else:
                    content = [{"type": "image", "image": f} for f in frames]
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[text], images=frames, return_tensors="pt", padding=True
                )
                inputs = {
                    k: v.to(device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
                if "pixel_values_videos" in inputs:
                    inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(
                        dtype=dtype
                    )

                with torch.autocast("cuda", dtype=dtype):
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        use_cache=True,
                    )
                prompt_len = inputs["input_ids"].shape[1]
                pred_text = processor.tokenizer.decode(
                    gen_ids[0][prompt_len:], skip_special_tokens=True
                )
                predictions = parse_response(pred_text)

                # Score with proper matching
                gt = filter_gt_actions(clip["actions"], clip["start_s"], fps)
                if coalesce:
                    gt = coalesce_gt_events(gt)
                preds_f = filter_predictions(predictions)
                result = match_clip(gt, preds_f, tolerance=tolerance)
                tp_total += len(result["matches"])
                fp_total += len(result["unmatched_preds"])
                fn_total += len(result["unmatched_gt"])

            except Exception as e:
                print(f"  eval clip {i} error: {e}")
                continue

            torch.cuda.empty_cache()

    model.train()
    return {"tp": tp_total, "fp": fp_total, "fn": fn_total}


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def _build_api_messages(
    frame_b64s: list[str], prompt: str, interleave_labels: bool = False
) -> list[dict]:
    """Build OpenAI-compatible messages with base64 images."""
    content = []
    for i, b64 in enumerate(frame_b64s):
        if interleave_labels:
            content.append({"type": "text", "text": f"Frame F{i:02d}:"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _print_report(
    total_tp: int,
    total_fp: int,
    total_fn: int,
    type_counts: dict,
    clip_details: list[dict],
) -> None:
    """Print evaluation report to stdout."""
    p, r, f1 = compute_prf(total_tp, total_fp, total_fn)
    print(f"\n=== Overall ===")
    print(f"Precision: {p:.2f}  Recall: {r:.2f}  F1: {f1:.2f}")
    print(f"TP={total_tp}  FP={total_fp}  FN={total_fn}")

    print(f"\n=== Per Type ===")
    for t in sorted(type_counts.keys()):
        c = type_counts[t]
        tp_, rp, f1p = compute_prf(c["tp"], c["fp"], c["fn"])
        print(
            f"{t:14s}  P={tp_:.2f}  R={rp:.2f}  F1={f1p:.2f}  "
            f"(TP={c['tp']}, FP={c['fp']}, FN={c['fn']})"
        )

    print(f"\n=== Per Clip ===")
    for cd in clip_details:
        _, _, f1c = compute_prf(cd["tp"], cd["fp"], cd["fn"])
        print(
            f"Clip {cd['clip_index']:02d} [{cd['tag']:20s}]: "
            f"GT={cd['gt_count']:3d}  Pred={cd['pred_count']:3d}  "
            f"TP={cd['tp']:3d}  FP={cd['fp']:3d}  FN={cd['fn']:3d}  F1={f1c:.2f}"
        )


def _score_results(data: dict, tolerance: int, coalesce: bool) -> dict:
    """Score a results JSON (from eval run or --rescore). Returns summary dict."""
    fps = data["fps"]
    total_tp, total_fp, total_fn = 0, 0, 0
    type_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    clip_details = []

    for clip in data["clips"]:
        gt = filter_gt_actions(clip["ground_truth"], clip["start_s"], fps)
        if coalesce:
            gt = coalesce_gt_events(gt)
        preds = filter_predictions(clip.get("predictions", []))
        result = match_clip(gt, preds, tolerance)

        tp = len(result["matches"])
        fp = len(result["unmatched_preds"])
        fn = len(result["unmatched_gt"])

        total_tp += tp
        total_fp += fp
        total_fn += fn

        for m in result["matches"]:
            type_counts[m["gt"]["type"]]["tp"] += 1
        for p in result["unmatched_preds"]:
            type_counts[p["type"]]["fp"] += 1
        for g in result["unmatched_gt"]:
            type_counts[g["type"]]["fn"] += 1

        _, _, clip_f1 = compute_prf(tp, fp, fn)
        clip_details.append(
            {
                "clip_index": clip["clip_index"],
                "clip_name": clip.get("clip_name", ""),
                "tag": clip.get("tag", ""),
                "gt_count": len(gt),
                "pred_count": len(preds),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "f1": round(clip_f1, 4),
            }
        )

    _print_report(total_tp, total_fp, total_fn, dict(type_counts), clip_details)

    p, r, f1 = compute_prf(total_tp, total_fp, total_fn)
    return {
        "tolerance": tolerance,
        "coalesce": coalesce,
        "overall": {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "per_type": {
            t: {
                **c,
                **dict(
                    zip(
                        ["precision", "recall", "f1"],
                        [round(x, 4) for x in compute_prf(c["tp"], c["fp"], c["fn"])],
                    )
                ),
            }
            for t, c in type_counts.items()
        },
        "clips": clip_details,
    }


def main():
    parser = argparse.ArgumentParser(description="IDM evaluation")
    # Eval mode
    parser.add_argument(
        "--clips-dir", help="Root directory containing clip mp4/json pairs"
    )
    parser.add_argument("--api-url", help="OpenAI-compatible API base URL")
    parser.add_argument(
        "--api-key", default="", help="API key (env OPENAI_API_KEY also works)"
    )
    parser.add_argument("--model-id", help="Model name/ID")
    parser.add_argument(
        "--fps", type=int, default=2, help="Frame extraction rate (default: 2)"
    )
    parser.add_argument(
        "--interleave-labels",
        action="store_true",
        help="Insert 'Frame F00:' text before each image",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Max output tokens"
    )
    parser.add_argument(
        "--sleep", type=float, default=1.0, help="Sleep between API calls (seconds)"
    )
    parser.add_argument(
        "--max-clips", type=int, default=None, help="Process only first N clips"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty (for sglang/vLLM)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode (Qwen3 via sglang)",
    )
    # Re-score mode
    parser.add_argument("--rescore", help="Path to existing results JSON to re-score")
    # Scoring settings
    parser.add_argument(
        "--tolerance", type=int, default=2, help="Frame tolerance for matching"
    )
    parser.add_argument(
        "--coalesce",
        action="store_true",
        help="Coalesce consecutive MouseScroll GT events",
    )
    # Output
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    # Re-score mode
    if args.rescore:
        with open(args.rescore) as f:
            data = json.load(f)
        print(
            f"Re-scoring {args.rescore} (tolerance={args.tolerance}, coalesce={args.coalesce})"
        )
        summary = _score_results(data, args.tolerance, args.coalesce)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSaved to {args.output}")
        return

    # Eval mode — validate required args
    if not args.clips_dir or not args.api_url or not args.model_id:
        parser.error(
            "--clips-dir, --api-url, and --model-id are required for eval mode"
        )

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")

    clips = discover_eval_clips(args.clips_dir)
    if not clips:
        print("No clips found!")
        return
    if args.max_clips:
        clips = clips[: args.max_clips]

    # Build extra_body for sglang/vLLM
    extra_body = {}
    if args.repetition_penalty:
        extra_body["repetition_penalty"] = args.repetition_penalty
    if args.thinking:
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = True

    print(
        f"Evaluating {len(clips)} clips: model={args.model_id}, fps={args.fps}, "
        f"tolerance={args.tolerance}, coalesce={args.coalesce}, interleave={args.interleave_labels}"
    )

    results = {
        "model_id": args.model_id,
        "api_url": args.api_url,
        "fps": args.fps,
        "tolerance": args.tolerance,
        "coalesce": args.coalesce,
        "interleave_labels": args.interleave_labels,
        "clips": [],
    }

    for i, clip in enumerate(clips):
        print(f"  [{i + 1}/{len(clips)}] {clip['clip_name']}...", end="", flush=True)

        frames = extract_frames(clip["mp4_path"], args.fps)
        if not frames:
            print(" no frames, skipping")
            continue
        frames = label_frames(frames)
        frame_b64s = encode_frames_base64(frames)

        prompt = build_prompt(args.fps, len(frames))
        messages = _build_api_messages(frame_b64s, prompt, args.interleave_labels)

        temperature = 0.6 if args.thinking else 0
        try:
            raw = call_api(
                messages,
                args.api_url,
                args.model_id,
                api_key=api_key,
                max_tokens=args.max_tokens,
                temperature=temperature,
                extra_body=extra_body or None,
            )
            predictions = parse_response(raw)
        except Exception as e:
            raw = str(e)
            predictions = []
            print(f" ERROR: {e}")
            continue

        clip_result = {
            "clip_index": i,
            "clip_name": clip["clip_name"],
            "tag": clip["tag"],
            "start_s": clip["start_s"],
            "ground_truth": clip["actions"],
            "predictions": predictions,
            "raw_response": raw,
        }
        results["clips"].append(clip_result)

        # Incremental save
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

        print(" done")
        if i < len(clips) - 1:
            time.sleep(args.sleep)

    # Score and print report
    print(f"\n{'=' * 60}")
    summary = _score_results(results, args.tolerance, args.coalesce)

    if args.output:
        results["summary"] = summary
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
