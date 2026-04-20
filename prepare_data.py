#!/usr/bin/env python3
"""Convert raw crowd-cast recordings (video + msgpack keylogs) into training clips.

Each clip is a short sequence of JPEG frames + sparse action annotations in the
same JSON format used by the eval pipeline.
"""

import json
import os
import random
import re
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import msgpack
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Key / button name normalization
# Maps raw crowd-cast event key names (e.g. "KeyA", "MetaLeft", "ShiftRight")
# to the display names used in the eval format (e.g. "A", "Cmd", "Shift").
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

# Keys that modify other keys (Cmd+C, Shift+A) rather than producing their own action.
MODIFIER_KEYS = {"Shift", "Ctrl", "Alt", "AltGr", "Cmd"}


def normalize_key_name(raw: str) -> str:
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


def format_key_with_modifiers(key: str, held_modifiers: set[str]) -> str:
    """Combine held modifiers with a key press, e.g. 'Cmd+C'."""
    if key in MODIFIER_KEYS:
        return key
    parts = []
    for mod in ["Cmd", "Ctrl", "Alt", "Shift"]:
        if mod in held_modifiers:
            parts.append(mod)
    parts.append(key)
    return "+".join(parts)


# ---------------------------------------------------------------------------
# Raw keylog → sparse events
# ---------------------------------------------------------------------------


def parse_keylog_events(entries: list, fps: int, num_frames: int) -> list[dict]:
    """Convert raw msgpack keylog entries to per-frame held-state events.

    Tracks KeyPress/KeyRelease and MousePress/MouseRelease to determine which
    keys/buttons are held on each frame. Emits one event per frame per held
    key/button. Scrolls are emitted as instantaneous events (one per tick).

    Returns list of {"frame_idx": int, "type": str, "details": str}.
    """
    held_modifiers: set[str] = set()

    # Sort by timestamp, then original order
    sortable = []
    for order_i, entry in enumerate(entries):
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        try:
            ts = int(entry[0])
        except (TypeError, ValueError):
            continue
        ev = entry[1]
        if not isinstance(ev, list) or len(ev) < 1:
            continue
        sortable.append((ts, order_i, ev))
    sortable.sort(key=lambda x: (x[0], x[1]))

    # Track held state as spans — keyed by raw key name, storing (frame, detail)
    held_keys: dict[str, tuple[int, str]] = {}  # raw_key -> (press_frame, detail)
    held_buttons: dict[str, int] = {}  # button -> press_frame
    key_spans: list[tuple[int, int, str]] = []  # (start, end, detail)
    button_spans: list[tuple[int, int, str]] = []  # (start, end, button)
    scroll_events: list[dict] = []

    def _release_all(frame: int) -> None:
        """Close all held key/button spans at the given frame."""
        for start, detail in held_keys.values():
            key_spans.append((start, frame, detail))
        held_keys.clear()
        held_modifiers.clear()
        for button, start in held_buttons.items():
            button_spans.append((start, frame, button))
        held_buttons.clear()

    for ts, _, ev in sortable:
        frame_idx = (ts * fps) // 1_000_000
        if frame_idx < 0 or frame_idx >= num_frames:
            continue

        etype = str(ev[0])
        payload = ev[1] if len(ev) > 1 else None

        if etype == "ContextChanged":
            # Release all held keys/buttons when switching to uncaptured app
            ctx = ev[1] if len(ev) > 1 else None
            if isinstance(ctx, list) and "UNCAPTURED" in ctx:
                _release_all(frame_idx)

        elif etype == "KeyPress":
            raw_key = _parse_key_name(payload)
            if raw_key == "UNKNOWN":
                continue
            key = normalize_key_name(raw_key)
            if key in MODIFIER_KEYS:
                held_modifiers.add(key)
                continue
            detail = format_key_with_modifiers(key, held_modifiers)
            if key not in held_keys:
                held_keys[key] = (frame_idx, detail)

        elif etype == "KeyRelease":
            raw_key = _parse_key_name(payload)
            if raw_key == "UNKNOWN":
                continue
            key = normalize_key_name(raw_key)
            if key in MODIFIER_KEYS:
                held_modifiers.discard(key)
                continue
            if key in held_keys:
                start, detail = held_keys.pop(key)
                key_spans.append((start, frame_idx, detail))

        elif etype == "MousePress":
            button = _parse_button(payload)
            if button != "UNKNOWN" and button not in held_buttons:
                held_buttons[button] = frame_idx

        elif etype == "MouseRelease":
            button = _parse_button(payload)
            if button != "UNKNOWN" and button in held_buttons:
                button_spans.append((held_buttons.pop(button), frame_idx, button))

        elif etype == "MouseScroll":
            direction = _parse_scroll_direction(payload)
            if direction:
                scroll_events.append(
                    {
                        "frame_idx": frame_idx,
                        "type": "MouseScroll",
                        "details": direction,
                    }
                )

    # Close unclosed spans at end of clip
    for detail, start in held_keys.items():
        key_spans.append((start, num_frames - 1, detail))
    for button, start in held_buttons.items():
        button_spans.append((start, num_frames - 1, button))

    # Emit per-frame events from spans
    events: list[dict] = []
    for start, end, detail in key_spans:
        for f in range(start, max(end, start + 1)):
            if f >= num_frames:
                break
            events.append({"frame_idx": f, "type": "KeyPress", "details": detail})

    for start, end, button in button_spans:
        for f in range(start, max(end, start + 1)):
            if f >= num_frames:
                break
            events.append({"frame_idx": f, "type": "MouseClick", "details": button})

    events.extend(scroll_events)
    events.sort(key=lambda x: (x["frame_idx"], x["type"], x["details"]))
    return events


def _parse_key_name(payload) -> str:
    if isinstance(payload, list) and len(payload) >= 2:
        return str(payload[1])
    return "UNKNOWN"


def _parse_button(payload) -> str:
    if isinstance(payload, list) and len(payload) >= 1:
        b = str(payload[0])
        if b in ("Left", "Right", "Middle"):
            return b
    return "UNKNOWN"


def _parse_scroll_direction(payload) -> str | None:
    """Extract scroll direction from [dx, dy, ...] payload."""
    if not isinstance(payload, list) or len(payload) < 2:
        return None
    try:
        dx, dy = float(payload[0]), float(payload[1])
    except (TypeError, ValueError):
        return None
    # Vertical scroll takes priority
    if abs(dy) > abs(dx):
        return "down" if dy < 0 else ("up" if dy > 0 else None)
    if abs(dx) > 0:
        return "down" if dx < 0 else "up"
    return None


def coalesce_scroll_events(events: list[dict]) -> list[dict]:
    """Merge consecutive MouseScroll events on adjacent frames into one."""
    if not events:
        return []
    result = []
    i = 0
    while i < len(events):
        ev = events[i]
        if ev["type"] != "MouseScroll":
            result.append(ev)
            i += 1
            continue
        # Start of a scroll gesture — consume consecutive scroll events
        gesture_frame = ev["frame_idx"]
        gesture_direction = ev["details"]
        last_frame = ev["frame_idx"]
        j = i + 1
        while j < len(events):
            nxt = events[j]
            if nxt["type"] != "MouseScroll":
                break
            if nxt["frame_idx"] > last_frame + 1:
                break  # gap > 1 frame
            if nxt["details"] != gesture_direction:
                break  # direction reversal
            last_frame = nxt["frame_idx"]
            j += 1
        result.append(
            {
                "frame_idx": gesture_frame,
                "type": "MouseScroll",
                "details": gesture_direction,
            }
        )
        i = j
    return result


def filter_event_types(events: list[dict], action_types: set[str]) -> list[dict]:
    """Keep only events whose type is in action_types."""
    return [e for e in events if e["type"] in action_types]


def events_to_eval_format(events: list[dict]) -> list[dict]:
    """Convert frame_idx-based events to eval format with 'frame' key."""
    return [
        {"frame": f"F{e['frame_idx']:02d}", "type": e["type"], "details": e["details"]}
        for e in events
    ]


# ---------------------------------------------------------------------------
# Frame extraction and filtering
# ---------------------------------------------------------------------------


def extract_frames(
    video_path: str,
    fps: int,
    output_dir: str,
    resolution: tuple[int, int] | None = None,
    top_bar_fraction: float = 0.0,
) -> list[str]:
    """Decode video to JPEG frames. Returns list of output paths."""
    vf_parts = []
    if top_bar_fraction > 0:
        vf_parts.append(
            f"crop=in_w:in_h*{1 - top_bar_fraction}:0:in_h*{top_bar_fraction}"
        )
    vf_parts.append(f"fps={fps}")
    if resolution:
        w, h = resolution
        vf_parts.append(f"scale={w}:{h}")
    vf = ",".join(vf_parts)

    pattern = os.path.join(output_dir, "frame_%06d.jpg")
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            vf,
            "-q:v",
            "2",
            pattern,
            "-y",
            "-loglevel",
            "error",
        ],
        check=True,
    )
    return sorted(str(p) for p in Path(output_dir).glob("frame_*.jpg"))


def is_black_frame(img: Image.Image, threshold: float = 0.95) -> bool:
    """Check if >threshold fraction of pixels are near-black."""
    arr = np.asarray(img)
    black_pixels = np.all(arr < 15, axis=-1)
    return float(black_pixels.mean()) > threshold


def filter_black_frames(
    frame_paths: list[str],
    events: list[dict],
    threshold: float = 0.95,
) -> tuple[list[str], list[dict]]:
    """Remove black frames and re-index events to match remaining frames."""
    if threshold >= 1.0:
        return frame_paths, events

    keep_mask = []
    for path in frame_paths:
        img = Image.open(path)
        keep_mask.append(not is_black_frame(img, threshold))

    # Build old→new frame index mapping
    old_to_new: dict[int, int] = {}
    new_idx = 0
    for old_idx, keep in enumerate(keep_mask):
        if keep:
            old_to_new[old_idx] = new_idx
            new_idx += 1

    filtered_paths = [p for p, k in zip(frame_paths, keep_mask) if k]
    filtered_events = []
    for ev in events:
        new_frame = old_to_new.get(ev["frame_idx"])
        if new_frame is not None:
            filtered_events.append({**ev, "frame_idx": new_frame})

    return filtered_paths, filtered_events


# ---------------------------------------------------------------------------
# Clip chunking
# ---------------------------------------------------------------------------


def chunk_into_clips(
    frame_paths: list[str],
    events: list[dict],
    clip_length: int,
    clip_stride: int,
) -> list[tuple[list[str], list[dict]]]:
    """Sliding-window chunk frames+events into clips.

    Returns list of (clip_frame_paths, clip_events) where clip_events have
    frame_idx relative to the clip start.
    """
    n = len(frame_paths)
    clips = []
    for start in range(0, max(1, n - clip_length + 1), clip_stride):
        end = min(start + clip_length, n)
        clip_frames = frame_paths[start:end]
        clip_events = []
        for ev in events:
            local_idx = ev["frame_idx"] - start
            if 0 <= local_idx < len(clip_frames):
                clip_events.append({**ev, "frame_idx": local_idx})
        clips.append((clip_frames, clip_events))
    return clips


# ---------------------------------------------------------------------------
# Frame labeling
# ---------------------------------------------------------------------------


def label_frame(img: Image.Image, label: str) -> Image.Image:
    """Burn a text label (e.g. 'F00') into the top-left corner."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12
        )
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 2
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill="black")
    draw.text((pad, pad), label, fill="white", font=font)
    return img


# ---------------------------------------------------------------------------
# Recording discovery
# ---------------------------------------------------------------------------

RECORDING_RE = re.compile(r"^recording_([0-9a-fA-F-]+)_seg(\d+)(_filtered)?\.mp4$")


def discover_recordings(input_dir: Path) -> list[tuple[Path, Path]]:
    """Find all (video, keylog) pairs under input_dir."""
    pairs = []
    for video_path in sorted(input_dir.rglob("recording_*.mp4")):
        match = RECORDING_RE.match(video_path.name)
        if not match:
            continue
        session_id = match.group(1)
        seg_idx = int(match.group(2))
        filtered_suffix = match.group(3) or ""
        keylog_name = f"input_{session_id}_seg{seg_idx:04d}{filtered_suffix}.msgpack"
        keylog_path = video_path.parent.parent / "keylogs" / keylog_name
        if keylog_path.exists():
            pairs.append((video_path, keylog_path))
    return pairs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# Maps --action-types CLI values to the set of event types to include in output.
ALLOWED_ACTION_TYPES = {
    "keyboard": {"KeyPress"},
    "keyboard,click": {"KeyPress", "MouseClick"},
    "keyboard,click,scroll": {"KeyPress", "MouseClick", "MouseScroll"},
    "all": {"KeyPress", "MouseClick", "MouseScroll"},
}


@dataclass
class Args:
    input_dir: str = ""
    output_dir: str = "./data"
    fps: int = 10
    resolution: str = "640x360"
    top_bar_fraction: float = 0.15
    black_threshold: float = 0.95
    clip_length: int = 30
    clip_stride: int = 15
    min_actions: int = (
        1  # minimum actions per clip to keep (clips below this are sampled at keep_empty_ratio)
    )
    keep_empty_ratio: float = (
        0.05  # fraction of below-threshold clips to keep as negative examples
    )
    label_frames: bool = False
    action_types: str = "keyboard,click,scroll"
    train_ratio: float = 0.85
    val_ratio: float = 0.15
    seed: int = 42
    num_workers: int = 1
    max_recordings: int = (
        0  # process only first N recordings (0=all, useful for debugging)
    )


def _parse_resolution(s: str) -> tuple[int, int]:
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid resolution: {s}. Expected WxH, e.g. 640x360")
    return int(parts[0]), int(parts[1])


def _short_clip_id(video_name: str, chunk_idx: int) -> str:
    """Build a short clip id from the recording name."""
    stem = video_name.replace(".mp4", "").replace("recording_", "rec_")
    # Truncate UUID to first 6 chars for brevity
    stem = re.sub(r"([0-9a-f]{6})[0-9a-f-]+", r"\1", stem)
    return f"{stem}_c{chunk_idx:03d}"


def process_recording(
    video_path: Path,
    keylog_path: Path,
    split_dir: Path,
    fps: int,
    resolution: tuple[int, int],
    top_bar_fraction: float,
    black_threshold: float,
    clip_length: int,
    clip_stride: int,
    min_actions: int,
    keep_empty_ratio: float,
    do_label_frames: bool,
    action_type_set: set[str],
) -> list[dict]:
    """Process one recording, returns list of JSONL entries for written clips."""
    # 1. Parse keylog
    keylog_bytes = keylog_path.read_bytes()
    entries = msgpack.unpackb(keylog_bytes, raw=False)

    # 2. Extract frames to tempdir
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = extract_frames(
            str(video_path),
            fps,
            tmpdir,
            resolution=resolution,
            top_bar_fraction=top_bar_fraction,
        )
        if not frame_paths:
            return []

        num_frames = len(frame_paths)

        # 3. Convert keylog → sparse events
        events = parse_keylog_events(entries, fps, num_frames)
        events = filter_event_types(events, action_type_set)
        events = coalesce_scroll_events(events)

        # 4. Filter black frames
        frame_paths, events = filter_black_frames(frame_paths, events, black_threshold)
        if not frame_paths:
            return []

        # 5. Chunk into clips
        clips = chunk_into_clips(frame_paths, events, clip_length, clip_stride)

        # 6. Write clips
        results = []
        for chunk_idx, (clip_frames, clip_events) in enumerate(clips):
            # Filter by action count
            n_actions = len(clip_events)
            if n_actions < min_actions:
                # Below threshold — keep a small fraction as negative examples
                if keep_empty_ratio <= 0 or random.random() > keep_empty_ratio:
                    continue

            clip_id = _short_clip_id(video_path.name, chunk_idx)
            clip_dir = split_dir / clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)

            # Copy/label frames
            for i, src_path in enumerate(clip_frames):
                img = Image.open(src_path)
                if do_label_frames:
                    img = label_frame(img, f"F{i:02d}")
                dst_path = clip_dir / f"F{i:02d}.jpg"
                img.save(dst_path, "JPEG", quality=90)

            # Build eval-format actions
            actions = events_to_eval_format(clip_events)
            entry = {
                "clip_id": clip_id,
                "clip_dir": str(clip_dir.relative_to(split_dir.parent)),
                "num_frames": len(clip_frames),
                "fps": fps,
                "actions": actions,
            }
            results.append(entry)

    return results


@dataclass
class RecordingJob:
    """Picklable job description for parallel processing."""

    video_path: str
    keylog_path: str
    split_dir: str
    fps: int
    resolution: tuple[int, int]
    top_bar_fraction: float
    black_threshold: float
    clip_length: int
    clip_stride: int
    min_actions: int
    keep_empty_ratio: float
    do_label_frames: bool
    action_types: list[str]  # list for pickling (sets aren't picklable)


def _process_one(job: RecordingJob) -> tuple[list[dict], float]:
    """Process one recording. Safe for multiprocessing (catches exceptions, times execution)."""
    t0 = time.monotonic()
    try:
        entries = process_recording(
            Path(job.video_path),
            Path(job.keylog_path),
            Path(job.split_dir),
            fps=job.fps,
            resolution=job.resolution,
            top_bar_fraction=job.top_bar_fraction,
            black_threshold=job.black_threshold,
            clip_length=job.clip_length,
            clip_stride=job.clip_stride,
            min_actions=job.min_actions,
            keep_empty_ratio=job.keep_empty_ratio,
            do_label_frames=job.do_label_frames,
            action_type_set=set(job.action_types),
        )
    except Exception as e:
        print(f"WARNING: Failed to process {job.video_path}: {e}", flush=True)
        entries = []
    return entries, time.monotonic() - t0


def main() -> None:
    args = tyro.cli(Args)
    if not args.input_dir:
        raise ValueError("--input-dir is required")
    if args.action_types not in ALLOWED_ACTION_TYPES:
        raise ValueError(
            f"Invalid --action-types: {args.action_types}. "
            f"Allowed: {list(ALLOWED_ACTION_TYPES.keys())}"
        )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    resolution = _parse_resolution(args.resolution)
    action_types_list = sorted(ALLOWED_ACTION_TYPES[args.action_types])

    # Discover recordings
    recordings = discover_recordings(input_dir)
    if args.max_recordings > 0:
        recordings = recordings[: args.max_recordings]
    if not recordings:
        print(f"No recordings found under {input_dir}")
        return
    print(f"Found {len(recordings)} recordings")

    # Split into train/val
    random.seed(args.seed)
    random.shuffle(recordings)
    n_train = int(len(recordings) * args.train_ratio)
    train_recs = recordings[:n_train]
    val_recs = recordings[n_train:]

    # Process
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    total_clips = 0
    num_failed = 0
    timing_s: list[float] = []
    num_workers = max(1, args.num_workers)

    for split_name, split_recs, split_dir in [
        ("train", train_recs, train_dir),
        ("val", val_recs, val_dir),
    ]:
        jobs = [
            RecordingJob(
                video_path=str(vp),
                keylog_path=str(kp),
                split_dir=str(split_dir),
                fps=args.fps,
                resolution=resolution,
                top_bar_fraction=args.top_bar_fraction,
                black_threshold=args.black_threshold,
                clip_length=args.clip_length,
                clip_stride=args.clip_stride,
                min_actions=args.min_actions,
                keep_empty_ratio=args.keep_empty_ratio,
                do_label_frames=args.label_frames,
                action_types=action_types_list,
            )
            for vp, kp in split_recs
        ]
        entries = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for clip_entries, elapsed in tqdm(
                executor.map(_process_one, jobs),
                total=len(jobs),
                desc=f"Processing {split_name} ({num_workers} workers)",
            ):
                timing_s.append(elapsed)
                if not clip_entries:
                    num_failed += 1
                entries.extend(clip_entries)

        # Write JSONL index
        jsonl_path = split_dir / f"{split_name}.jsonl"
        with open(jsonl_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print(f"{split_name}: {len(entries)} clips from {len(split_recs)} recordings")
        total_clips += len(entries)

    # Metadata
    meta = {
        "input_dir": str(input_dir),
        "fps": args.fps,
        "resolution": args.resolution,
        "top_bar_fraction": args.top_bar_fraction,
        "black_threshold": args.black_threshold,
        "clip_length": args.clip_length,
        "clip_stride": args.clip_stride,
        "action_types": args.action_types,
        "label_frames": args.label_frames,
        "total_clips": total_clips,
        "num_failed_recordings": num_failed,
        "train_recordings": len(train_recs),
        "val_recordings": len(val_recs),
        "seed": args.seed,
    }
    if timing_s:
        meta["timing"] = {
            "total_s": round(sum(timing_s), 1),
            "mean_per_recording_s": round(sum(timing_s) / len(timing_s), 2),
            "median_per_recording_s": round(sorted(timing_s)[len(timing_s) // 2], 2),
            "max_per_recording_s": round(max(timing_s), 2),
        }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Print timing summary
    if timing_s:
        total_t = sum(timing_s)
        mean_t = total_t / len(timing_s)
        median_t = sorted(timing_s)[len(timing_s) // 2]
        print(
            f"\nTiming: {total_t:.1f}s total, {mean_t:.2f}s/recording avg, "
            f"{median_t:.2f}s/recording median, {max(timing_s):.2f}s max"
        )
        est_all = mean_t * 2147  # ~2147 total recordings
        print(
            f"Estimated full dataset ({2147} recordings): {est_all/60:.0f}min = {est_all/3600:.1f}h"
        )

    if num_failed:
        print(f"WARNING: {num_failed} recordings failed (corrupt/unreadable)")
    print(f"Done. {total_clips} total clips written to {output_dir}")


if __name__ == "__main__":
    main()
