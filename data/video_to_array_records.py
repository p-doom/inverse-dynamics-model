import json
import multiprocessing as mp
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import msgpack
import numpy as np
import tyro
from array_record.python.array_record_module import ArrayRecordWriter
import ffmpeg

"""
This file processes video files by converting them into array records.
It splits videos into chunks of a specified size and saves them in a specified output folder.
The script uses multiprocessing to handle multiple videos concurrently and generates metadata for the processed videos.
"""

RECORDING_RE = re.compile(r"^recording_([0-9a-fA-F-]+)_seg(\d+)\.mp4$")
KEY_NAME_MAP = {
    "Return": "Enter",
    "Escape": "Esc",
    "Backspace": "Backspace",
    "Space": "Space",
    "Tab": "Tab",
    "ShiftLeft": "Shift",
    "ShiftRight": "Shift",
    "ControlLeft": "Ctrl",
    "ControlRight": "Ctrl",
    "Alt": "Alt",
    "AltGr": "AltGr",
    "MetaLeft": "Meta",
    "MetaRight": "Meta",
    "UpArrow": "Up",
    "DownArrow": "Down",
    "LeftArrow": "Left",
    "RightArrow": "Right",
}
BUTTON_NAME_MAP = {
    "Left": "LMB",
    "Right": "RMB",
    "Middle": "MMB",
}
MOUSE_X_QUANT_UNIT_F = 5.0
MOUSE_Y_QUANT_UNIT_F = 4.0
MOUSE_DELTA_CLIP_I = 1000
MOUSE_SCROLL_CLIP_I = 5


@dataclass
class Args:
    input_path: str
    output_path: str
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    target_width: int = 160
    target_height: int = 90
    target_fps: int = 10
    top_bar_fraction: float = 0.15
    black_ratio: float = 0.95
    chunk_size: int = 160
    chunks_per_file: int = 100
    filter_pure_noop_chunks: bool = False
    seed: int = 0
    num_workers: int = 0
    max_videos: int = 0


def _normalize_key_name(name_s: str) -> str:
    if name_s in KEY_NAME_MAP:
        return KEY_NAME_MAP[name_s]
    if name_s.startswith("Key") and len(name_s) == 4:
        return name_s[3:]
    if name_s.startswith("Digit") and len(name_s) == 6:
        return name_s[5:]
    if name_s.startswith("Num") and len(name_s) == 4:
        return name_s[3:]
    if name_s.startswith("Kp") and len(name_s) == 3:
        return "KP" + name_s[2:]
    return name_s


def _is_zero(value_f: float, eps_f: float = 1e-12) -> bool:
    if np.isnan(value_f):
        return False
    return abs(value_f) <= eps_f


def _parse_key_name(payload: object) -> str:
    if isinstance(payload, list) and len(payload) >= 2:
        return _normalize_key_name(str(payload[1]))
    return "UNKNOWN_KEY"


def _normalize_button_name(name_s: str) -> str:
    out_s = BUTTON_NAME_MAP.get(name_s, name_s).strip()
    if not out_s:
        return "UNKNOWN_BUTTON"
    return out_s.replace(" ", "_")


def _parse_button(payload: object) -> str:
    if isinstance(payload, list) and len(payload) >= 1:
        return _normalize_button_name(str(payload[0]))
    return "UNKNOWN_BUTTON"


def _parse_two_floats(payload: object) -> tuple[float, float]:
    if not isinstance(payload, list) or len(payload) < 2:
        return float("nan"), float("nan")
    try:
        return float(payload[0]), float(payload[1])
    except (TypeError, ValueError):
        return float("nan"), float("nan")


def _mouse_scroll_delta(payload: object) -> float:
    sx_f, sy_f = _parse_two_floats(payload)
    if not np.isnan(sy_f) and not _is_zero(sy_f):
        return sy_f
    if not np.isnan(sx_f) and not _is_zero(sx_f):
        return sx_f
    return 0.0


def _quantize_and_clip(
    value_f: float,
    quant_unit_f: float,
    clip_abs_i: int,
) -> int:
    if np.isnan(value_f):
        return 0
    quant_i = int(np.rint(value_f / quant_unit_f))
    return int(np.clip(quant_i, -clip_abs_i, clip_abs_i))


def _format_action(
    mouse_dx_i: int,
    mouse_dy_i: int,
    scroll_i: int,
    pressed_keys_s: set[str],
) -> str:
    keys_L = sorted(pressed_keys_s)
    if mouse_dx_i == 0 and mouse_dy_i == 0 and scroll_i == 0 and not keys_L:
        return "NO_OP"
    mouse_s = f"MOUSE:{mouse_dx_i},{mouse_dy_i},{scroll_i}"
    if not keys_L:
        return mouse_s
    return f"{mouse_s} ; {' '.join(keys_L)}"


def _get_keylog_path(filename: str) -> Path:
    match = RECORDING_RE.match(Path(filename).name)
    assert (
        match is not None
    ), f"Invalid recording filename: {filename}. Expected recording_<session-id>_seg####.mp4"
    session_id_s = match.group(1)
    seg_idx_i = int(match.group(2))
    return (
        Path(filename).parent.parent
        / "keylogs"
        / (f"input_{session_id_s}_seg{seg_idx_i:04d}.msgpack")
    )


def _actions_from_keylog_entries(
    entries: list[object],
    n_frames: int,
    target_fps: int,
) -> list[str]:
    if n_frames <= 0:
        return []

    frame_events_d: dict[int, list[tuple[str, object]]] = {}
    sortable_events_L: list[tuple[int, int, int, object]] = []
    for order_i, entry in enumerate(entries):
        if not isinstance(entry, list) or len(entry) < 2:
            continue

        timestamp = entry[0]
        event_tuple = entry[1]
        try:
            timestamp_i = int(timestamp)
        except (TypeError, ValueError):
            continue

        frame_idx_i = (timestamp_i * target_fps) // 1_000_000
        if frame_idx_i < 0 or frame_idx_i >= n_frames:
            continue
        sortable_events_L.append((timestamp_i, order_i, frame_idx_i, event_tuple))

    sortable_events_L.sort(key=lambda x: (x[0], x[1]))
    for _, _, frame_idx_i, event_tuple in sortable_events_L:
        if not isinstance(event_tuple, list) or len(event_tuple) < 1:
            continue
        event_type_s = str(event_tuple[0])
        payload = event_tuple[1] if len(event_tuple) > 1 else None
        frame_events_d.setdefault(frame_idx_i, []).append((event_type_s, payload))

    actions_L = []
    pressed_keys_s: set[str] = set()
    for frame_idx_i in range(n_frames):
        mouse_dx_f = 0.0
        mouse_dy_f = 0.0
        mouse_scroll_f = 0.0
        for event_type_s, payload in frame_events_d.get(frame_idx_i, []):
            if event_type_s == "KeyPress":
                key_s = _parse_key_name(payload)
                if key_s != "UNKNOWN_KEY":
                    pressed_keys_s.add(key_s)
                continue
            if event_type_s == "KeyRelease":
                key_s = _parse_key_name(payload)
                if key_s != "UNKNOWN_KEY":
                    pressed_keys_s.discard(key_s)
                continue
            if event_type_s == "MousePress":
                button_s = _parse_button(payload)
                if button_s != "UNKNOWN_BUTTON":
                    pressed_keys_s.add(button_s)
                continue
            if event_type_s == "MouseRelease":
                button_s = _parse_button(payload)
                if button_s != "UNKNOWN_BUTTON":
                    pressed_keys_s.discard(button_s)
                continue
            if event_type_s == "MouseMove":
                dx_f, dy_f = _parse_two_floats(payload)
                if not np.isnan(dx_f):
                    mouse_dx_f += dx_f
                if not np.isnan(dy_f):
                    mouse_dy_f += dy_f
                continue
            if event_type_s == "MouseScroll":
                mouse_scroll_f += _mouse_scroll_delta(payload)

        mouse_dx_i = _quantize_and_clip(
            value_f=mouse_dx_f,
            quant_unit_f=MOUSE_X_QUANT_UNIT_F,
            clip_abs_i=MOUSE_DELTA_CLIP_I,
        )
        mouse_dy_i = _quantize_and_clip(
            value_f=mouse_dy_f,
            quant_unit_f=MOUSE_Y_QUANT_UNIT_F,
            clip_abs_i=MOUSE_DELTA_CLIP_I,
        )
        mouse_scroll_i = _quantize_and_clip(
            value_f=mouse_scroll_f,
            quant_unit_f=1.0,
            clip_abs_i=MOUSE_SCROLL_CLIP_I,
        )
        actions_L.append(
            _format_action(
                mouse_dx_i=mouse_dx_i,
                mouse_dy_i=mouse_dy_i,
                scroll_i=mouse_scroll_i,
                pressed_keys_s=pressed_keys_s,
            )
        )
    return actions_L


def _actions_from_keylog_file(
    keylog_path: Path,
    n_frames: int,
    target_fps: int,
) -> list[str]:
    assert keylog_path.exists(), f"Missing keylog file: {keylog_path}"
    payload_b = keylog_path.read_bytes()
    assert payload_b, f"Keylog file is empty: {keylog_path}"
    entries = msgpack.unpackb(payload_b, raw=False)
    assert isinstance(entries, list), f"Keylog is not a msgpack list: {keylog_path}"
    return _actions_from_keylog_entries(
        entries, n_frames=n_frames, target_fps=target_fps
    )


def _failed_result(
    video_info: dict[str, object], skip_reason: str
) -> list[dict[str, object]]:
    return [
        {
            "filename": "",
            "length": 0,
            "source_filename": str(video_info.get("filename", "")),
            "path": str(video_info.get("path", "")),
            "num_chunks_in_file": 0,
            "skip_reason": skip_reason,
        }
    ]


def _chunk_video_records(
    video_tensor,
    video_info: dict[str, object],
    chunk_size: int,
    actions: list[str] | None = None,
    filter_pure_noop_chunks: bool = False,
) -> list[dict[str, bytes | int | list[str] | str]]:
    """
    Split one decoded video into fixed-length chunk records.

    Args:
        chunk_size: Number of frames per video chunk

    Returns:
        Chunk records ready to be serialized into ArrayRecord files
    """
    current_episode_len = video_tensor.shape[0]
    if current_episode_len < chunk_size:
        raise ValueError(
            f"Video has {current_episode_len} frames, need at least {chunk_size}."
        )
    if actions is not None and len(actions) != current_episode_len:
        raise ValueError(
            f"Action length mismatch: len(actions)={len(actions)} != frames={current_episode_len}"
        )

    file_chunks: list[dict[str, bytes | int | list[str] | str]] = []
    for start_idx in range(0, current_episode_len - chunk_size + 1, chunk_size):
        chunk = video_tensor[start_idx : start_idx + chunk_size]
        chunk_actions = (
            actions[start_idx : start_idx + chunk_size] if actions is not None else None
        )
        if (
            filter_pure_noop_chunks
            and chunk_actions is not None
            and all(action_s == "NO_OP" for action_s in chunk_actions)
        ):
            continue

        chunk_record = {
            "raw_video": chunk.tobytes(),
            "sequence_length": chunk_size,
            "path": str(video_info.get("path", "")),
        }
        if chunk_actions is not None:
            chunk_record["actions"] = chunk_actions

        file_chunks.append(chunk_record)
    return file_chunks


def _write_chunk_batch(
    batch_chunks: list[dict[str, bytes | int | list[str] | str]],
    output_folder: str,
    worker_idx: int,
    file_index: int,
) -> dict[str, object]:
    output_filename = f"chunked_videos_w{worker_idx:04d}_{file_index:06d}.array_record"
    output_file = os.path.join(output_folder, output_filename)

    writer = ArrayRecordWriter(output_file, "group_size:1")
    for chunk in batch_chunks:
        writer.write(pickle.dumps(chunk))
    writer.close()

    chunk_len = int(batch_chunks[0]["sequence_length"]) if batch_chunks else 0
    print(f"Created {output_filename} with {len(batch_chunks)} video chunks")
    return {
        "filename": output_file,
        "length": chunk_len,
        "source_filename": "",
        "path": "",
        "num_chunks_in_file": len(batch_chunks),
        "skip_reason": "",
    }


def _write_chunk_records(
    chunk_records: list[dict[str, bytes | int | list[str] | str]],
    output_folder: str,
    chunks_per_file: int,
    worker_idx: int,
    start_file_index: int = 0,
) -> list[dict[str, object]]:
    if not chunk_records:
        return []

    output_rows: list[dict[str, object]] = []
    file_index = start_file_index
    for i in range(0, len(chunk_records), chunks_per_file):
        batch_chunks = chunk_records[i : i + chunks_per_file]
        output_rows.append(
            _write_chunk_batch(
                batch_chunks=batch_chunks,
                output_folder=output_folder,
                worker_idx=worker_idx,
                file_index=file_index,
            )
        )
        file_index += 1
    return output_rows


def preprocess_video(
    idx,
    video_info,
    target_width,
    target_height,
    target_fps,
    chunk_size,
    top_bar_fraction,
    black_ratio,
    filter_pure_noop_chunks=False,
):
    """
    Preprocess a video file by reading it, resizing, changing its frame rate,
    and then chunking it into smaller segments to be saved as ArrayRecord files.

    Args:
        idx (int): Index of the video being processed.
        in_filename (str): Path to the input video file.
        target_width (int): The target width for resizing the video frames.
        target_height (int): The target height for resizing the video frames.
        target_fps (int): The target frames per second for the output video.
        chunk_size (int): Number of frames per chunk.
        top_bar_fraction (float): Top fraction of the frame ignored for black-frame detection.
        black_ratio (float): Fraction of pixels that must be below threshold.

    Returns:
        tuple: (chunk_records, failed_rows)
    """

    in_filename = str(video_info["filename"])
    print(f"Processing video {idx}, Filename: {in_filename}")
    try:
        out, _ = (
            ffmpeg.input(in_filename)
            .filter("fps", fps=target_fps, round="up")
            .filter(
                "scale",
                target_width,
                target_height,
                force_original_aspect_ratio="decrease",
            )
            .filter("pad", target_width, target_height, "(ow-iw)/2", "(oh-ih)/2")
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, quiet=True)
        )

        frame_size = target_height * target_width * 3
        n_frames = len(out) // frame_size
        if n_frames == 0:
            return [], _failed_result(video_info, "no_frames")
        if n_frames < chunk_size:
            print(f"Warning: Video has {n_frames} frames, skipping (need {chunk_size})")
            return [], _failed_result(video_info, "too_short")
        frames = np.frombuffer(out, np.uint8).reshape(
            n_frames, target_height, target_width, 3
        )
        keylog_path = _get_keylog_path(in_filename)
        actions = _actions_from_keylog_file(
            keylog_path=keylog_path,
            n_frames=n_frames,
            target_fps=target_fps,
        )

        segments = _filter_black_frames(
            frames=frames,
            actions=actions,
            black_ratio=black_ratio,
            top_bar_fraction=top_bar_fraction,
        )

        if len(segments) > 1:
            print(f"Split video {idx} into {len(segments)} segments at black frames")

        all_chunk_records = []
        for seg_idx, (seg_frames, seg_actions) in enumerate(segments):
            if len(seg_frames) < chunk_size:
                print(
                    f"Segment {seg_idx} of video {idx} has {len(seg_frames)} frames, skipping (need {chunk_size})"
                )
                continue

            chunk_records = _chunk_video_records(
                video_tensor=seg_frames,
                video_info=video_info,
                chunk_size=chunk_size,
                actions=seg_actions,
                filter_pure_noop_chunks=filter_pure_noop_chunks,
            )
            all_chunk_records.extend(chunk_records)

        if not all_chunk_records:
            return [], _failed_result(video_info, "all_segments_too_short")

        return all_chunk_records, []
    except Exception as e:
        print(f"Error processing video {idx} ({in_filename}): {e}")
        return [], _failed_result(video_info, f"decode_error:{e}")


def _process_video_shard(
    worker_idx: int,
    shard_args: list[tuple[object, ...]],
) -> list[dict[str, object]]:
    if not shard_args:
        return []

    output_folder = str(shard_args[0][2])
    chunks_per_file = int(shard_args[0][7])
    next_file_index = 0
    pending_chunks: list[dict[str, bytes | int | list[str] | str]] = []
    shard_results: list[dict[str, object]] = []

    for video_arg in shard_args:
        idx = int(video_arg[0])
        video_info = video_arg[1]
        target_width = int(video_arg[3])
        target_height = int(video_arg[4])
        target_fps = int(video_arg[5])
        chunk_size = int(video_arg[6])
        top_bar_fraction = float(video_arg[8]) if len(video_arg) > 8 else 0.15
        black_ratio = float(video_arg[9]) if len(video_arg) > 9 else 0.95
        filter_pure_noop_chunks = bool(video_arg[10]) if len(video_arg) > 10 else False

        chunk_records, failed_rows = preprocess_video(
            idx=idx,
            video_info=video_info,
            target_width=target_width,
            target_height=target_height,
            target_fps=target_fps,
            chunk_size=chunk_size,
            top_bar_fraction=top_bar_fraction,
            black_ratio=black_ratio,
            filter_pure_noop_chunks=filter_pure_noop_chunks,
        )
        if failed_rows:
            shard_results.extend(failed_rows)

        pending_chunks.extend(chunk_records)
        flush_n = (len(pending_chunks) // chunks_per_file) * chunks_per_file
        if flush_n <= 0:
            continue

        shard_results.extend(
            _write_chunk_records(
                chunk_records=pending_chunks[:flush_n],
                output_folder=output_folder,
                chunks_per_file=chunks_per_file,
                worker_idx=worker_idx,
                start_file_index=next_file_index,
            )
        )
        next_file_index += flush_n // chunks_per_file
        pending_chunks = pending_chunks[flush_n:]

    if pending_chunks:
        shard_results.extend(
            _write_chunk_records(
                chunk_records=pending_chunks,
                output_folder=output_folder,
                chunks_per_file=chunks_per_file,
                worker_idx=worker_idx,
                start_file_index=next_file_index,
            )
        )

    return shard_results


def _is_nearly_black(
    frame: np.ndarray,
    threshold: float = 10.0,
    black_ratio: float = 0.95,
    top_bar_fraction: float = 0.15,
) -> bool:
    """
    Check if a frame is nearly black.

    Args:
        frame: RGB frame of shape (H, W, 3)
        threshold: Pixel brightness threshold (0-255)
        black_ratio: Fraction of pixels that must be below threshold
        top_bar_fraction: Top fraction of the frame to ignore

    Returns:
        True if frame is nearly black
    """
    crop_fraction = min(max(top_bar_fraction, 0.0), 0.99)
    crop_start = int(frame.shape[0] * crop_fraction)
    brightness = frame[crop_start:].mean(axis=2)
    black_pixels = (brightness < threshold).sum()
    total_pixels = brightness.size
    return (black_pixels / total_pixels) >= black_ratio


def _filter_black_frames(
    frames: np.ndarray,
    actions: list[str] | None,
    threshold: float = 10.0,
    black_ratio: float = 0.95,
    top_bar_fraction: float = 0.15,
) -> list[tuple[np.ndarray, list[str] | None]]:
    """
    Split video at nearly-black frame sequences.

    Args:
        frames: Video frames of shape (N, H, W, 3)
        actions: List of action labels (or None)
        threshold: Pixel brightness threshold
        black_ratio: Fraction of pixels that must be below threshold
        top_bar_fraction: Top fraction of the frame to ignore

    Returns:
        List of (frames, actions) tuples for each non-black segment
    """
    n_frames = len(frames)
    if n_frames == 0:
        return []

    is_black = np.array(
        [
            _is_nearly_black(
                frame=frames[i],
                threshold=threshold,
                black_ratio=black_ratio,
                top_bar_fraction=top_bar_fraction,
            )
            for i in range(n_frames)
        ]
    )

    if not is_black.any():
        return [(frames, actions)]

    segments = []
    start_idx = None

    for i in range(n_frames):
        if not is_black[i]:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                seg_frames = frames[start_idx:i]
                seg_actions = actions[start_idx:i] if actions else None
                segments.append((seg_frames, seg_actions))
                start_idx = None

    if start_idx is not None:
        seg_frames = frames[start_idx:]
        seg_actions = actions[start_idx:] if actions else None
        segments.append((seg_frames, seg_actions))

    return segments


def save_split(pool_args, num_workers: int):
    if not pool_args:
        return []
    num_processes = num_workers if num_workers > 0 else mp.cpu_count()
    num_processes = min(num_processes, len(pool_args))
    print(f"Number of processes: {num_processes}")
    shards = [[] for _ in range(num_processes)]
    for i, pool_arg in enumerate(pool_args):
        shards[i % num_processes].append(pool_arg)
    worker_jobs = [
        (worker_idx, shard_args)
        for worker_idx, shard_args in enumerate(shards)
        if shard_args
    ]

    results = []
    with mp.Pool(processes=num_processes) as pool:
        for shard_result in pool.starmap(_process_video_shard, worker_jobs):
            results.extend(shard_result)
    return results


def _collect_input_videos(
    input_path: str,
) -> list[dict[str, object]]:
    input_root = Path(input_path).resolve()

    videos: list[dict[str, object]] = []
    for path in sorted(input_root.rglob("*.mp4")):
        if not path.is_file():
            continue
        videos.append(
            {
                "filename": str(path),
                "path": str(path.resolve()),
            }
        )
    return videos


def _split_videos(
    input_videos: list[dict[str, object]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, int]]]:
    rng = np.random.default_rng(seed)

    shuffled = list(input_videos)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = round(n_total * train_ratio)
    n_val = round(n_total * val_ratio)
    file_splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    for split, split_videos in file_splits.items():
        for video_info in split_videos:
            video_info["split"] = split
    split_stats = {
        split: {"videos": len(split_videos)}
        for split, split_videos in file_splits.items()
    }
    return file_splits, split_stats


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def main():
    args = tyro.cli(Args)

    print(f"Output path: {args.output_path}")

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    assert np.isclose(total_ratio, 1.0), "Ratios must sum to 1.0"
    assert 0.0 <= args.top_bar_fraction < 1.0, "top_bar_fraction must be in [0, 1)"
    assert 0.0 <= args.black_ratio <= 1.0, "black_ratio must be in [0, 1]"

    print("Converting video to array_record files...")
    input_files = _collect_input_videos(args.input_path)
    assert input_files, f"No input videos found under {args.input_path}"

    if args.max_videos > 0 and len(input_files) > args.max_videos:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(input_files)
        input_files = input_files[: args.max_videos]
        print(f"Truncated input to max_videos={args.max_videos}")

    file_splits, split_stats = _split_videos(
        input_videos=input_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    pool_args = dict()
    for split in file_splits.keys():
        pool_args[split] = []
        os.makedirs(os.path.join(args.output_path, split), exist_ok=True)
        for idx, video_info in enumerate(file_splits[split]):
            pool_args[split].append(
                (
                    idx,
                    video_info,
                    os.path.join(args.output_path, split),
                    args.target_width,
                    args.target_height,
                    args.target_fps,
                    args.chunk_size,
                    args.chunks_per_file,
                    args.top_bar_fraction,
                    args.black_ratio,
                    args.filter_pure_noop_chunks,
                )
            )

    print("\n========= Processing split: train =========\n")
    train_episode_metadata = save_split(pool_args["train"], args.num_workers)
    print("\n========= Processing split: val =========\n")
    val_episode_metadata = save_split(pool_args["val"], args.num_workers)
    print("\n========= Processing split: test =========\n")
    test_episode_metadata = save_split(pool_args["test"], args.num_workers)

    print("Done converting video to array_record files")

    results = train_episode_metadata + val_episode_metadata + test_episode_metadata
    # count the number of short and failed videos
    failed_videos = [result for result in results if result["length"] == 0]
    failed_video_names = {
        result["source_filename"]
        for result in failed_videos
        if result.get("source_filename")
    }
    num_failed_videos = len(failed_video_names)
    num_successful_videos = len(input_files) - num_failed_videos
    total_video_chunks = int(sum(result["num_chunks_in_file"] for result in results))
    print(f"Number of failed videos: {len(failed_videos)}")
    print(f"Number of successful videos: {num_successful_videos}")
    print(f"Number of total files: {len(input_files)}")
    print(f"Number of total chunk files: {len(results)}")
    print(f"Number of total video chunks: {total_video_chunks}")

    metadata = {
        "target_width": args.target_width,
        "target_height": args.target_height,
        "target_channels": 3,
        "target_fps": args.target_fps,
        "top_bar_fraction": args.top_bar_fraction,
        "black_ratio": args.black_ratio,
        "chunk_size": args.chunk_size,
        "filter_pure_noop_chunks": args.filter_pure_noop_chunks,
        "total_chunks": len(results),
        "total_video_chunks": total_video_chunks,
        "total_videos": len(input_files),
        "num_successful_videos": num_successful_videos,
        "num_failed_videos": num_failed_videos,
        "avg_episode_len_train": _safe_mean(
            [float(ep["length"]) for ep in train_episode_metadata]
        ),
        "avg_episode_len_val": _safe_mean(
            [float(ep["length"]) for ep in val_episode_metadata]
        ),
        "avg_episode_len_test": _safe_mean(
            [float(ep["length"]) for ep in test_episode_metadata]
        ),
        "episode_metadata_train": train_episode_metadata,
        "episode_metadata_val": val_episode_metadata,
        "episode_metadata_test": test_episode_metadata,
        "seed": args.seed,
        "num_workers": args.num_workers if args.num_workers > 0 else mp.cpu_count(),
        "split_stats": split_stats,
    }

    with open(os.path.join(args.output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    main()
