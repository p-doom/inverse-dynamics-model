import json
import multiprocessing as mp
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import tyro
from array_record.python.array_record_module import ArrayRecordWriter

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


@dataclass
class Args:
    input_path: str
    output_path: str
    env_name: str
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    target_width: int = 160
    target_height: int = 90
    target_fps: int = 10
    chunk_size: int = 160
    chunks_per_file: int = 100
    seed: int = 0
    num_workers: int = 0
    max_videos: int = 0


class _StepAccumulator:
    def __init__(self) -> None:
        self.tokens_s = set()
        self.has_active_input_b = False
        self.has_nonzero_motion_b = False
        self.has_nonzero_scroll_b = False
        self.has_zero_passive_event_b = False
        self.has_parse_error_b = False

    def consume(self, event_tuple: object) -> None:
        if not isinstance(event_tuple, list) or len(event_tuple) < 1:
            self.tokens_s.add("MALFORMED_EVENT")
            self.has_parse_error_b = True
            return

        event_type_s = str(event_tuple[0])
        payload = event_tuple[1] if len(event_tuple) > 1 else None

        if event_type_s == "KeyPress":
            self.tokens_s.add(f"KEY_DOWN:{_parse_key_name(payload)}")
            self.has_active_input_b = True
            return
        if event_type_s == "KeyRelease":
            self.tokens_s.add(f"KEY_UP:{_parse_key_name(payload)}")
            self.has_active_input_b = True
            return
        if event_type_s == "MousePress":
            self.tokens_s.add(f"MOUSE_DOWN:{_parse_button(payload)}")
            self.has_active_input_b = True
            return
        if event_type_s == "MouseRelease":
            self.tokens_s.add(f"MOUSE_UP:{_parse_button(payload)}")
            self.has_active_input_b = True
            return
        if event_type_s == "MouseMove":
            dx_f, dy_f = _parse_two_floats(payload)
            if _is_zero(dx_f) and _is_zero(dy_f):
                self.has_zero_passive_event_b = True
            else:
                self.tokens_s.add("MOUSE_MOVE")
                self.has_nonzero_motion_b = True
            return
        if event_type_s == "MouseScroll":
            sx_f, sy_f = _parse_two_floats(payload)
            if _is_zero(sx_f) and _is_zero(sy_f):
                self.has_zero_passive_event_b = True
            else:
                self.tokens_s.add("MOUSE_SCROLL")
                self.has_nonzero_scroll_b = True
            return

        self.tokens_s.add(f"OTHER:{event_type_s}")
        self.has_active_input_b = True

    def label(self) -> str:
        if (
            not self.has_active_input_b
            and not self.has_nonzero_motion_b
            and not self.has_nonzero_scroll_b
            and not self.has_parse_error_b
            and (self.has_zero_passive_event_b or not self.tokens_s)
        ):
            return "NO_OP"
        if self.tokens_s:
            return " + ".join(sorted(self.tokens_s))
        if self.has_parse_error_b:
            return "MALFORMED_STEP"
        return "EMPTY_STEP"


def _normalize_key_name(name_s: str) -> str:
    if name_s in KEY_NAME_MAP:
        return KEY_NAME_MAP[name_s]
    if name_s.startswith("Key") and len(name_s) == 4:
        return name_s[3:]
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


def _parse_button(payload: object) -> str:
    if isinstance(payload, list) and len(payload) >= 1:
        return str(payload[0])
    return "UNKNOWN_BUTTON"


def _parse_two_floats(payload: object) -> tuple[float, float]:
    if not isinstance(payload, list) or len(payload) < 2:
        return float("nan"), float("nan")
    try:
        return float(payload[0]), float(payload[1])
    except (TypeError, ValueError):
        return float("nan"), float("nan")


def _keylog_path_for_video_info(video_info: dict[str, object]) -> Path:
    session_id_s = str(video_info.get("session_id", ""))
    seg_idx_i = int(video_info.get("seg_idx", -1))
    if not session_id_s or seg_idx_i < 0:
        raise ValueError(
            "Cannot derive keylog path from video metadata. "
            "Need `session_id` and `seg_idx`."
        )
    video_path_p = Path(str(video_info["path"]))
    return video_path_p.parent.parent / "keylogs" / (
        f"input_{session_id_s}_seg{seg_idx_i:04d}.msgpack"
    )


def _actions_from_keylog_entries(
    entries: list[object],
    n_frames: int,
    target_fps: int,
) -> list[str]:
    if n_frames <= 0:
        return []

    accumulators_d: dict[int, _StepAccumulator] = {}
    for entry in entries:
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
        acc = accumulators_d.get(frame_idx_i)
        if acc is None:
            acc = _StepAccumulator()
            accumulators_d[frame_idx_i] = acc
        acc.consume(event_tuple)

    actions_L = ["NO_OP"] * n_frames
    for frame_idx_i, acc in accumulators_d.items():
        actions_L[frame_idx_i] = acc.label()
    return actions_L


def _actions_from_keylog_file(
    keylog_path: Path,
    n_frames: int,
    target_fps: int,
) -> list[str]:
    if not keylog_path.exists():
        raise FileNotFoundError(f"Missing keylog file: {keylog_path}")

    payload_b = keylog_path.read_bytes()
    if not payload_b:
        return ["NO_OP"] * n_frames

    entries = msgpack.unpackb(payload_b, raw=False)
    if not isinstance(entries, list):
        raise ValueError(f"Keylog is not a msgpack list: {keylog_path}")
    return _actions_from_keylog_entries(entries, n_frames=n_frames, target_fps=target_fps)


def _failed_result(video_info: dict[str, object], skip_reason: str) -> list[dict[str, object]]:
    return [
        {
            "path": "",
            "length": 0,
            "video_file_name": str(video_info.get("path", "")),
            "relative_path": str(video_info.get("relative_path", "")),
            "user_id": str(video_info.get("user_id", "")),
            "session_id": str(video_info.get("session_id", "")),
            "seg_idx": int(video_info.get("seg_idx", -1)),
            "split": str(video_info.get("split", "")),
            "num_chunks_in_file": 0,
            "skip_reason": skip_reason,
        }
    ]


def _chunk_and_save_video(
    video_tensor,
    video_info: dict[str, object],
    output_folder: str,
    chunk_size: int,
    chunks_per_file: int,
    file_index: int,
    actions: list[str] | None = None,
) -> list[dict[str, object]]:
    """
    Reprocess a single ArrayRecord file by splitting videos into chunks.

    Args:
        video_file_name: Name of the video file
        output_folder: Output folder for the chunked files
        chunk_size: Number of frames per video chunk
        chunks_per_file: Number of video chunks per output file
        file_index: Index for naming output files

    Returns:
        List of paths to created ArrayRecord files
    """
    file_chunks = []
    video_file_name = str(video_info["path"])

    current_episode_len = video_tensor.shape[0]
    if current_episode_len < chunk_size:
        print(
            f"Warning: Video has {current_episode_len} frames, skipping (need {chunk_size})"
        )
        return _failed_result(video_info, "too_short")
    if actions is not None and len(actions) != current_episode_len:
        raise ValueError(
            f"Action length mismatch: len(actions)={len(actions)} != frames={current_episode_len}"
        )

    for start_idx in range(0, current_episode_len - chunk_size + 1, chunk_size):
        chunk = video_tensor[start_idx : start_idx + chunk_size]

        chunk_record = {
            "raw_video": chunk.tobytes(),
            "sequence_length": chunk_size,
            "video_file_name": video_file_name,
            "relative_path": str(video_info.get("relative_path", "")),
            "user_id": str(video_info.get("user_id", "")),
            "session_id": str(video_info.get("session_id", "")),
            "seg_idx": int(video_info.get("seg_idx", -1)),
        }
        if actions is not None:
            chunk_record["actions"] = actions[start_idx : start_idx + chunk_size]

        file_chunks.append(chunk_record)

    # Write chunks to output files
    output_files = []
    for i in range(0, len(file_chunks), chunks_per_file):
        batch_chunks = file_chunks[i : i + chunks_per_file]
        output_filename = (
            f"chunked_videos_{file_index:04d}_{i//chunks_per_file:04d}.array_record"
        )
        output_file = os.path.join(output_folder, output_filename)

        writer = ArrayRecordWriter(output_file, "group_size:1")
        for chunk in batch_chunks:
            writer.write(pickle.dumps(chunk))
        writer.close()

        output_files.append(
            {
                "path": output_file,
                "length": chunk_size,
                "video_file_name": video_file_name,
                "relative_path": str(video_info.get("relative_path", "")),
                "user_id": str(video_info.get("user_id", "")),
                "session_id": str(video_info.get("session_id", "")),
                "seg_idx": int(video_info.get("seg_idx", -1)),
                "split": str(video_info.get("split", "")),
                "num_chunks_in_file": len(batch_chunks),
                "skip_reason": "",
            }
        )
        print(f"Created {output_filename} with {len(batch_chunks)} video chunks")

    print(
        f"Processed {video_file_name}: {len(file_chunks)} chunks -> {len(output_files)} files"
    )
    return output_files


def preprocess_video(
    idx,
    video_info,
    output_path,
    target_width,
    target_height,
    target_fps,
    chunk_size,
    chunks_per_file,
):
    """
    Preprocess a video file by reading it, resizing, changing its frame rate,
    and then chunking it into smaller segments to be saved as ArrayRecord files.

    Args:
        idx (int): Index of the video being processed.
        in_filename (str): Path to the input video file.
        output_path (str): Directory where the output ArrayRecord files will be saved.
        target_width (int): The target width for resizing the video frames.
        target_height (int): The target height for resizing the video frames.
        target_fps (int): The target frames per second for the output video.
        chunk_size (int): Number of frames per chunk.
        chunks_per_file (int): Number of chunks to be saved in each ArrayRecord file.

    Returns:
        list: A list of dictionaries containing metadata about the created ArrayRecord files.
    """

    in_filename = str(video_info["path"])
    print(f"Processing video {idx}, Filename: {in_filename}")
    try:
        import ffmpeg

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
            return _failed_result(video_info, "no_frames")
        frames = np.frombuffer(out, np.uint8).reshape(
            n_frames, target_height, target_width, 3
        )
        keylog_path = _keylog_path_for_video_info(video_info)
        actions = _actions_from_keylog_file(
            keylog_path=keylog_path,
            n_frames=n_frames,
            target_fps=target_fps,
        )

        result = _chunk_and_save_video(
            video_tensor=frames,
            video_info=video_info,
            output_folder=output_path,
            chunk_size=chunk_size,
            chunks_per_file=chunks_per_file,
            file_index=idx,
            actions=actions,
        )
        return result
    except Exception as e:
        print(f"Error processing video {idx} ({in_filename}): {e}")
        return _failed_result(video_info, f"decode_error:{e}")


def save_split(pool_args, num_workers: int):
    if not pool_args:
        return []
    num_processes = num_workers if num_workers > 0 else mp.cpu_count()
    num_processes = min(num_processes, len(pool_args))
    print(f"Number of processes: {num_processes}")
    results = []
    with mp.Pool(processes=num_processes) as pool:
        for result in pool.starmap(preprocess_video, pool_args):
            results.extend(result)
    return results


def _resolve_scan_root(input_path: Path) -> Path:
    uploads = input_path / "uploads"
    if uploads.exists() and uploads.is_dir():
        return uploads
    return input_path


def _extract_user_id(path: Path) -> str:
    parts = path.parts
    if "uploads" in parts:
        idx = parts.index("uploads")
        if idx + 2 < len(parts):
            return parts[idx + 2]
    if len(parts) >= 3 and parts[-2] == "recordings":
        return parts[-3]
    return "UNKNOWN_USER"


def _collect_input_videos(
    input_path: str,
) -> tuple[list[dict[str, object]], list[str], str, bool]:
    input_root = Path(input_path).resolve()
    scan_root = _resolve_scan_root(input_root)

    crowd_cast_videos: list[dict[str, object]] = []
    skipped_nonstandard: list[str] = []
    for path in scan_root.rglob("*.mp4"):
        if path.parent.name != "recordings":
            continue
        match = RECORDING_RE.match(path.name)
        if not match:
            skipped_nonstandard.append(str(path))
            continue
        crowd_cast_videos.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(scan_root)),
                "user_id": _extract_user_id(path),
                "session_id": match.group(1),
                "seg_idx": int(match.group(2)),
                "split": "",
            }
        )

    if crowd_cast_videos:
        return crowd_cast_videos, skipped_nonstandard, str(scan_root), True

    # Fallback for flat directories (legacy jasmine-style input layout).
    flat_videos: list[dict[str, object]] = []
    for in_filename in os.listdir(input_root):
        if not (in_filename.endswith(".mp4") or in_filename.endswith(".webm")):
            continue
        in_path = input_root / in_filename
        if not in_path.is_file():
            continue
        flat_videos.append(
            {
                "path": str(in_path),
                "relative_path": in_filename,
                "user_id": "",
                "session_id": "",
                "seg_idx": -1,
                "split": "",
            }
        )
    return flat_videos, skipped_nonstandard, str(input_root), False


def _split_videos(
    input_videos: list[dict[str, object]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    session_level: bool,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, int]]]:
    rng = np.random.default_rng(seed)

    if not session_level:
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
            split: {"videos": len(split_videos), "sessions": len(split_videos)}
            for split, split_videos in file_splits.items()
        }
        return file_splits, split_stats

    grouped_videos: dict[tuple[str, str], list[dict[str, object]]] = {}
    for video_info in input_videos:
        key = (str(video_info["user_id"]), str(video_info["session_id"]))
        grouped_videos.setdefault(key, []).append(video_info)

    session_keys = list(grouped_videos.keys())
    rng.shuffle(session_keys)

    n_total_sessions = len(session_keys)
    n_train = round(n_total_sessions * train_ratio)
    n_val = round(n_total_sessions * val_ratio)

    split_session_keys = {
        "train": session_keys[:n_train],
        "val": session_keys[n_train : n_train + n_val],
        "test": session_keys[n_train + n_val :],
    }

    file_splits: dict[str, list[dict[str, object]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for split, split_keys in split_session_keys.items():
        for session_key in split_keys:
            session_videos = sorted(
                grouped_videos[session_key],
                key=lambda info: (int(info["seg_idx"]), str(info["path"])),
            )
            for video_info in session_videos:
                video_info["split"] = split
                file_splits[split].append(video_info)

    split_stats = {
        split: {"videos": len(file_splits[split]), "sessions": len(split_keys)}
        for split, split_keys in split_session_keys.items()
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

    print("Converting video to array_record files...")
    input_files, skipped_nonstandard, scan_root, crowd_cast_mode = _collect_input_videos(
        args.input_path
    )
    if not input_files:
        raise ValueError(f"No input videos found under {args.input_path}")

    if args.max_videos > 0 and len(input_files) > args.max_videos:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(input_files)
        input_files = input_files[: args.max_videos]
        print(f"Truncated input to max_videos={args.max_videos}")

    file_splits, split_stats = _split_videos(
        input_videos=input_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        session_level=crowd_cast_mode,
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
                )
            )

    train_episode_metadata = save_split(pool_args["train"], args.num_workers)
    val_episode_metadata = save_split(pool_args["val"], args.num_workers)
    test_episode_metadata = save_split(pool_args["test"], args.num_workers)

    print("Done converting video to array_record files")

    results = train_episode_metadata + val_episode_metadata + test_episode_metadata
    # count the number of short and failed videos
    failed_videos = [result for result in results if result["length"] == 0]
    failed_video_names = {
        result["video_file_name"]
        for result in failed_videos
        if result.get("video_file_name")
    }
    num_failed_videos = len(failed_video_names)
    num_successful_videos = len(input_files) - num_failed_videos
    total_video_chunks = int(sum(result["num_chunks_in_file"] for result in results))
    unique_users = sorted(
        {
            str(video_info["user_id"])
            for video_info in input_files
            if str(video_info.get("user_id", ""))
        }
    )
    unique_sessions = {
        (str(video_info["user_id"]), str(video_info["session_id"]))
        for video_info in input_files
        if str(video_info.get("session_id", ""))
    }
    print(f"Number of failed videos: {len(failed_videos)}")
    print(f"Number of successful videos: {num_successful_videos}")
    print(f"Number of total files: {len(input_files)}")
    print(f"Number of total chunk files: {len(results)}")
    print(f"Number of total video chunks: {total_video_chunks}")
    print(f"Crowd-cast mode: {crowd_cast_mode}")
    print(
        f"Skipped nonstandard crowd-cast videos during indexing: {len(skipped_nonstandard)}"
    )

    metadata = {
        "env": args.env_name,
        "target_width": args.target_width,
        "target_height": args.target_height,
        "target_channels": 3,
        "target_fps": args.target_fps,
        "chunk_size": args.chunk_size,
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
        "scan_root": scan_root,
        "crowd_cast_mode": crowd_cast_mode,
        "split_strategy": "session" if crowd_cast_mode else "file",
        "seed": args.seed,
        "num_workers": args.num_workers if args.num_workers > 0 else mp.cpu_count(),
        "num_users": len(unique_users),
        "num_sessions": len(unique_sessions),
        "split_stats": split_stats,
        "skipped_nonstandard_videos": skipped_nonstandard,
    }

    with open(os.path.join(args.output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    main()
