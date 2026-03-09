import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

import ffmpeg
import numpy as np
import tyro
from PIL import Image
from video_to_array_records import (
    RECORDING_RE,
    _actions_from_keylog_file,
    _collect_input_videos,
    _get_keylog_path,
    _safe_mean,
    _split_videos,
)

"""
Save each video as a folder of JPEG frames + actions.json (one action string per frame).

Output layout:
    <output_path>/<split>/<video_stem>/
        frames/
            frame_000000.jpg
            frame_000001.jpg
            ...
        actions.json        # list[str], length == number of frames
"""


@dataclass
class Args:
    input_path: str
    output_path: str
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    target_height: int = 90
    target_fps: int = 10
    jpeg_quality: int = 85
    seed: int = 0
    num_workers: int = 0
    max_videos: int = 0


def _process_single_video(
    idx: int,
    total: int,
    video_info: dict[str, object],
    output_folder: str,
    target_height: int,
    target_fps: int,
    jpeg_quality: int,
) -> dict[str, object]:
    in_filename = str(video_info["filename"])
    video_stem = Path(in_filename).stem
    video_dir = os.path.join(output_folder, video_stem)
    frames_dir = os.path.join(video_dir, "frames")

    print(f"Processing video {idx + 1}/{total}: {in_filename}")
    try:
        probe = ffmpeg.probe(in_filename)
        video_stream = next(
            s for s in probe["streams"] if s["codec_type"] == "video"
        )
        in_width = int(video_stream["width"])
        in_height = int(video_stream["height"])
        out_width = round(target_height * in_width / in_height)
        out_width += out_width % 2  # ensure even for codec compatibility

        out, _ = (
            ffmpeg.input(in_filename)
            .filter("fps", fps=target_fps, round="up")
            .filter("scale", out_width, target_height)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, quiet=True)
        )
        frame_size = target_height * out_width * 3
        n_frames = len(out) // frame_size
        if n_frames == 0:
            return _skip_result(video_info, "no_frames")

        frames = np.frombuffer(out, np.uint8).reshape(
            n_frames, target_height, out_width, 3
        )

        keylog_path = _get_keylog_path(in_filename)
        actions = _actions_from_keylog_file(
            keylog_path=keylog_path,
            n_frames=n_frames,
            target_fps=target_fps,
        )

        os.makedirs(frames_dir, exist_ok=True)
        for fi in range(n_frames):
            frame_path = os.path.join(frames_dir, f"frame_{fi:06d}.jpg")
            Image.fromarray(frames[fi]).save(
                frame_path, format="JPEG", quality=jpeg_quality
            )

        with open(os.path.join(video_dir, "actions.json"), "w") as f:
            json.dump(actions, f)

        return {
            "source_filename": in_filename,
            "path": str(video_info.get("path", "")),
            "output_dir": video_dir,
            "n_frames": n_frames,
            "frame_width": out_width,
            "frame_height": target_height,
            "skip_reason": "",
        }
    except Exception as e:
        print(f"Error processing video {idx} ({in_filename}): {e}")
        return _skip_result(video_info, f"error:{e}")


def _skip_result(video_info: dict[str, object], reason: str) -> dict[str, object]:
    return {
        "source_filename": str(video_info.get("filename", "")),
        "path": str(video_info.get("path", "")),
        "output_dir": "",
        "n_frames": 0,
        "skip_reason": reason,
    }


def _process_video_shard(
    shard_args: list[tuple[object, ...]],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for args_tuple in shard_args:
        results.append(
            _process_single_video(
                idx=int(args_tuple[0]),
                total=int(args_tuple[1]),
                video_info=args_tuple[2],
                output_folder=str(args_tuple[3]),
                target_height=int(args_tuple[4]),
                target_fps=int(args_tuple[5]),
                jpeg_quality=int(args_tuple[6]),
            )
        )
    return results


def _save_split(
    pool_args: list[tuple[object, ...]],
    num_workers: int,
) -> list[dict[str, object]]:
    if not pool_args:
        return []
    num_processes = num_workers if num_workers > 0 else mp.cpu_count()
    num_processes = min(num_processes, len(pool_args))
    print(f"Number of processes: {num_processes}")

    shards: list[list[tuple[object, ...]]] = [[] for _ in range(num_processes)]
    for i, arg in enumerate(pool_args):
        shards[i % num_processes].append(arg)
    shards = [s for s in shards if s]

    results: list[dict[str, object]] = []
    with mp.Pool(processes=num_processes) as pool:
        async_results = [
            pool.apply_async(_process_video_shard, (shard,)) for shard in shards
        ]
        for ar in async_results:
            results.extend(ar.get())
    return results


def main():
    args = tyro.cli(Args)
    print(f"Output path: {args.output_path}")

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    assert np.isclose(total_ratio, 1.0), "Ratios must sum to 1.0"

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

    pool_args: dict[str, list[tuple[object, ...]]] = {}
    for split, videos in file_splits.items():
        split_dir = os.path.join(args.output_path, split)
        os.makedirs(split_dir, exist_ok=True)
        pool_args[split] = [
            (
                idx,
                len(videos),
                video_info,
                split_dir,
                args.target_height,
                args.target_fps,
                args.jpeg_quality,
            )
            for idx, video_info in enumerate(videos)
        ]

    all_results: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        print(f"\n========= Processing split: {split} =========\n")
        all_results.extend(_save_split(pool_args[split], args.num_workers))

    failed = [r for r in all_results if r["n_frames"] == 0]
    failed_names = {r["source_filename"] for r in failed if r["source_filename"]}
    num_failed = len(failed_names)
    num_ok = len(input_files) - num_failed
    total_frames = sum(int(r["n_frames"]) for r in all_results)

    print(f"\nFailed videos:     {num_failed}")
    print(f"Successful videos: {num_ok}")
    print(f"Total input files: {len(input_files)}")
    print(f"Total frames:      {total_frames}")

    metadata = {
        "target_height": args.target_height,
        "target_channels": 3,
        "target_fps": args.target_fps,
        "jpeg_quality": args.jpeg_quality,
        "total_videos": len(input_files),
        "num_successful_videos": num_ok,
        "num_failed_videos": num_failed,
        "total_frames": total_frames,
        "avg_frames_train": _safe_mean(
            [float(r["n_frames"]) for r in all_results if r["n_frames"] > 0]
        ),
        "seed": args.seed,
        "num_workers": args.num_workers if args.num_workers > 0 else mp.cpu_count(),
        "split_stats": split_stats,
        "video_results": all_results,
    }

    with open(os.path.join(args.output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    print(f"Metadata written to {os.path.join(args.output_path, 'metadata.json')}")


if __name__ == "__main__":
    main()
