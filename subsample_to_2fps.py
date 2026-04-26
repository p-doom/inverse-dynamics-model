#!/usr/bin/env python3
"""Subsample 5fps training data to 2fps by selecting every ~2.5th frame.

For a 25-frame clip at 5fps (5 seconds), we select 10 frames at 2fps:
  Frame indices: 0, 3, 5, 8, 10, 13, 15, 18, 20, 23
  These correspond to times: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5s

For clips with fewer than 25 frames (some val clips), we compute:
  duration = (num_frames - 1) / 5  (in seconds)
  num_2fps_frames = floor(duration * 2) + 1
  For each 2fps frame i: source_frame = round(i * 5 / 2) = round(i * 2.5)

Action frame remapping: original frame F_n at 5fps has time t = n/5.
  At 2fps, new frame index = round(t * 2) = round(n * 2 / 5), capped at (num_2fps_frames - 1).
"""

import argparse
import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def round_half_up(x: float) -> int:
    """Round half up (standard rounding), not banker's rounding."""
    return int(math.floor(x + 0.5))


def compute_2fps_indices(num_5fps_frames: int) -> list[int]:
    """Compute which 5fps frame indices to keep for 2fps subsampling."""
    duration = (num_5fps_frames - 1) / 5.0  # clip duration in seconds
    num_2fps = int(duration * 2) + 1  # number of 2fps frames
    # For each 2fps frame, find nearest 5fps frame
    indices = []
    for i in range(num_2fps):
        t = i / 2.0  # time in seconds for this 2fps frame
        src_idx = round_half_up(t * 5)  # nearest 5fps frame index
        src_idx = min(src_idx, num_5fps_frames - 1)
        indices.append(src_idx)
    return indices


def remap_action_frame(
    frame_str: str, num_5fps_frames: int, num_2fps_frames: int
) -> str:
    """Remap a 5fps frame label (e.g. 'F13') to 2fps frame index."""
    n = int(frame_str[1:])  # e.g. 13
    t = n / 5.0  # time in seconds
    new_idx = round_half_up(t * 2)  # 2fps frame index
    new_idx = min(new_idx, num_2fps_frames - 1)
    return f"F{new_idx:02d}"


def process_clip(args: tuple) -> dict | None:
    """Process a single clip: copy subsampled frames and return new JSONL entry."""
    record, input_base, output_base = args

    clip_dir = record["clip_dir"]  # e.g. "train/rec_93cf25_seg0000_c000"
    num_frames = record["num_frames"]
    src_dir = input_base / clip_dir
    dst_dir = output_base / clip_dir

    if not src_dir.exists():
        return None

    # Compute 2fps frame indices
    indices_5fps = compute_2fps_indices(num_frames)
    num_2fps = len(indices_5fps)

    # Create output directory
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy selected frames as F00.jpg, F01.jpg, ...
    for new_idx, src_idx in enumerate(indices_5fps):
        src_file = src_dir / f"F{src_idx:02d}.jpg"
        dst_file = dst_dir / f"F{new_idx:02d}.jpg"
        if src_file.exists():
            shutil.copy2(src_file, dst_file)

    # Remap actions
    new_actions = []
    for action in record.get("actions", []):
        new_action = dict(action)
        new_action["frame"] = remap_action_frame(action["frame"], num_frames, num_2fps)
        new_actions.append(new_action)

    # Build new record
    new_record = {
        "clip_id": record["clip_id"],
        "clip_dir": clip_dir,
        "num_frames": num_2fps,
        "fps": 2,
        "actions": new_actions,
    }
    return new_record


def process_split(split: str, input_dir: Path, output_dir: Path, workers: int) -> int:
    """Process one split (train or val). Returns count of clips processed."""
    jsonl_in = input_dir / split / f"{split}.jsonl"
    if not jsonl_in.exists():
        print(f"  Skipping {split}: {jsonl_in} not found")
        return 0

    # Read all records
    records = []
    with open(jsonl_in) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"  {split}: {len(records)} clips to process")

    # Prepare arguments
    task_args = [(r, input_dir, output_dir) for r in records]

    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_clip, a): i for i, a in enumerate(task_args)}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append((futures[future], result))
            done += 1
            if done % 10000 == 0:
                print(f"    {done}/{len(records)} clips done")

    # Sort by original order and write JSONL
    results.sort(key=lambda x: x[0])
    jsonl_out = output_dir / split / f"{split}.jsonl"
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_out, "w") as f:
        for _, record in results:
            f.write(json.dumps(record) + "\n")

    print(f"  {split}: wrote {len(results)} clips to {jsonl_out}")
    return len(results)


def main():
    parser = argparse.ArgumentParser(description="Subsample 5fps data to 2fps")
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Path to 5fps data dir"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Path to 2fps output dir"
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of parallel workers"
    )
    args = parser.parse_args()

    print(f"Subsampling 5fps -> 2fps")
    print(f"  Input:   {args.input_dir}")
    print(f"  Output:  {args.output_dir}")
    print(f"  Workers: {args.workers}")

    # Verify 25-frame clip mapping
    expected = [0, 3, 5, 8, 10, 13, 15, 18, 20, 23]
    actual = compute_2fps_indices(25)
    assert actual == expected, f"Expected {expected}, got {actual}"
    print(f"  25-frame mapping verified: {actual}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total = 0
    for split in ["train", "val"]:
        total += process_split(split, args.input_dir, args.output_dir, args.workers)
    elapsed = time.time() - t0

    # Write metadata
    metadata = {
        "source_dir": str(args.input_dir),
        "fps": 2,
        "resolution": "960x540",
        "subsampled_from": "5fps",
        "frame_indices_25": [0, 3, 5, 8, 10, 13, 15, 18, 20, 23],
        "total_clips": total,
    }
    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! {total} clips in {elapsed:.1f}s ({total/elapsed:.0f} clips/s)")


if __name__ == "__main__":
    main()
