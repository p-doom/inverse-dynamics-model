"""Convert a JPEG-frames dataset into chat-format JSONL for VLM fine-tuning.

Each video folder becomes one multi-turn conversation:
    assistant → frame_0   (image)
    user      → action_0  (text)
    assistant → frame_1   (image)
    user      → action_1  (text)
    …
    assistant → frame_N   (image)

The last assistant turn has no subsequent user action, which serves as the
final observation.

Input layout  (produced by video_to_jpegs.py):
    <input_path>/<split>/<video_stem>/
        frames/frame_000000.jpg …
        actions.json

Output:
    <output_path>/<split>.jsonl          – one JSON object per line
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import tyro
from tqdm import tqdm


@dataclass
class Args:
    input_path: str
    output_path: str
    splits: tuple[str, ...] = ("train", "val", "test")
    use_relative_paths: bool = True


def _build_messages(
    frames_dir: Path,
    actions: list[str],
    use_relative_paths: bool,
    base_path: Path,
) -> list[dict]:
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        return []

    n_frames = len(frame_files)
    if len(actions) != n_frames:
        print(
            f"  Warning: {frames_dir} has {n_frames} frames but "
            f"{len(actions)} actions – using min({n_frames}, {len(actions)})"
        )
    n = min(n_frames, len(actions))
    if n == 0:
        return []

    messages: list[dict] = []
    for i in range(n):
        img_path = frame_files[i]
        if use_relative_paths:
            img_ref = str(img_path.relative_to(base_path))
        else:
            img_ref = str(img_path)

        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "image", "image": img_ref}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": actions[i]}],
            }
        )

    # Append the frame that results from the last action (if it exists).
    if n < n_frames:
        last_img = frame_files[n]
        if use_relative_paths:
            img_ref = str(last_img.relative_to(base_path))
        else:
            img_ref = str(last_img)
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "image", "image": img_ref}],
            }
        )

    return messages


def main() -> None:
    args = tyro.cli(Args)
    input_root = Path(args.input_path)
    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    base_path = input_root if args.use_relative_paths else Path()

    for split in args.splits:
        split_dir = input_root / split
        if not split_dir.is_dir():
            print(f"Skipping missing split directory: {split_dir}")
            continue

        video_dirs = sorted(
            d for d in split_dir.iterdir() if d.is_dir() and (d / "actions.json").exists()
        )
        print(f"\n[{split}] Found {len(video_dirs)} video folders")

        out_file = output_root / f"{split}.jsonl"
        n_written = 0
        with open(out_file, "w") as fout:
            for vdir in tqdm(video_dirs, desc=split, unit="vid"):
                frames_dir = vdir / "frames"
                actions_file = vdir / "actions.json"
                if not frames_dir.is_dir():
                    continue

                with open(actions_file) as f:
                    actions: list[str] = json.load(f)

                messages = _build_messages(
                    frames_dir, actions, args.use_relative_paths, base_path
                )
                if not messages:
                    continue

                fout.write(json.dumps({"messages": messages}) + "\n")
                n_written += 1

        print(f"[{split}] Wrote {n_written} conversations → {out_file}")


if __name__ == "__main__":
    main()
