#!/usr/bin/env python3
"""Coalesce training data: merge consecutive identical actions on adjacent frames into single events.

This aligns training data with the coalesced eval scoring — the model learns to predict
one event per user action instead of one event per held-frame.
"""
import argparse
import json
import os
import shutil
from pathlib import Path


def coalesce_actions(actions: list[dict], gap: int = 1) -> list[dict]:
    """Coalesce consecutive MouseScroll and MouseClick events into gesture-level events.

    Matches the eval scorer's coalesce_gt_events exactly:
    - MouseScroll is coalesced (consecutive frames, same direction; splits on direction reversal)
    - MouseClick is coalesced (consecutive frames, same button)
    - KeyPress is LEFT UNTOUCHED
    """
    if not actions:
        return actions

    scrolls = [a for a in actions if a["type"] == "MouseScroll"]
    clicks = [a for a in actions if a["type"] == "MouseClick"]
    others = [a for a in actions if a["type"] not in ("MouseScroll", "MouseClick")]

    # --- Coalesce scrolls ---
    coalesced_scrolls = []
    if scrolls:
        scrolls.sort(key=lambda x: int(x["frame"].replace("F", "")))

        gestures = []
        current = [scrolls[0]]

        for s in scrolls[1:]:
            prev = current[-1]
            prev_f = int(prev["frame"].replace("F", ""))
            curr_f = int(s["frame"].replace("F", ""))
            direction_flip = s.get("details", "") != prev.get("details", "")

            if (curr_f - prev_f) <= gap and not direction_flip:
                current.append(s)
            else:
                gestures.append(current)
                current = [s]
        gestures.append(current)

        coalesced_scrolls = [g[0] for g in gestures]

    # --- Coalesce clicks ---
    coalesced_clicks = []
    if clicks:
        clicks.sort(key=lambda x: int(x["frame"].replace("F", "")))

        gestures = []
        current = [clicks[0]]

        for c in clicks[1:]:
            prev = current[-1]
            prev_f = int(prev["frame"].replace("F", ""))
            curr_f = int(c["frame"].replace("F", ""))
            button_change = c.get("details", "") != prev.get("details", "")

            if (curr_f - prev_f) <= gap and not button_change:
                current.append(c)
            else:
                gestures.append(current)
                current = [c]
        gestures.append(current)

        coalesced_clicks = [g[0] for g in gestures]

    # Merge back, sort by frame
    result = others + coalesced_scrolls + coalesced_clicks
    result.sort(key=lambda x: int(x["frame"].replace("F", "")))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", required=True, help="Original data dir (e.g. data_5fps_stateful)"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output dir for coalesced data"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    for split in ["train", "val"]:
        jsonl_in = input_dir / split / f"{split}.jsonl"
        if not jsonl_in.exists():
            print(f"Skipping {split}: {jsonl_in} not found")
            continue

        out_split = output_dir / split
        os.makedirs(out_split, exist_ok=True)

        total_clips = 0
        total_before = 0
        total_after = 0
        skipped_empty = 0

        with open(jsonl_in) as fin, open(out_split / f"{split}.jsonl", "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                clip = json.loads(line)
                before = len(clip["actions"])
                clip["actions"] = coalesce_actions(clip["actions"])
                after = len(clip["actions"])

                total_before += before
                total_after += after
                total_clips += 1

                if after == 0:
                    skipped_empty += 1
                    continue

                fout.write(json.dumps(clip) + "\n")

        # Symlink clip directories (frames don't change, only the JSONL)
        # Find all clip dirs in the input
        clip_dirs_src = input_dir / split
        clip_dirs_dst = output_dir / split
        for item in clip_dirs_src.iterdir():
            if item.is_dir():
                dst = clip_dirs_dst / item.name
                if not dst.exists():
                    os.symlink(item.resolve(), dst)

        removed_pct = 100 * (1 - total_after / max(total_before, 1))
        print(
            f"{split}: {total_clips} clips, {total_before} → {total_after} actions "
            f"({removed_pct:.0f}% removed), {skipped_empty} empty clips skipped"
        )

    # Copy metadata if exists
    meta = input_dir / "metadata.json"
    if meta.exists():
        shutil.copy2(meta, output_dir / "metadata.json")

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
