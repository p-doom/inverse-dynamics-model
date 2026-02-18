from __future__ import annotations

import glob
import json
import os
import pickle
from typing import Any

from array_record.python.array_record_module import ArrayRecordReader
import grain
import numpy as np


def derive_record_key(rec_d: dict[str, Any]) -> str:
    path_s = rec_d.get("path")
    if isinstance(path_s, str) and path_s:
        return path_s

    raise ValueError("Cannot derive record key. Need non-empty `path`.")


def _record_actions(rec_d: dict[str, Any], key_s: str) -> list[str]:
    actions_any = rec_d.get("actions")
    if actions_any is None:
        raise ValueError(
            f"Record `{key_s}` is missing `actions`. "
            "Regenerate ArrayRecords with in-record actions."
        )
    if not isinstance(actions_any, list) or not all(isinstance(x, str) for x in actions_any):
        raise ValueError(f"Actions for key `{key_s}` must be list[str].")
    return actions_any


def _decode_record(element_b: bytes) -> dict[str, Any]:
    if not isinstance(element_b, bytes):
        raise ValueError(f"Expected bytes record, got {type(element_b)}.")
    rec_d = pickle.loads(element_b)
    if not isinstance(rec_d, dict):
        raise ValueError("Decoded record must be dict.")
    return rec_d


class EpisodeLengthFilter(grain.transforms.Filter):
    def __init__(
        self,
        seq_len: int,
        image_h: int,
        image_w: int,
        image_c: int = 3,
    ):
        self.seq_len = seq_len
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c

    def filter(self, element: Any) -> bool:
        try:
            rec_d = _decode_record(element)
            T = int(rec_d["sequence_length"])
            raw_b = rec_d.get("raw_video")
            if T < self.seq_len or not isinstance(raw_b, bytes):
                return False
            exp_n = T * self.image_h * self.image_w * self.image_c
            if len(raw_b) != exp_n:
                return False
            key_s = derive_record_key(rec_d)
            actions_L = _record_actions(rec_d, key_s)
            return len(actions_L) == T
        except Exception:
            return False


class ProcessEpisodeAndSlice(grain.transforms.RandomMap):
    def __init__(
        self,
        seq_len: int,
        image_h: int,
        image_w: int,
        image_c: int = 3,
    ):
        self.seq_len = seq_len
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c

    def random_map(self, element: Any, rng: np.random.Generator) -> Any:
        rec_d = _decode_record(element)
        T = int(rec_d["sequence_length"])
        key_s = derive_record_key(rec_d)

        raw_b = rec_d["raw_video"]
        exp_n = T * self.image_h * self.image_w * self.image_c
        if len(raw_b) != exp_n:
            raise ValueError(
                f"raw_video size mismatch for key `{key_s}`: "
                f"got {len(raw_b)}, expected {exp_n}."
            )

        actions_L = _record_actions(rec_d, key_s)
        if len(actions_L) != T:
            raise ValueError(
                f"actions length mismatch for key `{key_s}`: "
                f"got {len(actions_L)}, expected {T}."
            )

        max_start_i = T - self.seq_len
        if max_start_i < 0:
            raise ValueError(f"Sequence too short: T={T}, seq_len={self.seq_len}.")
        start_i = int(rng.integers(0, max_start_i + 1)) if max_start_i > 0 else 0
        end_i = start_i + self.seq_len

        frames_THWC = np.frombuffer(raw_b, dtype=np.uint8).reshape(
            T, self.image_h, self.image_w, self.image_c
        )
        return {
            "frames": frames_THWC[start_i:end_i],
            "actions": list(actions_L[start_i:end_i]),
        }


class BuildSFTExampleFromFrames(grain.transforms.Map):
    def map(self, element: dict[str, Any]) -> dict[str, Any]:
        actions_L = element["actions"]
        if not isinstance(actions_L, list) or not all(
            isinstance(x, str) for x in actions_L
        ):
            raise ValueError("Expected actions as list[str].")

        target_text = "\n".join(f"Frame {i}: {a}" for i, a in enumerate(actions_L))
        return {
            "frames": element["frames"],
            "target_text": target_text,
        }


def find_array_record_paths(data_root: str, split: str) -> list[str]:
    split_dir = os.path.join(data_root, split)
    paths = sorted(glob.glob(os.path.join(split_dir, "*.array_record")))
    if not paths:
        raise ValueError(f"No .array_record files found in {split_dir}")
    return paths


def load_metadata_json(data_root: str) -> dict[str, Any]:
    meta_path = os.path.join(data_root, "metadata.json")
    if not os.path.exists(meta_path):
        raise ValueError(f"metadata.json not found in {data_root}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_image_hwc(
    meta_d: dict[str, Any],
    image_h: int | None,
    image_w: int | None,
    image_c: int | None,
) -> tuple[int, int, int]:
    if image_h and image_w and image_c:
        return int(image_h), int(image_w), int(image_c)

    h = image_h or meta_d.get("target_height")
    w = image_w or meta_d.get("target_width")
    c = image_c or meta_d.get("target_channels", 3)
    if h is None or w is None or c is None:
        raise ValueError(
            "Could not infer image dims. Pass --image_h/--image_w/--image_c "
            "or add target_height/target_width in metadata.json."
        )
    return int(h), int(w), int(c)


def get_dataloader(
    array_record_paths: list[str],
    seq_len: int,
    global_batch_size: int,
    image_h: int,
    image_w: int,
    image_c: int,
    rank: int,
    world_size: int,
    seed: int,
    epoch_i: int,
    num_workers: int = 0,
    prefetch_buffer_size: int = 1,
) -> grain.DataLoader:
    if not array_record_paths:
        raise ValueError("array_record_paths list cannot be empty.")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid shard config rank={rank}, world_size={world_size}.")
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible "
            f"by world_size ({world_size})."
        )

    source = grain.sources.ArrayRecordDataSource(array_record_paths)
    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.ShardOptions(
            shard_index=rank,
            shard_count=world_size,
            drop_remainder=True,
        ),
        shuffle=True,
        num_epochs=1,
        seed=seed + epoch_i,
    )
    ops = [
        EpisodeLengthFilter(
            seq_len=seq_len,
            image_h=image_h,
            image_w=image_w,
            image_c=image_c,
        ),
        ProcessEpisodeAndSlice(
            seq_len=seq_len,
            image_h=image_h,
            image_w=image_w,
            image_c=image_c,
        ),
        BuildSFTExampleFromFrames(),
        grain.transforms.Batch(
            batch_size=global_batch_size // world_size,
            drop_remainder=True,
        ),
    ]
    read_opts = grain.ReadOptions(
        prefetch_buffer_size=prefetch_buffer_size,
        num_threads=1,
    )
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=ops,
        worker_count=num_workers,
        worker_buffer_size=1,
        read_options=read_opts,
    )


def count_valid_records(
    array_record_paths: list[str],
    seq_len: int,
    image_h: int,
    image_w: int,
    image_c: int,
) -> int:
    filt = EpisodeLengthFilter(
        seq_len=seq_len,
        image_h=image_h,
        image_w=image_w,
        image_c=image_c,
    )
    valid_n = 0
    for path_s in array_record_paths:
        reader = ArrayRecordReader(path_s)
        for _ in range(reader.num_records()):
            valid_n += int(filt.filter(reader.read()))
        reader.close()
    return valid_n
