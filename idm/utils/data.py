from __future__ import annotations

import glob
import os
import pickle
from typing import Any

from array_record.python.array_record_module import ArrayRecordReader
import grain
import numpy as np

from idm.utils.actions import action_is_no_op_b


def derive_record_key(rec_d: dict[str, Any]) -> str:
    path_s = rec_d.get("path")
    if path_s:
        return path_s

    raise ValueError("Cannot derive record key. Need non-empty `path`.")


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
            rec_d = pickle.loads(element)
            T = int(rec_d["sequence_length"])
            raw_b = rec_d["raw_video"]
            actions_L = rec_d["actions"]
            if T < self.seq_len or not isinstance(raw_b, bytes):
                return False
            exp_n = T * self.image_h * self.image_w * self.image_c
            if len(raw_b) != exp_n:
                return False
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
        rec_d = pickle.loads(element)
        T = int(rec_d["sequence_length"])

        raw_b = rec_d["raw_video"]
        actions_L = rec_d.get("actions")
        if actions_L is None:
            raise ValueError("missing actions")
        if len(actions_L) != T:
            raise ValueError("actions length mismatch")

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


class ActionDensityFilter(grain.transforms.Filter):
    def __init__(self, min_action_density: float):
        self.min_action_density = float(min_action_density)

    def filter(self, element: Any) -> bool:
        if self.min_action_density <= 0.0:
            return True
        try:
            actions_L = element["actions"]
            if not actions_L:
                return False
            active_n = sum(
                1 for action_s in actions_L if not action_is_no_op_b(action_s)
            )
            density_f = float(active_n) / float(len(actions_L))
            return density_f >= self.min_action_density
        except Exception:
            return False


class ActionDensityRejectionSampler(grain.transforms.RandomMap):
    def __init__(self, random_sample_fraction: float):
        self.random_sample_fraction = float(random_sample_fraction)

    def random_map(self, element: Any, rng: np.random.Generator) -> Any:
        if self.random_sample_fraction >= 1.0:
            return element
        try:
            actions_L = element["actions"]
            if not actions_L:
                return None
            active_n = sum(
                1 for action_s in actions_L if not action_is_no_op_b(action_s)
            )
            density_f = float(active_n) / float(len(actions_L))
            keep_prob_f = (
                self.random_sample_fraction
                + (1.0 - self.random_sample_fraction) * density_f
            )
            return element if float(rng.random()) < keep_prob_f else None
        except Exception:
            return None


class NotNoneFilter(grain.transforms.Filter):
    def filter(self, element: Any) -> bool:
        return element is not None


class BuildSFTExampleFromFrames(grain.transforms.Map):
    def map(self, element: dict[str, Any]) -> dict[str, Any]:
        actions_L = element["actions"]
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


def count_source_records(array_record_paths: list[str]) -> int:
    if not array_record_paths:
        raise ValueError("array_record_paths list cannot be empty.")
    source = grain.sources.ArrayRecordDataSource(array_record_paths)
    return int(len(source))


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
    num_epochs: int | None = 1,
    num_workers: int = 4,
    prefetch_buffer_size: int = 8,
    read_num_threads: int = 4,
    worker_buffer_size: int = 4,
    min_action_density: float = 0.0,
    action_upsample_random_fraction: float = 1.0,
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
    if prefetch_buffer_size <= 0:
        raise ValueError("prefetch_buffer_size must be >= 1.")
    if read_num_threads <= 0:
        raise ValueError("read_num_threads must be >= 1.")
    if worker_buffer_size <= 0:
        raise ValueError("worker_buffer_size must be >= 1.")
    if min_action_density < 0.0 or min_action_density > 1.0:
        raise ValueError("min_action_density must be in [0, 1].")
    if action_upsample_random_fraction < 0.0 or action_upsample_random_fraction > 1.0:
        raise ValueError("action_upsample_random_fraction must be in [0, 1].")

    source = grain.sources.ArrayRecordDataSource(array_record_paths)
    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.ShardOptions(
            shard_index=rank,
            shard_count=world_size,
            drop_remainder=True,
        ),
        shuffle=True,
        num_epochs=num_epochs,
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
    ]
    if min_action_density > 0.0:
        ops.append(ActionDensityFilter(min_action_density=min_action_density))
    if action_upsample_random_fraction < 1.0:
        ops.extend(
            [
                ActionDensityRejectionSampler(
                    random_sample_fraction=action_upsample_random_fraction
                ),
                NotNoneFilter(),
            ]
        )
    ops.extend(
        [
            BuildSFTExampleFromFrames(),
            grain.transforms.Batch(
                batch_size=global_batch_size // world_size,
                drop_remainder=True,
            ),
        ]
    )
    read_opts = grain.ReadOptions(
        prefetch_buffer_size=prefetch_buffer_size,
        num_threads=read_num_threads,
    )
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=ops,
        worker_count=num_workers,
        worker_buffer_size=worker_buffer_size,
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
