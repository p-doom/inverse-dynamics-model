"""
Grain-based dataloader for video action datasets.

Supports two record formats:
  1. Legacy (pickle): keys "raw_video" (flat uint8 bytes), "sequence_length", "actions", "path"
  2. JPEG (msgpack):  keys "jpeg_frames" (list[bytes]), "sequence_length", "actions", "path"

The deserializer auto-detects the format per record.
"""

from __future__ import annotations

import glob
import io
import os
import pickle
from typing import Any

import cv2
import grain
import msgpack
import numpy as np

from idm.utils.actions import action_is_no_op_b


def derive_record_key(rec_d: dict[str, Any]) -> str:
    path_s = rec_d.get("path")
    if path_s:
        return path_s
    raise ValueError("Cannot derive record key. Need non-empty `path`.")


def _deserialize_record(raw_bytes: bytes) -> dict[str, Any]:
    """Try pickle first (legacy), fall back to msgpack (JPEG format)."""
    try:
        rec = pickle.loads(raw_bytes)
        if isinstance(rec, dict):
            return rec
    except Exception:
        pass
    rec = msgpack.unpackb(raw_bytes, raw=True)
    if isinstance(rec, dict):
        # msgpack may return bytes keys — normalise to str
        return {
            (k.decode() if isinstance(k, bytes) else k): v
            for k, v in rec.items()
        }
    raise ValueError("Record is neither valid pickle nor msgpack.")


def _decode_frames(
    rec_d: dict[str, Any],
    image_h: int,
    image_w: int,
    image_c: int,
) -> np.ndarray:
    """Return (T, H, W, C) uint8 array from either format."""
    T = int(rec_d["sequence_length"])

    # ── Legacy: raw flat bytes ───────────────────────────────────────────
    raw_b = rec_d.get("raw_video")
    if raw_b is not None:
        expected = T * image_h * image_w * image_c
        if len(raw_b) != expected:
            raise ValueError(
                f"raw_video length {len(raw_b)} != expected {expected}"
            )
        return np.frombuffer(raw_b, dtype=np.uint8).reshape(
            T, image_h, image_w, image_c
        )

    # ── JPEG: list of compressed bytes ───────────────────────────────────
    jpeg_frames = rec_d.get("jpeg_frames")
    if jpeg_frames is not None:
        if len(jpeg_frames) != T:
            raise ValueError(
                f"jpeg_frames count {len(jpeg_frames)} != sequence_length {T}"
            )
        frames = np.empty((T, image_h, image_w, image_c), dtype=np.uint8)
        for i, jpg_bytes in enumerate(jpeg_frames):
            if isinstance(jpg_bytes, memoryview):
                jpg_bytes = bytes(jpg_bytes)
            buf = np.frombuffer(jpg_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Failed to decode JPEG frame {i}")
            # Resize if needed (in case encoded size differs)
            if bgr.shape[0] != image_h or bgr.shape[1] != image_w:
                bgr = cv2.resize(bgr, (image_w, image_h))
            frames[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return frames

    raise ValueError("Record has neither 'raw_video' nor 'jpeg_frames'.")


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
            rec_d = _deserialize_record(element)
            T = int(rec_d["sequence_length"])
            if T < self.seq_len:
                return False
            actions_L = rec_d.get("actions")
            if actions_L is None or len(actions_L) != T:
                return False
            # Validate that frames are decodable (cheap check)
            if rec_d.get("raw_video") is not None:
                expected = T * self.image_h * self.image_w * self.image_c
                return len(rec_d["raw_video"]) == expected
            if rec_d.get("jpeg_frames") is not None:
                return len(rec_d["jpeg_frames"]) == T
            return False
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
        rec_d = _deserialize_record(element)
        T = int(rec_d["sequence_length"])
        actions_L = rec_d.get("actions")
        if actions_L is None:
            raise ValueError("missing actions")
        # Decode bytes keys in actions if needed (msgpack returns bytes)
        actions_L = [
            a.decode() if isinstance(a, bytes) else str(a)
            for a in actions_L
        ]
        if len(actions_L) != T:
            raise ValueError("actions length mismatch")

        max_start_i = T - self.seq_len
        if max_start_i < 0:
            raise ValueError(f"Sequence too short: T={T}, seq_len={self.seq_len}.")
        start_i = int(rng.integers(0, max_start_i + 1)) if max_start_i > 0 else 0
        end_i = start_i + self.seq_len

        frames_THWC = _decode_frames(
            rec_d, self.image_h, self.image_w, self.image_c,
        )

        # Also pass through jpeg_frames slice for visual logging
        result = {
            "frames": frames_THWC[start_i:end_i],
            "actions": list(actions_L[start_i:end_i]),
        }

        jpeg_frames = rec_d.get("jpeg_frames")
        if jpeg_frames is not None:
            result["jpeg_frames"] = list(jpeg_frames[start_i:end_i])

        return result


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


class BuildSFTExampleFromFrames(grain.transforms.Map):
    def map(self, element: dict[str, Any]) -> dict[str, Any]:
        actions_L = element["actions"]
        target_text = "\n".join(f"Frame {i}: {a}" for i, a in enumerate(actions_L))
        result = {
            "frames": element["frames"],
            "target_text": target_text,
        }
        # Pass through jpeg_frames if present (for visual logging)
        if "jpeg_frames" in element:
            result["jpeg_frames"] = element["jpeg_frames"]
        return result


import os

MIN_ARRAY_RECORD_SIZE = 64 * 1024  # 64KB

def _validate_array_record_file(path: str) -> bool:
    """Return True if the file can be opened and its record count read."""
    try:
        from array_record.python.array_record_module import ArrayRecordReader
        r = ArrayRecordReader(path)
        _ = r.num_records()
        r.close()
        return True
    except Exception:
        return False


def find_array_record_paths(data_root: str, split: str) -> list[str]:
    split_dir = os.path.join(data_root, split)
    paths = sorted(
        os.path.join(split_dir, f)
        for f in os.listdir(split_dir)
        if f.endswith(".array_record")
    )
    valid = []
    skipped_size = 0
    skipped_corrupt = 0
    for p in paths:
        if os.path.getsize(p) < MIN_ARRAY_RECORD_SIZE:
            skipped_size += 1
            continue
        if not _validate_array_record_file(p):
            skipped_corrupt += 1
            print(f"[{split}] Skipping corrupt file: {p}")
            continue
        valid.append(p)
    if skipped_size > 0:
        print(f"[{split}] Skipped {skipped_size} array_record files < 64KB")
    if skipped_corrupt > 0:
        print(f"[{split}] Skipped {skipped_corrupt} corrupt array_record files")
    return valid



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
        from array_record.python.array_record_module import ArrayRecordReader
        reader = ArrayRecordReader(path_s)
        for idx in range(reader.num_records()):
            raw = reader.read([idx])
            valid_n += int(filt.filter(raw[0]))
        reader.close()
    return valid_n
