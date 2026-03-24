import argparse
import json
import os
import multiprocessing as mp
from pathlib import Path

import cv2
import msgpack
import numpy as np
from scipy.signal import lfilter
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────
MOUSE_X_QUANT_UNIT_F = 5.0
MOUSE_Y_QUANT_UNIT_F = 4.0
MOUSE_DELTA_CLIP_I = 64
MOUSE_DELTA_EXP_CURVATURE_F = 1.0

DEFAULT_CHUNK_SIZE = 160
DEFAULT_CHUNKS_PER_FILE = 100
DEFAULT_JPEG_QUALITY = 85
DEFAULT_TARGET_FPS = 10
DEFAULT_NUM_FRAMES = 3_000_000
DEFAULT_NUM_WORKERS = 32


# ── Vectorized quantization ─────────────────────────────────────────────────

def quantize_mouse_deltas_vectorized(
    raw_d: np.ndarray,
    quant_unit_f: float,
    clip_abs_i: int,
    curvature_f: float = MOUSE_DELTA_EXP_CURVATURE_F,
) -> np.ndarray:
    max_value_f = quant_unit_f * float(clip_abs_i)
    if max_value_f <= 0.0:
        return np.zeros_like(raw_d, dtype=np.int32)
    sign = np.sign(raw_d)
    abs_d = np.abs(raw_d)
    normalized = np.clip(abs_d / max_value_f, 0.0, 1.0)
    curved = np.log1p(curvature_f * normalized) / np.log1p(curvature_f)
    quant_abs = np.rint(curved * float(clip_abs_i)).astype(np.int32)
    np.clip(quant_abs, 0, clip_abs_i, out=quant_abs)
    return (sign * quant_abs).astype(np.int32)


def format_actions_vectorized(dx: np.ndarray, dy: np.ndarray) -> list[str]:
    """Fully vectorized action string formatting."""
    n = len(dx)
    noop_mask = (dx == 0) & (dy == 0)
    actions = [None] * n

    # Build all MOUSE strings at once for non-noop entries
    noop_indices = np.where(noop_mask)[0]
    mouse_indices = np.where(~noop_mask)[0]

    for i in noop_indices:
        actions[i] = "NO_OP"

    # Batch-format mouse actions using pre-built strings
    if len(mouse_indices) > 0:
        dx_m = dx[mouse_indices]
        dy_m = dy[mouse_indices]
        for j, i in enumerate(mouse_indices):
            actions[i] = f"MOUSE:{dx_m[j]},{dy_m[j]},0"

    return actions


# ── Cursor polygon ──────────────────────────────────────────────────────────

def get_base_cursor_polygon(scale: float = 1.0) -> np.ndarray:
    raw_points = np.array([
        [0,  0], [0,  20], [4,  16], [8,  24],
        [11, 23], [7,  15], [13, 15],
    ], dtype=np.float64)
    raw_points *= scale
    return raw_points.astype(np.int32)


# ── Mouse trajectory simulation (vectorized) ────────────────────────────────

def simulate_mouse_trajectory(
    n_frames: int,
    img_w: int,
    img_h: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    decay = 0.95
    noise_scale = 12.0

    noise = rng.normal(0, noise_scale, size=(n_frames, 2))

    b = [1.0 - decay]
    a = [1.0, -decay]
    vx = lfilter(b, a, noise[:, 0])
    vy = lfilter(b, a, noise[:, 1])

    # ── Vectorized cumulative-sum-with-clipping ──────────────────────────
    # Replace the sequential Python loop with chunked numpy.
    # Pure cumsum then clip won't replicate the clamp-at-each-step semantics,
    # but we can do it in C-speed chunks to amortise the Python overhead.
    # Strategy: process in large blocks; within each block use cumsum, then
    # check if any value leaves [0, W-1]. If the block is fully in-bounds,
    # accept it in one shot.  Otherwise fall back to a small inner loop for
    # just that block.  For typical trajectories the vast majority of blocks
    # are in-bounds.

    BLOCK = 4096
    xs = np.empty(n_frames, dtype=np.float64)
    ys = np.empty(n_frames, dtype=np.float64)
    xs[0] = img_w / 2.0
    ys[0] = img_h / 2.0

    xmax = float(img_w - 1)
    ymax = float(img_h - 1)

    i = 1
    while i < n_frames:
        end = min(i + BLOCK, n_frames)
        # Optimistic: cumsum the whole block
        cx = np.cumsum(vx[i:end]) + xs[i - 1]
        cy = np.cumsum(vy[i:end]) + ys[i - 1]

        if cx.min() >= 0.0 and cx.max() <= xmax and cy.min() >= 0.0 and cy.max() <= ymax:
            xs[i:end] = cx
            ys[i:end] = cy
        else:
            # Slow path – only for blocks that hit boundaries
            for j in range(i, end):
                xs[j] = min(max(xs[j - 1] + vx[j], 0.0), xmax)
                ys[j] = min(max(ys[j - 1] + vy[j], 0.0), ymax)
        i = end

    raw_dx = np.diff(xs, prepend=xs[0])
    raw_dy = np.diff(ys, prepend=ys[0])
    raw_dx[0] = 0.0
    raw_dy[0] = 0.0

    dx_q = quantize_mouse_deltas_vectorized(raw_dx, MOUSE_X_QUANT_UNIT_F, MOUSE_DELTA_CLIP_I)
    dy_q = quantize_mouse_deltas_vectorized(raw_dy, MOUSE_Y_QUANT_UNIT_F, MOUSE_DELTA_CLIP_I)
    actions = format_actions_vectorized(dx_q, dy_q)

    return xs.astype(np.int32), ys.astype(np.int32), actions


# ── Worker ───────────────────────────────────────────────────────────────────

def _build_and_write_shard(
    worker_idx: int,
    frame_range: tuple[int, int],
    base_frame_bgr: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    actions: list[str],
    output_folder: str,
    chunk_size: int,
    chunks_per_file: int,
    jpeg_quality: int,
    source_path: str,
    cursor_scale: int,
) -> list[dict]:
    cv2.setNumThreads(0)

    from array_record.python.array_record_module import ArrayRecordWriter

    # ── JPEG encoder selection ───────────────────────────────────────────
    try:
        from turbojpeg import TurboJPEG
        _tj = TurboJPEG()
        def _encode_jpeg(frame_bgr):
            return _tj.encode(frame_bgr, quality=jpeg_quality)
    except ImportError:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        def _encode_jpeg(frame_bgr):
            ok, buf = cv2.imencode(".jpg", frame_bgr, encode_params)
            return buf.tobytes()

    n = len(xs)
    img_h, img_w = base_frame_bgr.shape[:2]

    base_polygon = get_base_cursor_polygon(cursor_scale)
    poly_min = base_polygon.min(axis=0)
    poly_max = base_polygon.max(axis=0)
    pad = 2

    # ── Pre-encode the bare background frame once ────────────────────────
    bare_jpeg = _encode_jpeg(base_frame_bgr)

    # ── Pre-compute unique cursor positions & their JPEG frames ──────────
    # Many frames share the same (x,y) → encode each unique position once.
    coords = np.stack([xs, ys], axis=1)                       # (N, 2)
    unique_coords, inverse_idx = np.unique(coords, axis=0, return_inverse=True)

    # Build a cache: unique_pos_index -> jpeg bytes
    pos_jpeg_cache: dict[int, bytes] = {}

    # Also detect "no-move" frames that map to (dx=0, dy=0).
    # The first frame of the trajectory has action NO_OP by construction;
    # for truly zero-delta frames, the cursor position is identical to the
    # previous frame, so the cache already handles dedup via unique coords.

    # Encode unique positions lazily inside the chunk loop (avoids encoding
    # millions of positions that might never be used if n is huge).

    file_index = 0
    pending: list[dict] = []
    results: list[dict] = []

    frame = base_frame_bgr.copy()  # working copy — restored after each draw

    def _flush(batch: list[dict]) -> dict:
        nonlocal file_index
        fname = f"chunked_videos_w{worker_idx:04d}_{file_index:06d}.array_record"
        fpath = os.path.join(output_folder, fname)
        writer = ArrayRecordWriter(fpath, "group_size:1")
        try:
            for rec in batch:
                writer.write(msgpack.packb(rec, use_bin_type=True))
        finally:
            writer.close()
        file_index += 1
        return {
            "filename": fpath,
            "length": int(batch[0]["sequence_length"]),
            "num_chunks_in_file": len(batch),
        }

    total_chunks = (n - chunk_size + 1 + chunk_size - 1) // chunk_size  # for progress
    pbar = tqdm(total=total_chunks, desc=f"Worker {worker_idx}", leave=False, position=worker_idx)

    for chunk_start in range(0, n - chunk_size + 1, chunk_size):
        chunk_end = chunk_start + chunk_size
        chunk_actions = actions[chunk_start:chunk_end]

        jpeg_frames: list[bytes] = []

        for i in range(chunk_start, chunk_end):
            uid = inverse_idx[i]
            cached = pos_jpeg_cache.get(uid)
            if cached is not None:
                jpeg_frames.append(cached)
                continue

            cur_x, cur_y = int(xs[i]), int(ys[i])

            # Compute ROI
            roi_x1 = max(cur_x + poly_min[0] - pad, 0)
            roi_y1 = max(cur_y + poly_min[1] - pad, 0)
            roi_x2 = min(cur_x + poly_max[0] + pad, img_w)
            roi_y2 = min(cur_y + poly_max[1] + pad, img_h)

            patch_backup = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()

            pts = base_polygon + np.array([cur_x, cur_y], dtype=np.int32)
            cv2.fillPoly(frame, [pts], (255, 255, 255), cv2.LINE_AA)
            cv2.polylines(frame, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)

            encoded = _encode_jpeg(frame)
            jpeg_frames.append(encoded)

            frame[roi_y1:roi_y2, roi_x1:roi_x2] = patch_backup

            # Cache (with a size cap to avoid OOM on huge runs)
            if len(pos_jpeg_cache) < 500_000:
                pos_jpeg_cache[uid] = encoded

        rec = {
            "jpeg_frames": jpeg_frames,
            "sequence_length": chunk_size,
            "path": source_path,
            "actions": chunk_actions,
        }
        pending.append(rec)
        pbar.update(1)

        if len(pending) >= chunks_per_file:
            results.append(_flush(pending))
            pending = []

    if len(pending) >= 2:
        results.append(_flush(pending))
    elif pending:
        pass  # drop tiny leftovers

    pbar.close()
    return results


def _shard_worker_star(args):
    return _build_and_write_shard(*args)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate simulated mouse-movement dataset "
                    "(array_record + msgpack, matching the video pipeline format)."
    )
    parser.add_argument("--frame_path", type=str, default="/home/hk-project-p0023960/tum_ddk3888/inverse-dynamics-model/data/frame.png")
    parser.add_argument("--output_path", type=str, default="/hkfs/work/workspace/scratch/tum_cte0515-crowd-cast/simulated_mouse_ds")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunks_per_file", type=int, default=DEFAULT_CHUNKS_PER_FILE)
    parser.add_argument("--jpeg_quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--target_fps", type=int, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--target_width", type=int, default=960)
    parser.add_argument("--target_height", type=int, default=540)
    parser.add_argument("--cursor_scale", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6
    assert 1 <= args.jpeg_quality <= 100

    # ── Load & resize frame ──────────────────────────────────────────────
    frame_bgr = cv2.imread(args.frame_path)
    assert frame_bgr is not None, f"Cannot read image: {args.frame_path}"
    frame_bgr = cv2.resize(
        frame_bgr, (args.target_width, args.target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    # Force contiguous C-order for faster copies / JPEG encoding
    frame_bgr = np.ascontiguousarray(frame_bgr)
    print(f"Frame resized to {args.target_width}x{args.target_height}")

    # ── Simulate trajectory ──────────────────────────────────────────────
    print(f"Simulating {args.num_frames:,} frames of mouse movement ...")
    xs, ys, actions = simulate_mouse_trajectory(
        n_frames=args.num_frames,
        img_w=args.target_width,
        img_h=args.target_height,
        seed=args.seed,
    )
    n_noop = sum(1 for a in actions if a == "NO_OP")
    print(f"  {n_noop:,} NO_OP frames ({100.0 * n_noop / len(actions):.1f}%)")

    # ── Split ────────────────────────────────────────────────────────────
    total_chunks = args.num_frames // args.chunk_size
    n_train = round(total_chunks * args.train_ratio)
    n_val = round(total_chunks * args.val_ratio)

    splits = {
        "train": (0, n_train * args.chunk_size),
        "val":   (n_train * args.chunk_size, (n_train + n_val) * args.chunk_size),
        "test":  ((n_train + n_val) * args.chunk_size, total_chunks * args.chunk_size),
    }

    all_results: dict[str, list[dict]] = {}

    for split, (s, e) in splits.items():
        if e <= s:
            all_results[split] = []
            continue
        split_dir = os.path.join(args.output_path, split)
        os.makedirs(split_dir, exist_ok=True)
        n_split = e - s

        n_workers = min(args.num_workers, mp.cpu_count(), 32)
        n_workers = max(1, min(n_workers, n_split // args.chunk_size))

        frames_per_worker = (n_split // n_workers // args.chunk_size) * args.chunk_size
        worker_jobs = []
        for w in range(n_workers):
            ws = s + w * frames_per_worker
            we = ws + frames_per_worker if w < n_workers - 1 else e
            we = ws + ((we - ws) // args.chunk_size) * args.chunk_size
            if we <= ws:
                continue
            worker_jobs.append((
                w, (0, we - ws), frame_bgr,
                xs[ws:we].copy(), ys[ws:we].copy(), actions[ws:we],
                split_dir, args.chunk_size, args.chunks_per_file,
                args.jpeg_quality, args.frame_path, args.cursor_scale,
            ))

        print(f"\n{'='*20} {split} ({n_split:,} frames, {len(worker_jobs)} workers) {'='*20}\n")

        if len(worker_jobs) <= 1:
            split_results = [_shard_worker_star(j) for j in worker_jobs]
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=len(worker_jobs)) as pool:
                split_results = list(pool.imap_unordered(_shard_worker_star, worker_jobs))

        all_results[split] = [r for shard in split_results for r in shard]

    # ── Metadata ─────────────────────────────────────────────────────────
    def _safe_mean(lst):
        return float(np.mean([r["length"] for r in lst])) if lst else 0.0

    total_video_chunks = sum(
        r["num_chunks_in_file"] for res in all_results.values() for r in res
    )
    metadata = {
        "target_width": args.target_width,
        "target_height": args.target_height,
        "target_channels": 3,
        "target_fps": args.target_fps,
        "chunk_size": args.chunk_size,
        "mouse_delta_clip": MOUSE_DELTA_CLIP_I,
        "mouse_scroll_clip": 0,
        "no_op_as_mouse_zero": False,
        "actions_stateful": True,
        "jpeg_quality": args.jpeg_quality,
        "frame_encoding": "jpeg",
        "cursor_rendered": True,
        "cursor_scale": args.cursor_scale,
        "simulated": True,
        "num_frames": args.num_frames,
        "total_chunks": sum(len(v) for v in all_results.values()),
        "total_video_chunks": total_video_chunks,
        "avg_episode_len_train": _safe_mean(all_results["train"]),
        "avg_episode_len_val": _safe_mean(all_results["val"]),
        "avg_episode_len_test": _safe_mean(all_results["test"]),
        "episode_metadata_train": all_results["train"],
        "episode_metadata_val": all_results["val"],
        "episode_metadata_test": all_results["test"],
        "seed": args.seed,
        "split_stats": {
            split: {
                "frames": splits[split][1] - splits[split][0],
                "chunks": sum(r["num_chunks_in_file"] for r in all_results[split]),
                "files": len(all_results[split]),
            }
            for split in splits
        },
    }
    meta_path = os.path.join(args.output_path, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written to {meta_path}")
    print(f"Total video chunks: {total_video_chunks:,}")
    print("Done.")


if __name__ == "__main__":
    main()
