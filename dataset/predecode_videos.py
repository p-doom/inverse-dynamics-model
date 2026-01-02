import argparse
import os
from pathlib import Path
import numpy as np
from decord import VideoReader, cpu

def open_npy_memmap(path: Path, shape, dtype=np.uint8):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(filename=str(path), mode="w+", dtype=dtype, shape=shape)

if __name__ == "__main__":
    root = Path("/data")

    mp4s = sorted(root.glob("*.mp4"))

    for mp4 in mp4s:
        vid_id = mp4.stem
        out_path = root / f"{vid_id}_frames_128.npy"

        vr = VideoReader(str(mp4), ctx=cpu(0), width=128, height=128, num_threads=4)
        n = len(vr)
        mm = open_npy_memmap(out_path, shape=(n, 128, 128, 3), dtype=np.uint8)

        chunk = 1024
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            idx = list(range(start, end))
            frames = vr.get_batch(idx).asnumpy()
            mm[start:end] = frames
            mm.flush()