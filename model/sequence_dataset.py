import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import numpy as np
import bisect
import random
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict

ACTION_DICT = {
    "tab": 0,
    "content": 1,
    "selection_command": 2,
    "selection_mouse": 3,
    "selection_keyboard": 4,
    "terminal_command": 5,
    "terminal_output": 6,
    "terminal_focus": 7,
    "git_branch_checkout": 8
} 

def get_dataloaders(data_root, batch_size, seq_len, frame_mode="diff", is_distributed=True):
    all_videos = sorted(list(Path(data_root).glob("*.mp4")))
    random.seed(42) 
    random.shuffle(all_videos)

    split_idx = int(len(all_videos) * 0.8)
    train_videos = all_videos[:split_idx]
    val_videos = all_videos[split_idx:]

    predecoded_root = "/data"

    train_ds = SequenceDataset(train_videos, data_root, seq_len, frame_mode, predecoded_root=predecoded_root)
    val_ds = SequenceDataset(val_videos, data_root, seq_len, frame_mode, predecoded_root=predecoded_root)

    train_sampler = DistributedSampler(train_ds) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    return train_loader, val_loader, train_sampler, val_sampler

class SequenceDataset(Dataset):
    def __init__(self, video_files, label_root, seq_len, frame_mode, predecoded_root, cache_size=2):
        super().__init__()
        self.seq_len = seq_len
        self.frame_mode = frame_mode
        self.label_root = Path(label_root)
        self.video_paths = video_files
        self.predecoded_root = Path(predecoded_root)
        self.frame_before_all = []
        self.frame_after_all = []
        self.action_id_all = []
        self.cumulative_sequences = [0]
        
        self.frame_cache = OrderedDict()
        self.cache_size = cache_size

        total_valid_sequences = 0
        for v_path in self.video_paths:
            vid_id = v_path.stem
            label_file = self.label_root / f"{vid_id}_labels" / "keystrokes.jsonl"
            
            with open(label_file, 'r') as f:
                vid_labels = [json.loads(line) for line in f]

            actual_frame_count = self._get_frame_count(v_path)
            
            frame_before = []
            frame_after = []
            action_id = []

            for l in vid_labels:
                fb = int(l.get("video_frame_before", 0))
                fa = int(l.get("video_frame_after", 0))
                if fb >= actual_frame_count or fa >= actual_frame_count:
                    continue
                frame_before.append(fb)
                frame_after.append(fa)
                action_id.append(ACTION_DICT[l["action_type"]])

            frame_before_arr = np.asarray(frame_before, dtype=np.int64)
            frame_after_arr = np.asarray(frame_after, dtype=np.int64)
            action_id_arr = np.asarray(action_id, dtype=np.int64)

            self.frame_before_all.append(frame_before_arr)
            self.frame_after_all.append(frame_after_arr)
            self.action_id_all.append(action_id_arr)

            num_sequences = max(0, len(action_id_arr) - seq_len + 1)
            total_valid_sequences += num_sequences
            self.cumulative_sequences.append(total_valid_sequences)

    def _get_frame_count(self, v_path):
        n = len(str(v_path))
        p = f"{str(v_path)[:n-4]}_frames_128.npy"
        arr = np.load(p, mmap_mode="r")
        return int(arr.shape[0])

    def _get_frames(self, v_path):
        n = len(str(v_path))
        p = f"{str(v_path)[:n-4]}_frames_128.npy"

        if p in self.frame_cache:
            self.frame_cache.move_to_end(p)
            return self.frame_cache[p]

        arr = np.load(p, mmap_mode="r")
        self.frame_cache[p] = arr
        if len(self.frame_cache) > self.cache_size:
            self.frame_cache.popitem(last=False)
        return arr

    def __len__(self):
        return self.cumulative_sequences[-1]

    def __getitem__(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sequences, idx) - 1
        local_idx = idx - self.cumulative_sequences[video_idx]
        
        v_path = str(self.video_paths[video_idx])
        fb_all = self.frame_before_all[video_idx]
        fa_all = self.frame_after_all[video_idx]
        action_all = self.action_id_all[video_idx]
        pre = self._get_frames(v_path)
        
        indices = np.empty(self.seq_len * 2, dtype=np.int64)
        indices[0::2] = fb_all[local_idx : local_idx + self.seq_len]
        indices[1::2] = fa_all[local_idx : local_idx + self.seq_len]

        all_frames = torch.from_numpy(pre[indices])

        all_frames = all_frames.permute(0, 3, 1, 2)
        all_frames = all_frames.view(self.seq_len, 2, all_frames.shape[1], all_frames.shape[2], all_frames.shape[3])
        img_before = all_frames[:, 0]
        img_after = all_frames[:, 1]

        if self.frame_mode == "diff":
            frames = (img_after - img_before).abs()
        elif self.frame_mode == "concat":
            frames = torch.cat([img_before, img_after], dim=1)
        else:
            raise NotImplementedError(f"Unknown frame_mode: {self.frame_mode}")

        actions = torch.from_numpy(action_all[local_idx : local_idx + self.seq_len])

        return {"frames": frames, "actions": actions}