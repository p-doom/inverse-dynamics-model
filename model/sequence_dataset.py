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
from actions import ACTION_DICT, KEY_DICT, NUM_KEYS

def text_to_key_id(text):
    if not text or text == "":
        return KEY_DICT["<pad>"]
    
    if text == "\\n" or text == "\n":
        return KEY_DICT["<enter>"]
    if text == "\\r" or text == "\r":
        return KEY_DICT["<enter>"]
    if text == " ":
        return KEY_DICT["<space>"]
    
    if len(text) == 1:
        return KEY_DICT.get(text, KEY_DICT["<unk>"])
    
    return KEY_DICT["<unk>"]


def get_dataloaders(data_root, batch_size, seq_len, frame_mode="diff", is_distributed=True):
    all_videos = sorted(list(Path(data_root).glob("*.mp4")))
    random.seed(42) 
    random.shuffle(all_videos)

    split_idx = int(len(all_videos) * 0.8)
    train_videos = all_videos[:split_idx]
    val_videos = all_videos[split_idx:]

    train_ds = SequenceDataset(train_videos, data_root, seq_len, frame_mode, predecoded_root=data_root)
    val_ds = SequenceDataset(val_videos, data_root, seq_len, frame_mode, predecoded_root=data_root)

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
        
        self.frame_indices_all = [] 
        self.action_id_all = []
        self.cursor_pos_all = []
        self.key_id_all = []
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
            
            action_map = {}
            cursor_map = {}
            key_map = {}
            
            for l in vid_labels:
                fb = int(l.get("video_frame_before", 0))
                fa = int(l.get("video_frame_after", 0))
                if fb >= actual_frame_count or fa >= actual_frame_count:
                    continue
                    
                action_type = l["action_type"]
                action_map[(fb, fa)] = ACTION_DICT[action_type]
                
                cursor_x = float(l.get("cursor", {}).get("x", 0.0))
                cursor_y = float(l.get("cursor", {}).get("y", 0.0))
                cursor_map[(fb, fa)] = (cursor_x, cursor_y)
                
                if action_type == "content":
                    text = l.get("text", "")
                    key_map[(fb, fa)] = text_to_key_id(text)
                else:
                    key_map[(fb, fa)] = KEY_DICT["<pad>"]
            
            frame_indices = []
            action_ids = []
            cursor_positions = []
            key_ids = []
            
            for i in range(actual_frame_count - 1):
                fb, fa = i, i + 1
                frame_indices.append((fb, fa))
                action_ids.append(action_map.get((fb, fa), ACTION_DICT["noop"]))
                cursor_positions.append(cursor_map.get((fb, fa), (0.0, 0.0)))
                key_ids.append(key_map.get((fb, fa), KEY_DICT["<pad>"]))
            
            self.frame_indices_all.append(np.asarray(frame_indices, dtype=np.int64))
            self.action_id_all.append(np.asarray(action_ids, dtype=np.int64))
            self.cursor_pos_all.append(np.asarray(cursor_positions, dtype=np.float32))
            self.key_id_all.append(np.asarray(key_ids, dtype=np.int64))

            num_sequences = max(0, len(action_ids) - seq_len + 1)
            total_valid_sequences += num_sequences
            self.cumulative_sequences.append(total_valid_sequences)

    def _get_frame_count(self, v_path):
        n = len(str(v_path))
        p = f"{str(v_path)[:n-4]}_frames_512.npy"
        arr = np.load(p, mmap_mode="r")
        return int(arr.shape[0])

    def _get_frames(self, v_path):
        n = len(str(v_path))
        p = f"{str(v_path)[:n-4]}_frames_512.npy"

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
        frame_indices = self.frame_indices_all[video_idx]
        action_all = self.action_id_all[video_idx]
        cursor_all = self.cursor_pos_all[video_idx]
        key_all = self.key_id_all[video_idx]
        pre = self._get_frames(v_path)
        
        seq_frame_indices = frame_indices[local_idx : local_idx + self.seq_len]
        
        indices = np.empty(self.seq_len * 2, dtype=np.int64)
        indices[0::2] = seq_frame_indices[:, 0] 
        indices[1::2] = seq_frame_indices[:, 1] 

        all_frames = torch.from_numpy(pre[indices].copy())

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

        actions = torch.from_numpy(action_all[local_idx : local_idx + self.seq_len].copy())
        cursor = torch.from_numpy(cursor_all[local_idx : local_idx + self.seq_len].copy())
        keys = torch.from_numpy(key_all[local_idx : local_idx + self.seq_len].copy())

        return {"frames": frames, "actions": actions, "cursor": cursor, "keys": keys, "video_path": str(v_path), "start_idx": local_idx}
