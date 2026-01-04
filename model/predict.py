import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from idm_video import KeystrokeIDM
from sequence_dataset import ACTION_DICT
from utils import render_annotated_video, compute_accuracy, print_results

ID_TO_ACTION = {v: k for k, v in ACTION_DICT.items()}
NUM_ACTIONS = len(ACTION_DICT)

def load_model(checkpoint_path, device, frame_mode="concat"):
    model = KeystrokeIDM(
        num_actions=NUM_ACTIONS,
        d_model=512,
        num_transformer_layers=4,
        num_heads=8,
        ff_dim=4096,
        frame_mode=frame_mode
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        print("info: removing 'module.' prefix from state_dict keys")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def load_ground_truth(video_path):
    video_stem = Path(video_path).stem.replace('_frames_128', '')
    label_file = Path(os.path.dirname(video_path)) / f"{video_stem}_labels" / "keystrokes.jsonl"
    
    frames = np.load(video_path, mmap_mode='r')
    num_frames = frames.shape[0]
    
    action_map = {}
    if label_file.exists():
        with open(label_file, 'r') as f:
            for line in f:
                label = json.loads(line)
                fb = int(label.get("video_frame_before", 0))
                fa = int(label.get("video_frame_after", 0))
                if fb < num_frames and fa < num_frames:
                    action_map[(fb, fa)] = ACTION_DICT[label["action_type"]]
    
    frame_indices = []
    actions = []
    
    for i in range(num_frames - 1):
        fb, fa = i, i + 1
        frame_indices.append((fb, fa))
        actions.append(action_map.get((fb, fa), ACTION_DICT["noop"]))
    
    frame_indices = np.array(frame_indices, dtype=np.int64)
    actions = np.array(actions, dtype=np.int64)
    
    print(f"Loaded {len(actions)} dense labels ({(actions != 0).sum()} non-noop)")
    return frame_indices, actions


def prepare_frame_pairs(frames, frame_indices, frame_mode="concat"):
    img_before = torch.from_numpy(frames[frame_indices[:, 0]]).float()
    img_after = torch.from_numpy(frames[frame_indices[:, 1]]).float()
    
    img_before = img_before.permute(0, 3, 1, 2)
    img_after = img_after.permute(0, 3, 1, 2)
    
    if frame_mode == "diff":
        prepared = (img_after - img_before).abs()
    elif frame_mode == "concat":
        prepared = torch.cat([img_before, img_after], dim=1)
    else:
        raise ValueError(f"Unknown frame_mode: {frame_mode}")
    
    return prepared


def run_inference(model, frames, frame_indices, device, frame_mode="concat", seq_len=16):
    num_samples = len(frame_indices)
    all_predictions = []
    all_logits = []
    
    prepared_frames = prepare_frame_pairs(frames, frame_indices, frame_mode)
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, seq_len):
            end_idx = min(start_idx + seq_len, num_samples)
            actual_len = end_idx - start_idx
            
            batch_frames = prepared_frames[start_idx:end_idx]
            
            if actual_len < seq_len:
                padding = torch.zeros(seq_len - actual_len, *batch_frames.shape[1:])
                batch_frames = torch.cat([batch_frames, padding], dim=0)
            
            batch_frames = batch_frames.unsqueeze(0).to(device)
            
            logits = model(batch_frames)
            logits = logits[0, :actual_len]
            
            preds = logits.argmax(dim=-1)
            
            all_predictions.extend(preds.cpu().tolist())
            all_logits.append(logits.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    return np.array(all_predictions), all_logits


if __name__ == "__main__":
    VIDEO_ID = "59814d20919e0b1"
    DATA_ROOT = "/data"
    OUTPUT_ROOT = "."
    
    npy_path = f"{DATA_ROOT}/vid_{VIDEO_ID}_frames_128.npy"
    video_path = f"{DATA_ROOT}/vid_{VIDEO_ID}.mp4"
    checkpoint = f"{OUTPUT_ROOT}/idm_checkpoint_ep1.pth"
    output_video_path = f"{OUTPUT_ROOT}/annotated_{VIDEO_ID}_predictions.mp4"
    
    FRAME_MODE = "concat"
    SEQ_LEN = 16
    
    RENDER = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(checkpoint, device, FRAME_MODE)
    
    frames = np.load(npy_path)
    frame_indices, ground_truth = load_ground_truth(npy_path)
    
    predictions, logits = run_inference(model, frames, frame_indices, device, FRAME_MODE, SEQ_LEN)
    
    print(f"Generated {len(predictions)} predictions")
    
    accuracy_results = compute_accuracy(predictions, ground_truth, ID_TO_ACTION, ACTION_DICT)
    print_results(accuracy_results)
    
    if RENDER:
        render_annotated_video(input_video_path=video_path, output_video_path=output_video_path, predictions=predictions, ground_truth=ground_truth, frame_indices=frame_indices, id_to_action=ID_TO_ACTION, action_dict=ACTION_DICT)
