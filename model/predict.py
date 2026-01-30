import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from idm_video import KeystrokeIDM
from actions import ACTION_DICT, NUM_ACTIONS, NUM_KEYS, KEY_DICT
from utils import render_annotated_video, compute_accuracy, print_results

ID_TO_ACTION = {v: k for k, v in ACTION_DICT.items()}

with open('default.yaml', 'r') as f:
    config = yaml.safe_load(f)

def plot_confusion_matrix(y_true, y_pred, num_keys, output_path):
    """Generates and saves a confusion matrix heatmap."""
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_keys))
    cm[0, 0] = 0
    
    # Plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=range(num_keys), yticklabels=range(num_keys))
    plt.xlabel('Predicted Key')
    plt.ylabel('True Key')
    plt.title('Keystroke Prediction Confusion Matrix')
    
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def load_model(checkpoint_path, device, frame_mode):
    """Load the trained KeystrokeIDM model from a checkpoint."""
    model = KeystrokeIDM(
        num_keys=NUM_KEYS,
        d_model=config["model"]["d_model"],
        num_transformer_layers=3,
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
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    if checkpoint.get('val_key_loss'):
        print(f"  Val Key Loss: {checkpoint.get('val_key_loss'):.4f}")
    
    return model


def load_ground_truth(video_path):
    """Load ground truth labels from the corresponding JSONL file."""
    video_stem = Path(video_path).stem.replace('_frames_512', '')
    label_file = Path(os.path.dirname(video_path)) / f"{video_stem}_labels" / "keystrokes.jsonl"
    
    frames = np.load(video_path, mmap_mode='r')
    num_frames = frames.shape[0]
    
    key_map = {}
    
    if label_file.exists():
        with open(label_file, 'r') as f:
            for line in f:
                label = json.loads(line)
                fb = int(label.get("video_frame_before", 0))
                fa = int(label.get("video_frame_after", 0))
                print(label, fb, fa)
                if fb < num_frames and fa < num_frames:
                    print("LABEL: ", label)
                    if "text" in label and len(label["text"]) == 1:
                        key_map[(fb, fa)] = KEY_DICT.get(label["text"], 0)
                    else:
                        key_map[(fb, fa)] = 0
    
    frame_indices = []
    keys = []
    
    for i in range(num_frames - 1):
        fb, fa = i, i + 1
        frame_indices.append((fb, fa))
        keys.append(key_map.get((fb, fa), 0))
    
    frame_indices = np.array(frame_indices, dtype=np.int64)
    keys = np.array(keys, dtype=np.int64)
    
    print(f"Loaded {len(keys)} dense labels ({(keys != 0).sum()} non-zero keys)")
    return frame_indices, keys


def prepare_frame_pairs(frames, frame_indices, frame_mode="concat"):
    """Prepare frame pairs for model input based on the specified mode."""
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
    """Run inference on the video frames and return key predictions."""
    num_samples = len(frame_indices)
    
    all_key_predictions = []
    all_key_logits = []
    
    prepared_frames = prepare_frame_pairs(frames, frame_indices, frame_mode)
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, seq_len):
            end_idx = min(start_idx + seq_len, num_samples)
            actual_len = end_idx - start_idx
            
            batch_frames = prepared_frames[start_idx:end_idx]
            
            # Pad if necessary
            if actual_len < seq_len:
                padding = torch.zeros(seq_len - actual_len, *batch_frames.shape[1:])
                batch_frames = torch.cat([batch_frames, padding], dim=0)
            
            batch_frames = batch_frames.unsqueeze(0).to(device)
            
            # Model returns only key_logits now
            key_logits = model(batch_frames)
            
            # Extract only the valid (non-padded) predictions
            key_logits = key_logits[0, :actual_len]
            
            key_preds = key_logits.argmax(dim=-1)
            
            all_key_predictions.extend(key_preds.cpu().tolist())
            all_key_logits.append(key_logits.cpu())
    
    all_key_logits = torch.cat(all_key_logits, dim=0)
    
    return {
        'key_predictions': np.array(all_key_predictions),
        'key_logits': all_key_logits
    }


def compute_key_accuracy(key_predictions, key_ground_truth):
    """Compute key prediction accuracy."""
    correct = (key_predictions == key_ground_truth).sum()
    total = len(key_ground_truth)
    return correct / total if total > 0 else 0.0


def compute_per_key_accuracy(key_predictions, key_ground_truth, num_keys=NUM_KEYS):
    """Compute per-key accuracy."""
    per_key_acc = {}
    per_key_counts = {}
    
    for i in range(num_keys):
        mask = key_ground_truth == i
        count = mask.sum()
        if count > 0:
            correct = ((key_predictions == i) & mask).sum()
            per_key_acc[i] = correct / count
            per_key_counts[i] = int(count)
        else:
            per_key_acc[i] = None
            per_key_counts[i] = 0
    
    return per_key_acc, per_key_counts


def print_detailed_results(predictions, ground_truth, output_dir=None):
    """Print detailed results for key predictions."""
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    
    # Key accuracy
    key_acc = compute_key_accuracy(
        predictions['key_predictions'],
        ground_truth['keys']
    )
    print("\n--- Key Predictions ---")
    print(f"Overall Key Accuracy: {key_acc:.2%}")
    
    # Per-key accuracy
    per_key_acc, per_key_counts = compute_per_key_accuracy(
        predictions['key_predictions'],
        ground_truth['keys']
    )
    
    print("\nPer-Key Accuracies:")
    for i in range(NUM_KEYS):
        if per_key_counts[i] > 0:
            print(f"  Key {i}: {per_key_acc[i]:.2%} (count: {per_key_counts[i]})")
    
    # Summary
    print("\n--- Summary ---")
    print(f"Total Frames Processed: {len(predictions['key_predictions'])}")
    non_zero = (predictions['key_predictions'] != 0).sum()
    print(f"Non-Zero Key Predictions: {non_zero}")

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        ground_truth['keys'], 
        predictions['key_predictions'], 
        NUM_KEYS, 
        cm_path
    )
    
    return {
        'key_accuracy': key_acc,
        'per_key_accuracy': per_key_acc,
        'per_key_counts': per_key_counts
    }


def save_predictions(predictions, ground_truth, output_path):
    """Save predictions to a JSON file for later analysis."""
    results = {
        'num_frames': len(predictions['key_predictions']),
        'predictions': {
            'keys': predictions['key_predictions'].tolist()
        },
        'ground_truth': {
            'keys': ground_truth['keys'].tolist()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    # Configuration
    VIDEO_ID = "b0916880d3f8ca2"
    DATA_ROOT = config["data_dir"]
    OUTPUT_ROOT = config["output_dir"]
    
    npy_path = f"{DATA_ROOT}/vid_{VIDEO_ID}_frames_512.npy"
    checkpoint = f"{OUTPUT_ROOT}/idm_checkpoint_ep9.pth"
    output_json_path = f"{OUTPUT_ROOT}/predictions_{VIDEO_ID}.json"
    
    SAVE_JSON = True
    RENDER = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(checkpoint, device, config["model"]["frame_mode"])
    
    # Load data
    frames = np.load(npy_path)
    frame_indices, gt_keys = load_ground_truth(npy_path)
    
    ground_truth = {
        'keys': gt_keys
    }
    
    # Run inference
    print(f"\nRunning inference on {len(frame_indices)} frame pairs...")
    predictions = run_inference(
        model, 
        frames, 
        frame_indices, 
        device, 
        config["model"]["frame_mode"], 
        config["model"]["seq_len"]
    )
    
    print(f"Generated {len(predictions['key_predictions'])} predictions")
    
    results = print_detailed_results(predictions, ground_truth, output_dir=OUTPUT_ROOT)
    
    if SAVE_JSON:
        save_predictions(predictions, ground_truth, output_json_path)
    if RENDER:
        video_path = f"{DATA_ROOT}/vid_{VIDEO_ID}.mp4"
        output_video_path = f"{OUTPUT_ROOT}/annotated_{VIDEO_ID}_keys.mp4"
        
        ID_TO_KEY = {v: k for k, v in KEY_DICT.items()}
        
        print(f"\nRendering annotated video to {output_video_path}...")
        render_annotated_video(
            input_video_path=video_path,
            output_video_path=output_video_path,
            key_predictions=predictions['key_predictions'],
            key_ground_truth=ground_truth['keys'],
            frame_indices=frame_indices,
            id_to_key=ID_TO_KEY
        )
        print("Video rendering complete!")
