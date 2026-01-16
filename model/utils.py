import cv2
import numpy as np
import torch.distributed as dist
COLOR_WHITE = (255, 255, 255)
COLOR_GT = (255, 200, 100)  # Light blue for ground truth
COLOR_CORRECT = (0, 255, 0)  # Green for correct predictions
COLOR_INCORRECT = (0, 0, 255)  # Red for incorrect predictions
COLOR_NOOP = (128, 128, 128)  # Gray for noop/no key


def render_annotated_video(input_video_path,  output_video_path, key_predictions, key_ground_truth, frame_indices, id_to_key=None, display_duration=10):
    cap = cv2.VideoCapture(input_video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Predictions: {len(key_predictions)}, GT: {len(key_ground_truth)}, Frame indices: {len(frame_indices)}")
    print(f"Display duration: {display_duration} frames (~{display_duration/fps:.1f}s)")
    
    non_zero_gt = (np.array(key_ground_truth) != 0).sum()
    non_zero_pred = (np.array(key_predictions) != 0).sum()
    print(f"Non-zero GT keys: {non_zero_gt}, Non-zero predicted keys: {non_zero_pred}")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    margin = 10
    
    annotations_list = []
    
    for idx in range(len(key_predictions)):
        key_gt = key_ground_truth[idx]
        key_pred = key_predictions[idx]
        
        if isinstance(frame_indices[idx], (list, tuple, np.ndarray)):
            frame_num = int(frame_indices[idx][0])
        else:
            frame_num = int(frame_indices[idx])
        
        if key_gt == 0 and key_pred == 0:
            continue
        
        ann = {
            'start_frame': frame_num,
            'end_frame': frame_num + display_duration,
            'key_pred': int(key_pred),
            'key_gt': int(key_gt),
            'key_correct': int(key_pred) == int(key_gt)
        }
        
        if id_to_key:
            ann['key_pred_name'] = id_to_key.get(int(key_pred), f"key_{key_pred}")
            ann['key_gt_name'] = id_to_key.get(int(key_gt), f"key_{key_gt}")
        else:
            ann['key_pred_name'] = str(key_pred)
            ann['key_gt_name'] = str(key_gt)
        
        annotations_list.append(ann)
    
    print(f"Total annotations to render: {len(annotations_list)}")
    
    frame_idx = 0
    rendered_annotations = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        y_offset = margin + 20
        
        # Frame counter
        cv2.putText(
            frame, f"Frame: {frame_idx}/{total_frames}", 
            (margin, y_offset), font, font_scale, COLOR_WHITE, thickness, cv2.LINE_AA
        )
        y_offset += 25
        
        active_annotations = [ann for ann in annotations_list if ann['start_frame'] <= frame_idx < ann['end_frame']]
        
        for ann in active_annotations:
            rendered_annotations += 1
            key_correct = ann['key_correct']
            key_gt = ann['key_gt']
            key_pred = ann['key_pred']
            
            frames_remaining = ann['end_frame'] - frame_idx
            alpha = min(1.0, frames_remaining / 10.0)  # Fade in last 10 frames
            
            if key_gt == 0 and key_pred == 0:
                color = COLOR_NOOP
            elif key_correct:
                color = COLOR_CORRECT
            else:
                color = COLOR_INCORRECT
            
            box_height = 70
            overlay = frame.copy()
            cv2.rectangle(
                overlay, 
                (margin - 5, y_offset - 15), 
                (margin + 300, y_offset + box_height), 
                (0, 0, 0), 
                -1
            )
            cv2.addWeighted(overlay, 0.6 * alpha, frame, 1 - 0.6 * alpha, 0, frame)
            
            # Key ground truth
            key_gt_text = f"GT Key: {ann['key_gt_name']}"
            cv2.putText(frame, key_gt_text, (margin, y_offset), font, font_scale, COLOR_GT, thickness, cv2.LINE_AA)
            y_offset += 22
            
            # Key prediction
            key_pred_text = f"Pred Key: {ann['key_pred_name']}"
            cv2.putText(frame, key_pred_text, (margin, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
            y_offset += 22
            
            # Status
            status = "KEY: OK" if key_correct else "KEY: WRONG"
            cv2.putText(frame, status, (margin, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
            y_offset += 30
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Rendered {rendered_annotations} annotation instances across {frame_idx} frames")
    print(f"Annotated video saved to: {output_video_path}")



def compute_accuracy(predictions, ground_truth, id_to_action, action_dict):
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    num_actions = len(action_dict)
    
    correct = (predictions == ground_truth).sum()
    total = len(ground_truth)
    overall_acc = correct / total if total > 0 else 0.0
    
    per_class_acc = {}
    confusion_matrix = np.zeros((num_actions, num_actions), dtype=np.int64)
    
    for pred, gt in zip(predictions, ground_truth):
        confusion_matrix[gt, pred] += 1
    
    for c in range(num_actions):
        class_name = id_to_action[c]
        class_total = (ground_truth == c).sum()
        class_correct = confusion_matrix[c, c]
        
        if class_total > 0:
            per_class_acc[class_name] = {'accuracy': class_correct / class_total, 'correct': int(class_correct), 'total': int(class_total)}
        else:
            per_class_acc[class_name] = {'accuracy': 0.0, 'correct': 0, 'total': 0}
    
    return {'overall_accuracy': overall_acc, 'correct': int(correct), 'total': int(total), 'per_class': per_class_acc}

def print_results(accuracy_results):   
    print(f"\nOverall Accuracy: {accuracy_results['overall_accuracy']:.2%}")
    print(f"Correct: {accuracy_results['correct']} / {accuracy_results['total']}")
    
    print("\nPer-Class Accuracy:")
    print("-" * 50)
    for class_name, stats in accuracy_results['per_class'].items():
        if stats['total'] > 0:
            print(f"  {class_name:20s}: {stats['accuracy']:6.2%} ({stats['correct']:5d}/{stats['total']:5d})")
        else:
            print(f"  {class_name:20s}: N/A (no samples)")

def compute_training_accuracy(metrics, conf_matrix, data_loader, local_rank, num_actions):
    avg_loss = metrics[0] / (len(data_loader) * dist.get_world_size())
    avg_acc = metrics[1] / metrics[2]

    if dist.get_rank() == 0:
        print("\nPer-Class Accuracy:")
        for c in range(num_actions):
            gt_total = conf_matrix[c].sum().item()
            correct_c = conf_matrix[c,c].item()
            acc_c = correct_c / gt_total if gt_total > 0 else 0.0
            print(f"  class {c:2d}: {acc_c:6.2%}  ({int(correct_c)}/{int(gt_total)})")
    return avg_loss, avg_acc
