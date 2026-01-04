import cv2
import numpy as np
import torch.distributed as dist

COLOR_CORRECT = (0, 255, 0)
COLOR_INCORRECT = (0, 0, 255)
COLOR_NOOP = (128, 128, 128)
COLOR_GT = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)

def render_annotated_video(input_video_path, output_video_path, predictions, ground_truth, frame_indices, id_to_action, action_dict):
    cap = cv2.VideoCapture(input_video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    margin = 10
    
    frame_annotations = {}
    
    for idx, (fb, fa) in enumerate(frame_indices):
        gt = ground_truth[idx]
        pred = predictions[idx]
        
        if gt == action_dict["noop"] and pred == action_dict["noop"]:
            continue
        
        if fa not in frame_annotations:
            frame_annotations[fa] = []
        
        frame_annotations[fa].append({'gt': gt, 'pred': pred, 'gt_name': id_to_action[gt], 'pred_name': id_to_action[pred], 'correct': gt == pred})
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        y_offset = margin + 20
        
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (margin, y_offset), font, font_scale, COLOR_WHITE, thickness, cv2.LINE_AA)
        y_offset += 25
        
        if frame_idx in frame_annotations:
            for ann in frame_annotations[frame_idx]:
                gt_name = ann['gt_name']
                pred_name = ann['pred_name']
                correct = ann['correct']
                
                if ann['gt'] == action_dict["noop"] and ann['pred'] == action_dict["noop"]:
                    color = COLOR_NOOP
                elif correct:
                    color = COLOR_CORRECT
                else:
                    color = COLOR_INCORRECT
                
                gt_text = f"GT: {gt_name}"
                cv2.putText(frame, gt_text, (margin, y_offset), font, font_scale, COLOR_GT, thickness, cv2.LINE_AA)
                y_offset += 25
                
                pred_text = f"Pred: {pred_name}"
                cv2.putText(frame, pred_text, (margin, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
                y_offset += 25
                
                status = "CORRECT" if correct else "WRONG"
                cv2.putText(frame, status, (margin, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
                y_offset += 30
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()

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
