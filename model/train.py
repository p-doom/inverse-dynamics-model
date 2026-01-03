import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import json

from sequence_dataset import get_dataloaders, ACTION_DICT
from idm_video import KeystrokeIDM

NUM_ACTIONS = len(ACTION_DICT)

def setup_dist():
    if 'LOCAL_RANK' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def get_acc(metrics, conf_matrix, data_loader, local_rank):
    avg_loss = metrics[0] / (len(data_loader) * dist.get_world_size())
    avg_acc = metrics[1] / metrics[2]

    if dist.get_rank() == 0:
        print("\nPer-Class Accuracy:")
        for c in range(NUM_ACTIONS):
            gt_total = conf_matrix[c].sum().item()
            correct_c = conf_matrix[c,c].item()
            acc_c = correct_c / gt_total if gt_total > 0 else 0.0
            print(f"  class {c:2d}: {acc_c:6.2%}  ({int(correct_c)}/{int(gt_total)})")
    return avg_loss, avg_acc

def log_val_predictions(model, val_loader, device, local_rank, epoch, log_path="val_predictions.jsonl"):
    model.eval()
    all_predictions = []
    
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    with torch.no_grad():
        for batch in val_loader:
            frames = batch["frames"].to(device, non_blocking=True).float()
            targets = batch["actions"].to(device, non_blocking=True).long()
            video_paths = batch["video_path"]
            start_indices = batch["start_idx"]
            
            logits = model(frames)
            preds = logits.argmax(dim=-1) 
            
            preds_cpu = preds.cpu().tolist()
            targets_cpu = targets.cpu().tolist()
            
            for i in range(len(video_paths)):
                all_predictions.append({
                    "video_path": video_paths[i],
                    "start_idx": start_indices[i].item() if torch.is_tensor(start_indices[i]) else start_indices[i],
                    "predictions": preds_cpu[i],
                    "targets": targets_cpu[i],
                })
    
    if world_size > 1:
        if global_rank == 0:
            gathered_results = [None] * world_size
            dist.gather_object(all_predictions, object_gather_list=gathered_results, dst=0)
            all_predictions = [item for sublist in gathered_results for item in sublist]
        else:
            dist.gather_object(all_predictions, dst=0)
    
    if global_rank == 0:
        with open(log_path, 'a') as f:
            f.write(f"# Epoch {epoch + 1}\n")
            for pred_entry in all_predictions:
                f.write(json.dumps(pred_entry) + "\n")
        print(f"Logged {len(all_predictions)} validation predictions to {log_path}")


def train(model, dataloader, criterion, optimizer, scheduler, device, local_rank):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    conf_matrix = torch.zeros((NUM_ACTIONS, NUM_ACTIONS), device=device)
    
    disable_tqdm = (local_rank != 0)
    loop = tqdm(dataloader, desc="Training", disable=disable_tqdm)

    for step, batch in enumerate(loop):
        frames = batch["frames"].to(device, non_blocking=True).float()
        targets = batch["actions"].to(device, non_blocking=True).long()

        optimizer.zero_grad()
        logits = model(frames)

        B, T, C = logits.shape
        loss = criterion(logits.view(B * T, C), targets.view(B * T))
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)

        correct += (preds == targets).sum().item()
        total += targets.numel()

        flat_t = targets.view(-1)
        flat_p = preds.view(-1)

        idx = flat_t * NUM_ACTIONS + flat_p
        bincount = torch.bincount(idx, minlength=NUM_ACTIONS * NUM_ACTIONS)
        conf_matrix += bincount.view(NUM_ACTIONS, NUM_ACTIONS)

        if not disable_tqdm:
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    
    # aggregate across GPUs
    metrics = torch.tensor([total_loss, correct, total], device=device).float()
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    dist.all_reduce(conf_matrix, op=dist.ReduceOp.SUM)

    avg_loss, avg_acc = get_acc(metrics, conf_matrix, dataloader, local_rank)

    return avg_loss, avg_acc

def validate(model, val_loader, criterion, device, local_rank):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    conf_matrix = torch.zeros((NUM_ACTIONS, NUM_ACTIONS), device=device)

    with torch.no_grad():
        for batch in val_loader:
            frames = batch["frames"].to(device, non_blocking=True).float()
            targets = batch["actions"].to(device, non_blocking=True).long()

            logits = model(frames)
            B, T, C = logits.shape

            loss = criterion(logits.view(B * T, C), targets.view(B * T))
            val_loss += loss.item()

            preds = logits.argmax(dim=-1)

            correct += (preds == targets).sum().item()
            total += targets.numel()

            flat_t = targets.view(-1)
            flat_p = preds.view(-1)

            idx = flat_t * NUM_ACTIONS + flat_p
            bincount = torch.bincount(idx, minlength=NUM_ACTIONS * NUM_ACTIONS)
            conf_matrix += bincount.view(NUM_ACTIONS, NUM_ACTIONS)

    # aggregate across GPUs
    metrics = torch.tensor([val_loss, correct, total], device=device).float()
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    dist.all_reduce(conf_matrix, op=dist.ReduceOp.SUM)

    avg_loss, avg_acc = get_acc(metrics, conf_matrix, val_loader, local_rank)

    return avg_loss, avg_acc

if __name__ == "__main__":
    local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    BATCH_SIZE = 32 
    LR = 3e-4
    EPOCHS = 10 
    SEQ_LEN = 16
    FRAME_MODE = "concat"

    train_loader, val_loader, train_sampler, val_sampler = get_dataloaders("/data", BATCH_SIZE, SEQ_LEN, frame_mode=FRAME_MODE, is_distributed=True)

    model = KeystrokeIDM(
        num_actions=NUM_ACTIONS,  
        d_model=512, 
        num_transformer_layers=4,
        num_heads=8,
        ff_dim=4096,
        frame_mode=FRAME_MODE
    ).to(device)
    
    model = DDP(model, device_ids=[local_rank])

    if dist.get_rank() == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Rank 0: {num_params:,} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = OneCycleLR(optimizer, max_lr=LR,  total_steps=total_steps, pct_start=0.1, anneal_strategy='cos')
    
    # NOTE: should be calculated from loaders
    counts = torch.tensor([48775, 296828, 565192, 170724, 179830, 14303, 1334684, 14661, 539], dtype=torch.float)
    weights = 1.0 / torch.sqrt(counts + 1) 
    weights = weights / weights.sum() * len(counts)
    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch) 
        
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, device, local_rank)
        val_loss, val_acc = validate(model, val_loader, criterion, device, local_rank)

        #log_val_predictions(model, val_loader, device, local_rank, epoch)

        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
            
            try:
                save_path = f"idm_checkpoint_ep{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
            except RuntimeError as e:
                print(f"Warning: Could not save checkpoint at epoch {epoch+1}: {e}")

    dist.destroy_process_group()