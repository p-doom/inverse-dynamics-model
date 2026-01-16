import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import json
import yaml

from sequence_dataset import get_dataloaders
from actions import ACTION_DICT, NUM_ACTIONS, NUM_KEYS
from idm_video import KeystrokeIDM
from utils import compute_training_accuracy


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


def train(model, dataloader, key_criterion, optimizer, scheduler, device, local_rank):
    model.train()
    total_key_loss = 0.0
    key_correct = 0
    total = 0

    disable_tqdm = (local_rank != 0)
    loop = tqdm(dataloader, desc="Training", disable=disable_tqdm)

    for step, batch in enumerate(loop):
        frames = batch["frames"].to(device, non_blocking=True).float()
        key_targets = batch["keys"].to(device, non_blocking=True).long()

        optimizer.zero_grad()
        key_logits = model(frames)
        B, T, C = key_logits.shape
        
        loss = key_criterion(key_logits.view(B * T, NUM_KEYS), key_targets.view(B * T))
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_key_loss += loss.item()

        key_preds = key_logits.argmax(dim=-1)
        key_correct += (key_preds == key_targets).sum().item()
        total += key_targets.numel()

        if not disable_tqdm:
            loop.set_postfix(key_loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    
    metrics = torch.tensor([total_key_loss, total, key_correct], device=device).float()
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    num_batches = len(dataloader) * dist.get_world_size()
    avg_key_loss = metrics[0].item() / num_batches
    avg_key_acc = metrics[2].item() / metrics[1].item()

    return avg_key_loss, avg_key_acc


def validate(model, val_loader, key_criterion, device):
    model.eval()
    total_key_loss = 0.0
    key_correct = 0
    total = 0

    key_conf_matrix = torch.zeros((NUM_KEYS, NUM_KEYS), device=device)

    with torch.no_grad():
        for batch in val_loader:
            frames = batch["frames"].to(device, non_blocking=True).float()
            key_targets = batch["keys"].to(device, non_blocking=True).long()

            key_logits = model(frames)
            B, T, C = key_logits.shape

            key_loss = key_criterion(key_logits.view(B * T, NUM_KEYS), key_targets.view(B * T))
            
            total_key_loss += key_loss.item()

            key_preds = key_logits.argmax(dim=-1)
            key_correct += (key_preds == key_targets).sum().item()
            total += key_targets.numel()

            flat_key_t = key_targets.view(-1)
            flat_key_p = key_preds.view(-1)
            key_idx = flat_key_t * NUM_KEYS + flat_key_p
            key_bincount = torch.bincount(key_idx, minlength=NUM_KEYS * NUM_KEYS)
            key_conf_matrix += key_bincount.view(NUM_KEYS, NUM_KEYS)

    metrics = torch.tensor([total_key_loss, total, key_correct], device=device).float()
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    dist.all_reduce(key_conf_matrix, op=dist.ReduceOp.SUM)

    num_batches = len(val_loader) * dist.get_world_size()
    avg_key_loss = metrics[0].item() / num_batches
    avg_key_acc = metrics[2].item() / metrics[1].item()

    per_key_acc = torch.nan_to_num(key_conf_matrix.diag() / key_conf_matrix.sum(1))

    return avg_key_loss, avg_key_acc, per_key_acc 


if __name__ == "__main__":
    local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    with open('default.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, train_sampler, val_sampler = get_dataloaders(config["data_dir"], config["training"]["batch_size"], config["model"]["seq_len"], frame_mode=config["model"]["frame_mode"], is_distributed=True)

    model = KeystrokeIDM(num_actions=NUM_ACTIONS, num_keys=NUM_KEYS, d_model=config["model"]["d_model"], num_transformer_layers=3, num_heads=8, ff_dim=4096, frame_mode=config["model"]["frame_mode"]).to(device)
    
    model = DDP(model, device_ids=[local_rank])

    if dist.get_rank() == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Rank 0: {num_params:,} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = OneCycleLR(optimizer, max_lr=config["training"]["lr"], total_steps=total_steps, pct_start=0.1, anneal_strategy='cos')
    
    counts = torch.tensor([1000, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    weights = 1.0 / torch.sqrt(counts + 1) 
    weights = weights / weights.sum() * len(counts)
    weights = weights.to(device)

    key_criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    for epoch in range(config["training"]["epochs"]):
        train_sampler.set_epoch(epoch) 
        
        train_key_loss, train_key_acc = train(model, train_loader, key_criterion, optimizer, scheduler, device, local_rank)
        val_key_loss, val_key_acc, per_key_acc = validate(model, val_loader, key_criterion, device)

        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1}/{config['training']['epochs']} | Train: key={train_key_loss:.4f} key_acc={train_key_acc:.2%} | Val: key={val_key_loss:.4f} key_acc={val_key_acc:.2%}")

            print("Per-key accuracies:")
            key_counts = per_key_acc.new_zeros(NUM_KEYS)
            with torch.no_grad():
                for batch in val_loader:
                    key_targets = batch["keys"].to(device, non_blocking=True).long()
                    for i in range(NUM_KEYS):
                        key_counts[i] += (key_targets == i).sum().item()
            for i, acc in enumerate(per_key_acc):
                print(f"  Key {i}: {acc:.2%} (count: {int(key_counts[i].item())})")

            if epoch % 2 == 0:
                try:
                    save_path = f"idm_checkpoint_ep{epoch+1}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_key_loss': val_key_loss
                    }, save_path)
                except RuntimeError as e:
                    print(f"Warning: Could not save checkpoint at epoch {epoch+1}: {e}")

    dist.destroy_process_group()