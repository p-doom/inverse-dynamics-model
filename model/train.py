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
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sequence_dataset import get_dataloaders
from actions import NUM_KEYS
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

def setup_wandb():
    wandb.init(
        entity=config["wandb_entity"],
        project=config["wandb_project"],
        name=f"ep{config['training']['epochs']}_lr{config['training']['lr']}",
        config={
            "epochs": config["training"]["epochs"],
            "lr": config["training"]["lr"],
            "batch_size": config["data"]["batch_size"] if "data" in config else config["training"]["batch_size"],
            "model": config["model"]["d_model"],
            "frame_mode": config["model"]["frame_mode"],
            "num_keys": NUM_KEYS,
        }
    )


def train(model, dataloader, key_criterion, optimizer, scheduler, device, local_rank):
    model.train()
    total_key_loss = 0.0
    key_correct = 0
    total = 0

    disable_tqdm = (local_rank != 0)
    loop = tqdm(dataloader, desc="Training", disable=disable_tqdm)

    for step, batch in enumerate(loop):
        frames = batch["frames"].to(device, non_blocking=True).float() / 255.0
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
        
        if dist.get_rank() == 0:
            wandb.log({"train/key_loss": loss.item(), "train/avg_lr": optimizer.param_groups[0]['lr']})
    
    metrics = torch.tensor([total_key_loss, total, key_correct], device=device).float()
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    num_batches = len(dataloader) * dist.get_world_size()
    avg_key_loss = metrics[0].item() / num_batches
    avg_key_acc = metrics[2].item() / metrics[1].item()

    return avg_key_loss, avg_key_acc


def validate(model, val_loader, key_criterion, device, epoch):
    model.eval()
    total_key_loss = 0.0
    key_correct = 0
    total = 0

    key_conf_matrix = torch.zeros((NUM_KEYS, NUM_KEYS), device=device)
    correct_counts = torch.zeros(NUM_KEYS, device=device)
    total_counts = torch.zeros(NUM_KEYS, device=device)

    with torch.no_grad():
        for batch in val_loader:
            frames = batch["frames"].to(device, non_blocking=True).float() / 255.0
            key_targets = batch["keys"].to(device, non_blocking=True).long()

            key_logits = model(frames)
            B, T, C = key_logits.shape

            key_loss = key_criterion(key_logits.view(B * T, NUM_KEYS), key_targets.view(B * T))
            total_key_loss += key_loss.item()

            key_preds = key_logits.argmax(dim=-1)
            key_correct += (key_preds == key_targets).sum().item()
            total += key_targets.numel()

            flat_targets = key_targets.view(-1)
            flat_preds = key_preds.view(-1)

            for i in range(NUM_KEYS):
                mask = (flat_targets == i)
                total_counts[i] += mask.sum()
                correct_counts[i] += (flat_preds[mask] == i).sum()

            key_idx = flat_targets * NUM_KEYS + flat_preds
            key_bincount = torch.bincount(key_idx, minlength=NUM_KEYS * NUM_KEYS)
            key_conf_matrix += key_bincount.view(NUM_KEYS, NUM_KEYS)

    metrics = torch.tensor([total_key_loss, total, key_correct], device=device).float()
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    dist.all_reduce(key_conf_matrix, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_counts, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_counts, op=dist.ReduceOp.SUM)

    num_batches = len(val_loader) * dist.get_world_size()
    avg_key_loss = metrics[0].item() / num_batches
    avg_key_acc = metrics[2].item() / metrics[1].item()

    if dist.get_rank() == 0:
        wandb.log({"val/key_loss": avg_key_loss, "val/key_acc": avg_key_acc})
        
        print("\nPer-key Performance:")
        for i in range(NUM_KEYS):
            correct = int(correct_counts[i].item())
            total_val = int(total_counts[i].item())
            acc = (correct / total_val) if total_val > 0 else 0.0
            print(f"Key {i}: {acc:.2%} ({correct}/{total_val})")
        plt.figure(figsize=(12, 10))
        
        conf_matrix_numpy = key_conf_matrix.cpu().numpy()
        
        row_sums = conf_matrix_numpy.sum(axis=1, keepdims=True)
        conf_matrix_norm = np.divide(conf_matrix_numpy, row_sums,  out=np.zeros_like(conf_matrix_numpy), where=row_sums != 0)

        sns.heatmap(conf_matrix_norm, annot=False, cmap='Blues', fmt='.2f')
        plt.title(f'Confusion Matrix (Epoch {epoch})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f"confusion_matrix_{epoch}.png")
        
        wandb.log({"val/confusion_matrix": wandb.Image(f"confusion_matrix_{epoch}.png")})
        plt.close()

    per_key_acc = torch.nan_to_num(key_conf_matrix.diag() / key_conf_matrix.sum(1))

    return avg_key_loss, avg_key_acc, per_key_acc


if __name__ == "__main__":
    local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    with open('default.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, train_sampler, val_sampler = get_dataloaders(config["data_dir"], config["training"]["batch_size"], config["model"]["seq_len"], frame_mode=config["model"]["frame_mode"], is_distributed=True, only_ops=config["model"]["only_ops"])

    model = KeystrokeIDM(num_keys=NUM_KEYS, d_model=config["model"]["d_model"], num_transformer_layers=3, num_heads=8, ff_dim=4096, frame_mode=config["model"]["frame_mode"]).to(device)
    
    model = DDP(model, device_ids=[local_rank])

    if dist.get_rank() == 0:
        setup_wandb()
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Rank 0: {num_params:,} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = OneCycleLR(optimizer, max_lr=config["training"]["lr"], total_steps=total_steps, pct_start=0.1, anneal_strategy='cos')
    
    if config["model"]["only_ops"]:
        counts = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    else: 
        counts = torch.tensor([5000, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    weights = 1.0 / torch.sqrt(counts + 1) 
    weights = weights / weights.sum() * len(counts)
    weights = weights.to(device)

    key_criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    for epoch in range(config["training"]["epochs"]):
        train_sampler.set_epoch(epoch) 
        
        train_key_loss, train_key_acc = train(model, train_loader, key_criterion, optimizer, scheduler, device, local_rank)
        val_key_loss, val_key_acc, per_key_acc = validate(model, val_loader, key_criterion, device, epoch)

        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1}/{config['training']['epochs']} | Train: key={train_key_loss:.4f} key_acc={train_key_acc:.2%} | Val: key={val_key_loss:.4f} key_acc={val_key_acc:.2%}")

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
            wandb.log({
                "epoch": epoch+1,
                "train_key_loss": train_key_loss,
                "train_key_acc": train_key_acc,
                "val_key_loss": val_key_loss,
                "val_key_acc": val_key_acc,
            })

    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()