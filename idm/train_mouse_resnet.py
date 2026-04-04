"""
Minimal ResNet-based Inverse Dynamics Model for mouse movement.

VPT-style: encode (frame_t, frame_{t+1}) with a shared ResNet18,
concatenate features, and classify action class + (dx, dy).

Usage (single GPU):
    cd idm && python train_mouse_resnet.py

Usage (multi-GPU):
    cd idm && torchrun --nproc_per_node=4 train_mouse_resnet.py
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import asdict, dataclass

import grain
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import ResNet18_Weights, resnet18
import tyro

sys.path.insert(0, os.path.dirname(__file__))
from mouse_actions import MOUSE_DELTA_CLIP_I, _parse_mouse_delta
from utils.data_jpeg import (
    EpisodeLengthFilter,
    ProcessEpisodeAndSlice,
    find_array_record_paths,
)

# ── Constants ─────────────────────────────────────────────────────────────────

DELTA_BINS   = 2 * MOUSE_DELTA_CLIP_I + 1  # 129 bins for dx/dy ∈ [-64, 64]
DELTA_OFFSET = MOUSE_DELTA_CLIP_I           # add to convert value → index

_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ── Args ──────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    data_root: str = "/p/scratch/envcomp/idm/sim_mouse_ds"
    image_h: int = 540
    image_w: int = 960
    seq_len: int = 16       # frames per episode slice; yields T-1 frame pairs
    batch_size: int = 8     # episode sequences per batch (per rank)
    resize: int = 224       # resize H/W for ResNet input
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_steps: int = 10_000
    warmup_steps: int = 200
    log_every: int = 10
    save_every: int = 1_000
    out_dir: str = "./runs/resnet_idm"
    num_workers: int = 4
    seed: int = 0
    no_op_weight: float = 0.3   # CE class weight for no-op (downweight majority)
    delta_sigma: float = 2.0    # std of Gaussian soft labels for dx/dy bins
    encoder_lr_scale: float = 0.1  # encoder LR = lr * encoder_lr_scale


# ── Model ─────────────────────────────────────────────────────────────────────

class MouseIDM(nn.Module):
    """
    ResNet18 IDM: (frame_t, frame_{t+1}) → mouse action.

    Both frames are encoded by a single shared ResNet18 (up to avgpool).
    Features are concatenated and fed into an MLP, which branches into:
      - cls: {no_op, mouse}
      - dx:  quantized x-delta ∈ [-64, 64]  (129 bins)
      - dy:  quantized y-delta ∈ [-64, 64]  (129 bins)
    """

    def __init__(self, hidden: int = 512):
        super().__init__()
        bb = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(bb.children())[:-1])  # (B, 512, 1, 1)
        feat = 512 * 3  # ft, ft1, and diff frame (ft1 - ft)

        self.mlp = nn.Sequential(
            nn.Linear(feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.cls_head = nn.Linear(hidden, 2)
        self.dx_head  = nn.Linear(hidden, DELTA_BINS)
        self.dy_head  = nn.Linear(hidden, DELTA_BINS)

    def forward(self, ft: torch.Tensor, ft1: torch.Tensor) -> dict[str, torch.Tensor]:
        """ft, ft1: (B, 3, H, W) normalised → logit dicts."""
        diff = ft1 - ft
        z = torch.cat([
            self.encoder(ft).flatten(1),
            self.encoder(ft1).flatten(1),
            self.encoder(diff).flatten(1),
        ], dim=1)
        h = self.mlp(z)
        return {"cls": self.cls_head(h), "dx": self.dx_head(h), "dy": self.dy_head(h)}


# ── Data ──────────────────────────────────────────────────────────────────────

def _get_dataloader(
    paths: list[str], args: Args, epoch: int, rank: int, world: int,
) -> grain.DataLoader:
    """Custom pipeline that keeps raw 'actions' (skips BuildSFTExampleFromFrames)."""
    source  = grain.sources.ArrayRecordDataSource(paths)
    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.ShardOptions(
            shard_index=rank, shard_count=world, drop_remainder=True,
        ),
        shuffle=True, num_epochs=1, seed=args.seed + epoch,
    )
    ops = [
        EpisodeLengthFilter(seq_len=args.seq_len, image_h=args.image_h, image_w=args.image_w),
        ProcessEpisodeAndSlice(seq_len=args.seq_len, image_h=args.image_h, image_w=args.image_w),
        # Deliberately skip BuildSFTExampleFromFrames to keep raw 'actions' in the batch
        grain.transforms.Batch(batch_size=args.batch_size, drop_remainder=True),
    ]
    return grain.DataLoader(
        data_source=source, sampler=sampler, operations=ops,
        worker_count=args.num_workers, worker_buffer_size=4,
        read_options=grain.ReadOptions(prefetch_buffer_size=8, num_threads=4),
    )


def _parse_action(action_s: str) -> tuple[int, int, int]:
    """Return (cls, dx_idx, dy_idx). cls ∈ {0=no_op, 1=mouse}."""
    parsed = _parse_mouse_delta(action_s)
    if parsed is None:
        return 0, DELTA_OFFSET, DELTA_OFFSET
    dx_q, dy_q, _ = parsed
    dx_q = int(np.clip(dx_q, -MOUSE_DELTA_CLIP_I, MOUSE_DELTA_CLIP_I))
    dy_q = int(np.clip(dy_q, -MOUSE_DELTA_CLIP_I, MOUSE_DELTA_CLIP_I))
    return 1, dx_q + DELTA_OFFSET, dy_q + DELTA_OFFSET


def _batch_to_tensors(
    batch: dict, device: torch.device, resize: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a grain batch into paired frame tensors + action labels.

    batch["frames"]:  (B, T, H, W, C) uint8
    batch["actions"]: (B, T) strings

    Returns ft, ft1: (N, 3, H, W);  cls, dx, dy: (N,) int64
    where N = B * (T - 1).
    """
    frames:  np.ndarray = batch["frames"]   # (B, T, H, W, C)
    actions               = batch["actions"] # (B, T) — str or object array
    B, T = frames.shape[:2]
    T_act = len(actions[0]) if hasattr(actions, '__len__') else actions.shape[1]  # may differ from T
    T_pairs = min(T - 1, T_act)

    # Stack all consecutive pairs as (N, H, W, C) numpy
    f0_np = frames[:, :T_pairs].reshape(-1, *frames.shape[2:])   # (N, H, W, C)
    f1_np = frames[:, 1:T_pairs + 1].reshape(-1, *frames.shape[2:])

    mean = _IMG_MEAN.to(device)
    std  = _IMG_STD.to(device)

    def to_tensor(arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).float().to(device) / 255.0
        t = F.interpolate(t, (resize, resize), mode="bilinear", align_corners=False)
        return (t - mean) / std

    ft  = to_tensor(f0_np)
    ft1 = to_tensor(f1_np)

    cls_list, dx_list, dy_list = [], [], []
    for b in range(B):
        for t in range(T_pairs):
            c, dx, dy = _parse_action(str(actions[b][t]))
            cls_list.append(c); dx_list.append(dx); dy_list.append(dy)

    cls = torch.tensor(cls_list, dtype=torch.long, device=device)
    dx  = torch.tensor(dx_list,  dtype=torch.long, device=device)
    dy  = torch.tensor(dy_list,  dtype=torch.long, device=device)
    return ft, ft1, cls, dx, dy


# ── Loss ──────────────────────────────────────────────────────────────────────

def _gaussian_soft_labels(targets: torch.Tensor, num_bins: int, sigma: float) -> torch.Tensor:
    """targets: (N,) int64  →  (N, num_bins) soft labels (sum to 1 per row)."""
    bins = torch.arange(num_bins, device=targets.device).float()
    dist = (bins.unsqueeze(0) - targets.unsqueeze(1).float()) ** 2
    labels = torch.exp(-dist / (2 * sigma ** 2))
    return labels / labels.sum(dim=1, keepdim=True)


def _soft_cross_entropy(logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_labels * log_probs).sum(dim=1).mean()


def _compute_loss(
    logits: dict[str, torch.Tensor],
    cls: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    no_op_weight: float,
    delta_sigma: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    cls_w  = torch.tensor([no_op_weight, 1.0], device=cls.device)
    l_cls  = F.cross_entropy(logits["cls"], cls, weight=cls_w)

    mouse  = cls == 1
    if mouse.any():
        dx_labels = _gaussian_soft_labels(dx[mouse], DELTA_BINS, delta_sigma)
        dy_labels = _gaussian_soft_labels(dy[mouse], DELTA_BINS, delta_sigma)
        l_dx = _soft_cross_entropy(logits["dx"][mouse], dx_labels)
        l_dy = _soft_cross_entropy(logits["dy"][mouse], dy_labels)
    else:
        # No mouse actions in batch — keep dx/dy heads in the graph for DDP.
        l_dx = (logits["dx"] * 0).sum()
        l_dy = (logits["dy"] * 0).sum()

    total = l_cls + l_dx + l_dy

    acc_cls = (logits["cls"].argmax(1) == cls).float().mean().item()
    if mouse.any():
        acc_dx = (logits["dx"][mouse].argmax(1) == dx[mouse]).float().mean().item()
        acc_dy = (logits["dy"][mouse].argmax(1) == dy[mouse]).float().mean().item()
    else:
        acc_dx = acc_dy = float("nan")

    return total, {
        "loss": total.item(),
        "l_cls": l_cls.item(),
        "l_dx": l_dx.item(),
        "l_dy": l_dy.item(),
        "acc_cls": acc_cls,
        "acc_dx": acc_dx,
        "acc_dy": acc_dy,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = tyro.cli(Args)
    os.makedirs(args.out_dir, exist_ok=True)

    rank  = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))

    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)

    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)

    model = MouseIDM().to(device)
    if world > 1:
        model = DDP(model, device_ids=[local], output_device=local)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"MouseIDM  params={n_params:,}  world={world}")

    m = model.module if world > 1 else model
    encoder_lr = args.lr * args.encoder_lr_scale
    optimizer = torch.optim.AdamW([
        {"params": m.encoder.parameters(), "lr": encoder_lr},
        {"params": [*m.mlp.parameters(), *m.cls_head.parameters(),
                    *m.dx_head.parameters(), *m.dy_head.parameters()]},
    ], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[encoder_lr, args.lr], total_steps=args.max_steps,
        pct_start=args.warmup_steps / args.max_steps,
    )

    paths   = find_array_record_paths(args.data_root, "train")
    step    = 0
    epoch   = 0
    log_acc:   dict[str, float] = {}
    log_count: dict[str, int]   = {}
    t0 = time.time()

    while step < args.max_steps:
        for batch in _get_dataloader(paths, args, epoch, rank, world):
            model.train()
            ft, ft1, cls, dx, dy = _batch_to_tensors(batch, device, args.resize)
            if ft.shape[0] == 0:
                continue

            logits      = model(ft, ft1)
            loss, stats = _compute_loss(logits, cls, dx, dy, args.no_op_weight, args.delta_sigma)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            for k, v in stats.items():
                if not math.isnan(v):
                    log_acc[k]    = log_acc.get(k, 0.0) + v
                    log_count[k]  = log_count.get(k, 0) + 1

            if rank == 0 and step % args.log_every == 0:
                parts = []
                for k, v in log_acc.items():
                    n = log_count.get(k, 1)
                    parts.append(f"{k}={v/n:.4f}")
                lr_enc  = optimizer.param_groups[0]["lr"]
                lr_head = optimizer.param_groups[1]["lr"]
                print(f"step={step} {' '.join(parts)} lr_enc={lr_enc:.2e} lr_head={lr_head:.2e} dt={time.time()-t0:.1f}s")
                log_acc.clear()
                log_count.clear()
                t0 = time.time()

            if rank == 0 and (step % args.save_every == 0 or step == args.max_steps):
                ckpt = os.path.join(args.out_dir, f"ckpt_{step:07d}.pt")
                m = model.module if world > 1 else model
                torch.save({"model": m.state_dict(), "step": step, "args": asdict(args)}, ckpt)
                print(f"Saved {ckpt}")

            if step >= args.max_steps:
                break

        epoch += 1

    if rank == 0:
        print(f"Done. step={step}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
