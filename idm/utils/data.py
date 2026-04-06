from __future__ import annotations

import os
from typing import Any

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _truncate(example: dict[str, Any], max_turns: int) -> dict[str, Any]:
    """Truncate to *max_turns* messages.

    Expected format: ``user=image, assistant=action``.
    A trailing user message (image with no paired action) is dropped.
    """
    messages = example["messages"][:max_turns]
    if messages and messages[-1]["role"] == "user":
        messages = messages[:-1]
    return {"messages": messages}


def load_chat_dataset(data_path: str, max_turns: int = 32) -> Dataset:
    """Load a JSONL chat dataset using HuggingFace *datasets*.

    Each line must be a JSON object with a ``"messages"`` key holding a
    conversation (list of ``{role, content}`` dicts).
    """
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(lambda ex: _truncate(ex, max_turns))
    dataset = dataset.filter(lambda ex: len(ex["messages"]) >= 2)
    return dataset


def get_chat_dataloader(
    dataset: Dataset,
    collate_fn: Any,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    """Build a :class:`DataLoader` with optional ``DistributedSampler``."""
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=True,
    )


def find_jsonl_path(data_root: str, split: str) -> str:
    """Return the path to ``{data_root}/{split}.jsonl`` if it exists."""
    path = os.path.join(data_root, f"{split}.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Expected JSONL file not found: {path}")
    return path
