from __future__ import annotations

from pathlib import Path
import pickle
import re
from typing import Any


STEP_RE = re.compile(r"step_(\d+)$")


def _root(out_dir: str) -> Path:
    return Path(out_dir) / "checkpoints"


def ckpt_dir_for_step(out_dir: str, step_i: int) -> Path:
    return _root(out_dir) / f"step_{step_i:08d}"


def find_latest_checkpoint(out_dir: str) -> str | None:
    root = _root(out_dir)
    if not root.exists():
        return None

    best_step = -1
    best_dir = None
    for path in root.iterdir():
        if not path.is_dir():
            continue
        m = STEP_RE.search(path.name)
        if m is None:
            continue
        step_i = int(m.group(1))
        if step_i > best_step:
            best_step = step_i
            best_dir = path
    return str(best_dir) if best_dir is not None else None


def _save_model(ckpt_dir: Path, model: Any, use_lora: bool) -> None:
    if use_lora:
        if not hasattr(model, "save_pretrained"):
            raise ValueError("LoRA save requires model.save_pretrained.")
        model.save_pretrained(str(ckpt_dir / "adapter"))
        return

    with open(ckpt_dir / "model_state.pkl", "wb") as f:
        pickle.dump(model.state_dict(), f)


def _load_model(ckpt_dir: Path, model: Any, use_lora: bool) -> None:
    if use_lora:
        adapter_dir = ckpt_dir / "adapter"
        if adapter_dir.exists() and hasattr(model, "load_adapter"):
            model.load_adapter(str(adapter_dir), adapter_name="default")
        return

    with open(ckpt_dir / "model_state.pkl", "rb") as f:
        state_d = pickle.load(f)
    model.load_state_dict(state_d)


def save_checkpoint(
    out_dir: str,
    step_i: int,
    model: Any,
    use_lora: bool,
    optimizer: Any,
    scheduler: Any,
    scaler_state: Any,
    train_state_d: dict[str, Any],
    grain_state_b: bytes | None,
    args_d: dict[str, Any],
    rank_i: int | None = None,
    save_model: bool = True,
) -> str:
    ckpt_dir = ckpt_dir_for_step(out_dir, step_i)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if save_model:
        _save_model(ckpt_dir, model, use_lora)

    payload_d = {
        "optimizer_state_d": optimizer.state_dict(),
        "scheduler_state_d": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_d": scaler_state,
        "train_state_d": train_state_d,
        "grain_state_b": grain_state_b,
        "args_d": args_d,
    }
    state_name = "trainer_state.pkl" if rank_i is None else f"trainer_state_rank{rank_i}.pkl"
    with open(ckpt_dir / state_name, "wb") as f:
        pickle.dump(payload_d, f)
    return str(ckpt_dir)


def load_checkpoint(
    ckpt_dir: str,
    model: Any,
    use_lora: bool,
    optimizer: Any,
    scheduler: Any,
    rank_i: int | None = None,
) -> dict[str, Any]:
    ckpt_dir_p = Path(ckpt_dir)
    if not ckpt_dir_p.exists():
        raise ValueError(f"Checkpoint dir not found: {ckpt_dir}")

    _load_model(ckpt_dir_p, model, use_lora)

    state_path = (
        ckpt_dir_p / "trainer_state.pkl"
        if rank_i is None
        else ckpt_dir_p / f"trainer_state_rank{rank_i}.pkl"
    )
    if rank_i is not None and not state_path.exists():
        state_path = ckpt_dir_p / "trainer_state.pkl"

    with open(state_path, "rb") as f:
        payload_d = pickle.load(f)

    optimizer.load_state_dict(payload_d["optimizer_state_d"])
    if scheduler is not None and payload_d.get("scheduler_state_d") is not None:
        scheduler.load_state_dict(payload_d["scheduler_state_d"])
    return payload_d
