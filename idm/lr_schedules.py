from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LRScheduleArgs:
    schedule: str
    init_lr: float
    max_lr: float
    decay_end: float
    max_steps: int
    warmup_steps: int
    wsd_decay_steps: int


def _linear(start_v: float, end_v: float, step_i: int, total_steps: int) -> float:
    if total_steps <= 0:
        return end_v
    frac = min(max(step_i / total_steps, 0.0), 1.0)
    return start_v + frac * (end_v - start_v)


def lr_at_step(cfg: LRScheduleArgs, step_i: int) -> float:
    step_i = min(max(step_i, 0), cfg.max_steps)

    if cfg.schedule not in {"cos", "wsd", "const"}:
        raise ValueError("Unsupported lr schedule. Use one of: cos, wsd, const")
    if cfg.warmup_steps > cfg.max_steps:
        raise ValueError("warmup_steps must be <= max_steps")
    if cfg.schedule == "wsd" and cfg.warmup_steps + cfg.wsd_decay_steps > cfg.max_steps:
        raise ValueError("warmup_steps + wsd_decay_steps must be <= max_steps")

    if step_i <= cfg.warmup_steps:
        return _linear(cfg.init_lr, cfg.max_lr, step_i, cfg.warmup_steps)

    if cfg.schedule == "const":
        return cfg.max_lr

    if cfg.schedule == "cos":
        decay_steps_i = cfg.max_steps - cfg.warmup_steps
        if decay_steps_i <= 0:
            return cfg.decay_end
        frac = (step_i - cfg.warmup_steps) / decay_steps_i
        cos_frac = 0.5 * (1.0 + math.cos(math.pi * frac))
        return cfg.decay_end + (cfg.max_lr - cfg.decay_end) * cos_frac

    decay_start_i = cfg.max_steps - cfg.wsd_decay_steps
    if step_i <= decay_start_i:
        return cfg.max_lr
    return _linear(
        cfg.max_lr,
        cfg.decay_end,
        step_i - decay_start_i,
        cfg.wsd_decay_steps,
    )
