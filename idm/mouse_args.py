from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Args:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    attn_implementation: str = "sdpa"
    data_root: str = ""
    image_h: int = 540
    image_w: int = 960
    image_c: int = 3
    video_fps: float = 10.0
    seq_len: int = 64
    train_min_action_density: float = 0.0
    global_batch_size: int = 4
    grad_accum: int = 8
    max_grad_norm: float = 1.0
    max_steps: int = 5000
    lr: float = 2e-5
    init_lr: float = 0.0
    decay_end: float = 0.0
    lr_schedule: str = "wsd"
    warmup_steps: int = 100
    wsd_decay_steps: int = 200
    weight_decay: float = 0.0
    precision: str = "bf16"
    grad_checkpointing: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 0
    num_workers: int = 4
    prefetch_buffer_size: int = 8
    read_num_threads: int = 4
    worker_buffer_size: int = 4
    collator_prefetch: bool = True
    log_every: int = 5
    val_every: int = 50
    val_steps: int = 32
    val_generate_max_new_tokens: int = 512
    val_log_examples: int = 2
    val_visual_every: int = 200
    """Log visual cursor overlay frames to WandB every N optimizer steps (0=off)."""
    val_visual_max_frames: int = 8
    """Max frames per visual sample to render."""
    val_visual_upscale: int = 4
    """Upscale factor for rendered cursor frames."""
    save_every: int = 100
    out_dir: str = "./runs/mouse_sim"
    resume_from: str = ""
    instruction_text: str = (
        "Given the video frames, output the mouse action for each frame in order."
    )
    no_op_loss_weight: float = 0.3
    """Loss weight for NO_OP action tokens. Lower than 1.0 to down-weight easy majority class."""
    mouse_loss_weight: float = 5.0
    """Loss weight for MOUSE action tokens. Higher than 1.0 to up-weight rare hard class."""
    format_loss_weight: float = 0.0
    """Loss weight for structural format tokens (Frame X:, newlines, etc.).
    0.0 = only train on action tokens (recommended). 1.0 = original behaviour."""
    class_balanced_loss: bool = True
    """Normalise loss per weight-bucket (one bucket per action class) then average buckets.
    This gives equal gradient contribution from NO_OP and MOUSE regardless of frequency."""
    wandb_enable: bool = True
    wandb_project: str = "idm-mouse"
    wandb_entity: str = "instant-uv"
    wandb_run_name: str = "idm_mouse_run"
    wandb_mode: str = "offline"
    mfu_peak_flops: float = 0.0

    # ── NEW: improvements for better accuracy ────────────────────────
    label_smoothing: float = 0.0
    """Label smoothing factor for cross-entropy loss (0.0 = off, try 0.05-0.1)."""

    focal_loss_gamma: float = 0.0
    """Focal loss gamma. 0 = standard CE. Try 1.0-2.0 to focus on hard tokens."""

    train_min_action_density_ramp_steps: int = 0
    """Linearly ramp train_min_action_density from 0 to its final value over
    this many steps.  Helps the model first learn NO_OP vs MOUSE distinction
    on easy batches before seeing harder high-density batches."""

    val_temperature: float = 1.0
    """Temperature for validation generation. <1 = sharper. Try 0.7-0.9."""

    ema_decay: float = 0.0
    """If >0, keep an EMA copy of the model and use it for validation.
    Try 0.999 or 0.9995."""

    cosine_aux_loss_weight: float = 0.0
    """If >0, add an auxiliary cosine-similarity loss on mouse-delta tokens
    to encourage directional correctness even when exact token doesn't match.
    Experimental -- try 0.1-0.5."""

    mouse_soft_label_sigma: float = 0.0
    """Sigma (in quantized delta units) for Gaussian soft-label loss on mouse
    numeric delta tokens.  0.0 = disabled.  When >0, predicting a value close
    to the true delta is penalised less than predicting a far-away value.
    Try 1.0-3.0 (units are the same as MOUSE_DELTA_CLIP_I=64 range)."""

    mouse_soft_label_weight: float = 0.0
    """Additive weight for the Gaussian soft-label auxiliary loss.
    Only active when mouse_soft_label_sigma > 0.  Try 0.1-1.0."""

    mouse_prox_px_threshold: float = 50.0
    """Pixel radius used to compute mouse proximity accuracy at validation time.
    A predicted MOUSE action is counted as 'close' if the (dx, dy) euclidean
    distance to the ground-truth in pixel space is within this threshold."""

    diversity_penalty: float = 0.0
    """If >0 during validation generation, apply a repetition/diversity penalty.
    Try 1.2-2.0 to break mode collapse at inference time."""

    train_action_density_curriculum: bool = False
    """If True, start training with low action density and increase over time."""

    noop_format: str = "token"
    """How to represent no-op frames in target text.
    'token' = 'NO_OP' (default).
    'zeros' = '0,0,0' (bare triplet; no special token, easier for the model to learn
    a continuous action space)."""

    skip_noop_frames: bool = False
    """If True, omit no-op frames from the target text entirely, keeping frame indices
    for non-noop frames only (e.g. 'Frame 8: MOUSE:0,1,0\\nFrame 10: MOUSE:1,0,0'
    where frame 9 was a no-op)."""
