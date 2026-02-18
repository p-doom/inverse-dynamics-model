# IDM SFT (Qwen3-VL + ArrayRecord + Grain + torchrun)

Minimal PyTorch DDP fine-tuning code for:
- `Qwen/Qwen3-VL-2B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`

Task:
- Video (multi-frame contiguous sequence) + text SFT.
- Input: `seq_len` contiguous frames.
- Target: action text for all `seq_len` frames in order.

Constraints:
- No TRL / bitsandbytes / accelerate / Trainer.
- Uses `torch`, `transformers`, `peft`, `grain`, `array_record`, `numpy`, `pickle`.

## Install

```bash
git clone git@github.com:p-doom/inverse-dynamics-model.git
cd inverse-dynamics-model
uv sync
# for tests:
uv sync --extra dev
```

## Dataset Format

Expected layout:

```text
data_root/
  metadata.json
  train/*.array_record
  val/*.array_record
  test/*.array_record
```

Each record is bytes that decode with `pickle.loads` into dict with:
- required: `raw_video` (bytes), `sequence_length` (int)
- required: `actions: list[str]` length `T`
- optional metadata: `relative_path`, `user_id`, `session_id`, `seg_idx`, `video_file_name`

Stable key derivation priority:
1. `relative_path`
2. `user_id|session_id|seg_idx`
3. `video_file_name|seg_idx`

## Launch

1 GPU:

```bash
torchrun --nproc_per_node=1 train.py \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --data-root /path/to/data_root \
  --image-h 90 --image-w 160 --image-c 3 \
  --seq-len 32 \
  --global-batch-size 8 \
  --grad-accum 1 \
  --max-steps 1000 \
  --lr 2e-5 \
  --lr-schedule wsd \
  --warmup-steps 100 \
  --wsd-decay-steps 300 \
  --precision bf16 \
  --use-lora True \
  --wandb-enable True \
  --wandb-project idm \
  --wandb-run-name qwen3vl2b_smoke \
  --out-dir ./runs/qwen3vl_2b
```

8 GPUs:

```bash
torchrun --nproc_per_node=8 train.py \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --data-root /path/to/data_root \
  --image-h 90 --image-w 160 --image-c 3 \
  --seq-len 32 \
  --global-batch-size 64 \
  --grad-accum 1 \
  --max-steps 5000 \
  --lr 1e-5 \
  --lr-schedule cos \
  --warmup-steps 300 \
  --precision bf16 \
  --use-lora True \
  --wandb-enable True \
  --wandb-project idm \
  --wandb-run-name qwen3vl4b_ddp \
  --out-dir ./runs/qwen3vl_4b
```

Resume:

```bash
torchrun --nproc_per_node=8 train.py ... --resume-from latest
```

W&B auth (if `wandb_mode=online`):

```bash
wandb login
```

## Key Behavior

- Frame sampling is contiguous only.
- Supervision is full sequence (all `seq_len` action texts in assistant output).
- DDP sharding uses `grain.sharding.ShardOptions(shard_index=rank, shard_count=world_size, drop_remainder=True)`.
- Training stops by `--max-steps`; script prints estimated epochs before training.
- LR schedules: `cos`, `wsd`, `const`.
- LoRA optional via `--use-lora`.
- W&B logging on rank 0 via `--wandb-enable`.

## Checkpoints

Saved to:

```text
out_dir/checkpoints/step_XXXXXXXX/
```

Includes:
- model payload (`adapter/` when LoRA, full model state when non-LoRA)
- rank-local trainer state (`trainer_state_rank{rank}.pkl`)
- optimizer/scheduler/scaler states
- RNG states
- Grain iterator state bytes
- args snapshot

## Sanity Checks

1. Overfit tiny subset (1 GPU):
- Use very small split / low `max_steps`.
- Expect rapid loss drop in early steps.

2. 1GPU vs DDP equivalence:
- Match effective global batch, seed, `max_steps`.
- Compare early loss trends; should be close.

## Troubleshooting

- OOM:
  - lower `--seq-len`
  - lower `--global-batch-size`
  - raise `--grad-accum`
  - use LoRA mode (`--use-lora True`)
- Missing actions:
  - regenerate ArrayRecords so each record contains `actions: list[str]` with `len(actions)==sequence_length`.
- Raw video shape mismatch:
  - set `--image-h/--image-w/--image-c` to match record encoding.
