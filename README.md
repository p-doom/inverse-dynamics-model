# Inverse Dynamics Model

Train an inverse dynamics model that predicts per-frame actions from screen-recording sequences.

## Install

```bash
uv sync
# optional test deps
uv sync --extra dev
uv run pre-commit install
```

## Data

For screen-recording data, we use [crowd-cast](https://pdoom.org/crowd_cast.html). The data is expected in the following form:

```text
crowd_cast_root/
  .../<user_id>/recordings/recording_<session-id>_seg####.mp4
  .../<user_id>/keylogs/input_<session-id>_seg####.msgpack
```

Preprocess the crowd-cast data into [ArrayRecord format](https://github.com/google/array-record) for IDM training:

```bash
uv run python data/idm_data/video_to_array_records.py \
  --input-path /path/to/crowd_cast_root \
  --output-path /path/to/idm_data \
  --target-width 160 \
  --target-height 90 \
  --target-fps 10 \
  --chunk-size 160 \
  --chunks-per-file 100 \
  --num-workers 16
```

The generated data directory looks like:

```text
idm_data/
  metadata.json
  train/*.array_record
  val/*.array_record
  test/*.array_record
```

## Train

Single GPU (baseline):

```bash
torchrun --nproc_per_node=1 train.py \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --data-root /path/to/idm_data \
  --image-h 90 --image-w 160 --image-c 3 \
  --seq-len 32 \
  --global-batch-size 8 \
  --grad-accum 1 \
  --max-steps 3000 \
  --lr 2e-5 \
  --lr-schedule wsd \
  --warmup-steps 200 \
  --wsd-decay-steps 600 \
  --precision bf16 \
  --use-lora True \
  --wandb-enable True \
  --wandb-project idm \
  --wandb-run-name idm_qwen2b_baseline \
  --out-dir ./runs/idm_qwen2b
```

If you are not using wandb, set `--wandb-enable False`.

Multi-GPU (example: 8 GPUs):

```bash
torchrun --nproc_per_node=8 train.py \
  --data-root /path/to/idm_data \
  --global-batch-size 64 \
  --out-dir ./runs/idm_8gpu \
  --wandb-enable False
```

Resume:

```bash
torchrun --nproc_per_node=8 train.py --data-root /path/to/idm_data --resume-from latest
```

Checkpoints are written under `out_dir/checkpoints/`.

## Test

```bash
uv run pytest
```
