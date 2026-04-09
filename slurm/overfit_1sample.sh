#!/usr/bin/env bash
#SBATCH --job-name=idm-overfit-1sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --qos=low
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/logs/%x_%j.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/logs/%x_%j.log

module load CUDA/12.8
export PYTHONUNBUFFERED=1

cd /fast/home/mihir.mahajan/Projects/idm/inverse-dynamics-model
source .venv/bin/activate

export HF_HOME=/fast/project/HFMI_SynergyUnit/tab_model/huggingface

DATA_DIR=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/data/data_sparse_960x540_5_pct_empty_w32/overfit_1

# Overfit on 1 sample: should drive loss to ~0 within 100 steps
python train.py \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --attn-implementation flash_attention_2 \
    --video-mode image \
    --data-dir "$DATA_DIR" \
    --max-length 16384 \
    --max-steps 200 \
    --batch-size 1 \
    --grad-accum 1 \
    --lr 1e-4 \
    --warmup-steps 0 \
    --wsd-decay-steps 0 \
    --save-every 9999 \
    --val-every 0 \
    --log-every 5 \
    --out-dir /fast/project/HFMI_SynergyUnit/p-doom/idm/train/runs/overfit_1sample_${SLURM_JOB_ID} \
    --wandb-project idm \
    --wandb-run-name "overfit-1sample-${SLURM_JOB_ID}" \
    --wandb-mode online \
    --seed 42
