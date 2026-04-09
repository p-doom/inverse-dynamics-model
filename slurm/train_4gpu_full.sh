#!/usr/bin/env bash
#SBATCH --job-name=idm-train-4gpu-full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --qos=low
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/logs/%x_%j.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/logs/%x_%j.log

module load CUDA/12.8
export PYTHONUNBUFFERED=1

cd /fast/home/mihir.mahajan/Projects/idm/inverse-dynamics-model
source .venv/bin/activate

export HF_HOME=/fast/project/HFMI_SynergyUnit/tab_model/huggingface
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29501

DATA_DIR=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/data/data_sparse_960x540_5_pct_empty_w32

# 4-GPU DDP, full fine-tune (no LoRA), image mode, flash-attn
# Effective batch = 4 GPUs × BS=1 × grad_accum=4 = 16
# Lower LR for full fine-tune
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --attn-implementation flash_attention_2 \
    --video-mode image \
    --data-dir "$DATA_DIR" \
    --max-length 16384 \
    --max-steps 10000 \
    --batch-size 1 \
    --grad-accum 4 \
    --lr 5e-6 \
    --warmup-steps 200 \
    --wsd-decay-steps 1000 \
    --max-grad-norm 1.0 \
    --no-use-lora \
    --save-every 1000 \
    --val-every 500 \
    --val-steps 20 \
    --log-every 10 \
    --out-dir /fast/project/HFMI_SynergyUnit/p-doom/idm/train/runs/train_4gpu_full_${SLURM_JOB_ID} \
    --wandb-project idm \
    --wandb-run-name "train-4gpu-full-${SLURM_JOB_ID}" \
    --wandb-mode online \
    --seed 42
