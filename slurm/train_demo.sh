#!/usr/bin/env bash
#SBATCH --job-name=idm-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --qos=low
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/logs/%x_%j.log

module load CUDA/12.8
module load FFmpeg/7.0.2-GCCcore-13.3.0

cd /fast/home/mihir.mahajan/Projects/idm/inverse-dynamics-model
source .venv/bin/activate
export HF_HOME=/fast/project/HFMI_SynergyUnit/tab_model/huggingface

torchrun --nnodes=1 --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$(hostname):29500 \
    train.py \
    --model-id Qwen/Qwen3-VL-8B-Instruct \
    --data-dir /fast/project/HFMI_SynergyUnit/p-doom/idm/train/data/data_2fps_coalesced \
    --interleave-labels \
    --use-lora --train-vision --lora-r 128 --lora-alpha 256 \
    --max-pixels 518400 \
    --max-length 8192 \
    --max-steps 3000 \
    --batch-size 2 --grad-accum 2 \
    --lr 2e-6 \
    --eval-clips-dir /fast/project/HFMI_SynergyUnit/p-doom/idm/val-set \
    --eval-coalesce \
    --eval-tolerance 2 \
    --run-id $SLURM_JOB_ID \
    --out-dir /fast/project/HFMI_SynergyUnit/p-doom/idm/train/runs/demo_$SLURM_JOB_ID \
    --wandb-project idm --wandb-run-name "demo-$SLURM_JOB_ID"
