#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --job-name=mouse_skip_noops

export CUDA_VISIBLE_DEVICES=0,1,2,3

source /p/project1/envcomp/idm/miniforge3/etc/profile.d/conda.sh
conda activate idm

export NCCL_SOCKET_IFNAME=eth0,en,eth,em,bond,enp
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export HF_HUB_CACHE=/p/scratch/envcomp/idm/huggingface
export HF_HOME=/p/scratch/envcomp/idm/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1


cd idm

# Mode: skip no-op frames entirely, keep frame indices for non-noop frames only
# e.g. "Frame 8: MOUSE:0,1,0\nFrame 10: MOUSE:1,0,0" (frame 9 was noop, omitted)
srun /p/project1/envcomp/idm/miniforge3/envs/idm/bin/torchrun \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_mouse.py \
    --data_root /p/scratch/envcomp/idm/sim_mouse_ds \
    --out_dir /p/scratch/envcomp/idm/mouse_log_dir_skip_noops \
    --focal_loss_gamma 2.0 \
    --label_smoothing 0.05 \
    --format_loss_weight 0.0 \
    --val_temperature 0.8 \
    --ema_decay 0.999 \
    --mouse_soft_label_sigma 1.5 \
    --mouse_soft_label_weight 0.3 \
    --train_min_action_density 0.1 \
    --train_min_action_density_ramp_steps 500 \
    --lora_r 32 \
    --lora_alpha 64 \
    --skip_noop_frames
