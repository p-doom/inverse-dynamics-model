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
#SBATCH --job-name=mouse_gen

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

srun /p/project1/envcomp/idm/miniforge3/envs/idm/bin/torchrun \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_mouse.py \
    --data_root /p/scratch/envcomp/idm/sim_mouse_ds \
    --out_dir ./runs/mouse_sim