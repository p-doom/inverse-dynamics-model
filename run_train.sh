#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --partition=accelerated
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:4
#SBATCH --output=/home/hk-project-p0023960/tum_ddk3888/inverse-dynamics-model/logs/mouse_%x_%j.out
#SBATCH --error=/home/hk-project-p0023960/tum_ddk3888/inverse-dynamics-model/logs/mouse_%x_%j.err
#SBATCH --job-name=preprocess_crowd_code

module unload mpi/openmpi/5.0
module unload devel/cuda/12.4

export CUDA_VISIBLE_DEVICES=0,1,2,3
conda init

cd data

uv run python mouse_generation.py \
    --frame_path /home/hk-project-p0023960/tum_ddk3888/inverse-dynamics-model/data/frame.png  \
    --output_path /hkfs/work/workspace/scratch/tum_cte0515-crowd-cast/simulated_mouse_ds  \
    --num_frames 1500000

cd ..
cd idm

srun uv run torchrun --nproc_per_node=4 train_mouse.py \
    --data_root /hkfs/work/workspace/scratch/tum_cte0515-crowd-cast/simulated_mouse_ds \
    --out_dir ./runs/mouse_sim