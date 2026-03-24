#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --partition=accelerated
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:4
#SBATCH --output=/home/hk-project-p0023960/tum_ddk3888/inverse-dynamics-model/logs/%x_%j.out
#SBATCH --error=/home/hk-project-p0023960/tum_ddk3888/inverse-dynamics-model/logs/%x_%j.err
#SBATCH --job-name=preprocess_crowd_code

module unload mpi/openmpi/5.0
module unload devel/cuda/12.4

export CUDA_VISIBLE_DEVICES=0,1,2,3
conda init

srun uv run python data/video_to_array_records.py \
        --input-path /hkfs/work/workspace/scratch/tum_cte0515-crowd-cast/crowd-cast/crowd-cast-2026-03-02/uploads \
        --output-path /hkfs/work/workspace/scratch/tum_cte0515-crowd-cast/crowd-cast/crowd-cast-2026-03-02/array_records_960x540_exp \
        --num-workers 8 \
        --chunks-per-file 100 \
        --seed 0 \
        --target-width 960 \
        --target-height 540 \
        --target-fps 30 \
        --chunk-size 160 \
        --decode-timeout-sec 600 \
        --mouse-delta-clip 64 \
        --mouse-scroll-clip 5 \
        --no-op-as-mouse-zero \
        --actions-stateful
