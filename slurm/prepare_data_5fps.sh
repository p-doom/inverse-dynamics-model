#!/usr/bin/env bash
#SBATCH --job-name=idm-prepare-data-5fps
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom/idm/preprocess/logs/%x_%j.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/p-doom/idm/preprocess/logs/%x_%j.log

module load CUDA/12.8
export PYTHONUNBUFFERED=1

cd /fast/home/mihir.mahajan/Projects/idm/inverse-dynamics-model
source .venv/bin/activate

mkdir -p /fast/project/HFMI_SynergyUnit/p-doom/idm/train/logs

# 5fps: 25 frames per clip (same 5-second duration), stride=12 (~50% overlap)
python prepare_data.py \
    --input-dir /fast/project/HFMI_SynergyUnit/p-doom/crowd-cast/crowd-cast-2026-02-25/uploads_valid_black_frame_filtered_crf_20 \
    --output-dir /fast/project/HFMI_SynergyUnit/p-doom/idm/train/data/data_sparse_960x540_5fps_w25 \
    --fps 5 \
    --resolution 960x540 \
    --clip-length 25 \
    --clip-stride 12 \
    --label-frames \
    --action-types keyboard,click,scroll \
    --num-workers 32
