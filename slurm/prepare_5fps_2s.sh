#!/usr/bin/env bash
#SBATCH --job-name=prep-5fps-2s
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --qos=low
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/logs/%x_%j.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/logs/%x_%j.log

module load FFmpeg/7.0.2-GCCcore-13.3.0
export PYTHONUNBUFFERED=1

cd /fast/home/mihir.mahajan/Projects/idm/inverse-dynamics-model
source .venv/bin/activate

: "${RESOLUTION:?Set RESOLUTION e.g. 854x480, 960x540, 1728x1080}"
: "${TAG:?Set TAG e.g. 480p, 540p, fullres}"

INPUT=/fast/project/HFMI_SynergyUnit/p-doom/crowd-cast/crowd-cast-2026-04-09/uploads
OUTPUT=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/data/data_5fps_2s_${TAG}

echo "=== Preparing 5fps 2s clips at ${TAG} (${RESOLUTION}) ==="
echo "Input: $INPUT"
echo "Output: $OUTPUT"

python prepare_data.py \
    --input-dir "$INPUT" \
    --output-dir "$OUTPUT" \
    --fps 5 \
    --resolution "$RESOLUTION" \
    --top-bar-fraction 0.15 \
    --black-threshold 0.95 \
    --clip-length 10 \
    --clip-stride 5 \
    --action-types keyboard,click,scroll \
    --label-frames \
    --train-ratio 0.85 \
    --val-ratio 0.15 \
    --seed 42 \
    --num-workers 16

echo "=== Done at $(date) ==="

echo "=== Coalescing ==="
python coalesce_training_data.py \
    --input-dir "$OUTPUT" \
    --output-dir "${OUTPUT}_coalesced"

echo "=== All done ==="
