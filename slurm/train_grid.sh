#!/usr/bin/env bash
#SBATCH --job-name=grid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --qos=low
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/logs/%x_%j.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/logs/%x_%j.log

module load CUDA/12.8
module load FFmpeg/7.0.2-GCCcore-13.3.0
export PYTHONUNBUFFERED=1
cd /fast/home/mihir.mahajan/Projects/idm/inverse-dynamics-model
source .venv/bin/activate
export HF_HOME=/fast/project/HFMI_SynergyUnit/tab_model/huggingface
export MASTER_ADDR=$(hostname)
export MASTER_PORT=${MASTER_PORT:-29500}

: "${DATA_DIR:?}"
: "${MAX_PIXELS:?}"
: "${MAX_LENGTH:?}"
: "${TAG:?}"
INTERLEAVE_FLAG="${INTERLEAVE_FLAG:-}"
EVAL_COALESCE_FLAG="${EVAL_COALESCE_FLAG:---eval-coalesce}"  # default: coalesce on. Set to "" for non-coalesced runs.
LR="${LR:-2e-6}"
MAX_STEPS="${MAX_STEPS:-2000}"
RESUME_FROM="${RESUME_FROM:-}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
# Allow overriding out-dir for resume (keeps checkpoints in original dir)
ORIG_JOB_ID="${ORIG_JOB_ID:-${SLURM_JOB_ID}}"
OUT_DIR="/fast/project/HFMI_SynergyUnit/p-doom/idm/train/runs/${TAG}_${ORIG_JOB_ID}"

RESUME_FLAG=""
if [ -n "$RESUME_FROM" ]; then
    RESUME_FLAG="--resume-from ${RESUME_FROM}"
    echo "=== Resuming from: ${RESUME_FROM} ==="
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"

echo "=== Grid search run: ${TAG} ==="
echo "DATA_DIR=${DATA_DIR}"
echo "MODEL_ID=${MODEL_ID}"
echo "MAX_PIXELS=${MAX_PIXELS}"
echo "INTERLEAVE=${INTERLEAVE_FLAG}"
echo "LORA_R=${LORA_R}, LORA_ALPHA=${LORA_ALPHA}"
echo "OUT_DIR=${OUT_DIR}"

torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --model-id ${MODEL_ID} \
    --attn-implementation flash_attention_2 \
    --video-mode image \
    ${INTERLEAVE_FLAG} \
    --data-dir ${DATA_DIR} \
    --max-length ${MAX_LENGTH} \
    --max-pixels ${MAX_PIXELS} \
    --max-steps ${MAX_STEPS} \
    --batch-size ${BATCH_SIZE:-1} --grad-accum ${GRAD_ACCUM:-4} \
    --lr ${LR} --warmup-steps 100 --wsd-decay-steps 300 \
    --weight-decay 0.01 --max-grad-norm 1.0 \
    ${LORA_FLAGS:---use-lora --train-vision --lora-r ${LORA_R:-16} --lora-alpha ${LORA_ALPHA:-32} --lora-dropout 0.05} \
    --save-every ${SAVE_EVERY:-400} --val-every ${VAL_EVERY:-400} --val-steps 20 --log-every 10 \
    --eval-clips-dir /fast/project/HFMI_SynergyUnit/p-doom/idm/val-set \
    ${EVAL_COALESCE_FLAG} \
    --eval-tolerance ${EVAL_TOLERANCE:-2} \
    --out-dir ${OUT_DIR} \
    ${RESUME_FLAG} \
    --run-id "${ORIG_JOB_ID}" \
    --wandb-project idm --wandb-run-name "${TAG}-${ORIG_JOB_ID}" --wandb-mode online --seed 42
