#!/usr/bin/env bash
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=00:10:00
#SBATCH --output=/fast/home/mihir.mahajan/Projects/idm/idm/optimize-throughput/slurm_logs/%x_%j.log
#SBATCH --error=/fast/home/mihir.mahajan/Projects/idm/idm/optimize-throughput/slurm_logs/%x_%j.log
#SBATCH --job-name=idm_bench

set -euo pipefail

REPO_DIR=/fast/home/mihir.mahajan/Projects/idm/idm/optimize-throughput
VENV_DIR=/home/mihir.mahajan/Projects/idm/inverse-dynamics-model/.venv

: "${DATA_ROOT:=/fast/project/HFMI_SynergyUnit/p-doom/crowd-cast/crowd-cast-2026-02-16/array_records_filter_noop}"
: "${OUT_DIR_BASE:=/fast/home/mihir.mahajan/Projects/idm/idm/optimize-throughput/runs_bench}"
: "${BENCH_TAG:=default}"
: "${MAX_STEPS:=40}"
: "${WARMUP_STEPS:=5}"
: "${WSD_DECAY_STEPS:=20}"
: "${GLOBAL_BS:=8}"
: "${SEQ_LEN:=128}"
: "${NUM_WORKERS:=0}"
: "${PREFETCH:=4}"
: "${READ_THREADS:=4}"
: "${WORKER_BUFFER:=1}"
: "${COLLATOR_PREFETCH:=1}"
: "${GRAD_CKPT:=1}"
: "${SKIP_RECORD_VALIDATION:=1}"
: "${NO_LORA:=1}"

source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1
# Avoid torchrun port collisions when multiple single-node jobs run on one host.
MASTER_PORT="${MASTER_PORT:-$((20000 + (SLURM_JOB_ID % 40000)))}"
export MASTER_PORT

mkdir -p "${OUT_DIR_BASE}"
RUN_DIR="${OUT_DIR_BASE}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${BENCH_TAG}"

cd "${REPO_DIR}"

LORA_FLAG="--use-lora"
if [[ "${NO_LORA}" == "1" ]]; then
  LORA_FLAG="--no-use-lora"
fi

PREFETCH_FLAG="--collator-prefetch"
if [[ "${COLLATOR_PREFETCH}" == "0" ]]; then
  PREFETCH_FLAG="--no-collator-prefetch"
fi

GRAD_CKPT_FLAG="--grad-checkpointing"
if [[ "${GRAD_CKPT}" == "0" ]]; then
  GRAD_CKPT_FLAG="--no-grad-checkpointing"
fi

RECORD_VALIDATION_FLAG="--skip-record-validation"
if [[ "${SKIP_RECORD_VALIDATION}" == "0" ]]; then
  RECORD_VALIDATION_FLAG="--no-skip-record-validation"
fi

if (( WARMUP_STEPS + WSD_DECAY_STEPS > MAX_STEPS )); then
  echo "Invalid schedule: WARMUP_STEPS(${WARMUP_STEPS}) + WSD_DECAY_STEPS(${WSD_DECAY_STEPS}) > MAX_STEPS(${MAX_STEPS})" >&2
  exit 2
fi

echo "JOB_ID=${SLURM_JOB_ID} TAG=${BENCH_TAG}"
echo "DATA_ROOT=${DATA_ROOT} MAX_STEPS=${MAX_STEPS} WARMUP=${WARMUP_STEPS} WSD_DECAY=${WSD_DECAY_STEPS} BS=${GLOBAL_BS} SEQ=${SEQ_LEN}"
echo "NUM_WORKERS=${NUM_WORKERS} PREFETCH=${PREFETCH} READ_THREADS=${READ_THREADS} WORKER_BUFFER=${WORKER_BUFFER}"
echo "COLLATOR_PREFETCH=${COLLATOR_PREFETCH} GRAD_CKPT=${GRAD_CKPT} SKIP_RECORD_VALIDATION=${SKIP_RECORD_VALIDATION} MASTER_PORT=${MASTER_PORT}"

torchrun --nproc_per_node=1 --master_port "${MASTER_PORT}" train.py \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --data-root "${DATA_ROOT}" \
  --max-steps "${MAX_STEPS}" \
  --global-batch-size "${GLOBAL_BS}" \
  --seq-len "${SEQ_LEN}" \
  --image-h 270 \
  --image-w 480 \
  --image-c 3 \
  --lr-schedule wsd \
  --warmup-steps "${WARMUP_STEPS}" \
  --wsd-decay-steps "${WSD_DECAY_STEPS}" \
  --lr 1e-5 \
  --init-lr 0 \
  --decay-end 0 \
  --log-every 10 \
  --save-every 1000000 \
  --val-every 0 \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-buffer-size "${PREFETCH}" \
  --read-num-threads "${READ_THREADS}" \
  --worker-buffer-size "${WORKER_BUFFER}" \
  ${PREFETCH_FLAG} \
  ${RECORD_VALIDATION_FLAG} \
  --precision bf16 \
  ${GRAD_CKPT_FLAG} \
  ${LORA_FLAG} \
  --mask-no-op-actions \
  --mask-mouse-actions \
  --no-wandb-enable \
  --out-dir "${RUN_DIR}"
