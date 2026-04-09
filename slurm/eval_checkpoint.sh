#!/usr/bin/env bash
#SBATCH --job-name=idm-eval-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --qos=low
#SBATCH --output=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/logs/%x_%j.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/logs/%x_%j.log

module load CUDA/12.8
module load FFmpeg/7.0.2-GCCcore-13.3.0
export PYTHONUNBUFFERED=1

# --- Configuration (set these before submitting) ---
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to the step_N directory}"
CKPT_NAME="${CKPT_NAME:?Set CKPT_NAME e.g. lora_step1000}"
IS_LORA="${IS_LORA:-true}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
# ---

export HF_HOME=/fast/project/HFMI_SynergyUnit/tab_model/huggingface
MERGED_DIR=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/merged_models/${CKPT_NAME}
EVAL_DIR=/fast/home/mihir.mahajan/Projects/idm/eval-models
TRAIN_DIR=/fast/home/mihir.mahajan/Projects/idm/inverse-dynamics-model
RESULTS_DIR=/fast/project/HFMI_SynergyUnit/p-doom/idm/eval/results
PORT=30010

# Step 1: Merge LoRA (if applicable) or copy full-finetune checkpoint
if [ "$IS_LORA" = "true" ]; then
    echo "=== Step 1: Merging LoRA checkpoint ==="
    cd $TRAIN_DIR
    source .venv/bin/activate
    python merge_and_save.py \
        --model-id "$MODEL_ID" \
        --checkpoint "${CKPT_DIR}/checkpoint.pt" \
        --output-dir "$MERGED_DIR"
    SERVE_MODEL="$MERGED_DIR"
else
    echo "=== Step 1: Loading full fine-tune checkpoint ==="
    # For full fine-tune, we need to load and save in HF format
    cd $TRAIN_DIR
    source .venv/bin/activate
    python -c "
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained('$MODEL_ID', torch_dtype=torch.bfloat16, trust_remote_code=True)
ckpt = torch.load('${CKPT_DIR}/checkpoint.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.save_pretrained('$MERGED_DIR')
AutoProcessor.from_pretrained('$MODEL_ID', trust_remote_code=True).save_pretrained('$MERGED_DIR')
print(f'Saved full model at step {ckpt[\"global_step\"]}')
"
    SERVE_MODEL="$MERGED_DIR"
fi

# Step 2: Start sglang server (using training venv which has sglang)
echo "=== Step 2: Starting sglang server on port $PORT ==="
cd $TRAIN_DIR
source .venv/bin/activate
python -m sglang.launch_server \
    --model-path "$SERVE_MODEL" \
    --port $PORT \
    --trust-remote-code \
    --dtype bfloat16 &
SERVER_PID=$!

echo "Waiting for server (PID=$SERVER_PID)..."
for i in $(seq 1 120); do
    if curl -s --max-time 5 "http://localhost:${PORT}/v1/models" | grep -q "model"; then
        echo "Server up after ${i} attempts."
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: Server did not come up. Exiting."
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 10
done

# Step 3: Run eval (switch to eval venv)
echo "=== Step 3: Running eval ==="
cd $EVAL_DIR
source .venv/bin/activate
python run_eval.py \
    --clips-dir /fast/project/HFMI_SynergyUnit/p-doom/idm/val-set \
    --provider sglang \
    --model "$SERVE_MODEL" \
    --base-url "http://localhost:${PORT}/v1" \
    --output "${RESULTS_DIR}/results_${CKPT_NAME}.json" \
    --fps 10 \
    --max-resolution 864

# Step 4: Score (10fps → tolerance=4 for equivalent ±400ms matching; no annotations since they were labeled at 5fps)
echo "=== Step 4: Scoring ==="
python score_eval.py \
    --results "${RESULTS_DIR}/results_${CKPT_NAME}.json" \
    --output "${RESULTS_DIR}/detailed_${CKPT_NAME}.json" \
    --coalesce \
    --tolerance 4

echo "=== Done ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
