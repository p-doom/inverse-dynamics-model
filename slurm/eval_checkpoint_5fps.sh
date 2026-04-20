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
TRAIN_VISION="${TRAIN_VISION:-false}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
PROMPT_STRATEGY="${PROMPT_STRATEGY:-baseline}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
INTERLEAVE="${INTERLEAVE:-false}"
# ---

export HF_HOME=/fast/project/HFMI_SynergyUnit/tab_model/huggingface
MERGED_DIR=/fast/project/HFMI_SynergyUnit/p-doom/idm/train/merged_models/${CKPT_NAME}
EVAL_DIR=/fast/home/mihir.mahajan/Projects/idm/eval-models
TRAIN_DIR=/fast/home/mihir.mahajan/Projects/idm/inverse-dynamics-model
RESULTS_DIR=/fast/project/HFMI_SynergyUnit/p-doom/idm/eval/results
# Use a unique port per job to avoid conflicts when multiple evals run on the same node
PORT=$((30000 + (SLURM_JOB_ID % 1000)))

# Step 1: Merge LoRA (if applicable) or copy full-finetune checkpoint
if [ "$IS_LORA" = "true" ]; then
    echo "=== Step 1: Merging LoRA checkpoint ==="
    cd $TRAIN_DIR
    source .venv/bin/activate
    VISION_FLAG=""
    if [ "$TRAIN_VISION" = "true" ]; then
        VISION_FLAG="--train-vision"
    fi
    python merge_and_save.py \
        --model-id "$MODEL_ID" \
        --checkpoint "${CKPT_DIR}/checkpoint.pt" \
        --output-dir "$MERGED_DIR" \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        $VISION_FLAG
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
# Strip _orig_mod. prefix from torch.compile'd checkpoints
state_dict = ckpt['model_state_dict']
state_dict = {k.replace('._orig_mod.', '.').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
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
INTERLEAVE_FLAG=""
if [ "$INTERLEAVE" = "true" ]; then
    INTERLEAVE_FLAG="--interleave-labels"
fi
python run_eval_v2.py \
    --clips-dir /fast/project/HFMI_SynergyUnit/p-doom/idm/val-set \
    --provider sglang \
    --model "$SERVE_MODEL" \
    --base-url "http://localhost:${PORT}/v1" \
    --output "${RESULTS_DIR}/results_${CKPT_NAME}.json" \
    --fps 5 \
    --max-resolution 864 \
    --prompt-strategy "$PROMPT_STRATEGY" \
    $INTERLEAVE_FLAG

# Step 4: Score
COALESCE_FLAG=""
if [ "${COALESCE:-true}" = "true" ]; then
    COALESCE_FLAG="--coalesce"
fi
echo "=== Step 4: Scoring (coalesce=${COALESCE:-true}) ==="
python score_eval.py \
    --results "${RESULTS_DIR}/results_${CKPT_NAME}.json" \
    --output "${RESULTS_DIR}/detailed_${CKPT_NAME}.json" \
    $COALESCE_FLAG \
    --tolerance 2

# Step 5: Log eval metrics to the training wandb run
TRAIN_JOB_ID="${TRAIN_JOB_ID:-}"
if [ -n "$TRAIN_JOB_ID" ] && [ -f "${RESULTS_DIR}/detailed_${CKPT_NAME}.json" ]; then
    echo "=== Step 5: Logging to wandb run train-${TRAIN_JOB_ID} ==="
    cd $TRAIN_DIR
    source .venv/bin/activate
    # Add eval-models to path so score_eval is importable
    export PYTHONPATH="${EVAL_DIR}:${PYTHONPATH:-}"
    python -c "
import json, os, re, sys
sys.path.insert(0, '${EVAL_DIR}')
import wandb

ckpt_name = '${CKPT_NAME}'
# Extract step number from checkpoint name (e.g. 'lora_5fps_step200_97939' -> 200)
m = re.search(r'step(\d+)', ckpt_name)
step = int(m.group(1)) if m else 0

with open('${RESULTS_DIR}/detailed_${CKPT_NAME}.json') as f:
    lines = f.read()
# Parse the overall line from the score output
# The detailed JSON might not have a clean structure, so parse from the results
with open('${RESULTS_DIR}/results_${CKPT_NAME}.json') as f:
    data = json.load(f)

# Re-score to get metrics
from score_eval import filter_gt_actions, coalesce_gt_events, filter_predictions, match_clip
fps = data['fps']
all_tp, all_fp, all_fn = 0, 0, 0
type_stats = {}
for clip in data['clips']:
    gt = filter_gt_actions(clip['ground_truth'], clip['start_s'], fps)
    if '${COALESCE:-true}' == 'true':
        gt = coalesce_gt_events(gt)
    preds = filter_predictions(clip.get('predictions', []))
    result = match_clip(gt, preds, tolerance=2)
    all_tp += len(result['matches']); all_fp += len(result['unmatched_preds']); all_fn += len(result['unmatched_gt'])

p = all_tp / max(all_tp + all_fp, 1)
r = all_tp / max(all_tp + all_fn, 1)
f1 = 2*p*r / max(p+r, 1e-9)

run = wandb.init(id='train-${TRAIN_JOB_ID}', resume='must', project='idm')
run.log({'step': step, 'eval/f1': f1, 'eval/precision': p, 'eval/recall': r, 'eval/tp': all_tp, 'eval/fp': all_fp, 'eval/fn': all_fn})
run.finish()
print(f'Logged eval step={step} f1={f1:.3f} to wandb run train-${TRAIN_JOB_ID}')
"
fi

echo "=== Done ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
