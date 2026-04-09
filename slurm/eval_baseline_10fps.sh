#!/usr/bin/env bash
#SBATCH --job-name=idm-eval-baseline-10fps
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

export HF_HOME=/fast/project/HFMI_SynergyUnit/tab_model/huggingface
EVAL_DIR=/fast/home/mihir.mahajan/Projects/idm/eval-models
TRAIN_DIR=/fast/home/mihir.mahajan/Projects/idm/inverse-dynamics-model
RESULTS_DIR=/fast/project/HFMI_SynergyUnit/p-doom/idm/eval/results
MODEL_ID=Qwen/Qwen3-VL-2B-Instruct
PORT=30010

# Start sglang server for off-the-shelf model
echo "=== Starting sglang server ==="
cd $TRAIN_DIR
source .venv/bin/activate
python -m sglang.launch_server \
    --model-path "$MODEL_ID" \
    --port $PORT \
    --trust-remote-code \
    --dtype bfloat16 &
SERVER_PID=$!

echo "Waiting for server..."
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

# Eval at 10fps
echo "=== Running eval at 10fps ==="
cd $EVAL_DIR
source .venv/bin/activate
python run_eval.py \
    --clips-dir /fast/project/HFMI_SynergyUnit/p-doom/idm/val-set \
    --provider sglang \
    --model "$MODEL_ID" \
    --base-url "http://localhost:${PORT}/v1" \
    --output "${RESULTS_DIR}/results_qwen3vl2b_10fps.json" \
    --fps 10 \
    --max-resolution 864

# Score with tolerance=4 (equivalent to tolerance=2 at 5fps)
echo "=== Scoring ==="
python score_eval.py \
    --results "${RESULTS_DIR}/results_qwen3vl2b_10fps.json" \
    --output "${RESULTS_DIR}/detailed_qwen3vl2b_10fps.json" \
    --coalesce \
    --tolerance 4

echo "=== Done ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
