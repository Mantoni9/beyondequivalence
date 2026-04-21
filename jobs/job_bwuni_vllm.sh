#!/bin/bash
#SBATCH --job-name=olala_bwuni_vllm
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/olala_bwuni_vllm_%j.out
#SBATCH --error=logs/olala_bwuni_vllm_%j.err

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.bwuni
set +a

# Ensure vLLM is installed (CUDA-only, not in environment.yml)
python -c "import vllm" 2>/dev/null || {
    echo "[setup] vllm not found — installing..."
    pip install vllm --quiet
}

# ── vLLM server ────────────────────────────────────────────────────────────────
PORT=$((8000 + (SLURM_JOB_ID % 1000)))
export VLLM_BASE_URL="http://localhost:${PORT}/v1"

echo "[vLLM] Starting server on port ${PORT}  model=${MODEL_PATH}"
echo "[vLLM] No quantization (2× A100 80GB)  tp=${VLLM_TENSOR_PARALLEL}  max_len=${VLLM_MAX_MODEL_LEN}"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL}" \
    --dtype bfloat16 \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --port "${PORT}" \
    --host 127.0.0.1 \
    --no-enable-log-requests \
    &
VLLM_PID=$!

trap 'echo "[vLLM] Shutting down server (PID ${VLLM_PID})"; kill ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null || true' EXIT

# ── Wait for server ready ──────────────────────────────────────────────────────
MAX_WAIT=900
WAITED=0
echo "[vLLM] Waiting for server to be ready (max ${MAX_WAIT}s)..."
until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
        echo "[vLLM] ERROR: server did not become ready within ${MAX_WAIT}s" >&2
        exit 1
    fi
    sleep 10
    WAITED=$((WAITED + 10))
    echo "[vLLM] Still waiting... ${WAITED}s elapsed"
done
echo "[vLLM] Server ready after ${WAITED}s  →  ${VLLM_BASE_URL}"

# ── Experiment ─────────────────────────────────────────────────────────────────
python run_experiment.py \
    --model "${MODEL_PATH}" \
    --wandb \
    --threshold 0.6
