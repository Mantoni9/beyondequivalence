#!/bin/bash
#SBATCH --job-name=olala_dws_vllm
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/olala_dws_vllm_%j.out
#SBATCH --error=logs/olala_dws_vllm_%j.err

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.dws
set +a

# Ensure vLLM is installed (CUDA-only, not in environment.yml)
python -c "import vllm" 2>/dev/null || {
    echo "[setup] vllm not found — installing..."
    pip install vllm --quiet
}

# ── vLLM server ────────────────────────────────────────────────────────────────
# Use a job-specific port to avoid collisions on shared nodes.
PORT=$((8000 + (SLURM_JOB_ID % 1000)))
export VLLM_BASE_URL="http://localhost:${PORT}/v1"

echo "[vLLM] Starting server on port ${PORT}  model=${MODEL_PATH}"
echo "[vLLM] Quantization: ${VLLM_QUANTIZATION}  tp=${VLLM_TENSOR_PARALLEL}  max_len=${VLLM_MAX_MODEL_LEN}"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL}" \
    --quantization "${VLLM_QUANTIZATION}" \
    --dtype bfloat16 \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --port "${PORT}" \
    --host 127.0.0.1 \
    --disable-log-requests \
    &
VLLM_PID=$!

# Ensure the server is killed when the job exits (success or error).
trap 'echo "[vLLM] Shutting down server (PID ${VLLM_PID})"; kill ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null || true' EXIT

# ── Wait for server ready ──────────────────────────────────────────────────────
MAX_WAIT=900   # 15 minutes — 70B model load can take a while
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
