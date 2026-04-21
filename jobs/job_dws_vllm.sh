#!/bin/bash
#SBATCH --job-name=olala_dws_vllm
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/olala_dws_vllm_%j.out
#SBATCH --error=logs/olala_dws_vllm_%j.err

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

# DWS system libstdc++ is too old for vLLM (missing CXXABI_1.3.15).
# Prepend the conda env lib so vLLM finds the newer libstdc++.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# DWS A6000 nodes: no NVLink P2P between GPUs → disable to avoid NCCL hang.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Use vLLM V0 engine to avoid shm_broadcast deadlock after CUDA graph capture.
export VLLM_USE_V1=0

set -a
source .env.dws
set +a

# Ensure vLLM is installed (CUDA-only, not in environment.yml)
python -c "import vllm" 2>/dev/null || {
    echo "[setup] vllm not found — installing..."
    python -m pip install vllm --quiet
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
    --no-enable-log-requests \
    &
VLLM_PID=$!

# Ensure the server is killed when the job exits (success or error).
trap 'echo "[vLLM] Shutting down server (PID ${VLLM_PID})"; kill ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null || true' EXIT

# ── Wait for server ready ──────────────────────────────────────────────────────
MAX_WAIT=1500  # 25 minutes — 70B load + torch.compile + CUDA graph capture
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
