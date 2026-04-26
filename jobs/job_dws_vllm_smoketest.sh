#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
#  job_dws_vllm_smoketest.sh
#
#  Purpose: minimal sanity check that vLLM + LLMOpenAI work on DWS.
#  Setup:   8B model, single GPU (TP=1), no quantization.
#  Goal:    isolate the vLLM ↔ LLMOpenAI plumbing from NCCL/TP/quant issues
#           that complicate the 70B path.
#
#  Run interactively (recommended for debugging):
#      srun --partition=gpu-vram-48gb --gres=gpu:1 --mem=50G --time=02:00:00 \
#           --pty bash jobs/job_dws_vllm_smoketest.sh
#
#  Or as a batch job:
#      sbatch jobs/job_dws_vllm_smoketest.sh
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=olala_dws_smoke
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --output=logs/olala_dws_smoke_%j.out
#SBATCH --error=logs/olala_dws_smoke_%j.err

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

# DWS system libstdc++ is too old for vLLM (missing CXXABI_1.3.15).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Verbose NCCL — helpful even on TP=1 (vLLM still initialises NCCL).
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL

# Use vLLM V0 — V1 has a known shm_broadcast deadlock after CUDA-graph capture.
export VLLM_USE_V1=0

set -a
source .env.dws
set +a

# Smoke-test overrides: 8B model, TP=1, no quantization.
SMOKE_MODEL_PATH="${SMOKE_MODEL_PATH:-/work/amarkic/models/Llama-3.1-8B-Instruct}"
SMOKE_TP=1
SMOKE_MAX_LEN=4096

# Sanity output — copy this into the chat if anything goes wrong.
echo "──────────────────────────────────────────────────────────────────"
echo "[smoketest] hostname:     $(hostname)"
echo "[smoketest] CUDA_VISIBLE: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "[smoketest] model:        ${SMOKE_MODEL_PATH}"
echo "[smoketest] TP:           ${SMOKE_TP}"
echo "[smoketest] max_len:      ${SMOKE_MAX_LEN}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv
python -c "import torch; print('[smoketest] torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python -c "import vllm; print('[smoketest] vllm:', vllm.__version__)"
echo "──────────────────────────────────────────────────────────────────"

# Ensure vLLM is installed (CUDA-only, not in environment.yml).
python -c "import vllm" 2>/dev/null || {
    echo "[setup] vllm not found — installing..."
    python -m pip install vllm --quiet
}

# ── vLLM server ───────────────────────────────────────────────────────────────
PORT=$((8000 + (SLURM_JOB_ID % 1000)))
export VLLM_BASE_URL="http://localhost:${PORT}/v1"

echo "[vLLM] Starting server on port ${PORT}"
echo "[vLLM] model=${SMOKE_MODEL_PATH}  tp=${SMOKE_TP}  max_len=${SMOKE_MAX_LEN}  no quantization"

python -m vllm.entrypoints.openai.api_server \
    --model "${SMOKE_MODEL_PATH}" \
    --tensor-parallel-size "${SMOKE_TP}" \
    --dtype bfloat16 \
    --max-model-len "${SMOKE_MAX_LEN}" \
    --port "${PORT}" \
    --host 127.0.0.1 \
    --enforce-eager \
    --gpu-memory-utilization 0.90 \
    --no-enable-log-requests \
    &
VLLM_PID=$!

# Always shut the server down on exit.
trap 'echo "[vLLM] Shutting down server (PID ${VLLM_PID})"; kill ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null || true' EXIT

# ── Wait for server ready ─────────────────────────────────────────────────────
MAX_WAIT=600  # 10 min — 8B without quant should load in 1-3 min
WAITED=0
echo "[vLLM] Waiting for server health (max ${MAX_WAIT}s)…"
until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
        echo "[vLLM] ERROR: server did not become ready within ${MAX_WAIT}s" >&2
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "[vLLM] still waiting… ${WAITED}s elapsed"
done
echo "[vLLM] Server ready after ${WAITED}s  →  ${VLLM_BASE_URL}"

curl -s "http://localhost:${PORT}/v1/models" | python -m json.tool || true

# ── Smoke-test client ─────────────────────────────────────────────────────────
echo "[smoketest] Running LLMOpenAI roundtrip…"
SMOKE_MODEL_PATH="${SMOKE_MODEL_PATH}" python - <<'PY'
import os, sys
from LLMOpenAI import LLMOpenAI
from prompt import Prompt

llm = LLMOpenAI(
    model_name=os.environ["SMOKE_MODEL_PATH"],
    base_url=os.environ["VLLM_BASE_URL"],
    api_key="EMPTY",
)
p = Prompt().user("Are Heart and Herz the same concept? Answer yes or no.")
print("[smoketest] text:        ", llm.get_text_completion([p]))
print("[smoketest] confidence:  ", llm.get_confidence_first_token([p]))
PY

echo "[smoketest] DONE — if you see text + confidence above, vLLM ↔ LLMOpenAI works."
