#!/bin/bash
#SBATCH --job-name=olala_stage2_smoke
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/olala_stage2_smoke_%j.out
#SBATCH --error=logs/olala_stage2_smoke_%j.err
# Exclude nodes where vLLM hangs after KV-cache init with no /health response.
# Observed 2026-06-02:
#   - dws-17 (job 255354): 12 shm_broadcast warnings, no progress
#   - dws-16 (job 255357): 20 shm_broadcast warnings, no progress
# Both have device capability 8.6 + "SymmMemCommunicator not supported" warnings;
# --enforce-eager alone is not enough on these nodes. Job 255327 ran fine on
# dws-09 earlier the same day (~10 min vLLM warmup, then reranker calls).
#SBATCH --exclude=dws-16,dws-17

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
export NCCL_DEBUG=WARN

# vLLM 0.19+ ignores VLLM_USE_V1; V1 engine is the only path. The shm_broadcast
# deadlock is avoided by --enforce-eager (no CUDA graph capture).

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
echo "[vLLM] Quantization: ${VLLM_QUANTIZATION}  dtype=${VLLM_DTYPE:-float16}  tp=${VLLM_TENSOR_PARALLEL}  max_len=${VLLM_MAX_MODEL_LEN}"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL}" \
    --quantization "${VLLM_QUANTIZATION}" \
    --dtype "${VLLM_DTYPE:-float16}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --port "${PORT}" \
    --host 127.0.0.1 \
    --enforce-eager \
    --gpu-memory-utilization 0.92 \
    --no-enable-log-requests \
    &
VLLM_PID=$!

# Ensure the server is killed when the job exits (success or error).
trap 'echo "[vLLM] Shutting down server (PID ${VLLM_PID})"; kill ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null || true' EXIT

# ── Wait for server ready ──────────────────────────────────────────────────────
# Job 255354 (2026-06-02, dws-17) was killed by our own timeout at 600s while
# the vLLM EngineCore was still alive but in a silent post-KV-cache
# initialisation step (shm_broadcast warnings every 60s, no /health response).
# Observed: 10+ min of "Still waiting" with the engine not yet serving. Cold
# disk cache on a previously-unused compute node is the likely cause; the
# 70B AWQ shard load on dws-17 took 39 s (vs the historical ~18 s on dws-09)
# and the post-load init never reached /health within our window. Raising
# MAX_WAIT to 1800 (30 min) is the conservative fix; the cost on a fast node
# (where vLLM is ready in <10 min) is zero because we still poll /health.
MAX_WAIT=1800
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

# ── Stage-1 predictions TSV ────────────────────────────────────────────────────
# Decoupled candidate gen: the embedder ran in a previous (Stage-1-only) job
# and persisted predictions.tsv. The reranker just loads it — no embedder on
# this GPU, so vLLM can keep tp=2 + gpu_memory_utilization=0.92 without
# triggering the OOM seen in job 255320 (2026-06-02).
#
# Default points at the stable symlink in results/stage1_frozen/. The
# underlying file is the Qwen3-noLoRA / path_context / T2 / asymmetric /
# g7-literature ablation-bidirectional run (SHA d11c97e), config:
#   ablbi_qwen3-embedding-8b_lora-off_A-path_context_B-sub_b_pin_g7-literature_d11c97e
# Set STAGE1_PREDICTIONS at submit time to use a different TSV.
STAGE1_PREDICTIONS="${STAGE1_PREDICTIONS:-results/stage1_frozen/g7-literature_qwen3-noLoRA_pathctx_T2_top20.tsv}"
STAGE1_TOP_K="${STAGE1_TOP_K:-20}"
STAGE1_DESCRIPTION="${STAGE1_DESCRIPTION:-description_path_context}"

if [ ! -f "${STAGE1_PREDICTIONS}" ]; then
    echo "[stage2] ERROR: Stage-1 predictions TSV not found at ${STAGE1_PREDICTIONS}" >&2
    echo "[stage2] Create the symlink first, or pass STAGE1_PREDICTIONS=/path/to.tsv." >&2
    exit 1
fi
echo "[stage2] Stage-1 TSV: ${STAGE1_PREDICTIONS}"
echo "[stage2] top_k_per_direction=${STAGE1_TOP_K}  reranker_description=${STAGE1_DESCRIPTION}"

# ── Stage-2 smoke ──────────────────────────────────────────────────────────────
# Single dataset, single LLM. No sweeps. Success = metrics.json with 4x4 CM.
DATASET="${DATASET:-g7-literature}"
THRESHOLD="${THRESHOLD:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
# d_subs_v2 places the "Relation: <label>" anchor on the FIRST line of the
# response, so even if the model continues reasoning afterwards the parser
# already has the label. Token budget is therefore mainly for the optional
# justification, not for waiting until the answer arrives. The previous
# attempt with d_subs + 256 hit the token cap before the anchor in 99.9%
# of calls — see job 255391 post-mortem 2026-06-02.
PROMPT_ID="${PROMPT_ID:-d_subs_v2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
# Parallel calls to vLLM. 16 is safe for Llama-3.3-70B-AWQ on 2x A40 with
# --enforce-eager (vLLM continuous batching handles the concurrency).
# Sequential (=1) costs ~60 s/call -> 17 h on g7; 16 brings that to ~1 h.
LLM_MAX_CONCURRENCY="${LLM_MAX_CONCURRENCY:-16}"
# Stufe-A arm A2 (registered 2026-06-12): set SWAP_PAIR_PRESENTATION=1 to fill
# the prompt slots with (target, source); directional labels invert exactly
# once at parse time. Prompt text and verbalizations untouched.
SWAP_PAIR_PRESENTATION="${SWAP_PAIR_PRESENTATION:-0}"
SWAP_FLAG=""
if [ "${SWAP_PAIR_PRESENTATION}" = "1" ]; then
    SWAP_FLAG="--swap-pair-presentation"
fi
echo "[stage2] prompt_id=${PROMPT_ID}  swap_pair_presentation=${SWAP_PAIR_PRESENTATION}"

python run_stage2_experiment.py \
    --dataset "${DATASET}" \
    --stage1-predictions "${STAGE1_PREDICTIONS}" \
    --stage1-top-k "${STAGE1_TOP_K}" \
    --stage1-description "${STAGE1_DESCRIPTION}" \
    --description "${STAGE1_DESCRIPTION}" \
    --llm-model "${MODEL_PATH}" \
    --prompt-id "${PROMPT_ID}" \
    --threshold "${THRESHOLD}" \
    --batch-size "${BATCH_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --llm-max-concurrency "${LLM_MAX_CONCURRENCY}" \
    ${SWAP_FLAG}
