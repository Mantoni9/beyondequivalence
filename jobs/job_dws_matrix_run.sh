#!/bin/bash
#SBATCH --job-name=mxrun
#SBATCH --partition=gpu-vram-48gb
#SBATCH --qos=max2gpu5d
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mxrun_%j.out
#SBATCH --error=logs/mxrun_%j.err
# NOTE: no node exclusion. The old dws-14..17 (A6000) exclusion rested on a
# weak premise — the "capability 8.6 / SymmMemCommunicator" warning also fires
# on the working A40 (dws-11), and all observed deadlocks were vLLM 0.19.1.
# The reasoner jobs use vLLM 0.23 (deadlock likely fixed). A40 is the only
# non-A6000 48GB node, so excluding A6000 would funnel every job onto dws-11.
# Safety net: bounded health-wait (MAX_WAIT) → exit 4 fast if a node hangs,
# then resubmit with a targeted --exclude.
#
# Stage-2 MATRIX run: ONE model x ONE dataset, single-order, FULL dataset, K=20.
# Server in env vllm-matrix (gpt-oss/gemma4/mistral, novel archs on vLLM 0.23) /
# melt-olala (Llama-AWQ, proven 0.19.1); reranker CLIENT always melt-olala.
# Per-model max_new_tokens + decoding from the Phase-0 probe. Walltime is set
# per-job at submit time (sbatch --time=...) — generous, variance-tolerant.
#
# Submit: MODEL=gpt-oss DATASET=g3-text SEED=42 sbatch --time=28:00:00 jobs/job_dws_matrix_run.sh

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
export HF_HOME=/work/amarkic/hf_cache
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN

MODEL="${MODEL:?set MODEL}"; DATASET="${DATASET:?set DATASET}"; SEED="${SEED:-42}"
PORT=$((8300 + (SLURM_JOB_ID % 600)))
export VLLM_BASE_URL="http://localhost:${PORT}/v1"
mkdir -p results logs

STAGE1="results/stage1_frozen/${DATASET}_qwen3-noLoRA_pathctx_T2_top20.tsv"
if [ ! -f "$STAGE1" ]; then echo "[mxrun] ERROR: missing $STAGE1" >&2; exit 1; fi

case "$MODEL" in
  gpt-oss)
    SERVE_ENV=vllm-matrix; MODEL_PATH="openai/gpt-oss-120b"
    SERVE_EXTRA="--max-model-len 8192"; MAX_NEW_TOKENS=1024; TEMP=1.0; TOP_P=1.0 ;;
  gemma4)
    SERVE_ENV=vllm-matrix; MODEL_PATH="google/gemma-4-31B-it"
    SERVE_EXTRA="--max-model-len 8192"; MAX_NEW_TOKENS=256; TEMP=1.0; TOP_P=0.95 ;;
  mistral)
    SERVE_ENV=vllm-matrix; MODEL_PATH="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    SERVE_EXTRA="--max-model-len 8192 --tokenizer-mode mistral --limit-mm-per-prompt {\"image\":0}"
    MAX_NEW_TOKENS=384; TEMP=0.0; TOP_P="" ;;
  llama)
    SERVE_ENV=melt-olala; MODEL_PATH="/work/amarkic/models/Llama-3.3-70B-Instruct-AWQ-INT4"
    SERVE_EXTRA="--quantization awq --dtype float16 --max-model-len 8192"
    MAX_NEW_TOKENS=256; TEMP=0.0; TOP_P="" ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 2 ;;
esac
TOP_P_FLAG=""; [ -n "$TOP_P" ] && TOP_P_FLAG="--top-p $TOP_P"

echo "=========================================================================="
echo "[mxrun] MODEL=$MODEL DATASET=$DATASET SEED=$SEED node=$(hostname) port=$PORT"
echo "[mxrun] serve_env=$SERVE_ENV mnt=$MAX_NEW_TOKENS temp=$TEMP top_p=${TOP_P:-none}"
echo "[mxrun] stage1=$STAGE1"
echo "=========================================================================="

conda activate "$SERVE_ENV"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python -c "import vllm; print('[mxrun] vllm', vllm.__version__)" || { echo "[mxrun] FATAL vllm" >&2; exit 3; }

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --tensor-parallel-size 2 --enforce-eager \
    --gpu-memory-utilization 0.92 --port "$PORT" --host 127.0.0.1 \
    --no-enable-log-requests $SERVE_EXTRA &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null || true' EXIT

MAX_WAIT=1800; WAITED=0
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then echo "[mxrun] FATAL: vLLM died during load" >&2; exit 4; fi
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then echo "[mxrun] FATAL: health timeout ${MAX_WAIT}s" >&2; exit 4; fi
    sleep 10; WAITED=$((WAITED+10)); [ $((WAITED % 60)) -eq 0 ] && echo "[mxrun] waiting vLLM ${WAITED}s"
done
echo "[mxrun] vLLM ready after ${WAITED}s"

OUT="results/matrix_${MODEL}_${DATASET}_seed${SEED}_$(git rev-parse --short HEAD)"
conda run -n melt-olala bash -lc "VLLM_BASE_URL='${VLLM_BASE_URL}' python run_stage2_experiment.py \
    --dataset '${DATASET}' \
    --stage1-predictions '${STAGE1}' --stage1-top-k 20 \
    --stage1-description description_path_context --description description_path_context \
    --llm-model '${MODEL_PATH}' --prompt-id d_subs_v2 \
    --max-new-tokens ${MAX_NEW_TOKENS} --temperature ${TEMP} ${TOP_P_FLAG} \
    --llm-max-concurrency 16 --seed ${SEED} --output-dir '${OUT}'"
RC=$?
echo "[mxrun] done MODEL=$MODEL DATASET=$DATASET SEED=$SEED rc=$RC out=$OUT"
exit $RC
