#!/bin/bash
#SBATCH --job-name=e15
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/e15_%j.out
#SBATCH --error=logs/e15_%j.err
#
# E15 few-shot ablation on bwUniCluster 3.0. ONE model x ONE dataset x ONE arm,
# single-order, K=20, otherwise identical to the Stage-2 matrix protocol.
# Serving env: vllm-e15 (vLLM 0.24, serves gpt-oss-120b + Llama-70B-AWQ).
# Reranker CLIENT env: melt-olala. Exemplars from held-out g1-web (native gold).
# Walltime is set per-submit (small g5/g7 ~8h; g3 ~24h) — generous, no timeout.
#
# Submit: MODEL=gpt-oss DATASET=g7-literature FEWSHOT_ARM=A2 sbatch --time=08:00:00 jobs/job_bwuni_e15.sh

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
WS=/pfs/work9/workspace/scratch/ma_amarkic-beyondeq_e15
export HF_HOME="$WS/hf_cache"
export ZENODO_BENCHMARK_ZIP="$HOME/melt-project/src/benchmark.zip"
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN

MODEL="${MODEL:?set MODEL}"; DATASET="${DATASET:?set DATASET}"
ARM="${FEWSHOT_ARM:?set FEWSHOT_ARM}"; SEED="${SEED:-42}"
PORT=$((8400 + (SLURM_JOB_ID % 500)))
export VLLM_BASE_URL="http://localhost:${PORT}/v1"
mkdir -p results logs

STAGE1="results/stage1_frozen/${DATASET}_qwen3-noLoRA_pathctx_T2_top20.tsv"
[ -f "$STAGE1" ] || { echo "[e15] ERROR missing $STAGE1" >&2; exit 1; }

case "$MODEL" in
  gpt-oss)
    MODEL_PATH="openai/gpt-oss-120b"; SERVE_EXTRA="--max-model-len 8192"
    MAX_NEW_TOKENS=1024; TEMP=1.0; TOP_P=1.0 ;;
  llama)
    MODEL_PATH="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"
    SERVE_EXTRA="--quantization awq --dtype float16 --max-model-len 8192"
    MAX_NEW_TOKENS=256; TEMP=0.0; TOP_P="" ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 2 ;;
esac
TOP_P_FLAG=""; [ -n "$TOP_P" ] && TOP_P_FLAG="--top-p $TOP_P"

echo "=========================================================================="
echo "[e15] MODEL=$MODEL DATASET=$DATASET ARM=$ARM SEED=$SEED node=$(hostname) port=$PORT"
echo "[e15] stage1=$STAGE1  exemplars=g1-web"
echo "=========================================================================="

conda activate vllm-e15
# CUDA toolkit (nvcc) — bwUni has it only as a module, NOT on PATH by default.
# vLLM 0.24 JIT-compiles the FlashInfer sampler kernel (and deep_gemm) at engine
# startup, which needs nvcc + headers; without it EngineCore dies with
# "Could not find nvcc". cuda/12.8 matches the env's torch cu128. (Path is stable
# on bwUni; `module load` is flaky non-interactively so we set it explicitly.)
export CUDA_HOME=/opt/bwhpc/common/devel/cuda/12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
python -c "import vllm; print('[e15] vllm', vllm.__version__)" || { echo "[e15] FATAL vllm" >&2; exit 3; }
command -v nvcc >/dev/null && echo "[e15] nvcc: $(nvcc --version | grep -oE 'release [0-9.]+')" || echo "[e15] WARN: nvcc not found"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --tensor-parallel-size 2 --enforce-eager \
    --gpu-memory-utilization 0.92 --port "$PORT" --host 127.0.0.1 \
    --no-enable-log-requests $SERVE_EXTRA &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null || true' EXIT

MAX_WAIT=1800; WAITED=0
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then echo "[e15] FATAL: vLLM died during load" >&2; exit 4; fi
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then echo "[e15] FATAL: health timeout ${MAX_WAIT}s" >&2; exit 4; fi
    sleep 10; WAITED=$((WAITED+10)); [ $((WAITED % 60)) -eq 0 ] && echo "[e15] waiting vLLM ${WAITED}s"
done
echo "[e15] vLLM ready after ${WAITED}s"

OUT="results/e15_${MODEL}_${DATASET}_${ARM}_seed${SEED}_$(git rev-parse --short HEAD)"
conda run -n melt-olala bash -lc "VLLM_BASE_URL='${VLLM_BASE_URL}' HF_HOME='${HF_HOME}' ZENODO_BENCHMARK_ZIP='${ZENODO_BENCHMARK_ZIP}' python run_stage2_experiment.py \
    --dataset '${DATASET}' \
    --stage1-predictions '${STAGE1}' --stage1-top-k 20 \
    --stage1-description description_path_context --description description_path_context \
    --llm-model '${MODEL_PATH}' --prompt-id d_subs_v2 \
    --few-shot-arm ${ARM} --exemplar-track g1-web \
    --max-new-tokens ${MAX_NEW_TOKENS} --temperature ${TEMP} ${TOP_P_FLAG} \
    --llm-max-concurrency 16 --seed ${SEED} --output-dir '${OUT}'"
RC=$?
echo "[e15] done MODEL=$MODEL DATASET=$DATASET ARM=$ARM rc=$RC out=$OUT"
exit $RC
