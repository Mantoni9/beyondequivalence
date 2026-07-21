#!/bin/bash
#SBATCH --job-name=e17verify
#SBATCH --partition=gpu-vram-48gb
#SBATCH --qos=max4gpu5d
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=logs/e17verify_%j.out
#SBATCH --error=logs/e17verify_%j.err
# E17 verification second-pass. Serves MODEL, then verifies its own assertions
# (V1 self-verify) over the datasets in DATASETS (default: dev pool g1+g2 for the
# viability checkpoint). Assertion source = the matrix cell predictions.tsv.
set -uo pipefail
cd /work/amarkic/beyondequivalence
source "$(conda info --base)/etc/profile.d/conda.sh"
export HF_HOME=/work/amarkic/hf_cache
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN
MODEL="${MODEL:?set MODEL}"
DATASETS="${DATASETS:-g1-web g2-diseases}"
TAG="${TAG:-_dev}"
PORT=$((8300 + (SLURM_JOB_ID % 600)))
export VLLM_BASE_URL="http://localhost:${PORT}/v1"
mkdir -p results/e17 logs

case "$MODEL" in
  gpt-oss) SERVE_ENV=vllm-matrix; MODEL_PATH="openai/gpt-oss-120b"; SERVE_EXTRA="--max-model-len 8192" ;;
  gemma4)  SERVE_ENV=vllm-matrix; MODEL_PATH="google/gemma-4-31B-it"; SERVE_EXTRA="--max-model-len 8192" ;;
  mistral) SERVE_ENV=vllm-matrix; MODEL_PATH="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
           SERVE_EXTRA="--max-model-len 8192 --tokenizer-mode mistral --limit-mm-per-prompt {\"image\":0}" ;;
  llama)   SERVE_ENV=melt-olala; MODEL_PATH="/work/amarkic/models/Llama-3.3-70B-Instruct-AWQ-INT4"
           SERVE_EXTRA="--quantization awq --dtype float16 --max-model-len 8192" ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 2 ;;
esac
case "$(hostname -s)" in
  dws-10|dws-14|dws-15|dws-16|dws-17) SERVE_EXTRA="$SERVE_EXTRA --disable-custom-all-reduce"
    echo "[e17] A6000 node -> --disable-custom-all-reduce" ;;
esac

echo "[e17] MODEL=$MODEL node=$(hostname) port=$PORT datasets='$DATASETS'"
conda activate "$SERVE_ENV"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --tensor-parallel-size 2 --enforce-eager \
    --gpu-memory-utilization 0.92 --port "$PORT" --host 127.0.0.1 \
    --no-enable-log-requests $SERVE_EXTRA &
VLLM_PID=$!
trap 'kill -9 ${VLLM_PID} 2>/dev/null || true' EXIT
MAX_WAIT=1800; WAITED=0
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then echo "[e17] FATAL vLLM died" >&2; scancel ${SLURM_JOB_ID}; exit 4; fi
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then echo "[e17] FATAL health timeout" >&2; kill -9 ${VLLM_PID} 2>/dev/null; scancel ${SLURM_JOB_ID}; exit 4; fi
    sleep 10; WAITED=$((WAITED+10)); [ $((WAITED % 60)) -eq 0 ] && echo "[e17] waiting vLLM ${WAITED}s"
done
echo "[e17] vLLM ready after ${WAITED}s"

for DS in $DATASETS; do
  CELL=$(ls -d results/matrix_${MODEL}_${DS}_seed42_* 2>/dev/null | grep -vE "_shard|_g2shard" | head -1)
  if [ -z "$CELL" ] || [ ! -f "$CELL/predictions.tsv" ]; then echo "[e17] SKIP $DS: no cell" >&2; continue; fi
  echo "[e17] verify $MODEL/$DS  <- $CELL"
  conda run -n melt-olala bash -lc "VLLM_BASE_URL='${VLLM_BASE_URL}' python scripts/e17_verify.py \
      --model '${MODEL}' --model-path '${MODEL_PATH}' --dataset '${DS}' \
      --assertions '${CELL}/predictions.tsv' --tag '${TAG}' --out results/e17 \
      --batch-size 256 --max-concurrency 64"
done
echo "[e17] DONE MODEL=$MODEL"
