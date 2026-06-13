#!/bin/bash
#SBATCH --job-name=mxprobe
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --exclude=dws-14,dws-15,dws-16,dws-17
#SBATCH --output=logs/mxprobe_%j.out
#SBATCH --error=logs/mxprobe_%j.err
#
# Stage-2 matrix Phase-0 PROBE: per-model serving + throughput + parse_fail +
# CoT-length check on a g7 smoke (single-order). The serving check is the
# show-stopper gate (gpt-oss MXFP4/harmony, gemma4, mistral3 are novel archs
# that vLLM 0.19.1 cannot serve). Server runs in env `vllm-matrix` (newer vLLM)
# for the novel archs / `melt-olala` (proven 0.19.1) for Llama-AWQ; the reranker
# CLIENT always runs in melt-olala against the HTTP endpoint.
#
# Submit: MODEL=gpt-oss sbatch jobs/job_dws_matrix_probe.sh   (gpt-oss FIRST)
#   MODEL in {gpt-oss, gemma4, mistral, llama}

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
export HF_HOME=/work/amarkic/hf_cache
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN

MODEL="${MODEL:-gpt-oss}"
PORT=$((8200 + (SLURM_JOB_ID % 700)))
export VLLM_BASE_URL="http://localhost:${PORT}/v1"
mkdir -p results logs

# ── per-model config ───────────────────────────────────────────────────────────
case "$MODEL" in
  gpt-oss)
    SERVE_ENV=vllm-matrix; MODEL_PATH="openai/gpt-oss-120b"; REASONER=1
    SERVE_EXTRA="--max-model-len 8192" ;;        # mxfp4 auto-detected from config
  gemma4)
    SERVE_ENV=vllm-matrix; MODEL_PATH="google/gemma-4-31B-it"; REASONER=1
    SERVE_EXTRA="--max-model-len 8192" ;;
  mistral)
    # Mistral-Small-3.2 is mistral3/pixtral (multimodal). vLLM 0.23.0 crashes
    # profiling a dummy image (MistralCommonImageProcessor.fetch_images). We
    # use it TEXT-ONLY → cap image inputs to 0 to skip the vision path.
    SERVE_ENV=vllm-matrix; MODEL_PATH="mistralai/Mistral-Small-3.2-24B-Instruct-2506"; REASONER=0
    SERVE_EXTRA="--max-model-len 8192 --tokenizer-mode mistral --limit-mm-per-prompt {\"image\":0}" ;;
  llama)
    SERVE_ENV=melt-olala; MODEL_PATH="/work/amarkic/models/Llama-3.3-70B-Instruct-AWQ-INT4"; REASONER=0
    SERVE_EXTRA="--quantization awq --dtype float16 --max-model-len 8192" ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 2 ;;
esac
# Reasoners need a realistic CoT budget (the 255391 lesson: 256 truncates CoT).
if [ "$REASONER" = "1" ]; then MAX_NEW_TOKENS=2048; else MAX_NEW_TOKENS=256; fi

echo "=========================================================================="
echo "[probe] MODEL=$MODEL  serve_env=$SERVE_ENV  reasoner=$REASONER  port=$PORT"
echo "[probe] model_path=$MODEL_PATH  max_new_tokens=$MAX_NEW_TOKENS  node=$(hostname)"
echo "[probe] serve_extra: $SERVE_EXTRA"
echo "=========================================================================="

# ── ensure serve env has libstdc++ + launch vLLM server ─────────────────────────
conda activate "$SERVE_ENV"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python -c "import vllm; print('[probe] serving with vllm', vllm.__version__)" || {
    echo "[probe] FATAL: vllm not importable in env $SERVE_ENV" >&2; exit 3; }

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --gpu-memory-utilization 0.92 \
    --port "$PORT" --host 127.0.0.1 \
    --no-enable-log-requests \
    $SERVE_EXTRA &
VLLM_PID=$!
trap 'echo "[probe] shutting down vLLM ${VLLM_PID}"; kill ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null || true' EXIT

# ── wait for /health (bounded — serving failure must surface fast) ──────────────
MAX_WAIT=1500; WAITED=0
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "[probe] RESULT MODEL=$MODEL serving=FAILED reason=vllm_process_died_during_load" ; exit 0
    fi
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "[probe] RESULT MODEL=$MODEL serving=FAILED reason=health_timeout_${MAX_WAIT}s" ; exit 0
    fi
    sleep 10; WAITED=$((WAITED+10))
    [ $((WAITED % 60)) -eq 0 ] && echo "[probe] waiting for vLLM... ${WAITED}s"
done
echo "[probe] vLLM ready after ${WAITED}s"

# ── client smoke (always melt-olala): g7 single-order, first 3 sources ──────────
OUT="results/mxprobe_${MODEL}_${SLURM_JOB_ID}"
T0=$(date +%s)
conda run -n melt-olala bash -lc "VLLM_BASE_URL='${VLLM_BASE_URL}' python run_stage2_experiment.py \
    --dataset g7-literature \
    --stage1-predictions results/stage1_frozen/g7-literature_qwen3-noLoRA_pathctx_T2_top20.tsv \
    --stage1-top-k 20 --stage1-description description_path_context \
    --description description_path_context \
    --llm-model '${MODEL_PATH}' --prompt-id d_subs_v2 \
    --max-new-tokens ${MAX_NEW_TOKENS} --llm-max-concurrency 16 \
    --smoke-test --output-dir '${OUT}'"
RC=$?
T1=$(date +%s); ELAPSED=$((T1-T0))

# ── report ──────────────────────────────────────────────────────────────────────
if [ $RC -ne 0 ] || [ ! -f "${OUT}/metrics.json" ]; then
    echo "[probe] RESULT MODEL=$MODEL serving=OK client=FAILED rc=$RC (check log)"; exit 0
fi
conda run -n melt-olala python - "$OUT" "$MODEL" "$ELAPSED" <<'PY'
import json, sys, csv, statistics as st
out, model, elapsed = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = json.load(open(f"{out}/metrics.json"))
toks=[]; n=0
with open(f"{out}/predictions.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        n+=1
        try: toks.append(int(r["n_tokens"]))
        except: pass
pf = m.get("reranker_parse_fail_rate")
rate = (n/elapsed*60) if elapsed else 0
tmax = max(toks) if toks else 0
tp95 = sorted(toks)[int(0.95*len(toks))] if toks else 0
print(f"[probe] RESULT MODEL={model} serving=OK pairs={n} elapsed={elapsed}s "
      f"throughput={rate:.1f}/min parse_fail={pf} cot_tokens_median={st.median(toks) if toks else 0} "
      f"p95={tp95} max={tmax}")
PY
echo "[probe] done MODEL=$MODEL"
