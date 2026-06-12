#!/bin/bash
# Stage-1 swapped-retrieval ablation (superclass recall ceiling fix).
# 12 runs: {qwen3-noLoRA (primary), nemo+LoRA (robustness side-run)} x 6 datasets.
# Each run produces ALL FOUR passes (s_broader, s_narrower, t_broader,
# t_narrower) from two invocations of the UNMODIFIED frozen matcher; the
# variants {baseline, v_sym, v_union} are offline pass subsets.
# Frozen levers: A=path_context, B=sub_b_pin (T2), top-50, seed 42.
# Built-in guard: the s-side passes must reproduce results/ablbi_*_d11c97e/
# predictions.tsv exactly (identity check; exit 3 on mismatch).

#SBATCH --job-name=swapabl
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --exclude=dws-16,dws-17
#SBATCH --output=logs/swapabl_%j.out
#SBATCH --error=logs/swapabl_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.dws
set +a
export HF_HOME=/work/amarkic/hf_cache

mkdir -p results logs

NEMO_ADAPTER="${NEMO_ADAPTER:-lora_adapters/nemo_subsumption_lora_extracted}"
if [ ! -d "$NEMO_ADAPTER" ]; then
    echo "ERROR: adapter dir not found: $NEMO_ADAPTER" >&2
    exit 1
fi

TS=$(date +%Y-%m-%d_%H-%M-%S)
SHA=$(git rev-parse --short HEAD)
WANDB_GROUP="${SWAP_WANDB_GROUP:-swap_ablation_${TS}_${SHA}}"

echo "=========================================================================="
echo "Swapped-retrieval ablation. Group=${WANDB_GROUP}"
echo "Job=${SLURM_JOB_ID}  Node=$(hostname)  GPU: $CUDA_VISIBLE_DEVICES  SHA=${SHA}"
echo "--------------------------------------------------------------------------"
echo "frozen: A=path_context  B=sub_b_pin(T2)  top-50  seed 42"
echo "  pin proof: SUBB_PIN_ASYM=$(python -c 'from subB_pinned_config import SUBB_PIN_ASYM; print(SUBB_PIN_ASYM)')"
echo "passes per run: s_broader s_narrower t_broader t_narrower (2 match calls)"
echo "configs: qwen3-noLoRA (primary) + nemo+LoRA (side-run)  x 6 datasets = 12 runs"
echo "identity check vs results/ablbi_*_d11c97e/ (must be present on this node)"
echo "NEMO adapter: ${NEMO_ADAPTER}"
echo "=========================================================================="

# Pre-flight: the d11c97e identity artifacts must be present — without them
# the runner's comparability gate would fail every run (exit 3).
echo ""
echo "--- Pre-flight: d11c97e identity artifacts ---"
MISSING=0
for ds in mouse-human g1-web g2-diseases g3-text g5-groceries g7-literature; do
    for prefix in "ablbi_qwen3-embedding-8b_lora-off" "ablbi_llama-embed-nemotron-8b_lora-on"; do
        d="results/${prefix}_A-path_context_B-sub_b_pin_${ds}_d11c97e"
        if [ ! -f "$d/predictions.tsv" ] || [ ! -f "$d/metrics.json" ]; then
            echo "MISSING: $d" >&2
            MISSING=1
        fi
    done
done
if [ "$MISSING" -ne 0 ]; then
    echo "ERROR: d11c97e identity artifacts incomplete — aborting before GPU use." >&2
    exit 1
fi
echo "--- All 12 identity targets present ---"

# Pre-run LoRA inference sanity for the nemo adapter (gate; a silent no-op
# load would reproduce baseline numbers otherwise). qwen3 runs without LoRA.
echo ""
echo "--- Pre-run LoRA inference sanity (nemo) ---"
python lora_inference_sanity.py --model nemo --adapter "$NEMO_ADAPTER" \
    --report-path "results/${WANDB_GROUP}_sanity_nemo.json"
echo "--- Sanity passed; proceeding to 12-run swap ablation ---"
echo ""

python scripts/ablation_swap.py \
    --configs qwen3-noLoRA nemo+LoRA \
    --datasets mouse-human g1-web g2-diseases g3-text g5-groceries g7-literature \
    --lora-adapter-nemo "$NEMO_ADAPTER" \
    --top-k-max 50 \
    --seed 42 \
    --identity-sha d11c97e \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee "results/${WANDB_GROUP}.log"

echo ""
echo "=========================================================================="
echo "Swap ablation done at $(date +%H:%M:%S). Group=${WANDB_GROUP}  SHA=${SHA}"
echo "Per-run artifacts: results/swap_*_${SHA}/{config.json,metrics.json,passes.tsv}"
echo "=========================================================================="
