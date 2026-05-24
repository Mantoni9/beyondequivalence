#!/bin/bash
# Main A x B ablation over the DIRECTED gold, bidirectional pipeline.
# 96 runs: 4 (A x B) perms x 2 models x 2 LoRA-states x 6 datasets.
#   matcher = MatcherAsymmetricRetrieval (emits '<' AND '>') — NOT the '<'-only
#   MatcherBidirectionalConsolidation. NO Lever C (cross-direction RRF is a
#   category error here). top-50, seed 42.
# Cross-model RRF fusion is a SEPARATE post-hoc CPU step
# (scripts/fuse_crossmodel_rrf.py) — not in this GPU job.

#SBATCH --job-name=ablbi
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablbi_%j.out
#SBATCH --error=logs/ablbi_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.dws
set +a
export HF_HOME=/work/amarkic/hf_cache

mkdir -p results logs

QWEN3_ADAPTER="${QWEN3_ADAPTER:-lora_adapters/qwen3_subsumption_lora_extracted}"
NEMO_ADAPTER="${NEMO_ADAPTER:-lora_adapters/nemo_subsumption_lora_extracted}"
for p in "$QWEN3_ADAPTER" "$NEMO_ADAPTER"; do
    if [ ! -d "$p" ]; then
        echo "ERROR: adapter dir not found: $p" >&2
        exit 1
    fi
done

TS=$(date +%Y-%m-%d_%H-%M-%S)
SHA=$(git rev-parse --short HEAD)
WANDB_GROUP="${ABLBI_WANDB_GROUP:-main_ablation_bidirectional_${TS}_${SHA}}"

echo "=========================================================================="
echo "Main A x B bidirectional ablation. Group=${WANDB_GROUP}"
echo "Job=${SLURM_JOB_ID}  Node=$(hostname)  GPU: $CUDA_VISIBLE_DEVICES  SHA=${SHA}"
echo "--------------------------------------------------------------------------"
echo "matcher = MatcherAsymmetricRetrieval ('<' AND '>')   Lever C = OFF   top-50"
echo "Lever A : turtle(description_one_gen) | path_context(description_path_context)"
echo "Lever B : default->$(python -c 'from subB_pinned_config import SUBB_DEFAULT_ASYM; print(SUBB_DEFAULT_ASYM[1])') (SUBB_DEFAULT_ASYM)  |  sub_b_pin->$(python -c 'from subB_pinned_config import SUBB_PIN_ASYM; print(SUBB_PIN_ASYM[1])') (SUBB_PIN_ASYM)"
echo "  pin proof: SUBB_DEFAULT_ASYM=$(python -c 'from subB_pinned_config import SUBB_DEFAULT_ASYM; print(SUBB_DEFAULT_ASYM)')  SUBB_PIN_ASYM=$(python -c 'from subB_pinned_config import SUBB_PIN_ASYM; print(SUBB_PIN_ASYM)')"
echo "4 perms: baseline(turtle,T1) A(pc,T1) B(turtle,T2) A+B(pc,T2)"
echo "Models x LoRA: qwen3{off,on}, nemo{off,on}  -> 4 x 2 x 2 x 6 = 96 runs"
echo "QWEN3 adapter: ${QWEN3_ADAPTER}"
echo "NEMO  adapter: ${NEMO_ADAPTER}"
echo "=========================================================================="

# Pre-run LoRA inference sanity for both adapters (gate; no-op load would
# silently reproduce baseline numbers otherwise).
echo ""
echo "--- Pre-run LoRA inference sanity ---"
python lora_inference_sanity.py --model qwen3 --adapter "$QWEN3_ADAPTER" \
    --report-path "results/${WANDB_GROUP}_sanity_qwen3.json"
python lora_inference_sanity.py --model nemo  --adapter "$NEMO_ADAPTER"  \
    --report-path "results/${WANDB_GROUP}_sanity_nemo.json"
echo "--- Sanity passed for both adapters; proceeding to 96-run ablation ---"
echo ""

python scripts/ablation_bidirectional.py \
    --lora-adapter-qwen3 "$QWEN3_ADAPTER" \
    --lora-adapter-nemo  "$NEMO_ADAPTER" \
    --datasets mouse-human g1-web g2-diseases g3-text g5-groceries g7-literature \
    --top-k-max 50 \
    --seed 42 \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee "results/${WANDB_GROUP}.log"

echo ""
echo "=========================================================================="
echo "Ablation done at $(date +%H:%M:%S). Group=${WANDB_GROUP}  SHA=${SHA}"
echo "Post-hoc fusion (CPU, run on login node after this finishes):"
echo "  python scripts/fuse_crossmodel_rrf.py --sha ${SHA}"
echo "=========================================================================="
