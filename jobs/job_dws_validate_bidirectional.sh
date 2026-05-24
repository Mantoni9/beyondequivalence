#!/bin/bash
# Bidirectional pipeline validation — single config, all 6 datasets, NOT a sweep.
# Functional test that the narrower-pass '>' direction reaches
# evaluation_recall.per_relation_strict.superclass with real numbers.
#
# Config: Nemo + LoRA, Lever A=path_context, MatcherAsymmetricRetrieval
# (emits '<' AND '>'), Lever C OFF, Lever B OFF (T1), top-20.
#
# Nothing is frozen and no model decision is made here. New W&B group
# validation_bidirectional_<TS>_<SHA> so it does not mix with the '<'-only runs.

#SBATCH --job-name=valbi
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/valbi_%j.out
#SBATCH --error=logs/valbi_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.dws
set +a
export HF_HOME=/work/amarkic/hf_cache

mkdir -p results logs

# Adapter path — override via env. Default = extracted Nemo adapter
# (training job 242725 + extraction commit 5df8e20).
NEMO_ADAPTER="${NEMO_ADAPTER:-lora_adapters/nemo_subsumption_lora_extracted}"
if [ ! -d "$NEMO_ADAPTER" ]; then
    echo "ERROR: NEMO_ADAPTER dir not found: $NEMO_ADAPTER" >&2
    exit 1
fi

TS=$(date +%Y-%m-%d_%H-%M-%S)
SHA=$(git rev-parse --short HEAD)
WANDB_GROUP="${VALBI_WANDB_GROUP:-validation_bidirectional_${TS}_${SHA}}"

echo "=========================================================================="
echo "Bidirectional validation. Group=${WANDB_GROUP}"
echo "Job=${SLURM_JOB_ID}  Node=$(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "model=llama-embed-nemotron-8b + LoRA  A=path_context  matcher=Asym  C=off"
echo "Lever B OFF -> SUBB_DEFAULT_ASYM=$(python -c 'from subB_pinned_config import SUBB_DEFAULT_ASYM; print(SUBB_DEFAULT_ASYM)')"
echo "Nemo adapter: ${NEMO_ADAPTER}"
echo "=========================================================================="

# Pre-run LoRA inference sanity (Nemo adapter), same gate as the lora-eval jobs.
echo ""
echo "--- Pre-run LoRA inference sanity (Nemo) ---"
python lora_inference_sanity.py --model nemo --adapter "$NEMO_ADAPTER" \
    --report-path "results/${WANDB_GROUP}_sanity_nemo.json"
echo "--- Sanity passed; proceeding to validation run ---"
echo ""

python scripts/validate_bidirectional.py \
    --model llama-embed-nemotron-8b \
    --lora-adapter-nemo "$NEMO_ADAPTER" \
    --description description_path_context \
    --asym-template-id T1 \
    --top-k-max 20 \
    --datasets mouse-human g1-web g2-diseases g3-text g5-groceries g7-literature \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee "results/${WANDB_GROUP}.log"

echo ""
echo "=========================================================================="
echo "Bidirectional validation done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "Result table is in the tail of results/${WANDB_GROUP}.log"
echo "=========================================================================="
