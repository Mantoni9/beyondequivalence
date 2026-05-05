#!/bin/bash
# T2-Pin validation — main ablation, with LoRA on Nemo only (Job B).
# 96 runs: identical permutation matrix to Job A, but Nemo runs attach the
# WordNet-LoRA adapter from job 242725. Qwen3 runs WITHOUT LoRA per the
# Slide-13 decision (catastrophic forgetting at LoRA-eval 2026-05-05).
# Adapter set only via --lora-adapter-nemo; --lora-adapter-qwen3 is left
# unset, which makes main_ablation_sweep emit qwen3 runs with lora-off
# infix and nemo runs with lora-on infix in the SAME wandb group.

#SBATCH --job-name=t2pin_with_lora
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/t2pin_with_lora_%j.out
#SBATCH --error=logs/t2pin_with_lora_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.bwuni
set +a

mkdir -p results logs

# Adapter path — set via env to override; default points at the
# extracted Nemo adapter from training job 242725 + extraction
# 2026-05-05 (commit 5df8e20 of lora-subsumption-finetune).
NEMO_ADAPTER="${NEMO_ADAPTER:-lora_adapters/nemo_subsumption_lora_extracted}"
if [ ! -d "$NEMO_ADAPTER" ]; then
    echo "ERROR: NEMO_ADAPTER dir not found: $NEMO_ADAPTER" >&2
    exit 1
fi

TS=$(date +%Y-%m-%d_%H-%M-%S)
SHA=$(git rev-parse --short HEAD)
WANDB_GROUP="${T2PIN_WITH_LORA_GROUP:-sweep_lora_t2pin_${TS}_${SHA}}"

echo "=========================================================================="
echo "T2-pin validation (with Nemo LoRA). Group=${WANDB_GROUP}"
echo "Job=${SLURM_JOB_ID}  Node=$(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Pin: SUBB_PIN_ASYM=$(python -c 'from subB_pinned_config import SUBB_PIN_ASYM; print(SUBB_PIN_ASYM)')"
echo "Nemo adapter: ${NEMO_ADAPTER}"
echo "Qwen3 adapter: <none, per Slide-13 decision>"
echo "=========================================================================="

# Pre-eval LoRA sanity for the Nemo adapter only — symmetric to the
# established lora_eval pattern but without the qwen3 sanity step.
echo ""
echo "--- Pre-eval LoRA inference sanity (Nemo only) ---"
python lora_inference_sanity.py --model nemo --adapter "$NEMO_ADAPTER" \
    --report-path "results/${WANDB_GROUP}_sanity_nemo.json"
echo "--- Sanity passed; proceeding to 96-run eval ---"
echo ""

python run_subsumption_experiment.py \
    --ablation-sweep \
    --ablation-models qwen3-embedding-8b llama-embed-nemotron-8b \
    --ablation-datasets mouse-human g1-web g2-diseases g3-text g5-groceries g7-literature \
    --lora-adapter-nemo "$NEMO_ADAPTER" \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee "results/${WANDB_GROUP}.log"

echo ""
echo "=========================================================================="
echo "T2-pin with-LoRA done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "=========================================================================="
