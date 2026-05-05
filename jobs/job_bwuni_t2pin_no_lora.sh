#!/bin/bash
# T2-Pin validation — main ablation, no LoRA (Job A).
# 96 runs: 2 models (qwen3, nemo) x 6 datasets x 8 (A x B x C) permutations.
# B=on now resolves to ("description_one_gen", "T2") via subB_pinned_config.
# B=off (default) and the 4 pin-INDEPENDENT permutations (baseline / A / C / A+C)
# MUST reproduce the T4-pin runs bit-for-bit — implicit pipeline reproducibility
# check. Any drift on those four cells is a pipeline bug, not a pin effect.

#SBATCH --job-name=t2pin_no_lora
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/t2pin_no_lora_%j.out
#SBATCH --error=logs/t2pin_no_lora_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.bwuni
set +a

mkdir -p results logs

TS=$(date +%Y-%m-%d_%H-%M-%S)
SHA=$(git rev-parse --short HEAD)
WANDB_GROUP="${T2PIN_NO_LORA_GROUP:-sweep_all6_t2pin_${TS}_${SHA}}"

echo "=========================================================================="
echo "T2-pin validation (no LoRA). Group=${WANDB_GROUP}"
echo "Job=${SLURM_JOB_ID}  Node=$(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Pin: SUBB_PIN_ASYM=$(python -c 'from subB_pinned_config import SUBB_PIN_ASYM; print(SUBB_PIN_ASYM)')"
echo "=========================================================================="

python run_subsumption_experiment.py \
    --ablation-sweep \
    --ablation-models qwen3-embedding-8b llama-embed-nemotron-8b \
    --ablation-datasets mouse-human g1-web g2-diseases g3-text g5-groceries g7-literature \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee "results/${WANDB_GROUP}.log"

echo ""
echo "=========================================================================="
echo "T2-pin no-LoRA done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "=========================================================================="
