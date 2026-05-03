#!/bin/bash
# Main ablation full sweep — A x B x C, asymmetric only, broader-direction.
# 96 runs: 2 models (qwen3, nemo) x 6 datasets x 8 permutations
#          (A in {turtle, path_context} x B in {default, sub_b_pin}
#                                       x C in {none, rrf}).
# Sequential on 1 GPU. Per-run VRAM cleanup in finally-block.

#SBATCH --job-name=ablation_full
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ablation_full_%j.out
#SBATCH --error=logs/ablation_full_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.dws
set +a
export HF_HOME=/work/amarkic/hf_cache

mkdir -p results logs

TS=$(date +%Y-%m-%d_%H-%M-%S)
SHA=$(git rev-parse --short HEAD)
WANDB_GROUP="${ABLATION_WANDB_GROUP:-ablation_full_${TS}_${SHA}}"

echo "=========================================================================="
echo "Ablation full sweep. Group=${WANDB_GROUP}"
echo "Job=${SLURM_JOB_ID}  Node=$(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES  Time-limit: 24h"
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
echo "Ablation full done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "=========================================================================="
