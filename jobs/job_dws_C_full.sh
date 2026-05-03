#!/bin/bash
# C full sweep — bidirectional consolidation via RRF, asymmetric only.
# 24 runs: 2 models (qwen3, nemo) x 6 datasets x 2 fusion configs (none, rrf).
#
# Embedder warm per model — 2 model loads total. KG cached per dataset
# (6 loads total). Resume-safe (group + SHA-tolerant). Direction is
# fixed to broader; output relation is '<'. Per THESIS_NOTES.md.

#SBATCH --job-name=C_full
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/C_full_%j.out
#SBATCH --error=logs/C_full_%j.err

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
WANDB_GROUP="${C_WANDB_GROUP:-C_bidirectional_${TS}_${SHA}}"

echo "=========================================================================="
echo "C full sweep. Group=${WANDB_GROUP}  Job=${SLURM_JOB_ID}  Node=$(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES  Time-limit: 4h"
echo "=========================================================================="

python run_subsumption_experiment.py \
    --C-sweep \
    --C-models qwen3-embedding-8b llama-embed-nemotron-8b \
    --C-datasets mouse-human g1-web g2-diseases g3-text g5-groceries g7-literature \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee "results/${WANDB_GROUP}.log"

echo ""
echo "=========================================================================="
echo "C full done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "=========================================================================="
