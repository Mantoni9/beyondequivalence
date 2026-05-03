#!/bin/bash
# Sub-B description-ablation — SYM job (sbert + qwen3 + nemotron, 330 runs).
# sbert is run with template_id=null (1 config × 5 depths × 6 datasets = 30).
# Embedder warm per (model, variant); KG cached per dataset; resume-safe.

#SBATCH --job-name=subB_sym
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/subB_sym_%j.out
#SBATCH --error=logs/subB_sym_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.dws
set +a
export HF_HOME=/work/amarkic/hf_cache

mkdir -p results logs

# Same SUBB_WANDB_GROUP convention as the ASYM job — see comment there.
TS=$(date +%Y-%m-%d_%H-%M-%S)
SHA=$(git rev-parse --short HEAD)
WANDB_GROUP="${SUBB_WANDB_GROUP:-subB_descablation_${TS}_${SHA}}"

echo "=========================================================================="
echo "Sub-B SYM job. Group=${WANDB_GROUP}  Job=${SLURM_JOB_ID}  Node=$(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES  Time-limit: 12h"
echo "=========================================================================="

python run_subsumption_experiment.py \
    --sub-b-sweep \
    --sub-b-models sbert qwen3-embedding-8b llama-embed-nemotron-8b \
    --sub-b-variants symmetric \
    --sub-b-datasets mouse-human g1-web g2-diseases g3-text g5-groceries g7-literature \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee "results/${WANDB_GROUP}_sym.log"

echo ""
echo "=========================================================================="
echo "SYM job done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "=========================================================================="
