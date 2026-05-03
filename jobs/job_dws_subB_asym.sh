#!/bin/bash
# Sub-B description-ablation — ASYM job (qwen3 + nemotron, 300 runs).
# Embedder warm per (model, variant); KG cached per dataset; resume-safe.

#SBATCH --job-name=subB_asym
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/subB_asym_%j.out
#SBATCH --error=logs/subB_asym_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.dws
set +a
export HF_HOME=/work/amarkic/hf_cache

mkdir -p results logs

# The W&B group is shared with the SYM job. Both jobs read it from the
# environment if SUBB_WANDB_GROUP is exported by the operator before sbatch;
# otherwise this job mints its own. To bundle them in W&B, set
#   export SUBB_WANDB_GROUP="subB_descablation_$(date +%Y-%m-%d_%H-%M-%S)_$(git -C /work/amarkic/beyondequivalence rev-parse --short HEAD)"
# and run sbatch on both jobs in the same shell.
TS=$(date +%Y-%m-%d_%H-%M-%S)
SHA=$(git rev-parse --short HEAD)
WANDB_GROUP="${SUBB_WANDB_GROUP:-subB_descablation_${TS}_${SHA}}"

echo "=========================================================================="
echo "Sub-B ASYM job. Group=${WANDB_GROUP}  Job=${SLURM_JOB_ID}  Node=$(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES  Time-limit: 12h"
echo "=========================================================================="

python run_subsumption_experiment.py \
    --sub-b-sweep \
    --sub-b-models qwen3-embedding-8b llama-embed-nemotron-8b \
    --sub-b-variants asymmetric \
    --sub-b-datasets mouse-human g1-web g2-diseases g3-text g5-groceries g7-literature \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee "results/${WANDB_GROUP}_asym.log"

echo ""
echo "=========================================================================="
echo "ASYM job done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "=========================================================================="
