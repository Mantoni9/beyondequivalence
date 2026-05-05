#!/bin/bash
# T2-pin validation — main ablation, no LoRA (Job A).
# Mirrors jobs/job_dws_ablation_full.sh exactly except for the active pin:
# subB_pinned_config now resolves SUBB_PIN_ASYM = ("description_one_gen", "T2").
# 96 runs: 2 models (qwen3, nemo) x 6 datasets x 8 (A x B x C) permutations.
#
# W&B group naming follows the T4 main-ablation pattern (ablation_full_*)
# so the W&B project shows the runs in the same series. Differentiation
# vs. the T4 group ablation_full_2026-05-04_00-37-30_9f60152 is via SHA
# (this job runs from branch t2-pin-validation, SHA differs from main-
# ablation-full's 9f60152) and timestamp.
#
# The 4 pin-INDEPENDENT permutations (B=off: baseline / A / C / A+C) MUST
# reproduce the T4 runs bit-for-bit — implicit pipeline reproducibility
# check called out in the brief.

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
echo "Ablation full sweep (T2 pin). Group=${WANDB_GROUP}"
echo "Job=${SLURM_JOB_ID}  Node=$(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES  Time-limit: 24h"
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
echo "Ablation full (T2) done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "=========================================================================="
