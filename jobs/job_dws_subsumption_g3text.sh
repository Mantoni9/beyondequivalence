#!/bin/bash
#SBATCH --job-name=stage1_g3text
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/stage1_g3text_%j.out
#SBATCH --error=logs/stage1_g3text_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.dws
set +a
export HF_HOME=/work/amarkic/hf_cache

DATASET=g3-text
TS=$(date +%Y-%m-%d_%H-%M-%S)

echo "=========================================================================="
echo "Stage-1 hauptlauf: dataset=${DATASET}"
echo "Job ID: ${SLURM_JOB_ID}  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Timestamp: ${TS}"
echo "Git SHA: $(git rev-parse --short HEAD)"
echo "=========================================================================="

CONFIGS=(
    "sbert symmetric"
    "qwen3-embedding-8b symmetric"
    "qwen3-embedding-8b asymmetric"
    "llama-embed-nemotron-8b symmetric"
    "llama-embed-nemotron-8b asymmetric"
)

mkdir -p results

for CFG in "${CONFIGS[@]}"; do
    read -r MODEL VARIANT <<< "$CFG"
    echo ""
    echo "--------------------------------------------------------------------------"
    echo "Run: model=${MODEL} variant=${VARIANT} dataset=${DATASET}  ($(date +%H:%M:%S))"
    echo "--------------------------------------------------------------------------"
    python run_subsumption_experiment.py \
        --model "$MODEL" \
        --instruction-variant "$VARIANT" \
        --dataset "$DATASET" \
        --wandb \
        2>&1 | tee "results/run_${DATASET}_${TS}_${MODEL}_${VARIANT}.log"
done

echo ""
echo "=========================================================================="
echo "All 5 runs on ${DATASET} completed at $(date +%H:%M:%S)"
echo "=========================================================================="
echo ""
printf "%-50s | %-7s | %-12s | %-13s\n" "Run" "stddev" "lax/super/20" "rel_strict/20"
echo "---------------------------------------------------+---------+--------------+---------------"
for log in results/run_${DATASET}_${TS}_*.log; do
    name=$(basename "$log" .log | sed "s/run_${DATASET}_${TS}_//")
    stddev=$(grep -oP '"stddev":\s*\K[0-9.]+' "$log" | head -1)
    rec_lax=$(grep -oP "recall_at_k_lax/superclass/k=20 = \K[0-9.]+" "$log" | head -1)
    rec_strict=$(grep -oP "recall_at_k_per_relation_strict/superclass/k=20 = \K[0-9.]+" "$log" | head -1)
    stddev_short=$(printf "%.4f" "${stddev:-0}" 2>/dev/null || echo "?")
    printf "%-50s | %-7s | %-12s | %-13s\n" \
        "$name" "${stddev_short}" "${rec_lax:-?}" "${rec_strict:-?}"
done