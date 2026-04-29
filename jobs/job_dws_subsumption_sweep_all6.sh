#!/bin/bash
#SBATCH --job-name=stage1_sweep_all6
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage1_sweep_all6_%j.out
#SBATCH --error=logs/stage1_sweep_all6_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.dws
set +a
export HF_HOME=/work/amarkic/hf_cache

DATASETS=(
    mouse-human
    g1-web
    g2-diseases
    g3-text
    g5-groceries
    g7-literature
)

CONFIGS=(
    "sbert symmetric"
    "qwen3-embedding-8b symmetric"
    "qwen3-embedding-8b asymmetric"
    "llama-embed-nemotron-8b symmetric"
    "llama-embed-nemotron-8b asymmetric"
)

TS=$(date +%Y-%m-%d_%H-%M-%S)
SHA=$(git rev-parse --short HEAD)
WANDB_GROUP="sweep_all6_${TS}_${SHA}"

echo "=========================================================================="
echo "Stage-1 sweep over ${#DATASETS[@]} datasets x ${#CONFIGS[@]} configs"
echo "Job ID: ${SLURM_JOB_ID}  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Timestamp: ${TS}  Git SHA: ${SHA}  W&B group: ${WANDB_GROUP}"
echo "=========================================================================="

mkdir -p results logs

for DATASET in "${DATASETS[@]}"; do
    for CFG in "${CONFIGS[@]}"; do
        read -r MODEL VARIANT <<< "$CFG"
        echo ""
        echo "--------------------------------------------------------------------------"
        echo "Run: dataset=${DATASET} model=${MODEL} variant=${VARIANT}  ($(date +%H:%M:%S))"
        echo "--------------------------------------------------------------------------"
        python run_subsumption_experiment.py \
            --model "$MODEL" \
            --instruction-variant "$VARIANT" \
            --dataset "$DATASET" \
            --wandb \
            --wandb-group "$WANDB_GROUP" \
            2>&1 | tee "results/run_${DATASET}_${TS}_${MODEL}_${VARIANT}.log"
    done
done

echo ""
echo "=========================================================================="
echo "Sweep ${WANDB_GROUP} completed at $(date +%H:%M:%S)"
echo "=========================================================================="
echo ""
printf "%-18s | %-30s | %-7s | %-12s | %-13s\n" \
    "dataset" "model/variant" "stddev" "lax/super/20" "rel_strict/20"
echo "-------------------+--------------------------------+---------+--------------+---------------"
for log in results/run_*_${TS}_*.log; do
    base=$(basename "$log" .log)
    rest=${base#run_}
    dataset=${rest%%_${TS}_*}
    name=${rest#*_${TS}_}
    stddev=$(grep -oP '"stddev":\s*\K[0-9.]+' "$log" | head -1)
    rec_lax=$(grep -oP "recall_at_k_lax/superclass/k=20 = \K[0-9.]+" "$log" | head -1)
    rec_strict=$(grep -oP "recall_at_k_per_relation_strict/superclass/k=20 = \K[0-9.]+" "$log" | head -1)
    stddev_short=$(printf "%.4f" "${stddev:-0}")
    printf "%-18s | %-30s | %-7s | %-12s | %-13s\n" \
        "$dataset" "$name" "$stddev_short" "${rec_lax:-NA}" "${rec_strict:-NA}"
done
