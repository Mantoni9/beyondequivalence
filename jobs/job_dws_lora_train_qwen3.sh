#!/bin/bash
# LoRA full training — Qwen3-Embedding-8B on the WordNet-triplet set.
# 176k triplets x 2 epochs ~10h on a single A6000 (smoke 5k/1ep was 8m).
# Loader kwargs auto-sourced from prompt.get_loader_kwargs(model_id):
#   tokenizer_kwargs={padding_side="left"} for Qwen3.

#SBATCH --job-name=lora_qwen3_full
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lora_qwen3_full_%j.out
#SBATCH --error=logs/lora_qwen3_full_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.dws
set +a
export HF_HOME=/work/amarkic/hf_cache

mkdir -p data results logs lora_adapters

if [ ! -f data/wordnet_triplets.train.jsonl ]; then
    echo "ERROR: data/wordnet_triplets.train.jsonl missing. Run prepare_wordnet_triplets.py first." >&2
    exit 1
fi

TS=$(date +%Y-%m-%d_%H-%M-%S)
SHA=$(git rev-parse --short HEAD)
WANDB_GROUP="${LORA_QWEN3_WANDB_GROUP:-lora_qwen3_full_${TS}_${SHA}}"

echo "=========================================================================="
echo "LoRA full Qwen3. Group=${WANDB_GROUP}  Job=${SLURM_JOB_ID}  Node=$(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES  Time-limit: 24h"
echo "=========================================================================="

python finetune_lora.py \
    --model qwen3 \
    --triplets-path data/wordnet_triplets.jsonl \
    --output-dir lora_adapters/qwen3_subsumption_lora \
    --batch-size 16 \
    --epochs 2 \
    --learning-rate 2e-5 \
    --gradient-checkpointing \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee logs/lora_qwen3_full_${SLURM_JOB_ID}_train.log

echo ""
echo "=========================================================================="
echo "Qwen3 full training done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "Adapter saved at: lora_adapters/qwen3_subsumption_lora"
echo "=========================================================================="
