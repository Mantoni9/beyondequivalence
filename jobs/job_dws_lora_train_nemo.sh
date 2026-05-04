#!/bin/bash
# LoRA full training — llama-embed-nemotron-8b on the WordNet-triplet set.
# Loader kwargs auto-sourced from prompt.get_loader_kwargs(model_id):
#   model_kwargs={attn_implementation="eager", torch_dtype="bfloat16"}
#   tokenizer_kwargs={padding_side="left"}
# These three pins are MANDATORY for Nemotron — eager-attn is required by
# the model card's bidirectional-attention path; bfloat16 matches released
# weights; left-padding matches the latent-attention pooler. The pins are
# verified before submission via:
#   conda run -n melt-olala python -c "from prompt import get_loader_kwargs; \
#       print(get_loader_kwargs('nvidia/llama-embed-nemotron-8b'))"
# Output (2026-05-04):
#   {'model_kwargs': {'attn_implementation': 'eager', 'torch_dtype': 'bfloat16'},
#    'tokenizer_kwargs': {'padding_side': 'left'}}

#SBATCH --job-name=lora_nemo_full
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lora_nemo_full_%j.out
#SBATCH --error=logs/lora_nemo_full_%j.err

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
WANDB_GROUP="${LORA_NEMO_WANDB_GROUP:-lora_nemo_full_${TS}_${SHA}}"

echo "=========================================================================="
echo "LoRA full Nemo. Group=${WANDB_GROUP}  Job=${SLURM_JOB_ID}  Node=$(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES  Time-limit: 24h"
echo "=========================================================================="

python finetune_lora.py \
    --model nemo \
    --triplets-path data/wordnet_triplets.jsonl \
    --output-dir lora_adapters/nemo_subsumption_lora \
    --batch-size 16 \
    --epochs 2 \
    --learning-rate 2e-5 \
    --gradient-checkpointing \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee logs/lora_nemo_full_${SLURM_JOB_ID}_train.log

echo ""
echo "=========================================================================="
echo "Nemo full training done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "Adapter saved at: lora_adapters/nemo_subsumption_lora"
echo "=========================================================================="
