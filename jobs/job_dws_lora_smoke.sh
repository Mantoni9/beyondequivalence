#!/bin/bash
# LoRA fine-tuning smoke — qwen3, 5000 train rows, 1 epoch.
# Validates: PEFT-adapter registers on top of pretrained model,
# trainable param ratio is in PEFT range (<<1%), gradient
# checkpointing works, train_loss decays, adapter changes embeddings
# vs. base on a 10-text sanity probe.

#SBATCH --job-name=lora_smoke
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lora_smoke_%j.out
#SBATCH --error=logs/lora_smoke_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.dws
set +a
export HF_HOME=/work/amarkic/hf_cache

mkdir -p data results logs lora_adapters

# 1) Ensure NLTK + WordNet are available before training.
python -m pip install --quiet nltk peft accelerate datasets
python -c "import nltk; nltk.download('wordnet', quiet=True)"

# 2) Build triplets if not already present.
if [ ! -f data/wordnet_triplets.train.jsonl ]; then
    echo "Generating WordNet triplets…"
    python prepare_wordnet_triplets.py \
        --output-path data/wordnet_triplets.jsonl \
        2>&1 | tee logs/lora_smoke_${SLURM_JOB_ID}_triplets.log
else
    echo "Re-using existing data/wordnet_triplets.{train,val}.jsonl"
fi

# 3) Smoke training: 5000 rows, 1 epoch, qwen3.
TS=$(date +%Y-%m-%d_%H-%M-%S)
SHA=$(git rev-parse --short HEAD)
WANDB_GROUP="lora_smoke_${TS}_${SHA}"

echo "=========================================================================="
echo "LoRA smoke. Group=${WANDB_GROUP}  Job=${SLURM_JOB_ID}  Node=$(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================================================="

python finetune_lora.py \
    --model qwen3 \
    --triplets-path data/wordnet_triplets.jsonl \
    --output-dir lora_adapters/qwen3_subsumption_lora_smoke \
    --smoke-triplets 5000 \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --max-seq-length 256 \
    --gradient-checkpointing \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee logs/lora_smoke_${SLURM_JOB_ID}_train.log

echo ""
echo "=========================================================================="
echo "Smoke done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "=========================================================================="
echo ""
# 4) Final assertion: smoke summary JSON must show adapter_has_effect=True.
python - <<'PY'
import json, glob, os, sys
files = sorted(glob.glob("results/lora_smoke_qwen3_*.json"))
if not files:
    sys.exit("No smoke summary written.")
summary = json.load(open(files[-1]))
print("Smoke summary:", json.dumps(summary, indent=2))
if not summary.get("adapter_has_effect"):
    sys.exit("FAIL: adapter has no measurable effect vs. base.")
print("PASS: adapter changes embeddings.")
PY
