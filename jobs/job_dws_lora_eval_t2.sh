#!/bin/bash
# T2-pin validation — main ablation with WordNet Nemo LoRA (Job B).
# Mirrors jobs/job_dws_lora_eval.sh except:
#   - Active pin is now T2 (subB_pinned_config: SUBB_PIN_ASYM = T2).
#   - Qwen3 runs WITHOUT adapter per Slide-13 decision (qwen3 catastrophic
#     forgetting at LoRA-eval 2026-05-05). --lora-adapter-qwen3 stays unset;
#     main_ablation_sweep emits qwen3 runs with run-name infix "lora-off"
#     and nemo runs with "lora-on" inside the same W&B group.
#
# W&B group naming follows the T4 LoRA-eval pattern
# (ablation_lora_finetune_*) so the W&B project shows the runs in the same
# series. Differentiation vs. the T4 group
# ablation_lora_finetune_2026-05-05_09-33-31_5df8e20 is via SHA (this
# job runs from branch t2-pin-validation) and timestamp.

#SBATCH --job-name=lora_eval
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lora_eval_%j.out
#SBATCH --error=logs/lora_eval_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.dws
set +a
export HF_HOME=/work/amarkic/hf_cache

mkdir -p results logs

# Adapter path — set via env to override; default points at the
# extracted Nemo adapter from training job 242725 + extraction
# 2026-05-05 (commit 5df8e20 on lora-subsumption-finetune).
NEMO_ADAPTER="${NEMO_ADAPTER:-lora_adapters/nemo_subsumption_lora_extracted}"
if [ ! -d "$NEMO_ADAPTER" ]; then
    echo "ERROR: NEMO_ADAPTER dir not found: $NEMO_ADAPTER" >&2
    exit 1
fi

TS=$(date +%Y-%m-%d_%H-%M-%S)
SHA=$(git rev-parse --short HEAD)
WANDB_GROUP="${LORA_EVAL_WANDB_GROUP:-ablation_lora_finetune_${TS}_${SHA}}"

echo "=========================================================================="
echo "LoRA eval (T2 pin, Nemo only). Group=${WANDB_GROUP}"
echo "Job=${SLURM_JOB_ID}  Node=$(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Pin: SUBB_PIN_ASYM=$(python -c 'from subB_pinned_config import SUBB_PIN_ASYM; print(SUBB_PIN_ASYM)')"
echo "Nemo adapter:  ${NEMO_ADAPTER}"
echo "Qwen3 adapter: <none, per Slide-13 decision>"
echo "=========================================================================="

# Pre-eval LoRA inference sanity for the Nemo adapter only.
echo ""
echo "--- Pre-eval LoRA inference sanity (Nemo only) ---"
python lora_inference_sanity.py --model nemo --adapter "$NEMO_ADAPTER" \
    --report-path "results/${WANDB_GROUP}_sanity_nemo.json"
echo "--- Sanity passed; proceeding to 96-run eval ---"
echo ""

python run_subsumption_experiment.py \
    --ablation-sweep \
    --ablation-models qwen3-embedding-8b llama-embed-nemotron-8b \
    --ablation-datasets mouse-human g1-web g2-diseases g3-text g5-groceries g7-literature \
    --lora-adapter-nemo "$NEMO_ADAPTER" \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee "results/${WANDB_GROUP}.log"

echo ""
echo "=========================================================================="
echo "LoRA eval (T2) done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "=========================================================================="
