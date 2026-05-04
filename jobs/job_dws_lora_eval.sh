#!/bin/bash
# LoRA-finetune evaluation — B+C config with the trained adapters.
# 24 runs: 2 models x 6 datasets x 1 (A=turtle) x 1 (B=sub_b_pin) x 1 (C=rrf).
#
# We piggyback on main_ablation_sweep with --ablation-models on the 2 models
# and let the spec-generator iterate the 8 permutations per (model, dataset).
# The post-hoc compare script filters down to the B+C cell only — we run
# all 8 permutations because skipping is cheap when only 2 of them have
# meaningful baselines, but keep options open for sub-questions later.
#
# Per-model adapter paths via --lora-adapter-qwen3 and --lora-adapter-nemo.

#SBATCH --job-name=lora_eval
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
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

TS=$(date +%Y-%m-%d_%H-%M-%S)
SHA=$(git rev-parse --short HEAD)
WANDB_GROUP="${LORA_EVAL_WANDB_GROUP:-ablation_lora_finetune_${TS}_${SHA}}"

# Adapter paths — set these as env vars before sbatch, or override here.
# Default points at the *_extracted dirs produced by
# extract_lora_from_hybrid_save.py (PEFT-conformant adapter_config.json
# + adapter_model.safetensors). The original *_subsumption_lora dirs
# from the Trainer are HYBRID saves (base+lora interleaved) and CANNOT
# be loaded via PeftModel.from_pretrained — see commit message of
# extract_lora_from_hybrid_save.py for the full forensics.
QWEN3_ADAPTER="${QWEN3_ADAPTER:-lora_adapters/qwen3_subsumption_lora_extracted}"
NEMO_ADAPTER="${NEMO_ADAPTER:-lora_adapters/nemo_subsumption_lora_extracted}"

# Sanity: refuse to start if adapters don't exist.
for p in "$QWEN3_ADAPTER" "$NEMO_ADAPTER"; do
    if [ ! -d "$p" ]; then
        echo "ERROR: adapter dir not found: $p" >&2
        exit 1
    fi
done

echo "=========================================================================="
echo "LoRA eval. Group=${WANDB_GROUP}  Job=${SLURM_JOB_ID}  Node=$(hostname)"
echo "QWEN3 adapter: ${QWEN3_ADAPTER}"
echo "NEMO  adapter: ${NEMO_ADAPTER}"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================================================="

# Pre-eval sanity: each adapter must measurably change embeddings vs.
# the base model. Without this, the 24-run sweep would silently
# reproduce baseline numbers if the inference-side adapter load no-ops.
# See finetune_lora.py / smoke 242696 for the rationale.
echo ""
echo "--- Pre-eval LoRA inference sanity ---"
python lora_inference_sanity.py --model qwen3 --adapter "$QWEN3_ADAPTER" \
    --report-path "results/${WANDB_GROUP}_sanity_qwen3.json"
python lora_inference_sanity.py --model nemo  --adapter "$NEMO_ADAPTER"  \
    --report-path "results/${WANDB_GROUP}_sanity_nemo.json"
echo "--- Sanity passed for both adapters; proceeding to 24-run eval ---"
echo ""

python run_subsumption_experiment.py \
    --ablation-sweep \
    --ablation-models qwen3-embedding-8b llama-embed-nemotron-8b \
    --ablation-datasets mouse-human g1-web g2-diseases g3-text g5-groceries g7-literature \
    --lora-adapter-qwen3 "$QWEN3_ADAPTER" \
    --lora-adapter-nemo  "$NEMO_ADAPTER" \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee "results/${WANDB_GROUP}.log"

echo ""
echo "=========================================================================="
echo "LoRA eval done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "=========================================================================="
