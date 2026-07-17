#!/bin/bash
#SBATCH --job-name=st1_vdi
#SBATCH --partition=gpu-vram-48gb
#SBATCH --qos=max2gpu5d
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/st1_vdi_%j.out
#SBATCH --error=logs/st1_vdi_%j.err
# Stage 1 on the VDI->eBay gold case, FROZEN config only (single cell):
# qwen3-embedding-8b, no LoRA, path_context (Lever A), T2/sub_b_pin (Lever B),
# MatcherAsymmetricRetrieval, top-50 deep list — the identical code path that
# produced the d11c97e stage1_frozen TSVs, via the grid filters of
# scripts/ablation_bidirectional.py. Afterwards the predictions are symlinked
# under the stage1_frozen naming convention the Stage-2 matrix runner expects.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala
set -a; source .env.dws; set +a
export HF_HOME=/work/amarkic/hf_cache

echo "=== Stage-1 vdi-ebay (frozen config) ==="
echo "Job ${SLURM_JOB_ID}  Node $(hostname)  SHA $(git rev-parse --short HEAD)  $(date +%F_%T)"

python scripts/ablation_bidirectional.py \
    --datasets vdi-ebay \
    --models qwen3-embedding-8b --lora-modes off \
    --A-labels path_context --B-labels sub_b_pin \
    --top-k-max 50 --seed 42 --wandb

SHA=$(git rev-parse --short HEAD)
SRC="results/ablbi_qwen3-embedding-8b_lora-off_A-path_context_B-sub_b_pin_vdi-ebay_${SHA}/predictions.tsv"
DST="results/stage1_frozen/vdi-ebay_qwen3-noLoRA_pathctx_T2_top20.tsv"
if [ ! -f "$SRC" ]; then echo "FATAL: $SRC fehlt" >&2; exit 5; fi
mkdir -p results/stage1_frozen
ln -sf "$(readlink -f "$SRC")" "$DST"
echo "stage1_frozen link: $DST -> $(readlink -f "$DST")  ($(wc -l < "$DST") Zeilen)"
echo "DONE"
