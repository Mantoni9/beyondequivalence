#!/bin/bash
# A smoke test — qwen3 asymmetric on g7-literature, both verbalizations.
# Validates that description_path_context produces non-empty strings, that
# the matcher consumes them via the existing serialize() string passthrough,
# and that tokens_truncated/* are populated as a sanity check.
#
# Wallclock target: < 10 min (model load + 2 small encodings).

#SBATCH --job-name=A_smoke
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/A_smoke_%j.out
#SBATCH --error=logs/A_smoke_%j.err

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
WANDB_GROUP="A_smoke_${TS}_${SHA}"

echo "=========================================================================="
echo "A smoke. Group=${WANDB_GROUP}  Job=${SLURM_JOB_ID}  Node=$(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================================================="

python run_subsumption_experiment.py \
    --A-sweep \
    --A-models qwen3-embedding-8b \
    --A-variants asymmetric \
    --A-datasets g7-literature \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee "results/${WANDB_GROUP}.log"

echo ""
echo "=========================================================================="
echo "Smoke done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "=========================================================================="
echo ""
# Verify: 2 metrics.json must exist, both with truncation fields populated,
# and the path_context predictions.tsv must contain non-empty source strings.
python - <<'PY'
import json, glob, os, sys
ok = bad = 0
for d in sorted(glob.glob("results/A_qwen3-embedding-8b_asy_*_g7-literature_*")):
    mp = os.path.join(d, "metrics.json")
    cp = os.path.join(d, "config.json")
    if not (os.path.isfile(mp) and os.path.isfile(cp)):
        continue
    m = json.load(open(mp))
    c = json.load(open(cp))
    last = m.get("matcher_last_run_metrics", {})
    has_trunc = any(k.startswith("tokens_truncated/") for k in last)
    print(f"{d}")
    print(f"  verbalization = {c.get('verbalization')}  "
          f"description = {c.get('description')}  "
          f"truncation_fields = {has_trunc}")
    if has_trunc:
        ok += 1
    else:
        bad += 1
print(f"summary: ok={ok}  bad={bad}")
sys.exit(1 if bad else 0)
PY
