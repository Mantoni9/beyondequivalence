#!/bin/bash
# Sub-B smoke test — validates the warm-embedder refactor, the truncation
# diagnostic, and the resume check before the 300/330 production sweeps.
#
# Two iterations, both on g7-literature (smallest dataset, ~80 refs):
#   1. sbert symmetric × description_one_gen × no instruction
#   2. qwen3-embedding-8b symmetric × description_one_gen × S1
#
# Wall-clock target: < 5 min. If anything in the embedder/cache pipeline
# misbehaves, it surfaces here, not in the 12h production run.

#SBATCH --job-name=subB_smoke
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/subB_smoke_%j.out
#SBATCH --error=logs/subB_smoke_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate melt-olala

set -a
source .env.dws
set +a
export HF_HOME=/work/amarkic/hf_cache

mkdir -p results logs

# Two-model smoke. We override SUBB_DESCRIPTION_METHODS / SUBB_SYM_TEMPLATE_IDS
# is not exposed at CLI, so we run two single-iteration sub-B sweeps each
# limited via --sub-b-models to one model — and rely on the smoke probe being
# skipped because --smoke-test is NOT set (we want full Recall@K to validate
# truncation fields). The default datasets list is forced to one tiny set.
TS=$(date +%Y-%m-%d_%H-%M-%S)
SHA=$(git rev-parse --short HEAD)
WANDB_GROUP="subB_smoke_${TS}_${SHA}"

echo "=========================================================================="
echo "Sub-B smoke. Group=${WANDB_GROUP}  Job=${SLURM_JOB_ID}  Node=$(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================================================="

# Smoke #1 — sbert (cheap, ~30 s) over all 5 description methods, 1 template (none).
# This validates the loop / KG cache / matcher cache / truncation diagnostic
# / resume check on a non-instruction-aware model.
python run_subsumption_experiment.py \
    --sub-b-sweep \
    --sub-b-models sbert \
    --sub-b-variants symmetric \
    --sub-b-datasets g7-literature \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee "results/${WANDB_GROUP}_sbert.log"

# Smoke #2 — qwen3 with one full sym template set on g7-lit.
# This exercises the prompt-prefix path on an 8B instruction-aware model
# and confirms tokens_truncated/* gets populated correctly with non-zero
# limit, plus the warm-embedder behaviour for 25 inner iterations
# (5 descriptions × 5 templates) without reloading weights.
python run_subsumption_experiment.py \
    --sub-b-sweep \
    --sub-b-models qwen3-embedding-8b \
    --sub-b-variants symmetric \
    --sub-b-datasets g7-literature \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee "results/${WANDB_GROUP}_qwen3.log"

echo ""
echo "=========================================================================="
echo "Smoke done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "=========================================================================="
echo ""
# Quick assertion: every metrics.json under group must carry the truncation fields.
python - <<'PY'
import json, glob, sys, os
ok = bad = 0
for d in glob.glob("results/subB_*"):
    mp = os.path.join(d, "metrics.json")
    if not os.path.isfile(mp):
        continue
    m = json.loads(open(mp).read())
    last = m.get("matcher_last_run_metrics", {})
    has_src = any(k.startswith("tokens_truncated/source") for k in last)
    has_tgt = any(k.startswith("tokens_truncated/target") for k in last)
    if has_src and has_tgt:
        ok += 1
    else:
        bad += 1
        print(f"MISSING TRUNCATION FIELDS in {d}: keys={sorted(last.keys())}")
print(f"truncation fields present: ok={ok}  bad={bad}")
sys.exit(1 if bad else 0)
PY
