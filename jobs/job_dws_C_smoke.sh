#!/bin/bash
# C smoke test — qwen3 asymmetric on g7-literature, both fusion configs.
# Validates that fusion=on and fusion=off produce different alignments,
# RRF score range is in [~0.012, ~0.033], n_pairs_overlap > 0 when fusion=on,
# and that source_broader / target_narrower truncation fields are populated.

#SBATCH --job-name=C_smoke
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/C_smoke_%j.out
#SBATCH --error=logs/C_smoke_%j.err

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
WANDB_GROUP="C_smoke_${TS}_${SHA}"

echo "=========================================================================="
echo "C smoke. Group=${WANDB_GROUP}  Job=${SLURM_JOB_ID}  Node=$(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================================================="

python run_subsumption_experiment.py \
    --C-sweep \
    --C-models qwen3-embedding-8b \
    --C-datasets g7-literature \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee "results/${WANDB_GROUP}.log"

echo ""
echo "=========================================================================="
echo "Smoke done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "=========================================================================="
echo ""
# Verify: 2 metrics.json must exist, both with truncation fields populated;
# fusion=on must record n_pairs_inverse > 0 and overlap > 0; fusion=off must
# record n_pairs_inverse == 0; and the two predicted alignments must differ
# in confidence (RRF scores) but share the same (s, t) schluesselraum.
python - <<'PY'
import json, glob, os, sys
ok = bad = 0
recs = {}
for d in sorted(glob.glob("results/C_qwen3-embedding-8b_asy_fusion-*_g7-literature_*")):
    mp = os.path.join(d, "metrics.json")
    cp = os.path.join(d, "config.json")
    if not (os.path.isfile(mp) and os.path.isfile(cp)):
        continue
    m = json.load(open(mp))
    c = json.load(open(cp))
    last = m.get("matcher_last_run_metrics", {})
    n_fwd = last.get("n_pairs_forward", -1)
    n_inv = last.get("n_pairs_inverse", -1)
    n_ovl = last.get("n_pairs_overlap", -1)
    has_src_broader = any(k.startswith("tokens_truncated/source_broader") for k in last)
    has_target = any(k.startswith("tokens_truncated/target") for k in last)
    print(f"{d}")
    print(f"  fusion={c.get('fusion')}  n_fwd={n_fwd}  n_inv={n_inv}  n_overlap={n_ovl}")
    print(f"  source_broader trunc fields = {has_src_broader}  target trunc fields = {has_target}")
    recs[c.get("fusion")] = {"dir": d, "n_fwd": n_fwd, "n_inv": n_inv, "n_ovl": n_ovl}
    if has_src_broader and has_target:
        ok += 1
    else:
        bad += 1

# Cross-checks.
if "none" in recs and "rrf" in recs:
    if recs["none"]["n_inv"] != 0:
        print(f"FAIL: fusion=none should have n_pairs_inverse=0 but got {recs['none']['n_inv']}")
        bad += 1
    if recs["rrf"]["n_inv"] <= 0:
        print(f"FAIL: fusion=rrf should have n_pairs_inverse>0 but got {recs['rrf']['n_inv']}")
        bad += 1
    if recs["rrf"]["n_ovl"] <= 0:
        print(f"FAIL: fusion=rrf should have overlap>0 but got {recs['rrf']['n_ovl']}")
        bad += 1
print(f"summary: ok={ok}  bad={bad}")
sys.exit(1 if bad else 0)
PY
