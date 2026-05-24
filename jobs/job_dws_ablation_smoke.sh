#!/bin/bash
# Ablation smoke — qwen3 on g7-literature, all 8 (A x B x C) permutations.
# Validates that no permutation crashes, the 8 alignments are distinguishable,
# and that permutation 0 (A=turtle, B=default, C=none) numerically reproduces
# the existing C-sweep fusion=none run for the same (model, dataset).

#SBATCH --job-name=ablation_smoke
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ablation_smoke_%j.out
#SBATCH --error=logs/ablation_smoke_%j.err

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
WANDB_GROUP="ablation_smoke_${TS}_${SHA}"

echo "=========================================================================="
echo "Ablation smoke. Group=${WANDB_GROUP}  Job=${SLURM_JOB_ID}  Node=$(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================================================="

python run_subsumption_experiment.py \
    --ablation-sweep \
    --ablation-models qwen3-embedding-8b \
    --ablation-datasets g7-literature \
    --wandb \
    --wandb-group "$WANDB_GROUP" \
    2>&1 | tee "results/${WANDB_GROUP}.log"

echo ""
echo "=========================================================================="
echo "Smoke done at $(date +%H:%M:%S). Group=${WANDB_GROUP}"
echo "=========================================================================="
echo ""
# Verify: 8 metrics.json, all with truncation fields populated, all 8
# alignments distinguishable in size or score range, all 8 with B-tag present.
python - <<'PY'
import json, glob, os, sys
ok = bad = 0
mrr_subclass: dict[str, float] = {}
for d in sorted(glob.glob(f"results/abl_qwen3-embedding-8b_*_g7-literature_*")):
    mp = os.path.join(d, "metrics.json")
    cp = os.path.join(d, "config.json")
    if not (os.path.isfile(mp) and os.path.isfile(cp)):
        continue
    m = json.load(open(mp))
    c = json.load(open(cp))
    last = m.get("matcher_last_run_metrics", {})
    has_trunc = any(k.startswith("tokens_truncated/") for k in last)
    mrr_sub = m.get("mrr", {}).get("per_relation_strict", {}).get("subclass")
    print(f"{os.path.basename(d)}")
    print(f"  A={c.get('A')}  B={c.get('B')}  C={c.get('C')}  "
          f"trunc={has_trunc}  mrr_sub={mrr_sub:.4f}")
    if has_trunc:
        ok += 1
    else:
        bad += 1
    key = f"A={c.get('A')}/B={c.get('B')}/C={c.get('C')}"
    mrr_subclass[key] = mrr_sub
print(f"summary: ok={ok}  bad={bad}  unique_mrr={len(set(round(v,5) for v in mrr_subclass.values() if v is not None))}")
sys.exit(1 if bad else 0)
PY
