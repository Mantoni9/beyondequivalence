# Stage-2 model matrix — pre-registration (Tier 1)

Filed and committed BEFORE any matrix run. Probe gate PASSED: all four models
serve, parse_fail 0; per-model max_new_tokens fixed from probed CoT length.

## Central question
Order-invariance (Stufe B, external repair) and reasoning (internal) are
ORTHOGONAL. The matrix asks: do reasoners (gpt-oss, gemma4) resolve direction
INTERNALLY better than non-reasoners — measured SINGLE-ORDER (each model's own
internal resolution, not external Both-Order repair)?

## BINDING SCOPE (Antonio)
- **K=20 EVERYWHERE**, all datasets, no exception (no K=10 anywhere).
- **FULL datasets** — the reranker classifies EVERY candidate pair; no gold-only
  subsets, no candidate pre-filtering/pruning. Maximum coverage.
- **FOUR models, ONE config each, single-order:** Llama-3.3-70B-AWQ (non-reasoning),
  Mistral-Small-3.2-24B (non-reasoning), gpt-oss-120b (reasoning), gemma-4-31B
  (reasoning). No Gemma thinking-split.
- **Decoding:** reasoners model-recommended (gpt-oss temp 1.0/top_p 1.0;
  gemma4 temp 1.0/top_p 0.95), non-reasoners temp 0.0 (Mistral, Llama).
- **max_new_tokens (probed):** gpt-oss 1024, gemma4 256, Mistral 384, Llama 256.
- Prompt d_subs_v2 answer-first, strict parser, parse_fail<5% gate per model.
- Frozen Stage-1 TSVs (Qwen3-noLoRA/path_context/T2, top-20). reranker-conditional
  primary; partOf excluded from primary scoring; end-to-end secondary.

## Serving (Phase-0 probe)
vLLM 0.23 (env vllm-matrix) for the novel archs — gpt-oss mxfp4, gemma4,
mistral3 (TEXT-ONLY: `--limit-mm-per-prompt image=0`, vLLM else crashes on the
pixtral vision path); Llama-AWQ on the proven vLLM 0.19.1 (env melt-olala,
untouched). Server/client split over HTTP; reranker client always melt-olala.
2 GPUs/job, QOS max2gpu5d → jobs serialize.

## Metrics (analyzer prints all; scripts/analyze_matrix.py)
- **PRIMARY: Macro-F1 over {<,>,=}**, reranker-conditional, per dataset + pooled.
- **FULL 4×4 confusion matrix {<,>,=,none}** per model per dataset, with the
  **none-row P/R/F1 as a first-class output** (probe: gpt-oss labelled 47/78 none —
  a reasoner that "fixes direction" while inflating none is a worse outcome that
  flip-rates alone hide).
- Per-class P/R/F1 for EVERY class. flip_rate_gt AND flip_rate_lt per model.
- Direction-accuracy (<↔> off-diagonal) — reported, explicitly NOT primary
  (Stufe A: misleads on '>'-heavy data).
- **Reference floor rows:** random-direction-guess + majority-class, on the same
  conditional pairs (so reasoner numbers read as GOOD vs just less-bad-than-Llama).
- **McNemar** per model-pair on the pinned directional gold sets (g7 67, g5 85,
  g3 541; named-26 g7 a designated subset) + **bootstrap CIs on Macro-F1**
  (small n → point estimates insufficient).
- **'<'-precision 3-cause decomposition** (explicit analysis question): poor '<'-F1
  has three candidate causes one model cannot separate — (a) positional bias,
  (b) gold-incompleteness (real unlabelled subclasses → FP), (c) genuine
  <-vs-none/= confusion. The matrix separates them: SAME poor '<'-precision across
  all 4 models → structural/gold (model-independent); a reasoner improving it
  without gold changing → model-specific. Do NOT call '<'-precision "bias" without
  this decomposition (it was overclaimed on one model in Stufe A).
- **foreign_audit** on a sample of '<'-FPs → the gold-gap component of corrected
  precision.
- parse_fail per model (gate). **Precision-confound column** (quantization per
  model; gpt-oss MXFP4-native, no BF16 reference; Llama AWQ-INT4; gemma4/mistral bf16).
- **Variance:** per-class F1 spread across the 3 g3 gpt-oss seeds = the noise floor
  for reasoner comparisons.

## Tiering (signal-per-cost; all K=20, all full datasets; QOS serialises)
- **Tier 1 (now):** {g7, g5, g3} × 4 models = 12 jobs, single-order, seed 42.
  ~36 h serialised (Llama 7h + Mistral 4.7h + gemma4 9.1h + gpt-oss 15.2h).
- **Variance (now, alongside):** gpt-oss × g3 × seeds {42, 123, 7}. Seed 42 IS the
  Tier-1 gpt-oss-g3 run; +2 extra jobs (seeds 123, 7), FULL g3 (8450 pairs each,
  ~12h). The variance is measured on the REAL run's full distribution — not a
  gold-only subset (which would misrepresent it).
- **Tier 2 (after):** mouse-human × 4, K=20 (full 68,135 pairs). Multi-day, accepted.
- **Tier 3 (optional):** g1 + g2 × 4 (high cost, low directional signal: 53/38
  directional pairs for 19.5k/35k candidates).

Total now = 12 Tier-1 + 2 variance = **14 jobs**.

## Walltime (generous, variance-tolerant; QOS max2gpu5d allows 5 days)
g7: gpt-oss 5h · gemma4 4h · mistral 3h · llama 3h.
g5: gpt-oss 6h · gemma4 5h · mistral 4h · llama 4h.
g3: gpt-oss 28h · gemma4 16h · mistral 10h · llama 12h (variance gpt-oss-g3 28h).

## Submission lines (one model × one dataset)
```bash
for ds_t in "g7-literature:3:4:5" "g5-groceries:4:5:6" "g3-text:12:16:28"; do
  ds=${ds_t%%:*}                              # walltimes encoded per tier in the loop
done
# explicit (Tier 1, seed 42):
MODEL=llama   DATASET=g7-literature sbatch --time=03:00:00 jobs/job_dws_matrix_run.sh
MODEL=mistral DATASET=g7-literature sbatch --time=03:00:00 jobs/job_dws_matrix_run.sh
MODEL=gemma4  DATASET=g7-literature sbatch --time=04:00:00 jobs/job_dws_matrix_run.sh
MODEL=gpt-oss DATASET=g7-literature sbatch --time=05:00:00 jobs/job_dws_matrix_run.sh
MODEL=llama   DATASET=g5-groceries  sbatch --time=04:00:00 jobs/job_dws_matrix_run.sh
MODEL=mistral DATASET=g5-groceries  sbatch --time=04:00:00 jobs/job_dws_matrix_run.sh
MODEL=gemma4  DATASET=g5-groceries  sbatch --time=05:00:00 jobs/job_dws_matrix_run.sh
MODEL=gpt-oss DATASET=g5-groceries  sbatch --time=06:00:00 jobs/job_dws_matrix_run.sh
MODEL=llama   DATASET=g3-text        sbatch --time=12:00:00 jobs/job_dws_matrix_run.sh
MODEL=mistral DATASET=g3-text        sbatch --time=10:00:00 jobs/job_dws_matrix_run.sh
MODEL=gemma4  DATASET=g3-text        sbatch --time=16:00:00 jobs/job_dws_matrix_run.sh
MODEL=gpt-oss DATASET=g3-text SEED=42  sbatch --time=28:00:00 jobs/job_dws_matrix_run.sh
MODEL=gpt-oss DATASET=g3-text SEED=123 sbatch --time=28:00:00 jobs/job_dws_matrix_run.sh
MODEL=gpt-oss DATASET=g3-text SEED=7   sbatch --time=28:00:00 jobs/job_dws_matrix_run.sh
```
Analyze (after results): `scripts/analyze_matrix.py` over the matrix_* dirs +
pinned directional sets + foreign_audit. Checkpoint after results before any
adoption / Tier 2.
