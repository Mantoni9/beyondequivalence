# Stage-2 Stufe A — pre-registration: bias attribution (label order vs argument position vs content prior)

Filed and committed BEFORE any Stufe-A run was submitted. The research
question is causal, not descriptive: WHERE does Llama's subclass-prior come
from? Each hypothesis implies a different downstream fix:

- **H-label** — the prior tracks LABEL-LIST POSITION (first-listed label wins).
- **H-position** — the prior tracks ARGUMENT POSITION (first-mentioned concept
  is read as the subclass).
- **H-content** — a genuine semantic prior, independent of presentation.

Stufe A deconfounds the three by manipulating each factor independently and
reading pre-registered outcome signatures — attribution by manipulation, not
post-hoc storytelling.

## Protocol

Dev/test split per THESIS_NOTES.md (registered 2026-06-12, commit `2d5313c`):
dev = {g7-literature, g5-groceries}, test untouched. All metrics
reranker-conditional. Model: Llama-3.3-70B-AWQ only, answer-first /
non-reasoning (binding: no CoT arms — reasoning attribution comes later
within-model via Gemma thinking-toggle / gpt-oss levels). No few-shot (dead
since v3). No content/wording changes beyond the registered order flip.

Decoding pinned to Run 255471 exactly (verified from its log config dump):
temperature 0.0 (greedy, hardcoded in LLMOpenAI), max_new_tokens 256,
seed 42, batch_size 8, llm_max_concurrency 16, description_path_context,
turtle, stage1_top_k 20, threshold 0.0; same job script
(`jobs/job_dws_stage2.sh`).

## Arms

- **R0** — `d_subs_v2` on g5 (standalone dev baseline; g7 baseline = Run
  255471). No bridge semantics: the query-swap was NOT adopted (job 262057),
  the frozen d11c97e TSVs stay.
- **A1** — `d_subs_v4b` on g7 + g5: label-ORDER flip. v2-byte-identical
  except the two (and only) label-order occurrences flip to
  superclass-first. **Padding control (GO addition #1):** v2's directional
  definitions carry 4-vs-2 trailing spaces (column alignment) — a latent
  tokenizer-level asymmetry. v4b equalizes both to two spaces so A1 tests
  label order with padding held constant. Offline sanity (test-pinned):
  v2-with-equalized-padding differs from v2 in exactly two space characters
  and nothing else. There is NO v4a: "v2 wording, subclass-first, zero-shot"
  is definitionally d_subs_v2 (Phase-0 audit) — 255471/R0 are the
  subclass-first arm.
- **A2** — `--swap-pair-presentation` on g7 + g5: prompt text byte-identical
  to v2; the harness fills the slots with (target, source); directional
  labels invert EXACTLY ONCE at parse time when mapping back to canonical
  (s, t) ('<'→'>', '>'→'<', '='/'none'/'partof'/parse_fail unchanged).
  **Verbalization-identity guard (GO addition #2):** verbalization is a
  function of (kg, concept) only — the same concept produces the same
  path_context string regardless of slot; pinned by
  `tests/test_stufeA_arms.py::test_verbalization_identity_across_slots`.
  The exactly-once inversion is pinned by
  `test_swap_inverts_directional_label_exactly_once_full_path`.
- **A3** — both flips combined (`d_subs_v4b` + swap). CONDITIONAL: prepared,
  submitted only if the registered trigger fires.

## Pre-registered outcomes

PRIMARY: flip_rate_gt (gold-'>' → predicted-'<' over directional
predictions), reranker-conditional, per dataset + dev-pooled.
SECONDARY: flip_rate_lt, Macro-F1, =-F1, label histogram, '<'-predictions on
the no-gold bucket (precision proxy; g7 baseline 324), parse_fail
(gate < 5%), and the named g7 flip-set (the 26 pairs of 255471) tracked per
arm as resolved / persisted / other.

Effect bands per arm (Δ flip_rate_gt vs same-dataset v2 baseline; positive =
improvement): **SOLID ≥ 0.15 · SMALL 0.05–0.15 · NO < 0.05 · REVERSE =
worse by ≥ 0.05.** Guard: dev-pooled =-F1 < 0.70 ⇒ the arm is REVERSE
regardless of flip rate (the v3 lesson: an "anti-bias" intervention that
collapses equivalence is worse than the bias).

Attribution signatures:
- **H-label:** A1 SOLID/SMALL ∧ A2 ≈ NO.
- **H-position:** A2 mirror signature — flip_rate_gt drops ≥ 0.30 AND
  flip_rate_lt rises ≥ 0.30 (the prior follows presentation).
- **H-content:** A1 NO ∧ A2 NO (errors stick to the same pairs regardless of
  presentation) → content levers / decomposition / reasoners next.

A3 trigger: A1 AND A2 both show ≥ 0.05 improvement (ambiguous attribution).
Consistency rule: per-dataset effects must agree in direction across g7 and
g5; a g7-only effect is flagged as a possible g7-tuning artifact.

Analyzer: `scripts/analyze_stufeA.py` (recomputes all registered metrics via
`evaluation_multiclass.compute_multiclass_metrics`, cross-validates against
each run's stored metrics.json, prints bands + signatures + verdict).

## Submission lines (Antonio submits; ~44 min g7 / ~62 min g5 per arm)

```bash
# R0 — v2 baseline on g5
DATASET=g5-groceries STAGE1_PREDICTIONS=results/stage1_frozen/g5-groceries_qwen3-noLoRA_pathctx_T2_top20.tsv \
  sbatch jobs/job_dws_stage2.sh
# A1 — v4b on g7 and g5
DATASET=g7-literature PROMPT_ID=d_subs_v4b sbatch jobs/job_dws_stage2.sh
DATASET=g5-groceries PROMPT_ID=d_subs_v4b STAGE1_PREDICTIONS=results/stage1_frozen/g5-groceries_qwen3-noLoRA_pathctx_T2_top20.tsv \
  sbatch jobs/job_dws_stage2.sh
# A2 — v2 + swapped presentation on g7 and g5
DATASET=g7-literature SWAP_PAIR_PRESENTATION=1 sbatch jobs/job_dws_stage2.sh
DATASET=g5-groceries SWAP_PAIR_PRESENTATION=1 STAGE1_PREDICTIONS=results/stage1_frozen/g5-groceries_qwen3-noLoRA_pathctx_T2_top20.tsv \
  sbatch jobs/job_dws_stage2.sh
# A3 — ONLY if the registered trigger fires (checkpoint 2 first):
# DATASET=... PROMPT_ID=d_subs_v4b SWAP_PAIR_PRESENTATION=1 sbatch jobs/job_dws_stage2.sh
```
