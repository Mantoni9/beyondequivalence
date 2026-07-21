# E17 — Registration: Verification Second-Pass (targeted FP rejection)

**Registered:** 2026-07-21, before any verification inference. Eligible per the
E16 contingency branch (H1 abstention below bar for **both** reasoning models:
gpt-oss +0.000, gemma −0.009 credited). This is the single registered follow-up;
no further stacked interventions. GPU inference is new but confined to
already-asserted pairs (not the candidate universe).

## Motivation (from E16)
E16 established: the models classify {`<`,`>`,`=`} well but massively
over-assert on `gold=none` pairs (dev FP:TP ≈ 17–56×), and this FP mass — not the
relation classifier — depresses F1. **Pruning does not fix it for the precise
models:** confidence abstention (H1) is below bar for all four; structural gating
(H3) helps only the over-asserting weak model (mistral) and the vdi-ebay case,
but *hurts* the reasoners by discarding correct multi-parent mappings. A robust
gain therefore requires **adding a rejection signal**, not removing edges: a
second, skeptical judgment over the asserted pairs that can reject genuine FPs
the forced-choice first pass wrongly asserted.

## Hypothesis
**H (verification):** A dedicated binary verification pass over each asserted
`(S, rel, T)` — framed skeptically, with both entities' hierarchy context, and
"no relation" primed as a common valid answer — rejects genuine FPs at a higher
rate than it rejects true positives, improving held-out **credited** macro-F1,
including for the reasoning models where E16's pruning failed.

## Scope
- **Input:** the kept `<`/`>`/`=` assertions of the final matrix cells (the E16
  assertion set). Verification is applied **only to asserted pairs** → cost scales
  with assertions, not with the candidate universe.
- **Cells:** 4 models × {g3-text, g5-groceries, g7-literature, mouse-human}
  (common test) + the four vdi-ebay gold cells (second arm). g1+g2 = the
  viability-check / τ-tuning pool (excluded from headline, as in E16).

## Verification prompt (skeptical binary; exact template in code)
Per asserted `(S, rel, T)`: present S and T each with their `description_path_context`
(label + ancestor path), state the asserted relation, and ask whether that exact
relation **actually holds** or whether the two are **not in that relation**.
"NO / not related" is explicitly stated as a common, valid answer. Answer strictly
YES/NO; confidence = logprob P(yes)/(P(yes)+P(no)) via `get_confidence_first_token`.
Decoding matches each model's matrix cell (temp/top-p). This is a **different
prompt** from the first pass (verification, not forced choice), registered fixed.

## Decision rule (parameter-free primary)
Keep the assertion iff the verifier answers YES (argmax P(yes) > P(no)); else
relabel to `none`. Verification can only **remove** assertions ⇒ recall is
non-increasing, precision non-decreasing; the risk is rejecting true positives,
measured directly.

## Arms
- **V1 — self-verification (primary):** each model verifies its own assertions.
  The clean "does a second pass fix over-assertion?" claim; attributable per model.
- **V2 — strong-judge (secondary):** gpt-oss verifies all four models' assertions
  (single strong universal gate). Reported separately; not stacked with V1.
- **V3 — thresholded (optional refinement):** keep iff P(yes) ≥ τ_v, with τ_v
  tuned on the g1+g2 dev pool (argmax over distinct dev values of dev credited
  macro-F1, tie-break higher τ_v) and applied frozen. Only reported if V1's
  argmax gate is promising but sub-optimal.

## Viability checkpoint (cheap, BEFORE the full test run)
Run V1 verification on the **g1+g2 dev pool only** first. Report, per model:
verifier YES-rate overall and split by TP vs FP (against gold-gap-aware credited
labels). **Proceed to the full test run only if** the verifier is non-degenerate
(YES-rate ∈ (5 %, 95 %)) **and** shows separation (FP YES-rate materially below
TP YES-rate — pre-stated: TP-YES minus FP-YES ≥ 0.10). A rubber-stamp verifier
(no separation) → negative result, one paragraph, no full run (saves GPU).

## Degeneracy / abort (pre-stated)
Verifier YES-rate > 95 % (rubber-stamp) or < 5 % (rejects everything) on dev →
degenerate, negative result for that model/arm, no test run.

## Success criteria (adoption bar, per model × arm)
- **Primary:** pooled held-out **credited** macro-F1 improves **≥ +0.03 absolute**,
  **and** no held-out test case loses more than **0.05** credited macro-F1.
- **Secondary (descriptive, no bar):** strict deltas; per-relation P/R; TP-retention
  vs FP-rejection rates; vdi-ebay arm; V2 vs V1 comparison.
- Below bar → honest negative/marginal paragraph; E16 findings stand unchanged.
- **Gold-gap guard:** credited is the headline (strict gains from rejecting
  gold-gap FPs are not real quality gains); an FP-rejection log records which
  rejected pairs are audit-consistent gold gaps.

## Metrics
Strict + **credited** macro-F1 over {`<`,`>`,`=`} on the e2e universe
(candidate ∪ gold), identical machinery to E16 (`closure_credit`), so numbers are
directly comparable to the E16 baseline/H1/H3 columns.

## Execution / cost
New GPU inference on DWS (4-GPU quota), served like the matrix campaign
(`--disable-custom-all-reduce` on A6000, 64/64 concurrency, sharding for large
cells), reusing the frozen serving configs. Only asserted pairs are scored →
well below a full re-inference. Verification runner is additive (new prompt id +
a verify-mode over an assertion list); does not modify Stage-1/Stage-2 or the
matrix cells.

## Timebox
Success-gated, not clock-gated (10 days, no competing jobs). Soft milestones:
dev viability check first; full test only if it passes. **Adopted into the thesis
only on success** (credited bar met for ≥1 model×arm); otherwise the topic stays
Future Work and E16's honest findings are the final word. No hard kill date — but
a degenerate/below-bar viability check ends it cheaply.

## Outputs (directory with commit hash)
`e17_verification_dev_viability.tsv` (model, arm, YES-rate TP/FP, separation,
degenerate y/n) · `e17_results.tsv` (model × testcase × arm × {strict,credited}
P/R/F1 + Δ vs baseline) · `e17_rejection_log.tsv` (rejected pair, was_TP,
was_gold_gap) · `e17_summary.md` (per model×arm: viability → credited Δ → bar y/n).
