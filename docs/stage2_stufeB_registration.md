# Stage-2 Stufe B — pre-registration: order-invariant directional classification (Both-Order-Voting)

Filed and committed BEFORE any Stufe-B run. Builds on the Stufe-A verdict
(H-position, clean): Llama's subclass-prior is POSITIONAL — it labels the
first-presented concept as the subclass, content-independent. Argument-swap
(A2) does NOT fix this — it redirects the bias (Macro-F1 barely moved
0.334→0.373; direction-accuracy rose only because dev is '>'-heavy). The
principled remedy is ORDER-INVARIANCE: query each pair in both argument orders
and reconcile, so the positional bias cancels instead of flipping sign.

## Literature grounding (Methods section)
- **Berglund et al. 2023 (Reversal Curse):** models trained on "A is B" fail
  to infer "B is A". The Stufe-A positional prior is exactly this; Berglund
  names the phenomenon but proposes no fix. Both-Order-Voting is our
  contribution to that open problem.
- **Yuan & Vlachos 2025 (symmetry):** a relation is symmetric iff
  r(x,y)⟹r(y,x), antisymmetric iff r(x,y)⟹¬r(y,x). This dictates the B3
  disagreement mapping (below).
- **Ensemble voting:** reconciling inconsistent single-model labels by voting
  over input perturbations is established practice; the argument-order
  perturbation is the natural one for a positional bias.

## Scope
feat/stage2-relation-classifier. Frozen Stage-1 TSVs (no swap). Llama-3.3-70B-AWQ
only (reasoners are a later matrix question, not Stufe B). Decoding pinned to
Run 255471 verbatim (temperature 0.0 greedy, max_new_tokens 256, seed 42,
concurrency 16); B2 additionally consumes answer-span logprobs (already greedy,
no decoding change). Dev = {g7-literature, g5-groceries}, reranker-conditional.
partOf is EXCLUDED from primary scoring (03.06 protocol) — folded to 'none'.

## Inference — two orders per pair, reconcile offline
For each deduped candidate (s, t): query AB = present (s, t) and BA = present
(t, s); invert BA's label to canonical via the A2 path
(MatcherSubsumptionReranker.relation_for_canonical_pair, tested exactly-once).
Persist both raw labels, both canonical predictions, both answer-span logprobs.
All three variants are OFFLINE recombinations of this ONE double-order run
(~2× baseline ≈ 90 min g7 / 125 min g5). Runner: run_stage2_bothorder.py.

## Derivation of the truth table (verified, not assumed)
A purely positional model says "first-presented = subclass". On a true X⊏Y:
- AB=(X,Y): X first → raw 'subclass' → canonical '<'.
- BA=(Y,X): Y first → raw 'subclass' AGAIN → canonical '>' (A2 inversion).
So a positional model yields canonical (<,>) DISAGREEMENT by giving the SAME
raw label both times. A content-tracking model flips its raw label with the
swap → canonical (<,<) AGREEMENT. **Empirically confirmed on the Stufe-A
named-26: 24/26 are raw (subclass, subclass) → canonical (<,>); 2/26 are
(subclass, equivalent) → canonical (<,=).**

Therefore:
- canonical AGREEMENT on a direction = the model tracked content → trust it.
- canonical directional DISAGREEMENT = same raw label both ways = X⊑Y ∧ X⊒Y =
  mutual subsumption = the equivalence signature (and, for a biased model on a
  directional pair, indistinguishable from it — B3 is honest about that).

## Reconciliation truth tables (canonical pred_AB, pred_BA ∈ {<,>,=,none})

| Row | (pred_AB, pred_BA) | B1 abstain | B3 symmetry | B2 confidence |
| --- | --- | --- | --- | --- |
| agreement-direction | (<,<) / (>,>) | that direction | that direction | that direction |
| agreement-equivalence | (=,=) | = | = | = |
| agreement-none | (none,none) | none | none | none |
| **directional conflict** | (<,>) / (>,<) | **none** | **=** | **tie-break\*** |
| **direction-meets-equivalence** | (<,=)/(>,=)/(=,<)/(=,>) | **none** | **none** | **tie-break\*** |
| directional-vs-none | (dir,none)/(none,dir) | none | none | tie-break\* |
| equivalence-vs-none | (=,none)/(none,=) | none | none | tie-break\* |

\* B2 tie-break: on any disagreement the frame with the higher answer-span
mean-logprob wins (emit that frame's canonical label); exact ties → AB.

- **B1 and B3 differ ONLY in the directional-conflict row** (none vs '=').
- The **direction-meets-equivalence row is explicit and registered**: it is
  neither a clean agreement nor a clean directional disagreement → none
  (B1/B3) / tie-break (B2). The named-26's 2 outliers (canonical (<,=)) live
  here and have a documented home.
- Consequence, acknowledged and intended: under B3 the named-26 map to '='
  (NOT the 24/26→'>' that A2 gave). A purely positional verdict on a
  directional pair is formally indistinguishable from equivalence; B3 is
  honest about that. B3 may therefore score WORSE on Macro-F1 than B1/B2
  (the '=' contamination hits both =-precision and >-recall). B3 is the
  theoretically-grounded variant, not necessarily the empirical winner. If
  B1 or B2 wins, that is a legitimate, reportable result — the contrast
  conflict→none (B1) vs conflict→'=' (B3) is itself a finding.

## Answer-span logprob (B2) + manual gate
`LLMOpenAI.get_text_completion_with_logprobs` now returns per-token strings
(additive). Answer span under answer-first = the first response line (the
'Relation: <label>' line): tokens up to and including the first token
containing '\n', mean of their logprobs (stage2_bothorder.answer_span_logprob).
**MANUAL GATE (255391 lesson — never trust the span parser blind):** before
B2 scoring is trusted, the run's token_dump.tsv (first 12 pairs' raw AB/BA
responses) must be manually inspected to confirm the first-line span actually
captures the label. This is an explicit gate, not a post-hoc check.

## Metrics — PRIMARY = Macro-F1 (NOT direction-accuracy)
Stufe A proved direction-accuracy misleads on '>'-heavy dev (A2 hit 0.835 acc
while Macro-F1 barely moved). Registered:
- PRIMARY: Macro-F1 over {<,>,=}, reranker-conditional, dev-pooled + per ds.
- flip_rate_gt AND flip_rate_lt (a real fix lowers BOTH; a redirect trades one).
- named-26 full destination distribution per variant (→<,>,=,none).
- disagreement rate between orders (a bias-magnitude measure); B1 abstain rate.
- =-F1 guard ≥ 0.70 (A1 lesson). B3 touches '=' directly — watch especially.
- parse_fail < 5% per order.

Bands vs the v2 baseline dev-pooled Macro-F1 **0.334** (Stufe-A 0c anchor):
**SOLID ≥ +0.10 · SMALL +0.03–0.10 · NO < 0.03 · REVERSE worse ≥ 0.03 OR
=-F1 < 0.70.** Precedence among guard-passing (SOLID/SMALL) variants: highest
Macro-F1; tie (Δ < 0.01) → higher recall (favors B2/B3 over B1's abstaining).

## '<'-heavy control slice (Antonio-approved, READ-ONLY)
docs/stufeB_guard_slice_mousehuman.tsv: 45 '<' gold pairs from mouse-human
(conditional on frozen d11c97e top-20, seed 42; pool 545 gold-'<', 495
conditional). docs/stufeB_guard_candidates_mousehuman.tsv is the minimal
Stage-1-shaped input so the guard run queries ONLY these 45 pairs. The
winning arm's <,>,= F1 on this slice is reported as a STANDALONE guard
readout — it does NOT enter arm selection or tuning (mouse-human is a test
set; disclosed guard sample). Rationale: dev is '>'-heavy, so a residual
'>'-lean would look fine on dev Macro-F1 but harm '<'-heavy data.

## Integrity check
The AB-order canonical predictions (no reconciliation) must equal the v2
single-order baseline (Run 255471) predicted_relation on shared kept pairs —
a free reproduction check that the double-order AB pass matches the baseline.
analyze_stufeB reports it (0 mismatches expected).

## Submission lines (Antonio submits; ~90 min g7 / ~125 min g5; guard ~5 min)
```bash
DATASET=g7-literature \
  STAGE1_PREDICTIONS=results/stage1_frozen/g7-literature_qwen3-noLoRA_pathctx_T2_top20.tsv \
  sbatch jobs/job_dws_stage2_bothorder.sh
DATASET=g5-groceries \
  STAGE1_PREDICTIONS=results/stage1_frozen/g5-groceries_qwen3-noLoRA_pathctx_T2_top20.tsv \
  sbatch jobs/job_dws_stage2_bothorder.sh
# Guard (only the 45 '<'-heavy pairs; run any time, read after the winner is known):
DATASET=mouse-human \
  STAGE1_PREDICTIONS=docs/stufeB_guard_candidates_mousehuman.tsv \
  sbatch jobs/job_dws_stage2_bothorder.sh
```
Analyze:
```bash
conda run -n melt-olala python scripts/analyze_stufeB.py \
  --bothorder g7-literature=<dir> g5-groceries=<dir> \
  --baseline g7-literature=<255471dir> g5-groceries=<R0dir> \
  --guard-bothorder mouse-human=<guarddir>
```

Checkpoints: after Phase 0 (done — baseline Macro-F1 0.334, guard slice
pinned), and after B1/B2/B3 results (Macro-F1 verdict + guard readout)
BEFORE adopting any arm.
