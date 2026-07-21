# E16 — Addendum H3: Structural Coherence Gate (registered before data)

**Registered:** 2026-07-21, before inspecting any prediction, confidence, or
confusion of the cells this arm is applied to. Extends `E16_registration_2026-07-19.md`;
on any conflict the base registration governs. Rules are **theory-driven**
(ontology-alignment repair), not derived from the data. CPU-only, post-hoc on the
existing `predictions.tsv`; no re-inference, no GPU, running jobs untouched.

## Motivation
Diagnosis from the aggregate confusion matrices (already visible, not blinded):
the reranker classifies well *among* {`<`,`>`,`=`} but over-asserts relations on
`gold=none` candidate pairs (semantically similar but not subsumption). F1 is
killed by this FP mass — a **none-gate** failure, not a relation-classifier
failure. H1 gates via confidence (model-dependent, may degenerate at temp=0).
H3 gates via **structure**, model-independent, and operationalizes the advisor's
point that these FPs are resolvable from the tree + inference rules
(alignment-coherence repair; cf. LogMap, AML, Meilicke & Stuckenschmidt).

## Hypothesis
**H3:** Removing structurally-incoherent asserted subsumption edges — using the
source and target ontology hierarchies (target closure from the audit instrument)
— removes ancestor-redundant and implausible assertions and improves strict
precision; recall behaviour is rule-specific and reported explicitly.

## Rules (parameter-free; fixed order; each application logged)
Applied per cell to the kept `<`/`>` assertions (`=` untouched; direction errors
gold∈{>,=} are not the target of H3 and pass through). All against
`closure_credit.build_closure` on the relevant hierarchy.

- **R1 — Transitive reduction (= base-reg H2).** Per source & direction, drop an
  assertion whose target is an ancestor (for `<`) / descendant (for `>`) of
  another kept target for the same source. Keep the most specific. *Construction
  invariant: strict recall EXACTLY unchanged* (only non-gold ancestors of a kept
  assertion are removable when gold is the lowest node) — checked exactly, a
  deviation is a bug, not a number.
- **R2 — Antisymmetry / cycle.** If `S<T` and `S>T` are both asserted (or a 2-cycle
  in the predicted subsumption graph), the pair is incoherent; drop **both**
  (no confidence tie-break — H3 is confidence-independent by design). Logged.
- **R3 — Single-chain fan-out (parameter fixed = 1).** After R1, if a source still
  asserts `<` to targets in **structurally disjoint** target subtrees (no
  ancestor/descendant relation between them), keep only the chain of the
  **highest Stage-1-score** target; drop the others. Rationale: a concept is a
  subclass of one target region, not several disjoint ones. This is the only rule
  that can remove a true positive → recall MAY fall; that is a genuine trade-off,
  reported (strict + credited), never suppressed. The tie-break (Stage-1 score) is
  fixed in advance, not tuned.

`H3 = R1∘R2∘R3` in that order. Also reported: **H2 alone (=R1)** for continuity
with the base registration, and the ablation R1, R1+R2, R1+R2+R3.

## Application scope
All cells, ALL datasets (no dev/test split — H3 has no tuned parameter):
4 models × {g1,g2,g3,g5,g7,mouse-human} + the four vdi-ebay gold cells.
Head-to-head vs H1 uses the base-reg common test set (excl. the H1 dev pool g1+g2);
H3 is additionally reported on g1+g2.

## Success criterion (adoption bar for main text)
Pooled held-out **credited** F1 improves by **≥ +0.02 absolute** for a model,
with R1's strict-recall invariant passing exactly. Strict deltas reported
alongside but **credited is the headline** — because ~38 % of strict `<`-FPs are
gold gaps (ltFP audit, ĝ=0.375), so a strict-precision gain from pruning partly
reflects discarding correct-but-unlabelled edges; only the credited delta is an
honest quality gain. Below bar → honest negative/marginal paragraph.

## Combined arm
`H1 then H3` reported descriptively only (no own criterion); contributions of
confidence-gating vs structural-gating kept attributable by also reporting each
alone. No further stacking.

## Outputs (into the E16 output directory, commit-hash in the name)
Extends the base-reg deliverables: `e16_results_frozen.tsv` gains arms
`{H2=R1, H3, R1+R2, combined}`; `e16_h3_repair_log.tsv` (per removed edge:
cell, source, target, rule, stage1_score, was_gold_gap_per_audit_if_known);
`e16_summary.md` gains one line per model for H3 (recall-invariant pass y/n →
credited ΔF1 → bar met y/n).

## Discipline
No peeking to design or tune: rules + tie-breaks + order are fixed here, before
touching the data. Same credited-first yardstick and exact construction checks as
the base registration. Timebox unchanged (analysis ≤ 2026-07-21 EOD).
