# E18 — Registration: Logical Consistency / Coherence Filter (post-hoc, CPU-only)

**Registered:** 2026-07-21, before running the filter. Theory-driven, parameter-free
rules (defined from the subsumption algebra + the ontology hierarchies, NOT tuned on
results) — register-before-data is satisfied by construction. Extends the E16/H3
structural line (transitive reduction) toward *coherence/conservativity* repair
(ALCOMO, Meilicke 2011 Uni Mannheim; LogMap-C, Solimando/Jiménez-Ruiz/Guerrini
ISWC 2014 / KAIS 2017). CPU-only, post-hoc on existing predictions.tsv; no GPU, no
re-inference; the matrix cells are untouched.

## Motivation
A predicted subsumption edge, read as a bridging axiom over the two class
hierarchies, can force a contradiction or a *novel* (non-conservative) entailment.
Deleting the minimal, lowest-confidence offender removes false positives that are
*provably* inconsistent — a model-independent, near-recall-safe precision gain
(works for every model incl. gpt-oss with no LLM call). Convergent #1 lever across
the E17 literature review.

## Setup
Per source s, collect its asserted `(t, rel, conf)`; use the TARGET-ontology
reflexive closure (`closure_credit.build_closure`): `anc(t)` = t + superclasses,
`desc(t)` = t + subclasses. Relations read as `⊑` (s<t ⟹ s⊑t), `⊒` (s>t ⟹ s⊒t),
`≡` (s=t ⟹ both). `=` edges never removed by L1/L3; `partof`/none out of scope.

## Rules (fixed order; each removal logged; confidence = reranker `confidence`)
- **L1 — direction contradiction (HARD).** If s has `<t1` and `>t2` with
  **t2 ∈ anc(t1)** (t2 at/above t1): s⊑t1⊑t2 and s⊒t2 forces t1≡t2, contradicting
  t1⊏t2 — inconsistent. Remove the **lower-confidence** of the two edges.
  (s<t1 & s>t2 with t2 strictly *below* t1 is consistent — s sits between them — and
  is kept.)
- **L2 — multiple equivalence (HARD).** If s has `=t1` and `=t2`, t1≠t2 and t1,t2 not
  already equivalent in the target: forces the novel entailment t1≡t2
  (conservativity violation). Remove the **lower-confidence** `=`.
- **L3 — ancestor-safe sibling exclusion (HEURISTIC, reported separately).** Let
  A(s) = s's highest-confidence accepted target (an `=`, else the top `<`). Remove a
  `<t` when **t is in a structurally DISJOINT branch from A(s)** (t ∉ anc(A(s)) ∪
  desc(A(s))) — a lexically-plausible but structurally-orphaned edge. **Never**
  remove t ∈ anc(A(s)): those ancestors are the true multi-level subsumptions behind
  our 38 % gold gaps (ancestor-safe guard, from BERTSubs/OntoLAMA negative design).
- Also reported: **R1** (E16 transitive reduction) alone, and **L1+L2+R1** (the full
  hard-consistency filter), and **L1+L2+L3+R1** (hard + heuristic).

## Arms
`baseline` · `L1` · `L2` · `L1L2` (hard consistency) · `L1L2R1` · `L1L2L3R1` (full).
Applied to all 24 matrix cells + 4 vdi-ebay gold cells.

## Metrics & success
Strict + **credited** macro-F1 over {<,>,=} on the e2e universe (candidate ∪ gold),
identical machinery to E16 (`closure_credit`), directly comparable. **Credited is the
headline** (gold-gap guard). L1/L2 remove provably-contradictory edges but may drop a
gold edge if it is the lower-confidence side of a conflict → strict recall may move;
reported, not assumed invariant. A rejection log records which removed edges were
gold (`was_gold`).
- **Adoption bar:** pooled held-out credited macro-F1 improves **≥ +0.02** for a
  model with no test case losing >0.05 credited. Below bar → honest negative.

## Outputs (results/e18/)
`e18_results.tsv` (model × dataset × arm × {strict,credited} macro-F1 + Δ),
`e18_removal_log.tsv` (edge, rule, conf, was_gold), `e18_summary.md`.
