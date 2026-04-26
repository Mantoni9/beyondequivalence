# Reference Relation Distribution in OAEI 2025 BeyondEquivalence

Empirical reference-relation counts across the BeyondEquivalence sub-datasets,
verified on 2026-04-26 by parsing each `reference.rdf` via `Alignment(...)` and
counting raw relation strings.

## Counts

| Dataset | total | `=` | `>` | `<` | PartOf | other | non-ASCII? |
|---|---:|---:|---:|---:|---:|---|---|
| g1-web         |    339 |   275 |    26 |    29 |   2 | `Related: 4`, `HasA: 3` | no |
| g2-diseases    |    355 |   316 |    11 |    27 |   0 | `HasA: 1` | no |
| g3-text        |    762 |    70 |   267 |   425 |   0 | — | no |
| g5-groceries   |    169 |    29 |   113 |    14 |   0 | `HasA: 2`, `Related: 11` | no |
| g7-literature  |     83 |    12 |    52 |    18 |   1 | — | no |
| gpc-unspsc     | 19 615 |   255 | 1 372 |   250 |   0 | `~: 17 738` | no |
| etim-eclass    |  3 678 | 2 371 |   380 |   636 |   0 | `~: 291` | no |
| eclass-gpc     | 12 895 |   251 |   431 | 1 199 |   0 | `~: 11 014` | no |
| mouse-human    |  1 833 |   676 |   612 |   545 |   0 | — | no |

### Double-annotated (source, target) pairs

Verified on 2026-04-26 across all five STROMA/TaSeR sub-datasets: **zero**
`(source, target)` pairs appear with more than one distinct relation. The
`gold_relation` column in `evaluation_recall.compute_recall_at_k`'s per-source
table is therefore always a single relation symbol on STROMA/TaSeR; the
defensive `"</>"`-style joined value is reachable in the code but not by data
on the Stage-1 datasets. Re-verify on g3-text and on Stage-3 product-classification
datasets when those become active.

Reproducer: see the inline ad-hoc script in the chat history; can be re-run via
`tracks.zenodo_loader.load_subdataset(name)` + `Alignment(reference_path)` +
`collections.Counter(c.relation for c in ref)`.

## Implications for the Stage 1 evaluation

### ASCII-only encoding
Every dataset uses ASCII relation strings (`=`, `<`, `>`, plus a few non-standard
identifiers below). The Unicode entries in
`evaluation_recall.RELATION_NORMALIZATION` (`≡`, `≥`, `≤`, `⊑`, `⊒`) are
defensive against future Reference variants and currently never fire in
practice.

### PartOf — in scope, but excluded for Stage 1 by data sparsity
PartOf is a hierarchical (meronymy) relation and **part of the master thesis
research question per the proposal** ("subsumption … or meronymy (part-of)").
Methodically it belongs alongside ⊑ and ⊒ in the asymmetric retrieval study.

The Stage 1 exclusion is not methodology-driven — it is data-driven. Across all
five STROMA/TaSeR sub-datasets there are only **3 PartOf reference mappings**
(n=1 in g7-literature, n=2 in g1-web, zero elsewhere), which is below the
threshold for a meaningful per-relation Recall@K estimator. PartOf is therefore
dropped at evaluation time and counted in `dropped_relations_breakdown`.

When a Reference with sufficient PartOf coverage becomes available (e.g. larger
or richer-annotated cross-ontology benchmarks in Stage 3 of the thesis), the
following two changes re-enable PartOf evaluation:

1. Add `"PartOf": "p"` (or similar ASCII canonical form) to
   `RELATION_NORMALIZATION` in `evaluation_recall.py`.
2. Add the matching label `"p": "partof"` to `RELATION_LABELS`.

The downstream `compute_recall_at_k()` machinery is already general over the
relation labels and needs no further change.

### `~` (Overlap / ≃) — unresolved Stage 3 question
The product-classification reference files use `~` as the dominant relation
(85–90 % of all mappings on gpc-unspsc and eclass-gpc; 7.9 % on etim-eclass).
This is OAEI's ASCII shorthand for ≃ Overlap / approximate match — exactly the
relation the Stage 1 design explicitly excludes alongside ⊘ Disjoint.

Currently `~` falls into the same drop-and-warn path as everything else not in
`RELATION_NORMALIZATION`, which is correct behaviour for Stage 1 (we never run
on these datasets in Stage 1). However, **before the first product-classification
run in Stage 3**, one of the following needs to be decided:

- **Option A — fourth recall breakdown.** Add `"~": "~"` to the normalisation
  table and `"~": "overlap"` to the labels. Recall@K and MRR get a fourth column
  for `overlap`. Pro: covers the dominant relation type in those datasets. Con:
  conflates a near-equivalence relation with the subsumption hypothesis under
  test; may need its own retrieval-direction setup.
- **Option B — restrict reporting to {=, <, >} subset.** Document explicitly in
  the Stage 3 results that recall is computed only over the `{=,<,>}` minority
  (~10 % of mappings on gpc-unspsc / eclass-gpc) and that the Overlap majority
  is excluded by design. The dropped-relations counter makes this transparent.
- **Option C — separate Overlap as its own retrieval task.** Evaluate Overlap
  with a dedicated symmetric retrieval pass (different instructions), report
  it as a parallel result independent of the subsumption study.

This is a Stage 3 decision; for Stage 1 the current behaviour (drop with
warning) is correct.

### HasA / Related — same rationale as PartOf
Single-digit counts on g1-web, g2-diseases, g5-groceries. Same data-sparsity
treatment as PartOf — dropped, logged, ignorable for Stage 1.
