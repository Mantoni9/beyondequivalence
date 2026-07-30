# Beyond Equivalence in E-Commerce

Code, registrations, and result artifacts for the MSc thesis **"Beyond Equivalence
in E-Commerce: Hierarchical Category Mapping using Large Language Models and
Advanced Embeddings"** (University of Mannheim, DWS group, 2026).

The thesis studies directed ontology alignment beyond equivalence, with the
relation set {subclass, superclass, equivalence, part-of, none}, using a
two-stage zero-shot pipeline: instruction-conditioned embedding retrieval
(Stage 1) and LLM relation classification (Stage 2), evaluated on six OAEI
BeyondEquivalence test cases and a newly built industrial gold standard of
2,048 directed correspondences between the VDI 4081 parts vocabulary and the
eBay Germany vehicle-parts taxonomy.

## Headline findings

- Instruction-conditioned retrieval makes directed candidates reachable, but
  carries no usable direction signal itself. Direction resolution is entirely
  Stage 2's burden.
- The direction decision separates model classes categorically. Reasoning-trained
  models orient hierarchical relations at 0.90 to 1.00 direction accuracy where
  instruction-tuned models range from 0.21 to 0.77 and systematically assign the
  subclass role by presentation position. Order-invariant voting (B2) repairs an
  instruct model by +0.087 Macro-F1 at doubled cost without reaching the other
  class.
- Of the 451 curated seed correspondences of the industrial reference, roughly
  nineteen of twenty are directional and about one in twenty is an equivalence.
  On the closure-expanded reference of 2,048 pairs the equivalence share falls
  to about one percent, which makes the point stronger: an equivalence-only
  matcher is blind to nearly the entire mapping.

## Where the results are

| What | Where |
|---|---|
| Curated final metrics, matrix analysis (`a24e146`), audit records, per-arm ablation records | `results-final/` (tracked in this repo) |
| Full per-pair predictions, complete run bundles, large TSVs | [GitHub Release **v1.0-thesis**](https://github.com/Mantoni9/beyondequivalence/releases/tag/v1.0-thesis) (assets) |
| Industrial gold standard (2,048 pairs) and ontologies | `data/gold-standard/` and Release assets |
| Annotator workbooks (anonymized) and adjudication record | `data/gold-standard/annotation/` |
| Pre-registration documents | `docs/` (see index below) |
| Experiment tracking | W&B project `beyondequivalence-retrieval-stage1` (source of truth for raw runs) |

The submitted thesis PDF is attached to Release **v1.0-thesis**.

## Thesis table and figure map

Every reported table is rebuilt by a script from stored run artifacts. A
validation pass over six hundred stored values reproduced the reported numbers
exactly.

| Thesis result | Script | Input artifact |
|---|---|---|
| Stage-1 ablation (templates, LoRA, verbalization) | `scripts/ablation_bidirectional.py` | `results/ablbi_*_d11c97e/metrics.json` |
| Stage-2 model matrix and analysis | `scripts/analyze_matrix.py`, `scripts/matrix_stats.py` | `results/matrix_*_seed42_*/predictions.tsv` |
| Order sensitivity and B2 voting (Stufe A/B) | `scripts/analyze_stufeA.py`, `scripts/analyze_stufeB.py`, `scripts/stage2_bothorder.py` | Stufe run outputs |
| Precision interventions E16/E17/E18 | `scripts/e16_analysis.py`, `scripts/e17_*.py`, `scripts/e18_consistency.py` | intervention run outputs, `results-final/E16_18_evidence.tsv` |
| Subclass-precision audit and correction | `scripts/build_ltfp_audit_sample.py`, `scripts/ltfp_corrected_precision.py` | `results-final/ltfp_audit_adjudicated_2026-07-19.tsv` |
| Hierarchy-credited metric | `scripts/closure_credit.py`, `scripts/closure_analysis.py` | prediction TSVs and target closures |
| Query-swap study | swap run outputs | `results/swap_*_e98c0b3/passes.tsv` |

## Reproducing

1. Clone and install (see `requirements` and the environment templates
   `.env.{bwuni,dws,local}.template`).
2. Benchmark data is not redistributed here. Download the OAEI
   BeyondEquivalence bundle from Zenodo (DOI `10.5281/zenodo.17091043`) and place
   `benchmark.zip` in the repository root. The loader picks it up directly.
3. The industrial gold standard is built by the tracked scripts under
   `goldstandard_ebay/` (`build_ebay_target.py`, `csv_to_owl.py`,
   `vdi/vdi2owl.py`, `make_reference.py`).
4. To rebuild a reported table, run the mapped script on the artifacts from
   `results-final/` or the Release bundle.

Cluster execution used two A6000 GPUs with vLLM serving. Exact model revisions,
quantizations, decoding parameters, and seeds are recorded in the thesis
reproducibility appendix and in the per-run configuration records.

## Pre-registrations

All confirmatory experiments were registered before their runs. See `docs/`:
Stage-2 matrix, Stufe A, Stufe B, few-shot ablation (E15), precision
interventions (E16 with the H3 structural-gate addendum, E17 verification
second pass, E18 coherence filter), and the Stage-1 query-swap registration.

## Thesis-cited revisions

The thesis text cites three code revisions. They are pinned by annotated tags:

| Tag | Revision | Scope |
|---|---|---|
| `thesis-stage1-ablation` | `d11c97e` | Stage-1 ablation layer |
| `thesis-query-swap` | `e98c0b3` | Query-swap study and registration record |
| `thesis-matrix-analysis` | `a24e146` | Stage-2 matrix analysis |

The submitted state is tagged `v1.0-thesis`.

## Data provenance and licenses

- Code: MIT License (see `LICENSE`).
- Own annotations and derived gold-standard relations: CC BY 4.0.
- OAEI BeyondEquivalence benchmark: not redistributed, obtain from Zenodo
  (DOI `10.5281/zenodo.17091043`).
- eBay Germany category taxonomy and VDI 4081 vocabulary: third-party
  material reproduced for research documentation only, all rights with their
  owners. See `THIRD_PARTY_NOTICES.md`.

## Citing

If you use the gold standard or the pipeline, please cite the thesis.
<!-- verify: add BibTeX entry once the library record exists -->