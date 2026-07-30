# Third-Party Notices & Data Licensing

This repository combines original code + annotations (see `LICENSE`) with
third-party material that remains under its owners' terms. This file documents
provenance and licensing for all non-original data. **Read this before making
the repository public or redistributing any data asset.**

---

## 1. Own annotations — CC BY 4.0

The VDI→eBay subsumption gold standard produced for this thesis is released
under **Creative Commons Attribution 4.0 International (CC BY 4.0)**:

- `data/gold-standard/reference_full.rdf` (2048 correspondences, incl. transitive closure)
- `data/gold-standard/reference_seed.rdf` (451 direct correspondences)
- `data/gold-standard/subsumption_gold/gold_relations_karosserie.tsv`
- `data/gold-standard/subsumption_gold/gold_subsumption_findings.md`

Attribution: Antonio Markic, *Beyond Equivalence* (Master's thesis, University of
Mannheim, 2026). These files encode the **mapping/annotation** (the subsumption
relations), which is the original intellectual contribution.

## 2. eBay category taxonomy — third-party (eBay Inc.)

`data/gold-standard/ebay_kfz_target.owl` and the `ebay.de/kfz#` URIs are derived
from eBay's motor-vehicle (KFZ) category hierarchy. eBay category names and the
taxonomy structure are © eBay Inc. and are included here **only** as the target
side of the alignment, for academic reproducibility. This is **not** an official
eBay dataset and carries no endorsement. Redistribution beyond academic use may
require eBay's permission — evaluate before any commercial or bulk redistribution.

## 3. VDI 4081 — third-party (Verein Deutscher Ingenieure)

`data/gold-standard/vdi_karosserie_source_pos.owl` and the `vdi.de/kfz#` URIs are
derived from the vehicle-body classification of **VDI 4081** (Verein Deutscher
Ingenieure e.V.). The VDI standard is copyrighted by VDI; only the class
labels/structure needed as the source side of the alignment are represented here,
for academic reproducibility. Consult the original VDI 4081 for authoritative use.

## 4. OAEI 2025 BeyondEquivalence benchmarks — NOT redistributed

The OAEI evaluation datasets (g1–g7, mouse–human, etc.) are **not** included in
this repository. They are obtained at run time from the official source:

- **Zenodo DOI: 10.5281/zenodo.17091043** (`benchmark.zip`)
- Loader: `tracks/zenodo_loader.py` (expects `benchmark.zip` at the project root
  or `$ZENODO_BENCHMARK_ZIP`).

Their licensing is governed by the OAEI / the respective dataset providers; see
the Zenodo record.

## 5. Models & libraries

Embedding/LLM weights (Qwen3-Embedding-8B, llama-embed-nemotron-8b, Llama-3.3-70B,
Mistral-Small, Gemma, gpt-oss, etc.) are downloaded from their providers under
their own licenses and are **not** included here. Python dependencies retain their
upstream licenses (see `environment.yml`).

---

**Summary for the public-release decision:** §1 is freely shareable (CC BY 4.0);
§4–§5 are already excluded (download-at-runtime). §2 (eBay) and §3 (VDI 4081) are
third-party derivatives currently tracked under `data/gold-standard/` — confirm
their redistribution is acceptable for academic release before flipping the repo
to public.
