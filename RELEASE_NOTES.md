# Release prep — v1.0-thesis

Prepared 2026-07-30 (Phase 4). **Antonio executes the release in the browser**
(no `gh` CLI on this machine). This file has three parts: (A) the release body to
paste, (B) the asset checksums, (C) the step-by-step.

---

## A. Release body (paste into the GitHub release description)

> ### Beyond Equivalence in E-Commerce — thesis submission state (v1.0-thesis)
>
> Frozen code, pre-registrations, and result artifacts accompanying the MSc thesis
> *"Beyond Equivalence in E-Commerce: Hierarchical Category Mapping using Large
> Language Models and Advanced Embeddings"* (University of Mannheim, DWS, 2026).
>
> This tag marks the submitted state. Curated small artifacts are tracked in the
> repository under `results-final/`; the large raw bundles (full per-pair
> predictions, complete run directories) are attached here as assets.
>
> **Thesis-cited code revisions** (annotated tags):
> - [`thesis-stage1-ablation`](https://github.com/Mantoni9/beyondequivalence/releases/tag/thesis-stage1-ablation) — `d11c97e`, Stage-1 ablation layer
>   - [`thesis-query-swap`](https://github.com/Mantoni9/beyondequivalence/releases/tag/thesis-query-swap) — `e98c0b3`, query-swap study + registration
>   - [`thesis-matrix-analysis`](https://github.com/Mantoni9/beyondequivalence/releases/tag/thesis-matrix-analysis) — `a24e146`, Stage-2 matrix analysis
>
> OAEI BeyondEquivalence benchmarks are not redistributed — obtain from Zenodo
> (DOI 10.5281/zenodo.17091043). See `THIRD_PARTY_NOTICES.md` for eBay/VDI data terms.
>
> SHA-256 checksums for all assets are listed at the bottom of this description.

---

## B. Asset checksums (SHA-256)

Ready assets (present, verified 2026-07-30):

| asset | size | sha256 |
|---|---|---|
| `thesis_results_final_2026-07-21.zip` | 70 MB | `398b02e87150454d09e38c43d1e656d6290d6133e25173c43c05f9154aec3f81` |
| `stage2_precision_E16_E17_E18_2026-07-22.zip` | 5.0 MB | `1cec08f67655bd3c1f5ec9f57e656e7fffaeab0b570b5bb66d308003cf01abea` |
| `sbert_e15_results_2026-07-17.zip` | 9.9 MB | `a02d1b759d96448d49164055c828f1e01b3cc60ecfc9803abed8ae64233474da` |
| `gold-standard-v1.zip` | 106 KB | `a444963f8221bc7890244329907353ee217a77209e749ef9ba30a032828e3f4f` |
| `Markic_2026_BeyondEquivalence_thesis.pdf` | 2.0 MB | `1f3f6eeb2ea56b77f9f415ac72168e51e5a85f7ea3a466c71c51019687a28397` |

`gold-standard-v1.zip` was built from the tracked `data/gold-standard/` (canonical
gold: reference_full/seed, eBay/VDI OWL, subsumption_gold). Regenerate + re-checksum:
`rm -f gold-standard-v1.zip && zip -rq gold-standard-v1.zip data/gold-standard && shasum -a 256 gold-standard-v1.zip`

The **5 assets above are ready** for the initial release. The PDF was copied from
the exchange folder (original: `Thesis___Beyond_Equivalence_..._Embeddings.pdf`).

**Follow-up asset (ready 2026-07-30 — optional extra on the existing release):**

| asset | status | sha256 |
|---|---|---|
| `annotation-workbooks-v1.zip` (Expert A/B/adjudication C, anonymized xlsx + CSV) | ✅ ready | `b8626274253d1016ab633937839f8dfc9603978e826b97e3d4ab6059fe5220bd` |

The workbooks are also tracked in-repo under `data/gold-standard/annotation/`; the
zip is an optional convenience asset. Add it to the existing v1.0-thesis release
(no re-release needed).

To checksum on arrival: `shasum -a 256 <file>`. Add it as an extra asset to the
existing release once scrubbed (no re-release needed).

---

## C. Step-by-step (Antonio, in the browser)

1. Confirm the **default branch is `main`** and it is at the intended tip
   (Settings → General → Default branch). The submission tip is the latest `main`
   commit at release time.
   2. Go to **Releases → Draft a new release**.
   3. **Choose a tag → create new tag `v1.0-thesis` on target `main`** (GitHub creates
      the annotated tag on the current main tip when you publish).
   4. **Title:** `v1.0-thesis — thesis submission state`.
   5. **Description:** paste section **A** above (including the checksum table from **B**).
   6. **Attach assets** (drag the four ready zips + the PDF + workbook bundle once present):
      - `thesis_results_final_2026-07-21.zip`
      - `stage2_precision_E16_E17_E18_2026-07-22.zip`
      - `sbert_e15_results_2026-07-17.zip`
      - `gold-standard-v1.zip`
      - `Markic_2026_BeyondEquivalence_thesis.pdf` (open slot)
      - annotator-workbooks bundle (open slot)
   7. Leave **"Set as the latest release"** checked. **Publish release.**
   8. **Repo Description + Topics** (Settings / repo homepage "About" gear):
      - Description: *Beyond Equivalence in E-Commerce — hierarchical category mapping with LLMs + embeddings (MSc thesis, Uni Mannheim 2026).*
      - Topics: `ontology-matching`, `llm`, `oaei`, `master-thesis`
   9. **Public switch — DO NOT flip yet.** Only after the separate explicit GO and after
      confirming `THIRD_PARTY_NOTICES.md` §2/§3 (eBay/VDI redistribution) is acceptable.

After publishing, the README release link
(`.../releases/tag/v1.0-thesis`) resolves live — no README edit needed.

---

### Notes
- The three `thesis-*` tags are already pushed and independent of this release;
  they preserve the cited SHAs even if the release is edited.
  - Assets are large binaries kept out of git; they live in the repo working dir now
    and move to `~/Desktop/repo_archive_2026-07/` in Phase 5 — **create the release
    before Phase 5 archival**, or upload from the archive location.
