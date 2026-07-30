# Post-Registration Exploration (Stage-2 precision)

> **Status: post-registration, exploratory. NOT part of the pre-registered thesis
> results.** All numbers here were produced **offline on already-logged Stage-2
> outputs** (the E17 verifier scores and matrix-cell confidences) — **no new
> inference and no new GPU runs were performed.** The harness baseline exactly
> reproduces the frozen `e17_test_results.tsv`.
>
> Referenced in the thesis appendix **"Post-Registration Exploration"** as work
> carried out after the results freeze; it does not alter any registered result.

Contents:
- `README_THESIS_HANDOFF.md` — full narrative + thesis-placement notes.
- `OFFLINE_PRECISION_FINDINGS.md` — detailed per-arm result tables.
- `data/` — machine-readable result tables (table1–4).
- `scripts/` — `dump_all.py`, `dump_t34.py` regenerate all tables (cwd = repo root,
  conda env `melt-olala`).
- `raw/` — the scored inputs (e17 verifier pairs + vdi gold references) for reproduction.

Main finding: a **coherence / cardinality extractor** for Stage-2 (per-source
chain constraint, W branches) is a GPU-free precision post-filter that generalizes
across OAEI and the vdi case. See `README_THESIS_HANDOFF.md` §0.
