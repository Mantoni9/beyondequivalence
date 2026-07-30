# results-final/ — curated thesis result artifacts

Small, tracked, curated outputs behind the thesis tables/figures. Large raw
artifacts (`predictions.tsv`, `passes.tsv`, full `results/`) stay gitignored;
they are reproducible from the code + the Zenodo benchmark and are published in
the GitHub Release (see repository README).

| file | thesis relevance | source / generator |
|---|---|---|
| `matrix_analysis_a24e146.md` / `.json` | Stage-2 model-matrix results (reasoner vs non-reasoner) — **thesis-cited revision a24e146** | `scripts/analyze_matrix.py` over `results/matrix_*_seed42_*` |
| `stage1_ablation_pooled.csv` | Stage-1 ablation (embedder × LoRA × verbalization A × template B × dataset), R@10/R@20 | pooled from `results/ablbi_*_d11c97e/metrics.json` (git `d11c97e`) |
| `E16_18_evidence.tsv` | per-cell long-format evidence for E16 (abstention/pruning) + E17 (verification/BC) + E18 (consistency); 2301 rows | `scripts/e16_analysis.py`, `e17_*`, `e18_consistency.py` |
| `ltfp_audit_adjudicated_2026-07-19.tsv` | adjudicated `<`-false-positive gold-gap audit (ĝ = 0.375) | `scripts/build_ltfp_audit_sample.py`, `ltfp_corrected_precision.py` |
| `stufeB_analysis.md` / `.json` | Both-Order-Voting (B1/B2/B3) reconciliation analysis | `scripts/analyze_stufeB.py`, `stage2_bothorder.py` |
| `matrix_cells/<model>_<ds>_seed<n>/metrics.json` | per-cell Stage-2 metrics (16 cells); `predictions.tsv` intentionally excluded (large, in Release) | `run_stage2_experiment.py` |

## Note on a24e146 vs 8a22646 (matrix analysis)
The thesis cites **a24e146**. A later regeneration exists (`8a22646`, a descendant
that added Micro-F1 / conditional-Macro columns). Their **point estimates are
identical** on all shared cells; only the bootstrap CI brackets differ by ≤0.002
(resampling noise, not a code/data/seed divergence). Only the a24e146 version is
placed here. See `EXECUTION_LOG.md` (Phase 2.1).
