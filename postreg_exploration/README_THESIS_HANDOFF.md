# Stage-2 Precision — Thesis Handoff (NEW work, not yet in the thesis)

**Compiled 2026-07-30.** This bundle contains everything from the precision-direction
work that the thesis text does **not** yet know about: the vdi F1 diagnosis, the
**coherence extractor** (the main new method), the offline lever exploration
(BC-stacking, conformal, margin), the GPU-free result, and the literature verdict
on whether the problem is solved. All numbers are **offline, already-logged data**
— no new GPU inference was run. Decide with your other chat what to fold in and where.

Headline metric throughout = **credited macro-F1** (`closure_credit`; Euzenat
semantic + Kiritchenko hierarchical credit). vdi scored vs `reference_full.rdf`
(Experte C's precision reference); OAEI vs standard gold. The harness baseline
**exactly reproduces** `e17_test_results.tsv` (validated).

---

## 0. TL;DR — the three things worth putting in the thesis

1. **A coherence / cardinality extractor for Stage-2** (NEW method). A per-source
   greedy filter: a concept's asserted superclasses must form a *comparable chain*
   in the target hierarchy; sibling/branching assertions are mutually exclusive;
   keep the max-confidence chain, allow up to **W** branches for multiple
   inheritance. It is OLaLa's cardinality filter **adapted to subsumption's
   1:chain cardinality**. **GPU-free** (ranks by the already-logged Stage-2
   confidence). Positive for all over-asserting models on **OAEI and vdi**.
   → `data/table1..3`.

2. **The vdi "0.09 catastrophe" is largely a metric artifact** — decomposed into
   (a) sparse seed reference, (b) macro-averaging over near-empty `>`/`=`,
   (c) genuine over-assertion. The real subsumption number (`<`-credited F1 vs
   full) is **0.18–0.26 baseline → up to 0.49 with the extractor**, not 0.09.
   → `data/table4`.

3. **The problem is genuinely open in the literature, not solved** — our numbers
   sit *inside* the field's envelope (OAEI-2025 BeyondEquivalence best F1 0.14–0.31;
   industrial cross-vocab 0.002–0.04). A principled improvement on an open problem
   is a solid contribution. → §5 below.

---

## 1. The vdi diagnosis (why F1 looked like 0.09)

`<`-relation is 96% of the vdi gold; the models over-assert massively.
Decomposition (all in `data/table4_vdi_decomposition.tsv`):

| level | number | what it is |
|---|---|---|
| strict / seed | ~0.04–0.06 | the raw "catastrophic" macro number |
| **credited / seed** | ~0.08–0.11 | the E17 headline (credited metric already bridges most of the seed→full gap) |
| `<`-credited / full, baseline | **0.18–0.26** | fair number on the actual subsumption relation |
| `<`-credited / full **+ extractor** | **up to 0.49** (gemma) | with the new method |

Two compounding artifacts, one real problem:
- **Seed vs full:** eval used `reference_seed.rdf` (451 direct pairs) for precision,
  but Experte C's `make_reference.py` prescribes `reference_full.rdf` (2048, incl.
  transitive closure) for precision — *"jede Vorhersage, die hier steht, ist korrekt."*
- **Macro drag:** gold is 1976 `<` / 50 `>` / 22 `=`; the near-empty `>`/`=` have
  F1≈0 and pull the 3-way macro down. Report `<`-F1 or a weighted macro.
- **Real over-assertion:** models assert `<` on ~50% of Stage-1 candidates, but only
  ~19% are truly `<` → 2.7–4.4× over-assertion. Root cause: cosine retrieval pulls
  topically-adjacent **siblings** (all brake parts cluster), Stage-2 forced-choice
  prefers a relation over "none". This is what the extractor fixes.

**Stage-2 ceiling:** a perfect precision post-filter on the current assertions caps
`<`-credited F1 at ~0.59–0.69 (set by recall ~0.52). The extractor reaches ~0.49;
the residual gap is precision the extractor doesn't yet capture. Going above ~0.69
needs *recall* → that's Stage-1 (retrieval), not Stage-2.

---

## 2. The coherence extractor (the main new method)

Motivation from OLaLa (Hertling & Paulheim, K-CAP 2023): after the LLM, they run a
**cardinality filter** (1:1 for equivalence) + a high-precision matcher. The LLM
scores each pair *independently* → no global constraint → over-assertion. The
cardinality filter injects the global constraint. Our Stage-2 subsumption pipeline
had **no such step**. Subsumption isn't 1:1 (a concept has many superclasses), so
the correct analog is **1:chain**: a source's asserted `<`-targets must lie on a
comparable ancestor path; sibling branches are mutually exclusive.

Results — Δ credited macro-F1 vs baseline (`data/table1`, p_yes-ranked):

| arm | llama | mistral | gemma | gpt-oss |
|---|---|---|---|---|
| EXT.W1 (OAEI-mean) | +0.061 | +0.069 | +0.031 | +0.001 |
| EXT.W2 (vdi) | +0.028 | +0.070 | +0.063 | −0.003 |
| EXT.W2 `<`-only (vdi) | +0.083 | +0.210 | +0.189 | −0.008 |

**GPU-FREE version** (ranked by Stage-2 confidence, no E17 needed; `data/table3`,
g3/g5/g7): W2 positive for all — llama +0.021, mistral +0.035, gemma +0.021,
gpt-oss +0.009. **The structure does the work, not the ranking signal.** Two tiers:
free (Stage-2-conf) vs E17-boosted (p_yes, ~2× the gain).

**W is the one knob:** W=2 is the robust default; W=1 is best when the ranking is
clean (gemma), noisier signals need W=2 to recover recall.

---

## 3. What else we tried offline (and what it says)

Full arm sweep in `data/table1` + `OFFLINE_PRECISION_FINDINGS` (below). Summary:

- **Batch Calibration (BC)** — keep if p_yes > per-model prior p̄. Best only for
  **mistral** (+0.118 OAEI; its p_yes is compressed near 0, prior-subtraction is
  decisive). Neutral/negative elsewhere on vdi.
- **Stacking EXT+BC ≈ max(EXT, BC), NOT additive** — redundant (both remove FPs).
  Pick one lever per model, don't stack. (Negative result.)
- **Conformal precision-gate FAILS** — strongly negative even oracle-calibrated on
  test. The p_yes score isn't discriminative enough to hit a precision target
  without collapsing recall. Threshold methods (BC-as-threshold, V3, conformal)
  all lose to the *structural* extractor. (Negative result — worth one paragraph.)
- **Direction-margin (p_yes − p_yes_rev)** — helps gemma as a gate (+0.060 OAEI)
  and llama as the extractor's tie-break ranking (EXTbyMargin.W1 +0.064); *hurts*
  llama/mistral on vdi and always hurts the reasoner. Model-specific, secondary.
- **Reasoner control:** gpt-oss gains ≤ +0.009 from any lever, hurt by aggressive
  ones — the over-asserter-vs-reasoner contrast holds in *every* arm (same as
  E16/E17).

Best arm per model (OAEI-mean absolute credited macro): llama EXTbyMargin.W1
0.303→0.367; mistral EXTthenBC 0.237→0.358; gemma MARGIN 0.373→0.433; gpt-oss ~flat.

---

## 4. Is it solved in research? — No (verdict from 4 parallel literature reviews)

- **Only benchmark that scores full subsumption P/R/F1** is OAEI-2025
  BeyondEquivalence: best macro-F1 **0.06–0.19**, isAmong best **~0.20**,
  industrial cross-vocab (eClass/GPC/UNSPSC — our regime) **0.002–0.04**. We are
  *inside* this band.
- **Bio-ML subsumption reports ranking only (MRR), deliberately no P/R/F1** —
  because incomplete references bias precision. So there is **no precision-sensitive
  subsumption SOTA to underperform**.
- Community calls it open verbatim: Thiéblin et al. (SWJ 2020) "no benchmark…
  metrics not adapted"; OAEI 2025 "remains a challenging and open research problem".
- **Genuine gaps = our contribution:** no published work (a) couples LLM
  *subsumption* per-pair scores with a global consistency layer, or (b) repairs an
  over-asserted *predicted-subsumption* graph. Our W-bounded-antichain extractor
  sits in that gap.
- **Untried proven levers** (if you want more): conformal/selective prediction done
  *right* (as risk-coverage, not our failed precision-target), ComEM-style
  *comparison* prompts (COLING 2025, +23 precision pts — needs re-inference), NLI
  entailment gate (OntoLAMA), type-compatibility gate (Krompaß +77% AUPRC).
- **Avoid:** self-consistency voting, multi-agent debate (ICLR 2024: don't beat
  voting, flip correct→incorrect on binary). Coherence *repair* harms subsumption
  (Pesquita OM 2013: 80–95% of =→< weakenings wrong) — use to detect, not delete.

Key peer-reviewed anchors: OAEI-2025 (CEUR Vol-4144), Thiéblin (SWJ 2020),
He et al. (OM@ISWC 2023, LLM precision collapse to 0.08), OLaLa (K-CAP 2023),
Meilicke-Stuckenschmidt (OM 2007, greedy≈optimal H3-null), Solimando (KAIS 2017),
ComEM (COLING 2025), OntoLAMA (ACL-F 2023), BC (ICLR 2024), Weeds (COLING 2014).

---

## 5. Suggested thesis placement (for you + the other chat to decide)

| finding | where it could go | strength |
|---|---|---|
| Coherence extractor (GPU-free, OAEI+vdi) | **new subsection in Ch. 4** (Stage-2 precision) or an own short chapter | strongest — novel method + positive results |
| vdi F1 decomposition (seed/full, macro drag, ceiling) | vdi/eBay case-study section; fixes the "0.09" framing | high — corrects a misleading number |
| "problem is open" literature framing | Related Work / Discussion | high — positions the contribution honestly |
| Conformal + stacking negatives | Discussion / limitations | medium — honest negative results |
| Margin, BC-per-model | ablation table | low/secondary |

**No new GPU run is required** for the core result. To complete the GPU-free 5-set
table you only need to **retrieve** the mh + vdi matrix cells from DWS (already
computed) — g3/g5/g7 are in `results/stage2_results_bundle/02_matrix_cells/` locally.

---

## 6. File index & reproduction

```
README_THESIS_HANDOFF.md          ← this file
data/
  table1_arm_comparison_pyes.tsv          all arms × model × {OAEI-mean,vdi} Δ credited (p_yes-ranked)
  table2_vdi_absolute.tsv                 vdi absolute credited macro + <-only P/R per arm
  table3_extractor_gpufree_stage2conf.tsv EXTRACTOR ranked by Stage-2 confidence (GPU-free), g3/g5/g7, W1/2/3
  table4_vdi_decomposition.tsv            vdi seed-vs-full, per-relation, oracle precision ceiling
scripts/
  dump_all.py     regenerates table1+table2   (conda run -n melt-olala python3 dump_all.py <out_dir>)
  dump_t34.py     regenerates table3+table4   (conda run -n melt-olala python3 dump_t34.py <out_dir>)
raw/
  e17_verify/     asserted pairs + p_yes + p_yes_rev, 4 models × 5 test sets (the core scored data)
  vdi_gold/       reference_seed.rdf, reference_full.rdf, ebay_kfz_target.owl
```

Matrix cells (Stage-2 confidence, 49 MB) not bundled — they live at
`results/stage2_results_bundle/02_matrix_cells/` in the repo. Closures/OAEI gold
load from `benchmark.zip` + `~/oaei_track_cache/zenodo/` via `tracks/zenodo_loader`.
All scripts assume cwd = repo root and the `melt-olala` conda env.
