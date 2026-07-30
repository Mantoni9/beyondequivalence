<!--
  MIRRORED registration document — do not edit here.
  Source of record: commit e98c0b3 (full: e98c0b3e0f748cb8ab7f03ba442e1d78ab66b6f6)
                    file docs/swap_ablation_registration.md
                    branch experiment/stage1-query-swap · tag thesis-query-swap
  Mirrored into main on 2026-07-30 so the pre-registration lives in the
  submission history; the query-swap experiment itself was NOT merged (it modifies
  evaluation_recall.py; kept as a separate experiment branch per REPO_INVENTORY.md).
-->

# Swapped-retrieval ablation — pre-registration record

All items below were filed and committed **before** the swap-ablation GPU job
was submitted (no swap results existed at registration time). Git history of
this file and of the referenced commits is the timestamp evidence.

Frozen baseline (Phase 0, computed from the d11c97e artifacts, exact-validated
against stored metrics and independently recomputed): pooled pair-coverage@20
`>` **0.608** (657/1081), `<` 0.928, `=` 0.998; volume 133,720 unique pairs at
the top-20-per-(query, direction) cut; per_relation_strict R@20 sub 0.902 /
sup 0.565 (headline metric, demoted to secondary comparability).

## Registered outcomes (as amended)

- **PRIMARY:** pooled `>`-pair-coverage at the candidate's budget vs the @20
  baseline 0.608 — apples-to-apples (coverage vs coverage).
- **Bands:** SOLID = ≥ 0.80 AND Δ ≥ +0.05 AND guards pass; "≥ 0.80 with
  Δ < 0.05" = baseline adequate, no switch; PARTIAL 0.65–0.80 (escalates);
  NO EFFECT < 0.65 (negative result); REVERSE = no candidate passes guards
  (investigate before adoption).
- **Guards:** pooled `<` and `=` coverage@20 drop ≤ 0.02 vs the same-run
  baseline variant; pooled volume ≤ 1.3 × baseline pairs@20 (= 173,836) —
  an absolute Stage-2 cost ceiling that does NOT shrink with Kt.
- **Structural-guard framing:** with the s-side fixed @20, v_3pass and
  v_union are supersets of the baseline pass set at any Kt — their `<`/`=`
  guards pass by construction and carry no information. Live guard
  questions: v_sym's `<`/`=` (quantifies the s_narrower cross-rescue) and
  the volume cap for every candidate. A positive `<`/`=` drop on a superset
  variant is flagged as data corruption, never reported as a guard result.

## Candidate set (closed before unblinding)

Exactly six adoption candidates: **{v_sym, v_3pass, v_union} × Kt ∈ {20, 10}**,
s-side fixed @20. Kt is a **side-level** cap (both t_broader and t_narrower in
v_union — confirmed). Kt=5 rows are reporting-only, never adoptable.
Precedence among guard-passers: highest `>`-coverage; ties (Δ < 0.01) break
toward lower volume. The decision runs only for the primary config
(Qwen3-noLoRA); Nemo+LoRA is a robustness side-run, reported in full, never
an adoption candidate; if its best swap candidate beats the primary's
`>`-coverage by ≥ 0.05 pooled, an escalation flag is raised (model-freeze
question to Antonio) — configs are never auto-blended.

Registered predictions: (a) mechanism — Δ`>`-coverage@20 concentrates on
mouse-human / g3-text / g5-groceries and is ≈ 0 on g1-web / g2-diseases /
g7-literature (at ceiling); misplaced gains revisit the mechanism story even
if SOLID. (b) K-sweep (v_sym, v_3pass) — fan-in lists concentrate gold at low
ranks, so Kt=10 loses < 0.02 `>`-coverage vs Kt=20.

Implementation: `evaluate_candidates()` in `scripts/analyze_swap_results.py`,
pinned by `tests/test_candidate_decision.py`.

## Adoption-consequence procedure (registered before unblinding)

**IF a swap candidate is adopted at the freeze gate:**

1. Generate new frozen TSVs for all 6 datasets from the winning
   (variant, Kt) configuration; the old frozen TSVs are retired and kept
   read-only.
2. Bridge runs: Llama + d_subs_v2, identical decoding/job setup, on the NEW
   g7 and g5 TSVs → these become the dev baselines for all Stage-2 work from
   Stufe B onward. Job 255471 and the Stufe-A numbers stay valid as
   attribution evidence but are never mixed with new-TSV numbers in one
   comparison table (different conditional pair mix).
3. Stufe-A conclusions (the attribution verdict) carry over unchanged — they
   characterize model+prompt, not the candidate set.

**IF no candidate is adopted:** baseline TSVs confirmed, no bridge, Stufe-A
baselines remain the reference.

**Either way:** one TSV regime per table, stated in every results note.

## Run protocol

12 GPU runs ({Qwen3-noLoRA, Nemo+LoRA} × 6 datasets), each producing all four
passes from two invocations of the unmodified frozen matcher
(A=path_context, B=T2, top-50, seed 42); variants are offline pass subsets.
Gates (all decided AFTER every artifact is persisted): d11c97e identity check
(row set + stored per_relation_strict metrics; skipped counts as failure),
v_sym superclass-zero suspect, pooled `<`/`=` guard. Job:
`jobs/job_dws_swap_ablation.sh`; analysis:
`conda run -n melt-olala python scripts/analyze_swap_results.py --sha <sha>`.
