# E15 — Few-shot ablation: pre-registration

Filed and committed BEFORE any E15 run (register-before-data). Answers K4:
"few-shot rejection under-evidenced". Zero-shot stays the PRIMARY setting in
every outcome; E15 only decides whether the rejection is reported as *proven*
or as a scope limit with a reported delta.

## Central question
Does adding in-context exemplars change the Stage-2 relation classifier's
directional behaviour — and does it help, or does it teach the position prior
(Stufe A) rather than the task?

## BINDING SCOPE
- **Arms (5), all otherwise identical to the matrix protocol** (single-order,
  K=20, prompt-format d_subs_v2, description_path_context, per-model decoding):
  - **A0** zero-shot — the existing matrix cell (NO new run).
  - **A1** N=1: one `<` exemplar (replicates the pilot claim).
  - **A2** balanced-3: one each `<`,`>`,`=`.
  - **A3** balanced-6: two each.
  - **A4** mirrored-6: balanced-6 where each exemplar ALSO appears swapped
    (source↔target, label inverted) — an anti-position-prior control.
- **Models (2):** Llama-3.3-70B-AWQ (order-sensitive representative) and
  gpt-oss-120b (reasoner representative).
- **Eval datasets (3):** g5-groceries, g7-literature, g3-text. Directional gold:
  g7 67 · g5 85 · **g3 541** (g3 carries the statistical power).
- **Exemplar source:** the held-out **g1-web** track — verified gold, native
  `<`/`>`/`=` (29/26/275, so NO swap-derivation), evaluation-disjoint from
  {g5, g7, g3}, rich hierarchy (1131 subClassOf edges). NOT used as an eval set.
- **Selection:** deterministic, seeded (`fewshot_exemplars.select_exemplars`),
  sorted pools, seed = run seed (42). Exemplars verbalized with the same
  description_path_context as the eval pairs. A4 mirror inverts label and swaps
  the origin ontology per slot.
- **Grid:** A1–A4 × 2 models × 3 datasets = **24 new runs** (A0 reused).

## Pre-registered signatures (decided before data)
- **S1 — pilot replication:** `=`-F1 collapse under A1 (the N=1 claim).
- **S2 — position-teaching:** flip_rate_gt RISES for Llama under A2/A3 vs A0,
  and A4 (mirrored) reduces/shifts the flips vs A2/A3.
- **S3 — reasoner robustness:** gpt-oss arm-deltas ≤ the seed-noise band
  (±0.009, from the D9/matrix variance).

## Acceptance logic (fixed in advance)
Zero-shot remains the primary setting in EVERY outcome. E15 decides only the
*framing* of the rejection:
- S1 ∧ S2 hold → few-shot rejection reported as **proven** (second bias
  evidence next to Stufe A); the 3.3.2 one-liner is replaced by the registered
  finding + a compact 4.3 result paragraph.
- otherwise → rejection reported as a **scope limit with the measured delta**.
Either way the primary results are unchanged.

## Execution
- **Cluster:** bwUniCluster 3.0, partition `gpu_h100` (3-day walltime cap, 12
  nodes), serving env `vllm-e15` (vLLM 0.24), reranker client `melt-olala`.
  Chosen to run PARALLEL to the DWS Tier-3 tail without blocking it.
- **Job script:** `jobs/job_bwuni_e15.sh` — one job per (model × dataset × arm),
  walltime right-sized per submit (g5/g7 ~8h, g3 ~24h), generous vs. timeout.
- **Config self-description:** each run records few_shot_arm, exemplar_track,
  exemplar_seed, and the full exemplar_manifest (uris/relation/mirrored) in
  config.json.

## Analysis (after runs)
Per (model × dataset × arm): Macro-F1, per-class F1 (`<`/`>`/`=`),
flip_rate_gt/lt, direction-accuracy — same evaluator as the matrix. Deltas vs
A0. S1/S2/S3 checked against the pre-registered thresholds. Checkpoint before
any thesis edit.
