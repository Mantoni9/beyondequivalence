# Directional-validity check — registration (BEFORE computation)

Filed and committed before any of the numbers below were computed. Purpose:
test whether Nemo+LoRA's swap coverage gain reflects GENUINE directional
(parent-finding / fan-in) retrieval or RIDE-ALONG coverage (plausible
hypernym neighbourhoods with gold riding along on list length). Gate: the
model-freeze reopen decision. Until this clears: no freeze-reopen, no TSV
changes, no Stage-2 submission; the DWS checkout stays as-is.

Setup: all-offline, read-only over `results/swap_*_e98c0b3/passes.tsv` +
local gold. Hypothesis model: **Nemo+LoRA**. Negative control:
**Qwen3-noLoRA** (Phase-3 evidence says it lacks genuine t-side `>` ability:
per_directed_query sup@20 0.609 vs sub 0.902). A check that cannot separate
the two models is uninformative and must be reported as such. "Covered" :=
within-list rank ≤ 20 (the budget) in the relevant pass; ranks are the
authoritative `rank` column of passes.tsv; gold is normalized + deduped as
everywhere else.

## Check 1 — rank distribution of '>' gold inside t_broader

For every covered gold (s, t, '>'): the rank of the correct s inside t's
broader list. Per dataset + pooled, both models; median, full histogram,
and shares at ranks 1–3 / 4–10 / 11–20. PRIMARY comparison on the SHARED
covered set (pairs covered by both models — removes composition effects);
full per-model sets reported as context. Uniform-over-20 reference: median
10.5, ranks-1–3 mass 0.15.

Registered bars (pooled, shared set):
- **GENUINE signature:** Nemo median ≤ 5 AND Nemo ranks-1–3 share ≥ 0.35,
  AND a clear gap to the control: Nemo median better than Qwen3's by ≥ 3
  ranks, OR Qwen3 fails both absolute bars while Nemo passes both.
- **RIDE-ALONG signature:** Nemo median ≥ 8 (≈ uniform), OR
  indistinguishable from the control (median gap < 2 ranks AND ranks-1–3
  share gap < 0.10).
- Anything else → MIXED.

## Check 2 — per-pair symmetric consistency (corroborative only)

Per gold '>' pair (s, t): fan-in hit = s in t_broader(t)@20; fan-out hit =
t in s_narrower(s)@20. Crosstab both / fan-in-only / fan-out-only / neither,
per model, per dataset + pooled. Agreement rate := both / (both + fan-in-only
+ fan-out-only).

Registered reading: the GAP is the signal, not Nemo's absolute number.
Corroborates GENUINE if Nemo agreement − Qwen3 agreement ≥ 0.15;
uninformative if |gap| < 0.05; in between = weak corroboration. Check 2
cannot decide alone.

## Check 3 — BLIND manual audit (decisive together with Check 1)

Sampling frame: gold '>' pairs covered by **Nemo's** t_broader@20 (the claim
under test), stratified 10 mouse-human + 10 g3-text, seed 42, deterministic
order before sampling. For each sampled pair: top-5 of t's broader list from
BOTH models. Audit rows are unique (pair, candidate) combinations — a
candidate returned by both models is judged once and the judgment applies to
both. Blind CSV columns: row_id, dataset, pair_id, t_label, candidate_label,
judgment(empty) — NO model column, NO gold-s column, rows shuffled seed 42.
The un-blinding key (row → model(s), URIs, ranks, gold s) lives in a separate
file, not shown to the judge. Judgments: `parent` (true superordinate of t) /
`neighbor` (related but not a superordinate) / `unsure`.

Registered bars (per-model true-parent precision over judged top-5 items,
`unsure` counts as not-parent):
- **GENUINE:** Nemo precision ≥ 0.50 AND Nemo − Qwen3 ≥ 0.15.
- **RIDE-ALONG:** gap < 0.10, OR both models < 0.40.
- Anything else → MIXED.

## Verdict logic (registered)

- **GENUINE** iff Check 1 = GENUINE AND Check 3 = GENUINE (Check 2
  corroborates) → mechanism claim holds; freeze-reopen is founded; proceed
  to verification package + bridge procedure.
- **RIDE-ALONG / INCONCLUSIVE** if Check 1 = RIDE-ALONG or Check 3 =
  RIDE-ALONG → honest finding: "swap lifts coverage but directional quality
  does not reliably transfer"; baseline stays; NO freeze-reopen on this
  evidence.
- **MIXED** otherwise → report both readings; decision stays with Antonio;
  no auto-adoption.

Deliverables: `results/swap_directional_validity_e98c0b3.md` (three tables +
verdict against this registration), blind CSV + separate key. Checkpoint
order: this registration first, then Check 1+2 results + blind CSV; the
final verdict completes only after the blind judging is un-blinded together.
