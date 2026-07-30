# Stage-2 model matrix — scaffolding prep (NOT pre-registered, NOT submittable)

> **STATUS: scaffolding only.** This documents what exists, what is reusable,
> and the OPEN design forks the matrix pre-registration must resolve. It is
> NOT a pre-registration and authorizes NO run. Hold for the matrix pre-reg
> prompt from the reviewing assistant. Nothing here is submitted.

## What is adopted going in
**B2 (confidence tie-break over a double-order run) is the adopted
order-invariant Stage-2 default for NON-REASONING models** (Stufe B, dev-pooled
Macro-F1 0.334→0.421, SMALL, guard-clean). Labeled honestly: order-invariant
voting HELPS moderately, it does NOT cleanly fix both directions
(flip_rate_gt 0.708→0.080 but flip_rate_lt 0.000→0.231 — it TRADES). B3
(symmetry-grounded) is a registered-predicted negative (=-F1 0.160, REVERSE).

## The central matrix question
Order-invariance (external repair) and reasoning (internal resolution) are
ORTHOGONAL. The matrix asks: **do reasoners (gpt-oss, Gemma-thinking) resolve
direction INTERNALLY better than external Both-Order-Voting repairs it from
outside?** B2 is the non-reasoning benchmark each reasoner is measured against.

## Matrix dimensions (per 03.06 scope — exact cells are the pre-reg's call)
- **Models (quartet):** Llama-3.3-70B-AWQ (non-reasoning), Mistral-Small-3.2
  (non-reasoning), gpt-oss-120b (reasoner), Gemma-3-31B (reasoner).
- **Reasoning toggle:** gpt-oss reasoning levels / Gemma thinking on|off —
  the within-model reasoning-attribution axis (Antonio's binding decision:
  reasoning attribution is within-model, not a CoT arm on Llama).
- **Inference mode per model:** non-reasoners → adopted B2 double-order.
  Reasoners → OPEN (see fork 1).
- **Datasets:** 03.06 scope. partOf excluded from primary scoring.
- **Primary metric:** Macro-F1 over {<,>,=}, reranker-conditional (dir-acc
  misleads on '>'-heavy data — Stufe-A/B proof). flip_rate_gt + flip_rate_lt,
  =-F1 ≥ 0.70 guard, parse_fail < 5% — all carry over.

## OPEN design forks the matrix pre-reg MUST resolve (do NOT decide here)
1. **B2 answer-span under reasoning.** B2's confidence = mean logprob of the
   FIRST response line ("Relation: <label>"). That holds for answer-first
   non-reasoners. Reasoners emit CoT FIRST, answer LAST — so "first line" would
   capture reasoning tokens, not the answer. The span definition for reasoners
   must be redefined (e.g. the tokens of the final "Relation: <label>" anchor
   line) AND re-gated (the 255391 manual-inspection gate re-applies per model).
2. **Do reasoners even use Both-Order?** If a reasoner resolves direction
   internally, single-order-with-reasoning may suffice (and Both-Order doubles
   cost). The matrix should likely run BOTH single-order and B2-double-order per
   reasoner to isolate "reasoning vs voting" — that is the orthogonality test.
3. **Model serving.** Each model is a different vLLM model/quantization; paths,
   tensor-parallel, max-model-len, and the thinking-toggle flags per model are
   cluster-specific and unspecified here. The dws-14/15/16/17 exclusion + the
   gpu:2 / enforce-eager setup carry over.
4. **Test-set release.** The dev/test protocol (dev={g7,g5}) governs until the
   final matrix; the matrix is presumably WHERE test is released — the pre-reg
   states the release rule explicitly.
5. **Cost.** Quartet × thinking-toggle × all datasets × (single+double order)
   is large; the pre-reg must scope it and the GPU budget.

## Reusable pieces (no new core code needed for non-reasoners)
- `run_stage2_bothorder.py` — model-agnostic via `--llm-model`; one double-order
  pass; persists per-order raw/canonical/span-logprob + token_dump (B2 gate).
- `stage2_bothorder.reconcile(..., variant="B2")` — the adopted reconciliation;
  `canonical_for_order` reuses the A2 exactly-once inversion.
- `scripts/analyze_stufeB.py` — Macro-F1 primary, both flip rates, =-F1 guard,
  named-flip-set destinations, AB-vs-baseline integrity, guard-slice readout.
- `jobs/job_dws_stage2_bothorder.sh` — env-parameterized (DATASET, PROMPT_ID,
  STAGE1_PREDICTIONS); model swap is a vLLM-config change.
- `docs/stufeB_guard_slice_mousehuman.tsv` (+ candidates) — the read-only
  '<'-heavy guard carries into the matrix.

## What a matrix runner would add (skeleton — build after pre-reg)
- a model registry: (alias → {hf/vllm path, tp, quant, thinking-toggle flag}).
- a loop over (model × thinking × dataset × {single, double}-order) calling the
  appropriate runner; reasoners get fork-1's redefined span + re-gate.
- one analyzer pass producing the matrix table (model × mode × Macro-F1 + flip
  rates + guard), with B2-non-reasoning as the registered benchmark column.

Until the matrix pre-reg lands, none of the above runner code is written and
no job is submitted.
