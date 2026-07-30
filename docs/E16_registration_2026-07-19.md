# E16 — Registration: Calibrated Abstention & Structural Repair (post-hoc, CPU-only)

**Registered:** 2026-07-19, before any inspection of confidence distributions.
**Status of underlying data:** existing `predictions.tsv` of the final matrix cells; no new inference. Analysis runs only after both Tag-2 queues (data integration, CC export) are cleared.

## Hypotheses

- **H1 (abstention):** The verbalized confidence logged with every Stage-2 verdict separates gold-pair assertions (TP) from non-gold assertions (FP) well enough that a single per-model global threshold τ, tuned on a development pair and applied frozen, improves strict F1 on held-out test cases.
- **H2 (repair):** Transitive reduction of the predicted subsumption graph (keep only the most specific assertion per chain, using the target-ontology closure already built for the audit) removes ancestor-redundant assertions and improves strict precision; strict recall is unchanged by construction (only non-gold ancestors of a kept assertion are removable when gold is the lowest node), credited recall may move and is reported.

## Data & split

- **Cells:** all four models × {g1-web, g2-diseases, g3-text, g5-groceries, g7-literature, mouse-human}; vdi-ebay gold cells added as a second application arm once landed.
- **Development pool (τ tuning): g1-web + g2-diseases pooled**, per model. Rationale: smallest pairs, least evidentially central; both excluded from headline claims.
- **Adequacy gate (pre-stated):** if the dev pool contains **fewer than 30 FP assertions** for a model, τ tuning for that model is declared underpowered; the experiment then reports distributions only for that model, no tuned operating point. **Registered fallback (used only if the gate fires for ≥2 models):** promote g3-text to dev and remove g3 from that model's test set — trading away the strongest test case rather than peeking.
- **Test set:** all remaining cells, thresholds applied frozen, no per-test-case adjustment.

## Threshold rule

τ_model = argmax over the grid of observed distinct dev confidence values of pooled dev strict F1 (assertions with confidence < τ are relabeled to none). Tie-break: the **higher** τ (more conservative). No smoothing, no per-dataset τ.

## Degeneracy / abort rule (pre-stated)

If a model emits **fewer than 5 distinct confidence values** across dev assertions, or **>90 % of assertions share a single value**, its confidence channel is declared degenerate: negative result (one paragraph, OLaLa's Llama-2 logit finding as precedent), no tuning, no further E16 arms for that model.

## Success criteria (adoption bar for the main text)

- **Primary (H1):** pooled held-out strict F1 improves by **≥ +0.03 absolute** for the model, **and** no single held-out test case loses more than 0.05 strict F1.
- **Primary (H2):** pooled strict precision improves by **≥ +0.02** with strict recall unchanged (construction check must pass exactly).
- **Secondary (descriptive, no bar):** number of test cases improved; credited-score deltas; combined arm (τ then reduction) reported without its own criterion.
- Below bar → honest negative/marginal paragraph; Future-Work text remains as written.

## Contingency branch (registered now, decided later)

**If** H1 is degenerate or below bar **for both reasoning models**, the verification second-pass (one additional prompt over asserted pairs only, feasible in hours on the new 4-GPU quota) becomes eligible as a single registered follow-up: separate mini-registration, decision on **2026-07-22**, hard kill **2026-07-24**. **If H1 succeeds, the second-pass stays future work** — no stacked interventions whose contributions cannot be attributed.

## Integration cap (on success)

One subsection in 4.3 ("From floor to operating point", ~¾ page + one table), one reworded limitation in 4.5, two sentences in Chapter 5, optional half-sentence in the abstract. Estimated 1.5 h writing, inside the Tag-2/3 integration window.

## Timebox

Analysis complete ≤ 2026-07-21 EOD · branch decision 2026-07-22 · hard kill 2026-07-24 (afterwards the topic returns to Future Work unchanged).

## Piggyback (independent of H1/H2 outcomes)

The analysis script also extracts `LLM_BATCH`/`LLM_CONC` from every cell's `config.json` (commit 73bb409 introduced the env overrides; defaults 8/16) and emits a per-cell serving-configuration table for the Appendix-A reproducibility note, including which Gold-Quartett cells ran at elevated settings.

---
*Addendum H3 (Structural Coherence Gate), registered 2026-07-21: see `E16_addendum_H3_structural_gate_2026-07-21.md`.*
