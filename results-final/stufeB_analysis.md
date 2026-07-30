# Stufe B — Both-Order-Voting: order-invariant directional classification

Registered analysis (docs/stage2_stufeB_registration.md). Primary = Macro-F1 over {<,>,=}, reranker-conditional; partOf folded to none. B1 abstain · B2 confidence tie-break · B3 symmetry-grounded.

## Per-dataset & dev-pooled metrics per variant

| Variant | Dataset | Macro-F1 | =-F1 | flip_gt | flip_lt | disagree | abstain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | g7-literature | 0.474 | 0.917 | 0.000 | — | 0.358 | 0.964 |
| B1 | g5-groceries | 0.305 | 0.704 | 0.000 | 0.000 | 0.263 | 0.935 |
| **B1** | **dev-pooled** | **0.347** | 0.769 | 0.000 | 0.000 | Δ=+0.013 → NO | |
| B2 | g7-literature | 0.502 | 0.846 | 0.062 | 0.267 | 0.358 | — |
| B2 | g5-groceries | 0.383 | 0.737 | 0.096 | 0.182 | 0.263 | — |
| **B2** | **dev-pooled** | **0.421** | 0.771 | 0.080 | 0.231 | Δ=+0.087 → SMALL | |
| B3 | g7-literature | 0.211 | 0.127 | 0.000 | — | 0.358 | — |
| B3 | g5-groceries | 0.131 | 0.182 | 0.000 | 0.000 | 0.263 | — |
| **B3** | **dev-pooled** | **0.144** | 0.160 | 0.000 | 0.000 | Δ=-0.190 → REVERSE (=-F1 0.160 < 0.7) | |

## Named-26 flip-set destinations (g7; n=26)

| Variant | →`<` | →`>` | →`=` | →none |
| --- | ---: | ---: | ---: | ---: |
| B1 | 0 | 0 | 0 | 26 |
| B2 | 3 | 22 | 1 | 0 |
| B3 | 0 | 0 | 24 | 2 |

*AB-order integrity vs v2 baseline (g7): 377 shared kept pairs, 0 mismatches (AB reproduces 255471).*

## Decision (registered)

- Baseline anchor dev-pooled Macro-F1 = 0.334.
- B1: Macro-F1 0.347 (Δ +0.013) → NO
- B2: Macro-F1 0.421 (Δ +0.087) → SMALL
- B3: Macro-F1 0.144 (Δ -0.190) → REVERSE (=-F1 0.160 < 0.7)
- **Winner: B2** (highest Macro-F1 among SOLID/SMALL guard-passers; tie→higher recall).

## '<'-heavy guard slice readout — winner B2 (mouse-human, n=45; READ-ONLY, not in selection)

- subclass-F1 0.784 (recall 0.644) · superclass-F1 0.000 · =-F1 0.000
- Guard check: the winner must NOT collapse subclass-F1 here (dev is '>'-heavy; this is the '<'-heavy sanity).
