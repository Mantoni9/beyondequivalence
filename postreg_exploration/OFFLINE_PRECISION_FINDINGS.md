# Offline precision-lever exploration — detailed arm results

Companion detail to `README_THESIS_HANDOFF.md`. Offline only, credited macro-F1,
harness baseline reproduces `e17_test_results.tsv`. Signals: `p_yes`/`p_yes_rev`
(E17 verifier, GPU) and Stage-2 `confidence` (logged, GPU-free).

## Δ credited macro-F1 vs baseline — OAEI-mean (g3,g5,g7,mh), p_yes-ranked
| arm | llama | mistral | gemma | gpt-oss |
|---|---|---|---|---|
| BC | −0.005 | +0.118 | +0.010 | +0.000 |
| EXT.W1 | +0.061 | +0.069 | +0.031 | +0.001 |
| EXT.W2 | +0.043 | +0.050 | +0.027 | +0.009 |
| EXT.W3 | +0.031 | +0.039 | +0.016 | +0.004 |
| BCthenEXT | +0.013 | +0.121 | +0.028 | +0.009 |
| EXTthenBC | +0.029 | +0.121 | +0.035 | +0.009 |
| MARGIN:0.0 | +0.031 | +0.023 | +0.046 | −0.026 |
| MARGIN:0.1 | +0.041 | +0.025 | +0.060 | −0.026 |
| MARGIN+EXT | +0.059 | +0.051 | +0.055 | −0.029 |
| EXTbyMargin.W1 | +0.064 | +0.059 | +0.031 | +0.001 |
| CONFdev(P=.5) | −0.028 | −0.237 | −0.373 | −0.479 |
| CONForacle(P=.5) | −0.120 | −0.052 | −0.373 | −0.479 |

## vdi (full ref), `<`-only credited F1, Δ vs baseline
| arm | llama | mistral | gemma | gpt-oss |
|---|---|---|---|---|
| BC | −0.047 | −0.158 | +0.013 | −0.008 |
| EXT.W1 | +0.051 | +0.171 | +0.257 | −0.052 |
| EXT.W2 | +0.083 | +0.210 | +0.189 | −0.008 |
| EXT.W3 | +0.083 | +0.176 | +0.133 | +0.031 |
| MARGIN+EXT | +0.004 | +0.217 | +0.205 | −0.117 |
| EXTbyMargin.W1 | +0.051 | +0.160 | +0.257 | −0.052 |

## GPU-free — EXT ranked by Stage-2 confidence (matrix cells, g3/g5/g7), W2
| model | g3 | g5 | g7 | OAEI3-mean |
|---|---|---|---|---|
| llama | +0.011 | +0.021 | +0.031 | +0.021 |
| mistral | +0.009 | +0.037 | +0.058 | +0.035 |
| gemma | +0.035 | +0.012 | +0.016 | +0.021 |
| gpt-oss | +0.021 | +0.005 | +0.002 | +0.009 |
(W1 occasionally negative with the noisier Stage-2 ranking; W2 robust default.)

## Best arm per model (OAEI-mean absolute credited macro)
| model | baseline | best arm | value | Δ |
|---|---|---|---|---|
| llama | 0.303 | EXTbyMargin.W1 | 0.367 | +0.064 |
| mistral | 0.237 | EXTthenBC | 0.358 | +0.121 |
| gemma | 0.373 | MARGIN:0.1 | 0.433 | +0.060 |
| gpt-oss | 0.479 | BCthenEXT | 0.488 | +0.009 |

## Findings
1. Coherence extractor generalizes vdi→OAEI and works GPU-free (Stage-2-conf) —
   the structure does the work, not the ranking signal. Main result.
2. p_yes-ranking ~doubles the gain vs Stage-2-conf (carries E17's independent signal).
3. Stacking EXT+BC ≈ max(EXT,BC), not additive — pick one lever per model.
4. Conformal precision-gate fails (score not discriminative); threshold methods lose
   to the structural method.
5. Direction-margin is model-specific (gemma gate, llama tie-break); hurts reasoner.
6. Over-asserter-vs-reasoner contrast holds in every arm (gpt-oss ≤ +0.009).

## Caveats
- p_yes arms need the E17 GPU pass; only Stage-2-conf extractor is truly GPU-free.
- mh+vdi absent from the GPU-free table (matrix cells on DWS; g3/g5/g7 local).
- vdi vs reference_full; OAEI vs standard gold; credited metric throughout.
- Conformal tested at target precision 0.5; lower targets → plain threshold tuning (≈V3).
