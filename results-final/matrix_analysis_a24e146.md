# Stage-2 model matrix — reasoner vs non-reasoner direction resolution

Registered: docs/stage2_matrix_registration.md. PRIMARY = Macro-F1 over {<,>,=}, reranker-conditional, single-order. partOf folded to none. Direction-accuracy reported but NOT primary.

## Primary table (per model × dataset, seed 42 canonical)

**Macro-cond / Micro-cond** = reranker-CONDITIONAL (Stage-1 misses excluded) — the FAIR comparison; **Micro for the TaSeR Table-5 comparison**, Macro for the per-class honesty story. **Macro-e2e / Micro-e2e** = end-to-end (candidate∪gold, misses as FN; Stufe-A/B basis). All over {<,>,=}; [CI] = percentile bootstrap. Per-class F1 / flip / dir-acc are on the e2e report.

| Model | Dataset | Macro-cond [CI] | Micro-cond [CI] | Macro-e2e | Micro-e2e | <-F1 | >-F1 | =-F1 | none-F1 | flip_gt | flip_lt | dir-acc | parse_fail | quant |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gemma4 | g3-text | **0.377** [0.348,0.405] | **0.384** [0.362,0.407] | 0.357 | 0.365 | 0.395 | 0.282 | 0.394 | 0.867 | 0.009 | 0.003 | 0.996 | 0.000 | bf16 |
| gemma4 | g5-groceries | **0.354** [0.296,0.409] | **0.298** [0.255,0.346] | 0.344 | 0.279 | 0.167 | 0.249 | 0.617 | 0.816 | 0.000 | 0.000 | 1.000 | 0.000 | bf16 |
| gemma4 | g7-literature | **0.405** [0.322,0.471] | **0.319** [0.255,0.382] | 0.403 | 0.317 | 0.078 | 0.547 | 0.585 | 0.873 | 0.050 | 0.000 | 0.957 | 0.000 | bf16 |
| gpt-oss | g3-text | **0.479** [0.445,0.511] | **0.465** [0.439,0.490] | 0.455 | 0.437 | 0.479 | 0.307 | 0.579 | 0.906 | 0.017 | 0.011 | 0.988 | 0.000 | MXFP4 (no BF16 ref) |
| gpt-oss | g5-groceries | **0.413** [0.345,0.473] | **0.316** [0.260,0.371] | 0.402 | 0.290 | 0.227 | 0.237 | 0.741 | 0.869 | 0.000 | 0.091 | 0.981 | 0.000 | MXFP4 (no BF16 ref) |
| gpt-oss | g7-literature | **0.544** [0.479,0.599] | **0.394** [0.323,0.460] | 0.542 | 0.390 | 0.153 | 0.558 | 0.917 | 0.895 | 0.068 | 0.200 | 0.898 | 0.000 | MXFP4 (no BF16 ref) |
| llama | g3-text | **0.229** [0.203,0.252] | **0.280** [0.259,0.301] | 0.226 | 0.267 | 0.279 | 0.007 | 0.392 | 0.849 | 0.991 | 0.000 | 0.766 | 0.000 | AWQ-INT4 |
| llama | g5-groceries | **0.278** [0.220,0.331] | **0.139** [0.100,0.182] | 0.270 | 0.129 | 0.059 | 0.095 | 0.655 | 0.837 | 0.824 | 0.000 | 0.311 | 0.000 | AWQ-INT4 |
| llama | g7-literature | **0.483** [0.398,0.548] | **0.202** [0.152,0.252] | 0.481 | 0.200 | 0.089 | 0.507 | 0.846 | 0.810 | 0.578 | 0.000 | 0.574 | 0.000 | AWQ-INT4 |
| mistral | g3-text | **0.233** [0.204,0.265] | **0.226** [0.208,0.244] | 0.214 | 0.217 | 0.214 | 0.069 | 0.358 | 0.790 | 0.917 | 0.000 | 0.766 | 0.000 | bf16 |
| mistral | g5-groceries | **0.247** [0.181,0.296] | **0.078** [0.053,0.104] | 0.242 | 0.074 | 0.036 | 0.048 | 0.642 | 0.687 | 0.938 | 0.000 | 0.211 | 0.000 | bf16 |
| mistral | g7-literature | **0.248** [0.157,0.310] | **0.074** [0.044,0.106] | 0.248 | 0.074 | 0.050 | 0.000 | 0.692 | 0.701 | 1.000 | 0.000 | 0.220 | 0.000 | bf16 |

## Reference floors (basis labelled — NOT the full-basis primary macro)

> The **direction floor is dir-accuracy 0.5** — compare it to each model's `dir-acc` column in the primary table (same directional-gold basis). The macro columns below are on the CONDITIONAL gold (excludes none-FPs/misses), so they are NOT directly comparable to the full-basis primary Macro-F1; they bound the conditional macro.

| Dataset | n_dir | random-dir: dir-acc | random-dir: dir-F1 (2-class) | majority-class: macro {<,>,=} (cond) (class) |
| --- | ---: | ---: | ---: | --- |
| g3-text | 541 | 0.500 | 0.468 | 0.265 (<) |
| g5-groceries | 85 | 0.499 | 0.424 | 0.260 (>) |
| g7-literature | 67 | 0.501 | 0.462 | 0.262 (>) |

## 4×4 confusion {rows=gold, cols=pred: <,>,=,none} (none-row is first-class)

**gemma4 · g3-text** (none P/R/F1 0.973/0.782/0.867)
| gold↓ pred→ | < | > | = | none |
| --- | ---: | ---: | ---: | ---: |
| < | 380 | 1 | 13 | 31 |
| > | 1 | 113 | 13 | 140 |
| = | 1 | 2 | 67 | 0 |
| none | 1117 | 418 | 177 | 6127 |

**gpt-oss · g3-text** (none P/R/F1 0.972/0.848/0.906)
| gold↓ pred→ | < | > | = | none |
| --- | ---: | ---: | ---: | ---: |
| < | 373 | 4 | 3 | 45 |
| > | 2 | 114 | 3 | 148 |
| = | 2 | 4 | 64 | 0 |
| none | 756 | 354 | 81 | 6648 |

**llama · g3-text** (none P/R/F1 0.971/0.754/0.849)
| gold↓ pred→ | < | > | = | none |
| --- | ---: | ---: | ---: | ---: |
| < | 368 | 0 | 21 | 36 |
| > | 113 | 1 | 14 | 139 |
| = | 1 | 0 | 69 | 0 |
| none | 1733 | 16 | 178 | 5912 |

**mistral · g3-text** (none P/R/F1 0.969/0.667/0.790)
| gold↓ pred→ | < | > | = | none |
| --- | ---: | ---: | ---: | ---: |
| < | 353 | 0 | 38 | 34 |
| > | 111 | 10 | 13 | 133 |
| = | 2 | 0 | 68 | 0 |
| none | 2405 | 11 | 191 | 5232 |

**gemma4 · g5-groceries** (none P/R/F1 0.948/0.716/0.816)
| gold↓ pred→ | < | > | = | none |
| --- | ---: | ---: | ---: | ---: |
| < | 11 | 0 | 1 | 2 |
| > | 0 | 54 | 8 | 51 |
| = | 0 | 4 | 25 | 0 |
| none | 107 | 262 | 18 | 974 |

**gpt-oss · g5-groceries** (none P/R/F1 0.937/0.810/0.869)
| gold↓ pred→ | < | > | = | none |
| --- | ---: | ---: | ---: | ---: |
| < | 10 | 1 | 0 | 3 |
| > | 0 | 42 | 2 | 69 |
| = | 1 | 6 | 20 | 2 |
| none | 63 | 193 | 3 | 1102 |

**llama · g5-groceries** (none P/R/F1 0.943/0.752/0.837)
| gold↓ pred→ | < | > | = | none |
| --- | ---: | ---: | ---: | ---: |
| < | 10 | 0 | 1 | 3 |
| > | 42 | 9 | 3 | 59 |
| = | 9 | 1 | 19 | 0 |
| none | 265 | 66 | 6 | 1024 |

**mistral · g5-groceries** (none P/R/F1 0.938/0.542/0.687)
| gold↓ pred→ | < | > | = | none |
| --- | ---: | ---: | ---: | ---: |
| < | 12 | 0 | 0 | 2 |
| > | 60 | 4 | 2 | 47 |
| = | 12 | 0 | 17 | 0 |
| none | 568 | 50 | 5 | 738 |

**gemma4 · g7-literature** (none P/R/F1 0.984/0.784/0.873)
| gold↓ pred→ | < | > | = | none |
| --- | ---: | ---: | ---: | ---: |
| < | 7 | 0 | 8 | 3 |
| > | 2 | 38 | 3 | 9 |
| = | 0 | 0 | 12 | 0 |
| none | 153 | 49 | 6 | 753 |

**gpt-oss · g7-literature** (none P/R/F1 0.987/0.819/0.895)
| gold↓ pred→ | < | > | = | none |
| --- | ---: | ---: | ---: | ---: |
| < | 12 | 3 | 0 | 3 |
| > | 3 | 41 | 1 | 7 |
| = | 1 | 0 | 11 | 0 |
| none | 123 | 51 | 0 | 787 |

**llama · g7-literature** (none P/R/F1 0.989/0.686/0.810)
| gold↓ pred→ | < | > | = | none |
| --- | ---: | ---: | ---: | ---: |
| < | 16 | 0 | 0 | 2 |
| > | 26 | 19 | 2 | 5 |
| = | 1 | 0 | 11 | 0 |
| none | 297 | 4 | 1 | 659 |

**mistral · g7-literature** (none P/R/F1 0.987/0.543/0.701)
| gold↓ pred→ | < | > | = | none |
| --- | ---: | ---: | ---: | ---: |
| < | 13 | 0 | 3 | 2 |
| > | 46 | 0 | 1 | 5 |
| = | 3 | 0 | 9 | 0 |
| none | 438 | 0 | 1 | 522 |

## McNemar (model-vs-model, paired on directional gold; exact binomial)

**g3-text** (pinned n_directional=541; paired n per row = actual shared candidate-present directional gold)
| A vs B | paired n | A✓B✗ | A✗B✓ | p (McNemar) |
| --- | ---: | ---: | ---: | ---: |
| gemma4 vs gpt-oss | 541 | 21 | 15 | 0.4050 |
| gemma4 vs llama | 541 | 135 | 11 | 0.0000 |
| gemma4 vs mistral | 541 | 139 | 9 | 0.0000 |
| gpt-oss vs llama | 541 | 136 | 18 | 0.0000 |
| gpt-oss vs mistral | 541 | 144 | 20 | 0.0000 |
| llama vs mistral | 541 | 27 | 21 | 0.4709 |

**g5-groceries** (pinned n_directional=85; paired n per row = actual shared candidate-present directional gold)
| A vs B | paired n | A✓B✗ | A✗B✓ | p (McNemar) |
| --- | ---: | ---: | ---: | ---: |
| gemma4 vs gpt-oss | 85 | 17 | 4 | 0.0072 |
| gemma4 vs llama | 85 | 47 | 1 | 0.0000 |
| gemma4 vs mistral | 85 | 50 | 1 | 0.0000 |
| gpt-oss vs llama | 85 | 34 | 1 | 0.0000 |
| gpt-oss vs mistral | 85 | 39 | 3 | 0.0000 |
| llama vs mistral | 85 | 8 | 5 | 0.5811 |

**g7-literature** (pinned n_directional=67; paired n per row = actual shared candidate-present directional gold)
| A vs B | paired n | A✓B✗ | A✗B✓ | p (McNemar) |
| --- | ---: | ---: | ---: | ---: |
| gemma4 vs gpt-oss | 67 | 1 | 9 | 0.0215 |
| gemma4 vs llama | 67 | 19 | 9 | 0.0872 |
| gemma4 vs mistral | 67 | 39 | 7 | 0.0000 |
| gpt-oss vs llama | 67 | 22 | 4 | 0.0005 |
| gpt-oss vs mistral | 67 | 44 | 4 | 0.0000 |
| llama vs mistral | 67 | 22 | 0 | 0.0000 |

## '<'-precision decomposition question (same across models → structural/gold; a reasoner improving it → model-specific)

| Dataset | gemma4 <-prec | gpt-oss <-prec | llama <-prec | mistral <-prec |
| --- | ---: | ---: | ---: | ---: |
| g3-text | 0.254 | 0.329 | 0.166 | 0.123 |  → ⚠ model-specific (spread≥0.10)
| g5-groceries | 0.093 | 0.135 | 0.031 | 0.018 |  → ⚠ model-specific (spread≥0.10)
| g7-literature | 0.043 | 0.086 | 0.047 | 0.026 |  → structural (spread<0.10)

*Gold-gap component quantified by the blind '<'-FP audit TSVs (see foreign_audit); corrected '<'-precision excludes gold_gap FPs.*

## Variance — per-class F1 spread across g3 seeds (reasoner noise floor)

| Model | dataset | seeds | <-F1 spread | >-F1 spread | =-F1 spread | Macro-F1 spread |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| gemma4 | g3-text | [7, 42, 123] | 0.002 | 0.001 | 0.005 | 0.002 |
| gpt-oss | g3-text | [7, 42, 123] | 0.004 | 0.014 | 0.025 | 0.009 |

## '<'-FP composition (gold-gap audit population vs direction errors)

| Dataset | gold=none '<'-FP (→ audit) | gold∈{>,=} '<'-FP (direction err) |
| --- | ---: | ---: |
| g3-text | 2738 | 233 |
| g5-groceries | 609 | 124 |
| g7-literature | 468 | 82 |