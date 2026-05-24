"""Pinned configurations from Sub-B description-ablation sweep.

Sweep group: subB_descablation_2026-05-03_12-18-11_2d92b24
630 runs total: 2 instruction-aware models × 6 datasets × 5 description methods × 5 templates × 2 modes (asym+sym)
                + 1 baseline model (sbert) × 6 datasets × 5 description methods (sym only)

Selection criterion (UPDATED 2026-05-05):
- Original Sub-B selection used MRR avg(sub,sup) and pinned T4 for asym.
- Stage-2-Reranker requires R@20 per_relation_strict.subclass as the
  primary input metric. On that metric T2 aggregates higher than T4
  (T2 0.818 vs T4 0.787 sub, +0.4% qwen3, +7.5% nemo).
- Pin therefore re-set to T2 to match the actual downstream metric.
  T4 results stay in W&B as historical evidence (group
  ablation_full_2026-05-04_00-37-30_9f60152 + lora group
  ablation_lora_finetune_2026-05-05_09-33-31_5df8e20). Validation
  sweeps on T2 land in groups sweep_all6_t2pin_* and sweep_lora_t2pin_*.

ASYM primary metric:  mean of MRR per_relation_strict for subclass and superclass
SYM primary metric:   MRR strict.equivalence

These pins define what 'B=on' means in the main ablation. When B=off,
the matchers fall back to SUBB_DEFAULT_* (the pre-Sub-B configuration).
"""

# B=on — pinned for the Stage-2-Reranker primary metric R@20 per_relation_strict.subclass
SUBB_PIN_ASYM = ("description_one_gen", "T2")  # R@20 sub aggregate = 0.818 (vs T4 0.787)
SUBB_PIN_SYM  = ("description_basic", "S1")    # MRR strict.equivalence = 0.8285

# B=off — pre-Sub-B baseline (used in main ablation when B is disabled)
SUBB_DEFAULT_ASYM = ("description_one_gen", "T1")  # MRR avg(sub,sup) = 0.4125
SUBB_DEFAULT_SYM  = ("description_one_gen", "S1")  # MRR strict.equivalence = 0.8007
