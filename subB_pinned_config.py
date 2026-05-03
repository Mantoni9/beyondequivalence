"""Pinned configurations from Sub-B description-ablation sweep.

Sweep group: subB_descablation_2026-05-03_12-18-11_2d92b24
630 runs total: 2 instruction-aware models × 6 datasets × 5 description methods × 5 templates × 2 modes (asym+sym)
                + 1 baseline model (sbert) × 6 datasets × 5 description methods (sym only)

Selection criterion: MRR strict, averaged over both 8B models and all 6 datasets.
- ASYM primary metric: mean of MRR per_relation_strict for subclass and superclass
- SYM primary metric:  MRR strict.equivalence

These pins define what 'B=on' means in the main ablation. When B=off,
the matchers fall back to SUBB_DEFAULT_* (the pre-Sub-B configuration).
"""

# B=on — best configuration found by Sub-B
SUBB_PIN_ASYM = ("description_one_gen", "T4")  # MRR avg(sub,sup) = 0.4269
SUBB_PIN_SYM  = ("description_basic", "S1")    # MRR strict.equivalence = 0.8285

# B=off — pre-Sub-B baseline (used in main ablation when B is disabled)
SUBB_DEFAULT_ASYM = ("description_one_gen", "T1")  # MRR avg(sub,sup) = 0.4125
SUBB_DEFAULT_SYM  = ("description_one_gen", "S1")  # MRR strict.equivalence = 0.8007
