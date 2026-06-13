"""
matrix_stats.py — statistical primitives for the Stage-2 matrix analyzer.

Pure, dependency-light (stdlib only) so it is unit-testable and runs anywhere:
  - macro_f1: Macro-F1 over the FIXED class set {<,>,=} (denominator always 3,
    so an absent class scores 0 — no silent inflation).
  - mcnemar: paired model-vs-model test (exact two-sided binomial on the
    discordant pairs; correct for the small directional-gold n of 67/85/541).
  - bootstrap_macro_f1_ci: percentile CI on Macro-F1 by resampling pairs.
  - random_direction_floor / majority_class_floor: the registered reference
    floors so reasoner numbers read as GOOD vs just less-bad-than-Llama.
"""

from __future__ import annotations

import random
from math import comb

PRIMARY = ("<", ">", "=")


def _prf(gold: list[str], pred: list[str], cls: str) -> tuple[float, float, float]:
    tp = sum(1 for g, p in zip(gold, pred) if g == cls and p == cls)
    fp = sum(1 for g, p in zip(gold, pred) if g != cls and p == cls)
    fn = sum(1 for g, p in zip(gold, pred) if g == cls and p != cls)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def macro_f1(gold: list[str], pred: list[str], classes: tuple[str, ...] = PRIMARY) -> float:
    """Macro-F1 over a FIXED class set (default {<,>,=})."""
    return sum(_prf(gold, pred, c)[2] for c in classes) / len(classes)


def mcnemar(b: int, c: int) -> dict:
    """Two-sided exact-binomial McNemar on a paired comparison.

    b = # pairs model-A correct, model-B wrong; c = # A wrong, B correct.
    The concordant cells are irrelevant. Exact binomial (n=b+c, p=0.5) is used
    throughout (not the chi-square approximation) because the directional-gold
    n here is small (67/85/541). Returns p_value, n_discordant, and the
    direction of the (uncorrected) effect."""
    n = b + c
    if n == 0:
        return {"p_value": 1.0, "n_discordant": 0, "b": b, "c": c}
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p = min(1.0, 2.0 * tail)
    return {"p_value": p, "n_discordant": n, "b": b, "c": c}


def bootstrap_macro_f1_ci(gold: list[str], pred: list[str], n_boot: int = 1000,
                          seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI on Macro-F1. Resamples (gold, pred) pairs with
    replacement n_boot times; returns the (alpha/2, 1-alpha/2) percentiles."""
    pairs = list(zip(gold, pred))
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        g = [s[0] for s in sample]
        p = [s[1] for s in sample]
        stats.append(macro_f1(g, p))
    stats.sort()
    lo = stats[max(0, int((alpha / 2) * n_boot))]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def majority_class_floor(gold: list[str]) -> dict:
    """Predict the single most-frequent gold class for every pair → Macro-F1.
    The trivial 'always the majority class' reference."""
    from collections import Counter
    if not gold:
        return {"majority_class": None, "macro_f1": 0.0}
    maj = Counter(gold).most_common(1)[0][0]
    pred = [maj] * len(gold)
    return {"majority_class": maj, "macro_f1": macro_f1(gold, pred)}


def random_direction_floor(gold: list[str], n_sim: int = 1000, seed: int = 42) -> dict:
    """Random '<'-or-'>' guess on the DIRECTIONAL gold pairs (the floor for the
    direction question). '=' pairs are excluded from the directional
    denominator. Averages direction-accuracy and Macro-F1 (over the directional
    subset) across n_sim simulations."""
    directional = [g for g in gold if g in ("<", ">")]
    nd = len(directional)
    if nd == 0:
        return {"direction_accuracy": None, "macro_f1": None, "n_directional": 0}
    rng = random.Random(seed)
    accs, f1s = [], []
    for _ in range(n_sim):
        pred = [rng.choice(("<", ">")) for _ in range(nd)]
        accs.append(sum(1 for g, p in zip(directional, pred) if g == p) / nd)
        f1s.append(macro_f1(directional, pred, classes=("<", ">")))
    return {"direction_accuracy": sum(accs) / n_sim,
            "macro_f1": sum(f1s) / n_sim, "n_directional": nd}
