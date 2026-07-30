"""Statistical primitives for the Stage-2 matrix analyzer: McNemar paired test,
bootstrap CI on Macro-F1, and the reference floor rows (random-direction guess,
majority-class). Correctness-critical — these decide whether reasoner numbers
read as real or noise."""

from __future__ import annotations

import math
import pytest

from matrix_stats import (
    macro_f1,
    micro_f1,
    mcnemar,
    bootstrap_macro_f1_ci,
    bootstrap_micro_f1_ci,
    random_direction_floor,
    majority_class_floor,
)


def test_micro_f1_perfect():
    g = ["<", ">", "=", "<"]
    assert micro_f1(g, list(g)) == pytest.approx(1.0)


def test_micro_f1_all_one_class():
    # gold <,>,= ; pred all < : micro TP=1, FP=2, FN=2 → P=R=F1=1/3
    assert micro_f1(["<", ">", "="], ["<", "<", "<"]) == pytest.approx(1.0 / 3, abs=1e-9)


def test_micro_differs_from_macro_on_absent_class():
    # only '<' present+correct: micro=1.0 (pooled), macro=1/3 (absent classes 0)
    g = ["<", "<"]
    assert micro_f1(g, list(g)) == pytest.approx(1.0)
    assert macro_f1(g, list(g)) == pytest.approx(1.0 / 3, abs=1e-9)


def test_micro_f1_none_excluded_from_classes_but_penalises():
    # a '<' gold predicted 'none' → FN for '<' (no class credit), micro drops
    # gold <,> ; pred <,none → TP=1 (<), FP=0, FN=1 (>) → P=1, R=1/2, F1=2/3
    assert micro_f1(["<", ">"], ["<", "none"]) == pytest.approx(2 / 3, abs=1e-9)


def test_bootstrap_micro_ci_brackets_and_deterministic():
    g = ["<", ">", "=", "<", ">", "=", "<", ">"]
    p = ["<", "<", "=", "<", ">", "=", ">", ">"]
    point = micro_f1(g, p)
    lo, hi = bootstrap_micro_f1_ci(g, p, n_boot=500, seed=42)
    assert lo <= point <= hi and 0.0 <= lo <= hi <= 1.0
    assert bootstrap_micro_f1_ci(g, p, n_boot=300, seed=7) == bootstrap_micro_f1_ci(g, p, n_boot=300, seed=7)


# --------------------------------------------------------------- macro_f1

def test_macro_f1_perfect():
    gold = ["<", ">", "=", "<", ">"]
    pred = ["<", ">", "=", "<", ">"]
    assert macro_f1(gold, pred) == pytest.approx(1.0)


def test_macro_f1_all_wrong_one_class():
    # everything predicted '<': only '<' class has any TP
    gold = ["<", ">", "="]
    pred = ["<", "<", "<"]
    # '<': P=1/3 R=1 F1=0.5 ; '>': 0 ; '=': 0 -> macro = 0.5/3
    assert macro_f1(gold, pred) == pytest.approx(0.5 / 3, abs=1e-9)


def test_macro_f1_only_over_present_classes_is_fixed_three():
    # macro is always over {<,>,=} (fixed denominator 3), even if a class absent
    gold = ["<", "<"]
    pred = ["<", "<"]
    # '<' F1=1, '>' and '=' have n=0 -> F1 0 -> macro = 1/3
    assert macro_f1(gold, pred) == pytest.approx(1.0 / 3, abs=1e-9)


# --------------------------------------------------------------- mcnemar

def test_mcnemar_no_disagreement_is_p1():
    # b=c=0 -> no discordant pairs -> p=1.0 (no evidence of difference)
    res = mcnemar(b=0, c=0)
    assert res["p_value"] == pytest.approx(1.0)
    assert res["n_discordant"] == 0


def test_mcnemar_symmetric_discordant_is_nonsignificant():
    res = mcnemar(b=10, c=10)
    assert res["p_value"] > 0.5


def test_mcnemar_strong_asymmetry_is_significant():
    # 0 vs 18 discordant -> exact binomial two-sided p = 2^-18*... tiny
    res = mcnemar(b=0, c=18)
    assert res["p_value"] < 0.01
    assert res["n_discordant"] == 18


def test_mcnemar_exact_binomial_small_n():
    # b=1,c=7: two-sided exact binomial p = 2 * sum_{k=0}^{1} C(8,k) 0.5^8
    # = 2 * (1 + 8) / 256 = 18/256 = 0.0703125
    res = mcnemar(b=1, c=7)
    assert res["p_value"] == pytest.approx(18 / 256, abs=1e-9)


def test_mcnemar_caps_p_at_one():
    res = mcnemar(b=4, c=5)  # near-symmetric -> p computes >1 before cap
    assert res["p_value"] <= 1.0


# ---------------------------------------------------- bootstrap CI

def test_bootstrap_ci_perfect_is_tight_at_one():
    gold = ["<", ">", "="] * 20
    pred = list(gold)
    lo, hi = bootstrap_macro_f1_ci(gold, pred, n_boot=200, seed=42)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_bootstrap_ci_brackets_point_estimate():
    gold = ["<", ">", "=", "<", ">", "=", "<", ">"]
    pred = ["<", "<", "=", "<", ">", "=", ">", ">"]
    point = macro_f1(gold, pred)
    lo, hi = bootstrap_macro_f1_ci(gold, pred, n_boot=500, seed=42)
    assert lo <= point <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_deterministic_under_seed():
    gold = ["<", ">", "=", "<", ">", "="]
    pred = ["<", "<", "=", ">", ">", "="]
    a = bootstrap_macro_f1_ci(gold, pred, n_boot=300, seed=7)
    b = bootstrap_macro_f1_ci(gold, pred, n_boot=300, seed=7)
    assert a == b


# ---------------------------------------------------- floor rows

def test_majority_class_floor_picks_most_frequent():
    gold = ["=", "=", "=", "<", ">"]   # '=' is majority (3/5)
    res = majority_class_floor(gold)
    assert res["majority_class"] == "="
    # predicting '=' for all: '=' F1 = 2*P*R/(P+R), P=3/5 R=1 -> 0.75; others 0
    assert res["macro_f1"] == pytest.approx(0.75 / 3, abs=1e-9)


def test_random_direction_floor_accuracy_near_half():
    # 100 directional gold pairs (mix < and >); random <-or-> guess -> dir-acc ~0.5
    gold = ["<"] * 50 + [">"] * 50
    res = random_direction_floor(gold, n_sim=200, seed=42)
    assert res["direction_accuracy"] == pytest.approx(0.5, abs=0.05)


def test_random_direction_floor_only_directional_pairs():
    # '=' pairs are not part of the directional floor's denominator
    gold = ["<", ">", "=", "<"]
    res = random_direction_floor(gold, n_sim=100, seed=1)
    assert res["n_directional"] == 3


def test_random_direction_floor_deterministic_under_seed():
    gold = ["<", ">", "<", ">", "<"]
    assert (random_direction_floor(gold, n_sim=100, seed=3)
            == random_direction_floor(gold, n_sim=100, seed=3))
