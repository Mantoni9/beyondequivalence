"""Tests for the registered candidate-set decision logic (candidate-set
registration 2026-06-12, filed before job-255613 unblinding):
exactly {v_sym, v_3pass, v_union} x Kt in {20, 10}; Kt=5 reporting only;
each candidate at its own budget vs the @20 baseline; absolute volume cap
(1.3x baseline pairs@20) that does NOT shrink with Kt; precedence = highest
>-coverage, ties (delta < 0.01) toward lower volume."""

from __future__ import annotations

from analyze_swap_results import (
    ADOPTION_KTS,
    CANDIDATE_VARIANTS,
    evaluate_candidates,
)


def _bucket(covered, n=1000):
    return {"covered": covered, "n": n}


BASE20 = {"subclass": _bucket(928), "superclass": _bucket(608),
          "equivalence": _bucket(998)}
BASE_PAIRS = 100_000          # absolute volume cap = 130,000


def _sweep(entries):
    """entries: {(variant, kt): (sub_cov, sup_cov, eq_cov, pairs)} on n=1000."""
    pooled, pairs = {}, {}
    for (v, kt), (sub, sup, eq, n_pairs) in entries.items():
        pooled[(v, kt)] = {
            "subclass": _bucket(round(sub * 1000)),
            "superclass": _bucket(round(sup * 1000)),
            "equivalence": _bucket(round(eq * 1000)),
        }
        pairs[(v, kt)] = n_pairs
    return pooled, pairs


def _good(sub=0.93, sup=0.82, eq=0.998, n_pairs=110_000):
    return (sub, sup, eq, n_pairs)


def _full_grid(overrides=None):
    entries = {(v, kt): _good() for v in CANDIDATE_VARIANTS for kt in (5, 10, 20)}
    if overrides:
        entries.update(overrides)
    return entries


def test_exactly_six_candidates_and_kt5_never_adoptable():
    pooled, pairs = _sweep(_full_grid())
    candidates, winner, band = evaluate_candidates(pooled, pairs, BASE20, BASE_PAIRS)
    assert set(candidates) == {f"{v}@t{kt}" for v in CANDIDATE_VARIANTS
                               for kt in ADOPTION_KTS}
    assert len(candidates) == 6
    assert not any(cid.endswith("@t5") for cid in candidates)
    assert winner is not None


def test_winner_is_highest_sup_among_guard_passers():
    pooled, pairs = _sweep(_full_grid({
        # best sup but breaches the absolute volume cap -> not adoptable
        ("v_union", 20): (0.93, 0.90, 0.998, 200_000),
        ("v_3pass", 20): (0.93, 0.85, 0.998, 120_000),
        ("v_sym", 20):   (0.93, 0.70, 0.998, 105_000),
    }))
    candidates, winner, band = evaluate_candidates(pooled, pairs, BASE20, BASE_PAIRS)
    assert candidates["v_union@t20"]["passes"] is False
    assert winner == "v_3pass@t20"
    assert band == "SOLID"     # 0.85 >= 0.80 and delta +0.242 >= 0.05


def test_tie_breaks_toward_lower_volume():
    pooled, pairs = _sweep(_full_grid({
        ("v_3pass", 20): (0.93, 0.858, 0.998, 125_000),
        ("v_sym", 10):   (0.93, 0.852, 0.998, 108_000),   # within 0.01 -> tie
        ("v_sym", 20):   (0.93, 0.700, 0.998, 115_000),
        ("v_3pass", 10): (0.93, 0.700, 0.998, 118_000),
        ("v_union", 20): (0.93, 0.700, 0.998, 128_000),
        ("v_union", 10): (0.93, 0.700, 0.998, 120_000),
    }))
    _c, winner, _b = evaluate_candidates(pooled, pairs, BASE20, BASE_PAIRS)
    assert winner == "v_sym@t10"


def test_volume_cap_is_absolute_and_does_not_shrink_with_kt():
    pooled, pairs = _sweep(_full_grid({
        ("v_union", 10): (0.93, 0.88, 0.998, 129_999),    # just under the @20-derived cap
    }))
    candidates, winner, _b = evaluate_candidates(pooled, pairs, BASE20, BASE_PAIRS)
    assert candidates["v_union@t10"]["passes"] is True
    assert winner == "v_union@t10"


def test_guard_drops_evaluated_at_own_budget_vs_at20_baseline():
    pooled, pairs = _sweep(_full_grid({
        # '<' coverage 0.90 vs baseline 0.928 -> drop 0.028 > 0.02 -> guard FAIL
        ("v_sym", 10): (0.90, 0.99, 0.998, 105_000),
    }))
    candidates, winner, _b = evaluate_candidates(pooled, pairs, BASE20, BASE_PAIRS)
    assert candidates["v_sym@t10"]["passes"] is False
    assert winner != "v_sym@t10"


def test_reverse_when_no_candidate_passes_guards():
    entries = {(v, kt): (0.93, 0.85, 0.90, 110_000)      # '=' drop 0.098 everywhere
               for v in CANDIDATE_VARIANTS for kt in (5, 10, 20)}
    pooled, pairs = _sweep(entries)
    _c, winner, band = evaluate_candidates(pooled, pairs, BASE20, BASE_PAIRS)
    assert winner is None
    assert band == "REVERSE"


def test_structural_guard_flags():
    # With s-side fixed @20, v_3pass/v_union are supersets of the baseline
    # pass set at ANY Kt — their '<'/'=' guards pass by construction and
    # must be marked structural; only v_sym's are empirically live.
    pooled, pairs = _sweep(_full_grid())
    candidates, _w, _b = evaluate_candidates(pooled, pairs, BASE20, BASE_PAIRS)
    for c in candidates.values():
        assert c["guards_structural"] == (c["variant"] != "v_sym")
        assert c["structural_violation"] is False


def test_structural_violation_flagged_on_impossible_drop():
    # A positive '<' drop on a superset variant is impossible by construction
    # — if it appears, the data is corrupt and must be flagged loudly.
    pooled, pairs = _sweep(_full_grid({
        ("v_3pass", 20): (0.90, 0.85, 0.998, 110_000),   # '<' 0.90 < base 0.928
    }))
    candidates, _w, _b = evaluate_candidates(pooled, pairs, BASE20, BASE_PAIRS)
    assert candidates["v_3pass@t20"]["structural_violation"] is True
    assert candidates["v_sym@t20"]["structural_violation"] is False


def test_band_classification_edges():
    pooled, pairs = _sweep(_full_grid({
        ("v_sym", 20): (0.93, 0.70, 0.998, 105_000),
        ("v_3pass", 20): (0.93, 0.70, 0.998, 106_000),
        ("v_union", 20): (0.93, 0.70, 0.998, 107_000),
        ("v_sym", 10): (0.93, 0.70, 0.998, 104_000),
        ("v_3pass", 10): (0.93, 0.70, 0.998, 104_500),
        ("v_union", 10): (0.93, 0.70, 0.998, 104_900),
    }))
    _c, winner, band = evaluate_candidates(pooled, pairs, BASE20, BASE_PAIRS)
    assert band == "PARTIAL"   # 0.65 <= 0.70 < 0.80

    pooled, pairs = _sweep({(v, kt): (0.93, 0.60, 0.998, 104_000)
                            for v in CANDIDATE_VARIANTS for kt in (5, 10, 20)})
    _c, _w, band = evaluate_candidates(pooled, pairs, BASE20, BASE_PAIRS)
    assert band == "NO_EFFECT"  # < 0.65

    # >= 0.80 but delta < +0.05 vs an already-high baseline -> no switch
    high_base = {"subclass": _bucket(928), "superclass": _bucket(790),
                 "equivalence": _bucket(998)}
    pooled, pairs = _sweep({(v, kt): (0.93, 0.82, 0.998, 104_000)
                            for v in CANDIDATE_VARIANTS for kt in (5, 10, 20)})
    _c, _w, band = evaluate_candidates(pooled, pairs, high_base, BASE_PAIRS)
    assert band == "NO_SWITCH"
