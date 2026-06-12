"""Unit tests for the swap-ablation evaluation extensions in evaluation_recall:
compute_per_directed_query_recall and compute_pair_coverage. Synthetic gold
covers one '<' pair, one '>' pair and one '=' pair (plus misses), per the
ablation spec. The legacy modes of compute_recall_at_k are not touched.
"""

from __future__ import annotations

from Alignment import Alignment
from Correspondence import Correspondence
from evaluation_recall import (
    compute_pair_coverage,
    compute_per_directed_query_recall,
)

S1, S2, S3, S4 = ("http://src/A", "http://src/B", "http://src/C", "http://src/D")
T1, T2, T3, T4 = ("http://tgt/X", "http://tgt/Y", "http://tgt/Z", "http://tgt/W")


def _gold(*triples) -> Alignment:
    a = Alignment()
    for s, t, rel in triples:
        a.add(Correspondence(s, t, rel, 1.0))
    return a


# ------------------------------------------------- per_directed_query recall

def test_per_directed_query_ranks_sub_in_s_list_and_sup_in_t_list():
    gold = _gold((S1, T1, "<"), (S2, T2, ">"), (S3, T3, "="))
    s_broader_lists = {S1: [T4, T1]}        # '<' gold at rank 2 of s-query list
    t_broader_lists = {T2: [S2, S4]}        # '>' gold at rank 1 of t-query list

    out = compute_per_directed_query_recall(
        gold, s_broader_lists, t_broader_lists, k_values=(1, 5))

    assert out["n"] == {"subclass": 1, "superclass": 1}
    assert out["recall_at_k"]["subclass"] == {1: 0.0, 5: 1.0}
    assert out["recall_at_k"]["superclass"] == {1: 1.0, 5: 1.0}
    assert out["mrr"]["subclass"] == 0.5
    assert out["mrr"]["superclass"] == 1.0


def test_per_directed_query_counts_missing_query_lists_as_misses():
    gold = _gold((S1, T1, "<"), (S2, T2, ">"))
    out = compute_per_directed_query_recall(gold, {}, {}, k_values=(20,))
    assert out["n"] == {"subclass": 1, "superclass": 1}
    assert out["recall_at_k"]["subclass"][20] == 0.0
    assert out["recall_at_k"]["superclass"][20] == 0.0
    assert out["mrr"] == {"subclass": 0.0, "superclass": 0.0}


def test_per_directed_query_normalizes_unicode_relations():
    gold = _gold((S1, T1, "⊑"), (S2, T2, "⊒"))
    out = compute_per_directed_query_recall(
        gold, {S1: [T1]}, {T2: [S2]}, k_values=(1,))
    assert out["recall_at_k"]["subclass"][1] == 1.0
    assert out["recall_at_k"]["superclass"][1] == 1.0


def test_per_directed_query_ignores_equivalence_and_dropped_relations():
    gold = _gold((S1, T1, "="), (S2, T2, "PartOf"))
    out = compute_per_directed_query_recall(gold, {}, {}, k_values=(1,))
    assert out["n"] == {"subclass": 0, "superclass": 0}


# --------------------------------------------------------- pair coverage

def test_pair_coverage_counts_pairs_any_direction_hint():
    gold = _gold((S1, T1, "<"), (S2, T2, ">"), (S3, T3, "="), (S4, T4, ">"))
    candidate_pairs = {(S1, T1), (S2, T2), (S3, T3)}

    cov = compute_pair_coverage(gold, candidate_pairs)

    assert cov["subclass"] == {"n": 1, "covered": 1, "coverage": 1.0}
    assert cov["superclass"] == {"n": 2, "covered": 1, "coverage": 0.5}
    assert cov["equivalence"] == {"n": 1, "covered": 1, "coverage": 1.0}


def test_pair_coverage_handles_empty_relation_bucket():
    gold = _gold((S1, T1, "<"))
    cov = compute_pair_coverage(gold, set())
    assert cov["subclass"] == {"n": 1, "covered": 0, "coverage": 0.0}
    assert cov["superclass"]["n"] == 0
    assert cov["superclass"]["coverage"] is None
