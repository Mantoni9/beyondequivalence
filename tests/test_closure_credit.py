"""P3 Closure-credit core (scripts/closure_credit.py). Correctness-critical:
these decide whether the hierarchy-credited precision the thesis reports is
sound. Covers the reflexive-transitive closure, the same-label ancestor/
descendant credit, existential (1:N) recall, cross-label isolation, and the
'credit == strict on an edgeless hierarchy' property behind the identity check."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from closure_credit import build_closure, credited_direction


# --------------------------------------------------------------- build_closure

def test_closure_chain_reflexive_transitive():
    # A ⊑ B ⊑ C
    anc, desc = build_closure([("A", "B"), ("B", "C")])
    assert anc["A"] == frozenset({"A", "B", "C"})
    assert anc["B"] == frozenset({"B", "C"})
    assert anc["C"] == frozenset({"C"})              # reflexive, no parents
    assert desc["C"] == frozenset({"A", "B", "C"})
    assert desc["A"] == frozenset({"A"})


def test_closure_diamond():
    # D ⊑ B, D ⊑ C, B ⊑ A, C ⊑ A
    anc, _ = build_closure([("D", "B"), ("D", "C"), ("B", "A"), ("C", "A")])
    assert anc["D"] == frozenset({"D", "B", "C", "A"})


# ------------------------------------------------- credit == strict (identity)

def test_edgeless_hierarchy_credit_equals_strict():
    # no subClassOf edges -> closure is reflexive-only -> credited must equal strict
    anc, desc = build_closure([])
    preds = {("e", "A"): "<", ("e", "B"): "<", ("e", "C"): ">"}
    gold = {("e", "A"): "<", ("e", "C"): ">"}
    U = {("e", "A"), ("e", "B"), ("e", "C")}
    r = credited_direction(preds, gold, anc, "<", U)
    assert r["credited"] == r["strict"]              # ('<' pred B is a real FP either way)
    assert r["strict"]["precision"] == pytest.approx(0.5)   # A hit, B FP
    r2 = credited_direction(preds, gold, desc, ">", U)
    assert r2["credited"] == r2["strict"]
    assert r2["strict"]["f1"] == pytest.approx(1.0)


# --------------------------------------------------- ancestor / descendant credit

def test_ancestor_credit_for_subclass():
    # gold: e '<' A ; pred: e '<' B where A ⊑ B  -> B is a coarser-but-valid superclass
    anc, _ = build_closure([("A", "B")])
    preds = {("e", "B"): "<"}
    gold = {("e", "A"): "<"}
    U = {("e", "A"), ("e", "B")}
    r = credited_direction(preds, gold, anc, "<", U)
    assert r["strict"]["precision"] == pytest.approx(0.0)
    assert r["credited"]["precision"] == pytest.approx(1.0)
    assert r["credited"]["recall"] == pytest.approx(1.0)
    assert r["fp_resolved"] == 1 and r["fp_resolved_frac"] == pytest.approx(1.0)
    assert r["flip_pairs"] == [("e", "B")]


def test_descendant_credit_for_superclass():
    # gold: e '>' B ; pred: e '>' A where A ⊑ B  -> A is descendant-or-self of B
    _, desc = build_closure([("A", "B")])
    preds = {("e", "A"): ">"}
    gold = {("e", "B"): ">"}
    U = {("e", "A"), ("e", "B")}
    r = credited_direction(preds, gold, desc, ">", U)
    assert r["credited"]["precision"] == pytest.approx(1.0)
    assert r["strict"]["precision"] == pytest.approx(0.0)


def test_wrong_direction_target_not_credited():
    # gold: e '<' A ; pred: e '<' Z where Z is NOT an ancestor of A -> stays FP
    anc, _ = build_closure([("A", "B")])   # Z isolated
    preds = {("e", "Z"): "<"}
    gold = {("e", "A"): "<"}
    U = {("e", "A"), ("e", "Z")}
    r = credited_direction(preds, gold, anc, "<", U)
    assert r["credited"]["precision"] == pytest.approx(0.0)
    assert r["fp_resolved"] == 0


def test_no_cross_label_credit():
    # a '<' pred must NOT credit a '>' gold even if hierarchy would match
    anc, _ = build_closure([("A", "B")])
    preds = {("e", "B"): "<"}
    gold = {("e", "A"): ">"}     # different label
    U = {("e", "A"), ("e", "B")}
    r = credited_direction(preds, gold, anc, "<", U)
    assert r["n_gold"] == 0                       # no '<' gold to credit against
    assert r["credited"]["precision"] == pytest.approx(0.0)


# -------------------------------------------------- existential (1:N) recall

def test_one_prediction_recall_covers_many_gold():
    # gold: e '<' A1, e '<' A2 (both ⊑ B); single pred e '<' B covers BOTH
    anc, _ = build_closure([("A1", "B"), ("A2", "B")])
    preds = {("e", "B"): "<"}
    gold = {("e", "A1"): "<", ("e", "A2"): "<"}
    U = {("e", "A1"), ("e", "A2"), ("e", "B")}
    r = credited_direction(preds, gold, anc, "<", U)
    assert r["credited"]["recall"] == pytest.approx(1.0)   # both gold recall-covered
    assert r["credited"]["precision"] == pytest.approx(1.0)
    assert r["n_gold"] == 2 and r["n_pred"] == 1


def test_exact_hit_stays_strict_and_credited():
    anc, _ = build_closure([("A", "B")])
    preds = {("e", "A"): "<"}
    gold = {("e", "A"): "<"}
    U = {("e", "A")}
    r = credited_direction(preds, gold, anc, "<", U)
    assert r["strict"]["f1"] == pytest.approx(1.0)
    assert r["credited"]["f1"] == pytest.approx(1.0)
    assert r["flip_pairs"] == []
