"""E15 few-shot exemplar selection (fewshot_exemplars.select_exemplars). The
selection is pre-registered, so determinism + the exact arm label-mix + the A4
mirror semantics (swap slots, invert label, swap origin ontology) are load-bearing."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from fewshot_exemplars import select_exemplars, ARM_SPECS

# a pool with enough of each relation for every arm (>= 2 each)
POOLS = {
    "<": [(f"s<{i}", f"t<{i}") for i in range(10)],
    ">": [(f"s>{i}", f"t>{i}") for i in range(10)],
    "=": [(f"s={i}", f"t={i}") for i in range(10)],
}


def test_a0_is_empty():
    assert select_exemplars(POOLS, "A0", 42) == []


def test_a1_one_subclass():
    s = select_exemplars(POOLS, "A1", 42)
    assert len(s) == 1
    assert s[0]["rel"] == "<" and s[0]["mirrored"] is False
    assert s[0]["source_onto"] == "src" and s[0]["target_onto"] == "tgt"


def test_a2_balanced_three():
    s = select_exemplars(POOLS, "A2", 42)
    assert len(s) == 3
    assert sorted(e["rel"] for e in s) == ["<", "=", ">"]
    assert all(not e["mirrored"] for e in s)


def test_a3_balanced_six():
    s = select_exemplars(POOLS, "A3", 42)
    assert len(s) == 6
    from collections import Counter
    assert Counter(e["rel"] for e in s) == {"<": 2, ">": 2, "=": 2}


def test_a4_mirrored_twelve():
    s = select_exemplars(POOLS, "A4", 42)
    assert len(s) == 12
    base = [e for e in s if not e["mirrored"]]
    mirror = [e for e in s if e["mirrored"]]
    assert len(base) == 6 and len(mirror) == 6
    # each mirror is a base with swapped slots + inverted label + swapped onto
    for b, m in zip(base, mirror):
        assert m["source"] == b["target"] and m["target"] == b["source"]
        assert m["source_onto"] == "tgt" and m["target_onto"] == "src"
        inv = {"<": ">", ">": "<", "=": "="}
        assert m["rel"] == inv[b["rel"]]


def test_a4_mirror_preserves_equivalence_label():
    s = select_exemplars(POOLS, "A4", 7)
    eq_mirror = [e for e in s if e["mirrored"] and e["source"].startswith("t=")]
    assert eq_mirror and all(e["rel"] == "=" for e in eq_mirror)   # '=' inverts to '='


def test_determinism_under_seed():
    assert select_exemplars(POOLS, "A4", 123) == select_exemplars(POOLS, "A4", 123)


def test_different_seed_differs():
    a = select_exemplars(POOLS, "A3", 1)
    b = select_exemplars(POOLS, "A3", 2)
    assert a != b   # overwhelmingly likely with 10-deep pools


def test_pool_too_small_raises():
    thin = {"<": [("a", "b")], ">": [("c", "d")], "=": [("e", "f")]}
    with pytest.raises(ValueError, match="pool"):
        select_exemplars(thin, "A3", 42)   # needs 2 each, only 1 available


def test_unknown_arm_raises():
    with pytest.raises(ValueError, match="unknown arm"):
        select_exemplars(POOLS, "A9", 42)


def test_arm_specs_cover_registered_arms():
    assert set(ARM_SPECS) == {"A0", "A1", "A2", "A3", "A4"}
