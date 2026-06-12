"""Unit tests for swap_retrieval — re-orientation, rank semantics, variant
assembly, budget capping per query, canonical-collision dedup, TSV roundtrip.

The swapped retrieval makes (t, s, '<') ≡ (s, t, '>') collisions routine;
these tests pin the canonicalisation rules before any cluster run.
"""

from __future__ import annotations

import pytest

from Alignment import Alignment
from Correspondence import Correspondence
from swap_retrieval import (
    PASS_SPECS,
    VARIANTS,
    PassRow,
    assemble_variant,
    candidate_pairs_at_budget,
    candidate_pairs_at_mixed_budget,
    candidate_triples,
    passes_from_alignment,
    read_passes_tsv,
    write_passes_tsv,
)

S1, S2 = "http://src/A", "http://src/B"
T1, T2, T3 = "http://tgt/X", "http://tgt/Y", "http://tgt/Z"


def _align(*cors) -> Alignment:
    a = Alignment()
    for c in cors:
        a.add(Correspondence(*c))
    return a


# ---------------------------------------------------------------- pass specs

def test_pass_specs_cover_all_four_passes_with_canonical_relations():
    assert set(PASS_SPECS) == {"s_broader", "s_narrower", "t_broader", "t_narrower"}
    assert PASS_SPECS["s_broader"].canonical_relation == "<"
    assert PASS_SPECS["s_narrower"].canonical_relation == ">"
    # t-side broader: t ⊑ s'  ⇒ canonical (s', t, '>')
    assert PASS_SPECS["t_broader"].canonical_relation == ">"
    assert PASS_SPECS["t_narrower"].canonical_relation == "<"
    assert VARIANTS["baseline"] == ("s_broader", "s_narrower")
    assert VARIANTS["v_sym"] == ("s_broader", "t_broader")
    # Amendment 2026-06-12: V-3pass keeps the s_narrower cross-rescue.
    assert VARIANTS["v_3pass"] == ("s_broader", "s_narrower", "t_broader")
    assert set(VARIANTS["v_union"]) == set(PASS_SPECS)


# ------------------------------------------------------- source-side passes

def test_source_side_passes_keep_orientation():
    raw = _align((S1, T1, "<", 0.9), (S1, T2, ">", 0.8))
    passes = passes_from_alignment(raw, query_side="source")
    assert set(passes) == {"s_broader", "s_narrower"}
    [b] = passes["s_broader"]
    assert b == PassRow(S1, T1, "<", 0.9, "s_broader", S1, 1)
    [n] = passes["s_narrower"]
    assert n == PassRow(S1, T2, ">", 0.8, "s_narrower", S1, 1)


# ------------------------------------------------- target-side re-orientation

def test_target_side_passes_reorient_to_canonical_orientation():
    # Raw output of the frozen matcher on the TRANSPOSED task: correspondences
    # are keyed (t, s). t-broader hit (t, s, '<') means t ⊑ s -> emit (s, t, '>').
    raw = _align((T1, S1, "<", 0.9), (T1, S2, ">", 0.7))
    passes = passes_from_alignment(raw, query_side="target")
    assert set(passes) == {"t_broader", "t_narrower"}
    [b] = passes["t_broader"]
    assert b == PassRow(S1, T1, ">", 0.9, "t_broader", T1, 1)
    [n] = passes["t_narrower"]
    assert n == PassRow(S2, T1, "<", 0.7, "t_narrower", T1, 1)


def test_rank_is_position_within_query_list_with_deterministic_ties():
    raw = _align(
        (S1, T2, "<", 0.9),
        (S1, T1, "<", 0.9),   # score tie with T2 -> broken by retrieved URI asc
        (S1, T3, "<", 0.8),
        (S2, T1, "<", 0.95),  # separate query -> own rank sequence
    )
    rows = passes_from_alignment(raw, query_side="source")["s_broader"]
    by_pair = {(r.query_uri, r.target_uri): r.rank for r in rows}
    assert by_pair[(S1, T1)] == 1   # tgt/X < tgt/Y lexicographically
    assert by_pair[(S1, T2)] == 2
    assert by_pair[(S1, T3)] == 3
    assert by_pair[(S2, T1)] == 1


def test_target_side_rank_ties_break_on_retrieved_source_uri():
    raw = _align((T1, S2, "<", 0.5), (T1, S1, "<", 0.5))
    rows = passes_from_alignment(raw, query_side="target")["t_broader"]
    by_src = {r.source_uri: r.rank for r in rows}
    assert by_src[S1] == 1
    assert by_src[S2] == 2


def test_passes_from_alignment_rejects_unknown_query_side():
    with pytest.raises(ValueError):
        passes_from_alignment(_align(), query_side="both")


# ------------------------------------------------------------------ variants

def test_assemble_variant_selects_pass_subsets():
    s_passes = passes_from_alignment(
        _align((S1, T1, "<", 0.9), (S1, T2, ">", 0.8)), query_side="source")
    t_passes = passes_from_alignment(
        _align((T3, S1, "<", 0.7), (T3, S2, ">", 0.6)), query_side="target")
    passes = {**s_passes, **t_passes}

    v_sym = assemble_variant(passes, "v_sym")
    assert {r.pass_id for r in v_sym} == {"s_broader", "t_broader"}
    assert all(r.relation in ("<", ">") for r in v_sym)
    # V-sym contains NO narrower-pass rows at all.
    assert not any("narrower" in r.pass_id for r in v_sym)

    v_3pass = assemble_variant(passes, "v_3pass")
    assert {r.pass_id for r in v_3pass} == {"s_broader", "s_narrower", "t_broader"}

    v_union = assemble_variant(passes, "v_union")
    assert {r.pass_id for r in v_union} == set(PASS_SPECS)

    baseline = assemble_variant(passes, "baseline")
    assert {r.pass_id for r in baseline} == {"s_broader", "s_narrower"}

    with pytest.raises(KeyError):
        assemble_variant(passes, "v_unknown")


# ------------------------------------------------------------ budget capping

def test_candidate_pairs_at_budget_caps_per_query_not_per_source():
    # Two DIFFERENT t-queries each retrieve the same source concept at rank 1.
    # Capping per (query, direction) must keep BOTH pairs at k=1 — capping per
    # source_uri would re-introduce the fan-out cap through the back door.
    t_passes = passes_from_alignment(
        _align((T1, S1, "<", 0.9), (T2, S1, "<", 0.8)), query_side="target")
    pairs = candidate_pairs_at_budget(t_passes["t_broader"], k=1)
    assert pairs == {(S1, T1), (S1, T2)}


def test_candidate_pairs_at_budget_respects_rank_cutoff():
    raw = _align((S1, T1, "<", 0.9), (S1, T2, "<", 0.8), (S1, T3, "<", 0.7))
    rows = passes_from_alignment(raw, query_side="source")["s_broader"]
    assert candidate_pairs_at_budget(rows, k=2) == {(S1, T1), (S1, T2)}
    assert candidate_pairs_at_budget(rows, k=50) == {(S1, T1), (S1, T2), (S1, T3)}


def test_candidate_pairs_at_mixed_budget_caps_per_pass():
    # t-side budget sweep (amendment point 2): different K per pass; passes
    # absent from the budget map contribute NOTHING.
    s_passes = passes_from_alignment(
        _align((S1, T1, "<", 0.9), (S1, T2, "<", 0.8), (S1, T3, ">", 0.7)),
        query_side="source")
    t_passes = passes_from_alignment(
        _align((T1, S1, "<", 0.9), (T1, S2, "<", 0.8)), query_side="target")
    passes = {**s_passes, **t_passes}

    pairs = candidate_pairs_at_mixed_budget(passes, {"s_broader": 2, "t_broader": 1})
    # s_broader both ranks in; t_broader only its rank-1 (S1, T1) — already
    # present via s_broader; s_narrower excluded entirely.
    assert pairs == {(S1, T1), (S1, T2)}

    pairs_kt2 = candidate_pairs_at_mixed_budget(passes, {"s_broader": 2, "t_broader": 2})
    assert pairs_kt2 == {(S1, T1), (S1, T2), (S2, T1)}


# ----------------------------------------------------- canonical-key dedup

def test_canonical_collision_between_passes_dedups():
    # s-narrower says (S1, T1, '>') and t-broader (T1, S1, '<') re-orients to
    # the SAME canonical triple (S1, T1, '>') — must collapse to one.
    s_passes = passes_from_alignment(_align((S1, T1, ">", 0.8)), query_side="source")
    t_passes = passes_from_alignment(_align((T1, S1, "<", 0.6)), query_side="target")
    rows = assemble_variant({**s_passes, **t_passes}, "v_union")
    assert candidate_triples(rows) == {(S1, T1, ">")}
    assert candidate_pairs_at_budget(rows, k=20) == {(S1, T1)}


# -------------------------------------------------------------- TSV roundtrip

def test_passes_tsv_roundtrip(tmp_path):
    s_passes = passes_from_alignment(
        _align((S1, T1, "<", 0.9), (S1, T2, ">", 0.8)), query_side="source")
    t_passes = passes_from_alignment(_align((T3, S2, "<", 0.7)), query_side="target")
    rows = assemble_variant({**s_passes, **t_passes}, "v_union")

    path = tmp_path / "passes.tsv"
    write_passes_tsv(path, rows)
    text = path.read_text(encoding="utf-8")
    # Schema contract documented in-file for the future Stage-2 loader.
    assert text.startswith("#")
    assert "cap top-K per (query_uri" in text
    assert "comparable ONLY within" in text

    rows_back = read_passes_tsv(path)
    assert sorted(rows_back) == sorted(rows)
