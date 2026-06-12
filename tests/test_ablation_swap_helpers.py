"""Regression tests for the cluster-gating helpers of scripts/ablation_swap.py:
the d11c97e identity check (exit-3 gate, row-set + stored-metrics comparison),
the pooled coverage guard (exit-4 gate), the provenance crosstab, and the
per-query list extraction feeding per_directed_query."""

from __future__ import annotations

import json
from pathlib import Path

from ablation_swap import (
    GUARD_MAX_DROP,
    _guard_violations,
    _identity_check,
    _pool_coverage,
    _provenance_crosstab,
    _query_lists,
)
from Alignment import Alignment
from Correspondence import Correspondence
from swap_retrieval import passes_from_alignment

S1, S2 = "http://src/A", "http://src/B"
T1, T2 = "http://tgt/X", "http://tgt/Y"


def _passes():
    s_align = Alignment()
    s_align.add(Correspondence(S1, T1, "<", 0.9))
    s_align.add(Correspondence(S1, T2, ">", 0.8))
    t_align = Alignment()
    t_align.add(Correspondence(T2, S2, "<", 0.7))
    return {
        **passes_from_alignment(s_align, query_side="source"),
        **passes_from_alignment(t_align, query_side="target"),
    }


# Legacy per_relation_strict of the _passes() baseline subset: one '<' and one
# '>' prediction for S1 — values here only need to be self-consistent.
LEGACY_PRS = {"subclass": {1: 1.0, 5: 1.0}, "superclass": {1: 0.0, 5: 1.0}}


def _write_ablbi_dir(d: Path, rows, prs=None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    with (d / "predictions.tsv").open("w", encoding="utf-8") as f:
        f.write("source_uri\ttarget_uri\trelation\tscore\n")
        for s, t, rel, sc in rows:
            f.write(f"{s}\t{t}\t{rel}\t{sc:.6f}\n")
    if prs is not None:
        (d / "metrics.json").write_text(json.dumps(
            {"recall_at_k": {"per_relation_strict":
                {rel: {str(k): v for k, v in by_k.items()} for rel, by_k in prs.items()}}}))
    return d


# ------------------------------------------------------------ identity check

def test_identity_check_ok_rows_only(tmp_path):
    d = _write_ablbi_dir(tmp_path / "run", [(S1, T1, "<", 0.9), (S1, T2, ">", 0.8)])
    out = _identity_check(_passes(), d, legacy_prs=None)
    assert out["status"] == "ok"
    assert out["metrics_check"] == "absent"
    assert out["n_rows"] == 2


def test_identity_check_ok_with_matching_stored_metrics(tmp_path):
    d = _write_ablbi_dir(tmp_path / "run", [(S1, T1, "<", 0.9), (S1, T2, ">", 0.8)],
                         prs=LEGACY_PRS)
    out = _identity_check(_passes(), d, legacy_prs=LEGACY_PRS)
    assert out["status"] == "ok"
    assert out["metrics_check"] == "ok"


def test_identity_check_detects_stored_metric_drift(tmp_path):
    drifted = {"subclass": {1: 1.0, 5: 1.0}, "superclass": {1: 0.0, 5: 0.5}}
    d = _write_ablbi_dir(tmp_path / "run", [(S1, T1, "<", 0.9), (S1, T2, ">", 0.8)],
                         prs=drifted)
    out = _identity_check(_passes(), d, legacy_prs=LEGACY_PRS)
    assert out["status"] == "mismatch"
    assert out["metrics_check"] == "mismatch"
    assert abs(out["metric_max_abs_diff"] - 0.5) < 1e-12


def test_identity_check_detects_score_drift(tmp_path):
    d = _write_ablbi_dir(tmp_path / "run", [(S1, T1, "<", 0.9), (S1, T2, ">", 0.81)])
    out = _identity_check(_passes(), d, legacy_prs=None)
    assert out["status"] == "mismatch"
    assert out["n_only_old"] == 0 and out["n_only_new"] == 0
    assert abs(out["max_score_delta_on_common"] - 0.01) < 1e-9


def test_identity_check_detects_row_set_drift(tmp_path):
    d = _write_ablbi_dir(tmp_path / "run", [(S1, T1, "<", 0.9), (S2, T2, ">", 0.5)])
    out = _identity_check(_passes(), d, legacy_prs=None)
    assert out["status"] == "mismatch"
    assert out["n_only_old"] == 1
    assert out["n_only_new"] == 1


def test_identity_check_skipped_when_reference_missing(tmp_path):
    out = _identity_check(_passes(), tmp_path / "missing_run", legacy_prs=None)
    assert out["status"] == "skipped"


# ------------------------------------------------------- pooled guard (exit 4)

def _cov(covered, n):
    return {"covered": covered, "n": n,
            "coverage": (covered / n) if n else None}


def test_pool_coverage_sums_across_datasets():
    acc = {}
    _pool_coverage(acc, "cfg", "v_sym",
                   {"subclass": _cov(8, 10), "superclass": _cov(5, 10),
                    "equivalence": _cov(10, 10)})
    _pool_coverage(acc, "cfg", "v_sym",
                   {"subclass": _cov(2, 10), "superclass": _cov(5, 10),
                    "equivalence": _cov(8, 10)})
    assert acc[("cfg", "v_sym", "subclass")] == {"covered": 10, "n": 20}
    assert acc[("cfg", "v_sym", "equivalence")] == {"covered": 18, "n": 20}


def test_guard_violations_flag_drops_beyond_threshold():
    acc = {}
    # baseline: sub 0.95, eq 1.00 — v_sym: sub 0.94 (ok), eq 0.90 (drop 0.10 > 0.02)
    _pool_coverage(acc, "cfg", "baseline",
                   {"subclass": _cov(95, 100), "superclass": _cov(50, 100),
                    "equivalence": _cov(100, 100)})
    _pool_coverage(acc, "cfg", "v_sym",
                   {"subclass": _cov(94, 100), "superclass": _cov(80, 100),
                    "equivalence": _cov(90, 100)})
    violations = _guard_violations(acc, ["cfg"])
    assert len(violations) == 1
    assert "equivalence" in violations[0] and "v_sym" in violations[0]
    # superclass is the PRIMARY outcome, never a guard — a rise must not flag.
    assert not any("superclass" in v for v in violations)
    assert GUARD_MAX_DROP == 0.02


def test_guard_violations_empty_when_within_threshold():
    acc = {}
    _pool_coverage(acc, "cfg", "baseline",
                   {"subclass": _cov(95, 100), "superclass": _cov(50, 100),
                    "equivalence": _cov(99, 100)})
    _pool_coverage(acc, "cfg", "v_union",
                   {"subclass": _cov(95, 100), "superclass": _cov(90, 100),
                    "equivalence": _cov(98, 100)})
    assert _guard_violations(acc, ["cfg"]) == []


# --------------------------------------------------------------- query lists

def test_query_lists_order_by_rank():
    a = Alignment()
    a.add(Correspondence(S1, T1, "<", 0.4))
    a.add(Correspondence(S1, T2, "<", 0.9))
    rows = passes_from_alignment(a, query_side="source")["s_broader"]
    assert _query_lists(rows, retrieved="target") == {S1: [T2, T1]}

    t = Alignment()
    t.add(Correspondence(T1, S1, "<", 0.3))
    t.add(Correspondence(T1, S2, "<", 0.6))
    t_rows = passes_from_alignment(t, query_side="target")["t_broader"]
    assert _query_lists(t_rows, retrieved="source") == {T1: [S2, S1]}


# ---------------------------------------------------------- provenance crosstab

def test_provenance_crosstab_buckets_by_pass_combination():
    gold = Alignment()
    gold.add(Correspondence(S1, T1, "<", 1.0))   # found by s_broader only
    gold.add(Correspondence(S2, T2, ">", 1.0))   # found by t_broader only
    gold.add(Correspondence(S2, T1, "=", 1.0))   # found by nothing
    crosstab = _provenance_crosstab(_passes(), gold, k=20)
    assert crosstab["s_broader"]["<"] == 1
    assert crosstab["t_broader"][">"] == 1
    assert crosstab["none"]["="] == 1
