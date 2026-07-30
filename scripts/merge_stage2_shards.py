#!/usr/bin/env python3
"""merge_stage2_shards.py — merge Stage-2 shard runs into one matrix cell.

Counterpart of scripts/shard_stage1_tsv.py. Takes the output dirs of the
per-shard Stage-2 runs, concatenates their predictions.tsv (shard-disjoint by
construction — asserted), and recomputes metrics.json + confusion_matrix.tsv
over the merged predictions via the SAME evaluator the runner uses
(evaluation_multiclass.compute_multiclass_metrics), with the same
reconstruction rule as scripts/analyze_matrix._load_run (kept=="True",
relation in {<,>,=}; candidate universe = all audited rows). The merged dir is
a drop-in matrix cell for analyze_matrix.

Usage:
    python scripts/merge_stage2_shards.py --dataset mouse-human \
        --out results/matrix_gpt-oss_mouse-human_seed42_<sha> \
        results/matrix_gpt-oss_mouse-human_seed42_shard1of6_<sha> [...more shards]

    --verify-against DIR   compare the recomputed core metrics against an
                           existing cell's metrics.json (identity harness).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Alignment import Alignment          # noqa: E402
from Correspondence import Correspondence  # noqa: E402
from evaluation_multiclass import (      # noqa: E402
    compute_multiclass_metrics, format_confusion_matrix_tsv,
)
from evaluation_recall import RELATION_NORMALIZATION  # noqa: E402
from tracks.zenodo_loader import load_subdataset  # noqa: E402

CORE_KEYS = ("macro_f1", "micro_f1", "per_class", "n_universe", "n_gold")


def _read_rows(d: Path) -> list[dict]:
    with (d / "predictions.tsv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("shards", nargs="+", help="Stage-2 shard output dirs (order = merge order)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True, help="Merged cell dir (created)")
    p.add_argument("--verify-against", default=None,
                   help="Existing cell dir; compare recomputed core metrics for identity")
    args = p.parse_args()

    shard_dirs = [Path(s) for s in args.shards]
    for d in shard_dirs:
        assert (d / "predictions.tsv").is_file(), f"missing predictions.tsv in {d}"

    # ── concatenate audits, assert shard-disjoint sources ────────────────────
    all_rows: list[dict] = []
    seen_sources: dict[str, str] = {}
    fieldnames: list[str] | None = None
    for d in shard_dirs:
        rows = _read_rows(d)
        if fieldnames is None:
            with (d / "predictions.tsv").open(encoding="utf-8") as f:
                fieldnames = next(csv.reader(f, delimiter="\t"))
        for r in rows:
            s = r["source_uri"]
            owner = seen_sources.setdefault(s, str(d))
            assert owner == str(d), (
                f"source {s} appears in {owner} AND {d} — shards not source-disjoint")
        all_rows.extend(rows)
        print(f"  + {d.name}: {len(rows)} rows")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(all_rows)

    # ── recompute metrics over the merged audit (analyzer reconstruction) ────
    _s, _t, ref_path = load_subdataset(args.dataset)
    reference = Alignment()
    for c in Alignment(str(ref_path)):
        n = RELATION_NORMALIZATION.get(c.relation.strip())
        if n:
            reference.add(Correspondence(c.source, c.target, n, 1.0))

    candidate_pairs = {(r["source_uri"], r["target_uri"]) for r in all_rows}
    predictions = Alignment()
    for r in all_rows:
        if r.get("kept") == "True" and r["predicted_relation"] in ("<", ">", "="):
            predictions.add(Correspondence(
                r["source_uri"], r["target_uri"], r["predicted_relation"], 1.0))

    report = compute_multiclass_metrics(
        reference=reference, predictions=predictions, candidate_pairs=candidate_pairs,
    )
    metrics = report.to_dict()

    canonical_counts: dict[str, int] = {}
    for r in all_rows:
        c = r.get("parsed_canonical", "")
        canonical_counts[c] = canonical_counts.get(c, 0) + 1
    metrics["reranker_canonical_counts"] = canonical_counts
    metrics["reranker_parse_fail_rate"] = (
        canonical_counts.get("parse_fail", 0) / max(1, sum(canonical_counts.values())))

    # runtime = sum of shard runtimes; provenance recorded.
    runtimes, shard_configs = [], []
    for d in shard_dirs:
        try:
            cfg = json.loads((d / "config.json").read_text())
            runtimes.append(cfg.get("runtime", {}).get("stage2_seconds", 0.0))
            shard_configs.append(cfg)
        except Exception:
            runtimes.append(0.0)
    metrics["runtime"] = {"stage2_seconds": sum(runtimes),
                          "shards": {d.name: r for d, r in zip(shard_dirs, runtimes)}}
    metrics["merged_from"] = [str(d) for d in shard_dirs]
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (out / "confusion_matrix.tsv").write_text(format_confusion_matrix_tsv(report.confusion))

    if shard_configs:
        cfg = dict(shard_configs[0])
        cfg["merged_from"] = [str(d) for d in shard_dirs]
        cfg["merge_note"] = ("Cell merged from source-disjoint Stage-1 shards "
                            "(walltime armour); prompts/decoding identical per shard.")
        (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    print(f"merged cell: {out}  rows={len(all_rows)}  "
          f"macro_f1={metrics.get('macro_f1')}  micro_f1={metrics.get('micro_f1')}  "
          f"parse_fail={metrics['reranker_parse_fail_rate']:.4f}")

    # ── optional identity harness ────────────────────────────────────────────
    if args.verify_against:
        ref_m = json.loads((Path(args.verify_against) / "metrics.json").read_text())
        mism = []
        for k in CORE_KEYS:
            if k in ref_m and json.dumps(ref_m.get(k), sort_keys=True) != \
                              json.dumps(metrics.get(k), sort_keys=True):
                mism.append(k)
        if mism:
            print(f"VERIFY: MISMATCH in {mism}")
            for k in mism:
                print(f"  {k}: ref={ref_m.get(k)}  merged={metrics.get(k)}")
            sys.exit(1)
        print(f"VERIFY: IDENTICAL on {[k for k in CORE_KEYS if k in ref_m]}")


if __name__ == "__main__":
    main()
