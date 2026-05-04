"""
ensemble_rrf.py — outer-RRF model ensemble for the B+C ablation cell.

Reads the 12 (qwen3, nemo) x 6 datasets B+C run dirs (run-name pattern
abl_*_A-turtle_B-sub_b_pin_C-rrf_*) and fuses each per-dataset model
pair into a single ranked list via Reciprocal Rank Fusion (k=60).

RRF semantics here:
  fused_score(s, t) = sum_m 1 / (k + rank_m(s, t))
  rank_m = position of t in m's per-source top-K ranking (1-based).
  rank_m = K+1 when the (s, t) pair is NOT in m's top-K — this gives
           every present-in-one-model pair a small penalty contribution
           from the absent model, so pairs that BOTH models rank gain
           a structural advantage (vs. the inner-RRF used in
           MatcherBidirectionalConsolidation, which assigns 0 to absent).

Output:
  results/ensemble_<group>.csv  — wide format, one row per dataset
                                  + AGGREGATE row (mean across 6).
  results/ensemble_<group>.json — same data plus per-dataset diagnostics
                                  (sanity checks, n_pairs counts).

Sanity checks (logged to stdout, also into the JSON):
  - Source-URI set intersection between qwen3 and nemo per dataset.
  - Top-K per-source length distribution per model.
  - Reference path identity per dataset (both runs must point at the
    same reference file — otherwise we'd be comparing apples to oranges).

Usage:
  python ensemble_rrf.py --group ablation_full_<TS>_<SHA>
  python ensemble_rrf.py --latest                     # picks latest ablation_full_*
  python ensemble_rrf.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ensemble_rrf")

DATASETS_ORDER = ("mouse-human", "g1-web", "g2-diseases",
                  "g3-text", "g5-groceries", "g7-literature")
MODELS = ("qwen3-embedding-8b", "llama-embed-nemotron-8b")
K_VALUES = (1, 5, 10, 20)
RRF_K = 60          # Cormack et al. (SIGIR 2009)
TOP_K = 20          # Per-source top-K, matches the ablation runs


def _read_predictions_tsv(path: Path) -> list[dict]:
    """Parse predictions.tsv into a list of dicts with the float-cast score."""
    out = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out.append({
                "source_uri": row["source_uri"],
                "target_uri": row["target_uri"],
                "predicted_relation": row["predicted_relation"],
                "score": float(row["score"]),
            })
    return out


def _build_per_source_ranking(rows: list[dict], top_k: int = TOP_K) -> dict[str, list[tuple[str, int]]]:
    """Group rows by source, sort by (-score, target_uri), keep top_k.
    Returns: source -> [(target, rank), ...] with rank 1-based.
    Same tie-break order as compute_recall_at_k uses internally.
    """
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source_uri"], []).append(r)
    ranking: dict[str, list[tuple[str, int]]] = {}
    for src, items in by_source.items():
        items.sort(key=lambda r: (-r["score"], r["target_uri"], r["predicted_relation"]))
        ranking[src] = [(r["target_uri"], i + 1) for i, r in enumerate(items[:top_k])]
    return ranking


def _rrf_fuse_two_models(
    ranking_a: dict[str, list[tuple[str, int]]],
    ranking_b: dict[str, list[tuple[str, int]]],
    k: int = RRF_K,
    top_k: int = TOP_K,
) -> dict[str, list[tuple[str, float]]]:
    """Fuse two per-source rankings via RRF with explicit "rank=top_k+1
    for absent" penalty. Returns: source -> [(target, fused_score), ...]
    sorted desc by fused_score, kept to top_k per source.
    """
    fused: dict[str, list[tuple[str, float]]] = {}
    all_sources = set(ranking_a) | set(ranking_b)
    penalty = 1.0 / (k + top_k + 1)
    for src in all_sources:
        a_ranks = dict(ranking_a.get(src, []))
        b_ranks = dict(ranking_b.get(src, []))
        all_targets = set(a_ranks) | set(b_ranks)
        scored: list[tuple[str, float]] = []
        for tgt in all_targets:
            score = 0.0
            score += 1.0 / (k + a_ranks[tgt]) if tgt in a_ranks else penalty
            score += 1.0 / (k + b_ranks[tgt]) if tgt in b_ranks else penalty
            scored.append((tgt, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        fused[src] = scored[:top_k]
    return fused


def _alignment_from_fused(fused: dict[str, list[tuple[str, float]]], relation: str = "<"):
    """Build an Alignment from the fused per-source ranking.
    All Correspondences get the same relation; confidence = fused_score.
    """
    from Alignment import Alignment
    from Correspondence import Correspondence
    al = Alignment()
    for src, items in fused.items():
        for tgt, score in items:
            al.add(Correspondence(src, tgt, relation, score))
    return al


def _evaluate(predictions, reference) -> dict:
    """Run compute_recall_at_k and return the per_relation_strict.subclass slice."""
    from evaluation_recall import compute_recall_at_k
    report = compute_recall_at_k(reference, predictions, k_values=K_VALUES)
    rec = report.recall_at_k.get("per_relation_strict", {}).get("subclass", {})
    mrr = report.mrr.get("per_relation_strict", {}).get("subclass")
    return {
        "mrr": mrr,
        "recall": {k: rec.get(k, rec.get(str(k))) for k in K_VALUES},
        "n_reference_total": report.n_reference_total,
        "n_reference_after_filter": report.n_reference_after_filter,
    }


def _discover_b_plus_c_runs(results_dir: Path, group: Optional[str]) -> list[dict]:
    """Find all abl_*_A-turtle_B-sub_b_pin_C-rrf_* run dirs whose
    config.json's wandb_group matches `group` (or any group if None).
    """
    out = []
    pattern = "abl_*_A-turtle_B-sub_b_pin_C-rrf_*"
    for d in sorted(results_dir.glob(pattern)):
        cfg_p = d / "config.json"
        if not cfg_p.is_file():
            continue
        try:
            cfg = json.loads(cfg_p.read_text())
        except Exception:
            continue
        if group is not None and cfg.get("wandb_group") != group:
            continue
        out.append({"dir": d, "config": cfg})
    return out


def _pick_group(runs: list[dict]) -> str:
    groups = {r["config"].get("wandb_group", "") for r in runs}
    candidates = [g for g in groups if g.startswith("ablation_full_") or g.startswith("ablation_smoke_")]
    if not candidates:
        sys.exit("No ablation_full_* / ablation_smoke_* group found in B+C runs.")
    return max(candidates)


def main() -> None:
    p = argparse.ArgumentParser(description="Outer-RRF model ensemble over B+C runs.")
    p.add_argument("--results-dir", default="results")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--group", default=None,
                   help="Explicit ablation W&B group to filter on.")
    g.add_argument("--latest", action="store_true",
                   help="Pick the most recent ablation_full_* group.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s")

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        sys.exit(f"results dir not found: {results_dir}")

    # First pass: discover any B+C runs (no group filter), then resolve group.
    all_bc_runs = _discover_b_plus_c_runs(results_dir, group=None)
    if not all_bc_runs:
        sys.exit(f"No B+C run dirs found in {results_dir}/abl_*_A-turtle_B-sub_b_pin_C-rrf_*")
    group = args.group or _pick_group(all_bc_runs)
    bc_runs = [r for r in all_bc_runs if r["config"].get("wandb_group") == group]
    logger.info("Group: %s — %d B+C run dirs found", group, len(bc_runs))

    # Index by (model, dataset).
    by_md: dict[tuple[str, str], dict] = {}
    for r in bc_runs:
        cfg = r["config"]
        model = cfg.get("model_arg") or cfg.get("model")
        dataset = cfg.get("dataset")
        key = (model, dataset)
        if key in by_md:
            logger.warning("Duplicate run for %s — keeping first, ignoring %s", key, r["dir"])
            continue
        by_md[key] = r

    missing = []
    for m in MODELS:
        for d in DATASETS_ORDER:
            if (m, d) not in by_md:
                missing.append((m, d))
    if missing:
        logger.warning("Missing B+C runs: %s", missing)

    rows: list[dict] = []
    diagnostics: list[dict] = []

    for ds in DATASETS_ORDER:
        if (MODELS[0], ds) not in by_md or (MODELS[1], ds) not in by_md:
            logger.warning("Skipping dataset '%s' — both model runs required, "
                           "have qwen3=%s nemo=%s",
                           ds, (MODELS[0], ds) in by_md, (MODELS[1], ds) in by_md)
            continue

        rec_qwen = by_md[(MODELS[0], ds)]
        rec_nemo = by_md[(MODELS[1], ds)]

        # ── Sanity checks. ──────────────────────────────────────────────
        ref_q = rec_qwen["config"].get("ref_path")
        ref_n = rec_nemo["config"].get("ref_path")
        ref_match = ref_q == ref_n
        if not ref_match:
            logger.warning("Dataset %s: ref_path mismatch q=%s n=%s — diagnostic only",
                           ds, ref_q, ref_n)

        rows_q = _read_predictions_tsv(rec_qwen["dir"] / "predictions.tsv")
        rows_n = _read_predictions_tsv(rec_nemo["dir"] / "predictions.tsv")

        rank_q = _build_per_source_ranking(rows_q, top_k=TOP_K)
        rank_n = _build_per_source_ranking(rows_n, top_k=TOP_K)

        srcs_q = set(rank_q)
        srcs_n = set(rank_n)
        n_intersect = len(srcs_q & srcs_n)
        n_q_only = len(srcs_q - srcs_n)
        n_n_only = len(srcs_n - srcs_q)
        len_q_dist = sorted({len(v) for v in rank_q.values()})
        len_n_dist = sorted({len(v) for v in rank_n.values()})
        logger.info(
            "[%s] sources qwen3=%d nemo=%d intersect=%d q_only=%d n_only=%d "
            "len_distinct qwen3=%s nemo=%s ref_match=%s",
            ds, len(srcs_q), len(srcs_n), n_intersect, n_q_only, n_n_only,
            len_q_dist, len_n_dist, ref_match,
        )

        if n_q_only or n_n_only:
            logger.warning(
                "[%s] source-set mismatch: qwen3-only=%d, nemo-only=%d. "
                "Continuing — RRF treats absent as rank=top_k+1 penalty.",
                ds, n_q_only, n_n_only,
            )

        # ── Fuse via outer RRF. ─────────────────────────────────────────
        fused = _rrf_fuse_two_models(rank_q, rank_n, k=RRF_K, top_k=TOP_K)
        ensemble_pred = _alignment_from_fused(fused, relation="<")

        # ── Reference: load from the qwen3 run's ref_path (assume consistent). ─
        from Alignment import Alignment
        reference = Alignment(ref_q)

        # ── Evaluate ensemble + pull baseline numbers from metrics.json. ─
        ens_eval = _evaluate(ensemble_pred, reference)

        def _baseline(rec):
            m = json.loads((rec["dir"] / "metrics.json").read_text())
            sub = m.get("mrr", {}).get("per_relation_strict", {}).get("subclass")
            rec_sub = m.get("recall_at_k", {}).get("per_relation_strict", {}).get("subclass", {})
            return {
                "mrr": sub,
                "recall": {k: rec_sub.get(k, rec_sub.get(str(k))) for k in K_VALUES},
            }
        q_eval = _baseline(rec_qwen)
        n_eval = _baseline(rec_nemo)

        rows.append({
            "dataset": ds,
            "qwen3_mrr":     q_eval["mrr"],
            "nemo_mrr":      n_eval["mrr"],
            "ensemble_mrr":  ens_eval["mrr"],
            **{f"qwen3_R@{k}":    q_eval["recall"][k] for k in K_VALUES},
            **{f"nemo_R@{k}":     n_eval["recall"][k] for k in K_VALUES},
            **{f"ensemble_R@{k}": ens_eval["recall"][k] for k in K_VALUES},
        })
        diagnostics.append({
            "dataset": ds,
            "ref_path_qwen3": ref_q,
            "ref_path_nemo":  ref_n,
            "ref_match": ref_match,
            "n_sources_qwen3": len(srcs_q),
            "n_sources_nemo":  len(srcs_n),
            "n_sources_intersect": n_intersect,
            "n_sources_qwen3_only": n_q_only,
            "n_sources_nemo_only":  n_n_only,
            "len_per_source_distinct_qwen3": len_q_dist,
            "len_per_source_distinct_nemo":  len_n_dist,
            "n_reference_total": ens_eval["n_reference_total"],
            "n_reference_after_filter": ens_eval["n_reference_after_filter"],
            "n_pairs_fused": sum(len(v) for v in fused.values()),
        })

    if not rows:
        sys.exit("No (qwen3, nemo) dataset pairs to evaluate. Aborting.")

    # ── Aggregate row: simple per-column mean across datasets. ──────────
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs)) if xs else None

    agg = {"dataset": "AGGREGATE"}
    for col in rows[0].keys():
        if col == "dataset":
            continue
        agg[col] = _mean(r[col] for r in rows)
    rows.append(agg)

    # ── Output: CSV + JSON. ─────────────────────────────────────────────
    safe_group = group.replace(":", "_").replace("/", "_")
    csv_path  = results_dir / f"ensemble_{safe_group}.csv"
    json_path = results_dir / f"ensemble_{safe_group}.json"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    payload = {
        "group": group,
        "rrf_k": RRF_K,
        "top_k": TOP_K,
        "models": list(MODELS),
        "datasets": list(DATASETS_ORDER),
        "rows": rows,
        "diagnostics": diagnostics,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # ── Stdout summary. ─────────────────────────────────────────────────
    print()
    print(f"# Ensemble RRF results for group {group}")
    print(f"# CSV  -> {csv_path}")
    print(f"# JSON -> {json_path}")
    print()
    headers = ["dataset", "q_mrr", "n_mrr", "ens_mrr",
               "ens_R@1", "ens_R@5", "ens_R@10", "ens_R@20"]
    fmt_row = "  ".join(f"{{:>{max(8, len(h))}}}" for h in headers)
    print(fmt_row.format(*headers))
    for r in rows:
        cells = [r["dataset"][:13]]
        for col in ("qwen3_mrr", "nemo_mrr", "ensemble_mrr",
                    "ensemble_R@1", "ensemble_R@5", "ensemble_R@10", "ensemble_R@20"):
            v = r.get(col)
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        print(fmt_row.format(*cells))

    # ── Verdict on the AGGREGATE row. ───────────────────────────────────
    agg = rows[-1]
    q, n, e = agg["qwen3_mrr"], agg["nemo_mrr"], agg["ensemble_mrr"]
    if e is None or q is None or n is None:
        print()
        print("Verdict: cannot evaluate (missing aggregate values).")
    else:
        better = e >= max(q, n)
        delta_max = e - max(q, n)
        delta_min = e - min(q, n)
        print()
        print(f"Verdict (AGGREGATE MRR per_relation_strict.subclass):")
        print(f"  qwen3   = {q:.4f}")
        print(f"  nemo    = {n:.4f}")
        print(f"  ensemble= {e:.4f}")
        print(f"  Δ vs max = {delta_max:+.4f}   Δ vs min = {delta_min:+.4f}")
        if better:
            print("  → Ensemble >= max(qwen3, nemo). RRF helps.")
        else:
            print("  → Ensemble < max(qwen3, nemo). Methodisch valides "
                  "Negativ-Resultat: model-asymmetry limits outer-RRF gains.")


if __name__ == "__main__":
    main()
