"""
phase0_swap_diagnostics.py — Phase 0 of the Stage-1 swapped-retrieval ablation.

Read-only diagnostics over the d11c97e re-ablation artifacts
(results/ablbi_*_d11c97e/) plus the local gold references. No GPU, no matcher.

0a  Per-dataset x per-direction per_relation_strict Recall@K for
    K in {5, 10, 20, 50}, for the frozen Stage-1 config (Qwen3-noLoRA /
    path_context / T2) and the Nemo+LoRA / path_context / T2 reference.
    The top-50 predictions.tsv lists are re-loaded into Alignments and
    re-scored with evaluation_recall.compute_recall_at_k; every stored
    metrics.json recall/MRR value (all three modes, K in {1, 5, 10, 20})
    is re-validated against the recomputation — an exact match proves the
    TSV round-trip AND that the local gold equals the cluster gold.

0c  Stage-2 cost baseline from the frozen config's TSVs:
      - unique (source, target) candidate pairs per dataset at the Stage-2
        budget cut (top-K per (source, direction), mirroring
        run_stage2_experiment's loader: sort by (-score, target), cap per
        direction, union) at K=20 — the 1.0x volume reference;
      - pair-level gold coverage at budget K in {5, 10, 20, 50} per relation
        (<, >, =): a gold pair counts as covered iff its (s, t) pair appears
        anywhere in the candidate set at that budget, any direction hint;
      - direction-list overlap stats at K=20.

Run (CPU-only, from the repo root):
    conda run -n melt-olala python scripts/phase0_swap_diagnostics.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# scripts/ is on sys.path[0] when run as `python scripts/<file>.py`; add repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Alignment import Alignment
from Correspondence import Correspondence
from evaluation_recall import _normalize_relation, compute_recall_at_k
from tracks.zenodo_loader import load_subdataset

logger = logging.getLogger("phase0_swap_diagnostics")

SHA = "d11c97e"
DATASETS = (
    "mouse-human", "g1-web", "g2-diseases", "g3-text", "g5-groceries", "g7-literature",
)
# (label, run-dir template). Both use the frozen levers A=path_context, B=sub_b_pin (T2).
CONFIGS = (
    ("qwen3-noLoRA", "ablbi_qwen3-embedding-8b_lora-off_A-path_context_B-sub_b_pin_{ds}_" + SHA),
    ("nemo+LoRA", "ablbi_llama-embed-nemotron-8b_lora-on_A-path_context_B-sub_b_pin_{ds}_" + SHA),
)
FROZEN_CONFIG = "qwen3-noLoRA"

EVAL_K_VALUES = (1, 5, 10, 20, 50)   # 1..20 only for metrics.json validation
REPORT_KS = (5, 10, 20, 50)
BUDGET_KS = (5, 10, 20, 50)
VOLUME_K = 20                        # the Stage-2 budget the 1.0x reference uses

GATE_SUP50 = 0.85        # spec Phase 0a: pooled sup@50 >= this -> ranking-depth problem
GATE_COVERAGE20 = 0.80   # amendment: pooled >-coverage@20 >= this -> stop before GPU


def _read_tsv(path: Path) -> list[tuple[str, str, str, float]]:
    rows: list[tuple[str, str, str, float]] = []
    with path.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["source_uri", "target_uri", "relation", "score"]:
            raise ValueError(f"{path}: unexpected header {header!r}")
        for line in f:
            s, t, rel, sc = line.rstrip("\n").split("\t")
            rows.append((s, t, rel, float(sc)))
    return rows


def _alignment_from_rows(rows: list[tuple[str, str, str, float]]) -> Alignment:
    alignment = Alignment()
    for s, t, rel, sc in rows:
        alignment.add(Correspondence(s, t, rel, sc))
    return alignment


def _gold_rel_counts(reference: Alignment) -> dict[str, int]:
    counts = {"<": 0, ">": 0, "=": 0}
    for cor in reference:
        norm = _normalize_relation(cor.relation)
        if norm is not None:
            counts[norm] += 1
    return counts


def _budget_pairs(
    rows: list[tuple[str, str, str, float]], k: int,
) -> tuple[set[tuple[str, str]], dict[str, set[tuple[str, str]]]]:
    """Stage-2 budget cut: top-k per (source, normalised relation), union of directions."""
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for s, t, rel, sc in rows:
        norm = _normalize_relation(rel)
        if norm is None:
            continue
        grouped.setdefault((s, norm), []).append((t, sc))
    pairs: set[tuple[str, str]] = set()
    per_direction: dict[str, set[tuple[str, str]]] = {"<": set(), ">": set()}
    for (s, norm), entries in grouped.items():
        entries.sort(key=lambda e: (-e[1], e[0]))
        for t, _sc in entries[:k]:
            pairs.add((s, t))
            per_direction.setdefault(norm, set()).add((s, t))
    return pairs, per_direction


def _validate_against_metrics(report, metrics: dict) -> tuple[int, float]:
    """Compare every stored recall/MRR value with the recomputation. Returns
    (n values checked, max absolute difference)."""
    n_checked, max_diff = 0, 0.0
    for mode, by_rel in metrics["recall_at_k"].items():
        for rel_label, by_k in by_rel.items():
            for k_str, stored in by_k.items():
                recomputed = report.recall_at_k[mode][rel_label][int(k_str)]
                max_diff = max(max_diff, abs(recomputed - stored))
                n_checked += 1
    for mode, by_rel in metrics["mrr"].items():
        for rel_label, stored in by_rel.items():
            recomputed = report.mrr[mode][rel_label]
            max_diff = max(max_diff, abs(recomputed - stored))
            n_checked += 1
    return n_checked, max_diff


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    )
    repo_root = Path(__file__).resolve().parent.parent
    results_root = repo_root / "results"

    # ---- Load gold once per dataset (local zenodo cache / benchmark.zip). ----
    references: dict[str, Alignment] = {}
    gold_counts: dict[str, dict[str, int]] = {}
    for ds in DATASETS:
        _src, _tgt, ref_path = load_subdataset(ds)
        references[ds] = Alignment(str(ref_path))
        gold_counts[ds] = _gold_rel_counts(references[ds])
        logger.info("gold[%s] (deduped): < %d  > %d  = %d", ds,
                    gold_counts[ds]["<"], gold_counts[ds][">"], gold_counts[ds]["="])

    out: dict = {"sha": SHA, "gold_counts": gold_counts, "0a": {}, "0c": {},
                 "validation": {}, "gates": {}}

    # =========================== Phase 0a ===========================
    rows_cache: dict[tuple[str, str], list] = {}
    for cfg_label, dir_tpl in CONFIGS:
        per_ds: dict[str, dict] = {}
        validation: dict[str, dict] = {}
        for ds in DATASETS:
            run_dir = results_root / dir_tpl.format(ds=ds)
            rows = _read_tsv(run_dir / "predictions.tsv")
            rows_cache[(cfg_label, ds)] = rows
            predictions = _alignment_from_rows(rows)
            report = compute_recall_at_k(
                references[ds], predictions, k_values=EVAL_K_VALUES,
            )
            metrics = json.loads((run_dir / "metrics.json").read_text())

            n_checked, max_diff = _validate_against_metrics(report, metrics)
            gold_match = (
                metrics["gold"]["subclass"] == gold_counts[ds]["<"]
                and metrics["gold"]["superclass"] == gold_counts[ds][">"]
                and metrics["gold"]["equivalence"] == gold_counts[ds]["="]
            )
            validation[ds] = {
                "n_values_checked": n_checked,
                "max_abs_diff": max_diff,
                "gold_counts_match": gold_match,
            }
            if max_diff > 1e-12 or not gold_match:
                logger.warning("VALIDATION DEVIATION [%s/%s]: max_abs_diff=%.3e gold_match=%s",
                               cfg_label, ds, max_diff, gold_match)

            prs = report.recall_at_k["per_relation_strict"]
            n_sub, n_sup = gold_counts[ds]["<"], gold_counts[ds][">"]
            per_ds[ds] = {
                "n_sub": n_sub, "n_sup": n_sup,
                "sub": {k: prs["subclass"][k] for k in REPORT_KS},
                "sup": {k: prs["superclass"][k] for k in REPORT_KS},
                "sub_hits": {k: round(prs["subclass"][k] * n_sub) for k in REPORT_KS},
                "sup_hits": {k: round(prs["superclass"][k] * n_sup) for k in REPORT_KS},
            }

        pooled_n_sub = sum(per_ds[ds]["n_sub"] for ds in DATASETS)
        pooled_n_sup = sum(per_ds[ds]["n_sup"] for ds in DATASETS)
        pooled = {
            "n_sub": pooled_n_sub, "n_sup": pooled_n_sup,
            "sub": {k: sum(per_ds[ds]["sub_hits"][k] for ds in DATASETS) / pooled_n_sub
                    for k in REPORT_KS},
            "sup": {k: sum(per_ds[ds]["sup_hits"][k] for ds in DATASETS) / pooled_n_sup
                    for k in REPORT_KS},
        }
        out["0a"][cfg_label] = {"per_dataset": per_ds, "pooled": pooled}
        out["validation"][cfg_label] = validation

    # =========================== Phase 0c ===========================
    # Frozen config only: volume + pair coverage per relation at budget K.
    coverage: dict[str, dict] = {}
    volume: dict[str, dict] = {}
    for ds in DATASETS:
        rows = rows_cache[(FROZEN_CONFIG, ds)]
        budget_sets = {k: _budget_pairs(rows, k) for k in BUDGET_KS}

        pairs20, per_dir20 = budget_sets[VOLUME_K]
        volume[ds] = {
            "unique_pairs@20": len(pairs20),
            "broader_pairs@20": len(per_dir20["<"]),
            "narrower_pairs@20": len(per_dir20[">"]),
            "direction_overlap@20": len(per_dir20["<"] & per_dir20[">"]),
        }

        cov_ds: dict[str, dict] = {}
        for rel_norm, rel_name in (("<", "sub"), (">", "sup"), ("=", "eq")):
            n_gold = gold_counts[ds][rel_norm]
            covered = {k: 0 for k in BUDGET_KS}
            for cor in references[ds]:
                if _normalize_relation(cor.relation) != rel_norm:
                    continue
                for k in BUDGET_KS:
                    if (cor.source, cor.target) in budget_sets[k][0]:
                        covered[k] += 1
            cov_ds[rel_name] = {
                "n_gold": n_gold,
                "covered": covered,
                "coverage": {k: (covered[k] / n_gold) if n_gold else None for k in BUDGET_KS},
            }
        coverage[ds] = cov_ds

    pooled_cov: dict[str, dict] = {}
    for rel_name in ("sub", "sup", "eq"):
        n_gold = sum(coverage[ds][rel_name]["n_gold"] for ds in DATASETS)
        covered = {k: sum(coverage[ds][rel_name]["covered"][k] for ds in DATASETS)
                   for k in BUDGET_KS}
        pooled_cov[rel_name] = {
            "n_gold": n_gold,
            "covered": covered,
            "coverage": {k: covered[k] / n_gold for k in BUDGET_KS},
        }
    out["0c"] = {
        "config": FROZEN_CONFIG,
        "volume": volume,
        "total_unique_pairs@20": sum(v["unique_pairs@20"] for v in volume.values()),
        "coverage_per_dataset": coverage,
        "coverage_pooled": pooled_cov,
    }

    # =========================== Decision gates ===========================
    pooled_sup50 = out["0a"][FROZEN_CONFIG]["pooled"]["sup"][50]
    pooled_cov20 = pooled_cov["sup"]["coverage"][20]
    out["gates"] = {
        "gate_a_sup50": {"value": pooled_sup50, "threshold": GATE_SUP50,
                         "stop": pooled_sup50 >= GATE_SUP50},
        "gate_b_sup_coverage20": {"value": pooled_cov20, "threshold": GATE_COVERAGE20,
                                  "stop": pooled_cov20 >= GATE_COVERAGE20},
    }

    # =========================== Markdown report ===========================
    lines: list[str] = []
    lines.append(f"# Phase 0 — swapped-retrieval diagnostics over {SHA} artifacts\n")
    lines.append("Frozen levers throughout: A=path_context, B=sub_b_pin (T2), "
                 "MatcherAsymmetricRetrieval, top-50, seed 42.\n")

    lines.append("## Validation (recomputed from predictions.tsv vs stored metrics.json)\n")
    lines.append("| Config | Dataset | values checked | max abs diff | gold counts match |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for cfg_label, _ in CONFIGS:
        for ds in DATASETS:
            v = out["validation"][cfg_label][ds]
            lines.append(f"| {cfg_label} | {ds} | {v['n_values_checked']} "
                         f"| {v['max_abs_diff']:.2e} | {v['gold_counts_match']} |")
    lines.append("")

    for cfg_label, _ in CONFIGS:
        data = out["0a"][cfg_label]
        lines.append(f"## 0a — per_relation_strict Recall@K — {cfg_label} (path_context · T2)\n")
        header = ("| Dataset | n< | n> | " +
                  " | ".join(f"sub@{k}" for k in REPORT_KS) + " | " +
                  " | ".join(f"sup@{k}" for k in REPORT_KS) + " |")
        lines.append(header)
        lines.append("| --- | ---: | ---: | " + " | ".join("---:" for _ in range(8)) + " |")
        for ds in DATASETS:
            d = data["per_dataset"][ds]
            lines.append(f"| {ds} | {d['n_sub']} | {d['n_sup']} | " +
                         " | ".join(_fmt(d["sub"][k]) for k in REPORT_KS) + " | " +
                         " | ".join(_fmt(d["sup"][k]) for k in REPORT_KS) + " |")
        p = data["pooled"]
        lines.append(f"| **pooled** | {p['n_sub']} | {p['n_sup']} | " +
                     " | ".join(f"**{_fmt(p['sub'][k])}**" for k in REPORT_KS) + " | " +
                     " | ".join(f"**{_fmt(p['sup'][k])}**" for k in REPORT_KS) + " |")
        lines.append("")

    lines.append(f"## 0c — Stage-2 cost baseline ({FROZEN_CONFIG}, budget cut per direction)\n")
    lines.append("| Dataset | unique pairs@20 | broader@20 | narrower@20 | direction overlap@20 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for ds in DATASETS:
        v = volume[ds]
        lines.append(f"| {ds} | {v['unique_pairs@20']} | {v['broader_pairs@20']} "
                     f"| {v['narrower_pairs@20']} | {v['direction_overlap@20']} |")
    lines.append(f"| **total** | **{out['0c']['total_unique_pairs@20']}** | | | |")
    lines.append("")

    lines.append("### Pair-level gold coverage at budget K (any direction hint)\n")
    for rel_name, rel_disp in (("sub", "`<` subclass"), ("sup", "`>` superclass"),
                               ("eq", "`=` equivalence")):
        lines.append(f"**{rel_disp}:**\n")
        lines.append("| Dataset | n gold | " + " | ".join(f"cov@{k}" for k in BUDGET_KS) + " |")
        lines.append("| --- | ---: | " + " | ".join("---:" for _ in BUDGET_KS) + " |")
        for ds in DATASETS:
            c = coverage[ds][rel_name]
            vals = " | ".join("—" if c["coverage"][k] is None else _fmt(c["coverage"][k])
                              for k in BUDGET_KS)
            lines.append(f"| {ds} | {c['n_gold']} | {vals} |")
        pc = pooled_cov[rel_name]
        lines.append(f"| **pooled** | {pc['n_gold']} | " +
                     " | ".join(f"**{_fmt(pc['coverage'][k])}**" for k in BUDGET_KS) + " |")
        lines.append("")

    lines.append("## Decision gates\n")
    ga, gb = out["gates"]["gate_a_sup50"], out["gates"]["gate_b_sup_coverage20"]
    lines.append(f"- Gate A (spec 0a): pooled sup@50 = {_fmt(ga['value'])} "
                 f"(threshold {ga['threshold']}) -> {'STOP: ranking-depth problem' if ga['stop'] else 'proceed (structural absence)'}")
    lines.append(f"- Gate B (amendment): pooled >-pair-coverage@20 = {_fmt(gb['value'])} "
                 f"(threshold {gb['threshold']}) -> {'STOP: baseline coverage adequate' if gb['stop'] else 'proceed'}")
    lines.append("")

    md = "\n".join(lines)
    md_path = results_root / f"phase0_swap_diagnostics_{SHA}.md"
    json_path = results_root / f"phase0_swap_diagnostics_{SHA}.json"
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n" + md)
    logger.info("written: %s + %s", md_path, json_path)


if __name__ == "__main__":
    main()
