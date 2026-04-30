"""
analyze_sweep.py — render Stage-1 sweep results as two pivot tables.

Reads results/subsumption_*/(config.json, metrics.json), filters by W&B
group, and prints:
  - one table for instruction_variant=symmetric
  - one table for instruction_variant=asymmetric
plus a single combined JSON dump (results/sweep_<group>.json) that can be
copied off-cluster for visualisation.

Symmetric columns:    per model × {R@1, R@5, R@10, R@20, MRR} from lax/all.
Asymmetric columns:   per model × {R@K-super, R@K-sub for K in 1,5,10,20,
                       MRR-super, MRR-sub} from per_relation_strict.

Usage:
    python analyze_sweep.py --latest
    python analyze_sweep.py --group sweep_all6_2026-04-30_12-34-56_abc1234
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATASETS = ("mouse-human", "g1-web", "g2-diseases", "g3-text", "g5-groceries", "g7-literature")
MODELS   = ("sbert", "qwen3-embedding-8b", "llama-embed-nemotron-8b")
K_VALUES = (1, 5, 10, 20)


def _discover_runs(results_dir: Path) -> list[dict]:
    runs: list[dict] = []
    for run_dir in sorted(results_dir.glob("subsumption_*")):
        cfg_p = run_dir / "config.json"
        met_p = run_dir / "metrics.json"
        if not (cfg_p.is_file() and met_p.is_file()):
            continue
        cfg = json.loads(cfg_p.read_text())
        met = json.loads(met_p.read_text())
        runs.append({"dir": str(run_dir), "config": cfg, "metrics": met})
    return runs


def _pick_group(runs: list[dict], explicit: str | None) -> str:
    if explicit:
        return explicit
    groups: dict[str, str] = {}
    for r in runs:
        g = r["config"].get("wandb_group")
        ts = r["config"].get("timestamp", "")
        if g and g.startswith("sweep_all6_"):
            if g not in groups or ts > groups[g]:
                groups[g] = ts
    if not groups:
        sys.exit("No sweep_all6_* group found in results/. Pass --group explicitly.")
    latest = max(groups, key=lambda g: groups[g])
    return latest


def _fmt(v) -> str:
    if v is None:
        return "  -  "
    return f"{float(v):.3f}"


def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _render_symmetric(runs_by_dm: dict) -> str:
    headers = ["dataset"]
    for m in MODELS:
        for k in K_VALUES:
            headers.append(f"{m}/R@{k}")
        headers.append(f"{m}/MRR")
    rows = []
    for ds in DATASETS:
        row = [ds]
        for m in MODELS:
            r = runs_by_dm.get((ds, m, "symmetric"))
            if r is None:
                row.extend(["  -  "] * (len(K_VALUES) + 1))
                continue
            recall = _get(r["metrics"], "recall_at_k", "lax", "all", default={})
            mrr    = _get(r["metrics"], "mrr",         "lax", "all")
            for k in K_VALUES:
                row.append(_fmt(recall.get(str(k), recall.get(k))))
            row.append(_fmt(mrr))
        rows.append(row)
    return _markdown_table(headers, rows)


def _render_asymmetric(runs_by_dm: dict) -> str:
    headers = ["dataset"]
    for m in MODELS:
        for k in K_VALUES:
            headers.append(f"{m}/R@{k}-super")
            headers.append(f"{m}/R@{k}-sub")
        headers.append(f"{m}/MRR-super")
        headers.append(f"{m}/MRR-sub")
    rows = []
    for ds in DATASETS:
        row = [ds]
        for m in MODELS:
            r = runs_by_dm.get((ds, m, "asymmetric"))
            if r is None:
                row.extend(["  -  "] * (2 * len(K_VALUES) + 2))
                continue
            recall_sup = _get(r["metrics"], "recall_at_k", "per_relation_strict", "superclass", default={})
            recall_sub = _get(r["metrics"], "recall_at_k", "per_relation_strict", "subclass",   default={})
            mrr_sup    = _get(r["metrics"], "mrr",         "per_relation_strict", "superclass")
            mrr_sub    = _get(r["metrics"], "mrr",         "per_relation_strict", "subclass")
            for k in K_VALUES:
                row.append(_fmt(recall_sup.get(str(k), recall_sup.get(k))))
                row.append(_fmt(recall_sub.get(str(k), recall_sub.get(k))))
            row.append(_fmt(mrr_sup))
            row.append(_fmt(mrr_sub))
        rows.append(row)
    return _markdown_table(headers, rows)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    body = "\n".join("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def main() -> None:
    p = argparse.ArgumentParser(description="Render Stage-1 sweep results.")
    p.add_argument("--results-dir", default="results")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--latest", action="store_true",
                   help="Pick the most recent sweep_all6_* W&B group.")
    g.add_argument("--group", default=None,
                   help="Explicit W&B group name.")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        sys.exit(f"results dir not found: {results_dir}")

    all_runs = _discover_runs(results_dir)
    if not all_runs:
        sys.exit(f"No subsumption_* runs in {results_dir}.")

    group = _pick_group(all_runs, args.group)
    runs = [r for r in all_runs if r["config"].get("wandb_group") == group]
    if not runs:
        sys.exit(f"No runs found for group={group!r}.")

    runs_by_dm: dict[tuple[str, str, str], dict] = {}
    for r in runs:
        cfg = r["config"]
        key = (cfg.get("dataset"), cfg.get("model_arg"), cfg.get("instruction_variant"))
        runs_by_dm[key] = r

    print(f"# Stage-1 sweep group: {group}")
    print(f"# {len(runs)} run(s) found across {len(set(k[0] for k in runs_by_dm))} dataset(s).")
    print()
    print("## Symmetric (Recall@K via lax/all)")
    print(_render_symmetric(runs_by_dm))
    print()
    print("## Asymmetric (Recall@K via per_relation_strict; super then sub)")
    print(_render_asymmetric(runs_by_dm))

    out_path = results_dir / f"sweep_{group}.json"
    payload = {
        "group": group,
        "datasets": list(DATASETS),
        "models": list(MODELS),
        "k_values": list(K_VALUES),
        "runs": [
            {
                "dir": r["dir"],
                "dataset": r["config"].get("dataset"),
                "model_arg": r["config"].get("model_arg"),
                "instruction_variant": r["config"].get("instruction_variant"),
                "git_sha": r["config"].get("git_sha"),
                "git_dirty": r["config"].get("git_dirty"),
                "timestamp": r["config"].get("timestamp"),
                "matcher_runtime_seconds": r["metrics"].get("matcher_runtime_seconds"),
                "recall_at_k": r["metrics"].get("recall_at_k"),
                "mrr": r["metrics"].get("mrr"),
                "score_diagnostics": r["metrics"].get("score_diagnostics"),
                "n_reference_total": r["metrics"].get("n_reference_total"),
                "n_reference_after_filter": r["metrics"].get("n_reference_after_filter"),
                "dropped_relations_breakdown": r["metrics"].get("dropped_relations_breakdown"),
            }
            for r in runs
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print()
    print(f"# Combined JSON written to: {out_path}")


if __name__ == "__main__":
    main()
