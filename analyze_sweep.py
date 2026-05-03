"""
analyze_sweep.py — render Stage-1 sweep results as two pivot tables.

Two input modes:
  - default (config.json + metrics.json): reads results/subsumption_*/
    and filters by W&B group.
  - --from-logs <glob>: parses recall/MRR lines straight from
    results/run_*.log when the per-run output dirs are empty (e.g. when a
    sweep finished but the JSON dumps got dropped on a flaky filesystem).

Both modes produce identical output:
  - symmetric:  per model × {R@1, R@5, R@10, R@20, MRR} from lax/all.
  - asymmetric: per model × {R@K-super, R@K-sub for K in 1,5,10,20,
                MRR-super, MRR-sub} from per_relation_strict.
Plus a combined JSON dump that can be copied off-cluster for visualisation.

Usage:
    python analyze_sweep.py --latest
    python analyze_sweep.py --group sweep_all6_2026-04-30_12-34-56_abc1234
    python analyze_sweep.py --from-logs 'results/run_*_2026-04-29_13-02-04_*.log'
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import re
import sys
from pathlib import Path

DATASETS = ("mouse-human", "g1-web", "g2-diseases", "g3-text", "g5-groceries", "g7-literature")
MODELS   = ("sbert", "qwen3-embedding-8b", "llama-embed-nemotron-8b")
K_VALUES = (1, 5, 10, 20)


def _discover_runs(results_dir: Path) -> list[dict]:
    runs: list[dict] = []
    # Single-run output dirs: results/subsumption_<TS>_<run_name>/
    # Sub-B sweep output dirs: results/subB_<run_name>/
    for pattern in ("subsumption_*", "subB_*"):
        for run_dir in sorted(results_dir.glob(pattern)):
            cfg_p = run_dir / "config.json"
            met_p = run_dir / "metrics.json"
            if not (cfg_p.is_file() and met_p.is_file()):
                continue
            try:
                cfg = json.loads(cfg_p.read_text())
                met = json.loads(met_p.read_text())
            except Exception:
                continue
            runs.append({"dir": str(run_dir), "config": cfg, "metrics": met})
    return runs


def _pick_group(runs: list[dict], explicit: str | None) -> str:
    if explicit:
        return explicit
    # Accept any of our known group prefixes; among multiple, pick the latest
    # by timestamp. New prefixes can be added here without touching callers.
    known_prefixes = ("sweep_all6_", "subB_descablation_", "subB_smoke_")
    groups: dict[str, str] = {}
    for r in runs:
        g = r["config"].get("wandb_group")
        ts = r["config"].get("timestamp", "")
        if g and any(g.startswith(p) for p in known_prefixes):
            if g not in groups or ts > groups[g]:
                groups[g] = ts
    if not groups:
        sys.exit(
            "No known sweep group found in results/. "
            f"Looked for prefixes {known_prefixes}. Pass --group explicitly."
        )
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


# ─── Sub-B heatmap rendering ──────────────────────────────────────────────────

SUBB_DESCRIPTIONS = (
    "description_text", "description_basic", "description_one_gen",
    "description_two_gen", "description_three_gen",
)
SUBB_SYM_TEMPLATES  = ("S1", "S2", "S3", "S4", "S5")
SUBB_ASYM_TEMPLATES = ("T1", "T2", "T3", "T4", "T5")


def _index_subB(runs: list[dict]) -> dict:
    """Index Sub-B runs by (model, variant, description, template_id, dataset).

    Returns a dict: {(model, variant, description, template_id): {dataset: run}}
    template_id is None for sbert. Values are the same run dicts as returned
    by _discover_runs (config + metrics).
    """
    out: dict = {}
    for r in runs:
        cfg = r["config"]
        # Only Sub-B runs have description/template_id at this granularity.
        if "description" not in cfg or "instruction_variant" not in cfg:
            continue
        if "template_id" not in cfg:
            continue
        model = cfg.get("model_arg") or cfg.get("model")
        variant = cfg.get("instruction_variant")
        description = cfg.get("description")
        template_id = cfg.get("template_id")
        dataset = cfg.get("dataset")
        if not all([model, variant, description, dataset]):
            continue
        key = (model, variant, description, template_id)
        out.setdefault(key, {})[dataset] = r
    return out


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _heatmap_value(run: dict, *path) -> float | None:
    """Drill into metrics dict via path of keys. Tolerates int/str key drift on K."""
    cur = run["metrics"]
    for k in path:
        if not isinstance(cur, dict):
            return None
        if k in cur:
            cur = cur[k]
        elif str(k) in cur:
            cur = cur[str(k)]
        else:
            return None
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def _render_heatmap(
    title: str,
    rows: tuple[str, ...],
    cols: tuple[str, ...],
    cell: callable,
) -> str:
    """Render a row × col heatmap as a Markdown table. `cell(row, col)` returns
    a float or None (rendered as " - ").
    """
    out = [f"### {title}", ""]
    header = "| description \\\\ template | " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join("-" * max(3, len(c)) for c in [""] + list(cols)) + " |"
    out.append(header)
    out.append(sep)
    for r in rows:
        cells = [r]
        for c in cols:
            v = cell(r, c)
            cells.append("  -  " if v is None else f"{v:.3f}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _render_subB_heatmaps(runs_by_key: dict) -> str:
    """One section per (model, variant). Inside each section: the metrics
    that are methodically meaningful for that variant (per_relation_strict
    @{10,20} for asym; strict.equivalence@{10,20} for sym). Cells are
    averaged across the 6 datasets.
    """
    out: list[str] = []

    # Group keys by (model, variant) for readability.
    by_mv: dict[tuple[str, str], list[tuple]] = {}
    for (model, variant, desc, tid), per_dataset in runs_by_key.items():
        by_mv.setdefault((model, variant), []).append((desc, tid, per_dataset))

    for (model, variant), entries in sorted(by_mv.items()):
        is_sbert = model == "sbert"
        templates: tuple = (None,) if is_sbert else (
            SUBB_SYM_TEMPLATES if variant == "symmetric" else SUBB_ASYM_TEMPLATES
        )
        out.append(f"## {model} / {variant}")
        out.append(f"_Datasets averaged: {len(DATASETS)} (per cell)._")
        out.append("")

        # Metric set differs per variant.
        if variant == "symmetric":
            metric_specs = [
                ("strict.equivalence@10", ("recall_at_k", "strict", "equivalence", 10)),
                ("strict.equivalence@20", ("recall_at_k", "strict", "equivalence", 20)),
                ("MRR strict.equivalence", ("mrr", "strict", "equivalence")),
            ]
        else:
            metric_specs = [
                ("per_relation_strict.subclass@10",   ("recall_at_k", "per_relation_strict", "subclass",   10)),
                ("per_relation_strict.subclass@20",   ("recall_at_k", "per_relation_strict", "subclass",   20)),
                ("per_relation_strict.superclass@10", ("recall_at_k", "per_relation_strict", "superclass", 10)),
                ("per_relation_strict.superclass@20", ("recall_at_k", "per_relation_strict", "superclass", 20)),
                ("MRR per_relation_strict.subclass",   ("mrr", "per_relation_strict", "subclass")),
                ("MRR per_relation_strict.superclass", ("mrr", "per_relation_strict", "superclass")),
            ]

        for metric_name, path in metric_specs:
            def cell(desc: str, tid_label: str, _path=path):
                tid = None if (is_sbert or tid_label == "—") else tid_label
                per_dataset = runs_by_key.get((model, variant, desc, tid), {})
                vals = [_heatmap_value(per_dataset[d], *_path) for d in DATASETS if d in per_dataset]
                return _mean(vals)
            cols = ("—",) if is_sbert else templates
            out.append(_render_heatmap(metric_name, SUBB_DESCRIPTIONS, cols, cell))
            out.append("")

    return "\n".join(out)


def _emit_subB_long_table(runs_by_key: dict) -> list[dict]:
    """Long-format records for downstream pivoting (one record per
    model × variant × description × template × dataset). Includes truncation
    diagnostics so a recall drop can be cross-checked against truncated counts.
    """
    rows: list[dict] = []
    for (model, variant, desc, tid), per_dataset in runs_by_key.items():
        for dataset, r in per_dataset.items():
            m = r["metrics"]
            last = m.get("matcher_last_run_metrics", {})
            rec = {
                "model": model,
                "variant": variant,
                "description": desc,
                "template_id": tid,
                "dataset": dataset,
                "recall_at_k": m.get("recall_at_k"),
                "mrr": m.get("mrr"),
                "n_reference_after_filter": m.get("n_reference_after_filter"),
                "matcher_runtime_seconds": m.get("matcher_runtime_seconds"),
                "tokens_truncated": {
                    k: v for k, v in last.items() if str(k).startswith("tokens_truncated/")
                },
            }
            rows.append(rec)
    return rows


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    body = "\n".join("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


_LOG_FILENAME_RE = re.compile(
    # run_<dataset>_<TS>_<model>_<variant>.log
    # dataset and model can both contain hyphens; variant is the only
    # sym/asym suffix, so anchor on that.
    r"^run_(?P<dataset>.+)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(?P<model>.+)_(?P<variant>symmetric|asymmetric)\.log$"
)
_RECALL_LINE_RE = re.compile(
    r"recall_at_k_(?P<mode>strict|lax|per_relation_strict)/(?P<label>[^/]+)/k=(?P<k>\d+) = (?P<val>[0-9.]+)"
)
_MRR_LINE_RE = re.compile(
    r"mrr_(?P<mode>strict|lax|per_relation_strict)/(?P<label>[^ ]+) = (?P<val>[0-9.]+)"
)


def _parse_log(path: Path) -> dict:
    recall_at_k: dict = {"strict": {}, "lax": {}, "per_relation_strict": {}}
    mrr:         dict = {"strict": {}, "lax": {}, "per_relation_strict": {}}
    text = path.read_text(errors="replace")
    for m in _RECALL_LINE_RE.finditer(text):
        mode, label, k = m["mode"], m["label"], int(m["k"])
        recall_at_k[mode].setdefault(label, {})[k] = float(m["val"])
    for m in _MRR_LINE_RE.finditer(text):
        mrr[m["mode"]][m["label"]] = float(m["val"])
    return {"recall_at_k": recall_at_k, "mrr": mrr}


def _discover_runs_from_logs(pattern: str) -> list[dict]:
    runs: list[dict] = []
    for log_path in sorted(_glob.glob(pattern)):
        p = Path(log_path)
        m = _LOG_FILENAME_RE.match(p.name)
        if not m:
            continue
        runs.append({
            "dir": str(p),
            "config": {
                "dataset": m["dataset"],
                "model_arg": m["model"],
                "instruction_variant": m["variant"],
                "timestamp": m["ts"],
                "wandb_group": f"from-logs:{m['ts']}",
            },
            "metrics": _parse_log(p),
        })
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description="Render Stage-1 sweep results.")
    p.add_argument("--results-dir", default="results")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--latest", action="store_true",
                   help="Pick the most recent sweep_all6_* W&B group.")
    g.add_argument("--group", default=None,
                   help="Explicit W&B group name.")
    g.add_argument("--from-logs", default=None,
                   help="Parse metrics straight from run_*.log files matching this glob "
                        "(use when subsumption_*/ output dirs are empty).")
    args = p.parse_args()

    if args.from_logs:
        runs = _discover_runs_from_logs(args.from_logs)
        if not runs:
            sys.exit(f"No log files matched: {args.from_logs}")
        group = runs[0]["config"]["wandb_group"]
        results_dir = Path(args.results_dir) if Path(args.results_dir).is_dir() else Path(".")
    else:
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
        # config.json carries `model` (from vars(args)); --from-logs carries
        # `model_arg`. Accept either.
        model = cfg.get("model_arg") or cfg.get("model")
        key = (cfg.get("dataset"), model, cfg.get("instruction_variant"))
        runs_by_dm[key] = r

    print(f"# Stage-1 sweep group: {group}")
    print(f"# {len(runs)} run(s) found across {len(set(k[0] for k in runs_by_dm))} dataset(s).")
    print()
    print("## Symmetric (Recall@K via lax/all)")
    print(_render_symmetric(runs_by_dm))
    print()
    print("## Asymmetric (Recall@K via per_relation_strict; super then sub)")
    print(_render_asymmetric(runs_by_dm))

    # Sub-B heatmaps — auto-emit when the group looks like a Sub-B sweep, or
    # when the runs carry description/template_id fields.
    is_subB = group.startswith(("subB_descablation_", "subB_smoke_")) or any(
        "template_id" in r["config"] for r in runs
    )
    subB_long: list[dict] | None = None
    if is_subB:
        subB_index = _index_subB(runs)
        if subB_index:
            print()
            print("# Sub-B description-ablation heatmaps")
            print(f"_(rows = description method, cols = template id; cells averaged across {len(DATASETS)} datasets)_")
            print()
            print(_render_subB_heatmaps(subB_index))
            subB_long = _emit_subB_long_table(subB_index)

    safe_group = group.replace(":", "_").replace("/", "_")
    out_path = results_dir / f"sweep_{safe_group}.json"
    extra_payload: dict = {}
    if subB_long is not None:
        extra_payload["subB_long"] = subB_long
    payload = {
        "group": group,
        "datasets": list(DATASETS),
        "models": list(MODELS),
        "k_values": list(K_VALUES),
        "runs": [
            {
                "dir": r["dir"],
                "dataset": r["config"].get("dataset"),
                "model_arg": r["config"].get("model_arg") or r["config"].get("model"),
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
        **extra_payload,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print()
    print(f"# Combined JSON written to: {out_path}")


if __name__ == "__main__":
    main()
