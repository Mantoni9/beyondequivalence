"""
ensemble_finetune_compare.py — compare LoRA-finetune B+C runs against
the existing baseline B+C runs from the main ablation sweep.

Inputs (auto-discovered from results/abl_*):
  - Baseline group:  --baseline-group <ablation_full_*>
  - Finetune group:  --finetune-group <ablation_lora_finetune_*>
Both filtered to A=turtle, B=sub_b_pin, C=rrf — the B+C cell.

Output: results/lora_compare_<finetune_group>.csv with one row per
(dataset, model) plus an AGGREGATE row averaged across 6 datasets x 2
models. Columns:
  dataset, model, baseline_mrr, finetune_mrr, delta_mrr_pct,
  baseline_R@K, finetune_R@K, delta_R@K_pct  for K in {1,5,10,20}

Verdict block at the end keys on the AGGREGATE delta_mrr_pct:
  > +3 %   -> "Fine-Tuning bringt Lift, Branch-Merge empfohlen"
  ±3 %     -> "Ambivalent — Branch nicht mergen, als Limitation in Thesis"
  < -3 %   -> "Catastrophic forgetting wahrscheinlich; methodisch
              wertvoller Negativ-Befund. Branch nicht mergen."
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _discover_runs(results_dir: Path, group: str, lora_tag_required: str | None):
    """Find all abl_*_A-turtle_B-sub_b_pin_C-rrf_<lora_tag>_*_<sha>* dirs
    in the given group. lora_tag_required is "lora-on" (finetune) or
    "lora-off" (baseline, for comparison) or None (any).
    """
    out = []
    for d in sorted(results_dir.glob("abl_*_A-turtle_B-sub_b_pin_C-rrf_*")):
        cfg_p = d / "config.json"
        met_p = d / "metrics.json"
        if not (cfg_p.is_file() and met_p.is_file()):
            continue
        try:
            cfg = json.loads(cfg_p.read_text())
            met = json.loads(met_p.read_text())
        except Exception:
            continue
        if cfg.get("wandb_group") != group:
            continue
        # Lora-tag filter — older baseline runs (pre-LoRA-patch) have no
        # lora-on/off infix in the run_name. Treat absence as lora-off.
        run_name = cfg.get("run_name", d.name)
        run_tag = "lora-on" if "_lora-on_" in run_name else "lora-off"
        if lora_tag_required and run_tag != lora_tag_required:
            continue
        out.append({"dir": d, "config": cfg, "metrics": met})
    return out


def _index_by_md(runs):
    """Index runs by (model_arg, dataset). Warn on duplicates."""
    out: dict = {}
    for r in runs:
        cfg = r["config"]
        model = cfg.get("model_arg") or cfg.get("model")
        dataset = cfg.get("dataset")
        key = (model, dataset)
        if key in out:
            print(f"WARN: duplicate run for {key} — keeping first.", file=sys.stderr)
            continue
        out[key] = r
    return out


def _metric_subclass(rec, k=None):
    """per_relation_strict.subclass MRR or R@K. K=None -> MRR."""
    if rec is None:
        return None
    m = rec["metrics"]
    if k is None:
        return m.get("mrr", {}).get("per_relation_strict", {}).get("subclass")
    rec_d = m.get("recall_at_k", {}).get("per_relation_strict", {}).get("subclass", {})
    return rec_d.get(k, rec_d.get(str(k)))


def _delta_pct(baseline, finetune):
    if baseline is None or finetune is None:
        return None
    if baseline == 0:
        return None
    return 100.0 * (finetune - baseline) / abs(baseline)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare LoRA-finetune B+C runs vs. baseline B+C runs.")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--baseline-group", required=True,
                   help="W&B group of the baseline ablation_full_* sweep "
                        "(used as B+C reference).")
    p.add_argument("--finetune-group", required=True,
                   help="W&B group of the LoRA-finetune ablation_lora_finetune_* sweep.")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    baseline_runs = _discover_runs(results_dir, args.baseline_group, lora_tag_required=None)
    finetune_runs = _discover_runs(results_dir, args.finetune_group, lora_tag_required="lora-on")
    if not baseline_runs:
        sys.exit(f"No baseline B+C runs in group {args.baseline_group!r}.")
    if not finetune_runs:
        sys.exit(f"No finetune B+C runs in group {args.finetune_group!r}.")
    print(f"Baseline runs: {len(baseline_runs)} in group {args.baseline_group}")
    print(f"Finetune runs: {len(finetune_runs)} in group {args.finetune_group}")

    bs = _index_by_md(baseline_runs)
    ft = _index_by_md(finetune_runs)

    # Cells to evaluate: intersection of (model, dataset) keys.
    keys = sorted(set(bs) & set(ft))
    missing_in_ft = sorted(set(bs) - set(ft))
    missing_in_bs = sorted(set(ft) - set(bs))
    if missing_in_ft:
        print(f"WARN: {len(missing_in_ft)} (model, dataset) pairs missing in finetune group: {missing_in_ft}")
    if missing_in_bs:
        print(f"WARN: {len(missing_in_bs)} pairs missing in baseline group: {missing_in_bs}")

    K_VALUES = (1, 5, 10, 20)
    rows = []
    for (model, dataset) in keys:
        b = bs[(model, dataset)]
        f = ft[(model, dataset)]
        row = {
            "dataset": dataset,
            "model":   model,
            "baseline_mrr": _metric_subclass(b),
            "finetune_mrr": _metric_subclass(f),
            "delta_mrr_pct": _delta_pct(_metric_subclass(b), _metric_subclass(f)),
        }
        for k in K_VALUES:
            bv = _metric_subclass(b, k)
            fv = _metric_subclass(f, k)
            row[f"baseline_R@{k}"]   = bv
            row[f"finetune_R@{k}"]   = fv
            row[f"delta_R@{k}_pct"]  = _delta_pct(bv, fv)
        rows.append(row)

    if not rows:
        sys.exit("No (model, dataset) overlap between baseline and finetune groups.")

    # Aggregate: arithmetic mean per column, ignoring None.
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs)) if xs else None

    # Per-model aggregate — averages each metric column across the
    # 6 datasets WITHIN one model. This is essential when the two
    # models behave bimodally (e.g. LoRA-eval 2026-05-05: nemo +16%,
    # qwen3 -22%, hiding inside a -2.67% global aggregate).
    per_model_rows: list[dict] = []
    models_in_data = sorted({r["model"] for r in rows})
    for m in models_in_data:
        m_rows = [r for r in rows if r["model"] == m]
        pm = {"dataset": f"AGG/{m}", "model": m}
        for col in rows[0].keys():
            if col in ("dataset", "model"):
                continue
            pm[col] = _mean(r[col] for r in m_rows)
        per_model_rows.append(pm)

    # Global aggregate across all (model, dataset) cells.
    agg = {"dataset": "AGGREGATE", "model": "AGGREGATE"}
    for col in rows[0].keys():
        if col in ("dataset", "model"):
            continue
        agg[col] = _mean(r[col] for r in rows)
    rows.extend(per_model_rows)
    rows.append(agg)

    # CSV out.
    safe = args.finetune_group.replace(":", "_").replace("/", "_")
    csv_path = results_dir / f"lora_compare_{safe}.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    print(f"\nWritten {csv_path}\n")

    # Stdout table.
    cols = ["dataset", "model", "baseline_mrr", "finetune_mrr", "delta_mrr_pct"] \
           + [c for k in K_VALUES for c in (f"baseline_R@{k}", f"finetune_R@{k}", f"delta_R@{k}_pct")]
    widths = [max(8, len(c)) for c in cols]
    fmt = "  ".join(f"{{:>{w}}}" for w in widths)
    print(fmt.format(*cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v) if v is not None else "  -  ")
        print(fmt.format(*cells))

    # Verdict — first per-model, then aggregate. Per-model verdicts are
    # the methodologically meaningful unit when the two models behave
    # bimodally; aggregate is the formal summary that drives the
    # branch-merge decision per the original spec.
    def _verdict_line(delta: float | None) -> str:
        if delta is None:
            return "  Cannot evaluate — delta is None."
        sign = f"{delta:+.2f}%"
        if delta > 3.0:
            return f"  Δ {sign}  → Lift; merge recommended (if isolated)."
        if delta < -3.0:
            return f"  Δ {sign}  → Catastrophic forgetting; do NOT merge."
        return f"  Δ {sign}  → Ambivalent; document as Limitation."

    print()
    print("=== VERDICT — per model (mean across 6 datasets) ===")
    for pm in per_model_rows:
        print(f"  model={pm['model']}")
        print(_verdict_line(pm.get("delta_mrr_pct")))

    delta = rows[-1].get("delta_mrr_pct")
    print()
    print("=== VERDICT — AGGREGATE (formal branch-merge gate) ===")
    if delta is None:
        print("  Cannot evaluate — aggregate delta is None.")
    else:
        print(f"  Δ MRR aggregate = {delta:+.2f}%")
        if delta > 3.0:
            print("  → Fine-Tuning bringt Lift. Branch-Merge empfohlen.")
        elif delta < -3.0:
            print("  → Catastrophic forgetting wahrscheinlich. Methodisch wertvoller "
                  "Negativ-Befund. Branch NICHT mergen.")
        else:
            print("  → Ambivalent. Branch NICHT mergen, als Limitation in Thesis "
                  "diskutieren.")
            # Bimodality alert: per-model deltas with opposite signs are the
            # clearest case where the aggregate verdict misleads.
            deltas = [pm.get("delta_mrr_pct") for pm in per_model_rows
                      if pm.get("delta_mrr_pct") is not None]
            if len(deltas) >= 2 and min(deltas) < -3.0 and max(deltas) > 3.0:
                print("  ⚠  BIMODAL: per-model deltas straddle ±3% in opposite "
                      "directions. The aggregate verdict masks a structural "
                      "split — see per-model section above and report both "
                      "outcomes separately in the Limitations chapter.")


if __name__ == "__main__":
    main()
