"""
analyze_matrix.py — Stage-2 multi-model matrix analysis (registered:
docs/stage2_matrix_registration.md). CPU-only; reads single-order run dirs.

Reports (all registered):
  - Macro-F1 over {<,>,=} (PRIMARY) per model per dataset + pooled, with
    bootstrap CIs.
  - FULL 4x4 confusion {<,>,=,none} per model per dataset, none-row P/R/F1
    first-class.
  - per-class P/R/F1; flip_rate_gt + flip_rate_lt; direction-accuracy
    (off-diagonal, explicitly NOT primary).
  - reference floor rows: random-direction-guess + majority-class.
  - McNemar per model-pair on the pinned directional gold sets (+ named-26 g7
    designated subset); bootstrap CIs on Macro-F1.
  - '<'-precision cross-model table → the 3-cause decomposition question
    (same poor '<'-precision across all 4 = structural/gold; a reasoner
    improving it = model-specific).
  - variance: per-class F1 spread across the g3 seeds per reasoner.
  - parse_fail per model (gate) + precision-confound (quantization) column.
  - exports a blind '<'-FP audit TSV per dataset (gold-gap component).

Run manifest: --run MODEL:DATASET[:SEED]=DIR ...  (seed default 42).
Llama g7/g5 reuse the existing baseline dirs (255471, 262089).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Alignment import Alignment
from Correspondence import Correspondence
from evaluation_multiclass import compute_multiclass_metrics
from evaluation_recall import _normalize_relation
from tracks.zenodo_loader import load_subdataset
from matrix_stats import (
    macro_f1, mcnemar, bootstrap_macro_f1_ci,
    random_direction_floor, majority_class_floor,
)

logger = logging.getLogger("analyze_matrix")

QUANT = {"llama": "AWQ-INT4", "mistral": "bf16", "gemma4": "bf16",
         "gpt-oss": "MXFP4 (no BF16 ref)"}
DIRECTIONAL_N = {"g7-literature": 67, "g5-groceries": 85, "g3-text": 541}


def _gold(ds: str) -> dict[tuple[str, str], str]:
    _s, _t, ref = load_subdataset(ds)
    g = {}
    for c in Alignment(str(ref)):
        n = _normalize_relation(c.relation)
        if n:
            g[(c.source, c.target)] = n
    return g


def _load_run(d: Path):
    """Return (candidate_pairs, pred_by_pair, parse_fail, raw_rows)."""
    rows = []
    with (d / "predictions.tsv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    cand = {(r["source_uri"], r["target_uri"]) for r in rows}
    pred_by_pair = {}
    for r in rows:
        kept = r.get("kept") == "True"
        rel = r["predicted_relation"] if kept and r["predicted_relation"] in ("<", ">", "=") else "none"
        pred_by_pair[(r["source_uri"], r["target_uri"])] = rel
    try:
        m = json.loads((d / "metrics.json").read_text())
        pf = m.get("reranker_parse_fail_rate")
    except Exception:
        pf = None
    return cand, pred_by_pair, pf, rows


def _report(ds: str, cand: set, pred_by_pair: dict) -> dict:
    gold = _gold(ds)
    reference = Alignment()
    for (s, t), rel in gold.items():
        reference.add(Correspondence(s, t, rel, 1.0))
    predictions = Alignment()
    for (s, t), rel in pred_by_pair.items():
        if rel in ("<", ">", "="):
            predictions.add(Correspondence(s, t, rel, 1.0))
    rep = compute_multiclass_metrics(reference=reference, predictions=predictions,
                                     candidate_pairs=cand).to_dict()
    # directional gold (gold,pred) for floors/mcnemar/bootstrap
    dir_gold, dir_pred = [], []
    cond_gold, cond_pred = [], []
    for (s, t), grel in gold.items():
        if (s, t) not in cand:
            continue
        prel = pred_by_pair.get((s, t), "none")
        cond_gold.append(grel); cond_pred.append(prel)
        if grel in ("<", ">"):
            dir_gold.append(grel); dir_pred.append(prel)
    return {"report": rep, "gold": gold,
            "dir_gold": dir_gold, "dir_pred": dir_pred,
            "cond_gold": cond_gold, "cond_pred": cond_pred}


def _correct_set(gold, pred_by_pair, cand):
    """{pair: bool correct} over directional gold pairs in the candidate set."""
    out = {}
    for (s, t), grel in gold.items():
        if grel in ("<", ">") and (s, t) in cand:
            out[(s, t)] = (pred_by_pair.get((s, t), "none") == grel)
    return out


def _pc(rep, cls, field):
    return rep.get("per_class", {}).get(cls, {}).get(field)


def _fmt(v, s=".3f"):
    return "—" if v is None else format(v, s)


def main():
    p = argparse.ArgumentParser(description="Stage-2 matrix analysis.")
    p.add_argument("--run", nargs="+", required=True, metavar="MODEL:DATASET[:SEED]=DIR")
    p.add_argument("--out-prefix", default="results/matrix_analysis")
    p.add_argument("--audit-fp-sample", type=int, default=40,
                   help="per-dataset blind '<'-FP audit sample size (gold-gap).")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s")

    # manifest: (model, dataset, seed) -> dir
    runs = {}
    for item in args.run:
        key, _, path = item.partition("=")
        parts = key.split(":")
        model, dataset = parts[0], parts[1]
        seed = int(parts[2]) if len(parts) > 2 else 42
        runs[(model, dataset, seed)] = Path(path)

    models = sorted({m for (m, _d, _s) in runs})
    datasets = sorted({d for (_m, d, _s) in runs})

    # per (model,dataset) at seed 42 (canonical) — load + metrics
    cells = {}        # (model,dataset) -> analysis dict
    correct = {}      # (model,dataset) -> {pair: correct} (directional)
    out = {"models": models, "datasets": datasets, "cells": {}, "mcnemar": {},
           "decomposition": {}, "variance": {}, "floors": {}}
    md = ["# Stage-2 model matrix — reasoner vs non-reasoner direction resolution\n"]
    md.append("Registered: docs/stage2_matrix_registration.md. PRIMARY = Macro-F1 "
              "over {<,>,=}, reranker-conditional, single-order. partOf folded to "
              "none. Direction-accuracy reported but NOT primary.\n")

    for (model, dataset, seed), d in runs.items():
        if seed != 42:
            continue
        cand, pbp, pf, rows = _load_run(d)
        a = _report(dataset, cand, pbp)
        a["parse_fail"] = pf
        a["pred_by_pair"] = pbp
        a["cand"] = cand
        cells[(model, dataset)] = a
        correct[(model, dataset)] = _correct_set(a["gold"], pbp, cand)

    # ---- main table: Macro-F1 (+CI), per-class, flip rates, dir-acc, none-row, parse_fail
    md.append("## Primary table (per model × dataset)\n")
    md.append("| Model | Dataset | Macro-F1 [95% CI] | <-F1 | >-F1 | =-F1 | none-F1 "
              "| flip_gt | flip_lt | dir-acc | parse_fail | quant |")
    md.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for model in models:
        for dataset in datasets:
            a = cells.get((model, dataset))
            if not a:
                continue
            rep = a["report"]
            lo, hi = bootstrap_macro_f1_ci(a["cond_gold"], a["cond_pred"], n_boot=1000, seed=42)
            out["cells"][f"{model}/{dataset}"] = {
                "macro_f1": rep.get("macro_f1"), "ci": [lo, hi],
                "per_class": rep.get("per_class"), "confusion": rep.get("confusion"),
                "flip_rate_gt": rep.get("flip_rate_gt"), "flip_rate_lt": rep.get("flip_rate_lt"),
                "direction_accuracy": rep.get("direction_accuracy"),
                "parse_fail": a["parse_fail"]}
            md.append(f"| {model} | {dataset} | {_fmt(rep.get('macro_f1'))} "
                      f"[{_fmt(lo)},{_fmt(hi)}] | {_fmt(_pc(rep,'<','f1'))} "
                      f"| {_fmt(_pc(rep,'>','f1'))} | {_fmt(_pc(rep,'=','f1'))} "
                      f"| {_fmt(_pc(rep,'none','f1'))} | {_fmt(rep.get('flip_rate_gt'))} "
                      f"| {_fmt(rep.get('flip_rate_lt'))} | {_fmt(rep.get('direction_accuracy'))} "
                      f"| {_fmt(a['parse_fail'])} | {QUANT.get(model,'?')} |")
    md.append("")

    # ---- floor rows per dataset
    md.append("## Reference floors (per dataset; on the conditional gold)\n")
    md.append("| Dataset | random-direction dir-acc | random-direction Macro-F1 | "
              "majority-class Macro-F1 (class) |")
    md.append("| --- | ---: | ---: | --- |")
    for dataset in datasets:
        any_cell = next((cells[(m, dataset)] for m in models if (m, dataset) in cells), None)
        if not any_cell:
            continue
        rdf = random_direction_floor(any_cell["cond_gold"], n_sim=1000, seed=42)
        mcf = majority_class_floor(any_cell["cond_gold"])
        out["floors"][dataset] = {"random_direction": rdf, "majority_class": mcf}
        md.append(f"| {dataset} | {_fmt(rdf.get('direction_accuracy'))} "
                  f"| {_fmt(rdf.get('macro_f1'))} | {_fmt(mcf['macro_f1'])} "
                  f"({mcf['majority_class']}) |")
    md.append("")

    # ---- 4x4 confusion per model per dataset
    md.append("## 4×4 confusion {rows=gold, cols=pred: <,>,=,none} (none-row is first-class)\n")
    for dataset in datasets:
        for model in models:
            a = cells.get((model, dataset))
            if not a:
                continue
            cm = a["report"].get("confusion", {})
            md.append(f"**{model} · {dataset}** "
                      f"(none P/R/F1 {_fmt(_pc(a['report'],'none','precision'))}/"
                      f"{_fmt(_pc(a['report'],'none','recall'))}/{_fmt(_pc(a['report'],'none','f1'))})")
            md.append("| gold↓ pred→ | < | > | = | none |")
            md.append("| --- | ---: | ---: | ---: | ---: |")
            for g in ("<", ">", "=", "none"):
                row = cm.get(g, {}) if isinstance(cm, dict) else {}
                md.append(f"| {g} | " + " | ".join(str(row.get(pp, 0)) for pp in ("<", ">", "=", "none")) + " |")
            md.append("")

    # ---- McNemar per model-pair on directional gold (per dataset)
    md.append("## McNemar (model-vs-model, paired on directional gold; exact binomial)\n")
    for dataset in datasets:
        present = [m for m in models if (m, dataset) in correct]
        if len(present) < 2:
            continue
        md.append(f"**{dataset}** (n_directional={DIRECTIONAL_N.get(dataset,'?')})")
        md.append("| A vs B | A✓B✗ | A✗B✓ | p (McNemar) |")
        md.append("| --- | ---: | ---: | ---: |")
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                A, B = present[i], present[j]
                ca, cb = correct[(A, dataset)], correct[(B, dataset)]
                shared = set(ca) & set(cb)
                b = sum(1 for k in shared if ca[k] and not cb[k])
                c = sum(1 for k in shared if not ca[k] and cb[k])
                mc = mcnemar(b, c)
                out["mcnemar"][f"{A}|{B}|{dataset}"] = mc
                md.append(f"| {A} vs {B} | {b} | {c} | {_fmt(mc['p_value'], '.4f')} |")
        md.append("")

    # ---- <-precision 3-cause decomposition (cross-model)
    md.append("## '<'-precision decomposition question (same across models → "
              "structural/gold; a reasoner improving it → model-specific)\n")
    md.append("| Dataset | " + " | ".join(f"{m} <-prec" for m in models) + " |")
    md.append("| --- | " + " | ".join("---:" for _ in models) + " |")
    for dataset in datasets:
        precs = {}
        cellrow = []
        for m in models:
            a = cells.get((m, dataset))
            v = _pc(a["report"], "<", "precision") if a else None
            precs[m] = v
            cellrow.append(_fmt(v))
        out["decomposition"][dataset] = precs
        vals = [v for v in precs.values() if v is not None]
        spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        flag = "structural (spread<0.10)" if spread < 0.10 else "⚠ model-specific (spread≥0.10)"
        md.append(f"| {dataset} | " + " | ".join(cellrow) + f" |  → {flag}")
    md.append("\n*Gold-gap component quantified by the blind '<'-FP audit TSVs "
              "(see foreign_audit); corrected '<'-precision excludes gold_gap FPs.*\n")

    # ---- variance across g3 seeds per reasoner
    md.append("## Variance — per-class F1 spread across g3 seeds (reasoner noise floor)\n")
    md.append("| Model | dataset | seeds | <-F1 spread | >-F1 spread | =-F1 spread | Macro-F1 spread |")
    md.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    seed_groups = defaultdict(dict)   # (model,dataset) -> {seed: dir}
    for (model, dataset, seed), d in runs.items():
        seed_groups[(model, dataset)][seed] = d
    for (model, dataset), sd in sorted(seed_groups.items()):
        if len(sd) < 2:
            continue
        per_seed = {}
        for seed, d in sd.items():
            cand, pbp, _pf, _rows = _load_run(d)
            rep = _report(dataset, cand, pbp)["report"]
            per_seed[seed] = {c: _pc(rep, c, "f1") for c in ("<", ">", "=")}
            per_seed[seed]["macro"] = rep.get("macro_f1")

        def spread(key):
            vs = [per_seed[s][key] for s in sd if per_seed[s][key] is not None]
            return (max(vs) - min(vs)) if len(vs) > 1 else 0.0
        out["variance"][f"{model}/{dataset}"] = {
            "seeds": sorted(sd), "per_seed": per_seed,
            "spread": {k: spread(k) for k in ("<", ">", "=", "macro")}}
        md.append(f"| {model} | {dataset} | {sorted(sd)} | {_fmt(spread('<'))} "
                  f"| {_fmt(spread('>'))} | {_fmt(spread('='))} | {_fmt(spread('macro'))} |")
    md.append("")

    # ---- blind '<'-FP audit export (gold-gap component)
    audit_dir = Path(args.out_prefix).parent
    import random as _r
    for dataset in datasets:
        fps = []
        for model in models:
            a = cells.get((model, dataset))
            if not a:
                continue
            gold = a["gold"]
            for (s, t), prel in a["pred_by_pair"].items():
                if prel == "<" and gold.get((s, t)) != "<":   # '<'-FP
                    fps.append((model, s, t))
        if not fps:
            continue
        rng = _r.Random(42)
        uniq = sorted({(s, t) for (_m, s, t) in fps})
        sample = rng.sample(uniq, min(args.audit_fp_sample, len(uniq)))
        rng.shuffle(sample)
        ap = audit_dir / f"matrix_ltFP_audit_{dataset}.tsv"
        with ap.open("w", encoding="utf-8") as f:
            f.write("# blind '<'-FP audit: pairs a model predicted '<' that are NOT gold '<'.\n")
            f.write("# judgment: gold_gap (real unlabelled subclass) | not_subclass | unsure\n")
            f.write("row_id\tsource_uri\ttarget_uri\tjudgment\n")
            for s, t in sample:
                rid = hashlib.sha1(f"{dataset}|{s}|{t}".encode()).hexdigest()[:8]
                f.write(f"{rid}\t{s}\t{t}\t\n")
        logger.info("wrote %s (%d of %d unique '<'-FP pairs)", ap, len(sample), len(uniq))

    Path(f"{args.out_prefix}.md").write_text("\n".join(md), encoding="utf-8")
    Path(f"{args.out_prefix}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\n" + "\n".join(md))
    logger.info("written: %s.md + .json", args.out_prefix)


if __name__ == "__main__":
    main()
