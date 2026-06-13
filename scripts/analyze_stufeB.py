"""
analyze_stufeB.py — registered Stufe-B analysis (Both-Order-Voting).
CPU-only; reads run_stage2_bothorder output dirs (bothorder_predictions.tsv)
and reconciles B1/B2/B3 OFFLINE.

Registered (docs/stage2_stufeB_registration.md, before any run):
  PRIMARY: Macro-F1 over {<,>,=}, reranker-conditional, dev-pooled + per ds.
  Bands vs v2 baseline dev-pooled Macro-F1 0.334:
    SOLID ≥ +0.10 · SMALL +0.03–0.10 · NO < 0.03 · REVERSE worse ≥0.03 OR =-F1<0.70.
  Also: flip_rate_gt AND flip_rate_lt (a real fix lowers BOTH); =-F1 guard ≥0.70;
  disagreement rate; B1 abstain rate; named-26 full destination distribution;
  parse_fail <5%/order. Precedence among guard-passers: highest Macro-F1;
  tie (Δ<0.01) → higher recall (favors B2/B3 over B1).
  partOf excluded from primary scoring (03.06 protocol) — folded to 'none'.

Integrity: the AB-order canonical predictions (no reconciliation) must equal
the single-order v2 baseline run's predicted_relation on the shared pairs —
a free cross-check that the double-order AB pass reproduces 255471. Reported.

The '<'-heavy guard slice (docs/stufeB_guard_slice_mousehuman.tsv) is read
ONLY for the winning arm's <,>,= F1 as a standalone guard readout — never
enters arm selection.

Usage:
  conda run -n melt-olala python scripts/analyze_stufeB.py \
    --bothorder g7-literature=<dir> g5-groceries=<dir> \
    [--baseline g7-literature=<255471dir> g5-groceries=<R0dir>] \
    [--guard-bothorder mouse-human=<dir>]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Alignment import Alignment
from Correspondence import Correspondence
from evaluation_multiclass import compute_multiclass_metrics
from evaluation_recall import _normalize_relation
from tracks.zenodo_loader import load_subdataset
from stage2_bothorder import RECON_VARIANTS, reconcile

logger = logging.getLogger("analyze_stufeB")

DEV = ("g7-literature", "g5-groceries")
BASELINE_MACRO_F1 = 0.334     # dev-pooled v2, from Stufe A (0c anchor)
EQ_F1_GUARD = 0.70
BANDS = {"SOLID": 0.10, "SMALL": 0.03}
PRECEDENCE_TIE = 0.01
GUARD_SLICE = (Path(__file__).resolve().parent.parent / "docs"
               / "stufeB_guard_slice_mousehuman.tsv")


def _parse_pairs(items):
    out = {}
    for it in items or []:
        ds, _, path = it.partition("=")
        out[ds] = Path(path)
    return out


def _load_bothorder(d: Path):
    rows = []
    with (d / "bothorder_predictions.tsv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    return rows


def _gold(ds):
    _s, _t, ref = load_subdataset(ds)
    g = {}
    for c in Alignment(str(ref)):
        n = _normalize_relation(c.relation)
        if n:
            g[(c.source, c.target)] = n
    return g


def _lp(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _reconciled_alignment(rows, variant):
    al = Alignment()
    for r in rows:
        rel = reconcile(r["ab_canonical"], r["ba_canonical"], variant=variant,
                        ab_lp=_lp(r.get("ab_span_logprob")),
                        ba_lp=_lp(r.get("ba_span_logprob")))
        if rel in ("<", ">", "="):
            al.add(Correspondence(r["source_uri"], r["target_uri"], rel, 1.0))
    return al


def _eqf1(rep):
    for k in ("equivalent", "equivalence", "="):
        if k in rep.get("per_class", {}):
            return rep["per_class"][k].get("f1")
    return None


def _metrics(rows, gold, variant):
    cand_pairs = {(r["source_uri"], r["target_uri"]) for r in rows}
    ref = Alignment()
    for (s, t), rel in gold.items():
        ref.add(Correspondence(s, t, rel, 1.0))
    rep = compute_multiclass_metrics(reference=ref, predictions=_reconciled_alignment(rows, variant),
                                     candidate_pairs=cand_pairs).to_dict()
    # disagreement / abstain
    disagree = sum(1 for r in rows if r["ab_canonical"] != r["ba_canonical"])
    n = max(1, len(rows))
    out = {"macro_f1": rep["macro_f1"], "eq_f1": _eqf1(rep),
           "flip_rate_gt": rep.get("flip_rate_gt"), "flip_rate_lt": rep.get("flip_rate_lt"),
           "disagreement_rate": disagree / n, "report": rep}
    if variant == "B1":
        out["abstain_rate"] = sum(
            1 for r in rows if reconcile(r["ab_canonical"], r["ba_canonical"], variant="B1") == "none"
        ) / n
    return out


def _named26_destinations(rows, baseline_rows, gold):
    """Where do the baseline gold-'>'→'<' flips land under each variant?"""
    if not baseline_rows:
        return None
    bl = {(r["source_uri"], r["target_uri"]): r for r in baseline_rows}
    flip = [(r["source_uri"], r["target_uri"]) for r in baseline_rows
            if gold.get((r["source_uri"], r["target_uri"])) == ">"
            and r.get("kept") == "True" and r.get("predicted_relation") == "<"]
    by = {(r["source_uri"], r["target_uri"]): r for r in rows}
    dest = {}
    for v in RECON_VARIANTS:
        d = {"<": 0, ">": 0, "=": 0, "none": 0, "missing": 0}
        for k in flip:
            r = by.get(k)
            if r is None:
                d["missing"] += 1
                continue
            rel = reconcile(r["ab_canonical"], r["ba_canonical"], variant=v,
                            ab_lp=_lp(r.get("ab_span_logprob")), ba_lp=_lp(r.get("ba_span_logprob")))
            d[rel if rel in ("<", ">", "=") else "none"] += 1
        dest[v] = d
    return {"n_flip": len(flip), "destinations": dest}


def _ab_baseline_integrity(rows, baseline_rows):
    """AB-order canonical preds must match the v2 baseline predicted_relation
    on shared kept pairs (free reproduction check of 255471)."""
    if not baseline_rows:
        return None
    bl = {(r["source_uri"], r["target_uri"]): r["predicted_relation"]
          for r in baseline_rows if r.get("kept") == "True"}
    checked = mismatch = 0
    for r in rows:
        k = (r["source_uri"], r["target_uri"])
        if k in bl and bl[k] in ("<", ">", "="):
            checked += 1
            if r["ab_canonical"] != bl[k]:
                mismatch += 1
    return {"checked": checked, "mismatch": mismatch}


def _pooled(rows_by_ds, golds, variant):
    ref, pred = Alignment(), Alignment()
    cand = set()
    for ds, rows in rows_by_ds.items():
        g = golds[ds]
        for (s, t), rel in g.items():
            ref.add(Correspondence(s, t, rel, 1.0))
        al = _reconciled_alignment(rows, variant)
        for c in al:
            pred.add(c)
        cand |= {(r["source_uri"], r["target_uri"]) for r in rows}
    rep = compute_multiclass_metrics(reference=ref, predictions=pred, candidate_pairs=cand).to_dict()
    return {"macro_f1": rep["macro_f1"], "eq_f1": _eqf1(rep),
            "flip_rate_gt": rep.get("flip_rate_gt"), "flip_rate_lt": rep.get("flip_rate_lt")}


def _band(delta, eq_f1):
    if eq_f1 is not None and eq_f1 < EQ_F1_GUARD:
        return f"REVERSE (=-F1 {eq_f1:.3f} < {EQ_F1_GUARD})"
    if delta <= -BANDS["SMALL"]:
        return "REVERSE"
    if delta >= BANDS["SOLID"]:
        return "SOLID"
    if delta >= BANDS["SMALL"]:
        return "SMALL"
    return "NO"


def _fmt(v, s=".3f"):
    return "—" if v is None else format(v, s)


def main():
    p = argparse.ArgumentParser(description="Stufe-B Both-Order analysis.")
    p.add_argument("--bothorder", nargs="+", required=True, metavar="DS=DIR")
    p.add_argument("--baseline", nargs="+", default=None, metavar="DS=DIR",
                   help="v2 single-order baselines (g7=255471, g5=R0) for "
                        "named-26 + AB integrity.")
    p.add_argument("--guard-bothorder", nargs="+", default=None, metavar="mouse-human=DIR",
                   help="double-order run restricted to the pinned '<'-heavy slice.")
    p.add_argument("--out-prefix", default="results/stufeB_analysis")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s")

    bo_dirs = _parse_pairs(args.bothorder)
    bl_dirs = _parse_pairs(args.baseline)
    rows_by_ds = {ds: _load_bothorder(d) for ds, d in bo_dirs.items()}
    baseline_by_ds = {ds: _load_bothorder_or_pred(d) for ds, d in bl_dirs.items()}
    golds = {ds: _gold(ds) for ds in rows_by_ds}

    out = {"baseline_macro_f1_anchor": BASELINE_MACRO_F1, "per_ds": {}, "pooled": {},
           "named26": {}, "integrity": {}, "guard": None}
    md = ["# Stufe B — Both-Order-Voting: order-invariant directional classification\n"]
    md.append("Registered analysis (docs/stage2_stufeB_registration.md). Primary = "
              "Macro-F1 over {<,>,=}, reranker-conditional; partOf folded to none. "
              "B1 abstain · B2 confidence tie-break · B3 symmetry-grounded.\n")

    # per-dataset + pooled metrics per variant
    md.append("## Per-dataset & dev-pooled metrics per variant\n")
    md.append("| Variant | Dataset | Macro-F1 | =-F1 | flip_gt | flip_lt | disagree | abstain |")
    md.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for v in RECON_VARIANTS:
        for ds, rows in rows_by_ds.items():
            m = _metrics(rows, golds[ds], v)
            out["per_ds"].setdefault(v, {})[ds] = {k: m[k] for k in m if k != "report"}
            md.append(f"| {v} | {ds} | {_fmt(m['macro_f1'])} | {_fmt(m['eq_f1'])} "
                      f"| {_fmt(m['flip_rate_gt'])} | {_fmt(m['flip_rate_lt'])} "
                      f"| {_fmt(m['disagreement_rate'])} | {_fmt(m.get('abstain_rate'))} |")
        pooled = _pooled(rows_by_ds, golds, v)
        out["pooled"][v] = pooled
        delta = (pooled["macro_f1"] or 0) - BASELINE_MACRO_F1
        band = _band(delta, pooled["eq_f1"])
        md.append(f"| **{v}** | **dev-pooled** | **{_fmt(pooled['macro_f1'])}** "
                  f"| {_fmt(pooled['eq_f1'])} | {_fmt(pooled['flip_rate_gt'])} "
                  f"| {_fmt(pooled['flip_rate_lt'])} | Δ={delta:+.3f} → {band} | |")
    md.append("")

    # named-26 destinations + AB integrity (need baseline)
    if baseline_by_ds.get("g7-literature"):
        bl = baseline_by_ds["g7-literature"]
        nd = _named26_destinations(rows_by_ds["g7-literature"], bl, golds["g7-literature"])
        out["named26"] = nd
        md.append(f"## Named-26 flip-set destinations (g7; n={nd['n_flip']})\n")
        md.append("| Variant | →`<` | →`>` | →`=` | →none |")
        md.append("| --- | ---: | ---: | ---: | ---: |")
        for v in RECON_VARIANTS:
            d = nd["destinations"][v]
            md.append(f"| {v} | {d['<']} | {d['>']} | {d['=']} | {d['none']} |")
        md.append("")
        integ = _ab_baseline_integrity(rows_by_ds["g7-literature"], bl)
        out["integrity"]["g7-literature"] = integ
        md.append(f"*AB-order integrity vs v2 baseline (g7): {integ['checked']} shared "
                  f"kept pairs, {integ['mismatch']} mismatches "
                  f"({'AB reproduces 255471' if integ['mismatch']==0 else 'DRIFT — investigate'}).*\n")

    # decision: precedence among guard-passers
    cands = {}
    for v in RECON_VARIANTS:
        pl = out["pooled"][v]
        delta = (pl["macro_f1"] or 0) - BASELINE_MACRO_F1
        band = _band(delta, pl["eq_f1"])
        cands[v] = {"macro_f1": pl["macro_f1"], "delta": delta, "band": band,
                    "passes": band in ("SOLID", "SMALL", "NO") and "REVERSE" not in band}
    passers = {v: c for v, c in cands.items() if c["passes"] and c["band"] in ("SOLID", "SMALL")}
    winner = None
    if passers:
        best = max(c["macro_f1"] for c in passers.values())
        tied = [v for v, c in passers.items() if best - (c["macro_f1"] or 0) < PRECEDENCE_TIE]
        # tie → higher recall: B2/B3 keep more directionals than B1's abstain
        order = {"B2": 0, "B3": 1, "B1": 2}
        winner = sorted(tied, key=lambda v: order.get(v, 9))[0]
    out["winner"] = winner
    md.append("## Decision (registered)\n")
    md.append(f"- Baseline anchor dev-pooled Macro-F1 = {BASELINE_MACRO_F1}.")
    for v in RECON_VARIANTS:
        c = cands[v]
        md.append(f"- {v}: Macro-F1 {_fmt(c['macro_f1'])} (Δ {c['delta']:+.3f}) → {c['band']}")
    md.append(f"- **Winner: {winner or 'NONE — all REVERSE/NO; baseline stays'}** "
              f"(highest Macro-F1 among SOLID/SMALL guard-passers; tie→higher recall).\n")

    # guard slice readout (winning arm only, standalone)
    if args.guard_bothorder and winner:
        gdir = _parse_pairs(args.guard_bothorder).get("mouse-human")
        if gdir:
            grows = _load_bothorder(gdir)
            slice_pairs = _load_guard_slice()
            grows = [r for r in grows if (r["source_uri"], r["target_uri"]) in slice_pairs]
            ggold = {p: "<" for p in slice_pairs}
            gm = _metrics(grows, ggold, winner)
            rep = gm["report"]["per_class"]
            out["guard"] = {"winner": winner, "n": len(grows),
                            "per_class": {k: rep.get(k, {}).get("f1") for k in
                                          ("subclass", "superclass", "equivalent")}}
            md.append(f"## '<'-heavy guard slice readout — winner {winner} "
                      f"(mouse-human, n={len(grows)}; READ-ONLY, not in selection)\n")
            md.append(f"- subclass-F1 {_fmt(out['guard']['per_class'].get('subclass'))} "
                      f"· superclass-F1 {_fmt(out['guard']['per_class'].get('superclass'))} "
                      f"· =-F1 {_fmt(out['guard']['per_class'].get('equivalent'))}")
            md.append("- Guard check: the winner must NOT collapse subclass-F1 here "
                      "(dev is '>'-heavy; this is the '<'-heavy sanity).\n")

    Path(f"{args.out_prefix}.md").write_text("\n".join(md), encoding="utf-8")
    Path(f"{args.out_prefix}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\n" + "\n".join(md))
    logger.info("written: %s.md + .json", args.out_prefix)


def _load_bothorder_or_pred(d: Path):
    """Baseline dirs are single-order predictions.tsv; bothorder dirs have the
    bothorder file. Detect which."""
    if (d / "predictions.tsv").is_file():
        with (d / "predictions.tsv").open(encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t"))
    return _load_bothorder(d)


def _load_guard_slice():
    pairs = set()
    with GUARD_SLICE.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("source_uri"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                pairs.add((parts[0], parts[1]))
    return pairs


if __name__ == "__main__":
    main()
