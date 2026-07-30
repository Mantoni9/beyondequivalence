#!/usr/bin/env python3
"""e17_viability.py — E17 dev viability checkpoint (go/no-go before the full run).

Reads the e17_verify_<model>_{g1-web,g2-diseases}_dev.tsv outputs, labels each
verified assertion credited-correct (TP) vs not (FP) using the same closure_credit
machinery as E16, and reports per model the verifier YES-rate for TP vs FP.

Registered gate (E17): proceed to the full test run only if, per model, the
verifier is non-degenerate (overall YES-rate in (5%,95%)) AND separates
(TP-YES minus FP-YES >= 0.10). Rubber-stamp => negative, no full run.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Alignment import Alignment                      # noqa: E402
from evaluation_recall import RELATION_NORMALIZATION  # noqa: E402
from tracks.zenodo_loader import load_subdataset      # noqa: E402
from closure_credit import build_closure             # noqa: E402
from rdflib import Graph, RDFS, URIRef               # noqa: E402

MODELS = ("llama", "mistral", "gemma4", "gpt-oss")
DEV = ("g1-web", "g2-diseases")


def load_gold(ds):
    _s, _t, r = load_subdataset(ds)
    g = {}
    for c in Alignment(str(r)):
        n = RELATION_NORMALIZATION.get(c.relation.strip())
        if n:
            g[(c.source, c.target)] = n
    return g


def target_closure(ds):
    _s, tp, _r = load_subdataset(ds)
    g = Graph(); g.parse(str(tp))
    edges = [(str(c), str(p)) for c, p in g.subject_objects(RDFS.subClassOf)
             if isinstance(c, URIRef) and isinstance(p, URIRef)]
    return build_closure(edges)


def credited_ok(s, t, rel, gold, anc, desc):
    """Is (s,t,rel) credited-correct? strict hit, or coarsening of a same-source
    gold under the target closure (< -> ancestor-or-self, > -> descendant-or-self)."""
    if gold.get((s, t)) == rel:
        return True
    if rel == "=":
        return False
    clo = anc if rel == "<" else desc
    for (gs, gt), gr in gold.items():
        if gs == s and gr == rel and t in clo.get(gt, frozenset({gt})):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-dir", default="results/e17")
    ap.add_argument("--tag", default="_dev")
    args = ap.parse_args()
    vd = Path(args.verify_dir)

    golds = {ds: load_gold(ds) for ds in DEV}
    clos = {ds: target_closure(ds) for ds in DEV}

    print(f"{'model':8} {'n':>7} {'YES-all':>8} {'TP n':>7} {'YES-TP':>7} "
          f"{'FP n':>7} {'YES-FP':>7} {'separation':>11} {'gate':>6}")
    summary = []
    for m in MODELS:
        tp_yes = tp_n = fp_yes = fp_n = 0
        for ds in DEV:
            f = vd / f"e17_verify_{m}_{ds}{args.tag}.tsv"
            if not f.is_file():
                print(f"  (missing {f.name})", file=sys.stderr); continue
            anc, desc = clos[ds]; gold = golds[ds]
            with f.open() as fh:
                for r in csv.DictReader(fh, delimiter="\t"):
                    s, t, rel = r["source_uri"], r["target_uri"], r["rel"]
                    yes = float(r["p_yes"]) > 0.5
                    if credited_ok(s, t, rel, gold, anc, desc):
                        tp_n += 1; tp_yes += yes
                    else:
                        fp_n += 1; fp_yes += yes
        n = tp_n + fp_n
        if n == 0:
            continue
        yes_all = (tp_yes + fp_yes) / n
        ytp = tp_yes / tp_n if tp_n else 0.0
        yfp = fp_yes / fp_n if fp_n else 0.0
        sep = ytp - yfp
        degen = yes_all > 0.95 or yes_all < 0.05
        gate = "PASS" if (not degen and sep >= 0.10) else "FAIL"
        print(f"{m:8} {n:>7} {yes_all:>8.3f} {tp_n:>7} {ytp:>7.3f} {fp_n:>7} {yfp:>7.3f} {sep:>+11.3f} {gate:>6}")
        summary.append(dict(model=m, n=n, yes_all=round(yes_all, 4), tp_n=tp_n, yes_tp=round(ytp, 4),
                            fp_n=fp_n, yes_fp=round(yfp, 4), separation=round(sep, 4), gate=gate))
    with (vd / "e17_verification_dev_viability.tsv").open("w", newline="") as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()), delimiter="\t")
            w.writeheader(); w.writerows(summary)
    print("\nGate: PASS = non-degenerate AND (YES-TP − YES-FP) ≥ 0.10  → eligible for full test run.")


if __name__ == "__main__":
    main()
