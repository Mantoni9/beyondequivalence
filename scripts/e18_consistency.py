#!/usr/bin/env python3
"""e18_consistency.py — Logical consistency / coherence filter (E18).

Post-hoc, CPU-only, on existing predictions.tsv. Protocol:
docs/E18_logical_consistency_filter_2026-07-21.md. Removes provably-inconsistent /
non-conservative predicted edges (confidence-based tie-break), model-independent.
Strict + credited macro-F1 vs baseline, identical machinery to E16.
"""
from __future__ import annotations
import argparse, csv, statistics, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Alignment import Alignment
from evaluation_recall import RELATION_NORMALIZATION
from tracks.zenodo_loader import load_subdataset
from closure_credit import build_closure, credited_direction
from rdflib import Graph, RDFS, URIRef

MODELS = ("llama", "mistral", "gemma4", "gpt-oss")
OAEI = ("g1-web", "g2-diseases", "g3-text", "g5-groceries", "g7-literature", "mouse-human")
ALLDS = OAEI + ("vdi-ebay",)
TEST = ("g3-text", "g5-groceries", "g7-literature", "mouse-human")
REL = ("<", ">", "=")


def cell_path(base, m, ds):
    p = (base/"gold_vdi_ebay"/"cells"/f"{m}_vdi-ebay"/"predictions.tsv") if ds == "vdi-ebay" \
        else (base/"cells_matrix"/f"{m}_{ds}"/"predictions.tsv")
    return p if p.is_file() else None


def load_gold(ds, repo):
    ref = (repo/"goldstandard_ebay"/"reference_seed.rdf") if ds == "vdi-ebay" else Path(load_subdataset(ds)[2])
    g = {}
    for c in Alignment(str(ref)):
        n = RELATION_NORMALIZATION.get(c.relation.strip())
        if n:
            g[(c.source, c.target)] = n
    return g


def target_closure(ds, repo):
    tp = (repo/"goldstandard_ebay"/"ebay_kfz_target.owl") if ds == "vdi-ebay" else Path(load_subdataset(ds)[1])
    g = Graph(); g.parse(str(tp))
    return build_closure([(str(c), str(p)) for c, p in g.subject_objects(RDFS.subClassOf)
                          if isinstance(c, URIRef) and isinstance(p, URIRef)])


def load_cell(p, gold):
    asserts, cand = [], set()
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            s, t = r["source_uri"], r["target_uri"]; cand.add((s, t))
            if r.get("kept") != "True" or r.get("predicted_relation") not in REL:
                continue
            try: conf = float(r["confidence"])
            except Exception: conf = 0.0
            asserts.append({"s": s, "t": t, "rel": r["predicted_relation"], "conf": conf})
    return asserts, cand


def preds_from(a): return {(x["s"], x["t"]): x["rel"] for x in a}


def _eq(preds, gold, U):
    pe = [p for p in U if preds.get(p) == "="]; ge = [p for p in U if gold.get(p) == "="]
    tp = sum(1 for p in pe if gold.get(p) == "="); P = tp/len(pe) if pe else 0.0; R = tp/len(ge) if ge else 0.0
    return 2*P*R/(P+R) if (P+R) else 0.0


def metric(preds, gold, U, anc, desc):
    lt = credited_direction(preds, gold, anc, "<", U); gt = credited_direction(preds, gold, desc, ">", U)
    eqf = _eq(preds, gold, U)
    sm = statistics.mean([lt["strict"]["f1"], gt["strict"]["f1"], eqf])
    cm = statistics.mean([lt["credited"]["f1"], gt["credited"]["f1"], eqf])
    return sm, cm, lt["strict"]["recall"], gt["strict"]["recall"]


# ── consistency rules ─────────────────────────────────────────────────────────
def apply_rules(asserts, anc, desc, rules):
    """Return (kept, removed) applying the named rules in fixed order."""
    keep = list(asserts); removed = []

    def by_src(items):
        d = defaultdict(list)
        for a in items: d[a["s"]].append(a)
        return d

    if "L1" in rules:  # direction contradiction: s<t1 & s>t2 with t2 in anc(t1)
        drop = set()
        for s, lst in by_src(keep).items():
            lts = [a for a in lst if a["rel"] == "<"]; gts = [a for a in lst if a["rel"] == ">"]
            for a1 in lts:
                for a2 in gts:
                    if a2["t"] != a1["t"] and a2["t"] in anc.get(a1["t"], frozenset({a1["t"]})):
                        lo = a1 if a1["conf"] <= a2["conf"] else a2
                        drop.add(id(lo)); removed.append({**lo, "rule": "L1"})
                    elif a2["t"] == a1["t"]:  # same target both < and > (shouldn't happen; guard)
                        lo = a1 if a1["conf"] <= a2["conf"] else a2
                        drop.add(id(lo)); removed.append({**lo, "rule": "L1"})
        keep = [a for a in keep if id(a) not in drop]

    if "L2" in rules:  # multiple non-equivalent '='
        drop = set()
        for s, lst in by_src(keep).items():
            eqs = [a for a in lst if a["rel"] == "="]
            for i in range(len(eqs)):
                for j in range(i+1, len(eqs)):
                    t1, t2 = eqs[i]["t"], eqs[j]["t"]
                    equiv = (t2 in anc.get(t1, frozenset({t1}))) and (t1 in anc.get(t2, frozenset({t2})))
                    if not equiv:
                        lo = eqs[i] if eqs[i]["conf"] <= eqs[j]["conf"] else eqs[j]
                        if id(lo) not in drop:
                            drop.add(id(lo)); removed.append({**lo, "rule": "L2"})
        keep = [a for a in keep if id(a) not in drop]

    if "L3" in rules:  # ancestor-safe sibling exclusion
        drop = set()
        for s, lst in by_src(keep).items():
            anchors = [a for a in lst if a["rel"] == "="] or [a for a in lst if a["rel"] == "<"]
            if not anchors:
                continue
            A = max(anchors, key=lambda a: a["conf"])["t"]
            safe = anc.get(A, frozenset({A})) | desc.get(A, frozenset({A}))
            for a in lst:
                if a["rel"] == "<" and a["t"] != A and a["t"] not in safe:
                    drop.add(id(a)); removed.append({**a, "rule": "L3"})
        keep = [a for a in keep if id(a) not in drop]

    if "R1" in rules:  # pure transitive reduction (keep most specific per source&dir)
        drop = set()
        for (s, rel), lst in {(k[0], k[1]): v for k, v in
                              _group_sr(keep).items()}.items() if False else _group_sr(keep).items():
            if rel not in ("<", ">"):
                continue
            tset = {a["t"] for a in lst}
            for a in lst:
                t = a["t"]; red = False
                pool = anc if rel == "<" else desc
                for tj in tset:
                    if tj != t and t in pool.get(tj, frozenset({tj})) and t != tj:
                        red = True; break
                if red:
                    drop.add(id(a)); removed.append({**a, "rule": "R1"})
        keep = [a for a in keep if id(a) not in drop]

    return keep, removed


def _group_sr(items):
    d = defaultdict(list)
    for a in items: d[(a["s"], a["rel"])].append(a)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True); ap.add_argument("--out-dir", default="results/e18")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent; base = Path(args.bundle)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print("loading gold + closures ...")
    golds = {ds: load_gold(ds, repo) for ds in ALLDS}
    clos = {ds: target_closure(ds, repo) for ds in ALLDS}
    CELL, CAND = {}, {}
    for m in MODELS:
        for ds in ALLDS:
            p = cell_path(base, m, ds)
            if p: CELL[(m, ds)], CAND[(m, ds)] = load_cell(p, golds[ds])

    ARMS = {"L1": ["L1"], "L2": ["L2"], "L1L2": ["L1", "L2"],
            "L1L2R1": ["L1", "L2", "R1"], "L1L2L3R1": ["L1", "L2", "L3", "R1"]}
    res, rmlog = [], []
    for (m, ds), asserts in CELL.items():
        gold = golds[ds]; anc, desc = clos[ds]; U = set(CAND[(m, ds)]) | set(gold)
        bs, bc, _, _ = metric(preds_from(asserts), gold, U, anc, desc)
        res.append({"model": m, "dataset": ds, "arm": "baseline", "strict": round(bs, 4),
                    "credited": round(bc, 4), "d_strict": 0.0, "d_credited": 0.0, "n": len(asserts)})
        for arm, rules in ARMS.items():
            kept, removed = apply_rules(asserts, anc, desc, rules)
            ss, cc, _, _ = metric(preds_from(kept), gold, U, anc, desc)
            res.append({"model": m, "dataset": ds, "arm": arm, "strict": round(ss, 4),
                        "credited": round(cc, 4), "d_strict": round(ss-bs, 4),
                        "d_credited": round(cc-bc, 4), "n": len(kept)})
            if arm == "L1L2L3R1":
                for rm in removed:
                    rmlog.append({"model": m, "dataset": ds, "source": rm["s"], "target": rm["t"],
                                  "rel": rm["rel"], "rule": rm["rule"], "conf": round(rm["conf"], 4),
                                  "was_gold": gold.get((rm["s"], rm["t"])) == rm["rel"]})
    with (out/"e18_results.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(res)
    with (out/"e18_removal_log.tsv").open("w", newline="") as f:
        if rmlog:
            w = csv.DictWriter(f, fieldnames=list(rmlog[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(rmlog)

    # summary: pooled test-set credited Δ per model per arm
    by = defaultdict(dict)
    for r in res: by[(r["model"], r["dataset"])][r["arm"]] = r
    print(f"\n{'model':8} {'arm':10} {'meanΔcred(test)':>16} {'meanΔstrict':>12} {'worstΔcred':>11}")
    md = ["# E18 — Logical consistency filter (mean Δ credited macro-F1, test g3/g5/g7/mh)\n",
          "| model | L1 | L2 | L1L2 | L1L2R1 | L1L2L3R1 |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        cells = [(m, ds) for ds in TEST if (m, ds) in CELL]
        row = [f"| {m} "]
        for arm in ("L1", "L2", "L1L2", "L1L2R1", "L1L2L3R1"):
            dc = [by[(m, ds)][arm]["d_credited"] for (m, ds) in cells]
            ds_ = [by[(m, ds)][arm]["d_strict"] for (m, ds) in cells]
            worst = min(dc) if dc else 0
            print(f"{m:8} {arm:10} {statistics.mean(dc):>+16.4f} {statistics.mean(ds_):>+12.4f} {worst:>+11.4f}")
            row.append(f"| {statistics.mean(dc):+.4f} ")
        md.append("".join(row) + "|")
    from collections import Counter
    rc = Counter(r["rule"] for r in rmlog); gc = Counter(r["rule"] for r in rmlog if r["was_gold"])
    md.append("\n## Removed edges by rule (of which gold = recall loss)")
    for rule in ("L1", "L2", "L3", "R1"):
        md.append(f"- {rule}: {rc.get(rule,0)} (gold {gc.get(rule,0)})")
    md.append("\n## vdi-ebay (second arm), credited baseline -> L1L2R1")
    for m in MODELS:
        b = by.get((m, "vdi-ebay"), {}).get("baseline"); h = by.get((m, "vdi-ebay"), {}).get("L1L2R1")
        if b and h: md.append(f"- {m}: {b['credited']:.3f} -> {h['credited']:.3f} ({h['d_credited']:+.4f})")
    (out/"e18_summary.md").write_text("\n".join(md))
    print("\ngeschrieben:", ", ".join(sorted(p.name for p in out.glob("e18_*"))))


if __name__ == "__main__":
    main()
