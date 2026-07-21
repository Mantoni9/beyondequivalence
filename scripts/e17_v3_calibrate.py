#!/usr/bin/env python3
"""e17_v3_calibrate.py — V3: per-model verification threshold, tuned on dev.

Registered E17 arm V3: tau_v = argmax over dev (g1+g2) of dev credited macro-F1
(tie-break higher tau_v), applied frozen to the test set. CPU-only; uses the
dev verify outputs (e17_verify_<m>_{g1,g2}_dev.tsv, first-token) and the test
verify outputs (e17_verify_<m>_<ds>_test.tsv). Continuous-confidence models only
(llama/mistral/gemma); gpt-oss reasoning verdicts are binary -> no threshold knob.
"""
from __future__ import annotations
import argparse, csv, glob, os, re, statistics, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Alignment import Alignment
from evaluation_recall import RELATION_NORMALIZATION
from tracks.zenodo_loader import load_subdataset
from closure_credit import build_closure, credited_direction
from rdflib import Graph, RDFS, URIRef

MODELS = ("llama", "mistral", "gemma4")   # continuous-confidence models
DEV = ("g1-web", "g2-diseases")
TEST = ("g3-text", "g5-groceries", "g7-literature", "mouse-human", "vdi-ebay")
REL = ("<", ">", "=")


def load_gold(ds, repo):
    ref = (repo/"goldstandard_ebay"/"reference_seed.rdf") if ds == "vdi-ebay" else Path(load_subdataset(ds)[2])
    g = {}
    for c in Alignment(str(ref)):
        n = RELATION_NORMALIZATION.get(c.relation.strip())
        if n: g[(c.source, c.target)] = n
    return g


def closure(ds, repo):
    tp = (repo/"goldstandard_ebay"/"ebay_kfz_target.owl") if ds == "vdi-ebay" else Path(load_subdataset(ds)[1])
    g = Graph(); g.parse(str(tp))
    return build_closure([(str(c), str(p)) for c, p in g.subject_objects(RDFS.subClassOf)
                          if isinstance(c, URIRef) and isinstance(p, URIRef)])


def resolve_cell(model, ds, results_root):
    for c in sorted(glob.glob(f"{results_root}/matrix_{model}_{ds}_seed42_*"), key=os.path.getmtime, reverse=True):
        if re.search(r"_shard|_g2shard|_thinkoff|_relow|_A[1-4]_", c):
            continue
        if os.path.isfile(f"{c}/predictions.tsv"):
            return Path(c)/"predictions.tsv"
    return None


def load_assertions(p):
    out, cand = [], set()
    for r in csv.DictReader(p.open(encoding="utf-8"), delimiter="\t"):
        s, t = r["source_uri"], r["target_uri"]; cand.add((s, t))
        if r.get("kept") == "True" and r.get("predicted_relation") in REL:
            out.append((s, t, r["predicted_relation"]))
    return out, cand


def load_verify(p):
    v = {}
    for r in csv.DictReader(p.open(encoding="utf-8"), delimiter="\t"):
        v[(r["source_uri"], r["target_uri"])] = float(r["p_yes"])
    return v


def _eq(preds, gold, U):
    pe = [p for p in U if preds.get(p) == "="]; ge = [p for p in U if gold.get(p) == "="]
    tp = sum(1 for p in pe if gold.get(p) == "="); P = tp/len(pe) if pe else 0; R = tp/len(ge) if ge else 0
    return 2*P*R/(P+R) if (P+R) else 0.0


def credited_macro(preds, gold, U, anc, desc):
    lt = credited_direction(preds, gold, anc, "<", U); gt = credited_direction(preds, gold, desc, ">", U)
    return statistics.mean([lt["credited"]["f1"], gt["credited"]["f1"], _eq(preds, gold, U)])


def strict_macro(preds, gold, U, anc, desc):
    lt = credited_direction(preds, gold, anc, "<", U); gt = credited_direction(preds, gold, desc, ">", U)
    return statistics.mean([lt["strict"]["f1"], gt["strict"]["f1"], _eq(preds, gold, U)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-dir", default="results/e17")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-dir", default="results/e17")
    ap.add_argument("--grid", type=int, default=60)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    out = Path(args.out_dir); vd = Path(args.verify_dir)
    golds = {ds: load_gold(ds, repo) for ds in set(DEV) | set(TEST)}
    clos = {ds: closure(ds, repo) for ds in set(DEV) | set(TEST)}

    rows = []
    print(f"{'model':8} {'tau_v':>6} {'dev cF1@tau':>11} {'dev cF1@.5':>11} || test credited-Δ per set")
    for m in MODELS:
        # ── build dev pool (assertions + p_yes + per-ds gold/closure) ────────
        dev = []   # (ds, s, t, rel, p_yes)
        dev_U = {}; dev_base = {}
        ok = True
        for ds in DEV:
            cell = resolve_cell(m, ds, args.results_root); vf = vd/f"e17_verify_{m}_{ds}_dev.tsv"
            if cell is None or not vf.is_file():
                ok = False; break
            asserts, cand = load_assertions(cell); v = load_verify(vf)
            dev_U[ds] = cand | set(golds[ds]); dev_base[ds] = {(s, t): r for (s, t, r) in asserts}
            for (s, t, rel) in asserts:
                dev.append((ds, s, t, rel, v.get((s, t), 1.0)))
        if not ok:
            print(f"{m}: dev outputs missing — skip", file=sys.stderr); continue

        # ── sweep tau_v: maximize mean dev credited-F1 over g1,g2 ────────────
        confs = sorted({d[4] for d in dev})
        if len(confs) > args.grid:
            step = len(confs)/args.grid
            confs = [confs[min(len(confs)-1, int(i*step))] for i in range(args.grid)]
        best = (-1.0, 0.0)
        for tau in confs:
            sc = []
            for ds in DEV:
                preds = {(s, t): rel for (dd, s, t, rel, p) in dev if dd == ds and p > tau}
                anc, desc = clos[ds]
                sc.append(credited_macro(preds, golds[ds], dev_U[ds], anc, desc))
            mean_sc = statistics.mean(sc)
            if mean_sc > best[0] or (mean_sc == best[0] and tau > best[1]):
                best = (mean_sc, tau)
        tau_v = best[1]
        # dev credited at tau_v and at 0.5 (for the record)
        def dev_cf1(tau):
            return statistics.mean([credited_macro(
                {(s, t): rel for (dd, s, t, rel, p) in dev if dd == ds and p > tau},
                golds[ds], dev_U[ds], *clos[ds]) for ds in DEV])
        dev_at, dev_50 = best[0], dev_cf1(0.5)

        # ── apply frozen to test ─────────────────────────────────────────────
        line = f"{m:8} {tau_v:>6.3f} {dev_at:>11.4f} {dev_50:>11.4f} ||"
        for ds in TEST:
            cell = resolve_cell(m, ds, args.results_root); vf = vd/f"e17_verify_{m}_{ds}_test.tsv"
            if cell is None or not vf.is_file():
                continue
            asserts, cand = load_assertions(cell); v = load_verify(vf)
            gold = golds[ds]; anc, desc = clos[ds]; U = cand | set(gold)
            base = {(s, t): r for (s, t, r) in asserts}
            v3 = {(s, t): r for (s, t, r) in asserts if v.get((s, t), 1.0) > tau_v}
            v1 = {(s, t): r for (s, t, r) in asserts if v.get((s, t), 1.0) > 0.5}
            bc = credited_macro(base, gold, U, anc, desc); bs = strict_macro(base, gold, U, anc, desc)
            v3c = credited_macro(v3, gold, U, anc, desc); v3s = strict_macro(v3, gold, U, anc, desc)
            v1c = credited_macro(v1, gold, U, anc, desc)
            rows.append({"model": m, "dataset": ds, "tau_v": round(tau_v, 4),
                         "base_cred": round(bc, 4), "V1_dcred": round(v1c-bc, 4),
                         "V3_dcred": round(v3c-bc, 4), "V3_dstrict": round(v3s-bs, 4)})
            line += f" {ds.split('-')[0]}:{v3c-bc:+.3f}"
        print(line)

    with (out/"e17_v3_results.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(rows)
    # summary: mean test credited Δ (OAEI) V1 vs V3
    by = defaultdict(dict)
    for r in rows: by[r["model"]][r["dataset"]] = r
    print(f"\n{'model':8} {'tau_v':>6} {'V1 Δcred(OAEI)':>15} {'V3 Δcred(OAEI)':>15}")
    md = ["# E17 V3 — calibrated per-model threshold (mean Δ credited macro-F1, OAEI test)\n",
          "| model | tau_v | V1(τ=.5) Δcred | V3(calibrated) Δcred |", "|---|---|---|---|"]
    oaei = [d for d in TEST if d != "vdi-ebay"]
    for m in MODELS:
        cells = [d for d in oaei if d in by[m]]
        if not cells: continue
        v1 = statistics.mean([by[m][d]["V1_dcred"] for d in cells])
        v3 = statistics.mean([by[m][d]["V3_dcred"] for d in cells])
        tau = by[m][cells[0]]["tau_v"]
        print(f"{m:8} {tau:>6.3f} {v1:>+15.4f} {v3:>+15.4f}")
        md.append(f"| {m} | {tau:.3f} | {v1:+.4f} | {v3:+.4f} |")
    (out/"e17_v3_summary.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
