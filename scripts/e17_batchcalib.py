#!/usr/bin/env python3
"""e17_batchcalib.py — Batch Calibration of the verification gate (E17).

Zhou et al., Batch Calibration (ICLR 2024): subtract the model's prior. For a
binary keep/reject with score p_yes, the log-odds BC decision at 0.5 is exactly
'keep if p_yes > p̄' where p̄ is the mean p_yes over a batch — parameter-free,
unsupervised, per-model, so it normalizes the incomparable p_yes scales
(mistral compressed near 0, gemma spread) without a dev-tuned threshold.

Arms: BC-model (p̄ = mean over the model's whole test set) and BC-cell (p̄ per
dataset). Reported next to V1(τ=.5) and the dev-tuned V3. Continuous-confidence
models only (llama/mistral/gemma); gpt-oss verdicts are binary.
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

MODELS = ("llama", "mistral", "gemma4")
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


def resolve_cell(model, ds, rr):
    for c in sorted(glob.glob(f"{rr}/matrix_{model}_{ds}_seed42_*"), key=os.path.getmtime, reverse=True):
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
    return {(r["source_uri"], r["target_uri"]): float(r["p_yes"])
            for r in csv.DictReader(p.open(encoding="utf-8"), delimiter="\t")}


def _eq(preds, gold, U):
    pe = [p for p in U if preds.get(p) == "="]; ge = [p for p in U if gold.get(p) == "="]
    tp = sum(1 for p in pe if gold.get(p) == "="); P = tp/len(pe) if pe else 0; R = tp/len(ge) if ge else 0
    return 2*P*R/(P+R) if (P+R) else 0.0


def cred(preds, gold, U, anc, desc):
    lt = credited_direction(preds, gold, anc, "<", U); gt = credited_direction(preds, gold, desc, ">", U)
    return statistics.mean([lt["credited"]["f1"], gt["credited"]["f1"], _eq(preds, gold, U)])


def strict(preds, gold, U, anc, desc):
    lt = credited_direction(preds, gold, anc, "<", U); gt = credited_direction(preds, gold, desc, ">", U)
    return statistics.mean([lt["strict"]["f1"], gt["strict"]["f1"], _eq(preds, gold, U)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-dir", default="results/e17"); ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-dir", default="results/e17")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent; vd = Path(args.verify_dir); out = Path(args.out_dir)
    golds = {ds: load_gold(ds, repo) for ds in TEST}; clos = {ds: closure(ds, repo) for ds in TEST}

    rows = []
    for m in MODELS:
        # gather available cells + verify, compute per-model prior p̄ (OAEI test)
        data = {}
        allp = []
        for ds in TEST:
            cell = resolve_cell(m, ds, args.results_root); vf = vd/f"e17_verify_{m}_{ds}_test.tsv"
            if cell is None or not vf.is_file():
                continue
            asserts, cand = load_assertions(cell); v = load_verify(vf)
            data[ds] = (asserts, cand, v)
            if ds != "vdi-ebay":
                allp += [v.get((s, t), 1.0) for (s, t, _) in asserts]
        if not data:
            continue
        pbar_model = statistics.mean(allp) if allp else 0.5
        for ds, (asserts, cand, v) in data.items():
            gold = golds[ds]; anc, desc = clos[ds]; U = cand | set(gold)
            base = {(s, t): r for (s, t, r) in asserts}
            pbar_cell = statistics.mean([v.get((s, t), 1.0) for (s, t, _) in asserts]) if asserts else 0.5
            bc_model = {(s, t): r for (s, t, r) in asserts if v.get((s, t), 1.0) > pbar_model}
            bc_cell = {(s, t): r for (s, t, r) in asserts if v.get((s, t), 1.0) > pbar_cell}
            bc = cred(base, gold, U, anc, desc); bstr = strict(base, gold, U, anc, desc)
            rows.append({"model": m, "dataset": ds, "pbar_model": round(pbar_model, 4),
                         "pbar_cell": round(pbar_cell, 4), "base_cred": round(bc, 4),
                         "BCmodel_dcred": round(cred(bc_model, gold, U, anc, desc)-bc, 4),
                         "BCcell_dcred": round(cred(bc_cell, gold, U, anc, desc)-bc, 4),
                         "BCmodel_dstrict": round(strict(bc_model, gold, U, anc, desc)-bstr, 4)})
    with (out/"e17_batchcalib_results.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(rows)

    by = defaultdict(dict)
    for r in rows: by[r["model"]][r["dataset"]] = r
    oaei = [d for d in TEST if d != "vdi-ebay"]
    print(f"{'model':8} {'p̄':>6} | {'BC-model Δcred':>14} {'BC-cell Δcred':>14} | per-set (BC-model)")
    md = ["# E17 Batch Calibration (mean Δ credited macro-F1, OAEI test)\n",
          "| model | p̄(model) | BC-model Δcred | BC-cell Δcred |", "|---|---|---|---|"]
    for m in MODELS:
        cells = [d for d in oaei if d in by[m]]
        if not cells: continue
        pbar = by[m][cells[0]]["pbar_model"]
        bcm = statistics.mean([by[m][d]["BCmodel_dcred"] for d in cells])
        bcc = statistics.mean([by[m][d]["BCcell_dcred"] for d in cells])
        perset = " ".join(f"{d.split('-')[0]}:{by[m][d]['BCmodel_dcred']:+.3f}" for d in TEST if d in by[m])
        print(f"{m:8} {pbar:>6.3f} | {bcm:>+14.4f} {bcc:>+14.4f} | {perset}")
        md.append(f"| {m} | {pbar:.3f} | {bcm:+.4f} | {bcc:+.4f} |")
    (out/"e17_batchcalib_summary.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
