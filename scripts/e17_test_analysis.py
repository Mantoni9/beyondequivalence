#!/usr/bin/env python3
"""e17_test_analysis.py — apply the verification gate to the test cells, score.

Reads e17_verify_<model>_<ds><tag>.tsv (p_yes [+ p_yes_rev]) and the matrix cell
predictions, gates the assertions, and recomputes strict + credited macro-F1 vs
baseline (same closure_credit machinery as E16/E18). Runs on DWS (resolves cells
from results/) or locally with --cells-root pointing at a bundle's cells_matrix.

Arms: baseline · V1(τ keep if p_yes>τ) · SYM(V1 and reject if p_yes_rev>τ).
Threshold τ default 0.5 (argmax); a small sweep is reported descriptively.
"""
from __future__ import annotations
import argparse, csv, glob, os, statistics, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Alignment import Alignment
from evaluation_recall import RELATION_NORMALIZATION
from tracks.zenodo_loader import load_subdataset
from closure_credit import build_closure, credited_direction
from rdflib import Graph, RDFS, URIRef

MODELS = ("llama", "mistral", "gemma4", "gpt-oss")
TEST = ("g3-text", "g5-groceries", "g7-literature", "mouse-human", "vdi-ebay")
REL = ("<", ">", "=")


def load_gold(ds, repo):
    ref = (repo/"goldstandard_ebay"/"reference_seed.rdf") if ds == "vdi-ebay" else Path(load_subdataset(ds)[2])
    g = {}
    for c in Alignment(str(ref)):
        n = RELATION_NORMALIZATION.get(c.relation.strip())
        if n: g[(c.source, c.target)] = n
    return g


def target_closure(ds, repo):
    tp = (repo/"goldstandard_ebay"/"ebay_kfz_target.owl") if ds == "vdi-ebay" else Path(load_subdataset(ds)[1])
    g = Graph(); g.parse(str(tp))
    return build_closure([(str(c), str(p)) for c, p in g.subject_objects(RDFS.subClassOf)
                          if isinstance(c, URIRef) and isinstance(p, URIRef)])


def resolve_cell(model, ds, cells_root, results_root):
    if cells_root:
        p = Path(cells_root) / f"{model}_{ds}" / "predictions.tsv"
        return p if p.is_file() else None
    for c in sorted(glob.glob(f"{results_root}/matrix_{model}_{ds}_seed42_*"),
                    key=os.path.getmtime, reverse=True):
        if "_shard" in c or "_g2shard" in c:
            continue
        if os.path.isfile(f"{c}/predictions.tsv"):
            return Path(c) / "predictions.tsv"
    return None


def load_assertions(p):
    out, cand = [], set()
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            s, t = r["source_uri"], r["target_uri"]; cand.add((s, t))
            if r.get("kept") == "True" and r.get("predicted_relation") in REL:
                out.append((s, t, r["predicted_relation"]))
    return out, cand


def load_verify(vpath):
    v = {}
    with vpath.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            v[(r["source_uri"], r["target_uri"])] = (
                float(r["p_yes"]),
                float(r["p_yes_rev"]) if r.get("p_yes_rev") not in (None, "",) else None)
    return v


def _eq(preds, gold, U):
    pe = [p for p in U if preds.get(p) == "="]; ge = [p for p in U if gold.get(p) == "="]
    tp = sum(1 for p in pe if gold.get(p) == "="); P = tp/len(pe) if pe else 0; R = tp/len(ge) if ge else 0
    return 2*P*R/(P+R) if (P+R) else 0.0


def metric(preds, gold, U, anc, desc):
    lt = credited_direction(preds, gold, anc, "<", U); gt = credited_direction(preds, gold, desc, ">", U)
    eqf = _eq(preds, gold, U)
    return (statistics.mean([lt["strict"]["f1"], gt["strict"]["f1"], eqf]),
            statistics.mean([lt["credited"]["f1"], gt["credited"]["f1"], eqf]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-dir", default="results/e17")
    ap.add_argument("--tag", default="_test")
    ap.add_argument("--cells-root", default=None)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-dir", default="results/e17")
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    vd = Path(args.verify_dir)

    golds = {ds: load_gold(ds, repo) for ds in TEST}
    clos = {ds: target_closure(ds, repo) for ds in TEST}
    res = []
    for m in MODELS:
        for ds in TEST:
            cell = resolve_cell(m, ds, args.cells_root, args.results_root)
            vpath = vd / f"e17_verify_{m}_{ds}{args.tag}.tsv"
            if cell is None or not vpath.is_file():
                print(f"skip {m}/{ds} (cell={cell is not None}, verify={vpath.is_file()})", file=sys.stderr); continue
            asserts, cand = load_assertions(cell)
            v = load_verify(vpath)
            gold = golds[ds]; anc, desc = clos[ds]; U = cand | set(gold)
            base = {(s, t): r for (s, t, r) in asserts}
            v1 = {(s, t): r for (s, t, r) in asserts if v.get((s, t), (1.0, None))[0] > args.tau}
            sym = {(s, t): r for (s, t, r) in asserts
                   if v.get((s, t), (1.0, None))[0] > args.tau
                   and not (v.get((s, t), (1.0, None))[1] is not None and v[(s, t)][1] > args.tau)}
            bs, bc = metric(base, gold, U, anc, desc)
            for arm, preds in (("baseline", base), ("V1", v1), ("SYM", sym)):
                ss, cc = metric(preds, gold, U, anc, desc)
                res.append({"model": m, "dataset": ds, "arm": arm, "n_kept": len(preds),
                            "strict": round(ss, 4), "credited": round(cc, 4),
                            "d_strict": round(ss-bs, 4), "d_credited": round(cc-bc, 4)})
    with (out/"e17_test_results.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(res)

    by = defaultdict(dict)
    for r in res: by[(r["model"], r["dataset"])][r["arm"]] = r
    oaei = [d for d in TEST if d != "vdi-ebay"]
    print(f"\n{'model':8} {'arm':5} {'meanΔcred(OAEI)':>16} {'meanΔstrict':>12} {'worstΔcred':>11} {'vdi Δcred':>10} {'bar':>5}")
    md = ["# E17 verification test-set results (mean Δ credited macro-F1)\n",
          "| model | V1 Δcred (OAEI) | SYM Δcred | vdi V1 | bar(+0.03) |", "|---|---|---|---|---|"]
    for m in MODELS:
        cells = [(m, ds) for ds in oaei if (m, ds) in by]
        line = f"| {m} "
        for arm in ("V1", "SYM"):
            dc = [by[(m, ds)][arm]["d_credited"] for (m, ds) in cells]
            ds_ = [by[(m, ds)][arm]["d_strict"] for (m, ds) in cells]
            worst = min(dc) if dc else 0
            vdi = by.get((m, "vdi-ebay"), {}).get(arm, {}).get("d_credited", 0)
            if arm == "V1":
                bar = "YES" if (dc and statistics.mean(dc) >= 0.03 and worst > -0.05) else "no"
                print(f"{m:8} {arm:5} {statistics.mean(dc):>+16.4f} {statistics.mean(ds_):>+12.4f} {worst:>+11.4f} {vdi:>+10.4f} {bar:>5}")
                line += f"| {statistics.mean(dc):+.4f} "
            else:
                print(f"{m:8} {arm:5} {statistics.mean(dc):>+16.4f} {statistics.mean(ds_):>+12.4f} {worst:>+11.4f} {vdi:>+10.4f}")
                line += f"| {statistics.mean(dc):+.4f} | {vdi:+.4f} | {bar} |"
        md.append(line)
    (out/"e17_test_summary.md").write_text("\n".join(md))
    print("\ngeschrieben:", ", ".join(sorted(p.name for p in out.glob("e17_test_*"))))


if __name__ == "__main__":
    main()
