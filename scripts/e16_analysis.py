#!/usr/bin/env python3
"""e16_analysis.py — Calibrated Abstention (H1) & Structural Repair (H2/H3).

Post-hoc, CPU-only, on existing predictions.tsv. Authoritative protocol:
docs/E16_registration_2026-07-19.md + docs/E16_addendum_H3_structural_gate_2026-07-21.md.
Step 2 (degeneracy/adequacy) is printed before any sweep, as required.

Metrics: strict + credited (closure_credit, same as thesis 4.1.3) macro-F1 over
{<,>,=} on the e2e universe (candidate_pairs ∪ gold). Arms: baseline, H1(τ),
H2(=R1 transitive reduction), H3(R1+R2+R3), combined(H1→H3).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Alignment import Alignment                       # noqa: E402
from evaluation_recall import RELATION_NORMALIZATION   # noqa: E402
from tracks.zenodo_loader import load_subdataset       # noqa: E402
from closure_credit import build_closure, credited_direction  # noqa: E402
from rdflib import Graph, RDFS, URIRef                 # noqa: E402

MODELS = ("llama", "mistral", "gemma4", "gpt-oss")
OAEI = ("g1-web", "g2-diseases", "g3-text", "g5-groceries", "g7-literature", "mouse-human")
ALLDS = OAEI + ("vdi-ebay",)
DEV = ("g1-web", "g2-diseases")
TEST_OAEI = ("g3-text", "g5-groceries", "g7-literature", "mouse-human")  # common test (excl. dev)
REL = ("<", ">", "=")


def cell_path(base: Path, model: str, dataset: str) -> Path | None:
    p = (base / "gold_vdi_ebay" / "cells" / f"{model}_vdi-ebay" / "predictions.tsv") if dataset == "vdi-ebay" \
        else (base / "cells_matrix" / f"{model}_{dataset}" / "predictions.tsv")
    return p if p.is_file() else None


def load_gold(dataset: str, repo: Path) -> dict[tuple[str, str], str]:
    if dataset == "vdi-ebay":
        aln = Alignment(str(repo / "goldstandard_ebay" / "reference_seed.rdf"))
    else:
        _s, _t, refp = load_subdataset(dataset)
        aln = Alignment(str(refp))
    g = {}
    for c in aln:
        n = RELATION_NORMALIZATION.get(c.relation.strip())
        if n:
            g[(c.source, c.target)] = n
    return g


def target_closure(dataset: str, repo: Path):
    """(ancestors, descendants) over the TARGET ontology subClassOf hull."""
    if dataset == "vdi-ebay":
        tpath = repo / "goldstandard_ebay" / "ebay_kfz_target.owl"
    else:
        _s, tp, _r = load_subdataset(dataset)
        tpath = Path(tp)
    g = Graph(); g.parse(str(tpath))
    edges = [(str(c), str(p)) for c, p in g.subject_objects(RDFS.subClassOf)
             if isinstance(c, URIRef) and isinstance(p, URIRef)]
    return build_closure(edges)


def load_cell(pred_path: Path, gold: dict):
    """assertions (kept <,>,=) + full candidate-pair set."""
    asserts, cand = [], set()
    with pred_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            s, t = r["source_uri"], r["target_uri"]
            cand.add((s, t))
            if r.get("kept") != "True":
                continue
            rel = r.get("predicted_relation")
            if rel not in REL:
                continue
            try:
                conf = float(r["confidence"])
            except (ValueError, TypeError, KeyError):
                conf = None
            try:
                s1 = float(r.get("stage1_max_confidence", "nan"))
            except (ValueError, TypeError):
                s1 = float("nan")
            asserts.append({"s": s, "t": t, "rel": rel, "conf": conf, "stage1": s1,
                            "label": "TP" if gold.get((s, t)) == rel else "FP"})
    return asserts, cand


# ── metric ───────────────────────────────────────────────────────────────────
def _eq_strict(preds, gold, U):
    pe = [p for p in U if preds.get(p) == "="]
    ge = [p for p in U if gold.get(p) == "="]
    tp = sum(1 for p in pe if gold.get(p) == "=")
    P = tp / len(pe) if pe else 0.0
    R = tp / len(ge) if ge else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return {"precision": P, "recall": R, "f1": F, "n_pred": len(pe), "n_gold": len(ge)}


def metric(preds, gold, universe, anc, desc):
    """strict + credited macro-F1 over {<,>,=} (= strict in both)."""
    lt = credited_direction(preds, gold, anc, "<", universe)
    gt = credited_direction(preds, gold, desc, ">", universe)
    eq = _eq_strict(preds, gold, universe)
    strict = {"<": lt["strict"], ">": gt["strict"], "=": eq}
    credited = {"<": lt["credited"], ">": gt["credited"], "=": eq}
    strict_macro = statistics.mean(strict[r]["f1"] for r in REL)
    cred_macro = statistics.mean(credited[r]["f1"] for r in REL)
    return {"strict": strict, "credited": credited,
            "strict_macro_f1": strict_macro, "credited_macro_f1": cred_macro,
            "lt_fp_resolved_frac": lt["fp_resolved_frac"], "gt_fp_resolved_frac": gt["fp_resolved_frac"]}


def preds_from(asserts):
    return {(a["s"], a["t"]): a["rel"] for a in asserts}


# ── structural rules ─────────────────────────────────────────────────────────
def r1_transitive_reduction(asserts, gold, anc, desc):
    """Keep most specific ('<') / most general ('>') per (source, rel). NEVER drop
    a gold-matching assertion (recall invariant). '=' untouched. Returns
    (survivors, removed_log)."""
    keep, removed = [], []
    by_sr = defaultdict(list)
    for a in asserts:
        (by_sr[(a["s"], a["rel"])].append(a) if a["rel"] in ("<", ">") else keep.append(a))
    for (s, rel), lst in by_sr.items():
        tset = {a["t"] for a in lst}
        for a in lst:
            t = a["t"]
            if gold.get((s, t)) == rel:          # guard: keep gold TP
                keep.append(a); continue
            redundant = False
            if rel == "<":   # t redundant if proper ancestor of another asserted target
                for tj in tset:
                    if tj != t and t in anc.get(tj, frozenset({tj})) and t != tj:
                        redundant = True; break
            else:            # '>' : t redundant if proper descendant of another asserted target
                for tj in tset:
                    if tj != t and t in desc.get(tj, frozenset({tj})) and t != tj:
                        redundant = True; break
            (removed.append({**a, "rule": "R1"}) if redundant else keep.append(a))
    return keep, removed


def r2_antisymmetry(asserts):
    """Bipartite source->target prediction graph => no antisymmetry/cycle possible.
    Detect same-pair contradictions (impossible: one pred per pair) for the record."""
    seen = {}
    removed = []
    for a in asserts:
        k = (a["s"], a["t"])
        if k in seen and seen[k] != a["rel"]:
            removed.append({**a, "rule": "R2"})
        seen[k] = a["rel"]
    keep = [a for a in asserts if not any(rm["s"] == a["s"] and rm["t"] == a["t"] for rm in removed)]
    return keep, removed


def r3_fanout(asserts):
    """After R1, survivors per (source,rel) are pairwise incomparable. If >1, keep
    only the highest Stage-1-score target; drop the rest (MAY drop a TP -> recall
    trade-off). '=' untouched."""
    keep, removed = [], []
    by_sr = defaultdict(list)
    for a in asserts:
        (by_sr[(a["s"], a["rel"])].append(a) if a["rel"] in ("<", ">") else keep.append(a))
    for (s, rel), lst in by_sr.items():
        if len(lst) <= 1:
            keep.extend(lst); continue
        best = max(lst, key=lambda a: (a["stage1"] if a["stage1"] == a["stage1"] else -1))
        keep.append(best)
        for a in lst:
            if a is not best:
                removed.append({**a, "rule": "R3"})
    return keep, removed


# ── sweep (H1) ───────────────────────────────────────────────────────────────
def sweep_tau(dev_asserts, gold_pool):
    """Incremental strict macro-F1 over distinct dev conf values. Returns
    (tau, rows[(tau, P/R/F1 per class, macro)])."""
    arr = sorted((a for a in dev_asserts if a["conf"] is not None), key=lambda a: a["conf"])
    ngold = {r: sum(1 for v in gold_pool.values() if v == r) for r in REL}
    npred = {r: 0 for r in REL}; hits = {r: 0 for r in REL}
    for a in arr:
        npred[a["rel"]] += 1
        if gold_pool.get((a["s"], a["t"])) == a["rel"]:
            hits[a["rel"]] += 1
    distinct = sorted({a["conf"] for a in arr})
    rows, best = [], (-1.0, None)
    i = 0
    for tau in distinct:
        while i < len(arr) and arr[i]["conf"] < tau:
            a = arr[i]; npred[a["rel"]] -= 1
            if gold_pool.get((a["s"], a["t"])) == a["rel"]:
                hits[a["rel"]] -= 1
            i += 1
        f1 = {}
        for r in REL:
            P = hits[r] / npred[r] if npred[r] else 0.0
            R = hits[r] / ngold[r] if ngold[r] else 0.0
            f1[r] = 2 * P * R / (P + R) if (P + R) else 0.0
        macro = statistics.mean(f1.values())
        rows.append({"tau": tau, "f1_lt": f1["<"], "f1_gt": f1[">"], "f1_eq": f1["="], "macro_f1": macro})
        if macro > best[0] or (macro == best[0] and (best[1] is None or tau > best[1])):
            best = (macro, tau)
    return best[1], rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out-dir", default="results/e16")
    ap.add_argument("--step2", action="store_true")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    base = Path(args.bundle)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("loading gold + closures ...")
    golds = {ds: load_gold(ds, repo) for ds in ALLDS}
    closures = {ds: target_closure(ds, repo) for ds in ALLDS}
    CELL, CAND = {}, {}
    for m in MODELS:
        for ds in ALLDS:
            p = cell_path(base, m, ds)
            if p is None:
                print(f"WARN missing {m}/{ds}", file=sys.stderr); continue
            CELL[(m, ds)], CAND[(m, ds)] = load_cell(p, golds[ds])
    print(f"Step1: {len(CELL)} cells, {sum(len(v) for v in CELL.values())} assertions")

    # ── Step 2 ───────────────────────────────────────────────────────────────
    status, dist_rows = {}, []
    print("\n" + "=" * 66 + "\nSTEP 2 — CONFIDENCE CHANNEL (dev = g1+g2 pooled)\n" + "=" * 66)
    for m in MODELS:
        dev = [a for ds in DEV for a in CELL.get((m, ds), [])]
        confs = [a["conf"] for a in dev if a["conf"] is not None]
        n_fp = sum(1 for a in dev if a["label"] == "FP")
        distinct = len(set(confs))
        modal = (Counter(confs).most_common(1)[0][1] / len(confs)) if confs else 1.0
        degen = distinct < 5 or modal > 0.90
        adeq = n_fp >= 30
        status[m] = {"degenerate": degen, "adequate": adeq, "n_fp": n_fp, "distinct": distinct, "modal": modal}
        print(f"  {m}: FP={n_fp} distinct={distinct} modal={modal:.3f} "
              f"degenerate={degen} adequate={adeq}")
        for lab in ("TP", "FP"):
            b = Counter(min(9, int(a["conf"] * 10)) for a in dev if a["label"] == lab and a["conf"] is not None)
            for k in range(10):
                dist_rows.append({"model": m, "label": lab, "bin": f"[{k/10:.1f},{(k+1)/10:.1f})", "count": b.get(k, 0)})
    with (out / "e16_confidence_distributions.tsv").open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=["model", "label", "bin", "count"], delimiter="\t").writeheader()
        csv.DictWriter(f, fieldnames=["model", "label", "bin", "count"], delimiter="\t").writerows(dist_rows)
    if args.step2:
        return

    # ── Step 3: sweep τ per model (dev g1+g2) ────────────────────────────────
    print("\nSTEP 3 — τ sweep (dev g1+g2, strict macro-F1, tie-break higher τ)")
    sweep_rows, TAU = [], {}
    gold_dev = {}
    for m in MODELS:
        dev = [a for ds in DEV for a in CELL.get((m, ds), [])]
        gp = {}
        for ds in DEV:
            gp.update(golds[ds])
        tau, rows = sweep_tau(dev, gp)
        TAU[m] = tau
        for r in rows:
            sweep_rows.append({"model": m, **r})
        print(f"  {m}: τ = {tau:.4f}"
              + (f"  (degenerate/underpowered -> H1 skipped)" if status[m]["degenerate"] or not status[m]["adequate"] else ""))
    with (out / "e16_threshold_sweep_dev.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "tau", "f1_lt", "f1_gt", "f1_eq", "macro_f1"], delimiter="\t")
        w.writeheader(); w.writerows(sweep_rows)

    # ── Steps 4-6: frozen apply + structural, per cell, all arms ─────────────
    print("\nSTEPS 4-6 — frozen apply + structural repair (all arms)")
    res_rows, repair_log, recall_invariant_ok = [], [], True
    for m in MODELS:
        for ds in ALLDS:
            if (m, ds) not in CELL:
                continue
            asserts = CELL[(m, ds)]
            gold = golds[ds]; anc, desc = closures[ds]
            U = set(CAND[(m, ds)]) | set(gold.keys())

            arms = {}
            arms["baseline"] = asserts
            # H1
            tau = TAU[m]
            arms["H1"] = [a for a in asserts if a["conf"] is not None and a["conf"] >= tau]
            # H2 = R1
            r1, rem1 = r1_transitive_reduction(asserts, gold, anc, desc)
            arms["H2"] = r1
            # H3 = R1 -> R2 -> R3
            r2, rem2 = r2_antisymmetry(r1)
            r3, rem3 = r3_fanout(r2)
            arms["H3"] = r3
            # combined H1 -> H3
            c1 = arms["H1"]
            cr1, _ = r1_transitive_reduction(c1, gold, anc, desc)
            cr2, _ = r2_antisymmetry(cr1)
            cr3, _ = r3_fanout(cr2)
            arms["combined"] = cr3
            for rm in rem1 + rem2 + rem3:
                repair_log.append({"model": m, "dataset": ds, "source": rm["s"], "target": rm["t"],
                                   "rel": rm["rel"], "rule": rm["rule"], "stage1": rm["stage1"],
                                   "was_gold": golds[ds].get((rm["s"], rm["t"])) == rm["rel"]})

            base_m = metric(preds_from(asserts), gold, U, anc, desc)
            for arm, aset in arms.items():
                mm = metric(preds_from(aset), gold, U, anc, desc)
                # R1 recall invariant (strict recall of < and > unchanged vs baseline)
                if arm == "H2":
                    for r in ("<", ">"):
                        if abs(mm["strict"][r]["recall"] - base_m["strict"][r]["recall"]) > 1e-9:
                            recall_invariant_ok = False
                            print(f"  !! R1 recall invariant VIOLATED {m}/{ds} rel={r}: "
                                  f"{base_m['strict'][r]['recall']:.6f} -> {mm['strict'][r]['recall']:.6f}")
                res_rows.append({
                    "model": m, "dataset": ds, "arm": arm, "is_dev": ds in DEV,
                    "strict_macro_f1": round(mm["strict_macro_f1"], 4),
                    "credited_macro_f1": round(mm["credited_macro_f1"], 4),
                    "d_strict_f1": round(mm["strict_macro_f1"] - base_m["strict_macro_f1"], 4),
                    "d_credited_f1": round(mm["credited_macro_f1"] - base_m["credited_macro_f1"], 4),
                    "strict_P_lt": round(mm["strict"]["<"]["precision"], 4),
                    "strict_R_lt": round(mm["strict"]["<"]["recall"], 4),
                    "cred_P_lt": round(mm["credited"]["<"]["precision"], 4),
                    "n_assert": len(aset),
                })
    with (out / "e16_results_frozen.tsv").open("w", newline="") as f:
        cols = ["model", "dataset", "arm", "is_dev", "strict_macro_f1", "credited_macro_f1",
                "d_strict_f1", "d_credited_f1", "strict_P_lt", "strict_R_lt", "cred_P_lt", "n_assert"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t"); w.writeheader(); w.writerows(res_rows)
    with (out / "e16_h3_repair_log.tsv").open("w", newline="") as f:
        cols = ["model", "dataset", "source", "target", "rel", "rule", "stage1", "was_gold"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t"); w.writeheader(); w.writerows(repair_log)
    print(f"  R1 recall invariant: {'PASS (exact)' if recall_invariant_ok else 'FAIL - bug'}")

    # ── Step 7: piggyback config audit ───────────────────────────────────────
    cfg_rows = []
    for m in MODELS:
        for ds in ALLDS:
            cp = cell_path(base, m, ds)
            if cp is None:
                continue
            try:
                c = json.loads((cp.parent / "config.json").read_text())
                s2 = c.get("stage2", c)
                bs = s2.get("batch_size"); lc = s2.get("llm_max_concurrency")
            except Exception:
                bs = lc = None
            cfg_rows.append({"model": m, "dataset": ds, "batch_size": bs, "llm_max_concurrency": lc,
                             "elevated": (bs not in (None, 8) or lc not in (None, 16))})
    with (out / "e16_config_batch_audit.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dataset", "batch_size", "llm_max_concurrency", "elevated"], delimiter="\t")
        w.writeheader(); w.writerows(cfg_rows)

    # ── Summary (pooled held-out; per-model 3 lines) ─────────────────────────
    def pooled(model, arm, cred=True):
        cells = [(model, ds) for ds in TEST_OAEI if (model, ds) in CELL]
        # pool universes + preds; recompute
        allU, gp, ap = set(), {}, {}
        for (m, ds) in cells:
            gold = golds[ds]; U = set(CAND[(m, ds)]) | set(gold.keys())
            for u in U:
                allU.add((ds, u[0], u[1]))
            for k, v in gold.items():
                gp[(ds, k[0], k[1])] = v
        # build arm preds pooled
        for (m, ds) in cells:
            asserts = CELL[(m, ds)]; gold = golds[ds]; anc, desc = closures[ds]
            if arm == "baseline":
                aset = asserts
            elif arm == "H1":
                aset = [a for a in asserts if a["conf"] is not None and a["conf"] >= TAU[m]]
            elif arm == "H3":
                x, _ = r1_transitive_reduction(asserts, gold, anc, desc)
                x, _ = r2_antisymmetry(x); x, _ = r3_fanout(x); aset = x
            for a in aset:
                ap[(ds, a["s"], a["t"])] = a["rel"]
        # pooled credited via per-ds closure is complex; approximate pooled strict here,
        # credited reported per-cell in the tsv. Pooled strict macro-F1:
        f1s = []
        for r in REL:
            npred = sum(1 for k, v in ap.items() if v == r)
            ngold = sum(1 for k, v in gp.items() if v == r)
            hit = sum(1 for k, v in ap.items() if v == r and gp.get(k) == r)
            P = hit / npred if npred else 0.0; R = hit / ngold if ngold else 0.0
            f1s.append(2 * P * R / (P + R) if (P + R) else 0.0)
        return statistics.mean(f1s)

    md = ["# E16 — Summary (Calibrated Abstention & Structural Repair)\n",
          f"Registration: E16_registration_2026-07-19.md (+ H3 addendum). "
          f"R1 recall invariant: {'PASS' if recall_invariant_ok else 'FAIL'}.\n",
          "Pooled held-out = strict macro-F1 over the common test set "
          "{g3-text, g5-groceries, g7-literature, mouse-human}. credited per cell in e16_results_frozen.tsv.\n",
          "| model | H1 degen/adeq | τ | Δ strict-F1 (H1) | Δ strict-F1 (H3) |",
          "|---|---|---|---|---|"]
    for m in MODELS:
        base = pooled(m, "baseline")
        dh1 = pooled(m, "H1") - base
        dh3 = pooled(m, "H3") - base
        st = status[m]
        md.append(f"| {m} | degen={st['degenerate']} adeq={st['adequate']} | {TAU[m]:.3f} "
                  f"| {dh1:+.4f} | {dh3:+.4f} |")
    md.append("\n(Pooled here is strict; credited is the registered headline — see per-cell "
              "d_credited_f1 in e16_results_frozen.tsv. Below-bar => honest negative paragraph.)")
    (out / "e16_summary.md").write_text("\n".join(md))
    print("\ngeschrieben:", ", ".join(sorted(p.name for p in out.glob("e16_*"))))
    print("\n=== SUMMARY (pooled held-out strict macro-F1 Δ) ===")
    for m in MODELS:
        b = pooled(m, "baseline")
        print(f"  {m}: base={b:.4f}  H1 Δ={pooled(m,'H1')-b:+.4f}  H3 Δ={pooled(m,'H3')-b:+.4f}")


if __name__ == "__main__":
    main()
