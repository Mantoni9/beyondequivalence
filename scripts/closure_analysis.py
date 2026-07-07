"""
closure_analysis.py — P3 Closure re-scoring driver (thesis 4.1.3, K5/K1).

For each matrix cell it computes, over BOTH the conditional (candidate-only) and
the e2e (candidate∪gold) universe:
  - STRICT '<'/'>'/'=' P/R/F1  (identity-checked against compute_multiclass_metrics,
    which is what produced the a24e146 numbers — so strict == a24e146 by reuse),
  - CREDITED '<'/'>' P/R/F1 via the reflexive-transitive subClassOf closure of the
    TARGET ontology (scripts/closure_credit.credited_direction),
  - Macro-credited = mean(cred'<'F1, cred'>'F1, strict'='F1),
  - the K1 core value: fraction of the strict '<'-FPs resolved by credit,
  - byproduct: per-prediction credit flags + which of the 40 audit row_ids are
    closure-resolvable (→ granularity, not gold-gap).

Layout mirrors matrix_analysis (a24e146). CPU-only.

Run (on the cluster, env melt-olala):
  python scripts/closure_analysis.py --run MODEL:DATASET:SEED=DIR ... \
      --audit-key results/ltfp_audit_key.json --out-prefix results/closure_analysis_<sha>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rdflib import RDFS
from rdflib.term import URIRef

from Alignment import Alignment, Correspondence
from RDFGraphWrapper import RDFGraphWrapper
from evaluation_recall import _normalize_relation
from evaluation_multiclass import compute_multiclass_metrics
from tracks.zenodo_loader import load_subdataset

from closure_credit import build_closure, credited_direction

logger = logging.getLogger("closure_analysis")

_IDENTITY_TOL = 1e-9


# ----------------------------------------------------------------- data loading

def _load_cell(d: Path):
    """(cand set, pred_by_pair) from a cell's predictions.tsv — identical rule
    to analyze_matrix._load_run so the strict basis matches a24e146."""
    rows = []
    with (d / "predictions.tsv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    cand = {(r["source_uri"], r["target_uri"]) for r in rows}
    pred = {}
    for r in rows:
        kept = r.get("kept") == "True"
        rel = r["predicted_relation"] if kept and r["predicted_relation"] in ("<", ">", "=") else "none"
        pred[(r["source_uri"], r["target_uri"])] = rel
    return cand, pred


_gold_cache: dict[str, dict] = {}
_closure_cache: dict[str, tuple] = {}


def _gold(ds: str) -> dict[tuple[str, str], str]:
    if ds not in _gold_cache:
        _s, _t, ref = load_subdataset(ds)
        g = {}
        for c in Alignment(str(ref)):
            n = _normalize_relation(c.relation)
            if n:
                g[(c.source, c.target)] = n
        _gold_cache[ds] = g
    return _gold_cache[ds]


def _target_closure(ds: str):
    """(ancestors, descendants) over the reflexive-transitive subClassOf hull of
    the TARGET ontology, keyed by str(URI)."""
    if ds not in _closure_cache:
        _s, tgt, _r = load_subdataset(ds)
        kg = RDFGraphWrapper(str(tgt))
        # Named-class subClassOf edges only; skip anonymous parents (owl:Restriction
        # blank nodes) — they carry no named ancestor a predicted/gold target could match.
        edges = [(str(c), str(pp)) for c, _, pp in kg.graph.triples((None, RDFS.subClassOf, None))
                 if isinstance(c, URIRef) and isinstance(pp, URIRef)]
        anc, desc = build_closure(edges)
        _closure_cache[ds] = (anc, desc)
        logger.info("closure[%s]: %d subClassOf edges, %d classes", ds, len(edges), len(anc))
    return _closure_cache[ds]


# ----------------------------------------------------------------- strict basis

def _strict_per_class(gold: dict, pred: dict, cand: set) -> dict:
    """compute_multiclass_metrics per_class (the EXACT a24e146 e2e basis)."""
    reference = Alignment()
    for (s, t), rel in gold.items():
        reference.add(Correspondence(s, t, rel, 1.0))
    predictions = Alignment()
    for (s, t), rel in pred.items():
        if rel in ("<", ">", "="):
            predictions.add(Correspondence(s, t, rel, 1.0))
    rep = compute_multiclass_metrics(reference=reference, predictions=predictions,
                                     candidate_pairs=cand).to_dict()
    return rep["per_class"]


# ----------------------------------------------------------------- per cell

def _analyze_cell(model: str, ds: str, d: Path, anc: dict, desc: dict) -> dict:
    cand, pred = _load_cell(d)
    gold = _gold(ds)
    e2e_universe = set(cand) | set(gold)
    cond_universe = set(cand)

    strict_pc = _strict_per_class(gold, pred, cand)

    out = {"model": model, "dataset": ds, "n_candidates": len(cand),
           "identity_ok": True, "identity_detail": {}}
    for basis, universe in (("cond", cond_universe), ("e2e", e2e_universe)):
        lt = credited_direction(pred, gold, anc, "<", universe)
        gt = credited_direction(pred, gold, desc, ">", universe)
        out[basis] = {
            "<": {"strict": lt["strict"], "credited": lt["credited"],
                  "strict_fp": lt["strict_fp"], "fp_resolved": lt["fp_resolved"],
                  "fp_resolved_frac": lt["fp_resolved_frac"]},
            ">": {"strict": gt["strict"], "credited": gt["credited"],
                  "strict_fp": gt["strict_fp"], "fp_resolved": gt["fp_resolved"],
                  "fp_resolved_frac": gt["fp_resolved_frac"]},
        }
        if basis == "e2e":
            out["flip_pairs_lt"] = lt["flip_pairs"]
            out["flip_pairs_gt"] = gt["flip_pairs"]
            out["k1_lt_fp_resolved_frac"] = lt["fp_resolved_frac"]
            # Macro-credited (e2e) over {cred<, cred>, strict=} — '=' unchanged.
            eq_f1 = strict_pc.get("=", {}).get("f1", 0.0)
            out["macro_credited_e2e"] = (lt["credited"]["f1"] + gt["credited"]["f1"] + eq_f1) / 3.0
            out["macro_strict_e2e"] = (lt["strict"]["f1"] + gt["strict"]["f1"] + eq_f1) / 3.0

    # ---- identity check: my strict '<'/'>' precision+recall == compute_multiclass
    for cls, r in (("<", out["e2e"]["<"]), (">", out["e2e"][">"])):
        for field in ("precision", "recall", "f1"):
            mine = r["strict"][field]
            ref = strict_pc.get(cls, {}).get(field, 0.0)
            if abs(mine - ref) > 1e-6:
                out["identity_ok"] = False
                out["identity_detail"][f"{cls}.{field}"] = {"mine": mine, "a24e146": ref}
    return out


# ----------------------------------------------------------------- audit x-check

def _audit_crosscheck(cells: list[dict], audit_key_path: Path) -> dict:
    """Which of the blind-audit row_ids are closure-resolvable (a model's '<'-FP
    for that pair flipped to credited) → granularity, not a gold gap."""
    if not audit_key_path or not audit_key_path.is_file():
        return {"note": "audit key not found", "path": str(audit_key_path)}
    key = json.loads(audit_key_path.read_text())
    # union of all '<'-flip pairs across cells, per dataset
    flipped: set = set()
    for c in cells:
        for (s, t) in c.get("flip_pairs_lt", []):
            flipped.add((c["dataset"], s, t))
    resolved_rows, unresolved_rows = [], []
    for rid, meta in key.items():
        triple = (meta.get("dataset"), meta.get("source_uri"), meta.get("target_uri"))
        (resolved_rows if triple in flipped else unresolved_rows).append(rid)
    return {"n_rows": len(key),
            "closure_resolved_row_ids": sorted(resolved_rows),
            "n_closure_resolved": len(resolved_rows),
            "n_unresolved": len(unresolved_rows),
            "note": "resolved rows = hierarchy-granularity FPs (NOT gold gaps); "
                    "cross-check against Antonio's adjudication of the same row_ids."}


# ----------------------------------------------------------------- reporting

def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "nosha"


def _fmt(v, s=".3f"):
    return "—" if v is None else format(v, s)


def _write(cells: list[dict], audit: dict, out_prefix: str):
    md = ["# Stage-2 Closure re-scoring (hierarchy-credited '<'/'>')",
          "",
          "Reflexive-transitive subClassOf closure of the TARGET ontology; same-label "
          "credit (ancestor for '<', descendant for '>'); existential set semantics; "
          "'=' strict. STRICT columns reproduce a24e146 (identity-checked). "
          "credited-R may exceed the strict coverage bound in e2e = **entailed coverage** "
          "(a coarse prediction recovers a Stage-1-missed gold that it entails).",
          ""]
    bad = [f"{c['model']}/{c['dataset']}" for c in cells if not c["identity_ok"]]
    md.append(f"**Identity check (strict == a24e146):** {'✅ all pass' if not bad else '❌ FAIL: ' + ', '.join(bad)}")
    md.append("")
    md.append("## '<' precision — strict vs credited (the K5 headline) + K1 FP-resolution")
    md.append("")
    md.append("| Model | Dataset | strict-P< | cred-P< (cond) | cred-P< (e2e) | strict-F< | cred-F< | strict-F> | cred-F> | Macro-str | Macro-cred | **K1: <-FP resolved** |")
    md.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c in sorted(cells, key=lambda x: (x["model"], x["dataset"])):
        lt_c, gt_c = c["cond"]["<"], c["cond"][">"]
        lt_e = c["e2e"]["<"]
        md.append("| {m} | {d} | {sp} | {cpc} | {cpe} | {sf} | {cf} | {sfg} | {cfg} | {ms} | {mc} | **{k1}** |".format(
            m=c["model"], d=c["dataset"],
            sp=_fmt(lt_e["strict"]["precision"]),
            cpc=_fmt(lt_c["credited"]["precision"]),
            cpe=_fmt(lt_e["credited"]["precision"]),
            sf=_fmt(lt_e["strict"]["f1"]), cf=_fmt(lt_e["credited"]["f1"]),
            sfg=_fmt(gt_c["strict"]["f1"]), cfg=_fmt(gt_c["credited"]["f1"]),
            ms=_fmt(c["macro_strict_e2e"]), mc=_fmt(c["macro_credited_e2e"]),
            k1=_fmt(c["k1_lt_fp_resolved_frac"])))
    md.append("")
    md.append("## Audit row_id cross-check (closure-resolvable = granularity, not gold gap)")
    md.append(f"- rows: {audit.get('n_rows','?')} · closure-resolved: **{audit.get('n_closure_resolved','?')}** · unresolved: {audit.get('n_unresolved','?')}")
    if audit.get("closure_resolved_row_ids"):
        md.append(f"- resolved row_ids: `{', '.join(audit['closure_resolved_row_ids'])}`")
    md.append(f"- {audit.get('note','')}")
    md.append("")

    Path(out_prefix + ".md").write_text("\n".join(md), encoding="utf-8")
    Path(out_prefix + ".json").write_text(json.dumps(
        {"cells": cells, "audit_crosscheck": audit, "sha": _git_sha()},
        indent=2), encoding="utf-8")
    logger.info("written: %s.md + .json", out_prefix)
    return bad


def main():
    p = argparse.ArgumentParser(description="P3 Closure re-scoring.")
    p.add_argument("--run", nargs="+", required=True, metavar="MODEL:DATASET[:SEED]=DIR")
    p.add_argument("--audit-key", default="results/ltfp_audit_key.json")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s")

    cells = []
    for item in args.run:
        key, _, path = item.partition("=")
        parts = key.split(":")
        model, ds = parts[0], parts[1]
        d = Path(path)
        if not (d / "predictions.tsv").is_file():
            logger.warning("SKIP %s/%s — no predictions.tsv at %s", model, ds, d)
            continue
        anc, desc = _target_closure(ds)
        cells.append(_analyze_cell(model, ds, d, anc, desc))
        logger.info("done %s/%s identity_ok=%s", model, ds, cells[-1]["identity_ok"])

    audit = _audit_crosscheck(cells, Path(args.audit_key))
    out_prefix = args.out_prefix or f"results/closure_analysis_{_git_sha()}"
    bad = _write(cells, audit, out_prefix)
    if bad:
        logger.error("IDENTITY CHECK FAILED for: %s", bad)
        sys.exit(1)


if __name__ == "__main__":
    main()
