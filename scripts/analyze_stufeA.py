"""
analyze_stufeA.py — registered Stufe-A attribution analyzer (label order vs
argument position vs content prior). CPU-only, reads finished stage2 run dirs.

The research question is CAUSAL: where does Llama's subclass-prior come from?
  H-label    — tracks LABEL-LIST POSITION  -> A1 (order flip) moves it, A2 not.
  H-position — tracks ARGUMENT POSITION    -> A2 shows the mirror signature
               (flip_rate_gt drops >= 0.30 AND flip_rate_lt rises >= 0.30).
  H-content  — genuine semantic prior      -> A1 and A2 both NO; errors stick
               to the same pairs regardless of presentation.

Registered outcome machinery (docs/stage2_stufeA_registration.md, filed
before any Stufe-A run):
  - PRIMARY: flip_rate_gt, reranker-conditional, per dataset + dev-pooled.
  - Effect bands per arm (delta = baseline − arm, positive = improvement):
    SOLID >= 0.15 · SMALL 0.05–0.15 · NO < 0.05 · REVERSE = worse by >= 0.05.
  - Guard: dev-pooled =-F1 < 0.70 -> the arm is REVERSE regardless of flip
    rate (the v3 lesson).
  - A3 trigger: A1 AND A2 both show >= 0.05 improvement (ambiguous
    attribution).
  - Consistency rule: per-dataset deltas must agree in direction across
    g7 and g5; a g7-only effect is flagged as a possible g7-tuning artifact.
  - Named flip-set: the gold-'>'-predicted-'<' pairs of the g7 baseline
    (Run 255471; 26 pairs) tracked per arm as resolved / persisted / other.

All metrics are recomputed through evaluation_multiclass.compute_multiclass_metrics
(definitional consistency with the runners) and cross-validated against each
run's stored metrics.json — any mismatch is reported and exits non-zero.

Usage (run dirs are the stage2_* output dirs):
  conda run -n melt-olala python scripts/analyze_stufeA.py \
      --baseline g7-literature=<dir255471> g5-groceries=<dirR0> \
      --a1 g7-literature=<dir> g5-groceries=<dir> \
      --a2 g7-literature=<dir> g5-groceries=<dir> \
      [--a3 g7-literature=<dir> g5-groceries=<dir>]
Outputs: results/stufeA_analysis.md + .json
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

logger = logging.getLogger("analyze_stufeA")

DEV_DATASETS = ("g7-literature", "g5-groceries")
SOLID, SMALL, NO_BAND, REVERSE = "SOLID", "SMALL", "NO", "REVERSE"
EQ_F1_GUARD = 0.70
MIRROR_THRESHOLD = 0.30
PARSE_FAIL_GATE = 0.05


def _parse_run_args(pairs: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in pairs or []:
        ds, _, path = item.partition("=")
        if ds not in DEV_DATASETS:
            sys.exit(f"ERROR: dataset {ds!r} is not a dev dataset {DEV_DATASETS} "
                     "— tuning runs on test datasets void the protocol.")
        out[ds] = Path(path)
    return out


def _load_gold(dataset: str) -> dict[tuple[str, str], str]:
    _s, _t, ref_path = load_subdataset(dataset)
    gold: dict[tuple[str, str], str] = {}
    for cor in Alignment(str(ref_path)):
        norm = _normalize_relation(cor.relation)
        if norm is not None:
            gold[(cor.source, cor.target)] = norm
    return gold


def _load_run(run_dir: Path) -> dict:
    rows = []
    with (run_dir / "predictions.tsv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(row)
    stored = json.loads((run_dir / "metrics.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    return {"rows": rows, "stored": stored, "config": config, "dir": str(run_dir)}


def _alignments_for(rows: list[dict], reference_gold: dict) -> tuple[Alignment, Alignment, set]:
    """(reference, predictions, candidate_pairs) for compute_multiclass_metrics."""
    candidate_pairs = {(r["source_uri"], r["target_uri"]) for r in rows}
    reference = Alignment()
    for (s, t), rel in reference_gold.items():
        reference.add(Correspondence(s, t, rel, 1.0))
    predictions = Alignment()
    for r in rows:
        if r["kept"] == "True":
            predictions.add(Correspondence(
                r["source_uri"], r["target_uri"], r["predicted_relation"],
                float(r["confidence"])))
    return reference, predictions, candidate_pairs


def _eq_f1(report_dict: dict) -> float | None:
    per_class = report_dict.get("per_class", {})
    for key in ("equivalent", "equivalence", "="):
        if key in per_class:
            return per_class[key].get("f1")
    return None


def _run_metrics(run: dict, gold: dict) -> dict:
    reference, predictions, candidate_pairs = _alignments_for(run["rows"], gold)
    report = compute_multiclass_metrics(
        reference=reference, predictions=predictions, candidate_pairs=candidate_pairs,
    ).to_dict()

    histogram: dict[str, int] = {}
    no_gold_lt = 0
    for r in run["rows"]:
        histogram[r["parsed_canonical"]] = histogram.get(r["parsed_canonical"], 0) + 1
        if (r["kept"] == "True" and r["predicted_relation"] == "<"
                and (r["source_uri"], r["target_uri"]) not in gold):
            no_gold_lt += 1
    n_rows = max(1, len(run["rows"]))
    parse_fail_rate = histogram.get("parse_fail", 0) / n_rows

    # Cross-validation against the runner's stored metrics.
    stored = run["stored"]
    deviations = []
    for key in ("flip_rate_gt", "flip_rate_lt", "macro_f1", "direction_accuracy"):
        ours, theirs = report.get(key), stored.get(key)
        if ours is not None and theirs is not None and abs(ours - theirs) > 1e-9:
            deviations.append(f"{key}: recomputed {ours} != stored {theirs}")

    return {"report": report, "histogram": histogram, "no_gold_lt": no_gold_lt,
            "parse_fail_rate": parse_fail_rate, "eq_f1": _eq_f1(report),
            "deviations": deviations}


def _pooled_metrics(runs: dict[str, dict], golds: dict[str, dict]) -> dict:
    reference, predictions = Alignment(), Alignment()
    candidate_pairs: set = set()
    for ds, run in runs.items():
        ref, pred, cands = _alignments_for(run["rows"], golds[ds])
        for c in ref:
            reference.add(c)
        for c in pred:
            predictions.add(c)
        candidate_pairs |= cands
    report = compute_multiclass_metrics(
        reference=reference, predictions=predictions, candidate_pairs=candidate_pairs,
    ).to_dict()
    return {"report": report, "eq_f1": _eq_f1(report)}


def _flip_set(rows: list[dict], gold: dict) -> set[tuple[str, str]]:
    return {(r["source_uri"], r["target_uri"]) for r in rows
            if gold.get((r["source_uri"], r["target_uri"])) == ">"
            and r["kept"] == "True" and r["predicted_relation"] == "<"}


def _flip_resolution(rows: list[dict], flip_set: set) -> dict[str, int]:
    out = {"resolved": 0, "persisted": 0, "equivalent": 0, "dropped": 0, "other": 0}
    by_pair = {(r["source_uri"], r["target_uri"]): r for r in rows}
    for pair in flip_set:
        r = by_pair.get(pair)
        if r is None or r["kept"] != "True":
            out["dropped"] += 1
        elif r["predicted_relation"] == ">":
            out["resolved"] += 1
        elif r["predicted_relation"] == "<":
            out["persisted"] += 1
        elif r["predicted_relation"] == "=":
            out["equivalent"] += 1
        else:
            out["other"] += 1
    return out


def _a2_symmetry_sanity(base_rows: list[dict], a2_rows: list[dict]) -> dict:
    """Measurement guard for the H-position mirror signature (registered
    addition, GO for submission 2026-06-12): the mirror reading assumes the
    presentation swap leaves the slot-SYMMETRIC classes invariant — a pair v2
    calls '=' should stay '=' under A2. Reports '='/none prediction counts on
    the SAME conditional pairs plus pair-level '='-retention; a > 10% relative
    shift flags that the flip_rate_lt arm of the signature carries noise.
    Not a pass/fail band — it calibrates how literally to read the mirror."""
    base_by = {(r["source_uri"], r["target_uri"]): r for r in base_rows}
    a2_by = {(r["source_uri"], r["target_uri"]): r for r in a2_rows}
    shared = set(base_by) & set(a2_by)

    def _counts(by: dict) -> tuple[int, int]:
        eq = sum(1 for p in shared if by[p]["parsed_canonical"] == "equivalent")
        none = sum(1 for p in shared if by[p]["parsed_canonical"] == "none")
        return eq, none

    b_eq, b_none = _counts(base_by)
    a_eq, a_none = _counts(a2_by)
    eq_shift = ((a_eq - b_eq) / b_eq) if b_eq else None
    none_shift = ((a_none - b_none) / b_none) if b_none else None
    base_eq_pairs = [p for p in shared
                     if base_by[p]["parsed_canonical"] == "equivalent"]
    retention = (sum(1 for p in base_eq_pairs
                     if a2_by[p]["parsed_canonical"] == "equivalent")
                 / len(base_eq_pairs)) if base_eq_pairs else None
    flagged = any(s is not None and abs(s) > 0.10 for s in (eq_shift, none_shift))
    return {"n_shared_pairs": len(shared),
            "v2_eq": b_eq, "a2_eq": a_eq, "eq_rel_shift": eq_shift,
            "eq_retention": retention,
            "v2_none": b_none, "a2_none": a_none, "none_rel_shift": none_shift,
            "flagged": flagged}


def _band(base_gt: float, arm_gt: float, pooled_eq_f1: float | None) -> str:
    if pooled_eq_f1 is not None and pooled_eq_f1 < EQ_F1_GUARD:
        return f"{REVERSE} (guard: dev-pooled =-F1 {pooled_eq_f1:.3f} < {EQ_F1_GUARD})"
    if arm_gt - base_gt >= 0.05:
        return REVERSE
    delta = base_gt - arm_gt
    if delta >= 0.15:
        return SOLID
    if delta >= 0.05:
        return SMALL
    return NO_BAND


def _fmt(v, spec=".3f"):
    return "—" if v is None else format(v, spec)


def main() -> None:
    p = argparse.ArgumentParser(description="Registered Stufe-A attribution analysis.")
    p.add_argument("--baseline", nargs="+", required=True, metavar="DS=DIR",
                   help="v2 baselines: g7 (Run 255471) and g5 (R0).")
    p.add_argument("--a1", nargs="+", default=None, metavar="DS=DIR",
                   help="A1 = d_subs_v4b (label-order flip, padding constant).")
    p.add_argument("--a2", nargs="+", default=None, metavar="DS=DIR",
                   help="A2 = swap-pair-presentation (argument position).")
    p.add_argument("--a3", nargs="+", default=None, metavar="DS=DIR",
                   help="A3 = both flips (conditional arm).")
    p.add_argument("--out-prefix", default="results/stufeA_analysis")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s")

    arms_dirs = {"baseline": _parse_run_args(args.baseline)}
    for name, val in (("A1", args.a1), ("A2", args.a2), ("A3", args.a3)):
        if val:
            arms_dirs[name] = _parse_run_args(val)

    golds = {ds: _load_gold(ds) for ds in
             {ds for dirs in arms_dirs.values() for ds in dirs}}

    arms: dict[str, dict] = {}
    all_deviations: list[str] = []
    for arm_name, dirs in arms_dirs.items():
        runs = {ds: _load_run(d) for ds, d in dirs.items()}
        per_ds = {ds: _run_metrics(run, golds[ds]) for ds, run in runs.items()}
        pooled = _pooled_metrics(runs, golds) if len(runs) > 1 else None
        arms[arm_name] = {"runs": runs, "per_ds": per_ds, "pooled": pooled}
        for ds, m in per_ds.items():
            for dev in m["deviations"]:
                all_deviations.append(f"{arm_name}/{ds}: {dev}")

    base = arms["baseline"]

    # Named g7 flip-set from the baseline (expected: the 26 pairs of 255471).
    g7 = "g7-literature"
    flip_set = (_flip_set(base["runs"][g7]["rows"], golds[g7])
                if g7 in base["runs"] else set())

    md: list[str] = []
    md.append("# Stufe A — Bias attribution: label order vs argument position "
              "vs content prior\n")
    md.append("Registered analysis (docs/stage2_stufeA_registration.md). All "
              "metrics reranker-conditional on the frozen d11c97e candidates; "
              "dev protocol = {g7-literature, g5-groceries} (THESIS_NOTES.md). "
              "Interpretation language is CAUSAL: each arm manipulates exactly "
              "one presentation factor; the verdict reads the pre-registered "
              "signatures, not post-hoc stories.\n")

    md.append("## Per-arm registered metrics\n")
    md.append("| Arm | Dataset | flip_rate_gt | flip_rate_lt | Macro-F1 | =-F1 "
              "| dir-acc | parse_fail | '<' on no-gold | label histogram |")
    md.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for arm_name, arm in arms.items():
        for ds, m in arm["per_ds"].items():
            r = m["report"]
            hist = ", ".join(f"{k}:{v}" for k, v in sorted(m["histogram"].items()))
            gate = "" if m["parse_fail_rate"] < PARSE_FAIL_GATE else " ⚠GATE"
            md.append(f"| {arm_name} | {ds} | {_fmt(r.get('flip_rate_gt'))} "
                      f"| {_fmt(r.get('flip_rate_lt'))} | {_fmt(r.get('macro_f1'))} "
                      f"| {_fmt(m['eq_f1'])} | {_fmt(r.get('direction_accuracy'))} "
                      f"| {m['parse_fail_rate']:.3f}{gate} | {m['no_gold_lt']} "
                      f"| {hist} |")
        if arm["pooled"]:
            rp = arm["pooled"]["report"]
            md.append(f"| **{arm_name}** | **dev-pooled** "
                      f"| **{_fmt(rp.get('flip_rate_gt'))}** "
                      f"| {_fmt(rp.get('flip_rate_lt'))} | {_fmt(rp.get('macro_f1'))} "
                      f"| {_fmt(arm['pooled']['eq_f1'])} "
                      f"| {_fmt(rp.get('direction_accuracy'))} | | | |")
    md.append("")

    # ---- effect bands + consistency ----
    bands: dict[str, dict] = {}
    if base["pooled"]:
        base_gt_pooled = base["pooled"]["report"].get("flip_rate_gt")
        md.append("## Effect bands (Δ flip_rate_gt vs same-dataset baseline; "
                  "positive Δ = improvement)\n")
        md.append("| Arm | g7 Δ | g5 Δ | pooled Δ | pooled band | consistency |")
        md.append("| --- | ---: | ---: | ---: | --- | --- |")
        for arm_name in ("A1", "A2", "A3"):
            if arm_name not in arms or not arms[arm_name]["pooled"]:
                continue
            arm = arms[arm_name]
            deltas = {}
            for ds in arm["per_ds"]:
                b = base["per_ds"].get(ds)
                if b:
                    deltas[ds] = (b["report"].get("flip_rate_gt") or 0.0) - \
                                 (arm["per_ds"][ds]["report"].get("flip_rate_gt") or 0.0)
            arm_gt_pooled = arm["pooled"]["report"].get("flip_rate_gt")
            band = _band(base_gt_pooled, arm_gt_pooled, arm["pooled"]["eq_f1"])
            signs = {ds: (1 if d >= 0.0 else -1) for ds, d in deltas.items()}
            consistent = len(set(signs.values())) <= 1
            consistency = ("consistent" if consistent else
                           "⚠ INCONSISTENT across dev — possible g7-tuning artifact")
            bands[arm_name] = {"deltas": deltas,
                               "pooled_delta": base_gt_pooled - arm_gt_pooled,
                               "band": band, "consistent": consistent,
                               "flip_rate_lt_pooled": arm["pooled"]["report"].get("flip_rate_lt")}
            md.append(f"| {arm_name} "
                      + " | ".join(f"{deltas.get(ds, 0.0):+.3f}"
                                   for ds in DEV_DATASETS)
                      + f" | {bands[arm_name]['pooled_delta']:+.3f} | {band} "
                      f"| {consistency} |")
        md.append("")

    # ---- A2 symmetry sanity (measurement guard for the mirror signature) ----
    a2_sanity: dict[str, dict] = {}
    if "A2" in arms:
        md.append("## A2 symmetry sanity (measurement guard — slot-symmetric "
                  "classes must be invariant under presentation swap)\n")
        md.append("| Dataset | shared pairs | v2 '=' | A2 '=' | Δrel | "
                  "'='-retention | v2 none | A2 none | Δrel | flag |")
        md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
                  "| ---: | --- |")
        for ds, a2_run in arms["A2"]["runs"].items():
            b_run = base["runs"].get(ds)
            if b_run is None:
                continue
            s = _a2_symmetry_sanity(b_run["rows"], a2_run["rows"])
            a2_sanity[ds] = s
            md.append(f"| {ds} | {s['n_shared_pairs']} | {s['v2_eq']} "
                      f"| {s['a2_eq']} | {_fmt(s['eq_rel_shift'], '+.1%')} "
                      f"| {_fmt(s['eq_retention'], '.3f')} | {s['v2_none']} "
                      f"| {s['a2_none']} | {_fmt(s['none_rel_shift'], '+.1%')} "
                      f"| {'⚠ >10% — mirror flip_rate_lt arm carries noise' if s['flagged'] else 'ok'} |")
        md.append("")

    # ---- flip-set resolution ----
    md.append(f"## Named g7 flip-set ({len(flip_set)} gold-'>' pairs predicted "
              "'<' by the baseline)\n")
    md.append("| Arm | resolved (→'>') | persisted (→'<') | →'=' | dropped | other |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for arm_name, arm in arms.items():
        if g7 in arm["runs"]:
            res = _flip_resolution(arm["runs"][g7]["rows"], flip_set)
            md.append(f"| {arm_name} | {res['resolved']} | {res['persisted']} "
                      f"| {res['equivalent']} | {res['dropped']} | {res['other']} |")
    md.append("")

    # ---- attribution verdict ----
    verdict = "PENDING — arms missing"
    if "A1" in bands and "A2" in bands:
        a1b = bands["A1"]["band"]
        a2b = bands["A2"]["band"]
        base_lt = base["pooled"]["report"].get("flip_rate_lt") or 0.0
        a2_lt = bands["A2"]["flip_rate_lt_pooled"] or 0.0
        mirror = (bands["A2"]["pooled_delta"] >= MIRROR_THRESHOLD
                  and (a2_lt - base_lt) >= MIRROR_THRESHOLD)
        h_label = a1b in (SOLID, SMALL) and a2b == NO_BAND
        h_content = a1b == NO_BAND and a2b == NO_BAND
        a3_trigger = (bands["A1"]["pooled_delta"] >= 0.05
                      and bands["A2"]["pooled_delta"] >= 0.05)
        md.append("## Attribution verdict (registered signatures)\n")
        md.append(f"- H-label (A1 SOLID/SMALL ∧ A2 NO): **{h_label}**")
        md.append(f"- H-position mirror (A2 Δgt ≥ {MIRROR_THRESHOLD} ∧ "
                  f"Δlt rise ≥ {MIRROR_THRESHOLD}): **{mirror}** "
                  f"(Δgt {bands['A2']['pooled_delta']:+.3f}, "
                  f"lt {base_lt:.3f}→{a2_lt:.3f})")
        md.append(f"- H-content (A1 NO ∧ A2 NO): **{h_content}**")
        md.append(f"- A3 trigger (both ≥ 0.05): **{a3_trigger}**")
        if h_label:
            verdict = ("H-label — the prior tracks label-list position; the "
                       "downstream fix is prompt-side (order-neutral labels).")
        elif mirror:
            verdict = ("H-position — the prior follows the presented argument "
                       "order; the fix is presentation-side (e.g. both-order "
                       "voting), not wording.")
        elif h_content:
            verdict = ("H-content — a genuine semantic prior, robust to "
                       "presentation; content levers / decomposition / "
                       "reasoners are the next stage.")
        else:
            verdict = ("MIXED — no single registered signature fires; "
                       "decision stays with Antonio"
                       + (" (A3 trigger FIRED — submit A3)" if a3_trigger else ""))
        inconsistent = [a for a, b in bands.items() if not b["consistent"]]
        if inconsistent:
            verdict += f" [⚠ direction-inconsistent across dev: {inconsistent}]"
        md.append(f"\n**Verdict: {verdict}**\n")

    if all_deviations:
        md.append("## ⚠ Cross-validation deviations vs stored metrics.json\n")
        md.extend(f"- {d}" for d in all_deviations)
    else:
        md.append(f"*Cross-validation vs stored metrics.json: all checked keys "
                  f"exact across {sum(len(a['per_ds']) for a in arms.values())} runs.*")

    out_md = Path(f"{args.out_prefix}.md")
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json = Path(f"{args.out_prefix}.json")
    out_json.write_text(json.dumps({
        "arms": {a: {"per_ds": {ds: {k: v for k, v in m.items() if k != "report"}
                                | {"report": m["report"]}
                                for ds, m in arm["per_ds"].items()},
                     "pooled": arm["pooled"]}
                 for a, arm in arms.items()},
        "bands": bands, "verdict": verdict,
        "a2_symmetry_sanity": a2_sanity,
        "flip_set": sorted(flip_set),
    }, indent=2, default=str), encoding="utf-8")
    print("\n" + "\n".join(md))
    logger.info("written: %s + %s", out_md, out_json)
    if all_deviations:
        sys.exit(5)


if __name__ == "__main__":
    main()
