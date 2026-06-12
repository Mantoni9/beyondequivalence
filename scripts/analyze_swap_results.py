"""
analyze_swap_results.py — Phase-3 offline analysis over swap_*_<sha> run dirs
(passes.tsv + local gold). CPU-only: no GPU, no model — runs on a login node
or locally once the run dirs are rsynced.

Implements the pre-registered analysis incl. the 2026-06-12 amendment (filed
before job-255613 results were seen):
  - variants {baseline, v_sym, v_3pass, v_union} via identical machinery;
  - pair coverage per relation at K in {5,10,20,50}, per dataset + micro-pooled;
  - guards (relative): pooled '<'/'=' coverage@20 drop <= 0.02 vs baseline,
    pooled volume <= 1.3 x baseline pairs@20;
  - t-side budget sweep (volume valve): ALL t-side passes of a variant capped
    at Kt in {5,10,20} with s-side passes at 20, for v_sym/v_3pass/v_union.
    Registered prediction (v_sym, v_3pass): fan-in lists concentrate gold at
    low ranks, so Kt=10 loses < 0.02 >-coverage vs Kt=20;
  - candidate-set registration (2026-06-12, pre-unblinding): adoption
    candidates are EXACTLY {v_sym, v_3pass, v_union} x Kt in {20, 10}; Kt=5
    rows are reporting-only. Each candidate is evaluated at its own budget
    against the unchanged absolute band and guards; the volume cap is an
    absolute Stage-2 cost ceiling (1.3 x baseline pairs@20) that does NOT
    shrink with Kt. Precedence: highest >-coverage; ties (delta < 0.01) break
    toward lower volume. A v_sym '<'-guard trip is a FINDING (quantifies the
    s_narrower cross-rescue), not a failure;
  - the decision section runs ONLY for the primary config (--primary-config,
    default qwen3-noLoRA); other configs are reported in full but are never
    adoption candidates. If Nemo+LoRA's best swap candidate beats the
    primary's >-coverage by >= 0.05 pooled, an escalation flag is raised
    (model-freeze question to Antonio) — configs are never auto-blended;
  - outcome bands on the winner: SOLID >= 0.80 AND delta >= +0.05 over the
    @20 baseline AND guards pass; '>= 0.80 with delta < 0.05' = baseline
    adequate, no switch; PARTIAL 0.65-0.80; NO EFFECT < 0.65; REVERSE = no
    candidate passes guards -> investigate before adoption;
  - mechanism check (secondary, no threshold): per-dataset delta >-coverage@20
    predicted to concentrate on mouse-human / g3-text / g5-groceries and be
    ~0 on g1-web / g2-diseases / g7-literature (at ceiling);
  - per_directed_query recall pooled from the pass lists (TSV ranks are
    authoritative); pooled provenance crosstab x gold relation (log only);
  - cross-validation: recomputed {covered, n, pairs} for every variant present
    in each run's metrics.json must match exactly.

Legacy per_relation_strict is READ from metrics.json, never recomputed from
passes.tsv: stored scores are 6-dp rounded and recomputation could rank-flip
inside tie clusters. The rank column is used for capping only.

Usage:
    conda run -n melt-olala python scripts/analyze_swap_results.py --sha <sha> \
        [--configs qwen3-noLoRA nemo+LoRA] [--datasets ...] [--results-root results]
Outputs: results/swap_analysis_<sha>.md + .json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# scripts/ is on sys.path[0] when run as `python scripts/<file>.py`; add repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Alignment import Alignment
from tracks.zenodo_loader import load_subdataset
from evaluation_recall import (
    _normalize_relation,
    compute_pair_coverage,
    compute_per_directed_query_recall,
)
from swap_retrieval import (
    PASS_SPECS,
    VARIANTS,
    assemble_variant,
    candidate_pairs_at_budget,
    candidate_pairs_at_mixed_budget,
    read_passes_tsv,
)

logger = logging.getLogger("analyze_swap_results")

DEFAULT_DATASETS = (
    "mouse-human", "g1-web", "g2-diseases", "g3-text", "g5-groceries", "g7-literature",
)
# config name -> (alias used in run-dir names, lora tag)
CONFIG_DIRS = {
    "qwen3-noLoRA": ("qwen3-embedding-8b", "lora-off"),
    "nemo+LoRA":    ("llama-embed-nemotron-8b", "lora-on"),
    "sbert-noLoRA": ("sbert", "lora-off"),
}

BUDGET_KS = (5, 10, 20, 50)
PDQ_KS = (1, 5, 10, 20, 50)
BUDGET = 20                      # the registered Stage-2 budget
T_SWEEP_KS = (5, 10, 20)         # amendment point 2; Kt=5 is REPORTING ONLY
# Candidate-set registration 2026-06-12 (filed pre-unblinding): the adoption
# candidates are EXACTLY CANDIDATE_VARIANTS x ADOPTION_KTS (s-side fixed @20).
CANDIDATE_VARIANTS = ("v_sym", "v_3pass", "v_union")
ADOPTION_KTS = (20, 10)
PREDICTION_VARIANTS = ("v_sym", "v_3pass")   # registered Kt=10 prediction scope
GUARD_MAX_DROP = 0.02
VOLUME_CAP_RATIO = 1.3
PRECEDENCE_TIE = 0.01
RELS = ("subclass", "superclass", "equivalence")
# Mechanism prediction groups (amendment point 5 keeps these unchanged).
PREDICTED_GAIN = ("mouse-human", "g3-text", "g5-groceries")
PREDICTED_CEILING = ("g1-web", "g2-diseases", "g7-literature")


def _query_lists(rows, *, retrieved: str) -> dict[str, list[str]]:
    by_query: dict[str, list[str]] = {}
    for r in sorted(rows, key=lambda r: (r.query_uri, r.rank)):
        by_query.setdefault(r.query_uri, []).append(
            r.source_uri if retrieved == "source" else r.target_uri)
    return by_query


def _pool(acc: dict, key, cov: dict) -> None:
    for rel in RELS:
        b = acc.setdefault(key, {}).setdefault(rel, {"covered": 0, "n": 0})
        b["covered"] += cov[rel]["covered"]
        b["n"] += cov[rel]["n"]


def _ratio(bucket: dict) -> float | None:
    return (bucket["covered"] / bucket["n"]) if bucket["n"] else None


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}"


BAND_TEXT = {
    "SOLID": "SOLID — adopt; Stage-2 candidate TSVs change → bridge run needed",
    "NO_SWITCH": "baseline adequate — no switch (≥ 0.80 but Δ < 0.05)",
    "PARTIAL": "PARTIAL — decision escalates to Antonio",
    "NO_EFFECT": "NO EFFECT — negative result; baseline stays",
    "REVERSE": "REVERSE — no candidate passes guards; investigate before adoption",
}


def evaluate_candidates(pooled_sweep: dict, pooled_sweep_pairs: dict,
                        base20: dict, base_pairs20: int):
    """Registered candidate-set decision (registration 2026-06-12, filed
    before job-255613 unblinding). Candidates are EXACTLY
    CANDIDATE_VARIANTS x ADOPTION_KTS, each evaluated at its own budget
    (s@20, t@Kt): >-coverage against the unchanged absolute band, guards
    ('<'/'=' pooled drop <= GUARD_MAX_DROP vs the same-run @20 baseline)
    and the ABSOLUTE volume cap (VOLUME_CAP_RATIO x baseline pairs@20 —
    a Stage-2 cost ceiling that does NOT shrink with Kt). Kt=5 rows are
    reporting-only and never enter this function's candidate set.
    Precedence: highest >-coverage; ties (delta < PRECEDENCE_TIE) break
    toward lower volume. Returns (candidates, winner_id, band_code) with
    band_code in {SOLID, NO_SWITCH, PARTIAL, NO_EFFECT, REVERSE}."""
    volume_cap = VOLUME_CAP_RATIO * base_pairs20
    base_sub = _ratio(base20["subclass"])
    base_eq = _ratio(base20["equivalence"])
    base_sup = _ratio(base20["superclass"])

    candidates: dict[str, dict] = {}
    for variant in CANDIDATE_VARIANTS:
        for kt in ADOPTION_KTS:
            cov = pooled_sweep[(variant, kt)]
            pairs = pooled_sweep_pairs[(variant, kt)]
            sup = _ratio(cov["superclass"])
            drop_sub = base_sub - _ratio(cov["subclass"])
            drop_eq = base_eq - _ratio(cov["equivalence"])
            # With the s-side fixed @20, v_3pass/v_union contain the full
            # baseline pass set at ANY Kt — their '<'/'=' guards pass by
            # construction (structural, no information). Only v_sym's are
            # empirically live. A positive drop on a structural candidate is
            # impossible -> flag as data corruption, never as a guard result.
            structural = variant != "v_sym"
            candidates[f"{variant}@t{kt}"] = {
                "variant": variant, "kt": kt,
                "sup_cov": sup, "delta": sup - base_sup,
                "drop_sub": drop_sub, "drop_eq": drop_eq,
                "pairs": pairs, "volume_ratio": pairs / base_pairs20,
                "guards_structural": structural,
                "structural_violation": structural and (drop_sub > 1e-12
                                                        or drop_eq > 1e-12),
                "passes": (drop_sub <= GUARD_MAX_DROP
                           and drop_eq <= GUARD_MAX_DROP
                           and pairs <= volume_cap),
            }

    passers = {cid: c for cid, c in candidates.items() if c["passes"]}
    winner = None
    if passers:
        best_sup = max(c["sup_cov"] for c in passers.values())
        tied = [cid for cid, c in passers.items()
                if best_sup - c["sup_cov"] < PRECEDENCE_TIE]
        winner = min(tied, key=lambda cid: passers[cid]["pairs"])

    if winner is None:
        band = "REVERSE"
    else:
        w = candidates[winner]
        if w["sup_cov"] >= 0.80 and w["delta"] >= 0.05:
            band = "SOLID"
        elif w["sup_cov"] >= 0.80:
            band = "NO_SWITCH"
        elif w["sup_cov"] >= 0.65:
            band = "PARTIAL"
        else:
            band = "NO_EFFECT"
    return candidates, winner, band


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-3 analysis over swap_* run dirs.")
    p.add_argument("--sha", required=True, help="git SHA suffix of the swap_* run dirs")
    p.add_argument("--configs", nargs="+", choices=sorted(CONFIG_DIRS),
                   default=["qwen3-noLoRA", "nemo+LoRA"])
    p.add_argument("--primary-config", default="qwen3-noLoRA", choices=sorted(CONFIG_DIRS),
                   help="Only this config gets the adoption decision; all others "
                        "are reported in full but are never adoption candidates.")
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--results-root", default="results")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s")
    results_root = Path(args.results_root)

    references: dict[str, Alignment] = {}
    for ds in args.datasets:
        _s, _t, ref_path = load_subdataset(ds)
        references[ds] = Alignment(str(ref_path))

    out: dict = {"sha": args.sha, "configs": {}, "validation": []}
    md: list[str] = [f"# Swap ablation — Phase-3 analysis over swap_*_{args.sha}\n"]
    md.append("Variants: baseline = s_broader+s_narrower (frozen) · v_sym = "
              "s_broader+t_broader · v_3pass = +s_narrower (amendment 2026-06-12) · "
              "v_union = all four. Coverage = pair-level gold coverage at the "
              "per-(query, pass) budget cut. Pooled = micro (Σcovered/Σn).\n")

    for cfg_name in args.configs:
        alias, lora_tag = CONFIG_DIRS[cfg_name]
        per_ds: dict[str, dict] = {}
        pooled_cov: dict = {}       # (variant, k) -> {rel: {covered, n}}
        pooled_pairs: dict = {}     # (variant, k) -> int
        pooled_sweep: dict = {}     # (variant, kt) -> {rel: {covered, n}}
        pooled_sweep_pairs: dict = {}
        pooled_sweep_pass_pairs: dict = {}   # (variant, kt, pass_id) -> n pairs
        pdq_acc: dict = {}          # label -> {k: hits, n}
        crosstab_pool: dict = {}
        legacy_pool_note: dict = {}

        for ds in args.datasets:
            run_dir = results_root / f"swap_{alias}_{lora_tag}_{ds}_{args.sha}"
            rows = read_passes_tsv(run_dir / "passes.tsv")
            passes: dict[str, list] = {}
            for r in rows:
                passes.setdefault(r.pass_id, []).append(r)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            reference = references[ds]

            ds_out: dict = {"variants": {}, "sweep": {}, "volume": metrics.get("volume", {}),
                            "identity": metrics.get("identity_check", {}).get("status")}

            # --- variants: coverage + volume at each K (+ cross-validation) ---
            for variant in VARIANTS:
                vrows = assemble_variant(passes, variant)
                v_out = {"coverage": {}, "pairs": {}}
                for k in BUDGET_KS:
                    pair_set = candidate_pairs_at_budget(vrows, k)
                    cov = compute_pair_coverage(reference, pair_set)
                    v_out["coverage"][k] = cov
                    v_out["pairs"][k] = len(pair_set)
                    _pool(pooled_cov, (variant, k), cov)
                    pooled_pairs[(variant, k)] = pooled_pairs.get((variant, k), 0) + len(pair_set)
                ds_out["variants"][variant] = v_out

                stored = metrics.get("variants", {}).get(variant)
                if stored is not None:
                    max_diff = 0
                    for k in BUDGET_KS:
                        sk = stored["pair_coverage"][str(k)]
                        for rel in RELS:
                            max_diff = max(max_diff,
                                           abs(sk[rel]["covered"] - v_out["coverage"][k][rel]["covered"]),
                                           abs(sk[rel]["n"] - v_out["coverage"][k][rel]["n"]))
                        max_diff = max(max_diff,
                                       abs(stored["pairs_at_budget"][str(k)] - v_out["pairs"][k]))
                    out["validation"].append({"config": cfg_name, "dataset": ds,
                                              "variant": variant, "max_int_diff": max_diff})
                    if max_diff:
                        logger.error("CROSS-VALIDATION MISMATCH %s/%s/%s: %d",
                                     cfg_name, ds, variant, max_diff)

            # --- t-side budget sweep (s-side fixed at BUDGET). Kt caps ALL
            #     t-side passes of the variant — t_broader, plus t_narrower
            #     for v_union. Kt=5 rows are reporting-only (registered). ---
            for variant in CANDIDATE_VARIANTS:
                for kt in T_SWEEP_KS:
                    k_by_pass = {
                        pid: (kt if PASS_SPECS[pid].query_side == "target" else BUDGET)
                        for pid in VARIANTS[variant]
                    }
                    pair_set = candidate_pairs_at_mixed_budget(passes, k_by_pass)
                    cov = compute_pair_coverage(reference, pair_set)
                    per_pass = {pid: len(candidate_pairs_at_budget(passes.get(pid, []), k))
                                for pid, k in k_by_pass.items()}
                    ds_out["sweep"][f"{variant}@t{kt}"] = {
                        "coverage": cov, "pairs": len(pair_set),
                        "pairs_per_pass": per_pass}
                    _pool(pooled_sweep, (variant, kt), cov)
                    pooled_sweep_pairs[(variant, kt)] = \
                        pooled_sweep_pairs.get((variant, kt), 0) + len(pair_set)
                    for pid, n_p in per_pass.items():
                        key = (variant, kt, pid)
                        pooled_sweep_pass_pairs[key] = \
                            pooled_sweep_pass_pairs.get(key, 0) + n_p

            # --- per_directed_query (pooled) ---
            pdq = compute_per_directed_query_recall(
                reference,
                _query_lists(passes.get("s_broader", []), retrieved="target"),
                _query_lists(passes.get("t_broader", []), retrieved="source"),
                k_values=PDQ_KS)
            for label in ("subclass", "superclass"):
                acc = pdq_acc.setdefault(label, {"n": 0, "hits": {k: 0 for k in PDQ_KS}})
                acc["n"] += pdq["n"][label]
                for k in PDQ_KS:
                    acc["hits"][k] += pdq["hits_at_k"][label][k]

            # --- provenance crosstab (pooled; log only) ---
            for bucket, by_rel in metrics.get("provenance_crosstab_at_20", {}).items():
                tgt = crosstab_pool.setdefault(bucket, {"<": 0, ">": 0, "=": 0})
                for rel, n in by_rel.items():
                    tgt[rel] += n

            legacy_pool_note[ds] = metrics.get("legacy_per_relation_strict", {})
            per_ds[ds] = ds_out

        # ---------- pooled tables ----------
        md.append(f"## {cfg_name}" + (
            "  *(robustness side-run — does NOT reopen the model freeze)*"
            if cfg_name == "nemo+LoRA" else "  *(primary — frozen Stage-1 model)*") + "\n")

        base20 = pooled_cov[("baseline", BUDGET)]
        base_pairs20 = pooled_pairs[("baseline", BUDGET)]
        volume_cap = VOLUME_CAP_RATIO * base_pairs20

        md.append("### Pooled variants @ budget 20 (guards in brackets)\n")
        md.append("| Variant | >cov@5 | >cov@10 | >cov@20 | >cov@50 | <cov@20 | =cov@20 "
                  "| pairs@20 | ×base | guards |")
        md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        guard_status: dict[str, dict] = {}
        for variant in VARIANTS:
            cov20 = pooled_cov[(variant, BUDGET)]
            pairs20 = pooled_pairs[(variant, BUDGET)]
            drop_sub = _ratio(base20["subclass"]) - _ratio(cov20["subclass"])
            drop_eq = _ratio(base20["equivalence"]) - _ratio(cov20["equivalence"])
            vol_ratio = pairs20 / base_pairs20
            passes_guards = (variant == "baseline") or (
                drop_sub <= GUARD_MAX_DROP and drop_eq <= GUARD_MAX_DROP
                and pairs20 <= volume_cap)
            guard_status[variant] = {
                "drop_sub": drop_sub, "drop_eq": drop_eq, "volume_ratio": vol_ratio,
                "passes": passes_guards,
                "sup_cov20": _ratio(cov20["superclass"]),
            }
            guard_str = "—" if variant == "baseline" else (
                ("PASS" if passes_guards else "FAIL") +
                f" (<drop {drop_sub:+.3f}, =drop {drop_eq:+.3f}, vol {vol_ratio:.2f}×)")
            md.append(
                f"| {variant} | "
                + " | ".join(_fmt(_ratio(pooled_cov[(variant, k)]["superclass"]))
                             for k in BUDGET_KS)
                + f" | {_fmt(_ratio(cov20['subclass']))} | {_fmt(_ratio(cov20['equivalence']))} "
                + f"| {pairs20} | {pairs20 / base_pairs20:.2f} | {guard_str} |")
        md.append("")

        # ---------- per-dataset mechanism table ----------
        md.append("### Per-dataset >-coverage@20 (mechanism check — predicted gains on "
                  "mouse-human/g3-text/g5-groceries, ≈0 on g1/g2/g7)\n")
        md.append("| Dataset | predicted | n> | baseline | v_sym | Δv_sym | v_3pass | Δv_3pass "
                  "| v_union | nS | nT | base pairs@20 | v_sym pairs@20 |")
        md.append("| --- | --- | ---: " + "| ---: " * 10 + "|")
        for ds in args.datasets:
            d = per_ds[ds]
            covs = {v: _ratio(d["variants"][v]["coverage"][BUDGET]["superclass"])
                    for v in VARIANTS}
            n_sup = d["variants"]["baseline"]["coverage"][BUDGET]["superclass"]["n"]
            pred = "gain" if ds in PREDICTED_GAIN else (
                "ceiling" if ds in PREDICTED_CEILING else "?")
            vol = d.get("volume", {})
            md.append(
                f"| {ds} | {pred} | {n_sup} | {_fmt(covs['baseline'])} "
                f"| {_fmt(covs['v_sym'])} | {covs['v_sym'] - covs['baseline']:+.3f} "
                f"| {_fmt(covs['v_3pass'])} | {covs['v_3pass'] - covs['baseline']:+.3f} "
                f"| {_fmt(covs['v_union'])} | {vol.get('n_source_classes', '—')} "
                f"| {vol.get('n_target_classes', '—')} "
                f"| {d['variants']['baseline']['pairs'][BUDGET]} "
                f"| {d['variants']['v_sym']['pairs'][BUDGET]} |")
        md.append("")

        # ---------- t-side budget sweep ----------
        md.append("### t-side budget sweep (s-side @20; Kt caps ALL t-side passes; "
                  "Kt=5 is reporting-only, never adoptable; registered prediction: "
                  "Kt=10 loses < 0.02 >-coverage vs Kt=20)\n")
        md.append("Per-pass columns show each pass's pair count at the candidate's "
                  "budgets (s-side @20, t-side @Kt) — the union total is ≤ their sum "
                  "because pairs overlap across passes. This makes visible which "
                  "pass the Kt valve squeezes (for v_union both t-side passes).\n")
        md.append("| Variant | Kt | adoptable | >cov | <cov | =cov "
                  "| sB pairs | sN pairs | tB pairs | tN pairs | pairs (∪) | ×base |")
        md.append("| --- | ---: | --- | ---: | ---: | ---: "
                  "| ---: | ---: | ---: | ---: | ---: | ---: |")
        sweep_summary = {}
        for variant in CANDIDATE_VARIANTS:
            for kt in T_SWEEP_KS:
                cov = pooled_sweep[(variant, kt)]
                pairs = pooled_sweep_pairs[(variant, kt)]
                sweep_summary[(variant, kt)] = _ratio(cov["superclass"])
                per_pass_cells = " | ".join(
                    str(pooled_sweep_pass_pairs[(variant, kt, pid)])
                    if (variant, kt, pid) in pooled_sweep_pass_pairs else "—"
                    for pid in ("s_broader", "s_narrower", "t_broader", "t_narrower"))
                md.append(f"| {variant} | {kt} | {'yes' if kt in ADOPTION_KTS else 'no'} "
                          f"| {_fmt(_ratio(cov['superclass']))} "
                          f"| {_fmt(_ratio(cov['subclass']))} "
                          f"| {_fmt(_ratio(cov['equivalence']))} "
                          f"| {per_pass_cells} "
                          f"| {pairs} | {pairs / base_pairs20:.2f} |")
        for variant in PREDICTION_VARIANTS:
            loss = sweep_summary[(variant, 20)] - sweep_summary[(variant, 10)]
            md.append(f"\n- {variant}: Kt=10 vs Kt=20 >-coverage loss = {loss:+.4f} → "
                      f"prediction (< 0.02) {'HOLDS' if loss < 0.02 else 'FAILS'}")
        md.append("")

        # ---------- per_directed_query pooled ----------
        md.append("### per_directed_query (pooled, pass-level — variant-independent)\n")
        md.append("| Direction | n | " + " | ".join(f"R@{k}" for k in PDQ_KS) + " |")
        md.append("| --- | ---: | " + " | ".join("---:" for _ in PDQ_KS) + " |")
        for label in ("subclass", "superclass"):
            acc = pdq_acc[label]
            md.append(f"| {label} | {acc['n']} | "
                      + " | ".join(f"{acc['hits'][k] / acc['n']:.3f}" if acc["n"] else "—"
                                   for k in PDQ_KS) + " |")
        md.append("")

        # ---------- provenance crosstab ----------
        md.append("### Provenance crosstab @20 (pooled; log only — no claims)\n")
        md.append("| Pass combination | gold < | gold > | gold = |")
        md.append("| --- | ---: | ---: | ---: |")
        for bucket in sorted(crosstab_pool):
            row = crosstab_pool[bucket]
            md.append(f"| {bucket} | {row['<']} | {row['>']} | {row['=']} |")
        md.append("")

        # ---------- decision (registered candidate set; primary config only) ----------
        candidates, winner, band = evaluate_candidates(
            pooled_sweep, pooled_sweep_pairs, base20, base_pairs20)
        base_sup = _ratio(base20["superclass"])
        if cfg_name == args.primary_config:
            md.append("### Pre-registered decision — candidate set = "
                      "{v_sym, v_3pass, v_union} × Kt ∈ {20, 10} (registration "
                      "2026-06-12; Kt=5 reporting-only; each candidate at its own "
                      "budget; volume cap absolute)\n")
            md.append(f"- Baseline pooled >-coverage@20: **{_fmt(base_sup)}** "
                      f"({base20['superclass']['covered']}/{base20['superclass']['n']}); "
                      f"volume base {base_pairs20} (absolute cap {volume_cap:.0f}).")
            md.append("")
            md.append("**Structural guard note:** with the s-side fixed @20, v_3pass "
                      "and v_union are supersets of the baseline pass set at ANY Kt — "
                      "their '<'/'=' guards pass **by construction** and carry no "
                      "information (marked *struct* below). The empirically live "
                      "guard questions are exactly: (a) v_sym's '<'/'=' (quantifying "
                      "the s_narrower cross-rescue) and (b) the volume cap for every "
                      "candidate.")
            md.append("")
            md.append("| Candidate | >cov | Δ vs base | <drop | =drop | pairs | ×base | guards |")
            md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
            for cid, c in sorted(candidates.items(),
                                 key=lambda kv: -kv[1]["sup_cov"]):
                tripped = [g for g, hit in (
                    ("<", c["drop_sub"] > GUARD_MAX_DROP),
                    ("=", c["drop_eq"] > GUARD_MAX_DROP),
                    ("vol", c["pairs"] > volume_cap)) if hit]
                if c["guards_structural"]:
                    cell = (("PASS" if c["passes"] else "FAIL: " + ",".join(tripped))
                            + " — <,= struct, vol live")
                    if c["structural_violation"]:
                        cell += " ⚠ STRUCTURAL VIOLATION (data corrupt?)"
                else:
                    cell = ("PASS — all live" if c["passes"]
                            else "FAIL: " + ",".join(tripped))
                md.append(f"| {cid} | {_fmt(c['sup_cov'])} | {c['delta']:+.3f} "
                          f"| {c['drop_sub']:+.3f} | {c['drop_eq']:+.3f} "
                          f"| {c['pairs']} | {c['volume_ratio']:.2f} | {cell} |")
            md.append("")
            if winner:
                md.append(f"- **Winner: {winner}** (highest >-coverage among "
                          f"guard-passers; ties Δ<{PRECEDENCE_TIE} → lower volume) → "
                          f"**{BAND_TEXT[band]}**")
            else:
                md.append(f"- **{BAND_TEXT['REVERSE']}**")
        else:
            md.append("### Decision: n/a — side-run\n")
            best_cid = max(candidates, key=lambda cid: candidates[cid]["sup_cov"])
            md.append(f"- {cfg_name} is reported in full but is never an adoption "
                      f"candidate (candidate-set registration 2026-06-12). Best "
                      f"candidate: {best_cid} with >-coverage "
                      f"{_fmt(candidates[best_cid]['sup_cov'])}.")
        md.append("")

        out["configs"][cfg_name] = {
            "per_dataset": per_ds,
            "pooled_coverage": {f"{v}@{k}": {rel: dict(b) for rel, b in d.items()}
                                for (v, k), d in pooled_cov.items()},
            "pooled_pairs": {f"{v}@{k}": n for (v, k), n in pooled_pairs.items()},
            "sweep_pooled": {f"{v}@t{kt}": {rel: dict(b) for rel, b in d.items()}
                             for (v, kt), d in pooled_sweep.items()},
            "sweep_pairs": {f"{v}@t{kt}": n for (v, kt), n in pooled_sweep_pairs.items()},
            "sweep_pass_pairs": {f"{v}@t{kt}/{pid}": n
                                 for (v, kt, pid), n in pooled_sweep_pass_pairs.items()},
            "pdq_pooled": pdq_acc,
            "crosstab_pooled": crosstab_pool,
            "guards_at_20": guard_status,
            "candidates": candidates,
            "winner": winner if cfg_name == args.primary_config else None,
            "band": band if cfg_name == args.primary_config else "n/a (side-run)",
            "legacy_per_relation_strict_per_dataset": legacy_pool_note,
        }

    # ---------- model-freeze escalation check (registration point 4) ----------
    if (args.primary_config in out["configs"] and "nemo+LoRA" in out["configs"]
            and args.primary_config != "nemo+LoRA"):
        p_best = max(c["sup_cov"]
                     for c in out["configs"][args.primary_config]["candidates"].values())
        n_best = max(c["sup_cov"]
                     for c in out["configs"]["nemo+LoRA"]["candidates"].values())
        out["escalation"] = {"primary_best_sup": p_best, "nemo_best_sup": n_best,
                             "flagged": n_best - p_best >= 0.05}
        if out["escalation"]["flagged"]:
            md.append(f"\n**ESCALATION FLAG (model freeze):** Nemo+LoRA's best swap "
                      f"candidate >-coverage ({n_best:.3f}) beats the primary's "
                      f"({p_best:.3f}) by ≥ 0.05 pooled — the model-freeze question "
                      f"goes to Antonio. Configs are NOT auto-blended.\n")
        else:
            md.append(f"\n*No model-freeze escalation: Nemo+LoRA best swap-candidate "
                      f">-coverage {n_best:.3f} vs primary {p_best:.3f} "
                      f"(threshold +0.05).*\n")

    n_bad = sum(1 for v in out["validation"] if v["max_int_diff"])
    md.append(f"\n*Cross-validation vs runner metrics.json: "
              f"{len(out['validation'])} (config, dataset, variant) cells checked, "
              f"{n_bad} mismatches.*\n")

    md_text = "\n".join(md)
    md_path = results_root / f"swap_analysis_{args.sha}.md"
    json_path = results_root / f"swap_analysis_{args.sha}.json"
    md_path.write_text(md_text, encoding="utf-8")
    json_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\n" + md_text)
    logger.info("written: %s + %s", md_path, json_path)
    if n_bad:
        sys.exit(5)


if __name__ == "__main__":
    main()
