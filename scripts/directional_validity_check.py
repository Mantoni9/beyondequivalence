"""
directional_validity_check.py — registered directional-validity check for the
swap ablation (docs/swap_directional_validity_registration.md, committed
BEFORE this script produced numbers). Gate for the freeze-reopen decision.

All-offline, read-only over results/swap_*_<sha>/passes.tsv + local gold.
Hypothesis model: Nemo+LoRA. Negative control: Qwen3-noLoRA.

Check 1  rank distribution of '>' gold inside t_broader (per model; primary
         comparison on the SHARED covered set, pooled).
Check 2  per-pair symmetric consistency: fan-in (s in t_broader(t)@20) vs
         fan-out (t in s_narrower(s)@20); agreement gap Nemo - Qwen3.
Check 3  blind-audit material: 10 mouse-human + 10 g3-text gold '>' pairs
         covered by Nemo's t_broader@20 (seed 42), top-5 of t's broader list
         from BOTH models; blind CSV (no model column, shuffled) + separate
         un-blinding key. Judging is manual (Antonio), un-blinding together.

Outputs: results/swap_directional_validity_<sha>.md + .json,
         results/directional_audit_blind_<sha>.csv,
         results/directional_audit_key_<sha>.json.

Run: conda run -n melt-olala python scripts/directional_validity_check.py --sha e98c0b3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Alignment import Alignment
from tracks.zenodo_loader import load_subdataset
from evaluation_recall import _normalize_relation
from run_subsumption_experiment import _load_kg_with_labels
from swap_retrieval import read_passes_tsv

logger = logging.getLogger("directional_validity")

DATASETS = (
    "mouse-human", "g1-web", "g2-diseases", "g3-text", "g5-groceries", "g7-literature",
)
CONFIG_DIRS = {
    "nemo+LoRA":    ("llama-embed-nemotron-8b", "lora-on"),    # hypothesis model
    "qwen3-noLoRA": ("qwen3-embedding-8b", "lora-off"),        # negative control
}
BUDGET = 20
TOP_AUDIT = 5

# Registered bars (docs/swap_directional_validity_registration.md).
C1_GENUINE_MEDIAN = 5
C1_GENUINE_R13 = 0.35
C1_GENUINE_MEDIAN_GAP = 3
C1_RIDEALONG_MEDIAN = 8
C1_INDIST_MEDIAN_GAP = 2
C1_INDIST_R13_GAP = 0.10
C2_CORROBORATE_GAP = 0.15
C2_UNINFORMATIVE_GAP = 0.05


def _hash_id(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:8]


def _rank_stats(ranks: list[int]) -> dict:
    if not ranks:
        return {"n": 0, "median": None, "r1_3": None, "r4_10": None, "r11_20": None,
                "histogram": {}}
    n = len(ranks)
    return {
        "n": n,
        "median": statistics.median(ranks),
        "r1_3": sum(1 for r in ranks if r <= 3) / n,
        "r4_10": sum(1 for r in ranks if 4 <= r <= 10) / n,
        "r11_20": sum(1 for r in ranks if r >= 11) / n,
        "histogram": {str(r): sum(1 for x in ranks if x == r) for r in range(1, 21)},
    }


def _fmt(v, spec=".3f") -> str:
    return "—" if v is None else format(v, spec)


def main() -> None:
    p = argparse.ArgumentParser(description="Registered directional-validity check.")
    p.add_argument("--sha", default="e98c0b3")
    p.add_argument("--results-root", default="results")
    p.add_argument("--audit-datasets", nargs="+", default=["mouse-human", "g3-text"])
    p.add_argument("--audit-n", type=int, default=10, help="pairs per audit dataset")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s")
    results_root = Path(args.results_root)

    # ---------------- load passes + gold ----------------
    # lookups[config][ds] = {"tb": {(t, s): rank}, "sn": {(s, t): rank}}
    lookups: dict[str, dict[str, dict]] = {}
    for cfg, (alias, lora_tag) in CONFIG_DIRS.items():
        lookups[cfg] = {}
        for ds in DATASETS:
            run_dir = results_root / f"swap_{alias}_{lora_tag}_{ds}_{args.sha}"
            rows = read_passes_tsv(run_dir / "passes.tsv")
            tb = {(r.query_uri, r.source_uri): r.rank for r in rows if r.pass_id == "t_broader"}
            sn = {(r.query_uri, r.target_uri): r.rank for r in rows if r.pass_id == "s_narrower"}
            lookups[cfg][ds] = {"tb": tb, "sn": sn}

    gold_sup: dict[str, list[tuple[str, str]]] = {}
    for ds in DATASETS:
        _s, _t, ref_path = load_subdataset(ds)
        reference = Alignment(str(ref_path))
        gold_sup[ds] = sorted({(c.source, c.target) for c in reference
                               if _normalize_relation(c.relation) == ">"})

    out: dict = {"sha": args.sha, "check1": {}, "check2": {}, "check3": {}}
    md: list[str] = []
    md.append(f"# Directional-validity check — swap_*_{args.sha}\n")
    md.append("Registration: `docs/swap_directional_validity_registration.md` "
              "(committed before computation). Hypothesis model: Nemo+LoRA; "
              "negative control: Qwen3-noLoRA. Covered := rank ≤ 20 in the pass.\n")

    # ---------------- Check 1 ----------------
    ranks_all: dict[str, dict[str, list[int]]] = {cfg: {} for cfg in CONFIG_DIRS}
    covered_sets: dict[str, dict[str, set]] = {cfg: {} for cfg in CONFIG_DIRS}
    for cfg in CONFIG_DIRS:
        for ds in DATASETS:
            tb = lookups[cfg][ds]["tb"]
            rs, cov = [], set()
            for s, t in gold_sup[ds]:
                r = tb.get((t, s))
                if r is not None and r <= BUDGET:
                    rs.append(r)
                    cov.add((s, t))
            ranks_all[cfg][ds] = rs
            covered_sets[cfg][ds] = cov

    shared_ranks: dict[str, list[int]] = {cfg: [] for cfg in CONFIG_DIRS}
    n_shared = 0
    for ds in DATASETS:
        shared = covered_sets["nemo+LoRA"][ds] & covered_sets["qwen3-noLoRA"][ds]
        n_shared += len(shared)
        for cfg in CONFIG_DIRS:
            tb = lookups[cfg][ds]["tb"]
            shared_ranks[cfg].extend(tb[(t, s)] for s, t in shared)

    md.append("## Check 1 — rank of the gold parent inside t_broader (covered '>' gold)\n")
    md.append("| Model | Set | n | median | ranks 1–3 | ranks 4–10 | ranks 11–20 |")
    md.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    c1: dict[str, dict] = {}
    for cfg in CONFIG_DIRS:
        for ds in DATASETS:
            st = _rank_stats(ranks_all[cfg][ds])
            c1.setdefault(cfg, {})[ds] = st
            md.append(f"| {cfg} | {ds} | {st['n']} | {_fmt(st['median'], '.1f')} "
                      f"| {_fmt(st['r1_3'])} | {_fmt(st['r4_10'])} | {_fmt(st['r11_20'])} |")
        pooled = _rank_stats([r for ds in DATASETS for r in ranks_all[cfg][ds]])
        c1[cfg]["pooled"] = pooled
        md.append(f"| **{cfg}** | **pooled (own covered)** | {pooled['n']} "
                  f"| **{_fmt(pooled['median'], '.1f')}** | **{_fmt(pooled['r1_3'])}** "
                  f"| {_fmt(pooled['r4_10'])} | {_fmt(pooled['r11_20'])} |")
        sh = _rank_stats(shared_ranks[cfg])
        c1[cfg]["shared_pooled"] = sh
        md.append(f"| **{cfg}** | **pooled (SHARED, primary)** | {sh['n']} "
                  f"| **{_fmt(sh['median'], '.1f')}** | **{_fmt(sh['r1_3'])}** "
                  f"| {_fmt(sh['r4_10'])} | {_fmt(sh['r11_20'])} |")
    md.append(f"\nShared covered pairs: {n_shared}. Uniform-over-20 reference: "
              "median 10.5, ranks-1–3 mass 0.15.\n")

    nemo_sh, qwen_sh = c1["nemo+LoRA"]["shared_pooled"], c1["qwen3-noLoRA"]["shared_pooled"]
    genuine_abs = (nemo_sh["median"] is not None and nemo_sh["median"] <= C1_GENUINE_MEDIAN
                   and nemo_sh["r1_3"] >= C1_GENUINE_R13)
    qwen_fails_abs = (qwen_sh["median"] is None or qwen_sh["median"] > C1_GENUINE_MEDIAN
                      or qwen_sh["r1_3"] < C1_GENUINE_R13)
    median_gap = (qwen_sh["median"] - nemo_sh["median"]
                  if None not in (qwen_sh["median"], nemo_sh["median"]) else None)
    r13_gap = (nemo_sh["r1_3"] - qwen_sh["r1_3"]
               if None not in (nemo_sh["r1_3"], qwen_sh["r1_3"]) else None)
    gap_ok = (median_gap is not None and median_gap >= C1_GENUINE_MEDIAN_GAP) or \
             (genuine_abs and qwen_fails_abs)
    indistinguishable = (median_gap is not None and abs(median_gap) < C1_INDIST_MEDIAN_GAP
                         and r13_gap is not None and abs(r13_gap) < C1_INDIST_R13_GAP)
    if genuine_abs and gap_ok:
        c1_verdict = "GENUINE"
    elif (nemo_sh["median"] is not None and nemo_sh["median"] >= C1_RIDEALONG_MEDIAN) \
            or indistinguishable:
        c1_verdict = "RIDE-ALONG"
    else:
        c1_verdict = "MIXED"
    out["check1"] = {"stats": c1, "median_gap": median_gap, "r1_3_gap": r13_gap,
                     "n_shared": n_shared, "verdict": c1_verdict}
    md.append(f"**Check-1 verdict vs registered bars (shared set): {c1_verdict}** — "
              f"Nemo median {_fmt(nemo_sh['median'], '.1f')} (bar ≤ {C1_GENUINE_MEDIAN}), "
              f"ranks-1–3 {_fmt(nemo_sh['r1_3'])} (bar ≥ {C1_GENUINE_R13}); median gap "
              f"{_fmt(median_gap, '.1f')} (bar ≥ {C1_GENUINE_MEDIAN_GAP}); "
              f"ranks-1–3 gap {_fmt(r13_gap)}.\n")

    # ---------------- Check 2 ----------------
    md.append("## Check 2 — symmetric consistency (fan-in vs fan-out, @20)\n")
    md.append("| Model | both | fan-in only | fan-out only | neither | agreement |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    c2: dict[str, dict] = {}
    for cfg in CONFIG_DIRS:
        both = fi_only = fo_only = neither = 0
        for ds in DATASETS:
            tb, sn = lookups[cfg][ds]["tb"], lookups[cfg][ds]["sn"]
            for s, t in gold_sup[ds]:
                fi = (r := tb.get((t, s))) is not None and r <= BUDGET
                fo = (r2 := sn.get((s, t))) is not None and r2 <= BUDGET
                if fi and fo:
                    both += 1
                elif fi:
                    fi_only += 1
                elif fo:
                    fo_only += 1
                else:
                    neither += 1
        denom = both + fi_only + fo_only
        agreement = both / denom if denom else None
        c2[cfg] = {"both": both, "fan_in_only": fi_only, "fan_out_only": fo_only,
                   "neither": neither, "agreement": agreement}
        md.append(f"| {cfg} | {both} | {fi_only} | {fo_only} | {neither} "
                  f"| {_fmt(agreement)} |")
    gap2 = (c2["nemo+LoRA"]["agreement"] - c2["qwen3-noLoRA"]["agreement"]
            if None not in (c2["nemo+LoRA"]["agreement"], c2["qwen3-noLoRA"]["agreement"])
            else None)
    if gap2 is None or abs(gap2) < C2_UNINFORMATIVE_GAP:
        c2_verdict = "UNINFORMATIVE"
    elif gap2 >= C2_CORROBORATE_GAP:
        c2_verdict = "CORROBORATES GENUINE"
    else:
        c2_verdict = "WEAK"
    out["check2"] = {**c2, "agreement_gap": gap2, "verdict": c2_verdict}
    md.append(f"\n**Check-2 reading: {c2_verdict}** — agreement gap Nemo−Qwen3 = "
              f"{_fmt(gap2)} (corroborates ≥ {C2_CORROBORATE_GAP}, uninformative "
              f"< {C2_UNINFORMATIVE_GAP}). Corroborative only.\n")

    # ---------------- Check 3 — blind audit material ----------------
    rng = random.Random(args.seed)
    audit_rows: list[dict] = []
    key: dict[str, dict] = {}
    sampled: dict[str, list] = {}
    for ds in args.audit_datasets:
        src_path, tgt_path, _ref = load_subdataset(ds)
        _kg_s, s_labels = _load_kg_with_labels(src_path)
        _kg_t, t_labels = _load_kg_with_labels(tgt_path)
        frame = sorted(covered_sets["nemo+LoRA"][ds])
        chosen = rng.sample(frame, min(args.audit_n, len(frame)))
        sampled[ds] = chosen
        for s, t in chosen:
            pair_id = _hash_id(ds, s, t)
            # top-5 of t's broader list per model
            per_model: dict[str, list[tuple[str, int]]] = {}
            for cfg in CONFIG_DIRS:
                tb = lookups[cfg][ds]["tb"]
                lst = sorted(((cand, r) for (q, cand), r in tb.items()
                              if q == t and r <= TOP_AUDIT), key=lambda e: e[1])
                per_model[cfg] = lst
            cand_models: dict[str, dict[str, int]] = {}
            for cfg, lst in per_model.items():
                for cand, r in lst:
                    cand_models.setdefault(cand, {})[cfg] = r
            for cand, models in sorted(cand_models.items()):
                row_id = _hash_id(ds, t, cand, pair_id)
                audit_rows.append({
                    "row_id": row_id, "dataset": ds, "pair_id": pair_id,
                    "t_label": t_labels.get(t) or t.rsplit("/", 1)[-1],
                    "candidate_label": s_labels.get(cand) or cand.rsplit("/", 1)[-1],
                })
                key[row_id] = {
                    "dataset": ds, "pair_id": pair_id, "t_uri": t,
                    "candidate_uri": cand, "models": models,
                    "gold_s_uri": s,
                    "gold_s_label": s_labels.get(s) or s.rsplit("/", 1)[-1],
                    "candidate_is_gold_s": cand == s,
                }
    rng.shuffle(audit_rows)
    blind_path = results_root / f"directional_audit_blind_{args.sha}.csv"
    with blind_path.open("w", encoding="utf-8") as f:
        f.write("row_id\tdataset\tpair_id\tt_label\tcandidate_label\tjudgment\n")
        for row in audit_rows:
            f.write(f"{row['row_id']}\t{row['dataset']}\t{row['pair_id']}"
                    f"\t{row['t_label']}\t{row['candidate_label']}\t\n")
    key_path = results_root / f"directional_audit_key_{args.sha}.json"
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    out["check3"] = {"n_pairs": {ds: len(v) for ds, v in sampled.items()},
                     "n_rows": len(audit_rows),
                     "blind_csv": str(blind_path), "key": str(key_path),
                     "verdict": "PENDING blind judging"}

    md.append("## Check 3 — blind audit material (judging pending)\n")
    md.append(f"- Sampled pairs: " + ", ".join(f"{ds}: {len(v)}" for ds, v in sampled.items())
              + f" (frame: Nemo-covered '>' gold; seed {args.seed}).")
    md.append(f"- Blind CSV: `{blind_path}` — {len(audit_rows)} unique (pair, candidate) "
              f"rows, top-{TOP_AUDIT} per model, model identity stripped, shuffled. "
              "Judgments: `parent` / `neighbor` / `unsure`.")
    md.append(f"- Un-blinding key (do not open before judging): `{key_path}`.\n")

    # ---------------- verdict ----------------
    md.append("## Verdict against the registered logic\n")
    md.append(f"- Check 1: **{c1_verdict}**; Check 2: {c2_verdict}; Check 3: pending.")
    if c1_verdict == "RIDE-ALONG":
        overall = ("**RIDE-ALONG / INCONCLUSIVE** — registered logic decides already "
                   "(Check 1 = RIDE-ALONG): no freeze-reopen on this evidence. The blind "
                   "audit can still be completed for the record.")
    elif c1_verdict == "GENUINE":
        overall = ("Pending Check 3: GENUINE iff the blind audit also lands GENUINE; "
                   "RIDE-ALONG if it lands RIDE-ALONG; otherwise MIXED.")
    else:
        overall = ("Pending Check 3; Check 1 = MIXED means the final verdict can be at "
                   "most MIXED (decision stays with Antonio).")
    md.append(f"- {overall}\n")

    md_text = "\n".join(md)
    md_path = results_root / f"swap_directional_validity_{args.sha}.md"
    json_path = results_root / f"swap_directional_validity_{args.sha}.json"
    md_path.write_text(md_text, encoding="utf-8")
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n" + md_text)
    logger.info("written: %s + %s", md_path, json_path)


if __name__ == "__main__":
    main()
