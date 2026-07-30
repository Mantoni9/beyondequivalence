"""
build_ltfp_audit_sample.py — stratified, MODEL-BLIND sample of '<'-FP gold-gap
candidates for manual judgment (matrix '<'-precision decomposition).

A '<'-FP gold-gap candidate = a pair a model predicted '<' where the gold has
NO relation (gold=none). 2738 (g3) such pairs is the UPPER BOUND of possible
gold gaps; the hand-audit measures the real fraction (real unlabelled subclass
vs model hallucination). This draws ~N pairs stratified across the 4 models
(so per-model gap-rates can be estimated), model identity hidden, with human-
readable concept labels, plus a separate un-blinding key.

Outputs:
  results/ltfp_audit_blind.tsv   (row_id, dataset, source_concept, target_concept, judgment[empty]) — shuffled, NO model column
  results/ltfp_audit_key.json    (row_id → dataset, URIs, labels, flagging models, gold=none)

Run: conda run -n melt-olala python scripts/build_ltfp_audit_sample.py --sha a24e146 --n 40
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Alignment import Alignment
from evaluation_recall import _normalize_relation
from tracks.zenodo_loader import load_subdataset

MODELS = ("llama", "mistral", "gemma4", "gpt-oss")
DATASETS = ("g3-text", "g5-groceries", "g7-literature")
RESULTS = Path("results")

# seed-42 cell dir per (model, dataset). Llama g7/g5 are the reused baselines.
REUSE = {
    ("llama", "g7-literature"): "2026-06-02_19-01-38_stage2_g7-literature_s1-qwen3-embedding-8b-asy-T2-description_path_context",
    ("llama", "g5-groceries"): "2026-06-13_11-32-15_stage2_g5-groceries_s1-qwen3-embedding-8b-asy-T2-description_path_context_p-d_subs_v2",
}


def _cell_dir(model: str, dataset: str, sha: str) -> Path | None:
    if (model, dataset) in REUSE:
        d = RESULTS / REUSE[(model, dataset)]
    else:
        d = RESULTS / f"matrix_{model}_{dataset}_seed42_{sha}"
    return d if (d / "predictions.tsv").is_file() else None


def _gold_none_lt_fps(d: Path, gold: dict) -> dict[tuple[str, str], tuple[str, str]]:
    """(s,t) → (source_label, target_label) for pairs predicted '<' (kept) with
    gold=none."""
    out = {}
    with (d / "predictions.tsv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("kept") != "True" or r.get("predicted_relation") != "<":
                continue
            key = (r["source_uri"], r["target_uri"])
            if key in gold:           # gold has a relation → not a gold-gap candidate
                continue
            sl = r.get("source_label") or r["source_uri"].rsplit("/", 1)[-1].rsplit("#", 1)[-1]
            tl = r.get("target_label") or r["target_uri"].rsplit("/", 1)[-1].rsplit("#", 1)[-1]
            out[key] = (sl, tl)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sha", default="a24e146")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    golds = {}
    for ds in DATASETS:
        _s, _t, ref = load_subdataset(ds)
        g = {}
        for c in Alignment(str(ref)):
            n = _normalize_relation(c.relation)
            if n:
                g[(c.source, c.target)] = n
        golds[ds] = g

    # pair → {models flagging it}, labels, dataset ; and per-model pair lists.
    info: dict[tuple[str, str, str], dict] = {}
    per_model: dict[str, list] = {m: [] for m in MODELS}
    for model in MODELS:
        for ds in DATASETS:
            d = _cell_dir(model, ds, args.sha)
            if d is None:
                print(f"WARN missing cell {model}/{ds}", file=sys.stderr)
                continue
            fps = _gold_none_lt_fps(d, golds[ds])
            for (s, t), (sl, tl) in fps.items():
                k = (ds, s, t)
                if k not in info:
                    info[k] = {"dataset": ds, "source_uri": s, "target_uri": t,
                               "source_label": sl, "target_label": tl, "models": set()}
                info[k]["models"].add(model)
                per_model[model].append(k)

    # Stratified round-robin across models for model coverage; dedup to unique pairs.
    rng = random.Random(args.seed)
    for m in MODELS:
        rng.shuffle(per_model[m])
    chosen: list = []
    seen = set()
    idx = {m: 0 for m in MODELS}
    while len(chosen) < args.n and any(idx[m] < len(per_model[m]) for m in MODELS):
        for m in MODELS:
            if len(chosen) >= args.n:
                break
            while idx[m] < len(per_model[m]):
                k = per_model[m][idx[m]]; idx[m] += 1
                if k not in seen:
                    seen.add(k); chosen.append(k); break

    rng.shuffle(chosen)
    blind_rows, key = [], {}
    for k in chosen:
        ds, s, t = k
        rid = hashlib.sha1(("|".join(k)).encode()).hexdigest()[:8]
        i = info[k]
        blind_rows.append((rid, ds, i["source_label"], i["target_label"]))
        key[rid] = {"dataset": ds, "source_uri": s, "target_uri": t,
                    "source_label": i["source_label"], "target_label": i["target_label"],
                    "flagging_models": sorted(i["models"]), "gold": "none"}

    blind = RESULTS / "ltfp_audit_blind.tsv"
    with blind.open("w", encoding="utf-8") as f:
        f.write("# Blind '<'-FP gold-gap audit. Each row: a pair some model labelled "
                "'<' but the gold has NO relation. Judge whether SOURCE is really a "
                "subclass of TARGET.\n")
        f.write("# judgment: gold_gap (real subclass, just unlabelled) | real_error "
                "(source is NOT a subclass of target) | unsure\n")
        f.write("row_id\tdataset\tsource_concept\ttarget_concept\tjudgment\n")
        for rid, ds, sl, tl in blind_rows:
            f.write(f"{rid}\t{ds}\t{sl}\t{tl}\t\n")
    (RESULTS / "ltfp_audit_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")

    # provenance / coverage summary
    from collections import Counter
    by_ds = Counter(k[0] for k in chosen)
    by_model = Counter(m for k in chosen for m in info[k]["models"])
    print(f"sampled {len(chosen)} unique '<'-FP gold-gap pairs (target {args.n})")
    print("  per dataset:", dict(by_ds))
    print("  flagging-model coverage (pair counted per flagging model):", dict(by_model))
    print(f"  blind: {blind}  (NO model column)  | key: {RESULTS/'ltfp_audit_key.json'}")
    # blindness self-check
    txt = blind.read_text()
    leak = [m for m in MODELS if m in txt]
    print("  blindness check — model names in blind file:", leak or "none ✓")


if __name__ == "__main__":
    main()
