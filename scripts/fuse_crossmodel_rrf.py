"""
fuse_crossmodel_rrf.py — post-hoc cross-model Reciprocal Rank Fusion for the
bidirectional ablation. NO new model runs: reads the deep top-50 ranked lists
persisted by ablation_bidirectional.py and fuses them.

Fusion is STRICTLY within the same direction (< with <, > with >, never across)
and across the two models (Qwen3 (+) Nemo), per (lever-perm x direction x
dataset). Fused lists are cut to k=10 / k=20 and evaluated with the same
per_relation_strict metric as the single-model runs.

Pairings (Qwen3 lora-state (+) Nemo lora-state):
  1. qwen3 lora-off (+) nemo lora-on    (current leader)
  2. qwen3 lora-off (+) nemo lora-off
  3. qwen3 lora-on  (+) nemo lora-on
  4. qwen3 lora-on  (+) nemo lora-off

Reports fused R@k vs. best-single R@k (the better of the two models for that
direction/dataset/perm — NOT their average), per dataset + aggregate, plus the
headline check fused R@10 vs. best-single R@20 (>= means Stage-2 candidate count
halved at equal recall).

Torch-free: only Alignment / evaluation_recall / zenodo_loader. Runs on a login
node. Additive — touches no Stage-1 file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Alignment import Alignment
from Correspondence import Correspondence
from evaluation_recall import compute_recall_at_k
from tracks.zenodo_loader import load_subdataset

QWEN = "qwen3-embedding-8b"
NEMO = "llama-embed-nemotron-8b"

# (label, qwen3 lora_tag, nemo lora_tag)
PAIRINGS = (
    ("q-noLoRA (+) n-LoRA",   "lora-off", "lora-on"),
    ("q-noLoRA (+) n-noLoRA", "lora-off", "lora-off"),
    ("q-LoRA (+) n-LoRA",     "lora-on",  "lora-on"),
    ("q-LoRA (+) n-noLoRA",   "lora-on",  "lora-off"),
)
PERMS = (("turtle", "default"), ("path_context", "default"),
         ("turtle", "sub_b_pin"), ("path_context", "sub_b_pin"))
DIRECTIONS = (("<", "subclass"), (">", "superclass"))


def _rrf(list_a: list[str], list_b: list[str], k: int) -> list[str]:
    sc: dict[str, float] = defaultdict(float)
    for rank, t in enumerate(list_a, 1):
        sc[t] += 1.0 / (k + rank)
    for rank, t in enumerate(list_b, 1):
        sc[t] += 1.0 / (k + rank)
    return sorted(sc, key=lambda t: (-sc[t], t))


def _load_ranked(run_dir: Path, direction: str) -> dict[str, list[str]]:
    """Per-source ranked target list for one direction, from predictions.tsv."""
    per_src: dict[str, list[tuple[float, str]]] = defaultdict(list)
    with (run_dir / "predictions.tsv").open(encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            s, t, rel, score = line.rstrip("\n").split("\t")
            if rel != direction:
                continue
            per_src[s].append((float(score), t))
    return {s: [t for _, t in sorted(lst, key=lambda x: (-x[0], x[1]))]
            for s, lst in per_src.items()}


def _build_index(results_root: Path, sha: str) -> dict[tuple, Path]:
    """Index run dirs by (alias, lora_tag, A, B, dataset) via their config.json."""
    index: dict[tuple, Path] = {}
    for cfg_p in sorted(results_root.glob(f"ablbi_*_{sha}/config.json")):
        cfg = json.loads(cfg_p.read_text())
        key = (cfg["model_alias"], cfg["lora"], cfg["A"], cfg["B"], cfg["dataset"])
        index[key] = cfg_p.parent
    return index


def _single_r(run_dir: Path, side: str) -> dict[int, float]:
    m = json.loads((run_dir / "metrics.json").read_text())["r10_r20"][side]
    return {10: m["10"], 20: m["20"]}


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-model RRF fusion (post-hoc).")
    p.add_argument("--sha", required=True, help="git short SHA stamped on the ablation run dirs.")
    p.add_argument("--results-root", default="results")
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--datasets", nargs="+",
                   default=["mouse-human", "g1-web", "g2-diseases", "g3-text", "g5-groceries", "g7-literature"])
    p.add_argument("--out-tsv", default=None, help="Full per-(pairing,perm,dataset,direction) TSV.")
    args = p.parse_args()

    root = Path(args.results_root)
    index = _build_index(root, args.sha)
    if not index:
        sys.exit(f"ERROR: no ablbi_*_{args.sha} run dirs under {root}. Run the ablation first.")

    # Cache references + gold counts per dataset.
    ref_cache: dict[str, Alignment] = {}
    for d in args.datasets:
        _, _, ref_path = load_subdataset(d)
        ref_cache[d] = Alignment(str(ref_path))

    out_tsv = Path(args.out_tsv) if args.out_tsv \
        else root / f"fusion_crossmodel_{args.sha}.tsv"
    tsv_rows: list[str] = ["pairing\tA\tB\tdataset\tdirection\tn_gold\t"
                           "fused_R@10\tfused_R@20\tbest_single_R@10\tbest_single_R@20\t"
                           "fused_hits@10\tfused_hits@20\tbest_single_model@20"]

    print("\n" + "=" * 100)
    print(f"CROSS-MODEL RRF FUSION  (k={args.rrf_k})  —  fused vs best-single per_relation_strict")
    print("=" * 100)

    for pair_label, q_lora, n_lora in PAIRINGS:
        # Aggregate accumulators keyed by (A,B,direction).
        agg: dict[tuple, dict] = defaultdict(lambda: {
            "n": 0, "fused_hits": {10: 0, 20: 0}, "bestsingle_hits": {10: 0, 20: 0}})

        for a_label, b_label in PERMS:
            for direction, label in DIRECTIONS:
                for dataset in args.datasets:
                    qk = (QWEN, q_lora, a_label, b_label, dataset)
                    nk = (NEMO, n_lora, a_label, b_label, dataset)
                    if qk not in index or nk not in index:
                        sys.exit(f"ERROR: missing run dir for {qk if qk not in index else nk}")
                    q_dir, n_dir = index[qk], index[nk]

                    q_ranked = _load_ranked(q_dir, direction)
                    n_ranked = _load_ranked(n_dir, direction)
                    sources = set(q_ranked) | set(n_ranked)

                    fused = Alignment()
                    for s in sources:
                        order = _rrf(q_ranked.get(s, []), n_ranked.get(s, []), args.rrf_k)
                        for rank, t in enumerate(order[:20], 1):
                            fused.add(Correspondence(s, t, direction, 1.0 / rank))

                    rep = compute_recall_at_k(ref_cache[dataset], fused, k_values=(10, 20))
                    fr = rep.recall_at_k["per_relation_strict"][label]
                    # n for this direction = per_relation_strict denominator.
                    gold = json.loads((q_dir / "metrics.json").read_text())["gold"]
                    n_gold = gold[label]

                    q_single = _single_r(q_dir, "sub" if direction == "<" else "sup")
                    n_single = _single_r(n_dir, "sub" if direction == "<" else "sup")
                    best = {k: max(q_single[k], n_single[k]) for k in (10, 20)}
                    best_model = {k: (QWEN if q_single[k] >= n_single[k] else NEMO) for k in (10, 20)}

                    fhits = {k: round(fr.get(k, 0.0) * n_gold) for k in (10, 20)}
                    bhits = {k: round(best[k] * n_gold) for k in (10, 20)}

                    a = agg[(a_label, b_label, direction)]
                    a["n"] += n_gold
                    for k in (10, 20):
                        a["fused_hits"][k] += fhits[k]
                        a["bestsingle_hits"][k] += bhits[k]

                    tsv_rows.append(
                        f"{pair_label}\t{a_label}\t{b_label}\t{dataset}\t{direction}\t{n_gold}\t"
                        f"{fr.get(10,0.0):.4f}\t{fr.get(20,0.0):.4f}\t{best[10]:.4f}\t{best[20]:.4f}\t"
                        f"{fhits[10]}\t{fhits[20]}\t{best_model[20]}")

        # Print aggregate table for this pairing.
        print(f"\n### Pairing: {pair_label}   (aggregate, pooled hits/n across {len(args.datasets)} datasets)")
        hdr = (f"{'perm':<22} {'dir':<10} {'n':>5} | {'fusedR@10':>9} {'fusedR@20':>9} | "
               f"{'bestR@10':>8} {'bestR@20':>8} | headline(f@10>=best@20)")
        print(hdr); print("-" * len(hdr))
        for a_label, b_label in PERMS:
            for direction, label in DIRECTIONS:
                a = agg[(a_label, b_label, direction)]
                n = a["n"] or 1
                f10 = a["fused_hits"][10] / n
                f20 = a["fused_hits"][20] / n
                b10 = a["bestsingle_hits"][10] / n
                b20 = a["bestsingle_hits"][20] / n
                flag = "HALVED ✓" if f10 >= b20 else ""
                perm = f"{a_label}+{b_label}"
                print(f"{perm:<22} {label:<10} {a['n']:>5} | {f10:>9.4f} {f20:>9.4f} | "
                      f"{b10:>8.4f} {b20:>8.4f} | {flag}")

    out_tsv.write_text("\n".join(tsv_rows) + "\n", encoding="utf-8")
    print(f"\nFull per-dataset detail written to: {out_tsv}")
    print("=" * 100)


if __name__ == "__main__":
    main()
