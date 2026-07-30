#!/usr/bin/env python3
"""shard_stage1_tsv.py — split a frozen Stage-1 predictions TSV into N shards.

Walltime armour for expensive Stage-2 cells (gpt-oss × mouse-human): each shard
runs as its own SLURM job (well under the 120h QOS cap), shards can run in
parallel under the 4-GPU quota, and a wall-kill costs one shard instead of the
whole cell. Scientifically a no-op: rows are grouped **source-wise**, so the
Stage-2 loader's top-20-per-(source, direction) cut sees exactly the same rows
per source as in the unsharded file.

Deterministic: sources keep first-appearance order; greedy least-loaded
assignment by row count. Same input + same N -> byte-identical shards.

Usage:
    python scripts/shard_stage1_tsv.py results/stage1_frozen/mouse-human_....tsv \
        --num-shards 6 [--out-dir results/stage1_shards]

Merge counterpart after the Stage-2 shard runs: scripts/merge_stage2_shards.py.
"""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("tsv", help="Stage-1 predictions TSV (source_uri/target_uri/relation/score)")
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: directory of the input TSV)")
    args = p.parse_args()

    src = Path(args.tsv)
    n = args.num_shards
    assert n >= 2, "--num-shards must be >= 2"
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with src.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = list(reader)
    src_col = header.index("source_uri")

    groups: "OrderedDict[str, list[list[str]]]" = OrderedDict()
    for r in rows:
        groups.setdefault(r[src_col], []).append(r)

    # Greedy least-loaded (stable: ties broken by shard index).
    loads = [0] * n
    assign: list[list[str]] = [[] for _ in range(n)]  # shard -> source keys
    for key, g in groups.items():
        i = min(range(n), key=lambda k: (loads[k], k))
        assign[i].append(key)
        loads[i] += len(g)

    stem = src.stem
    print(f"input: {src}  ({len(rows)} rows, {len(groups)} sources)")
    for i in range(n):
        out = out_dir / f"{stem}_shard{i + 1}of{n}.tsv"
        n_rows = 0
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(header)
            for key in assign[i]:
                for r in groups[key]:
                    w.writerow(r)
                    n_rows += 1
        print(f"  shard {i + 1}/{n}: {len(assign[i]):5d} sources  {n_rows:7d} rows  -> {out}")

    total = sum(loads)
    assert total == len(rows), f"row loss: {total} != {len(rows)}"
    print("OK — no row loss, sources are shard-disjoint.")


if __name__ == "__main__":
    main()
