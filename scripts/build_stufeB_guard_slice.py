"""
build_stufeB_guard_slice.py — pin the read-only '<'-heavy control slice for
Stufe B (Antonio-approved 2026-06-13). One-shot, deterministic; the output
TSV is committed and treated as READ-ONLY thereafter.

Rationale: dev {g7, g5} is '>'-heavy, so a residual '>'-lean would look fine
on dev Macro-F1 but harm '<'-heavy data. This fixed slice of mouse-human '<'
gold pairs is used ONLY to verify the winning Stufe-B arm does not degrade
'<' — its numbers NEVER enter arm selection or tuning (mouse-human is a test
set; this is a disclosed guard sample, not a tuning target).

Selection (deterministic): mouse-human gold pairs with normalized relation
'<' that are reranker-CONDITIONAL (present in the frozen d11c97e candidate
set after the top-20-per-(source,direction) cut the Stage-2 loader applies),
sorted by (source_uri, target_uri), then a fixed-seed (42) sample of N=45.

Run once:  conda run -n melt-olala python scripts/build_stufeB_guard_slice.py
Output:    docs/stufeB_guard_slice_mousehuman.tsv  (committed, read-only)
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Alignment import Alignment
from evaluation_recall import _normalize_relation
from tracks.zenodo_loader import load_subdataset

DATASET = "mouse-human"
SHA = "d11c97e"
TOP_K = 20
SEED = 42
N_SLICE = 45
OUT = Path(__file__).resolve().parent.parent / "docs" / "stufeB_guard_slice_mousehuman.tsv"
CAND_TSV = (Path(__file__).resolve().parent.parent / "results" /
            f"ablbi_qwen3-embedding-8b_lora-off_A-path_context_B-sub_b_pin_{DATASET}_{SHA}"
            / "predictions.tsv")


def _conditional_pairs(tsv: Path, top_k: int) -> set[tuple[str, str]]:
    """Replicate the Stage-2 loader cut: top-k per (source, normalized rel),
    union of (source, target) over both directions."""
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = {}
    with tsv.open(encoding="utf-8") as f:
        f.readline()
        for line in f:
            s, t, rel, sc = line.rstrip("\n").split("\t")
            norm = _normalize_relation(rel)
            if norm is None:
                continue
            grouped.setdefault((s, norm), []).append((t, float(sc)))
    pairs: set[tuple[str, str]] = set()
    for (s, _norm), entries in grouped.items():
        entries.sort(key=lambda e: (-e[1], e[0]))
        for t, _sc in entries[:top_k]:
            pairs.add((s, t))
    return pairs


def main() -> None:
    if OUT.exists():
        sys.exit(f"REFUSING to overwrite the pinned read-only guard slice: {OUT}\n"
                 "Delete it manually only if you intend to re-pin (and document why).")
    if not CAND_TSV.is_file():
        sys.exit(f"mouse-human candidate TSV not found: {CAND_TSV}")

    _s, _t, ref_path = load_subdataset(DATASET)
    reference = Alignment(str(ref_path))
    gold_lt = sorted({(c.source, c.target) for c in reference
                      if _normalize_relation(c.relation) == "<"})
    conditional = _conditional_pairs(CAND_TSV, TOP_K)
    eligible = sorted(p for p in gold_lt if p in conditional)

    rng = random.Random(SEED)
    n = min(N_SLICE, len(eligible))
    chosen = sorted(rng.sample(eligible, n))

    lines = [
        "# Stufe-B '<'-heavy control slice (Antonio-approved 2026-06-13). READ-ONLY.",
        f"# dataset={DATASET} relation=< conditional_on=frozen d11c97e top-{TOP_K} "
        f"seed={SEED} N={n}",
        f"# pool: gold '<' = {len(gold_lt)}, conditional '<' = {len(eligible)}; "
        f"sampled {n}.",
        "# Used ONLY to check the winning arm does not degrade '<' — NEVER enters "
        "arm selection/tuning.",
        "source_uri\ttarget_uri\tgold_relation",
    ]
    for s, t in chosen:
        lines.append(f"{s}\t{t}\t<")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT.chmod(0o444)  # read-only on disk
    print(f"pinned {n} pairs -> {OUT}")
    print(f"pool: gold '<' = {len(gold_lt)}, conditional '<' = {len(eligible)}")
    print("first 3:", chosen[:3])


if __name__ == "__main__":
    main()
