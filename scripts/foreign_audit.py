"""
foreign_audit.py — list ALL 'foreign' top-5 occupants from the superclass-miss
analysis, as a blind, manually-classifiable TSV.

Scope = same as analyze_superclass_misses.py H3: focus config
(Qwen3/no-LoRA/path_context/T2), datasets g3-text + g5-groceries (the two
fan-out datasets; g2-diseases has too few misses). A 'foreign' = a target in
S's '>' top-5 that has NO gold relation (<, >, =) to S in the reference.

One TSV row per foreign hit, with an empty 'judgment' column for manual filling
against a FIXED three-class criterion (defined in the TSV comment header). Rows
are randomly shuffled (seed 42) so judgment stays blind — not grouped by dataset
or source. A stable row_id (hash of dataset|source|foreign) makes each verdict
reproducibly referenceable regardless of shuffle order.

Torch-free, no new model run. Reads the persisted predictions.tsv.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Alignment import Alignment
from tracks.zenodo_loader import load_subdataset
from analyze_superclass_misses import _ranked_by_source, _gold_by_source, _load_labels, _lab, _run_dir

CRITERION_HEADER = [
    "# foreign_audit — manual classification of 'foreign' top-5 occupants",
    "# focus: Qwen3 / no-LoRA / path_context / T2 ; datasets: g3-text + g5-groceries",
    "# A 'foreign' = a target in S's '>' top-5 with NO gold relation (<,>,=) to S in the reference.",
    "# Fill 'judgment' with EXACTLY one of:",
    "#   gold_gap          — foreign is plausibly a CHILD of S (more specific, same domain), just not annotated.  e.g. S=Location, foreign=Region",
    "#   sibling_or_parent — semantically related but NOT a child: sibling, broader term, or spelling-variant of S itself.  e.g. S=Sauces, foreign=Sauces",
    "#   real_foreign      — semantically unrelated, genuine retrieval error.  e.g. S=Science, foreign=Football",
    "# Rows are randomly shuffled (seed 42) to keep judgment blind (not grouped by dataset/source).",
]


def main() -> None:
    p = argparse.ArgumentParser(description="List all 'foreign' top-5 occupants for manual audit.")
    p.add_argument("--sha", required=True)
    p.add_argument("--results-root", default="results")
    p.add_argument("--model", default="qwen3-embedding-8b")
    p.add_argument("--lora", default="lora-off")
    p.add_argument("--a", default="path_context")
    p.add_argument("--b", default="sub_b_pin")
    p.add_argument("--datasets", nargs="+", default=["g3-text", "g5-groceries"])
    p.add_argument("--out-tsv", default=None)
    p.add_argument("--sample", type=int, default=0, help="If >0, random subsample to this many rows (seed 42).")
    args = p.parse_args()

    root = Path(args.results_root)
    rows: list[dict] = []
    per_dataset: dict[str, int] = {}

    for dataset in args.datasets:
        run_dir = _run_dir(root, args.model, args.lora, args.a, args.b, dataset, args.sha)
        src_path, tgt_path, ref_path = load_subdataset(dataset)
        reference = Alignment(str(ref_path))
        gold = _gold_by_source(reference)
        sup_ranked = _ranked_by_source(run_dir, ">")
        src_labels = _load_labels(str(src_path))
        tgt_labels = _load_labels(str(tgt_path))

        count = 0
        for s in gold:
            if not gold[s][">"]:
                continue
            order = sup_ranked.get(s, [])
            pos = {t: i + 1 for i, t in enumerate(order)}
            missed = gold[s][">"] - set(order[:20])
            if not missed:
                continue  # only missed sources, matching H3 scope
            missed_lbls = sorted(_lab(t, tgt_labels) for t in missed)
            missed_str = "; ".join(missed_lbls[:5]) + (f" (+{len(missed_lbls)-5} more)" if len(missed_lbls) > 5 else "")
            for t in order[:5]:
                if t in gold[s][">"] or t in gold[s]["<"] or t in gold[s]["="]:
                    continue  # not foreign
                rid = hashlib.md5(f"{dataset}|{s}|{t}".encode()).hexdigest()[:8]
                rows.append({
                    "row_id": rid,
                    "source_label": _lab(s, src_labels),
                    "foreign_label": _lab(t, tgt_labels),
                    "foreign_rank": pos.get(t, ""),
                    "missed_children": missed_str,
                    "source_uri": s,
                    "foreign_uri": t,
                })
                count += 1
        per_dataset[dataset] = count

    total = len(rows)
    random.seed(42)
    random.shuffle(rows)

    sampled = rows
    if args.sample and total > args.sample:
        sampled = rows[:args.sample]

    out = Path(args.out_tsv) if args.out_tsv else root / f"foreign_audit_{args.sha}.tsv"
    cols = ["row_id", "source_label", "foreign_label", "foreign_rank",
            "missed_children", "source_uri", "foreign_uri", "judgment"]
    with out.open("w", encoding="utf-8") as f:
        for line in CRITERION_HEADER:
            f.write(line + "\n")
        f.write("\t".join(cols) + "\n")
        for r in sampled:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    print("=" * 72)
    print("FOREIGN AUDIT")
    print(f"  total foreign hits (g3-text + g5-groceries): {total}")
    for d, c in per_dataset.items():
        print(f"    {d}: {c}")
    if total > 150:
        print(f"  >150 — consider a manageable random subsample, e.g.:")
        print(f"    python scripts/foreign_audit.py --sha {args.sha} --sample 80")
        print(f"  (current file has {'all ' + str(total) if not args.sample else str(len(sampled))} rows)")
    else:
        print(f"  <=150 — all {total} listed (no sampling needed).")
    print(f"  written: {out}  ({len(sampled)} rows + empty 'judgment' column)")
    print("=" * 72)


if __name__ == "__main__":
    main()
