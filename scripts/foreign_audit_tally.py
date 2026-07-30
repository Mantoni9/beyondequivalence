"""
foreign_audit_tally.py — tally the manually-filled judgment column of
foreign_audit_<sha>.tsv into reproducible counts.

Reads the TSV (skipping the '#' criterion header), counts the three judgment
classes overall and per dataset (dataset re-derived from the source_uri
namespace, since the audit TSV is intentionally dataset-blind), and reports the
gold_gap share — the quantity that tells us how much the measured '>' recall is
depressed by reference incompleteness vs genuine error.

Torch-free. Flags any blank / invalid judgment so partial fills are visible.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

VALID = {"gold_gap", "sibling_or_parent", "real_foreign"}


def _dataset_of(source_uri: str) -> str:
    if "OpenCalais.com" in source_uri or "AlchemyAPI.com" in source_uri:
        return "g3-text"
    if "Groceries.com" in source_uri or "Groceries" in source_uri:
        return "g5-groceries"
    return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description="Tally the filled foreign_audit judgment column.")
    p.add_argument("--tsv", required=True, help="Path to the filled foreign_audit_<sha>.tsv")
    args = p.parse_args()

    path = Path(args.tsv)
    header: list[str] = []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if not header:
            header = parts
            continue
        rows.append(dict(zip(header, parts)))

    overall = Counter()
    per_ds = defaultdict(Counter)
    blank = invalid = 0
    for r in rows:
        j = (r.get("judgment") or "").strip()
        ds = _dataset_of(r.get("source_uri", ""))
        if not j:
            blank += 1
            continue
        if j not in VALID:
            invalid += 1
            print(f"  ! invalid judgment '{j}' in row {r.get('row_id')}")
            continue
        overall[j] += 1
        per_ds[ds][j] += 1

    judged = sum(overall.values())
    print("=" * 64)
    print(f"FOREIGN AUDIT TALLY  ({path.name})")
    print(f"  rows={len(rows)}  judged={judged}  blank={blank}  invalid={invalid}")
    print("-" * 64)

    def _block(title: str, c: Counter) -> None:
        tot = sum(c.values()) or 1
        print(f"  {title}  (n={sum(c.values())})")
        for k in ("gold_gap", "sibling_or_parent", "real_foreign"):
            print(f"    {k:<18} {c.get(k,0):>4}  ({c.get(k,0)/tot*100:5.1f}%)")

    _block("OVERALL", overall)
    for ds in sorted(per_ds):
        _block(ds, per_ds[ds])
    print("-" * 64)
    if judged:
        gg = overall.get("gold_gap", 0)
        sp = overall.get("sibling_or_parent", 0)
        rf = overall.get("real_foreign", 0)
        print(f"  gold_gap share of judged foreign: {gg/judged*100:.1f}%")
        print(f"  not-an-error (gold_gap + sibling_or_parent): {(gg+sp)/judged*100:.1f}%")
        print(f"  genuine real_foreign: {rf/judged*100:.1f}%")
    if blank:
        print(f"  NOTE: {blank} rows still unjudged — fill them for a complete count.")
    print("=" * 64)


if __name__ == "__main__":
    main()
