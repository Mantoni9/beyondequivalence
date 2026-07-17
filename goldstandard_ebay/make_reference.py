#!/usr/bin/env python3
"""make_reference.py — VDI→eBay Subsumptions-Gold → OAEI-Alignment-Referenzen.

Konvertiert das von Experte C gelieferte `subsumption_gold/gold_relations_karosserie.tsv`
(Spalten: source, source_label, source_level, target, target_label, relation,
tier, hops_source, hops_target, rules) in das OAEI-Alignment-XML, das
`Alignment(path)` im Runner erwartet (gleiches Format wie die Zenodo-Referenzen).

Zwei Ausgaben, gemäß dem Eval-Protokoll der Lieferung:
  reference_seed.rdf — nur tier=seed (die 451 direkten Konzept-Paare).
                       Primärreferenz für Recall@K/MRR: die deduktive Hülle
                       würde den Recall mit trivialen Fernbeziehungen fluten.
  reference_full.rdf — alle Zeilen (2.055) inkl. transitiver Hülle. Für
                       Precision-artige Analysen / Hierarchie-Kredit: jede
                       Vorhersage, die hier steht, ist korrekt.

URIs: source → http://vdi.de/kfz#{id}, target → http://ebay.de/kfz#{id}
(identisch zu vdi_karosserie_source_pos.owl / ebay_kfz_target.owl).

Aufruf (aus goldstandard_ebay/):
    python make_reference.py
"""
from __future__ import annotations

import csv
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
GOLD_TSV = HERE / "subsumption_gold" / "gold_relations_karosserie.tsv"
VDI_BASE = "http://vdi.de/kfz#"
EBAY_BASE = "http://ebay.de/kfz#"

HEADER = """<?xml version="1.0" encoding="utf-8"?>
<rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns:xsd="http://www.w3.org/2001/XMLSchema#">
<Alignment>
  <xml>yes</xml>
  <level>0</level>
  <type>??</type>
"""
FOOTER = """</Alignment>
</rdf:RDF>
"""
CELL = """  <map>
    <Cell>
      <entity1 rdf:resource="{src}"/>
      <entity2 rdf:resource="{tgt}"/>
      <relation>{rel}</relation>
      <measure rdf:datatype="xsd:float">1.0</measure>
    </Cell>
  </map>
"""


def write_reference(rows: list[dict], out_path: Path) -> int:
    with out_path.open("w", encoding="utf-8") as f:
        f.write(HEADER)
        for r in rows:
            f.write(CELL.format(
                src=VDI_BASE + r["source"],
                tgt=EBAY_BASE + r["target"],
                rel=escape(r["relation"]),
            ))
        f.write(FOOTER)
    return len(rows)


def main() -> None:
    with GOLD_TSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows, f"leeres Gold: {GOLD_TSV}"
    bad = [r for r in rows if r["relation"] not in ("<", ">", "=")]
    assert not bad, f"unerwartete Relation(en): {[r['relation'] for r in bad[:5]]}"

    seed = [r for r in rows if r["tier"] == "seed"]
    n_full = write_reference(rows, HERE / "reference_full.rdf")
    n_seed = write_reference(seed, HERE / "reference_seed.rdf")
    print(f"reference_seed.rdf : {n_seed} Cells (tier=seed)")
    print(f"reference_full.rdf : {n_full} Cells (Seeds + Hülle)")


if __name__ == "__main__":
    main()
