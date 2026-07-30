"""
csv_to_owl.py — regenerate the eBay Kfz target ontology from the control CSV.

Same OWL as build_ebay_target.py, but sourced from ebay_kfz_tree.csv
(columns: id,label,parent_id,depth,path). Future edits to the tree touch ONLY
the CSV; re-run this to rebuild the OWL.

Model (pipeline-conformant, verified against RDFGraphWrapper):
  - one owl:Class per id, URI = {BASE_URI}{id}
  - rdfs:label = label, language-tagged @de
  - rdfs:subClassOf = parent id's URI; empty parent_id -> owl:Thing (root)
  - a repeated id with different parent_id (eBay artefact, e.g. 263026) yields
    multiple subClassOf edges on ONE class (rdflib dedups the class/label triples)

URI scheme: change BASE_URI in ONE place. The gold alignment file MUST use the
identical URIs.

Run: python csv_to_owl.py ebay_kfz_tree.csv ebay_kfz_target.owl
"""

from __future__ import annotations

import csv
import sys

from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL

BASE_URI = "http://ebay.de/kfz#e"      # <- single place to change the URI scheme
ONTOLOGY_URI = "http://ebay.de/kfz"
LANG = "de"


def _uri(cid: str) -> URIRef:
    return URIRef(f"{BASE_URI}{cid}")


def build_graph(csv_path: str) -> Graph:
    rows = []
    ids = set()
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            ids.add(r["id"].strip())

    g = Graph()
    g.add((URIRef(ONTOLOGY_URI), RDF.type, OWL.Ontology))
    g.add((URIRef(ONTOLOGY_URI), RDFS.comment, Literal(
        "eBay.de Kfz-Kategoriebaum (kuratierter Subtree, Root 131090) als "
        "Target-Ontologie. Generiert aus ebay_kfz_tree.csv via csv_to_owl.py.",
        lang=LANG)))

    roots = 0
    parents_per_id: dict[str, int] = {}
    for r in rows:
        cid = r["id"].strip()
        label = r["label"]
        pid = (r.get("parent_id") or "").strip()
        c = _uri(cid)
        g.add((c, RDF.type, OWL.Class))
        g.add((c, RDFS.label, Literal(label, lang=LANG)))
        if pid:
            if pid not in ids:
                print(f"WARN: parent_id {pid} of {cid} is not a class in the CSV", file=sys.stderr)
            g.add((c, RDFS.subClassOf, _uri(pid)))
            parents_per_id[cid] = parents_per_id.get(cid, 0) + 1
        else:
            g.add((c, RDFS.subClassOf, OWL.Thing))
            roots += 1

    # integrity report
    n_class = len(set(g.subjects(RDF.type, OWL.Class)))
    n_sub = sum(1 for _ in g.triples((None, RDFS.subClassOf, None)))
    multi = {c: n for c, n in parents_per_id.items() if n > 1}
    print(f"[csv->owl] classes={n_class} subClassOf={n_sub} roots={roots} "
          f"multi_parent={multi or 'none'}", file=sys.stderr)
    if roots != 1:
        print(f"WARN: expected exactly 1 root, got {roots}", file=sys.stderr)
    return g


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "ebay_kfz_tree.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "ebay_kfz_target.owl"
    g = build_graph(csv_path)
    g.serialize(destination=out_path, format="pretty-xml")
    print(f"wrote {out_path}  ({len(g)} triples)")
