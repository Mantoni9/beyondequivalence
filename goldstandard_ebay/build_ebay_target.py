#!/usr/bin/env python3
"""
build_ebay_target.py — eBay-Kfz-Kategoriebaum (HTML) -> Target-Ontologie
========================================================================
Input : ebay_kfz.html  (verschachtelte <li><p>Label | ID: N</p><ul>...-Struktur,
        kuratierter Kfz-Subtree des eBay.de-Kategoriebaums, Root 131090)
Output: ebay_kfz_tree.csv   (id,label,parent_id,depth,path)  — Kontrolle/Analyse
        ebay_kfz_target.owl (RDF/XML: owl:Class, rdfs:label@de, rdfs:subClassOf)

Design-Entscheidungen:
- URI-Schema als Konstante BASE_URI + "e{ID}" (NCName-sicher, da nicht zifferbeginnend).
  Bei Bedarf an das Schema der Pipeline/der OAEI-Datensätze anpassen und neu ausführen.
- Root (131090) -> rdfs:subClassOf owl:Thing.
- Labels verbatim aus eBay (de), keine Normalisierung; Pfad nur in der CSV.
- 'Sonstige'-Knoten bleiben erhalten (echte eBay-Kategorien mit eigener ID).

Usage: python3 build_ebay_target.py [html_in] [csv_out] [owl_out]
"""
import csv
import re
import sys
from bs4 import BeautifulSoup

BASE_URI = "http://ebay.de/kfz#"          # <- bei Bedarf anpassen
ONTOLOGY_URI = "http://ebay.de/kfz"
LANG = "de"

HTML_IN = sys.argv[1] if len(sys.argv) > 1 else "ebay_kfz.html"
CSV_OUT = sys.argv[2] if len(sys.argv) > 2 else "ebay_kfz_tree.csv"
OWL_OUT = sys.argv[3] if len(sys.argv) > 3 else "ebay_kfz_target.owl"

raw = open(HTML_IN, encoding="utf-8").read()

# ---------- Integritätschecks auf dem Rohtext (VOR dem reparierenden Parser) ----------
n_li_open, n_li_close = len(re.findall(r"<li\b", raw)), raw.count("</li>")
n_ul_open, n_ul_close = len(re.findall(r"<ul\b", raw)), raw.count("</ul>")
n_idmark = raw.count("| ID:")
assert n_li_open == n_li_close, f"li-Tags unbalanciert: {n_li_open} vs {n_li_close}"
assert n_ul_open == n_ul_close, f"ul-Tags unbalanciert: {n_ul_open} vs {n_ul_close}"
assert n_li_open == n_idmark, f"li-Anzahl {n_li_open} != ID-Marker {n_idmark}"

soup = BeautifulSoup(f"<ul>{raw}</ul>", "lxml")
root_ul = soup.find("ul")

edges = []          # (id, label, parent_id, depth, path) — eine Zeile je Vorkommen
seen_ids = {}       # id -> label (Erstvorkommen)
multi_parent = {}   # id -> [parent_ids] bei Mehrfachvorkommen (Quell-Artefakt)

def walk(ul, parent_id, parent_path, depth):
    for li in ul.find_all("li", recursive=False):
        p = li.find("p", recursive=False)
        assert p is not None, f"li ohne <p> unter parent {parent_id}"
        text = p.get_text()
        label, sep, id_part = text.rpartition(" | ID: ")
        assert sep, f"Kein ID-Marker in: {text!r}"
        label = label.strip()
        cid = int(id_part.strip())
        assert label and "ID:" not in label, f"Labelproblem: {text!r}"
        if cid in seen_ids:
            # Duplikat nur tolerieren, wenn das Label exakt identisch ist
            # (sonst deutet es auf einen Transkriptions-/Quellfehler hin).
            assert seen_ids[cid] == label, (
                f"ID {cid} mit abweichendem Label: {label!r} vs {seen_ids[cid]!r}")
            multi_parent.setdefault(cid, []).append(parent_id)
        else:
            seen_ids[cid] = label
        path = f"{parent_path} > {label}" if parent_path else label
        edges.append((cid, label, parent_id, depth, path))
        sub = li.find("ul", recursive=False)
        if sub is not None:
            walk(sub, cid, path, depth + 1)

walk(root_ul, None, "", 0)
assert len(edges) == n_idmark, f"Parser verlor Knoten: {len(edges)} vs {n_idmark}"
roots = [e for e in edges if e[2] is None]
assert len(roots) == 1 and roots[0][0] == 131090, f"Root-Anomalie: {roots}"
nodes = edges  # CSV = Kantenliste; OWL dedupliziert Klassen unten

# ---------- CSV ----------
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "label", "parent_id", "depth", "path"])
    for cid, label, pid, depth, path in nodes:
        w.writerow([cid, label, pid if pid is not None else "", depth, path])

# ---------- OWL (RDF/XML) ----------
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL

EB = Namespace(BASE_URI)
g = Graph()
g.bind("owl", OWL)
g.bind("rdfs", RDFS)
g.bind("ebay", EB)
onto = URIRef(ONTOLOGY_URI)
g.add((onto, RDF.type, OWL.Ontology))
g.add((onto, RDFS.comment, Literal(
    "eBay.de Kfz-Kategoriebaum (kuratierter Subtree, Root 131090) als Target-Ontologie. "
    "Generiert aus ebay_kfz.html via build_ebay_target.py.", lang=LANG)))

def uri(cid: int) -> URIRef:
    return EB[f"e{cid}"]

for cid, label, pid, depth, path in nodes:
    c = uri(cid)
    g.add((c, RDF.type, OWL.Class))          # Graph dedupliziert identische Tripel
    g.add((c, RDFS.label, Literal(label, lang=LANG)))
    g.add((c, RDFS.subClassOf, uri(pid) if pid is not None else OWL.Thing))

g.serialize(destination=OWL_OUT, format="pretty-xml")

# ---------- Statistik ----------
from collections import Counter
depths = Counter(n[3] for n in nodes)
parents = {n[2] for n in nodes if n[2] is not None}
uniq = {n[0] for n in nodes}
leaves = [i for i in uniq if i not in parents]
lbl_counts = Counter(n[1] for n in nodes)
dups = {l: c for l, c in lbl_counts.items() if c > 1}
print(f"Kanten gesamt      : {len(edges)} | eindeutige Klassen: {len(seen_ids)}")
if multi_parent:
    print("QUELL-ARTEFAKT — IDs mit mehreren Eltern (Label identisch, als Mehrfach-subClassOf modelliert):")
    for cid, extra in multi_parent.items():
        allp = [e[2] for e in edges if e[0] == cid]
        print(f"   {cid} {seen_ids[cid]!r}: Eltern {allp}")
print(f"Max. Tiefe         : {max(depths)} | Verteilung: {dict(sorted(depths.items()))}")
print(f"Blätter / Eltern   : {len(leaves)} / {len(parents)}")
print(f"'Sonstige'-Knoten  : {lbl_counts.get('Sonstige', 0)}")
print(f"Mehrfach-Labels    : {len(dups)} verschiedene (IDs bleiben eindeutig)")
print(f"Direkte Root-Kinder ({sum(1 for n in nodes if n[2] == 131090)}):")
for cid, label, pid, depth, path in nodes:
    if pid == 131090:
        print(f"   {cid:6d}  {label}")
