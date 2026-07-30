# VDI → eBay Subsumption Gold Standard

Industrial cross-vocabulary subsumption benchmark built for this thesis: it maps
concepts of the **VDI 4081** vehicle-body classification (source) to the **eBay**
motor-vehicle (KFZ) category taxonomy (target), with directed relations
`<` (subclass), `>` (superclass), `=` (equivalent).

## Files
| file | what |
|---|---|
| `reference_full.rdf` | OAEI-alignment format, **2048** correspondences = seeds + deductive (transitive) closure. Use for **precision / hierarchy-credit** analysis ("every pair here is correct"). |
| `reference_seed.rdf` | **451** direct (tier=seed) correspondences. Use for **Recall@K / MRR** (the closure would flood recall with trivial distant pairs). |
| `subsumption_gold/gold_relations_karosserie.tsv` | source TSV from the annotation process (source, source_label, target, target_label, relation, tier, hops, rules). |
| `subsumption_gold/derive_subsumption_gold.py` | derivation of the closure/tiers from the adjudicated annotations. |
| `subsumption_gold/gold_subsumption_findings.md` | annotation findings / rule notes. |
| `ebay_kfz_target.owl` | eBay KFZ taxonomy as OWL (target). **Third-party — see `../../THIRD_PARTY_NOTICES.md` §2.** |
| `vdi_karosserie_source_pos.owl` | VDI 4081 body classification as OWL (source, positive slice). **Third-party — §3.** |

Build inputs (scripts) are tracked at repo root under `goldstandard_ebay/`
(`build_ebay_target.py`, `csv_to_owl.py`, `vdi/vdi2owl.py`); the large raw HTML/xlsx
sources stay out of git.

## Annotation protocol (summary)
Two independent annotators (**A**, **B**) plus an adjudicator (**C**) produced the
directed relations; annotation rules **K1–K5** (relation criteria) and **R1/R2**
(reconciliation) governed the process; findings **H1–H7** are documented in the
thesis. The three annotator workbooks (A / B / adjudication C), **anonymized** and
CSV-exported, are under [`annotation/`](annotation/) — see its README for the
protocol and the privacy note.

## Licensing
Own annotations (the mapping itself) → **CC BY 4.0**. Third-party taxonomy content
(eBay, VDI 4081) → owners' terms. See `../../THIRD_PARTY_NOTICES.md`.
