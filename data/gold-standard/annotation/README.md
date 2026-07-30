# Annotation workbooks — VDI→eBay gold standard

Anonymized source workbooks behind the industrial gold standard. **All annotator
identities are removed** (document metadata set to `anonymized`; one company
mention in an instruction cell replaced with `Partnerbetrieb`). Two independent
annotators (**A**, **B**) plus an adjudicator (**C**) produced the directed
relations. Person names are referred to only by role (Expert A / B / C).

## Files
| file | role | sheets |
|---|---|---|
| `annotator_A.xlsx` (+ `annotator_A__*.csv`) | Expert A — independent tree | Anleitung, Hauptgruppen_Ref, Baugruppen (2180 rows), NeueZwischenkategorien |
| `annotator_B.xlsx` (+ `annotator_B__*.csv`) | Expert B — independent tree | Anleitung, Hauptgruppen_Ref, Baugruppen (2180 rows), NeueZwischenkategorien |
| `adjudication_C.xlsx` (+ `adjudication_C__*.csv`) | Expert C — adjudication of disagreements | Anleitung_C, Muster, Entscheidungen (454), Statistik, Befunde_Datei_A |

Each sheet is also exported to a plain-UTF-8 CSV (`<workbook>__<sheet>.csv`) for
tool-independent access. No hidden sheets, no cell comments.

## Protocol (as given to the annotators; full text in the `Anleitung` sheets)
- **Task:** for each of the 2,180 VDI Baugruppen, assign the superordinate category
  via **`subClassOf`** ("ist eine Art von" — taxonomic is-a) and/or **`partOf`**
  ("ist Teil von" — meronymy). A valid parent is a Hauptgruppe id (1–5) or another
  Baugruppe id; annotators may introduce new intermediate categories.
- **Independence:** A and B work **separately, without prior coordination** — the
  point is to measure inter-rater agreement.
- **Mandatory scope:** all Baugruppen with focus = "Karosserie" (body parts) are
  required; the rest is optional.
- **Adjudication:** annotator **C** resolves every A/B disagreement (sheet
  `Entscheidungen`), yielding the reconciled gold used to build `reference_seed.rdf`
  / `reference_full.rdf`.

## Rule / finding labels
The formal annotation criteria **K1–K5**, the reconciliation rules **R1/R2**, and
the annotation findings **H1–H7** referenced in the thesis correspond to this
process; their authoritative definitions are in the thesis text (annotation
chapter) and are reflected in the `Anleitung*`/`Muster`/`Befunde_Datei_A` sheets.

## Provenance / privacy
Scrubbed 2026-07-30 from the original workbooks (kept out of the repo). Metadata
`creator`/`lastModifiedBy` were `anonymized`; the single company reference
(instruction cell) became `Partnerbetrieb`. The full original→anonymized mapping
was handed to the thesis author for review and is intentionally **not** committed.
Own annotations are CC BY 4.0; see `../../../THIRD_PARTY_NOTICES.md`.
