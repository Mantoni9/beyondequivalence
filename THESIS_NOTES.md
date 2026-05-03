# Thesis Methods Notes

## Versuchsaufbau (Stand 2026-05-03)

Drei Hebel für Stage-1-Retrieval:
- **A** — Text-Verbalisierung (BERTSubs Path Context)
- **B** — Description-Tuning (one_gen × T4 für asym, basic × S1 für sym)
- **C** — Bidirektionale Subsumption-Konsolidierung via RRF

Hebel **D** (Sym ∪ Asym Union) wurde verworfen — siehe Diskussion 2026-05-03.

## Vertagte Punkte

- **Konsistenz-Check via Bidirektional-Symmetrie als post-hoc Filter:**
  Wenn ein Source-Target-Paar in beiden broader-Richtungen (Source-zu-Target UND Target-zu-Source) hoch rankt, ist die Subsumption-Richtung ambivalent. Könnte als Precision-Filter im Stage-2-Reranker eingesetzt werden. Aktuell nicht implementiert weil es Recall (Hauptproblem) nicht löst, sondern Precision.

- **description_three_gen rauswerfen aus Hauptsweep:**
  Sub-B hat gezeigt: bitidentisch zu description_two_gen (Ontology-Tiefe reicht nicht für depth=3). Im Hauptablations-Sweep nicht mit testen.

- **Hebel C — bewusste Reduktion auf Subsumption-Richtung `<` (Source ⊂ Target):**
  C testet ausschließlich die `<`-Richtung. Pass 1 (broader-Pass anchored at Source) und Pass 2 (narrower-Pass anchored at Target) emittieren beide Korrespondenzen mit Relation `<` über `(s, t)`-Paare; die RRF-Fusion läuft über genau diesen einheitlichen Schlüsselraum. Konsequenzen, die methodisch dokumentiert sein müssen, nicht als Bug behandelt:
  1. **`per_relation_strict.superclass` ist by design 0 für C-Runs** — wir emittieren keine `>`-Predictions. Hauptmetrik ist `per_relation_strict.subclass.@K` und `mrr_per_relation_strict.subclass`.
  2. **C=off ist auch nur der broader-Pass von Source aus, mit Output-Relation `<` only** — nicht der existierende `MatcherAsymmetricRetrieval`-Zweipass-Modus, der zusätzlich `>`-Predictions emittiert. Das hält A=off vs. A=on (im C-Kontext) auf der gleichen Output-Relation und damit metrisch vergleichbar.
  3. **Die spiegelbildliche Direction `>` (Source ⊃ Target) ist ein separates Experiment**, das analog mit narrower-anchored Pass 1 + broader-anchored Pass 2 fahren würde. Vertagt; nicht Teil des C-Sweeps.
