# Thesis Methods Notes

## Stage-2 Dev/Test-Protokoll (registriert 2026-06-12 — BINDEND für alles weitere Stage-2-Tuning)

- **Dev = {g7-literature, g5-groceries}** · **Test = {mouse-human, g3-text, g1-web, g2-diseases}** — Test bleibt unangetastet bis zur finalen prä-registrierten Modell-Matrix.
- **Rationale:** g7 und g5 sind die `>`-lastigen Fälle (g7: 52 von 82 gerichteten Gold-Paaren; g5: 113 von 127) — genau dort sitzt der zu attribuierende Subclass-Prior. **Ehrliche Offenlegung:** g7 ist KEIN unberührtes Dev-Set — die v2-Baseline (Run 255471) und die v3a/v3b-Iterationen liefen bereits auf g7; alle bisherigen Prompt-Entscheidungen sind also g7-informiert. g5 kommt als zweites, bisher Stage-2-unberührtes Dev-Dataset hinzu; die Konsistenz-Regel (Effekte müssen über g7 UND g5 gleichgerichtet sein) existiert, um g7-Tuning-Artefakte zu fangen.
- **Regel:** Jeder Tuning-Run auf einem Test-Dataset macht dieses Protokoll ungültig. Keine Ausnahmen; die Test-Datasets erscheinen erst in der finalen Matrix wieder.
- **Metrik-Konvention:** Alle Tuning-Metriken sind **reranker-konditional** (nur Gold-Paare, die im Stage-1-Kandidatenset vorhanden sind) — isoliert die Stage-2-Klassifikationsleistung von der Stage-1-Recall-Decke (auf `⊐` bindend, siehe Coverage-Befunde).
- Stage-1-Kandidaten: die eingefrorenen d11c97e-TSVs (Qwen3-noLoRA / path_context / T2, top-20 pro Richtung). Der Query-Swap wurde getestet und NICHT adoptiert (Job 262057, RIDE-ALONG-Verdict) — es gibt keine neuen TSVs und keinen Bridge-Lauf; R0 (v2 auf g5) ist ein eigenständiger Baseline-Lauf.

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

## LoRA-Fine-Tune-Ergebnis (2026-05-05) — bimodaler Befund

Branch `lora-subsumption-finetune` wurde nach drei Trainings- und Inferenz-Iterationen evaluiert. Das Aggregat über beide 8B-Modelle × 6 Datasets liegt bei Δ MRR per_relation_strict.subclass = **−2.67 %** und damit nominell im "Ambivalent"-Band des dreigeteilten Verdict-Schemas (>+3 % / ±3 % / <−3 %). **Branch wird NICHT in `main` gemergt.**

Das Aggregat verdeckt aber einen strukturell bimodalen Befund pro Modell:

| Modell | Mittel-Δ MRR über 6 Datasets | Befund |
|---|---:|---|
| `llama-embed-nemotron-8b` | **+16.3 %** | Systematischer Lift. Alle 6 Datasets profitieren oder sind neutral. |
| `qwen3-embedding-8b`      | **−21.6 %** | Systematische Verschlechterung. Catastrophic Forgetting auf 4/6 Datasets, am stärksten g7-literature mit −73.9 %. |

**Methodische Erklärung — "Stacked-LoRA-Hypothese":**

Qwen3-Embedding-8B wird im HF-Repo bereits als LoRA-getrainte Variante released — die Modelkarte beschreibt das Modell als Base + eingebrannten LoRA-Adapter (im `model.safetensors` gemergt, nicht als separater PEFT-Adapter). Unser zusätzlicher LoRA-Adapter (r=16) sitzt in derselben `q_proj`/`k_proj`/`v_proj`/`o_proj`-Subspace wie der Pretrained-Adapter und überschreibt dessen Gradient-Information mit signifikant kleiner Trainings-Set (177 k WordNet-Triplets vs. presumably mehrere Millionen Pretraining-Triplets). Das Ergebnis ist Catastrophic Forgetting der Pretrained-Adapter-Information.

Llama-embed-nemotron-8b basiert auf Llama-3.1-base + custom latent-attention pooler ohne pretrained LoRA-Schicht. Unser LoRA findet hier den vorgesehenen "leeren" Adapter-Slot und kann sauber lernen — daher der systematische Lift.

**Konsequenz für die Thesis:**
Die Ergebnisse werden im Limitations-Block als bimodaler Befund berichtet, nicht als einfaches "Negativ-Resultat". Konkret:
1. LoRA-Fine-Tuning auf Embedding-Modellen funktioniert reproduzierbar gegen ein Llama-3-Base-Derivat ohne Pretrained-LoRA (Nemotron: +16 %).
2. Dasselbe Verfahren auf einem bereits LoRA-getuneten Modell (Qwen3-Embedding) führt zu Catastrophic Forgetting — die Wahl der "Fine-Tunability" eines Modells hängt vom Pretraining-Stack ab, nicht vom Task oder den LoRA-Hyperparametern.
3. Die formale Aggregat-Schwelle von ±3 % ist methodisch unzureichend bei zwei Modellen mit gegensätzlichem Verhalten. Pro-Modell-Verdicts werden ab dem Compare-Skript 2026-05-05 explizit ausgegeben (siehe `ensemble_finetune_compare.py`).

**Deferred:**
- Re-Training auf Qwen3 mit höherem Rang (r=64), niedrigerer Lernrate, oder layer-spezifischem Targeting (z.B. nur die letzten 8 Layer) zur Vermeidung der Pretrained-Adapter-Kollision. Würde testen, ob der Forgetting-Befund modell-spezifisch oder hyperparameter-spezifisch ist. Nicht im Stage-1-Scope.
- LoRA-Adapter für Nemotron in eigenem Branch mergen, ohne Qwen3 — würde das positive Ergebnis isoliert in die Hauptablation tragen, kollidiert aber mit der ursprünglichen Branch-Disziplin (alle-oder-keinen-Merge). Antonio entscheidet bei Stage-2-Implementation.
- Methodisches Side-Finding zu PEFT 0.19.1 (siehe Limitations-Block in `extract_lora_from_hybrid_save.py`): `inference_mode=True` in `adapter_config.json` deaktiviert beim Reload den Adapter ohne lautes Versagen; `.<adapter_name>.weight`-Suffix in den safetensors-Keys führt zu unsichtbaren `unexpected_keys` beim PeftModel-Load, weil `set_peft_model_state_dict` den Suffix selbst rein-mapped. Beide Findings sind reproduzierbar dokumentiert und für andere Researcher-Pipelines mit hand-rolled Adapter-Extraktion relevant.

## Daten-Notiz: mouse-human Gold-Zählung (rohes File vs. Eval-Denominator)

Beim Bidirektional-Validierungslauf fiel auf, dass die Gold-Zahlen für **mouse-human** je nach Messpunkt differieren: ein rohes `grep` der `<relation>`-Tags in `reference.rdf` ergibt **600 `<` / 671 `>` / 676 `=`** (1947 `<Cell>`-Blöcke), während die Evaluations-Pipeline mit **545 `<` / 612 `>` / 676 `=`** (1833 Korrespondenzen) rechnet.

**Erklärung — kein Datenverlust, kein Filter:** `mouse-human/reference.rdf` enthält **114 exakte Duplikat-Cells** (55× `<`, 59× `>`, 0× `=`). `Alignment.add()` dedupliziert auf den Korrespondenz-Key `(source, target, relation)` (`Correspondence.key`), behält also die 1833 **distinct** Gold-Korrespondenzen. Es gibt **0** `(source, target)`-Paare mit widersprüchlichen Relationen — die Duplikate sind reine exakte Wiederholungen, keine Annotations-Konflikte. Es erscheint keine „Dropped non-evaluable"-Warnung, weil nichts als nicht-evaluierbar verworfen wird; die Wiederholungen werden von der Datenstruktur kollabiert.

**Konsequenz für die Thesis:** Der korrekte Recall-Denominator ist die **distinct**-Zählung (545 `<` / 612 `>`); die rohe File-Zählung (600/671) ist durch Datei-Duplikate inflationiert. Nur mouse-human ist betroffen — die anderen fünf STROMA/TaSeR-Datasets (g1-web, g2-diseases, g3-text, g5-groceries, g7-literature) haben 0 Duplikate, dort sind rohe und distinct Zählung identisch.
