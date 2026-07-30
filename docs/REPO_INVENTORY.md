# REPO_INVENTORY.md — beyondequivalence Repo-Inventur

**Erstellt:** 2026-07-30 · **Modus:** Discovery / Report-only (keine Änderungen,
keine Commits, kein History-Rewrite). **Repo:** `git@github.com:Mantoni9/beyondequivalence.git`
· HEAD `1cfa348` auf `feat/stage2-relation-classifier`. Grundlage für spätere
Aufräumarbeiten; Ausführung erfolgt als separater Auftrag nach deiner Entscheidung.

---

## ⚠️ Kernbefund zuerst (das eine echte Risiko)

**Die drei in der Thesis zitierten Revisionen sind NICHT auf `main` und von KEINEM
Tag erreichbar — jede hängt an genau einem ungemergten Feature-Branch:**

| SHA | Thesis-Bezug | erreichbar via | auf main? | tag? |
|---|---|---|---|---|
| `d11c97e` | Stage-1-Ablation | `origin/feat/stage2-relation-classifier` | ✗ | ✗ |
| `e98c0b3` | Query-Swap | `origin/feat/stage1-swapped-retrieval` | ✗ | ✗ |
| `a24e146` | Matrix-Analyse | `origin/feat/stage2-relation-classifier` | ✗ | ✗ |

→ Werden diese Branches bei einer Aufräumaktion gelöscht, werden die Commits
**unerreichbar und irgendwann GC-gelöscht** — die Thesis-Footnote-Links brechen.
**Vor jeder Branch-Bereinigung müssen diese SHAs mit annotierten Tags fixiert
werden** (siehe Empfehlung #1). Kein Rebase/Rewrite nötig oder erlaubt — reines
Tagging genügt.

---

## 1. Branch-/Tag-Lage

**8 lokale Branches (alle zu `origin` gespiegelt).** ahead/behind vs `main`:

| Branch | behind | ahead | Status |
|---|---|---|---|
| `main` | 0 | 0 | Basis |
| `feat/stage2-relation-classifier` | 0 | **69** | **NICHT gemergt** — trägt Stage-2 (Matrix, E16/E17, `d11c97e`, `a24e146`) |
| `feat/stage1-swapped-retrieval` | 0 | **12** | **NICHT gemergt** — Query-Swap (`e98c0b3`) |
| `A-textverbalization` | 19 | 0 | vollständig in main enthalten (subsumiert) |
| `C-bidirectional-consolidation` | 21 | 0 | vollständig in main enthalten |
| `lora-subsumption-finetune` | 3 | 0 | vollständig in main enthalten |
| `main-ablation-full` | 12 | 0 | vollständig in main enthalten |
| `t2-pin-validation` | 1 | 0 | vollständig in main enthalten |

**Finaler Experimentstand auf main gemerged? NEIN.** Der Stage-2-Endstand (69
Commits inkl. der zitierten SHAs) lebt nur auf `feat/stage2-relation-classifier`.
Stage-1 wurde einst gemergt (Tag `stage1-final-v1` → `f7d6268` „Merge Stage 1 …
into main"), aber der spätere Stage-2-Block nicht.

**Tags (2):**
- `equivalence-baseline-v1` → `4621bcd` (Zenodo-Loader)
- `stage1-final-v1` → `f7d6268` (Stage-1-Merge in main)

Keiner der Tags deckt eine der drei zitierten SHAs ab.

---

## 2. Arbeitszustand

- **Modifizierte getrackte Dateien:** 0. **Stashes:** keine.
- **Untracked:** 24 Einträge (`git status --porcelain`), u. a.: 9 Zips (s. §3),
  `goldstandard_ebay/` (eBay/VDI-Golddaten, s. §5), `precision_handoff/`,
  `ltfp_audit_adjudicated_2026-07-19.tsv`, `review_deliverables_2026-07-07/`,
  `vdi-ebay.xlsx`, `ebay_owl.zip`.
- Historie ist **sauber**: 129 getrackte Dateien, alle < 0.1 MB.

---

## 3. Größen

- **Working Tree gesamt: 1.8 GB** · **`.git`: 15 MB** (schlanke, saubere Historie —
  größtes Blob je committet: `run_subsumption_experiment.py`, ~0.1 MB).
- Der gesamte Ballast ist **untracked oder gitignored**, nicht in der Historie.

**Top-Dateien im Tree (alle gitignored unter `results/` oder untracked Zips):**

| Größe | Pfad | Zustand |
|---|---|---|
| 171 MB | `results/swap_llama-embed-nemotron-8b_lora-on_g2-diseases_e98c0b3/passes.tsv` | gitignored |
| 170 MB | `results/swap_qwen3-embedding-8b_lora-off_g2-diseases_e98c0b3/passes.tsv` | gitignored |
| 104 MB | `results.zip` | untracked |
| 70 MB | `thesis_results_final_2026-07-21.zip` | untracked |
| 42 MB | `stage2_all_results_2026-07-12.zip` | untracked |
| ~18 MB ×15 | `results/ablbi_*_g2-diseases_d11c97e/predictions.tsv` | gitignored |
| 17 MB | `results/e16/e16_h3_repair_log.tsv` | gitignored |

**Untracked Zips (Summe der porcelain-Untracked ≈ 244 MB):** `results.zip` 104M,
`thesis_results_final_2026-07-21.zip` 70M, `stage2_all_results_2026-07-12.zip` 42M,
`sbert_e15_results_2026-07-17.zip` 9.9M, `stage2_precision_E16_E17_E18_2026-07-22.zip`
5.0M, `vdi_owl_neu.zip` 648K, `review_deliverables_2026-07-07.zip` 436K,
`ebay_owl.zip` 96K, `subsumption_gold.zip` 36K.

**Caches im Tree (gitignored, nur lokale Hygiene):** `__pycache__/` (root, `scripts/`,
`tests/`, `tracks/`), `.pytest_cache/`. **Nicht committet** — reine Working-Tree-Reste.

> Fazit: Es besteht **kein History-Bloat-Problem**. `.gitignore` fängt
> `results/`, `logs/`, `models/`, `data/`, `benchmark.zip`, `wandb/`, `__pycache__`,
> `.env*` sauber ab. Aufräumen betrifft nur das Working-Verzeichnis, nicht die Historie.

---

## 4. Secrets-Scan (Tree UND History)

- **`gitleaks detect` über 158 Commits → `no leaks found`** (Exit 0).
- Regex-Muster (WANDB/HF/OpenAI-Keys, `-----BEGIN … PRIVATE KEY`, `password=…`)
  über `git log --all -p`: **0 Treffer.**
- **Keine `.env`/Credential-Datei je getrackt** — nur die erwarteten Templates
  `.env.{bwuni,dws,local}.template` (keine Werte).
- 2 grep-Treffer in *untracked* `review_deliverables_.../P2_sbert/query_*.py` sind
  **False Positives** (`SBERT_TOKENS = ("all-minilm","sbert")` — Modellnamen, keine Keys).

**→ Secrets-Lage sauber.** (Hinweis am Rande: W&B-Auth liegt in `~/.netrc` außerhalb
des Repos — kein Repo-Problem, aber beim Teilen des Rechners beachten.)

---

## 5. Daten & Lizenz

- **LICENSE: FEHLT** (weder Root noch getrackt). Für ein zitiertes/öffentliches Repo
  eine echte Lücke.
- **OAEI-Benchmarks (g1–g7, mouse–human): NICHT im Repo** — korrekt gitignored
  (`benchmark.zip`), offizielle Quelle in `.gitignore` dokumentiert (Zenodo DOI
  `10.5281/zenodo.17091043`). Loader erwartet `benchmark.zip` im Root. Sauber gelöst.
- **eBay-/VDI-Golddump:** liegt **untracked** im Working Tree (`goldstandard_ebay/`,
  `vdi-ebay.xlsx`, diverse Zips). Getrackt ist davon nur `goldstandard_ebay/make_reference.py`.
- **PII: unkritisch.** Spaltenköpfe der Golddaten sind reine Taxonomie-Struktur —
  `vdi_ebay_gold_clean.csv`: `vdi_code,ebay_id,ebay_uri,state`; `ebay_kfz_tree.csv`:
  `id,label,parent_id,depth,path`. **Keine** Seller/Buyer/Order-IDs, keine E-Mails,
  keine Klarnamen. `ebay_id` ist eine Kategorie-Knoten-ID, kein Nutzer. Restrisiko:
  eBay-Kategoriedaten sind fremdes Urheberrecht/ToS — beim Veröffentlichen prüfen
  (kein PII-, aber ein Lizenz-Thema).

---

## 6. Ergebnis-Auffindbarkeit

**Zentrales Problem: Im getrackten Repo liegt KEIN einziges Ergebnis.** Ein externer
Klon liefert Code + `docs/` + Registrierungen, aber alle Resultate sind in
gitignored `results/`, in untracked Zips oder außerhalb des Repos. Kombiniert mit
**fehlendem README** kann Dr. Hertling weder Ergebnisse finden noch reproduzieren.

| gesuchtes Artefakt | Fundort | Zustand |
|---|---|---|
| `stage1_master.csv` | **nirgends gefunden** (existiert nicht unter dem Namen) | ✗ fehlt |
| Matrix-Zellen | `results/stage2_results_bundle/02_matrix_cells/` (125 Dateien, 57 MB) | gitignored |
| 2.301-Zeilen-Langformat (`E16_18_evidence.tsv`) | `~/Desktop/benchmark/cc_dumps_r4/` | **außerhalb Repo** |
| ltfp-Audit | `ltfp_audit_adjudicated_2026-07-19.tsv` (Root) + `results/matrix_ltFP_audit_*.tsv` | untracked / gitignored |
| Gold-2048 | `goldstandard_ebay/reference_full.rdf` | untracked |
| stage1_pulls | `review_deliverables_2026-07-07/stage1_pulls.md` | untracked |
| stufeB-Analyse | `results/stufeB_analysis.{json,md}` + Bundle | gitignored / untracked |
| Matrix-Analyse | `results/matrix_analysis_8a22646*.{md,json}` | gitignored |

**Thesis → Skript → Eingabe (rekonstruierbar aus `scripts/` + `docs/`):**
- Stage-1-Ablation (T1/T2, LoRA, A/B-Lever) → `scripts/ablation_bidirectional.py` →
  `results/ablbi_*_d11c97e/metrics.json`
- Matrix-Zellen/Analyse → `scripts/analyze_matrix.py`, `matrix_stats.py` →
  `results/matrix_*_seed42_*/predictions.tsv`
- StufeA/B → `scripts/analyze_stufeA.py`, `analyze_stufeB.py`, `stage2_bothorder.py`
- E16/E17/E18 → `scripts/e16_analysis.py`, `e17_{verify,batchcalib,v3_calibrate,test_analysis,viability}.py`, `e18_consistency.py`
- ltFP-Audit → `scripts/build_ltfp_audit_sample.py`, `ltfp_corrected_precision.py`,
  `foreign_audit*.py` → `ltfp_audit_adjudicated_2026-07-19.tsv`
- credited-Metrik → `scripts/closure_credit.py`, `closure_analysis.py`
- Query-Swap → `results/swap_*_e98c0b3/passes.tsv`

**README-Zustand:** Es existiert **kein** Repo-README (nur `.pytest_cache/README.md`
und mein `precision_handoff/README_THESIS_HANDOFF.md`). Für einen externen Leser
fehlt damit: Einstieg, Reproduktions-Anleitung, Daten-Provenienz (Zenodo-DOI, eBay-Gold),
und das Tabellen→Skript→Artefakt-Mapping.

---

## 7. Registrierungen (Pre-Registration)

**Im Repo getrackt (`docs/`):**
- `E16_registration_2026-07-19.md` (+ `E16_addendum_H3_structural_gate_2026-07-21.md`)
- `E17_verification_secondpass_registration_2026-07-21.md`
- `E18_logical_consistency_filter_2026-07-21.md`
- `stage2_matrix_registration.md`, `stage2_stufeA_registration.md`,
  `stage2_stufeB_registration.md`, `stage2_e15_fewshot_registration.md`

**Lücken/Hinweise:**
- Die **Stage-1-Swap-Registrierung** ist der Commit `e98c0b3` selbst („registration
  record — outcomes, candidate set, adoption-consequence procedure") und lebt auf
  `feat/stage1-swapped-retrieval`, **nicht in main/`docs/`** → mit den SHAs am selben
  Löschrisiko (§ Kernbefund).
- Der **Coherence-Extractor** (jüngste Offline-Arbeit, `precision_handoff/`) ist
  **nicht registriert** — bewusst post-Thesis; nur relevant, falls er noch einfließt.

---

## 8. Tote Enden — Kandidatenliste (nur Vorschläge; NICHT löschen ohne Freigabe)

> Achtung: registrierte Negativ-Arme sind **keine** Kandidaten. Vieles hier ist
> tatsächlich Teil registrierter Ablationen — daher als „vor Anfassen
> Registrierungsstatus prüfen" markiert, nicht als Löschempfehlung.

| Kandidat | Begründung | Verdikt |
|---|---|---|
| `ensemble_rrf.py`, `scripts/fuse_crossmodel_rrf.py`, `ensemble_finetune_compare.py` | Cross-Model-RRF-Fusion — laut Projektstand „0/64 negativ, gedroppt" | **wahrscheinlich registrierter Negativ-Arm → behalten**; nur prüfen, ob Ergebnis dokumentiert ist |
| `finetune_lora.py`, `extract_lora_from_hybrid_save.py`, `lora_inference_sanity.py`, `prepare_wordnet_triplets.py`, `jobs/job_dws_lora_*.sh` (5) | LoRA-Training/Eval | **LoRA on/off ist registrierte Stage-1-Ablations-Achse (`d11c97e`) → behalten** |
| `cleanup_wandb_runs.py` | reines Utility (W&B-Aufräumen) | Utility — behalten oder nach `scripts/tools/` |
| `analyze_sweep.py` (Root) | früher Sweep-Analyzer, evtl. durch `scripts/analyze_matrix.py` abgelöst | **prüfen**, ob noch referenziert |
| untracked Zips (§3, ~244 MB) | Snapshot-Archive (`results.zip`, `thesis_results_final*.zip`, …) | Working-Tree-Altlast — archivieren/auslagern, nicht ins Repo |
| Working-Tree-Caches (`__pycache__`, `.pytest_cache`) | gitignored, nur lokal | lokal löschbar, kein Repo-Effekt |

**Keine eindeutig verwaisten getrackten Kern-Skripte gefunden** — die 129 getrackten
Dateien sind überwiegend Pipeline/Analyse/Registrierung.

---

## 9. Empfehlungen (priorisiert, Aufwand S/M/L)

1. **[S] Thesis-SHAs mit annotierten Tags fixieren — VOR jeder Branch-Bereinigung.**
   `git tag -a thesis-stage1-ablation d11c97e`, `thesis-query-swap e98c0b3`,
   `thesis-matrix-analysis a24e146`, dann `git push origin --tags`. Macht die
   Footnote-Commits dauerhaft erreichbar, ohne History-Rewrite. **Höchste Priorität.**
2. **[M] Repo-README erstellen** mit: Überblick, Reproduktions-Schritte, Daten-Provenienz
   (Zenodo-DOI, eBay-Gold-Herkunft), und dem Tabellen/Figur → Skript → Artefakt-Mapping
   aus §6. Ohne das findet ein externer Leser nichts.
3. **[S] LICENSE hinzufügen** (Code-Lizenz wählen; eBay-/OAEI-Daten separat als
   Fremd-Lizenz kennzeichnen).
4. **[M] Ergebnis-Hosting entscheiden.** Da `results/` gitignored und nicht
   aus dem Repo reproduzierbar ist: kuratiertes Bundle (`stage2_results_bundle/`,
   `thesis_results_final_2026-07-21.zip`) als GitHub-Release oder Zenodo-Deposit
   veröffentlichen und im README verlinken. Klärt „wo sind die Ergebnisse?".
5. **[S] Stage-1-Swap-Registrierung nach main/`docs/` spiegeln** (Kopie des Inhalts
   von `e98c0b3`), damit alle Pre-Registrations an einem Ort und nicht am Branch-Löschrisiko hängen.
6. **[S] Entscheiden, ob `feat/stage2-relation-classifier` (+69) nach main gemergt wird.**
   Falls ja: no-fast-forward-Merge (kein Rewrite), dann sind die Thesis-SHAs auch über
   main erreichbar. Falls nein: mindestens Tags (#1) sind Pflicht.
7. **[S] Fully-merged Stale-Branches aufräumen** (`A-textverbalization`,
   `C-bidirectional-consolidation`, `lora-subsumption-finetune`, `main-ablation-full`,
   `t2-pin-validation` — alle 0 ahead, in main enthalten). **Niemals** die zwei
   ungemergten Feature-Branches oder Branches mit Thesis-SHAs anfassen.
8. **[S] Untracked Snapshot-Zips (~244 MB) auslagern** (nach `~/Desktop/…` oder Release),
   aus dem Repo-Arbeitsverzeichnis entfernen — reduziert Verwirrung, nicht die Historie.
9. **[S] Gold-Build-Provenienz sichern:** `goldstandard_ebay/`-Build-Skripte
   (`build_ebay_target.py`, `csv_to_owl.py`, `vdi/vdi2owl.py`) tracken (nur Skripte,
   nicht die großen Daten), damit das 2048er-Gold reproduzierbar dokumentiert ist.
10. **[S] `stage1_master.csv` klären:** in der Thesis referenziert(?), aber nicht
    auffindbar — entweder Herkunft dokumentieren oder aus einem `results/`-Artefakt
    rekonstruieren und im README benennen.

---

### Anhang — verwendete read-only-Kommandos (Reproduzierbarkeit)
`git for-each-ref` (ahead/behind) · `git merge-base --is-ancestor` · `git tag`/`rev-list`
· `git status --porcelain` · `git stash list` · `du -k` (Tree) ·
`git rev-list --objects --all | git cat-file --batch-check` (History-Blobs) ·
`git check-ignore` · `git ls-files` · `gitleaks detect --redact` ·
`git log --all -p | grep <patterns>`. Keine schreibenden Operationen.
