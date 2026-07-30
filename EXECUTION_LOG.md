# EXECUTION_LOG.md — Repo-Finalisierung beyondequivalence

Basis: `docs/REPO_INVENTORY.md`. Regeln eingehalten: kein Rebase, kein
History-Rewrite, kein Force-Push. Branch-Löschung/Zip-Auslagerung erst Phase 5.

---

## Phase 0 — Sicherheitsnetz ✅ (2026-07-30)

Annotierte Tags auf die Thesis-zitierten SHAs gesetzt und gepusht:

| Tag | SHA |
|---|---|
| `thesis-stage1-ablation` | d11c97e |
| `thesis-query-swap` | e98c0b3 |
| `thesis-matrix-analysis` | a24e146 |

`git push origin --tags` → 3 `[new tag]` bestätigt.

**VERIFY (`git tag --contains <sha>`):**
- d11c97e → `thesis-matrix-analysis thesis-stage1-ablation` (letzterer ist der eigene;
  ersterer, weil a24e146 von d11c97e abstammt — harmlos) ✓
- e98c0b3 → `thesis-query-swap` ✓
- a24e146 → `thesis-matrix-analysis` ✓

Alle drei SHAs jetzt tag-erreichbar (vorher: von keinem Tag, nur Feature-Branch).

---

## Phase 1 (revidiert) — main = Abgabestand ✅ (2026-07-30)

**pytest-Baseline VOR Merge** (feat/stage2-Tip 1cfa348): **81 passed / 0 failed** (10.4s).

**1–2. Merge feat/stage2-relation-classifier → main (`--no-ff`)**
- Merge-Commit `b1935fd`.
- **VERIFY-A** `git diff feat/stage2-relation-classifier main --stat` → **LEER ✓**
  (main-Baum == finaler Pipeline-Stand).

**3. SWAP-GATE** `git diff main...feat/stage1-swapped-retrieval --name-status`:
```
M   THESIS_NOTES.md
M   evaluation_recall.py     <-- Modifikation an Pipeline-Evaluationscode -> Gate greift
A   docs/swap_ablation_registration.md
A   ... (sonst nur A-Einträge: scripts/, jobs/, tests/, swap_retrieval.py)
```
→ Regel: **NICHT mergen.** `git branch -m feat/stage1-swapped-retrieval
experiment/stage1-query-swap`; neuen Namen gepusht (`[new branch]`), Upstream gesetzt.
Alter Remote-Ref `origin/feat/stage1-swapped-retrieval` bleibt bis Phase 5.
e98c0b3 doppelt gesichert: Branch `experiment/stage1-query-swap` + Tag `thesis-query-swap`.

**4. Swap-Registrierung gespiegelt** → `docs/stage1_swap_registration.md`
(Quell-Header: commit e98c0b3, Branch/Tag genannt; Original bleibt Source-of-Record).

**5. postreg_exploration/** (ex `precision_handoff/`) committet `e3ddd46`, mit
README-Kopf (post-registration, offline auf geloggten Stage-2-Outputs, keine neue
Inferenz, Bezug Thesis-Appendix "Post-Registration Exploration"). `.DS_Store` entfernt.

**6. docs/REPO_INVENTORY.md** + Gold-Build-Skripte (`goldstandard_ebay/build_ebay_target.py`,
`csv_to_owl.py`, `vdi/vdi2owl.py`) getrackt → commit `bc1718b`.

**VERIFY-B** (`git merge-base --is-ancestor <sha> main`):
- d11c97e → **OK ✓**
- a24e146 → **OK ✓**
- e98c0b3 → **nicht auf main — BY DESIGN** (Swap-Gate → rename); erreichbar via
  Tag `thesis-query-swap` + Branch `experiment/stage1-query-swap`.

**VERIFY-C** (pytest auf gemergtem main): **81 passed / 0 failed** (2.3s) —
identisch zur Baseline, **keine neuen Fehler**. Kein benchmark.zip-Altfehler aufgetreten.

**Push:** `git push origin main` → `f7d6268..bc1718b` (forward-only, kein force).

**main-Historie (Spitze):**
```
bc1718b docs: REPO_INVENTORY + swap reg + gold scripts
e3ddd46 Add post-registration Stage-2 precision exploration
b1935fd Merge Stage-2 final experiment state into main (thesis submission)
```

**Phase-1-Status: abgeschlossen, alle Verifies grün** (e98c0b3-Nichtmerge ist die
gewollte Konsequenz des Swap-Gates, nicht ein Fehlschlag).

---

## Phase 2 — Klärungen ✅ (2026-07-30)

**2.1 — 8a22646 vs a24e146:** a24e146-Fassung existiert direkt
(`results/matrix_analysis_a24e146.{md,json}`, 2026-06-19; 3 identische Kopien,
JSON-md5 `b5da2623`). a24e146 ist **ancestor** von 8a22646 (8a ist späterer
Nachkomme mit Zusatzmetriken Micro-F1/conditional-Macro). **Keine Regeneration
durchgeführt** → keine Regenerations-Divergenz. Divergenz-Check der geteilten Zellen:
**Punktschätzer IDENTISCH** (g3-text Macro-cond: gemma 0.377/0.377, gpt-oss
0.479/0.479, llama 0.229/0.229, mistral 0.233/0.233); **nur Bootstrap-CI-Klammern
≤0.002 abweichend** (Resampling-Rauschen, Seeds [7,42,123]; keine Code-/Daten-/Seed-
Divergenz der Metrik). Verdikt: Zahlen stimmen überein → a24e146 nach results-final/.
CI-Notiz transparent an Antonio gemeldet (keine stille Auswahl).

**2.2 — `stage1_ablation_pooled.csv`:** gebaut aus 96 `results/ablbi_*_d11c97e/metrics.json`
(2 Embedder × 2 LoRA × 2 Verbalization-A × 2 Template-B × 6 Datasets), Felder
subclass/superclass R@20 + R@10 + pooled_R20. → `results-final/`.

**2.3 — `E16_18_evidence.tsv`** (2301 Zeilen) aus `~/Desktop/benchmark/cc_dumps_r4/`
→ `results-final/`.

## Phase 3.1 / 3.3 / 3.4 ✅ (2026-07-30)  — commit c2d5315

**3.1 results-final/** (getrackt, kuratiert, klein — 236K): `matrix_analysis_a24e146.{md,json}`,
`stage1_ablation_pooled.csv`, `E16_18_evidence.tsv`, `ltfp_audit_adjudicated_2026-07-19.tsv`,
`stufeB_analysis.{md,json}`, `matrix_cells/<cell>/metrics.json` (16 Zellen, **keine**
`predictions.tsv`), `README.md` (Index + a24/8a-Notiz).

**3.3 LICENSE (MIT)** für Code + **THIRD_PARTY_NOTICES.md**: eigene Annotationen CC BY 4.0;
eBay-Kategorien (§2) und VDI 4081 (§3) als Fremdmaterial; OAEI via Zenodo-DOI
10.5281/zenodo.17091043 verlinkt, **nicht redistribuiert**. ⚠️ Copyright-Halter in
LICENSE = "Antonio Markic" (Git-User zeigt "Anton Klevers" — **von Antonio bestätigen**).

**3.4 data/gold-standard/** (getrackt via `.gitignore`-Negation `!data/gold-standard/`;
1.5M): `reference_full.rdf` (2048), `reference_seed.rdf` (451), `ebay_kfz_target.owl`,
`vdi_karosserie_source_pos.owl`, `subsumption_gold/` (TSV + derive-Skript + findings),
`README.md` (Protokoll A/B/C, K1–K5, R1/R2, H1–H7-Verweis). Gold-Build-Skripte bereits
in Phase 1 getrackt. ⚠️ eBay/VDI-OWL sind Fremd-Derivate → Redistributions-Freigabe vor
"public" prüfen (in THIRD_PARTY_NOTICES §Summary vermerkt).

**Push:** `d57886d..c2d5315` (main, forward-only).

---

## Phase 3.2 — README ✅ (2026-07-30)  — commit 4fb7db5

Antonios reviewter Draft (Repo-Root `README.md`) 1:1 übernommen. **Release-Link-Verify
aufgelöst** → deterministische URL `.../releases/tag/v1.0-thesis` (Zeile 35).
**Zahlenabgleich gegen results-final — alle Headline-Claims konsistent, keine Abweichung:**
- dir-acc: reasoning 0.898–1.000 (≈"0.90–1.00"), instruct 0.211–0.766 ("0.21–0.77") ✓
- B2 "+0.087": = B2(0.4210) − baseline_anchor(0.334) = **+0.0870** ✓ (nicht B2−B1=+0.074)
- vdi "19/20 directional, 1/20 equivalence": = Seed-Ratio 429/451 dir + 22/451 eq ✓
- Der 3. `<!-- verify -->` (BibTeX, Zeile 108) blieb wie im Draft (kein Library-Record;
  von dir nicht als aufzulösen benannt).
- Minor-Notiz (ursprünglich): "1 in 20 equivalence" gilt gegen die 451-Seed; gegen
  die 2048-Full ist Äquivalenz ~1 % (22/2048). **Follow-up-Commit (2026-07-30):**
  Headline-Finding 3 im README auf **Basis-Klarstellung Seed vs. Vollreferenz**
  präzisiert (451-Seed: ~19/20 dir + ~1/20 eq; 2048-Full: eq-Anteil ~1 % → Argument
  wird stärker). **Der Thesis-Satz bleibt unberührt (Druckstand)** — nur die README-
  Formulierung geschärft. Verteidigungsnotiz für Antonio: die scheinbare Diskrepanz
  ist reiner Bezugsrahmen (Seed-Denominator vs. Closure-Denominator), kein Zahlenfehler.

## Phase 4 — Vorbereitung ✅ (2026-07-30, ohne Austauschordner-Content)  — commit siehe unten

`RELEASE_NOTES.md` entworfen: (A) Release-Body inkl. Verweis auf die drei `thesis-*`-Tags
+ `v1.0-thesis`, (B) sha256-Checksummen, (C) Browser-Schrittfolge für Antonio.
**Ready-Assets (sha256 berechnet):**
- `thesis_results_final_2026-07-21.zip` (70M) `398b02e8…`
- `stage2_precision_E16_E17_E18_2026-07-22.zip` (5.0M) `1cec08f6…`
- `sbert_e15_results_2026-07-17.zip` (9.9M) `a02d1b75…`
- `gold-standard-v1.zip` (106K, aus getracktem data/gold-standard/ gebaut) `a444963f…`
**Offene Asset-Slots:** `Markic_2026_BeyondEquivalence_thesis.pdf`, Annotator-Workbook-Bundle
(A/B/C) — sha256 bei Ankunft im Austauschordner.

---

## PROVENANCE-VERMERK — Git-Autor (Punkt 4)

**Sämtliche Commits dieses Repos wurden unter der Git-Identität `Anton Klevers`
erstellt** (`user.name=Anton Klevers`), während **Autor/Urheber der Arbeit
Antonio Markic** ist (Thesis `Markic_2026_BeyondEquivalence`; LICENSE-Copyright =
Antonio Markic). Die Diskrepanz ist gewollt belassen — die Git-Identität wird NICHT
umkonfiguriert und die Historie NICHT umgeschrieben. `git blame`/`git log` zeigen
daher "Anton Klevers"; die maßgebliche Autorschaft ist Antonio Markic.

---

## Offen (Checkpoint)

- **Phase 4.1** (Annotator-Workbooks A/B/C scrubben + CSV-Export + annotation/-README) —
  gated bis Workbooks im Austauschordner.
- **Phase 4.2–4.4** (Release erstellen, Topics/Description, public) — Antonio im Browser
  per `RELEASE_NOTES.md`; **public nur nach separatem OK**.
- **Phase 5** (Branch-Löschung lokal+origin, Zip-Auslagerung nach `~/Desktop/repo_archive_2026-07/`,
  Cache-Löschung) — erst nach separatem, explizitem OK. `experiment/stage1-query-swap`
  bleibt DAUERHAFT. Hinweis: Release VOR Zip-Auslagerung erstellen (Assets liegen noch im Arbeitsdir).
- **Klärpunkt:** LICENSE-Copyright-Halter = "Antonio Markic" (gehalten bis Bestätigung).
