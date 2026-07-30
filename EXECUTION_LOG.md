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

## Phase 2–5 — offen (Checkpoint)

- **Phase 2** (Klärungen: 8a22646-vs-a24e146, stage1_ablation_pooled.csv,
  E16_18_evidence holen) — bereit, offline ausführbar.
- **Phase 3.1/3.3/3.4** (results-final/, LICENSE, THIRD_PARTY, data/gold-standard) — bereit.
- **Phase 3.2 README** — BLOCKIERT: Draft kommt nach `~/Desktop/benchmark/README.md`.
- **Phase 4** — BLOCKIERT: kein `gh` CLI (→ Variante A: RELEASE_NOTES + Schritte
  vorbereiten, Antonio führt im Browser aus); Annotator-Workbooks A/B/C + Abgabe-PDF
  kommen in den Austauschordner.
- **Phase 5** (Branch-Löschung, Zip-Auslagerung) — erst nach separatem, explizitem OK.
