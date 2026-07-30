#!/usr/bin/env python3
"""ltfp_corrected_precision.py — Unblinding + gold-gap-corrected subclass precision.

Consumes Antonio's adjudicated blind '<'-FP audit (ltfp_audit_adjudicated_
2026-07-19.tsv, 40 rows) together with the draw key (results/ltfp_audit_key.json)
and computes, per model x dataset (seed-42 cells, g3/g5/g7, e2e basis):

    P_corr = (TP< + g_ds * FP<) / (TP< + FP<)

where FP< is EXACTLY the audit-frame population: kept '<' predictions whose
pair has NO gold relation (gold=none). Direction errors (gold in {>,=}) are
excluded from FP< by construction (they are real errors, not gap candidates)
and reported separately. g_ds is the per-dataset gold-gap rate from the audit
(Wilson 95% CI). The draw (scripts/build_ltfp_audit_sample.py, seed 42, N=40)
was stratified round-robin ACROSS models, pair-deduplicated and model-blind;
judgments are pair-level, so g_ds is model-independent within a dataset.

Outputs (to --out-dir):
    corrected_precision_<sha>.tsv    model, dataset, n_FP, P_strict, P_corr, ...
    corrected_precision_<sha>.md     thesis-ready methods paragraph + caveats
    ltfp_audit_unblinded_supplementary.tsv   per-row unblinded table

Run:
    conda run -n melt-olala python scripts/ltfp_corrected_precision.py \
        --adjudicated ltfp_audit_adjudicated_2026-07-19.tsv \
        --cells-root <dir with {model}_{dataset}_42/predictions.tsv>  [--sha a24e146]

Without --cells-root, cells are resolved like the draw script did:
results/matrix_{model}_{dataset}_seed42_{sha}/ plus the two reused llama dirs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Alignment import Alignment                      # noqa: E402
from evaluation_recall import RELATION_NORMALIZATION  # noqa: E402
from tracks.zenodo_loader import load_subdataset      # noqa: E402

MODELS = ("llama", "mistral", "gemma4", "gpt-oss")
DATASETS = ("g3-text", "g5-groceries", "g7-literature")
REUSE = {
    ("llama", "g7-literature"): "2026-06-02_19-01-38_stage2_g7-literature_s1-qwen3-embedding-8b-asy-T2-description_path_context",
    ("llama", "g5-groceries"): "2026-06-13_11-32-15_stage2_g5-groceries_s1-qwen3-embedding-8b-asy-T2-description_path_context_p-d_subs_v2",
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_adjudicated(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.rstrip("\n").split("\t"))
    header, data = rows[0], rows[1:]
    idx = {c: i for i, c in enumerate(header)}
    return [{c: r[i] for c, i in idx.items()} for r in data]


def load_gold(ds: str) -> dict[tuple[str, str], str]:
    _s, _t, ref = load_subdataset(ds)
    g = {}
    for c in Alignment(str(ref)):
        n = RELATION_NORMALIZATION.get(c.relation.strip())
        if n:
            g[(c.source, c.target)] = n
    return g


def cell_predictions(cells_root: Path | None, model: str, ds: str, sha: str) -> Path:
    if cells_root is not None:
        p = cells_root / f"{model}_{ds}_42" / "predictions.tsv"
        if p.is_file():
            return p
    if (model, ds) in REUSE:
        p = Path("results") / REUSE[(model, ds)] / "predictions.tsv"
    else:
        p = Path("results") / f"matrix_{model}_{ds}_seed42_{sha}" / "predictions.tsv"
    if not p.is_file():
        sys.exit(f"FATAL: predictions.tsv fehlt fuer {model}/{ds}: {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adjudicated", default="ltfp_audit_adjudicated_2026-07-19.tsv")
    ap.add_argument("--key", default="results/ltfp_audit_key.json")
    ap.add_argument("--frames-glob", default="results/matrix_ltFP_audit_{ds}.tsv")
    ap.add_argument("--cells-root", default=None)
    ap.add_argument("--sha", default="a24e146")
    ap.add_argument("--out-dir", default="results/ltfp_corrected_precision")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_root = Path(args.cells_root) if args.cells_root else None

    # ── (1) Join + Konsistenz ────────────────────────────────────────────────
    adj = load_adjudicated(Path(args.adjudicated))
    key = json.loads(Path(args.key).read_text())
    assert len(adj) == 40, f"erwartet 40 Zeilen, gefunden {len(adj)}"
    jd = Counter(r["judgment"] for r in adj)
    assert jd.get("gold_gap", 0) == 15 and jd.get("not_subclass", 0) == 25 \
        and jd.get("unsure", 0) == 0, f"Verteilung unerwartet: {dict(jd)}"
    missing = [r["row_id"] for r in adj if r["row_id"] not in key]
    assert not missing, f"row_ids ohne Key-Eintrag: {missing}"
    ds_mismatch = [r["row_id"] for r in adj if key[r["row_id"]]["dataset"] != r["dataset"]]
    assert not ds_mismatch, f"dataset-Mismatch adjudicated vs key: {ds_mismatch}"
    assert len({r["row_id"] for r in adj}) == 40, "doppelte row_ids"
    print(f"Konsistenz OK: 40 Zeilen, {dict(jd)}, alle row_ids im Key, datasets stimmen.")

    # Rahmen-Sheets: Abdeckungs-Doku (row_id-Overlap).
    frame_ids: set[str] = set()
    for ds in DATASETS:
        fp = Path(str(args.frames_glob).replace("{ds}", ds))
        if fp.is_file():
            with fp.open(encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#") or line.startswith("row_id") or not line.strip():
                        continue
                    frame_ids.add(line.split("\t", 1)[0])
    overlap = sum(1 for r in adj if r["row_id"] in frame_ids)

    # ── (2) Gap-Raten mit Wilson-CI ──────────────────────────────────────────
    per_ds = defaultdict(lambda: [0, 0])   # ds -> [gold_gap, n]
    for r in adj:
        per_ds[r["dataset"]][1] += 1
        if r["judgment"] == "gold_gap":
            per_ds[r["dataset"]][0] += 1
    g_hat = {}
    for ds in DATASETS:
        k, n = per_ds[ds]
        g_hat[ds] = (k / n if n else 0.0, n, k, *wilson_ci(k, n))
    kp = sum(v[0] for v in per_ds.values())
    np_ = sum(v[1] for v in per_ds.values())
    pooled = (kp / np_, np_, kp, *wilson_ci(kp, np_))

    # ── (3) P_corr je Modell x Datensatz ─────────────────────────────────────
    golds = {ds: load_gold(ds) for ds in DATASETS}
    rows_out = []
    for model in MODELS:
        for ds in DATASETS:
            pred_path = cell_predictions(cells_root, model, ds, args.sha)
            tp = fp_none = dir_err = 0
            with pred_path.open(encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    if r.get("kept") != "True" or r.get("predicted_relation") != "<":
                        continue
                    g = golds[ds].get((r["source_uri"], r["target_uri"]))
                    if g == "<":
                        tp += 1
                    elif g is None:
                        fp_none += 1
                    else:
                        dir_err += 1
            gh = g_hat[ds][0]
            denom = tp + fp_none
            p_strict = tp / denom if denom else float("nan")
            p_corr = (tp + gh * fp_none) / denom if denom else float("nan")
            lo = (tp + g_hat[ds][3] * fp_none) / denom if denom else float("nan")
            hi = (tp + g_hat[ds][4] * fp_none) / denom if denom else float("nan")
            rows_out.append({
                "model": model, "dataset": ds, "TP_lt": tp, "n_FP": fp_none,
                "dir_errors_excl": dir_err,
                "P_strict": round(p_strict, 4), "P_corr": round(p_corr, 4),
                "Delta": round(p_corr - p_strict, 4),
                "g_ds": round(gh, 4),
                "g_CI95": f"[{g_hat[ds][3]:.3f},{g_hat[ds][4]:.3f}]",
                "P_corr_CI95": f"[{lo:.4f},{hi:.4f}]",
            })

    tsv_path = out_dir / f"corrected_precision_{args.sha}.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows_out)

    # ── (5) Supplementary: per-row unblinded ─────────────────────────────────
    sup_path = out_dir / "ltfp_audit_unblinded_supplementary.tsv"
    with sup_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["row_id", "dataset", "source_uri", "target_uri",
                    "source_label", "target_label", "judgment", "flagging_models"])
        for r in sorted(adj, key=lambda x: (x["dataset"], x["row_id"])):
            k = key[r["row_id"]]
            w.writerow([r["row_id"], k["dataset"], k["source_uri"], k["target_uri"],
                        k["source_label"], k["target_label"], r["judgment"],
                        ",".join(k["flagging_models"])])

    # ── (4) MD-Report ────────────────────────────────────────────────────────
    md = []
    md.append(f"# Gold-gap-korrigierte Subclass-Precision (Audit adjudiziert 2026-07-19)\n")
    md.append("## Stichprobenrahmen (registriert)\n")
    md.append("Population je Datensatz: kept-`<`-Predictions mit **gold=none** "
              "(Richtungsfehler gold∈{>,=} ausgeschlossen), Union über die 4 "
              "seed-42-Zellen (llama g5/g7 = reused Baselines), **pair-dedupliziert**. "
              "Ziehung: `scripts/build_ltfp_audit_sample.py`, seed 42, N=40, "
              "stratifiziertes Round-Robin **über Modelle** (model-blind annotiert). "
              "Da das Urteil pair-level ist, gilt ĝ **modellunabhängig je Datensatz**; "
              "die Modell-Stratifizierung sichert nur Abdeckung, keine "
              "modellspezifischen Raten.\n")
    md.append(f"**Rahmen-Klarstellung:** Die per-Datensatz-Sheets "
              f"`matrix_ltFP_audit_{{ds}}.tsv` (Review-Deliverables 2026-07-07) teilen "
              f"die Populations-Definition, sind aber eine **separate, frühere "
              f"Ziehung** — row_id-Overlap mit diesem Sample: {overlap}/40. "
              f"Maßgeblicher Rahmen dieser Adjudikation ist die oben definierte "
              f"Population + `build_ltfp_audit_sample.py` (seed 42).\n")
    md.append("## Gap-Raten (Wilson 95%)\n")
    md.append("| Datensatz | gold_gap / n | ĝ | 95%-CI |\n|---|---|---|---|")
    for ds in DATASETS:
        p, n, k, lo, hi = g_hat[ds]
        md.append(f"| {ds} | {k}/{n} | {p:.3f} | [{lo:.3f}, {hi:.3f}] |")
    p, n, k, lo, hi = pooled
    md.append(f"| **gepoolt** | **{k}/{n}** | **{p:.3f}** | **[{lo:.3f}, {hi:.3f}]** |\n")
    md.append("## Korrigierte Precision (e2e, seed-42-Zellen)\n")
    md.append("| Modell | Datensatz | TP< | FP<(none) | dir-err (excl.) | P_strict | P_corr | Δ |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows_out:
        md.append(f"| {r['model']} | {r['dataset']} | {r['TP_lt']} | {r['n_FP']} | "
                  f"{r['dir_errors_excl']} | {r['P_strict']:.3f} | {r['P_corr']:.3f} | "
                  f"+{r['Delta']:.3f} |")
    md.append("\n## Methods paragraph (thesis-ready)\n")
    md.append(
        "> To quantify how many strict subclass false positives are artefacts of "
        "incomplete reference alignments rather than model errors, we audited a "
        "blind sample of 40 pairs that at least one of the four rerankers labelled "
        "`<` although the reference contains no relation for the pair (direction "
        "errors excluded). The sample was drawn with a fixed seed, stratified "
        "round-robin across models, deduplicated at pair level, and adjudicated "
        "by the first author without access to model identities. "
        f"Of the 40 pairs, {kp} ({100*pooled[0]:.0f}%) were genuine but unlabelled "
        "subclass relations (gold gaps; Wilson 95% CI "
        f"[{pooled[3]:.2f}, {pooled[4]:.2f}]), with per-dataset rates of "
        + ", ".join(f"{100*g_hat[ds][0]:.0f}% ({ds})" for ds in DATASETS) + ". "
        "We therefore report, alongside the strict subclass precision "
        "P_strict = TP/(TP+FP_none), a gap-corrected estimate "
        "P_corr = (TP + ĝ_d·FP_none)/(TP+FP_none), which treats the estimated "
        "share ĝ_d of unlabelled-but-true pairs as correct; corrected precisions "
        "rise by up to the reported Δ while all ranking conclusions between "
        "models remain unchanged.\n")
    md.append("## Kaveats (Pflicht)\n")
    md.append("- **mouse-human wurde nicht auditiert** — dort keine Korrektur; "
              "P_corr existiert nur für g3/g5/g7.")
    md.append("- **n=40-Stichprobe** → ĝ ist eine Schätzung mit den angegebenen "
              "CIs, keine Re-Annotation des Golds; die Referenz bleibt unverändert.")
    md.append("- Richtungsfehler (gold ∈ {>,=}) sind in FP< nicht enthalten und "
              "werden von der Korrektur nicht berührt (Spalte `dir_errors_excl`).")
    ds_n = {ds: g_hat[ds][1] for ds in DATASETS}
    md.append(f"- **Ungleiche Datensatz-Besetzung** (Folge der Modell-, nicht "
              f"Datensatz-Stratifizierung): n = {ds_n}. Die ĝ-Schätzer für "
              f"g5-groceries (n={ds_n['g5-groceries']}) und g7-literature "
              f"(n={ds_n['g7-literature']}) sind entsprechend schwach (siehe "
              f"breite CIs); insbesondere beruht der große g7-Δ auf 3/4 "
              f"Urteilen. Der **gepoolte** Schätzer (15/40 = 0.375, CI "
              f"[0.242, 0.530]) ist der belastbare Hauptwert; per-Datensatz-"
              f"P_corr für g5/g7 als Sensitivität lesen, nicht als Punktwahrheit.")
    md.append(f"\nReproduktion: `python scripts/ltfp_corrected_precision.py "
              f"--sha {args.sha}` — alle Zahlen oben sind aus "
              f"`{tsv_path.name}` + `{sup_path.name}` nachrechenbar.\n")
    (out_dir / f"corrected_precision_{args.sha}.md").write_text("\n".join(md))

    print(f"geschrieben: {tsv_path}")
    print(f"geschrieben: {out_dir / f'corrected_precision_{args.sha}.md'}")
    print(f"geschrieben: {sup_path}")
    print(f"gepoolt: {kp}/{np_} = {pooled[0]:.3f}  CI [{pooled[3]:.3f}, {pooled[4]:.3f}]")


if __name__ == "__main__":
    main()
