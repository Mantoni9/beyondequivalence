"""
analyze_superclass_misses.py — error analysis of the superclass ('>') recall
weakness, on EXISTING ablation outputs. No new model run, nothing frozen.

Reads the persisted top-50 ranked lists (predictions.tsv) + the reference, for
the focus config Qwen3 / no-LoRA / path_context + T2, on three contrast datasets
(g3-text weak, g2-diseases perfect, g5-groceries tie-breaker). Tests four
hypotheses for WHY '>' recall (~0.58) trails '<' recall (~0.90).

Methodological discipline: a pattern counts as a finding only if it is
CONSISTENT across all three datasets. Per-dataset numbers are printed so a
pattern that holds on only one dataset is visible as an anecdote, not a cause.
No interpretation is hard-coded — the script prints the supporting numbers; the
verdict (confirmed / refuted / dataset-dependent) is drawn from them afterwards.

H0  random baseline           : 1-(1-1/n_target)^k  per dataset (effect-size frame)
H1  fan-out                   : corr(#gold-children of S, per-source '>' recall@20)
H2  rank distribution of misses: hit@20 / near (21-50) / absent (>50) for '>' gold
H3  what it finds instead     : for missed '>' gold, classify S's '>' top-5 vs gold.
    'foreign' = a target in S's '>' top-5 that has NO gold relation (<, >, =) to S
    in the reference. NOTE: a 'foreign' can be a genuine error OR a correct-but-
    unannotated relation (gold gap) — the --examples inspection exposes which.
H4  path_context direction bias: '>' recall@20 path_context vs turtle (B=T2 held)

Torch-free. Run on DWS where results/ablbi_*_<sha>/ live.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Alignment import Alignment
from evaluation_recall import _normalize_relation
from tracks.zenodo_loader import load_subdataset

FOCUS_DATASETS = ("g3-text", "g2-diseases", "g5-groceries")


def _run_dir(results_root: Path, model: str, lora: str, a: str, b: str, dataset: str, sha: str) -> Path:
    return results_root / f"ablbi_{model}_{lora}_A-{a}_B-{b}_{dataset}_{sha}"


def _ranked_by_source(run_dir: Path, direction: str) -> dict[str, list[str]]:
    """Per-source ranked target list (score desc) for one direction, from predictions.tsv."""
    per_src: dict[str, list[tuple[float, str]]] = defaultdict(list)
    pred = run_dir / "predictions.tsv"
    if not pred.is_file():
        sys.exit(f"ERROR: predictions.tsv missing: {pred}\n"
                 f"The top-50 lists for this run are not on disk — a targeted re-run of "
                 f"this config is needed before the analysis can proceed.")
    with pred.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            s, t, rel, score = line.rstrip("\n").split("\t")
            if rel != direction:
                continue
            per_src[s].append((float(score), t))
    return {s: [t for _, t in sorted(v, key=lambda x: (-x[0], x[1]))] for s, v in per_src.items()}


def _gold_by_source(reference: Alignment) -> dict[str, dict[str, set]]:
    """gold[source] = {'<': {targets}, '>': {targets}, '=': {targets}} (normalized)."""
    g: dict[str, dict[str, set]] = defaultdict(lambda: {"<": set(), ">": set(), "=": set()})
    for cor in reference:
        norm = _normalize_relation(cor.relation)
        if norm in ("<", ">", "="):
            g[cor.source][norm].add(cor.target)
    return g


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def _spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return _pearson(ranks(xs), ranks(ys))


def _n_target(dataset: str) -> int:
    _, tgt_path, _ = load_subdataset(dataset)
    txt = Path(tgt_path).read_text(encoding="utf-8")
    return len(re.findall(r"<owl:Class\b", txt)) or len(set(re.findall(r'rdf:about="([^"]+)"', txt)))


def _load_labels(path: str) -> dict[str, str]:
    """URI -> rdfs:label via rdflib (torch-free). Empty dict on failure."""
    try:
        from rdflib import Graph, RDFS
        g = Graph(); g.parse(path)
        return {str(s): str(o) for s, o in g.subject_objects(RDFS.label)}
    except Exception as e:
        print(f"  (label load failed for {path}: {e})")
        return {}


def _lab(uri: str, labels: dict[str, str]) -> str:
    return labels.get(uri) or uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def main() -> None:
    p = argparse.ArgumentParser(description="Superclass-miss error analysis (4 hypotheses).")
    p.add_argument("--sha", required=True)
    p.add_argument("--results-root", default="results")
    p.add_argument("--model", default="qwen3-embedding-8b")
    p.add_argument("--lora", default="lora-off")
    p.add_argument("--a", default="path_context")
    p.add_argument("--b", default="sub_b_pin")
    p.add_argument("--a-contrast", default="turtle", help="A value for the H4 contrast run.")
    p.add_argument("--datasets", nargs="+", default=list(FOCUS_DATASETS))
    p.add_argument("--examples", type=int, default=4,
                   help="Per dataset, print this many missed-source examples with labels (foreign inspection). 0 to skip.")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    root = Path(args.results_root)
    report: dict = {"config": vars(args), "datasets": {}}

    print("=" * 96)
    print(f"SUPERCLASS-MISS ANALYSIS  focus={args.model}/{args.lora}/A-{args.a}/B-{args.b}  sha={args.sha}")
    print("=" * 96)

    for dataset in args.datasets:
        focus = _run_dir(root, args.model, args.lora, args.a, args.b, dataset, args.sha)
        contrast = _run_dir(root, args.model, args.lora, args.a_contrast, args.b, dataset, args.sha)
        src_path, tgt_path, ref_path = load_subdataset(dataset)
        reference = Alignment(str(ref_path))
        gold = _gold_by_source(reference)
        sup_ranked = _ranked_by_source(focus, ">")          # focus '>' top-50 per source
        n_tgt = _n_target(dataset)

        d: dict = {"n_target": n_tgt}
        print(f"\n{'#'*88}\n## {dataset}   n_target~{n_tgt}\n{'#'*88}")

        # ── H0 random baseline ────────────────────────────────────────────
        b10 = 1 - (1 - 1 / n_tgt) ** 10
        b20 = 1 - (1 - 1 / n_tgt) ** 20
        d["H0_random"] = {"R@10": b10, "R@20": b20}
        print(f"[H0] random baseline: R@10~{b10*100:.2f}%  R@20~{b20*100:.2f}%")

        # Per-source '>' recall@20 + miss ranks.
        sources_with_sup = [s for s in gold if gold[s][">"]]
        n_sup_total = sum(len(gold[s][">"]) for s in sources_with_sup)

        # ── H1 fan-out ────────────────────────────────────────────────────
        nchild, rec_s = [], []
        for s in sources_with_sup:
            top20 = set(sup_ranked.get(s, [])[:20])
            hits = len(gold[s][">"] & top20)
            nchild.append(len(gold[s][">"]))
            rec_s.append(hits / len(gold[s][">"]))
        pear = _pearson([float(x) for x in nchild], rec_s)
        spear = _spearman([float(x) for x in nchild], rec_s)
        lo = [r for n, r in zip(nchild, rec_s) if n <= 2]
        hi = [r for n, r in zip(nchild, rec_s) if n >= 10]
        d["H1_fanout"] = {
            "n_sources": len(sources_with_sup),
            "pearson": pear, "spearman": spear,
            "mean_recall_nchild_le2": (sum(lo) / len(lo)) if lo else None, "n_le2": len(lo),
            "mean_recall_nchild_ge10": (sum(hi) / len(hi)) if hi else None, "n_ge10": len(hi),
            "max_children": max(nchild) if nchild else 0,
        }
        print(f"[H1] fan-out: sources={len(sources_with_sup)} maxChildren={max(nchild) if nchild else 0} "
              f"Pearson(nChildren,recall)={pear:+.3f} Spearman={spear:+.3f}")
        print(f"      mean recall | nChildren<=2: {d['H1_fanout']['mean_recall_nchild_le2']} (n={len(lo)})"
              f" | nChildren>=10: {d['H1_fanout']['mean_recall_nchild_ge10']} (n={len(hi)})")

        # ── H2 rank distribution of misses ───────────────────────────────
        hit20 = near = absent = 0
        for s in sources_with_sup:
            order = sup_ranked.get(s, [])
            pos = {t: i + 1 for i, t in enumerate(order)}
            for t in gold[s][">"]:
                r = pos.get(t)
                if r is None:
                    absent += 1
                elif r <= 20:
                    hit20 += 1
                else:
                    near += 1
        d["H2_miss_ranks"] = {"total_sup_gold": n_sup_total,
                              "hit@20": hit20, "near_21_50": near, "absent_gt50": absent}
        print(f"[H2] miss ranks (of {n_sup_total} '>' gold): hit@20={hit20} "
              f"near(21-50)={near} absent(>50)={absent}  "
              f"-> of misses, near={near/(near+absent)*100:.1f}% absent={absent/(near+absent)*100:.1f}%"
              if (near + absent) else f"[H2] (no misses)")

        # ── H3 what it finds instead ──────────────────────────────────────
        # For sources with >=1 missed '>' gold, classify the focus '>' top-5
        # occupants vs that source's gold.
        # Mutually-exclusive 3-way classification; one count per top-5 slot.
        cls = {"sup_gold_child": 0, "wrong_dir_sub_or_eq": 0, "foreign": 0}
        examined = 0
        for s in sources_with_sup:
            top20 = set(sup_ranked.get(s, [])[:20])
            if gold[s][">"] <= top20:
                continue  # no miss for this source
            examined += 1
            for t in sup_ranked.get(s, [])[:5]:
                if t in gold[s][">"]:
                    cls["sup_gold_child"] += 1
                elif t in gold[s]["<"] or t in gold[s]["="]:
                    cls["wrong_dir_sub_or_eq"] += 1
                else:
                    cls["foreign"] += 1
        slots = sum(cls.values())
        den = max(1, slots)
        d["H3_what_instead"] = {"sources_with_miss": examined, "top5_slots": slots,
                                "counts": cls, "pct": {k: v / den for k, v in cls.items()}}
        print(f"[H3] sources with >=1 '>' miss={examined}; top-5 slots={slots} "
              f"(foreign = no gold relation to S): "
              f"sup_gold_child={cls['sup_gold_child']} ({cls['sup_gold_child']/den*100:.0f}%) "
              f"wrong_dir(<,=)={cls['wrong_dir_sub_or_eq']} ({cls['wrong_dir_sub_or_eq']/den*100:.0f}%) "
              f"foreign={cls['foreign']} ({cls['foreign']/den*100:.0f}%)")

        # ── H4 path_context vs turtle on '>' ──────────────────────────────
        sup_ranked_c = _ranked_by_source(contrast, ">")
        worse = better = equal = 0
        rec_focus = rec_contrast = 0
        for s in sources_with_sup:
            n = len(gold[s][">"])
            hf = len(gold[s][">"] & set(sup_ranked.get(s, [])[:20]))
            hc = len(gold[s][">"] & set(sup_ranked_c.get(s, [])[:20]))
            rec_focus += hf
            rec_contrast += hc
            if hf < hc:
                worse += 1
            elif hf > hc:
                better += 1
            else:
                equal += 1
        d["H4_pathctx_vs_turtle"] = {
            "recall20_path_context": rec_focus / n_sup_total if n_sup_total else 0,
            "recall20_turtle": rec_contrast / n_sup_total if n_sup_total else 0,
            "sources_pathctx_worse": worse, "better": better, "equal": equal,
        }
        print(f"[H4] '>' R@20  path_context={rec_focus/n_sup_total:.4f}  turtle={rec_contrast/n_sup_total:.4f}"
              f"  (per-source: pc worse={worse} better={better} equal={equal})")

        # ── Foreign inspection: concrete missed-source examples with labels ──
        if args.examples > 0:
            src_labels = _load_labels(str(src_path))
            tgt_labels = _load_labels(str(tgt_path))
            print(f"   --- examples (foreign = NO gold relation to S; check: real error vs gold gap) ---")
            shown = 0
            for s in sources_with_sup:
                order = sup_ranked.get(s, [])
                pos = {t: i + 1 for i, t in enumerate(order)}
                missed = gold[s][">"] - set(order[:20])
                if not missed:
                    continue
                shown += 1
                md = []
                for t in list(missed)[:3]:
                    r = pos.get(t)
                    md.append(f"{_lab(t, tgt_labels)} [{'rank '+str(r) if r else '>50/absent'}]")
                top5 = []
                for t in order[:5]:
                    cat = ("sup_gold" if t in gold[s][">"]
                           else "wrong_dir" if (t in gold[s]["<"] or t in gold[s]["="])
                           else "foreign")
                    top5.append(f"{_lab(t, tgt_labels)}[{cat}]")
                print(f"   • S={_lab(s, src_labels)}  (children={len(gold[s]['>'])}, missed={len(missed)})")
                print(f"       missed: {'; '.join(md)}")
                print(f"       '>' top-5: {' | '.join(top5)}")
                if shown >= args.examples:
                    break

        report["datasets"][dataset] = d

    out = Path(args.out_json) if args.out_json else root / f"superclass_analysis_{args.sha}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nJSON written: {out}")
    print("=" * 96)
    print("Consistency rule: a pattern is a finding only if it holds across ALL three datasets.")
    print("=" * 96)


if __name__ == "__main__":
    main()
