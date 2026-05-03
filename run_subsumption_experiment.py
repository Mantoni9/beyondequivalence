"""
run_subsumption_experiment.py — BeyondEquivalence Stage 1 retrieval runner.

One W&B run per (model, instruction-variant, dataset) configuration. Computes
strict / lax / per_relation_strict Recall@K and MRR via evaluation_recall, plus
score-distribution diagnostics that surface whether broader and narrower
asymmetric runs have systematically different score scales (the methodological
explanation for any divergence between strict and per_relation_strict).

Usage (interactive on a compute node, the primary path during development):

    conda activate melt-olala
    python run_subsumption_experiment.py \\
        --model qwen3-embedding-8b \\
        --instruction-variant asymmetric \\
        --dataset g7-literature \\
        --wandb

Local MacBook smoke (downloads sbert once, runs in seconds):

    python run_subsumption_experiment.py \\
        --model sbert \\
        --instruction-variant symmetric \\
        --dataset g7-literature \\
        --smoke-test

W&B project routing:
  - regular runs: beyondequivalence-retrieval-stage1
  - --smoke-test: beyondequivalence-smoke (with tag mode:smoke)
  - --wandb-project: explicit override
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from dotenv import load_dotenv
load_dotenv()


MODEL_ALIASES: dict[str, str] = {
    "sbert":                   "sentence-transformers/all-MiniLM-L6-v2",
    "qwen3-embedding-8b":      "Qwen/Qwen3-Embedding-8B",
    "llama-embed-nemotron-8b": "nvidia/llama-embed-nemotron-8b",
    "e5-mistral":              "intfloat/e5-mistral-7b-instruct",
}

DEFAULT_K_VALUES: tuple[int, ...] = (1, 5, 10, 20)


# ─── small helpers ────────────────────────────────────────────────────────────

def _git_sha_and_dirty() -> tuple[str, bool, str]:
    """Return (short SHA, is_dirty, dirty_paths).

    `dirty_paths` is the verbatim ``git status --porcelain`` output — empty
    when the tree is clean. We surface it in the run log so a dirty stamp
    triggered by a stray output file can be diagnosed directly from the
    stdout.log of the affected run, instead of requiring a manual
    ``git status`` after the fact.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True,
        ).rstrip()
        return sha, bool(porcelain), porcelain
    except Exception:
        return "unknown", False, ""


def _resolve_model(arg: str) -> str:
    return MODEL_ALIASES.get(arg, arg)


def _alias_for_naming(arg: str) -> str:
    if arg in MODEL_ALIASES:
        return arg
    safe = arg.split("/")[-1].replace(":", "-").replace("\\", "-").replace(" ", "_")
    return safe or "model"


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _score_stats(scores: list[float]) -> dict[str, float | int]:
    if not scores:
        return {"count": 0, "mean": 0.0, "median": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count":  len(scores),
        "mean":   float(statistics.fmean(scores)),
        "median": float(statistics.median(scores)),
        "stddev": float(statistics.pstdev(scores)) if len(scores) >= 2 else 0.0,
        "min":    float(min(scores)),
        "max":    float(max(scores)),
    }


def _score_diagnostics(alignment, instruction_variant: str) -> dict:
    """Side-metric: do broader/narrower runs have systematically different score scales?

    Why this matters:
      - In the asymmetric setting, a large gap between broader_run.mean and
        narrower_run.mean (or large stddev difference) means the joint top-K
        ranking is dominated by one direction; that is the methodological
        explanation for any divergence between strict and per_relation_strict
        recall metrics in the same run.

    Sanity heuristic — apply at run-review time, not enforced in code:
      - On a competent instruction-aware embedding model (Qwen3-Embedding-8B,
        llama-embed-nemotron-8b, e5-mistral-7b-instruct), expect stddev >= 0.05 over a
        retrieval-style top-K population. Stddev < 0.05 indicates a near-flat
        score distribution and is a strong red flag for a broken encoding path
        (wrong pooling, wrong instruction format, single-token-degenerate
        embedding). On the 2026-04-26 sbert smoke runs we saw stddev ~0.02
        for symmetric and ~0.03 per-run for asymmetric — that is expected for a
        22M-parameter non-instruction-trained model and is NOT comparable to a
        Qwen3 / Llama-Embed-Nemotron run on the same data.

    Interpretation per model class (essential for the thesis argument):
      - SBERT (non-instruction-trained, e.g. all-MiniLM-L6-v2): in BOTH modes the
        broader_run / narrower_run distributions look near-identical (e.g. on
        2026-04-26 g7-literature smoke: broader mean=0.650, narrower mean=0.626,
        delta=0.024, stddevs ~0.028 each). This is EXPECTED, not a bug. SBERT
        cannot interpret the instruction as a steering signal — from its
        perspective the instruction is just "text-before-the-description". The
        SBERT numbers are scientifically valuable as an INSTRUCTION-NAIVE
        BASELINE, but they are NOT informative about the effect of asymmetric
        instructions on retrieval. The asymmetric-instruction effect can only
        manifest on instruction-aware models.
      - Instruction-aware models (Qwen3-Embedding-8B,
        llama-embed-nemotron-8b, e5-mistral-7b-instruct): if the broader/narrower
        distributions also look near-
        identical, that is SURPRISING and demands a written explanation:
          (a) the model is silently ignoring the instructions (e.g. wrong
              prompt format for that model class — verify against the model
              card),
          (b) the instructions themselves are too weak / underspecified to
              shift the embedding (iterate on SUBSUMPTION_INSTRUCTIONS),
          (c) the subsumption research question is genuinely not influenceable
              by instruction-conditioning at the bi-encoder level — a result
              worth reporting on its own.
      The strict-vs-per_relation_strict gap on the same run further disambiguates:
      a large gap with similar broader/narrower distributions points to (b);
      no gap and similar distributions points to (a) or (c).
    """
    by_rel: dict[str, list[float]] = {}
    for cor in alignment:
        by_rel.setdefault(cor.relation, []).append(float(cor.confidence))
    if instruction_variant == "symmetric":
        return {"all": _score_stats([s for v in by_rel.values() for s in v])}
    return {
        "broader_run":  _score_stats(by_rel.get("<", [])),
        "narrower_run": _score_stats(by_rel.get(">", [])),
    }


def _safe_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v).replace("\t", " ").replace("\n", " ")


def _write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(_safe_cell(c) for c in row) + "\n")


# ─── KG / labels ──────────────────────────────────────────────────────────────

def _load_kg_with_labels(rdf_path: Path):
    from RDFGraphWrapper import RDFGraphWrapper
    kg = RDFGraphWrapper(str(rdf_path))
    labels: dict[str, str] = {}
    for cls in kg.get_classes():
        ls = kg.get_labels(cls)
        labels[str(cls)] = next(iter(ls)) if ls else ""
    return kg, labels


def _filter_kg_to_smoke(kg, n: int = 3) -> list[str]:
    """Restrict kg.get_classes() to the first N URIs alphabetically. Returns the kept URIs (as str)."""
    classes_sorted = sorted(kg.get_classes(), key=str)
    keep_uris = classes_sorted[:n]
    keep_set = set(keep_uris)
    kg.get_classes = lambda: keep_set      # noqa: ARG005 — intentional monkey-patch
    return [str(c) for c in keep_uris]


# ─── smoke probe ──────────────────────────────────────────────────────────────

def _smoke_probe(
    logger: logging.Logger,
    matcher,
    kg_source,
    smoke_source_uris: list[str],
    predictions,
    reference,
    source_labels: dict[str, str],
    target_labels: dict[str, str],
    description: str,
    kg_format: str,
) -> None:
    """Per-source verbose probe + reference HIT/MISS check + embedding sanity."""
    from RDFGraphWrapper import RDFGraphWrapper
    from prompt import build_instruct_query_prompt
    from evaluation_recall import _normalize_relation

    logger.info("=" * 72)
    logger.info("SMOKE TEST: per-source detail for %d source class(es)", len(smoke_source_uris))
    logger.info("=" * 72)

    # Per-source prediction ranking.
    per_src_preds: dict[str, list] = {}
    for cor in predictions:
        per_src_preds.setdefault(cor.source, []).append(cor)
    for lst in per_src_preds.values():
        lst.sort(key=lambda c: (-c.confidence, c.target, c.relation))

    # Reference indexed by source.
    ref_by_src: dict[str, list] = {}
    for cor in reference:
        ref_by_src.setdefault(cor.source, []).append(cor)

    # Embedding-vector sanity probe — re-encode the 3 smoke sources via the
    # already-loaded matcher embedder, dump norm + first 5 dims.
    description_method = getattr(kg_source, description)
    smoke_texts = [
        RDFGraphWrapper.serialize(description_method(uri_obj_from_str(kg_source, src_uri)),
                                   format=kg_format)
        for src_uri in smoke_source_uris
    ]
    # Use any query instruction the matcher carries; for asymmetric, broader is fine.
    instr = getattr(matcher, "query_instruction", None) \
            or getattr(matcher, "broader_query_instruction", "")
    prompt = build_instruct_query_prompt(instr) or None
    embs = matcher._embedder.encode(
        smoke_texts, prompt=prompt, convert_to_tensor=True, show_progress_bar=False,
    )

    for i, src_uri in enumerate(smoke_source_uris):
        label = source_labels.get(src_uri, "")
        logger.info("")
        logger.info("--- Source: %s   (label='%s')", src_uri, label)
        # Description snippet (first 200 chars of the Turtle serialisation).
        snippet = smoke_texts[i].replace("\n", " ")[:200]
        logger.info("    description[:200]: %s", snippet)
        # Embedding sanity.
        emb = embs[i].detach().float().cpu()
        norm = float(emb.norm().item())
        first5 = [float(x) for x in emb[:5].tolist()]
        logger.info("    embedding: norm=%.6f  first5=%s", norm, [f"{x:+.4f}" for x in first5])

        # Top-5 predictions.
        preds = per_src_preds.get(src_uri, [])
        logger.info("    top-5 predictions:")
        if not preds:
            logger.warning("      (none)")
        for rank, cor in enumerate(preds[:5], start=1):
            tgt_lbl = target_labels.get(cor.target, "")
            logger.info("      %2d. %s  rel=%s  score=%.4f  label='%s'",
                        rank, cor.target, cor.relation, cor.confidence, tgt_lbl)

        # Reference probe.
        refs = ref_by_src.get(src_uri, [])
        logger.info("    reference mappings for this source: %d", len(refs))
        for ref in refs:
            rank_lax = None
            rank_strict = None
            ref_norm = _normalize_relation(ref.relation)
            for r, cor in enumerate(preds, start=1):
                if cor.target == ref.target:
                    if rank_lax is None:
                        rank_lax = r
                    if ref_norm is not None and cor.relation == ref_norm and rank_strict is None:
                        rank_strict = r
            tgt_lbl = target_labels.get(ref.target, "")
            verdict_lax    = f"HIT@{rank_lax}"    if rank_lax    is not None else "MISS"
            verdict_strict = f"HIT@{rank_strict}" if rank_strict is not None else "MISS"
            logger.info("      REF -> %s  rel=%s (norm=%s)  label='%s' :: lax=%s strict=%s",
                        ref.target, ref.relation, ref_norm, tgt_lbl, verdict_lax, verdict_strict)

    logger.info("=" * 72)


def uri_obj_from_str(kg, uri_str: str):
    """Find the URIRef in kg.get_classes() that string-matches uri_str."""
    for c in sorted(kg.get_classes(), key=str):
        if str(c) == uri_str:
            return c
    raise KeyError(f"URI {uri_str} not in kg.get_classes()")


# ─── Sub-B sweep helpers ──────────────────────────────────────────────────────

# Fields a finished metrics.json must carry. Resume-check considers a run
# unfinished if any of these keys is missing or the JSON itself is corrupt.
_REQUIRED_METRICS_FIELDS = (
    "recall_at_k", "mrr",
    "n_reference_total", "n_reference_after_filter",
    "score_diagnostics", "matcher_runtime_seconds",
    "matcher_last_run_metrics",
)


def _resume_complete(output_dir: Path, expected_group: str | None = None) -> bool:
    """Return True iff <output_dir>/metrics.json exists, is valid JSON, and
    carries every required field. When `expected_group` is given, also require
    that <output_dir>/config.json's wandb_group matches — otherwise an old
    Smoke / earlier-sweep run with the same run_name (same SHA, same model,
    same description, same dataset, same template) would falsely satisfy the
    resume check.
    """
    metrics_p = output_dir / "metrics.json"
    if not metrics_p.is_file():
        return False
    try:
        m = json.loads(metrics_p.read_text())
    except Exception:
        return False
    if not all(field in m for field in _REQUIRED_METRICS_FIELDS):
        return False
    if expected_group is not None:
        cfg_p = output_dir / "config.json"
        if not cfg_p.is_file():
            return False
        try:
            cfg = json.loads(cfg_p.read_text())
        except Exception:
            return False
        if cfg.get("wandb_group") != expected_group:
            return False
    return True


def _find_completed_run(
    *,
    alias_short: str,
    variant_short: str,
    description: str,
    template_id_str: str,
    dataset: str,
    expected_group: str | None,
) -> Path | None:
    """Find any results/subB_<...>/ directory matching the run-name stem
    (model, variant, description, template, dataset) regardless of git SHA
    suffix, whose metrics.json + config.json satisfy the group-aware resume
    check. Returns the matching Path, or None.

    Why: output_dir embeds the current short SHA. After a fix-up commit
    (e.g. group-aware resume), a re-launch builds output_dir with a new SHA
    and would re-run every previously completed iteration. This lookup
    matches by content (group), not by output-dir name, so already-correct
    runs from the same logical group are reused regardless of the SHA they
    were originally written under.
    """
    import glob as _glob_mod
    pattern = f"results/subB_{alias_short}_{variant_short}_{description}_{template_id_str}_{dataset}_*"
    for d in sorted(_glob_mod.glob(pattern)):
        p = Path(d)
        if _resume_complete(p, expected_group=expected_group):
            return p
    return None


def _attach_iteration_log(output_dir: Path) -> logging.FileHandler:
    """Add an output-dir-scoped FileHandler. Caller must remove it after
    the iteration so the next run gets a fresh per-output-dir log file.
    """
    fh = logging.FileHandler(output_dir / "stdout.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s]: %(message)s"))
    logging.getLogger().addHandler(fh)
    return fh


def _detach_iteration_log(handler: logging.FileHandler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()


def _iter_subB_specs(
    models: list[str],
    variants: list[str],
    datasets: list[str],
    descriptions: tuple[str, ...],
    sym_template_ids: tuple[str, ...],
    asym_template_ids: tuple[str, ...],
):
    """Generate Sub-B run specs in deterministic order: model → dataset →
    description → template. sbert is forced to symmetric only and gets a
    single template_id=None (the model is not instruction-aware).

    Yields dicts, one per run. The order here also determines KG / matcher
    cache reuse: the inner-most axis is template, so within one
    (model, dataset, description) block 5 templates run back-to-back without
    re-serialising the KG.
    """
    for model in models:
        for dataset in datasets:
            for description in descriptions:
                for variant in variants:
                    if model == "sbert" and variant != "symmetric":
                        # sbert is sym-only by construction; document elsewhere.
                        continue
                    if model == "sbert":
                        ids: tuple = (None,)
                    elif variant == "symmetric":
                        ids = sym_template_ids
                    else:
                        ids = asym_template_ids
                    for template_id in ids:
                        yield {
                            "model": model,
                            "variant": variant,
                            "dataset": dataset,
                            "description": description,
                            "template_id": template_id,
                        }


def _persist_artefacts(
    output_dir: Path,
    config_dump: dict,
    matcher,
    predictions,
    report,
    reference,
    source_labels: dict[str, str],
    target_labels: dict[str, str],
    score_diag: dict,
    t_elapsed: float,
) -> dict:
    """Write config.json, metrics.json, predictions.tsv, reference.tsv,
    per_source_long.tsv to <output_dir>. Returns the metrics dict for
    optional W&B re-use.
    """
    (output_dir / "config.json").write_text(
        json.dumps(config_dump, indent=2, ensure_ascii=False)
    )

    metrics = {
        "recall_at_k": report.recall_at_k,
        "mrr": report.mrr,
        "n_reference_total": report.n_reference_total,
        "n_reference_after_filter": report.n_reference_after_filter,
        "dropped_relations_count": report.dropped_relations_count,
        "dropped_relations_breakdown": report.dropped_relations_breakdown,
        "score_diagnostics": score_diag,
        "matcher_runtime_seconds": t_elapsed,
        "matcher_last_run_metrics": matcher.last_run_metrics,
        "distance_breakdown": None,
        "distance_breakdown_reason": (
            "Cross-ontology distance not well-defined on STROMA/TaSeR sub-datasets; "
            "deferred — see docs/relations_per_dataset.md."
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False)
    )

    pred_rows = sorted(
        ([cor.source, source_labels.get(cor.source, ""), cor.target,
          target_labels.get(cor.target, ""), cor.relation, float(cor.confidence)]
         for cor in predictions),
        key=lambda r: (r[0], -r[5], r[2], r[4]),
    )
    _write_tsv(
        output_dir / "predictions.tsv",
        ["source_uri", "source_label", "target_uri", "target_label",
         "predicted_relation", "score"],
        pred_rows,
    )

    from evaluation_recall import _normalize_relation
    ref_rows = sorted(
        ([cor.source, cor.target, cor.relation, _normalize_relation(cor.relation) or ""]
         for cor in reference),
        key=lambda r: (r[0], r[1], r[2]),
    )
    _write_tsv(
        output_dir / "reference.tsv",
        ["source_uri", "target_uri", "relation_raw", "relation_normalized"],
        ref_rows,
    )

    if report.per_source_rows:
        cols = list(report.per_source_rows[0].keys())
        _write_tsv(
            output_dir / "per_source_long.tsv",
            cols,
            [[row[c] for c in cols] for row in report.per_source_rows],
        )

    return metrics


def _log_run_to_wandb(wandb_run, wandb_module, report, score_diag, matcher, t_elapsed) -> None:
    flat = report.to_wandb_metrics()
    flat["runtime/matcher_seconds"] = t_elapsed
    for k, v in matcher.last_run_metrics.items():
        if isinstance(v, (int, float)) or v is None:
            if v is not None:
                flat[f"encoding/{k}"] = v
        elif isinstance(v, dict) and k == "per_run":
            for run_label, sub in v.items():
                for kk, vv in sub.items():
                    if isinstance(vv, (int, float)):
                        flat[f"encoding/{run_label}/{kk}"] = vv
    for run_label, stats in score_diag.items():
        for kk, vv in stats.items():
            flat[f"score_diagnostics/{run_label}/{kk}"] = vv
    wandb_module.log(flat)
    if report.per_source_rows:
        cols = list(report.per_source_rows[0].keys())
        data = [[row[c] for c in cols] for row in report.per_source_rows]
        wandb_run.log({"per_source_top_k": wandb_module.Table(columns=cols, data=data)})
    wandb_run.finish()


# ─── argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BeyondEquivalence Stage 1 retrieval experiment.")
    # Single-run path (kept fully backwards-compatible).
    p.add_argument("--model", default=None,
                   help="Model alias or HF id / local path. Aliases: " + ", ".join(MODEL_ALIASES))
    p.add_argument("--instruction-variant", default=None,
                   choices=("symmetric", "asymmetric"))
    p.add_argument("--wandb-group", default=None,
                   help="W&B run group (used to bundle a multi-dataset sweep).")
    # Sub-B sweep path (description × template ablation, embedder warm).
    p.add_argument("--sub-b-sweep", action="store_true",
                   help="Run the Sub-B description/template ablation sweep — outer loop "
                        "model → dataset → description → template, embedder is loaded once "
                        "per model and reused. Ignores --model / --instruction-variant / "
                        "--dataset; reads --sub-b-models / --sub-b-variants / --sub-b-datasets "
                        "instead.")
    p.add_argument("--sub-b-models", nargs="+", default=None,
                   help="Models to sweep in --sub-b-sweep mode (e.g. sbert qwen3-embedding-8b).")
    p.add_argument("--sub-b-variants", nargs="+", default=None,
                   choices=("symmetric", "asymmetric"),
                   help="Variants to sweep in --sub-b-sweep mode. sbert is silently restricted "
                        "to symmetric.")
    p.add_argument("--sub-b-datasets", nargs="+", default=None,
                   help="Datasets to sweep in --sub-b-sweep mode.")
    p.add_argument("--symmetric-instruction-id", default="sym_v1",
                   help="SUBSUMPTION_INSTRUCTIONS id used on both sides in --instruction-variant=symmetric.")
    p.add_argument("--broader-instruction-id", default="asym_broader_v1",
                   help="Instruction for the broader run's query side (asymmetric only).")
    p.add_argument("--narrower-instruction-id", default="asym_narrower_v1",
                   help="Instruction for the narrower run's query side (asymmetric only).")
    p.add_argument("--document-instruction-id", default=None,
                   help=("Document-side instruction id. If unset (default), it is resolved "
                         "per variant: symmetric -> --symmetric-instruction-id (same "
                         "instruction on both sides, the actual symmetric setup); "
                         "asymmetric -> 'none' (empty document instruction, per the "
                         "Stage-1 spec). Pass an explicit id (e.g. 'sym_v1' or 'none') "
                         "to override."))
    p.add_argument("--dataset", default="g7-literature",
                   help="BeyondEquivalence sub-dataset name (see tracks.zenodo_loader).")
    p.add_argument("--description", default="description_one_gen",
                   help="RDFGraphWrapper description method.")
    p.add_argument("--top-k-max", type=int, default=20,
                   help="Top-K to retrieve per source. Recall@K is reported for K in {1,5,10,20} ∩ ≤ this.")
    p.add_argument("--kg-format", default="turtle")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--output-dir", default=None,
                   help="Override output dir (default: results/subsumption[_smoke]_<ts>_<run-name>/).")
    p.add_argument("--smoke-test", action="store_true",
                   help="Restrict to first 3 source classes alphabetically; verbose per-source logs.")
    return p.parse_args()


# ─── Sub-B sweep main ─────────────────────────────────────────────────────────

def main_subB_sweep(args: argparse.Namespace) -> None:
    """Outer-loop sweep: model → dataset → description → template.

    Embedder is loaded once per (model, variant) pair and held warm across
    all 5×5×N inner iterations. KG is loaded once per dataset and reused
    across all (description × template) inner iterations.
    """
    if not args.sub_b_models or not args.sub_b_variants or not args.sub_b_datasets:
        sys.exit("--sub-b-sweep requires --sub-b-models, --sub-b-variants, --sub-b-datasets.")

    sha, dirty, dirty_paths = _git_sha_and_dirty()
    sweep_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not args.wandb_group:
        args.wandb_group = f"subB_descablation_{sweep_ts}_{sha}"

    # Sweep-level logging — file mirror in results/<group>.log so a long
    # sweep is grep-able from a single file.
    Path("results").mkdir(parents=True, exist_ok=True)
    sweep_log_path = Path("results") / f"{args.wandb_group}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s]: %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); root.addHandler(sh)
    sweep_fh = logging.FileHandler(sweep_log_path, encoding="utf-8")
    sweep_fh.setFormatter(fmt); root.addHandler(sweep_fh)
    logger = logging.getLogger("run_subsumption.subB")

    logger.info("Sub-B sweep starting. Group=%s SHA=%s dirty=%s", args.wandb_group, sha, dirty)
    if dirty:
        logger.warning("Working tree is DIRTY:\n%s", dirty_paths)
    logger.info("Models: %s", args.sub_b_models)
    logger.info("Variants: %s", args.sub_b_variants)
    logger.info("Datasets: %s", args.sub_b_datasets)

    _set_seeds(args.seed)
    device = _detect_device()
    logger.info("Device: %s", device)

    from prompt import (
        SUBB_DESCRIPTION_METHODS, SUBB_SYM_TEMPLATE_IDS, SUBB_ASYM_TEMPLATE_IDS,
        get_subb_sym_template, get_subb_asym_templates,
    )
    from Alignment import Alignment
    from tracks.zenodo_loader import load_subdataset
    from evaluation_recall import compute_recall_at_k
    from MatcherEmbeddingRetrieval import MatcherEmbeddingRetrieval
    from MatcherAsymmetricRetrieval import MatcherAsymmetricRetrieval

    specs = list(_iter_subB_specs(
        args.sub_b_models, args.sub_b_variants, args.sub_b_datasets,
        SUBB_DESCRIPTION_METHODS, SUBB_SYM_TEMPLATE_IDS, SUBB_ASYM_TEMPLATE_IDS,
    ))
    logger.info("Total run specs: %d", len(specs))

    matcher_cache: dict[tuple[str, str], object] = {}
    kg_cache: dict[str, tuple] = {}

    wandb = None
    if args.wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            logger.error("wandb not installed; install with: pip install wandb")
            sys.exit(1)

    project = args.wandb_project or "beyondequivalence-retrieval-stage1"
    n_done = n_skipped = n_failed = 0
    sweep_t0 = time.perf_counter()

    for i, spec in enumerate(specs, start=1):
        model = spec["model"]
        variant = spec["variant"]
        dataset = spec["dataset"]
        description = spec["description"]
        template_id = spec["template_id"]
        alias_short = _alias_for_naming(model)
        template_id_str = template_id if template_id is not None else "noinstr"

        run_name = (
            f"subB_{alias_short}_{variant[:3]}_{description}_{template_id_str}_{dataset}_{sha}"
            + ("_dirty" if dirty else "")
        )
        output_dir = Path("results") / run_name

        # Resume check BEFORE mkdir, so a hit doesn't leave an empty dir
        # behind. Lookup is SHA-tolerant: any older Sub-B run-dir with the
        # same (model, variant, description, template, dataset) stem and a
        # matching wandb_group counts as completed.
        existing = _find_completed_run(
            alias_short=alias_short, variant_short=variant[:3],
            description=description, template_id_str=template_id_str,
            dataset=dataset, expected_group=args.wandb_group,
        )
        if existing is not None:
            logger.info("[%d/%d] SKIP (resume): %s -> %s",
                        i, len(specs), run_name, existing.name)
            n_skipped += 1
            continue
        output_dir.mkdir(parents=True, exist_ok=True)

        iter_handler = _attach_iteration_log(output_dir)
        try:
            logger.info("[%d/%d] RUN: %s", i, len(specs), run_name)
            _set_seeds(args.seed)

            # Resolve instructions for this iteration.
            doc_instr = ""  # Document-side stays empty for both sym (matched
                            # by query side below) and asym (per Sub-B spec).
            if variant == "symmetric":
                if template_id is None:
                    sym_instr = ""
                else:
                    sym_instr = get_subb_sym_template(template_id)
                broader_instr = narrower_instr = ""
                doc_instr = sym_instr  # sym = same instruction on both sides
            else:
                sym_instr = ""
                broader_instr, narrower_instr = get_subb_asym_templates(template_id)
                doc_instr = ""  # explicit: empty document side for all asym templates

            # KG cache.
            if dataset not in kg_cache:
                src_path, tgt_path, ref_path = load_subdataset(dataset)
                kg_source, source_labels = _load_kg_with_labels(src_path)
                kg_target, target_labels = _load_kg_with_labels(tgt_path)
                reference = Alignment(str(ref_path))
                kg_cache[dataset] = (kg_source, kg_target, reference, source_labels, target_labels,
                                     str(src_path), str(tgt_path), str(ref_path))
                logger.info("Loaded KG '%s' (src=%d tgt=%d refs=%d)", dataset,
                            len(kg_source.get_classes()), len(kg_target.get_classes()), len(reference))
            kg_source, kg_target, reference, source_labels, target_labels, src_path_s, tgt_path_s, ref_path_s = kg_cache[dataset]

            # Matcher cache (one per (model, variant); embedder lazy-loaded on first match()).
            resolved_model = _resolve_model(model)
            cache_key = (model, variant)
            if cache_key not in matcher_cache:
                if variant == "symmetric":
                    matcher_cache[cache_key] = MatcherEmbeddingRetrieval(
                        model=resolved_model,
                        description=description,
                        query_instruction=sym_instr,
                        document_instruction=doc_instr,
                        output_relation="=",
                        top_k=args.top_k_max,
                        kg_format=args.kg_format,
                    )
                else:
                    matcher_cache[cache_key] = MatcherAsymmetricRetrieval(
                        model=resolved_model,
                        description=description,
                        broader_query_instruction=broader_instr,
                        narrower_query_instruction=narrower_instr,
                        document_instruction=doc_instr,
                        top_k=args.top_k_max,
                        kg_format=args.kg_format,
                    )
            matcher = matcher_cache[cache_key]
            # Hot-update per-iteration attributes.
            matcher.description = description
            if variant == "symmetric":
                matcher.query_instruction = sym_instr
                matcher.document_instruction = doc_instr
            else:
                matcher.broader_query_instruction = broader_instr
                matcher.narrower_query_instruction = narrower_instr
                matcher.document_instruction = doc_instr

            # W&B init for this iteration.
            wandb_run = None
            if args.wandb:
                tags = [
                    "phase:sub-B", "axis:description", "axis:template",
                    f"model:{alias_short}", f"variant:{variant}",
                    f"dataset:{dataset}", f"description:{description}",
                    f"template:{template_id_str}",
                ]
                wandb_config = {
                    "git_sha": sha, "git_dirty": dirty,
                    "model_arg": model, "model_resolved": resolved_model,
                    "instruction_variant": variant,
                    "description": description,
                    "template_id": template_id,
                    "symmetric_instruction_text":  sym_instr      if variant == "symmetric"  else None,
                    "broader_instruction_text":    broader_instr  if variant == "asymmetric" else None,
                    "narrower_instruction_text":   narrower_instr if variant == "asymmetric" else None,
                    "document_instruction_text":   doc_instr,
                    "dataset": dataset,
                    "top_k_max": args.top_k_max,
                    "kg_format": args.kg_format,
                    "seed": args.seed,
                    "device": device,
                    "cluster": os.getenv("CLUSTER", ""),
                }
                wandb_run = wandb.init(
                    project=project, name=run_name, config=wandb_config, tags=tags,
                    group=args.wandb_group, reinit=True,
                )
                logger.info("W&B run: %s", wandb_run.url)

            # Run matcher.
            t_start = time.perf_counter()
            predictions = matcher.match(kg_source, kg_target, Alignment(), parameters={})
            t_elapsed = time.perf_counter() - t_start
            logger.info("Matcher run: %.2fs, alignment size=%d", t_elapsed, len(predictions))

            score_diag = _score_diagnostics(predictions, variant)

            k_values = tuple(k for k in DEFAULT_K_VALUES if k <= args.top_k_max) or (args.top_k_max,)
            report = compute_recall_at_k(
                reference, predictions, k_values=k_values,
                source_labels=source_labels, target_labels=target_labels,
            )
            for mode in ("strict", "lax", "per_relation_strict"):
                for label, by_k in report.recall_at_k[mode].items():
                    for k, v in by_k.items():
                        logger.info("recall_at_k_%s/%s/k=%d = %.4f", mode, label, k, v)
                for label, v in report.mrr[mode].items():
                    logger.info("mrr_%s/%s = %.4f", mode, label, v)

            # Persist.
            config_dump = {
                "git_sha": sha, "git_dirty": dirty,
                "model": model, "model_arg": model, "model_resolved": resolved_model,
                "instruction_variant": variant,
                "dataset": dataset,
                "description": description,
                "template_id": template_id,
                "symmetric_instruction_text":  sym_instr      if variant == "symmetric"  else None,
                "broader_instruction_text":    broader_instr  if variant == "asymmetric" else None,
                "narrower_instruction_text":   narrower_instr if variant == "asymmetric" else None,
                "document_instruction_text":   doc_instr,
                "wandb_group": args.wandb_group,
                "top_k_max": args.top_k_max,
                "kg_format": args.kg_format,
                "seed": args.seed,
                "device": device,
                "run_name": run_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "src_path": src_path_s, "tgt_path": tgt_path_s, "ref_path": ref_path_s,
            }
            _persist_artefacts(
                output_dir, config_dump, matcher, predictions, report, reference,
                source_labels, target_labels, score_diag, t_elapsed,
            )
            if wandb_run is not None:
                _log_run_to_wandb(wandb_run, wandb, report, score_diag, matcher, t_elapsed)

            n_done += 1
        except KeyboardInterrupt:
            logger.error("KeyboardInterrupt — stopping sweep.")
            raise
        except Exception:
            logger.exception("[%d/%d] FAIL: %s", i, len(specs), run_name)
            n_failed += 1
        finally:
            _detach_iteration_log(iter_handler)

    sweep_elapsed = time.perf_counter() - sweep_t0
    logger.info(
        "Sub-B sweep finished in %.1fs. done=%d skipped=%d failed=%d (total=%d)",
        sweep_elapsed, n_done, n_skipped, n_failed, len(specs),
    )


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.sub_b_sweep:
        main_subB_sweep(args)
        return
    if not args.model or not args.instruction_variant:
        sys.exit("Single-run mode requires --model and --instruction-variant. "
                 "Use --sub-b-sweep for the description/template ablation.")
    sha, dirty, dirty_paths = _git_sha_and_dirty()
    alias_short = _alias_for_naming(args.model)
    run_name = (
        f"{alias_short}_{args.instruction_variant}_{args.dataset}_{sha}"
        + ("_dirty" if dirty else "")
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_prefix = "subsumption_smoke" if args.smoke_test else "subsumption"
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("results") / f"{out_prefix}_{timestamp}_{run_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging — stdout + file mirror.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s]: %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); root.addHandler(sh)
    fh = logging.FileHandler(output_dir / "stdout.log", encoding="utf-8"); fh.setFormatter(fmt)
    root.addHandler(fh)
    logger = logging.getLogger("run_subsumption")

    logger.info("Run name: %s", run_name)
    logger.info("Output dir: %s", output_dir)
    logger.info("Git: sha=%s dirty=%s", sha, dirty)
    if dirty:
        logger.warning("Working tree is DIRTY — run is not reproducible from sha alone.")
        logger.warning("git status --porcelain output:\n%s", dirty_paths)

    _set_seeds(args.seed)
    device = _detect_device()
    logger.info("Device: %s", device)

    # Resolve instruction texts.
    # document_instruction_id default depends on the variant:
    #   symmetric  -> same instruction as the query side (i.e. symmetric_instruction_id),
    #                 because that is what "symmetric" actually means.
    #   asymmetric -> 'none' (empty) per the Stage-1 spec — document side stays
    #                 symmetric across the two runs and direction-agnostic.
    # Explicit --document-instruction-id always wins.
    from prompt import get_subsumption_instruction
    if args.document_instruction_id is None:
        args.document_instruction_id = (
            args.symmetric_instruction_id if args.instruction_variant == "symmetric" else "none"
        )
        logger.info(
            "document_instruction_id defaulted to '%s' (variant=%s)",
            args.document_instruction_id, args.instruction_variant,
        )
    document_instruction = get_subsumption_instruction(args.document_instruction_id)
    if args.instruction_variant == "symmetric":
        sym_instr = get_subsumption_instruction(args.symmetric_instruction_id)
        broader_instr = narrower_instr = ""
    else:
        sym_instr = ""
        broader_instr  = get_subsumption_instruction(args.broader_instruction_id)
        narrower_instr = get_subsumption_instruction(args.narrower_instruction_id)

    # Load dataset.
    from tracks.zenodo_loader import load_subdataset
    src_path, tgt_path, ref_path = load_subdataset(args.dataset)
    logger.info("Dataset '%s' paths: %s | %s | %s", args.dataset, src_path, tgt_path, ref_path)

    from Alignment import Alignment
    kg_source, source_labels = _load_kg_with_labels(src_path)
    kg_target, target_labels = _load_kg_with_labels(tgt_path)
    reference = Alignment(str(ref_path))
    logger.info(
        "Source classes=%d  target classes=%d  reference correspondences=%d",
        len(kg_source.get_classes()), len(kg_target.get_classes()), len(reference),
    )

    smoke_source_uris: list[str] = []
    if args.smoke_test:
        smoke_source_uris = _filter_kg_to_smoke(kg_source, n=3)
        logger.info("--smoke-test: limited source side to %d classes: %s",
                    len(smoke_source_uris), smoke_source_uris)

    # Build matcher.
    resolved_model = _resolve_model(args.model)
    if args.instruction_variant == "symmetric":
        from MatcherEmbeddingRetrieval import MatcherEmbeddingRetrieval
        matcher = MatcherEmbeddingRetrieval(
            model=resolved_model,
            description=args.description,
            query_instruction=sym_instr,
            document_instruction=document_instruction,
            output_relation="=",
            top_k=args.top_k_max,
            kg_format=args.kg_format,
        )
    else:
        from MatcherAsymmetricRetrieval import MatcherAsymmetricRetrieval
        matcher = MatcherAsymmetricRetrieval(
            model=resolved_model,
            description=args.description,
            broader_query_instruction=broader_instr,
            narrower_query_instruction=narrower_instr,
            document_instruction=document_instruction,
            top_k=args.top_k_max,
            kg_format=args.kg_format,
        )
    logger.info("Matcher: %s", matcher)

    # W&B.
    wandb_run = None
    wandb = None
    if args.wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            logger.error("wandb not installed; install with: pip install wandb")
            sys.exit(1)
        project = args.wandb_project or (
            "beyondequivalence-smoke" if args.smoke_test else "beyondequivalence-retrieval-stage1"
        )
        wandb_config = {
            "git_sha": sha,
            "git_dirty": dirty,
            "model_arg": args.model,
            "model_resolved": resolved_model,
            "instruction_variant": args.instruction_variant,
            "symmetric_instruction_id":   args.symmetric_instruction_id   if args.instruction_variant == "symmetric"  else None,
            "symmetric_instruction_text": sym_instr                       if args.instruction_variant == "symmetric"  else None,
            "broader_instruction_id":     args.broader_instruction_id     if args.instruction_variant == "asymmetric" else None,
            "broader_instruction_text":   broader_instr                   if args.instruction_variant == "asymmetric" else None,
            "narrower_instruction_id":    args.narrower_instruction_id    if args.instruction_variant == "asymmetric" else None,
            "narrower_instruction_text":  narrower_instr                  if args.instruction_variant == "asymmetric" else None,
            "document_instruction_id":    args.document_instruction_id,
            "document_instruction_text":  document_instruction,
            "dataset": args.dataset,
            "description": args.description,
            "top_k_max": args.top_k_max,
            "kg_format": args.kg_format,
            "seed": args.seed,
            "smoke_test": args.smoke_test,
            "device": device,
            "distance_logging_enabled": False,  # documented in docs/relations_per_dataset.md
            "cluster": os.getenv("CLUSTER", ""),
        }
        tags = [
            f"model:{alias_short}",
            f"variant:{args.instruction_variant}",
            f"dataset:{args.dataset}",
        ]
        if args.smoke_test:
            tags.append("mode:smoke")
        wandb_run = wandb.init(
            project=project, name=run_name, config=wandb_config, tags=tags,
            group=args.wandb_group,
        )
        logger.info("W&B run initialised: %s  group=%s", wandb_run.url, args.wandb_group)

    # Run matcher.
    t_start = time.perf_counter()
    predictions = matcher.match(kg_source, kg_target, Alignment(), parameters={})
    t_elapsed = time.perf_counter() - t_start
    logger.info("Matcher run finished in %.2fs; alignment size = %d", t_elapsed, len(predictions))

    # Score-distribution diagnostics (per-run for asym, single bucket for sym).
    score_diag = _score_diagnostics(predictions, args.instruction_variant)
    logger.info("Score diagnostics: %s", json.dumps(score_diag, indent=2))

    # Recall + MRR.
    from evaluation_recall import compute_recall_at_k
    k_values = tuple(k for k in DEFAULT_K_VALUES if k <= args.top_k_max) or (args.top_k_max,)
    report = compute_recall_at_k(
        reference, predictions,
        k_values=k_values,
        source_labels=source_labels,
        target_labels=target_labels,
    )
    for mode in ("strict", "lax", "per_relation_strict"):
        for label, by_k in report.recall_at_k[mode].items():
            for k, v in by_k.items():
                logger.info("recall_at_k_%s/%s/k=%d = %.4f", mode, label, k, v)
        for label, v in report.mrr[mode].items():
            logger.info("mrr_%s/%s = %.4f", mode, label, v)
    if report.dropped_relations_count:
        logger.warning("Dropped %d reference correspondence(s): %s",
                       report.dropped_relations_count, report.dropped_relations_breakdown)

    # Smoke probe.
    if args.smoke_test:
        _smoke_probe(
            logger, matcher, kg_source, smoke_source_uris, predictions, reference,
            source_labels, target_labels, args.description, args.kg_format,
        )

    # Persist artefacts.
    config_dump = vars(args).copy()
    config_dump.update({
        "git_sha": sha,
        "git_dirty": dirty,
        "model_resolved": resolved_model,
        "run_name": run_name,
        "timestamp": timestamp,
        "device": device,
        "smoke_source_uris": smoke_source_uris if args.smoke_test else None,
    })
    (output_dir / "config.json").write_text(json.dumps(config_dump, indent=2, ensure_ascii=False))

    metrics = {
        "recall_at_k": report.recall_at_k,
        "mrr": report.mrr,
        "n_reference_total": report.n_reference_total,
        "n_reference_after_filter": report.n_reference_after_filter,
        "dropped_relations_count": report.dropped_relations_count,
        "dropped_relations_breakdown": report.dropped_relations_breakdown,
        "score_diagnostics": score_diag,
        "matcher_runtime_seconds": t_elapsed,
        "matcher_last_run_metrics": matcher.last_run_metrics,
        "distance_breakdown": None,
        "distance_breakdown_reason": (
            "Cross-ontology distance not well-defined on STROMA/TaSeR sub-datasets; "
            "deferred — see docs/relations_per_dataset.md."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # predictions.tsv (sorted by source, then desc score, then target, then relation).
    pred_rows = sorted(
        ([cor.source, source_labels.get(cor.source, ""), cor.target,
          target_labels.get(cor.target, ""), cor.relation, float(cor.confidence)]
         for cor in predictions),
        key=lambda r: (r[0], -r[5], r[2], r[4]),
    )
    _write_tsv(
        output_dir / "predictions.tsv",
        ["source_uri", "source_label", "target_uri", "target_label", "predicted_relation", "score"],
        pred_rows,
    )
    # reference.tsv (raw + normalized relation).
    from evaluation_recall import _normalize_relation
    ref_rows = sorted(
        ([cor.source, cor.target, cor.relation, _normalize_relation(cor.relation) or ""]
         for cor in reference),
        key=lambda r: (r[0], r[1], r[2]),
    )
    _write_tsv(
        output_dir / "reference.tsv",
        ["source_uri", "target_uri", "relation_raw", "relation_normalized"],
        ref_rows,
    )
    # per_source_long.tsv
    if report.per_source_rows:
        cols = list(report.per_source_rows[0].keys())
        _write_tsv(
            output_dir / "per_source_long.tsv",
            cols,
            [[row[c] for c in cols] for row in report.per_source_rows],
        )

    # W&B logging.
    if wandb_run is not None:
        flat = report.to_wandb_metrics()
        flat["runtime/matcher_seconds"] = t_elapsed
        # Encoding side-metrics.
        for k, v in matcher.last_run_metrics.items():
            if isinstance(v, (int, float)) or v is None:
                if v is not None:
                    flat[f"encoding/{k}"] = v
            elif isinstance(v, dict) and k == "per_run":
                for run_label, sub in v.items():
                    for kk, vv in sub.items():
                        if isinstance(vv, (int, float)):
                            flat[f"encoding/{run_label}/{kk}"] = vv
        # Score diagnostics.
        for run_label, stats in score_diag.items():
            for kk, vv in stats.items():
                flat[f"score_diagnostics/{run_label}/{kk}"] = vv
        wandb.log(flat)
        if report.per_source_rows:
            cols = list(report.per_source_rows[0].keys())
            data = [[row[c] for c in cols] for row in report.per_source_rows]
            wandb_run.log({"per_source_top_k": wandb.Table(columns=cols, data=data)})
        wandb_run.finish()

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
