"""
run_stage2_experiment.py — Stage-2 multi-class relation classifier smoke runner.

Two candidate-gen modes:
  - --stage1-predictions <path.tsv>  (default for the smoke):
      Load a pre-computed Stage-1 predictions.tsv into an Alignment.
      Skips the embedder entirely — no GPU conflict with the vLLM server.
      This is the path that fixes the 2026-06-02 OOM (job 255320): with
      vLLM holding ~44/47 GB on GPU 0 via tp=2 + gpu_memory_utilization=0.92,
      the Qwen3-8B embedder cannot also load on cuda:0. Decoupling
      retrieval and reranking by reading a persisted TSV solves it
      structurally instead of via memory-knob tuning.
  - --stage1-{model,variant,template-id,description,top-k}:
      In-process candidate gen via MatcherAsymmetricRetrieval /
      MatcherEmbeddingRetrieval. Only safe when the LLM backend lives
      out-of-process (vLLM on different GPUs / different node) or when
      no LLM runs at all. Kept available for local dev / future use.

Pipeline:
    candidates Alignment
      → dedup to unique (s, t)        (inside MatcherSubsumptionReranker)
      → Stage-2 reranker              (MatcherSubsumptionReranker on vLLM-served LLM)
      → metrics                       (evaluation_multiclass: 4x4 CM + per-rel P/R/F1)

LLM backend: LLMOpenAI when VLLM_BASE_URL is set (cluster default), else
LLMHuggingFace in-process (local fallback).

Output: metrics.json, confusion_matrix.tsv, predictions.tsv,
stage1_candidates.tsv, config.json in
``results/stage2_<timestamp>_<run-name>/`` (or --output-dir override).

Usage (TSV-loader mode — default for the smoke):
    python run_stage2_experiment.py \\
        --dataset g7-literature \\
        --stage1-predictions results/stage1_frozen/g7-literature_qwen3-noLoRA_pathctx_T2_top20.tsv \\
        --stage1-top-k 20 \\
        --llm-model "${MODEL_PATH}"

Usage (in-process candidate gen — local dev only, OOMs if vLLM on same GPU):
    python run_stage2_experiment.py \\
        --dataset g7-literature \\
        --stage1-model qwen3-embedding-8b \\
        --stage1-variant asymmetric \\
        --stage1-template-id T2 \\
        --stage1-description description_one_gen \\
        --llm-model "${MODEL_PATH}"

This is a single-run smoke; no W&B, no sweep machinery. Success = a
metrics.json with a 4x4 confusion matrix on one dataset.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from dotenv import load_dotenv
load_dotenv()


# Mirror of run_subsumption_experiment.MODEL_ALIASES — kept in sync manually
# because that module triggers a heavy import on first use.
MODEL_ALIASES: dict[str, str] = {
    "sbert":                   "sentence-transformers/all-MiniLM-L6-v2",
    "qwen3-embedding-8b":      "Qwen/Qwen3-Embedding-8B",
    "llama-embed-nemotron-8b": "nvidia/llama-embed-nemotron-8b",
    "e5-mistral":              "intfloat/e5-mistral-7b-instruct",
}


def _resolve_model(arg: str) -> str:
    return MODEL_ALIASES.get(arg, arg)


def _alias_for_naming(arg: str) -> str:
    if arg in MODEL_ALIASES:
        return arg
    safe = arg.split("/")[-1].replace(":", "-").replace("\\", "-").replace(" ", "_")
    return safe or "model"


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


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_kg_with_labels(rdf_path: Path):
    """Same pattern as run_subsumption_experiment._load_kg_with_labels."""
    from RDFGraphWrapper import RDFGraphWrapper
    kg = RDFGraphWrapper(str(rdf_path))
    labels: dict[str, str] = {}
    for cls in kg.get_classes():
        ls = kg.get_labels(cls)
        labels[str(cls)] = next(iter(ls)) if ls else ""
    return kg, labels


def _filter_kg_to_smoke(kg, n: int = 3) -> list[str]:
    """Restrict kg.get_classes() to first N URIs alphabetically. Same monkey-patch
    pattern as run_subsumption_experiment._filter_kg_to_smoke.
    """
    classes_sorted = sorted(kg.get_classes(), key=str)
    keep_uris = classes_sorted[:n]
    keep_set = set(keep_uris)
    kg.get_classes = lambda: keep_set  # noqa: ARG005 — intentional monkey-patch
    return [str(c) for c in keep_uris]


def _safe_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(_safe_cell(c) for c in row) + "\n")


# Schema aliases for Stage-1 predictions.tsv. Required logical columns are
# source / target / relation / score. Labels are optional (older runs omit
# them — they're recovered from kg_*_labels at runtime).
_TSV_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "source":   ("source_uri", "source"),
    "target":   ("target_uri", "target"),
    "relation": ("predicted_relation", "relation", "stage1_relation"),
    "score":    ("score", "stage1_score", "confidence"),
}


def _load_stage1_predictions_tsv(
    path: Path, top_k_per_direction: int,
) -> tuple["Alignment", dict]:
    """Load a Stage-1 predictions.tsv into an Alignment.

    Schema is header-driven (case-insensitive). Accepted column names:
      - source:   source_uri | source
      - target:   target_uri | target
      - relation: predicted_relation | relation | stage1_relation
      - score:    score | stage1_score | confidence

    The relation is normalised via ``evaluation_recall._normalize_relation``
    so legacy Unicode variants (≤, ⊑, ⊒, ≥) collapse to ASCII {=, <, >}.
    Rows whose relation does not normalise to one of those are dropped and
    counted (they wouldn't reach a {=, ⊏, ⊐} reranker decision anyway).

    Per (source, normalised_relation), entries are sorted by score DESC and
    capped to ``top_k_per_direction``. For an asymmetric Stage-1 run that
    emits '<' and '>', each source therefore keeps its top-K broader hits
    AND its top-K narrower hits — i.e. up to 2*K rows per source. This
    matches the cap used by MatcherAsymmetricRetrieval at retrieval time.

    Returns
    -------
    (alignment, stats) where ``stats`` records pre/post-cap counts plus
    any dropped-relation breakdown.
    """
    import csv
    from Alignment import Alignment
    from Correspondence import Correspondence
    from evaluation_recall import _normalize_relation

    if not path.is_file():
        raise FileNotFoundError(f"Stage-1 predictions TSV not found: {path}")

    raw_rows: list[tuple[str, str, str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"TSV {path} is empty")
        header_lc = [h.strip().lower() for h in header]

        col_idx: dict[str, int] = {}
        for logical, aliases in _TSV_COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in header_lc:
                    col_idx[logical] = header_lc.index(alias)
                    break
        missing = [k for k in _TSV_COLUMN_ALIASES if k not in col_idx]
        if missing:
            raise ValueError(
                f"TSV {path}: header missing columns {missing}. Header={header}. "
                f"Accepted aliases: {_TSV_COLUMN_ALIASES}"
            )

        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            try:
                src = row[col_idx["source"]].strip()
                tgt = row[col_idx["target"]].strip()
                rel = row[col_idx["relation"]].strip()
                sc = float(row[col_idx["score"]])
            except (IndexError, ValueError) as e:
                raise ValueError(f"TSV {path}: malformed row {row!r}: {e}")
            raw_rows.append((src, tgt, rel, sc))

    # Normalise relation; track drops.
    normalised: list[tuple[str, str, str, float]] = []
    dropped_rel: dict[str, int] = {}
    for src, tgt, rel, sc in raw_rows:
        norm = _normalize_relation(rel)
        if norm is None:
            dropped_rel[rel] = dropped_rel.get(rel, 0) + 1
            continue
        normalised.append((src, tgt, norm, sc))

    # Cap top-K per (source, normalised_relation), score desc; deterministic
    # tie-break by target then full key.
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for src, tgt, rel, sc in normalised:
        grouped.setdefault((src, rel), []).append((tgt, sc))
    capped: list[tuple[str, str, str, float]] = []
    for (src, rel), entries in grouped.items():
        entries.sort(key=lambda e: (-e[1], e[0]))
        for tgt, sc in entries[:top_k_per_direction]:
            capped.append((src, tgt, rel, sc))

    alignment = Alignment()
    for src, tgt, rel, sc in capped:
        alignment.add(Correspondence(src, tgt, rel, sc))

    stats = {
        "tsv_path":               str(path),
        "n_rows_raw":             len(raw_rows),
        "n_rows_after_normalize": len(normalised),
        "n_rows_after_cap":       len(capped),
        "top_k_per_direction":    top_k_per_direction,
        "dropped_relation_breakdown": dropped_rel,
    }
    return alignment, stats


def _log_candidate_stats(alignment: "Alignment", logger: logging.Logger) -> dict:
    """Log per-(s,t) and per-direction stats. Useful pre-reranker sanity:
    direction_accuracy is only statistically meaningful when both '<' and '>'
    populations are non-trivial.
    """
    by_pair: dict[tuple[str, str], set[str]] = {}
    rel_counts: dict[str, int] = {}
    for cor in alignment:
        by_pair.setdefault((cor.source, cor.target), set()).add(cor.relation)
        rel_counts[cor.relation] = rel_counts.get(cor.relation, 0) + 1

    only_lt = sum(1 for rels in by_pair.values() if rels == {"<"})
    only_gt = sum(1 for rels in by_pair.values() if rels == {">"})
    only_eq = sum(1 for rels in by_pair.values() if rels == {"="})
    both_lt_gt = sum(1 for rels in by_pair.values() if rels >= {"<", ">"})
    other_combinations = (
        len(by_pair) - only_lt - only_gt - only_eq - both_lt_gt
    )

    sources = {src for src, _ in by_pair.keys()}
    stats = {
        "n_correspondences":   len(alignment),
        "n_unique_pairs":      len(by_pair),
        "n_unique_sources":    len(sources),
        "per_relation_rows":   rel_counts,
        "pairs_only_<":        only_lt,
        "pairs_only_>":        only_gt,
        "pairs_only_=":        only_eq,
        "pairs_both_<_>":      both_lt_gt,
        "pairs_other_combos":  other_combinations,
    }
    logger.info("Candidate stats (pre-reranker):")
    logger.info("  rows=%d  unique_pairs=%d  unique_sources=%d  per_relation=%s",
                stats["n_correspondences"], stats["n_unique_pairs"],
                stats["n_unique_sources"], stats["per_relation_rows"])
    logger.info("  pair direction breakdown: only_<=%d  only_>=%d  only_==%d  "
                "both_<>=%d  other=%d",
                only_lt, only_gt, only_eq, both_lt_gt, other_combinations)
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-2 multi-class relation classifier (smoke).")

    # ── Dataset ───────────────────────────────────────────────────────────────
    p.add_argument("--dataset", default="g7-literature",
                   help="BeyondEquivalence sub-dataset (see tracks.zenodo_loader).")

    # ── TSV-loader mode (preferred — decouples retrieval from reranking). ────
    p.add_argument("--stage1-predictions", default=None,
                   help=("Path to a pre-computed Stage-1 predictions.tsv. "
                         "When set, the in-process candidate gen below is "
                         "SKIPPED — no embedder is loaded, no GPU conflict "
                         "with vLLM. Header-driven schema (source_uri, "
                         "target_uri, predicted_relation|relation, score; "
                         "labels optional)."))

    # ── Stage-1 in-process candidate-gen config (only used when
    # ── --stage1-predictions is NOT set). Parametrised, NOT hardcoded. ───────
    p.add_argument("--stage1-model", default="qwen3-embedding-8b",
                   help=("Stage-1 encoder. Alias or HF id / local path. "
                         "Aliases: " + ", ".join(MODEL_ALIASES) + ". "
                         "DEFAULT is Antonio's interim guess for the 2026-05-27 "
                         "frozen Stage-1 config — override before running."))
    p.add_argument("--stage1-variant", default="asymmetric",
                   choices=("symmetric", "asymmetric"),
                   help="Stage-1 retrieval variant.")
    p.add_argument("--stage1-template-id", default="T2",
                   help=("Stage-1 instruction-template id. "
                         "For asymmetric: T1..T5. For symmetric: S1..S5. "
                         "Pass 'noinstr' or empty to disable instructions "
                         "(sbert / non-instruction-tuned encoders)."))
    p.add_argument("--stage1-description", default="description_one_gen",
                   help="RDFGraphWrapper description method used by Stage-1.")
    p.add_argument("--stage1-top-k", type=int, default=20,
                   help=("Top-K candidates per source. In in-process mode "
                         "this is the matcher's top_k; in TSV-loader mode "
                         "this caps top-K per (source, normalised_relation) "
                         "pair, so an asymmetric TSV with rows in '<' and "
                         "'>' keeps up to 2*K rows per source."))

    # ── Stage-2 reranker config. ──────────────────────────────────────────────
    p.add_argument("--llm-model", default=None,
                   help=("Model name / path for the Stage-2 LLM. "
                         "When VLLM_BASE_URL is set the value is passed to "
                         "LLMOpenAI as model_name; otherwise LLMHuggingFace "
                         "loads it in-process. Defaults to env MODEL_PATH."))
    p.add_argument("--prompt-id", default="d_subs_v2",
                   help=("RERANKING_PROMPTS key for the multi-class prompt. "
                         "d_subs_v2 (default) is the answer-first variant; "
                         "d_subs is the original 'think then answer' variant "
                         "which broke under max_new_tokens=256 on Llama "
                         "(see job 255391 post-mortem 2026-06-02)."))
    p.add_argument("--description", default=None,
                   help=("RDFGraphWrapper description method used by the Stage-2 "
                         "reranker. Defaults to --stage1-description."))
    p.add_argument("--kg-format", default="turtle")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--swap-pair-presentation", action="store_true",
                   help=("Stufe-A arm A2: fill the prompt slots with (target, "
                         "source) instead of (source, target); directional "
                         "labels are inverted exactly once at parse time when "
                         "mapping back to the canonical (s, t) pair. Prompt "
                         "text and verbalizations are untouched."))
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Decoding temperature. Stage-2 matrix: reasoners "
                        "model-recommended (>0), non-reasoners 0.0 (default).")
    p.add_argument("--top-p", type=float, default=None,
                   help="Optional nucleus top_p (only sent when set).")
    p.add_argument("--threshold", type=float, default=0.0,
                   help=("Optional confidence cutoff applied AFTER the 'none' "
                         "filter. 0.0 = keep all non-none predictions."))
    p.add_argument("--llm-max-concurrency", type=int, default=16,
                   help=("Concurrent in-flight requests to the vLLM endpoint "
                         "(LLMOpenAI ThreadPoolExecutor workers). 1 = serial "
                         "(legacy behaviour). 16 is a safe default for "
                         "Llama-3.3-70B-AWQ on 2x A40; vLLM's continuous "
                         "batching handles the concurrency natively. Only "
                         "consumed when the vLLM/OpenAI backend is active."))
    # ── Reasoning ablation (D9). Only sent to the backend when set. ────────────
    p.add_argument("--reasoning-effort", default=None,
                   choices=("low", "medium", "high"),
                   help=("gpt-oss (Harmony): request-body reasoning_effort. Only "
                         "sent when set; otherwise the server default is used."))
    p.add_argument("--disable-thinking", action="store_true",
                   help=("Hybrid reasoners (Gemma-4): send "
                         "chat_template_kwargs.enable_thinking=False so the model "
                         "answers without a CoT span. Verified: gemma-4-31B-it's "
                         "chat_template honours enable_thinking."))
    # ── Few-shot ablation (E15). A0 = zero-shot (plain d_subs_v2). ─────────────
    p.add_argument("--few-shot-arm", default="A0",
                   choices=("A0", "A1", "A2", "A3", "A4"),
                   help=("E15 arm. A0=zero-shot; A1=N=1 '<'; A2=balanced-3; "
                         "A3=balanced-6; A4=mirrored-6. A1..A4 inject held-out "
                         "exemplars via d_subs_v2_fs."))
    p.add_argument("--exemplar-track", default="g1-web",
                   help="Held-out track for few-shot exemplars (eval-disjoint from g5,g7,g3).")
    p.add_argument("--exemplar-seed", type=int, default=None,
                   help="Seed for exemplar selection; defaults to --seed.")

    # ── I/O. ──────────────────────────────────────────────────────────────────
    p.add_argument("--output-dir", default=None,
                   help="Override output dir.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true",
                   help="Restrict to first 3 source classes alphabetically.")

    return p.parse_args()


def _build_stage1_matcher(args, logger):
    """Construct MatcherAsymmetricRetrieval or MatcherEmbeddingRetrieval per
    --stage1-variant. Resolves instruction strings from --stage1-template-id.
    """
    from MatcherEmbeddingRetrieval import MatcherEmbeddingRetrieval
    from MatcherAsymmetricRetrieval import MatcherAsymmetricRetrieval
    from prompt import (
        get_subb_sym_template, get_subb_asym_templates,
        SUBB_SYM_TEMPLATE_IDS, SUBB_ASYM_TEMPLATE_IDS,
    )

    resolved = _resolve_model(args.stage1_model)
    template = args.stage1_template_id or ""
    template_no = template.lower() in ("", "noinstr", "none")

    if args.stage1_variant == "symmetric":
        sym_instr = "" if template_no else get_subb_sym_template(template)
        logger.info(
            "Stage-1 matcher: MatcherEmbeddingRetrieval(model=%s, desc=%s, sym=%r)",
            resolved, args.stage1_description, sym_instr[:60],
        )
        return MatcherEmbeddingRetrieval(
            model=resolved,
            description=args.stage1_description,
            query_instruction=sym_instr,
            document_instruction=sym_instr,  # symmetric = same on both sides
            output_relation="=",
            top_k=args.stage1_top_k,
            kg_format=args.kg_format,
        )

    # asymmetric
    if template_no:
        broader_instr = narrower_instr = ""
    else:
        broader_instr, narrower_instr = get_subb_asym_templates(template)
    logger.info(
        "Stage-1 matcher: MatcherAsymmetricRetrieval(model=%s, desc=%s, broader=%r, narrower=%r)",
        resolved, args.stage1_description, broader_instr[:60], narrower_instr[:60],
    )
    return MatcherAsymmetricRetrieval(
        model=resolved,
        description=args.stage1_description,
        broader_query_instruction=broader_instr,
        narrower_query_instruction=narrower_instr,
        document_instruction="",  # explicit: empty document side for all asym
        top_k=args.stage1_top_k,
        kg_format=args.kg_format,
    )


def _reasoning_extra_body(reasoning_effort=None, disable_thinking=False) -> dict:
    """Pure map from the D9 ablation flags to the OpenAI/vLLM extra_body dict.
    Empty when neither is set (server defaults). Kept pure for unit testing."""
    eb: dict = {}
    if reasoning_effort:
        eb["reasoning_effort"] = reasoning_effort
    if disable_thinking:
        eb["chat_template_kwargs"] = {"enable_thinking": False}
    return eb


def _build_llm(args, logger):
    """Pick LLMOpenAI (vLLM) or LLMHuggingFace based on VLLM_BASE_URL."""
    model = args.llm_model or os.getenv("MODEL_PATH")
    if not model:
        sys.exit("Either --llm-model or env MODEL_PATH must be set.")
    vllm_url = os.getenv("VLLM_BASE_URL")
    if vllm_url:
        from LLMOpenAI import LLMOpenAI
        extra_body = _reasoning_extra_body(
            getattr(args, "reasoning_effort", None),
            getattr(args, "disable_thinking", False),
        )
        logger.info(
            "Stage-2 LLM backend: LLMOpenAI -> vLLM at %s  model=%s  max_concurrency=%d  extra_body=%s",
            vllm_url, model, args.llm_max_concurrency, extra_body or None,
        )
        return LLMOpenAI(
            model_name=model,
            base_url=vllm_url,
            api_key="EMPTY",
            max_concurrency=args.llm_max_concurrency,
            extra_body=extra_body or None,
        )
    from LLMHuggingFace import LLMHuggingFace
    logger.info("Stage-2 LLM backend: LLMHuggingFace (in-process)  model=%s", model)
    return LLMHuggingFace(model)


def main() -> None:
    args = parse_args()
    if args.description is None:
        args.description = args.stage1_description

    # ── Logging + output dir. ─────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    alias = _alias_for_naming(args.stage1_model)
    run_name = (
        f"stage2_{args.dataset}_s1-{alias}-{args.stage1_variant[:3]}-"
        f"{args.stage1_template_id or 'noinstr'}-{args.stage1_description}"
        f"_p-{args.prompt_id}"
        + (f"_fs{args.few_shot_arm}" if args.few_shot_arm != "A0" else "")
        + ("_swapped" if args.swap_pair_presentation else "")
        + ("_smoke" if args.smoke_test else "")
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("results") / f"{ts}_{run_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s]: %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); root.addHandler(sh)
    fh = logging.FileHandler(output_dir / "stdout.log", encoding="utf-8")
    fh.setFormatter(fmt); root.addHandler(fh)
    logger = logging.getLogger("run_stage2")

    _set_seeds(args.seed)
    device = _detect_device()
    logger.info("Stage-2 smoke runner. output_dir=%s device=%s", output_dir, device)
    logger.info("args=%s", vars(args))

    # ── Load dataset. ─────────────────────────────────────────────────────────
    from Alignment import Alignment
    from tracks.zenodo_loader import load_subdataset
    src_path, tgt_path, ref_path = load_subdataset(args.dataset)
    kg_source, source_labels = _load_kg_with_labels(src_path)
    kg_target, target_labels = _load_kg_with_labels(tgt_path)
    reference = Alignment(str(ref_path))
    logger.info("Dataset %s: src=%d tgt=%d refs=%d",
                args.dataset,
                len(kg_source.get_classes()), len(kg_target.get_classes()),
                len(reference))

    if args.smoke_test:
        kept = _filter_kg_to_smoke(kg_source, n=3)
        logger.info("SMOKE-TEST: restricted source to %d classes: %s",
                    len(kept), kept)

    # ── Stage 1: candidate gen (TSV-loader OR in-process). ────────────────────
    stage1_mode: str
    stage1_loader_stats: dict = {}
    if args.stage1_predictions:
        stage1_mode = "tsv_loader"
        logger.info("Stage-1 mode: TSV LOADER  path=%s  top_k_per_direction=%d",
                    args.stage1_predictions, args.stage1_top_k)
        t0 = time.perf_counter()
        candidates, stage1_loader_stats = _load_stage1_predictions_tsv(
            Path(args.stage1_predictions), top_k_per_direction=args.stage1_top_k,
        )
        t_stage1 = time.perf_counter() - t0
        logger.info(
            "Loaded TSV: raw=%d  after_normalize=%d  after_cap=%d  dropped_rels=%s",
            stage1_loader_stats["n_rows_raw"],
            stage1_loader_stats["n_rows_after_normalize"],
            stage1_loader_stats["n_rows_after_cap"],
            stage1_loader_stats["dropped_relation_breakdown"],
        )
        if args.smoke_test:
            # In TSV mode, smoke-restriction means: drop candidates whose
            # source is no longer in the (monkey-patched) get_classes set.
            keep_sources = {str(c) for c in kg_source.get_classes()}
            filtered = Alignment()
            for cor in candidates:
                if cor.source in keep_sources:
                    filtered.add(cor)
            n_before = len(candidates)
            candidates = filtered
            logger.info(
                "SMOKE-TEST filter applied to TSV candidates: %d -> %d",
                n_before, len(candidates),
            )
    else:
        stage1_mode = "in_process"
        logger.warning(
            "Stage-1 mode: IN-PROCESS retrieval. This path OOMs on a single "
            "GPU shared with the vLLM server (verified on job 255320, "
            "2026-06-02). Prefer --stage1-predictions when vLLM is up on the "
            "same node."
        )
        stage1 = _build_stage1_matcher(args, logger)
        t0 = time.perf_counter()
        candidates = stage1.match(kg_source, kg_target, Alignment(), parameters={})
        t_stage1 = time.perf_counter() - t0
        logger.info("Stage-1 done: %.1fs  alignment_size=%d", t_stage1, len(candidates))

        # Free Stage-1 embedder before loading the LLM (mostly relevant in the
        # HF fallback path where both live in-process). vLLM is out-of-process;
        # safe.
        if hasattr(stage1, "_embedder") and stage1._embedder is not None:
            stage1._embedder = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Pre-reranker candidate stats — separate broader '<' / narrower '>'
    # populations. Direction_accuracy downstream is only statistically
    # meaningful when both populations are non-trivial.
    candidate_stats = _log_candidate_stats(candidates, logger)

    # ── Stage 2: reranker. ────────────────────────────────────────────────────
    from MatcherSubsumptionReranker import MatcherSubsumptionReranker
    llm = _build_llm(args, logger)
    # E15 few-shot: build the exemplar block from the held-out track; auto-swap
    # to the {exemplars}-carrying prompt for A1..A4 (A0 stays plain d_subs_v2,
    # byte-identical to the matrix). Empty block for A0.
    from fewshot_exemplars import build_fewshot_block
    few_shot_block, exemplar_manifest = build_fewshot_block(
        arm=args.few_shot_arm, exemplar_track=args.exemplar_track,
        description=args.description, kg_format=args.kg_format,
        seed=args.exemplar_seed if args.exemplar_seed is not None else args.seed,
    )
    prompt_id = args.prompt_id
    if args.few_shot_arm != "A0" and prompt_id == "d_subs_v2":
        prompt_id = "d_subs_v2_fs"
    if args.few_shot_arm != "A0":
        logger.info("E15 few-shot: arm=%s track=%s n_exemplars=%d prompt=%s",
                    args.few_shot_arm, args.exemplar_track, len(exemplar_manifest), prompt_id)
    reranker = MatcherSubsumptionReranker(
        llm=llm,
        prompt_id=prompt_id,
        description=args.description,
        kg_format=args.kg_format,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        batch_size=args.batch_size,
        swap_pair_presentation=args.swap_pair_presentation,
        temperature=args.temperature,
        top_p=args.top_p,
        few_shot_block=few_shot_block,
    )
    logger.info("Stage-2 reranker: %s  (temp=%.2f top_p=%s)",
                reranker, args.temperature, args.top_p)

    t0 = time.perf_counter()
    predictions = reranker.match(kg_source, kg_target, candidates, parameters={})
    t_stage2 = time.perf_counter() - t0
    logger.info("Stage-2 done: %.1fs  predictions_size=%d", t_stage2, len(predictions))

    # ── Evaluate. ─────────────────────────────────────────────────────────────
    from evaluation_multiclass import (
        compute_multiclass_metrics, format_confusion_matrix_tsv, DISPLAY_LABELS,
    )
    candidate_pairs = {(c.source, c.target) for c in candidates}
    report = compute_multiclass_metrics(
        reference=reference, predictions=predictions, candidate_pairs=candidate_pairs,
    )

    # ── Persist artefacts. ────────────────────────────────────────────────────
    config_dump = {
        "timestamp":  ts,
        "run_name":   run_name,
        "dataset":    args.dataset,
        "stage1": {
            "mode":               stage1_mode,
            "predictions_tsv":    args.stage1_predictions,
            "model":              args.stage1_model,
            "model_resolved":     _resolve_model(args.stage1_model),
            "variant":            args.stage1_variant,
            "template_id":        args.stage1_template_id,
            "description":        args.stage1_description,
            "top_k":              args.stage1_top_k,
            "loader_stats":       stage1_loader_stats,
            "candidate_stats":    candidate_stats,
        },
        "stage2": {
            "llm_model":           args.llm_model or os.getenv("MODEL_PATH"),
            "vllm_base_url":       os.getenv("VLLM_BASE_URL"),
            "backend":             "vllm" if os.getenv("VLLM_BASE_URL") else "huggingface",
            "prompt_id":           prompt_id,
            "few_shot_arm":        args.few_shot_arm,
            "exemplar_track":      args.exemplar_track,
            "exemplar_seed":       args.exemplar_seed if args.exemplar_seed is not None else args.seed,
            "exemplar_manifest":   exemplar_manifest,
            "swap_pair_presentation": args.swap_pair_presentation,
            "temperature":         args.temperature,
            "top_p":               args.top_p,
            "description":         args.description,
            "kg_format":           args.kg_format,
            "batch_size":          args.batch_size,
            "max_new_tokens":      args.max_new_tokens,
            "threshold":           args.threshold,
            "llm_max_concurrency": args.llm_max_concurrency,
            "reasoning_effort":    args.reasoning_effort,
            "disable_thinking":    args.disable_thinking,
        },
        "smoke_test":   args.smoke_test,
        "seed":         args.seed,
        "device":       device,
        "src_path":     str(src_path),
        "tgt_path":     str(tgt_path),
        "ref_path":     str(ref_path),
        "runtime": {
            "stage1_seconds": t_stage1,
            "stage2_seconds": t_stage2,
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_dump, indent=2, ensure_ascii=False)
    )

    metrics = report.to_dict()
    metrics["runtime"] = config_dump["runtime"]
    # Surface per-canonical reranker outcomes (incl. parse_fail rate) at the
    # top of metrics.json — a >5% parse_fail rate is a load-bearing red flag
    # for format-compliance regressions and should be the first thing seen.
    canonical_counts: dict[str, int] = {}
    for d in reranker.last_run_details:
        c = d["parsed_canonical"]
        canonical_counts[c] = canonical_counts.get(c, 0) + 1
    metrics["reranker_canonical_counts"] = canonical_counts
    n_total = max(1, sum(canonical_counts.values()))
    metrics["reranker_parse_fail_rate"] = canonical_counts.get("parse_fail", 0) / n_total
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False)
    )

    (output_dir / "confusion_matrix.tsv").write_text(
        format_confusion_matrix_tsv(report.confusion)
    )

    # predictions.tsv — full per-candidate audit, including dropped ones.
    pred_rows = []
    for d in reranker.last_run_details:
        pred_rows.append([
            d["source"],
            source_labels.get(d["source"], ""),
            d["target"],
            target_labels.get(d["target"], ""),
            d["stage1_relations"],
            d["stage1_max_confidence"],
            d["parsed_canonical"],
            d["predicted_relation"],
            d["confidence"],
            d["sum_logprob"],
            d["n_tokens"],
            d["kept"],
            d["drop_reason"],
            (d["raw_response"] or "").replace("\n", "  "),
        ])
    _write_tsv(
        output_dir / "predictions.tsv",
        ["source_uri", "source_label", "target_uri", "target_label",
         "stage1_relations", "stage1_max_confidence",
         "parsed_canonical", "predicted_relation",
         "confidence", "sum_logprob", "n_tokens", "kept", "drop_reason",
         "raw_response"],
        pred_rows,
    )

    # stage1_candidates.tsv — the dedup-input candidate set for audit.
    cand_rows = sorted(
        ([c.source, source_labels.get(c.source, ""), c.target,
          target_labels.get(c.target, ""), c.relation, float(c.confidence)]
         for c in candidates),
        key=lambda r: (r[0], -r[5], r[2], r[4]),
    )
    _write_tsv(
        output_dir / "stage1_candidates.tsv",
        ["source_uri", "source_label", "target_uri", "target_label",
         "stage1_relation", "stage1_score"],
        cand_rows,
    )

    # ── Console summary. ──────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("Stage-2 smoke summary")
    logger.info("  universe=%d  candidates=%d  refs(kept/total)=%d/%d  "
                "stage1_misses=%d  predicted_partof=%d",
                report.n_universe, report.n_candidates_total,
                report.n_reference_after_filter, report.n_reference_total,
                report.n_gold_not_in_candidates, report.predicted_partof_count)
    for c in ("=", "<", ">"):
        pc = report.per_class[c]
        logger.info("  rel=%s  P=%.3f  R=%.3f  F1=%.3f  (tp=%d fp=%d fn=%d support=%d)",
                    c, pc["precision"], pc["recall"], pc["f1"],
                    pc["tp"], pc["fp"], pc["fn"], pc["support"])
    logger.info("  macro_F1=%.3f  micro_F1=%.3f  direction_acc=%s",
                report.macro_f1, report.micro_f1,
                f"{report.direction_accuracy:.3f}" if report.direction_accuracy is not None else "n/a")
    # Flip headline metrics — the headline for prompt-asymmetry diagnosis.
    logger.info(
        "  Flip metrics: gold->-pred< = %d (rate=%s) | gold<-pred> = %d (rate=%s) | "
        "direction_asymmetry = %s  (positive = pro-subclass bias)",
        report.flip_gt_to_lt,
        f"{report.flip_rate_gt:.3f}" if report.flip_rate_gt is not None else "n/a",
        report.flip_lt_to_gt,
        f"{report.flip_rate_lt:.3f}" if report.flip_rate_lt is not None else "n/a",
        f"{report.direction_asymmetry:+.3f}" if report.direction_asymmetry is not None else "n/a",
    )
    logger.info("Confusion matrix (gold rows x pred cols, labels=%s):",
                list(DISPLAY_LABELS))
    for g in DISPLAY_LABELS:
        row = [str(report.confusion[g][p]) for p in DISPLAY_LABELS]
        logger.info("    %s: %s", g, "  ".join(f"{x:>5}" for x in row))
    logger.info("=" * 72)
    logger.info("Artefacts in %s", output_dir)


if __name__ == "__main__":
    main()
