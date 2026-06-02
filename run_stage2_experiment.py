"""
run_stage2_experiment.py — Stage-2 multi-class relation classifier smoke runner.

Pipeline:
    Stage-1 candidate gen  (MatcherAsymmetricRetrieval / MatcherEmbeddingRetrieval)
      → dedup to unique (s, t)
      → Stage-2 reranker    (MatcherSubsumptionReranker on vLLM-served LLM)
      → metrics            (evaluation_multiclass: 4x4 CM + per-rel P/R/F1)

The Stage-1 config (encoder, variant, template, description) is a PARAMETER,
not hardcoded. Pass --stage1-model / --stage1-variant / --stage1-template-id /
--stage1-description; the smoke job script wires sensible defaults.

LLM backend: LLMOpenAI when VLLM_BASE_URL is set (cluster default), else
LLMHuggingFace in-process (local fallback).

Output: metrics.json, confusion_matrix.tsv, predictions.tsv,
stage1_candidates.tsv, config.json in
``results/stage2_<timestamp>_<run-name>/`` (or --output-dir override).

Usage:
    python run_stage2_experiment.py \\
        --dataset g7-literature \\
        --stage1-model qwen3-embedding-8b \\
        --stage1-variant asymmetric \\
        --stage1-template-id T2 \\
        --stage1-description description_one_gen \\
        --llm-model "${MODEL_PATH}" \\
        --threshold 0.0

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-2 multi-class relation classifier (smoke).")

    # ── Dataset ───────────────────────────────────────────────────────────────
    p.add_argument("--dataset", default="g7-literature",
                   help="BeyondEquivalence sub-dataset (see tracks.zenodo_loader).")

    # ── Stage-1 candidate-gen config (parametrised, NOT hardcoded). ──────────
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
                   help="Top-K candidates per source from Stage-1.")

    # ── Stage-2 reranker config. ──────────────────────────────────────────────
    p.add_argument("--llm-model", default=None,
                   help=("Model name / path for the Stage-2 LLM. "
                         "When VLLM_BASE_URL is set the value is passed to "
                         "LLMOpenAI as model_name; otherwise LLMHuggingFace "
                         "loads it in-process. Defaults to env MODEL_PATH."))
    p.add_argument("--prompt-id", default="d_subs",
                   help="RERANKING_PROMPTS key for the multi-class prompt.")
    p.add_argument("--description", default=None,
                   help=("RDFGraphWrapper description method used by the Stage-2 "
                         "reranker. Defaults to --stage1-description."))
    p.add_argument("--kg-format", default="turtle")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.0,
                   help=("Optional confidence cutoff applied AFTER the 'none' "
                         "filter. 0.0 = keep all non-none predictions."))

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


def _build_llm(args, logger):
    """Pick LLMOpenAI (vLLM) or LLMHuggingFace based on VLLM_BASE_URL."""
    model = args.llm_model or os.getenv("MODEL_PATH")
    if not model:
        sys.exit("Either --llm-model or env MODEL_PATH must be set.")
    vllm_url = os.getenv("VLLM_BASE_URL")
    if vllm_url:
        from LLMOpenAI import LLMOpenAI
        logger.info("Stage-2 LLM backend: LLMOpenAI -> vLLM at %s  model=%s", vllm_url, model)
        return LLMOpenAI(model_name=model, base_url=vllm_url, api_key="EMPTY")
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

    # ── Stage 1: candidate gen. ───────────────────────────────────────────────
    stage1 = _build_stage1_matcher(args, logger)
    t0 = time.perf_counter()
    candidates = stage1.match(kg_source, kg_target, Alignment(), parameters={})
    t_stage1 = time.perf_counter() - t0
    logger.info("Stage-1 done: %.1fs  alignment_size=%d", t_stage1, len(candidates))

    # Free Stage-1 embedder before loading the LLM (mostly relevant in the HF
    # fallback path where both live in-process). vLLM is out-of-process; safe.
    if hasattr(stage1, "_embedder") and stage1._embedder is not None:
        stage1._embedder = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Stage 2: reranker. ────────────────────────────────────────────────────
    from MatcherSubsumptionReranker import MatcherSubsumptionReranker
    llm = _build_llm(args, logger)
    reranker = MatcherSubsumptionReranker(
        llm=llm,
        prompt_id=args.prompt_id,
        description=args.description,
        kg_format=args.kg_format,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    logger.info("Stage-2 reranker: %s", reranker)

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
            "model":          args.stage1_model,
            "model_resolved": _resolve_model(args.stage1_model),
            "variant":        args.stage1_variant,
            "template_id":    args.stage1_template_id,
            "description":    args.stage1_description,
            "top_k":          args.stage1_top_k,
        },
        "stage2": {
            "llm_model":      args.llm_model or os.getenv("MODEL_PATH"),
            "vllm_base_url":  os.getenv("VLLM_BASE_URL"),
            "backend":        "vllm" if os.getenv("VLLM_BASE_URL") else "huggingface",
            "prompt_id":      args.prompt_id,
            "description":    args.description,
            "kg_format":      args.kg_format,
            "batch_size":     args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "threshold":      args.threshold,
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
    logger.info("Confusion matrix (gold rows x pred cols, labels=%s):",
                list(DISPLAY_LABELS))
    for g in DISPLAY_LABELS:
        row = [str(report.confusion[g][p]) for p in DISPLAY_LABELS]
        logger.info("    %s: %s", g, "  ".join(f"{x:>5}" for x in row))
    logger.info("=" * 72)
    logger.info("Artefacts in %s", output_dir)


if __name__ == "__main__":
    main()
