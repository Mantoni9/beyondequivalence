"""
run_stage2_bothorder.py — Stufe-B Both-Order-Voting double-order inference.

Queries each frozen Stage-1 candidate pair (deduped to unique (s, t)) in BOTH
argument orders (AB = present (s,t); BA = present (t,s), A2-inverted) and
persists both raw labels + both canonical predictions + both answer-span
logprobs. The three reconciliation variants (B1/B2/B3) are computed OFFLINE
by scripts/analyze_stufeB.py — this runner does ONE double-order pass
(~2x baseline), no reconciliation, no metrics.

Decoding pinned to Run 255471 verbatim: Llama-3.3-70B-AWQ via vLLM,
temperature 0.0 (greedy), max_new_tokens 256, seed 42.

Output dir results/<ts>_stage2bo_<dataset>_<prompt>/:
  - bothorder_predictions.tsv  (per pair: gold + both orders' raw/canonical/span-lp)
  - token_dump.tsv             (first N pairs' raw tokens for the MANUAL B2 gate)
  - config.json

Usage:
  python run_stage2_bothorder.py --dataset g7-literature \
      --stage1-predictions results/stage1_frozen/g7-literature_qwen3-noLoRA_pathctx_T2_top20.tsv \
      --stage1-top-k 20 --stage1-description description_path_context \
      --llm-model "$MODEL_PATH"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Alignment import Alignment
from evaluation_recall import _normalize_relation
from run_stage2_experiment import (
    _alias_for_naming, _build_llm, _load_kg_with_labels,
    _load_stage1_predictions_tsv, _resolve_model, _set_seeds, _write_tsv,
)
from stage2_bothorder import run_both_orders

logger = logging.getLogger("run_stage2_bothorder")

TOKEN_DUMP_N = 12   # first N pairs' tokens persisted for the manual B2 gate


def parse_args():
    p = argparse.ArgumentParser(description="Stufe-B double-order inference (one pass).")
    p.add_argument("--dataset", required=True)
    p.add_argument("--stage1-predictions", required=True)
    p.add_argument("--stage1-top-k", type=int, default=20)
    p.add_argument("--stage1-description", default="description_path_context")
    p.add_argument("--stage1-model", default="qwen3-embedding-8b")
    p.add_argument("--stage1-variant", default="asymmetric")
    p.add_argument("--stage1-template-id", default="T2")
    p.add_argument("--description", default=None,
                   help="Reranker verbalization; defaults to --stage1-description.")
    p.add_argument("--prompt-id", default="d_subs_v2",
                   help="Both-order voting uses the v2 wording (order is the "
                        "manipulation, not the prompt text).")
    p.add_argument("--kg-format", default="turtle")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-max-concurrency", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--smoke-test", action="store_true",
                   help="Restrict to first 3 source classes alphabetically.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.description is None:
        args.description = args.stage1_description

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    alias = _alias_for_naming(args.stage1_model)
    run_name = (f"stage2bo_{args.dataset}_s1-{alias}-{args.stage1_variant[:3]}-"
                f"{args.stage1_template_id}-{args.stage1_description}_p-{args.prompt_id}"
                + ("_smoke" if args.smoke_test else ""))
    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / f"{ts}_{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seeds(args.seed)

    from tracks.zenodo_loader import load_subdataset
    src_path, tgt_path, ref_path = load_subdataset(args.dataset)
    kg_source, source_labels = _load_kg_with_labels(src_path)
    kg_target, target_labels = _load_kg_with_labels(tgt_path)
    reference = Alignment(str(ref_path))

    candidates, loader_stats = _load_stage1_predictions_tsv(
        Path(args.stage1_predictions), args.stage1_top_k)
    if args.smoke_test:
        keep = sorted({c.source for c in candidates})[:3]
        keep_set = set(keep)
        sub = Alignment()
        for c in candidates:
            if c.source in keep_set:
                sub.add(c)
        candidates = sub
    logger.info("Loaded %d candidates (%s)", len(candidates), loader_stats)

    gold = {}
    for c in reference:
        n = _normalize_relation(c.relation)
        if n:
            gold[(c.source, c.target)] = n

    llm = _build_llm(args, logger)
    logger.info("Stufe-B double-order: dataset=%s prompt=%s desc=%s top_k=%d",
                args.dataset, args.prompt_id, args.description, args.stage1_top_k)

    t0 = time.perf_counter()
    records = run_both_orders(
        llm, kg_source, kg_target, candidates,
        prompt_id=args.prompt_id, description=args.description,
        kg_format=args.kg_format, max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    t_elapsed = time.perf_counter() - t0
    logger.info("Double-order done: %.1fs  pairs=%d", t_elapsed, len(records))

    # parse_fail per order (gate < 5%).
    n = max(1, len(records))
    ab_pf = sum(1 for r in records if r["ab_raw"] == "parse_fail") / n
    ba_pf = sum(1 for r in records if r["ba_raw"] == "parse_fail") / n
    span_missing = sum(1 for r in records
                       if r["ab_span_logprob"] is None or r["ba_span_logprob"] is None)
    logger.info("parse_fail: AB=%.3f BA=%.3f | span-logprob missing on %d/%d pairs",
                ab_pf, ba_pf, span_missing, len(records))
    if ab_pf > 0.05 or ba_pf > 0.05:
        logger.warning("parse_fail GATE exceeded (>5%%) — check prompt/truncation.")

    rows = []
    for r in records:
        rows.append([
            r["source"], source_labels.get(r["source"], ""),
            r["target"], target_labels.get(r["target"], ""),
            gold.get((r["source"], r["target"]), ""),
            r["stage1_relations"],
            r["ab_raw"], r["ab_canonical"], r["ab_span_logprob"],
            r["ba_raw"], r["ba_canonical"], r["ba_span_logprob"],
            r["ab_text"], r["ba_text"],
        ])
    _write_tsv(output_dir / "bothorder_predictions.tsv",
               ["source_uri", "source_label", "target_uri", "target_label",
                "gold_relation", "stage1_relations",
                "ab_raw", "ab_canonical", "ab_span_logprob",
                "ba_raw", "ba_canonical", "ba_span_logprob",
                "ab_text", "ba_text"],
               rows)

    # MANUAL-GATE token dump: first N pairs' raw AB/BA token sequences, so the
    # answer-span parser can be eyeballed before B2 scoring is trusted
    # (255391 lesson — never trust the span parser blind).
    _write_tsv(output_dir / "token_dump.tsv",
               ["pair_index", "order", "ab_or_ba_text"],
               [[i, "note", "Inspect that the first-line span captures the "
                 "'Relation: <label>' token before trusting B2 logprobs."]
                for i in range(0)] +
               [[i, "AB", records[i]["ab_text"]] for i in range(min(TOKEN_DUMP_N, len(records)))] +
               [[i, "BA", records[i]["ba_text"]] for i in range(min(TOKEN_DUMP_N, len(records)))])

    config_dump = {
        "timestamp": ts, "run_name": run_name, "dataset": args.dataset,
        "stage": "stufe-B double-order (B1/B2/B3 reconciled offline)",
        "stage1_predictions": args.stage1_predictions,
        "stage1_model": args.stage1_model, "stage1_variant": args.stage1_variant,
        "stage1_template_id": args.stage1_template_id,
        "stage1_description": args.stage1_description, "stage1_top_k": args.stage1_top_k,
        "llm_model": args.llm_model or os.getenv("MODEL_PATH"),
        "prompt_id": args.prompt_id, "description": args.description,
        "max_new_tokens": args.max_new_tokens, "seed": args.seed,
        "n_pairs": len(records), "parse_fail_ab": ab_pf, "parse_fail_ba": ba_pf,
        "runtime_seconds": t_elapsed, "loader_stats": loader_stats,
    }
    (output_dir / "config.json").write_text(json.dumps(config_dump, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", output_dir)


if __name__ == "__main__":
    main()
