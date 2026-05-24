"""
validate_bidirectional.py — functional test of the bidirectional (< and >)
Stage-1 retrieval pipeline.

This is a VALIDATION run, not a sweep and not a freeze. It exists to verify one
thing end-to-end: that the narrower-pass '>' direction flows through
MatcherAsymmetricRetrieval into evaluation_recall.per_relation_strict.superclass
with real, non-zero numbers — i.e. that evaluation verdict (A) holds in practice.

Fixed configuration (per the validation task):
  - model:    llama-embed-nemotron-8b + optional Nemo LoRA adapter
  - Lever A:  path_context  (description_path_context)
  - matcher:  MatcherAsymmetricRetrieval — emits BOTH '<' (broader) and '>'
              (narrower). NOT the '<'-only MatcherBidirectionalConsolidation.
  - Lever C:  off (no RRF fusion)
  - Lever B:  off -> SUBB_DEFAULT_ASYM template (T1) for the asym instructions
  - top-20, asymmetric, all 6 STROMA/TaSeR sub-datasets

Additive only: imports the frozen Stage-1 matchers / evaluator and composes
them. Does NOT modify run_subsumption_experiment.py, the matchers, or
evaluation_recall.py.

Exit codes: 0 = ok; 2 = superclass R@20 aggregate is still 0.000 (the failure
the task asks us to catch — '>' is not reaching the metric).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from dotenv import load_dotenv
load_dotenv()

# Reuse frozen Stage-1 helpers (import, not modify).
from run_subsumption_experiment import (
    MODEL_ALIASES, _resolve_model, _alias_for_naming,
    _load_kg_with_labels, _set_seeds, _git_sha_and_dirty, _detect_device,
)
from Alignment import Alignment
from tracks.zenodo_loader import load_subdataset
from evaluation_recall import compute_recall_at_k, _normalize_relation
from MatcherAsymmetricRetrieval import MatcherAsymmetricRetrieval
from prompt import get_subb_asym_templates
from subB_pinned_config import SUBB_DEFAULT_ASYM

DEFAULT_DATASETS = (
    "mouse-human", "g1-web", "g2-diseases", "g3-text", "g5-groceries", "g7-literature",
)

logger = logging.getLogger("validate_bidirectional")


def _attach_lora(matcher: MatcherAsymmetricRetrieval, adapter_path: str) -> None:
    """Attach a PEFT LoRA adapter to the matcher's embedder, mirroring the
    working ablation path (run_subsumption_experiment.py:1733-1751).

    We wrap model[0].auto_model with PeftModel.from_pretrained, NOT
    SentenceTransformer.load_adapter — the latter is a silent no-op on
    sentence-transformers 5.4.1 + peft 0.19.1 (documented in finetune_lora.py /
    THESIS_NOTES.md). Loading via PeftModel is the only path that actually
    activates the adapter.
    """
    matcher._ensure_embedder()
    from peft import PeftModel
    inner = matcher._embedder[0].auto_model
    if isinstance(inner, PeftModel):
        logger.warning("Inner model already a PeftModel — skipping load to avoid stacking.")
        return
    matcher._embedder[0].auto_model = PeftModel.from_pretrained(inner, adapter_path)
    logger.info("PeftModel attached: peft_config keys=%s",
                list(matcher._embedder[0].auto_model.peft_config.keys()))


def _gold_direction_counts(reference: Alignment) -> dict[str, int]:
    """Count normalized gold relations. Lets us cross-check against the
    relation audit (e.g. g7 = 18 '<' / 52 '>')."""
    counts = {"<": 0, ">": 0, "=": 0, "dropped": 0}
    for cor in reference:
        norm = _normalize_relation(cor.relation)
        counts[norm if norm is not None else "dropped"] += 1
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description="Bidirectional pipeline validation run (single config).")
    p.add_argument("--model", default="llama-embed-nemotron-8b",
                   help="Model alias or HF id. Aliases: " + ", ".join(MODEL_ALIASES))
    p.add_argument("--lora-adapter-nemo", default="lora_adapters/nemo_subsumption_lora_extracted",
                   help="Path to the Nemo LoRA adapter dir. Pass 'none' to run without LoRA.")
    p.add_argument("--description", default="description_path_context",
                   help="RDFGraphWrapper description method (Lever A = path_context).")
    p.add_argument("--asym-template-id", default=SUBB_DEFAULT_ASYM[1],
                   help="Asym instruction template id (Lever B OFF -> SUBB_DEFAULT_ASYM = T1).")
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--top-k-max", type=int, default=20)
    p.add_argument("--kg-format", default="turtle")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-group", default=None)
    args = p.parse_args()

    Path("results").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    )

    sha, dirty, dirty_paths = _git_sha_and_dirty()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_group = args.wandb_group or f"validation_bidirectional_{ts}_{sha}"

    use_lora = args.lora_adapter_nemo and args.lora_adapter_nemo.lower() != "none"
    if use_lora and not Path(args.lora_adapter_nemo).is_dir():
        sys.exit(f"ERROR: LoRA adapter dir not found: {args.lora_adapter_nemo}")

    _set_seeds(args.seed)
    device = _detect_device()
    resolved_model = _resolve_model(args.model)
    broader_instr, narrower_instr = get_subb_asym_templates(args.asym_template_id)

    # ── Config proof header (analogous to the SLURM job pin proof) ──────────
    logger.info("=" * 78)
    logger.info("BIDIRECTIONAL VALIDATION RUN  group=%s", wandb_group)
    logger.info("git SHA=%s dirty=%s  device=%s  seed=%d", sha, dirty, device, args.seed)
    if dirty:
        logger.warning("Working tree DIRTY:\n%s", dirty_paths)
    logger.info("model=%s  (resolved=%s)", args.model, resolved_model)
    logger.info("Lever A = path_context -> description=%s", args.description)
    logger.info("matcher = MatcherAsymmetricRetrieval (emits '<' AND '>')")
    logger.info("Lever C = OFF (no RRF fusion)")
    logger.info("Lever B = OFF -> SUBB_DEFAULT_ASYM=%s, asym_template_id=%s",
                SUBB_DEFAULT_ASYM, args.asym_template_id)
    logger.info("LoRA = %s", args.lora_adapter_nemo if use_lora else "<none>")
    logger.info("top_k=%d  datasets=%s", args.top_k_max, args.datasets)
    logger.info("=" * 78)

    wandb = None
    if args.wandb:
        import wandb as _wandb
        wandb = _wandb
    project = args.wandb_project or "beyondequivalence-retrieval-stage1"

    # ── Build the matcher ONCE; warm across all datasets (same model + adapter,
    #    only the KG changes per dataset). ──────────────────────────────────
    matcher = MatcherAsymmetricRetrieval(
        model=resolved_model,
        broader_query_instruction=broader_instr,
        narrower_query_instruction=narrower_instr,
        document_instruction="",
        description=args.description,
        top_k=args.top_k_max,
        kg_format=args.kg_format,
    )
    if use_lora:
        logger.info("Loading LoRA adapter: %s", args.lora_adapter_nemo)
        _attach_lora(matcher, args.lora_adapter_nemo)

    K = 20
    rows: list[dict] = []
    tot = {"hits_sub": 0, "n_sub": 0, "hits_sup": 0, "n_sup": 0}

    try:
        for dataset in args.datasets:
            _set_seeds(args.seed)
            src_path, tgt_path, ref_path = load_subdataset(dataset)
            kg_source, source_labels = _load_kg_with_labels(src_path)
            kg_target, target_labels = _load_kg_with_labels(tgt_path)
            reference = Alignment(str(ref_path))
            gold = _gold_direction_counts(reference)

            t0 = time.perf_counter()
            predictions = matcher.match(kg_source, kg_target, Alignment(), parameters={})
            t_elapsed = time.perf_counter() - t0

            report = compute_recall_at_k(
                reference, predictions, k_values=(1, 5, 10, K),
                source_labels=source_labels, target_labels=target_labels,
            )
            r20_sub = report.recall_at_k["per_relation_strict"]["subclass"].get(K, 0.0)
            r20_sup = report.recall_at_k["per_relation_strict"]["superclass"].get(K, 0.0)
            n_sub, n_sup = gold["<"], gold[">"]
            hits_sub = round(r20_sub * n_sub)
            hits_sup = round(r20_sup * n_sup)

            tot["hits_sub"] += hits_sub; tot["n_sub"] += n_sub
            tot["hits_sup"] += hits_sup; tot["n_sup"] += n_sup

            row = {
                "dataset": dataset,
                "n_sub": n_sub, "n_sup": n_sup, "n_eq": gold["="], "n_dropped": gold["dropped"],
                "r20_sub": r20_sub, "r20_sup": r20_sup,
                "hits_sub": hits_sub, "hits_sup": hits_sup,
                "align_size": len(predictions), "runtime_s": round(t_elapsed, 1),
            }
            rows.append(row)
            logger.info(
                "[%s] R@20 subclass=%.4f (%d/%d) | superclass=%.4f (%d/%d) | align=%d | %.1fs",
                dataset, r20_sub, hits_sub, n_sub, r20_sup, hits_sup, n_sup,
                len(predictions), t_elapsed,
            )

            if wandb is not None:
                run = wandb.init(
                    project=project, group=wandb_group, reinit=True,
                    name=f"valbi_{_alias_for_naming(args.model)}_{dataset}_{sha}",
                    tags=["phase:validation", "axis:bidirectional",
                          f"dataset:{dataset}", f"lora:{'on' if use_lora else 'off'}"],
                    config={
                        "git_sha": sha, "git_dirty": dirty, "model_resolved": resolved_model,
                        "description": args.description, "matcher": "AsymmetricRetrieval",
                        "fusion": False, "asym_template_id": args.asym_template_id,
                        "lora_adapter": args.lora_adapter_nemo if use_lora else None,
                        "dataset": dataset, "top_k_max": args.top_k_max, "seed": args.seed,
                        "cluster": os.getenv("CLUSTER", ""),
                    },
                )
                wandb.log({
                    "per_relation_strict/subclass/R@20": r20_sub,
                    "per_relation_strict/superclass/R@20": r20_sup,
                    "gold/n_subclass": n_sub, "gold/n_superclass": n_sup,
                    "hits/subclass@20": hits_sub, "hits/superclass@20": hits_sup,
                })
                run.finish()
    finally:
        if getattr(matcher, "_embedder", None) is not None:
            matcher._embedder = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Results table ───────────────────────────────────────────────────────
    agg_sub = (tot["hits_sub"] / tot["n_sub"]) if tot["n_sub"] else 0.0
    agg_sup = (tot["hits_sup"] / tot["n_sup"]) if tot["n_sup"] else 0.0

    print("\n" + "=" * 96)
    print("BIDIRECTIONAL VALIDATION — R@20 per_relation_strict (subclass='<', superclass='>')")
    print("=" * 96)
    hdr = f"{'dataset':<15} | {'gold <':>6} {'gold >':>6} | {'R@20 <':>8} {'hits<':>7} | {'R@20 >':>8} {'hits>':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['dataset']:<15} | {r['n_sub']:>6} {r['n_sup']:>6} | "
              f"{r['r20_sub']:>8.4f} {r['hits_sub']:>3}/{r['n_sub']:<3} | "
              f"{r['r20_sup']:>8.4f} {r['hits_sup']:>3}/{r['n_sup']:<3}")
    print("-" * len(hdr))
    print(f"{'AGGREGATE':<15} | {tot['n_sub']:>6} {tot['n_sup']:>6} | "
          f"{agg_sub:>8.4f} {tot['hits_sub']:>3}/{tot['n_sub']:<3} | "
          f"{agg_sup:>8.4f} {tot['hits_sup']:>3}/{tot['n_sup']:<3}")
    print("=" * 96)

    # ── Sanity verdict ────────────────────────────────────────────────────────
    if tot["hits_sup"] == 0:
        logger.error(
            "SANITY FAIL: aggregate superclass R@20 = 0.000 (0/%d). The narrower "
            "pass emitted no '>' or it did not reach the metric. STOP and inspect.",
            tot["n_sup"],
        )
        sys.exit(2)
    logger.info("SANITY OK: aggregate superclass R@20 = %.4f (%d/%d) >> 0.",
                agg_sup, tot["hits_sup"], tot["n_sup"])


if __name__ == "__main__":
    main()
