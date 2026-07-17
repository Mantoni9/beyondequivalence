"""
ablation_bidirectional.py — main A x B ablation over the DIRECTED gold, on the
validated bidirectional scaffold.

Built on scripts/validate_bidirectional.py: same MatcherAsymmetricRetrieval
(emits BOTH '<' and '>'), same PeftModel.from_pretrained LoRA attach, same
per-direction evaluation. Deliberately NOT the old --ablation-sweep, which uses
the '<'-only MatcherBidirectionalConsolidation (Lever C, cross-direction RRF)
and the silent load_adapter no-op.

Matrix (96 runs):
  Lever A : {turtle -> description_one_gen, path_context -> description_path_context}
  Lever B : {default -> T1 (SUBB_DEFAULT_ASYM), sub_b_pin -> T2 (SUBB_PIN_ASYM)}
            -> 4 perms: baseline (turtle,T1), A (pc,T1), B (turtle,T2), A+B (pc,T2)
  Models  : {qwen3-embedding-8b, llama-embed-nemotron-8b}
  LoRA    : {off, on} per model
  Datasets: all 6 STROMA/TaSeR sub-datasets
  -> 4 x 2 x 2 x 6 = 96 runs.  No Lever C.  top-50, seed 42.

A controls the description method; B controls the instruction template only
(get_subb_asym_templates). Deep top-50 ranked lists are persisted per run
(predictions.tsv) so scripts/fuse_crossmodel_rrf.py can do post-hoc cross-model
RRF without re-running any model.

Additive: imports the frozen Stage-1 matchers/evaluator and the validated
validate_bidirectional helpers; modifies nothing under Stage 1.

Exit codes: 0 = ok; 2 = a run produced superclass R@20 = 0 on a dataset that has
'>' gold (signals the '>' path silently broke mid-sweep).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from dotenv import load_dotenv
load_dotenv()

# scripts/ is on sys.path[0] when run as `python scripts/<file>.py`; add repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_subsumption_experiment import (
    _resolve_model, _alias_for_naming, _load_kg_with_labels,
    _set_seeds, _git_sha_and_dirty, _detect_device,
)
from validate_bidirectional import _attach_lora, _gold_direction_counts
from Alignment import Alignment
from tracks.zenodo_loader import load_subdataset
from evaluation_recall import compute_recall_at_k
from MatcherAsymmetricRetrieval import MatcherAsymmetricRetrieval
from prompt import get_subb_asym_templates
from subB_pinned_config import SUBB_DEFAULT_ASYM, SUBB_PIN_ASYM

logger = logging.getLogger("ablation_bidirectional")

DEFAULT_DATASETS = (
    "mouse-human", "g1-web", "g2-diseases", "g3-text", "g5-groceries", "g7-literature",
)

# Lever A: (label, description method). A controls the description.
A_VALUES = (("turtle", "description_one_gen"), ("path_context", "description_path_context"))
# Lever B: (label, template_id). B controls the instruction template only.
B_VALUES = (("default", SUBB_DEFAULT_ASYM[1]), ("sub_b_pin", SUBB_PIN_ASYM[1]))


def _write_predictions_tsv(path: Path, predictions: Alignment) -> None:
    rows = sorted(
        ((c.source, c.target, c.relation, float(c.confidence)) for c in predictions),
        key=lambda r: (r[0], -r[3], r[2], r[1]),
    )
    with path.open("w", encoding="utf-8") as f:
        f.write("source_uri\ttarget_uri\trelation\tscore\n")
        for s, t, rel, sc in rows:
            f.write(f"{s}\t{t}\t{rel}\t{sc:.6f}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Main A x B bidirectional ablation (96 runs).")
    p.add_argument("--lora-adapter-qwen3", default="lora_adapters/qwen3_subsumption_lora_extracted")
    p.add_argument("--lora-adapter-nemo", default="lora_adapters/nemo_subsumption_lora_extracted")
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--top-k-max", type=int, default=50)
    p.add_argument("--kg-format", default="turtle")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--results-root", default="results")
    # Grid filters (additive; defaults reproduce the full 96-run grid). Used to
    # run single frozen-config cells on new datasets (e.g. the VDI->eBay gold
    # case: --models qwen3-embedding-8b --lora-modes off
    #       --A-labels path_context --B-labels sub_b_pin).
    p.add_argument("--models", nargs="+",
                   default=["qwen3-embedding-8b", "llama-embed-nemotron-8b"],
                   choices=["qwen3-embedding-8b", "llama-embed-nemotron-8b"])
    p.add_argument("--lora-modes", nargs="+", default=["off", "on"],
                   choices=["off", "on"])
    p.add_argument("--A-labels", nargs="+", default=[a[0] for a in A_VALUES],
                   choices=[a[0] for a in A_VALUES])
    p.add_argument("--B-labels", nargs="+", default=[b[0] for b in B_VALUES],
                   choices=[b[0] for b in B_VALUES])
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    )
    Path(args.results_root).mkdir(parents=True, exist_ok=True)

    sha, dirty, dirty_paths = _git_sha_and_dirty()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    group = args.wandb_group or f"main_ablation_bidirectional_{ts}_{sha}"

    # Resolve adapters per model. Hard-fail if a requested adapter is missing.
    adapters = {
        "qwen3-embedding-8b": args.lora_adapter_qwen3,
        "llama-embed-nemotron-8b": args.lora_adapter_nemo,
    }
    for model, ap in adapters.items():
        if model in args.models and "on" in args.lora_modes and not Path(ap).is_dir():
            sys.exit(f"ERROR: LoRA adapter dir for {model} not found: {ap}")

    _set_seeds(args.seed)
    device = _detect_device()

    logger.info("=" * 84)
    logger.info("MAIN A x B BIDIRECTIONAL ABLATION  group=%s", group)
    logger.info("git SHA=%s dirty=%s  device=%s  seed=%d", sha, dirty, device, args.seed)
    if dirty:
        logger.warning("Working tree DIRTY:\n%s", dirty_paths)
    logger.info("matcher=MatcherAsymmetricRetrieval ('<' AND '>')  NO Lever C  top_k=%d", args.top_k_max)
    logger.info("Lever A: %s", [a[0] for a in A_VALUES])
    logger.info("Lever B: default->%s (SUBB_DEFAULT_ASYM), sub_b_pin->%s (SUBB_PIN_ASYM)",
                SUBB_DEFAULT_ASYM[1], SUBB_PIN_ASYM[1])
    logger.info("Models x LoRA: qwen3{off,on}, nemo{off,on}   datasets=%s", args.datasets)
    logger.info("=" * 84)

    wandb = None
    if args.wandb:
        import wandb as _wandb
        wandb = _wandb
    project = args.wandb_project or "beyondequivalence-retrieval-stage1"

    n_done = 0
    failures: list[str] = []
    sweep_t0 = time.perf_counter()

    # Outer: (model, lora_state) -> one embedder load per block (4 loads total).
    for model in args.models:
        resolved = _resolve_model(model)
        alias = _alias_for_naming(model)
        for lora_on in tuple("on" == m for m in args.lora_modes):
            lora_tag = "lora-on" if lora_on else "lora-off"
            adapter = adapters[model] if lora_on else None

            # Build matcher once; warm across all perms x datasets in this block.
            matcher = MatcherAsymmetricRetrieval(
                model=resolved,
                broader_query_instruction="", narrower_query_instruction="",
                document_instruction="", description="description_one_gen",
                top_k=args.top_k_max, kg_format=args.kg_format,
            )
            matcher._ensure_embedder()
            if lora_on:
                logger.info("[%s/%s] attaching LoRA: %s", alias, lora_tag, adapter)
                _attach_lora(matcher, adapter)

            try:
                for dataset in args.datasets:
                    src_path, tgt_path, ref_path = load_subdataset(dataset)
                    kg_source, source_labels = _load_kg_with_labels(src_path)
                    kg_target, target_labels = _load_kg_with_labels(tgt_path)
                    reference = Alignment(str(ref_path))
                    gold = _gold_direction_counts(reference)

                    for a_label, desc_method in A_VALUES:
                        if a_label not in args.A_labels:
                            continue
                        for b_label, template_id in B_VALUES:
                            if b_label not in args.B_labels:
                                continue
                            _set_seeds(args.seed)
                            broader_instr, narrower_instr = get_subb_asym_templates(template_id)
                            matcher.description = desc_method
                            matcher.broader_query_instruction = broader_instr
                            matcher.narrower_query_instruction = narrower_instr

                            run_name = f"ablbi_{alias}_{lora_tag}_A-{a_label}_B-{b_label}_{dataset}_{sha}"
                            out_dir = Path(args.results_root) / run_name
                            out_dir.mkdir(parents=True, exist_ok=True)

                            t0 = time.perf_counter()
                            predictions = matcher.match(kg_source, kg_target, Alignment(), parameters={})
                            t_elapsed = time.perf_counter() - t0

                            report = compute_recall_at_k(
                                reference, predictions, k_values=(1, 5, 10, 20),
                                source_labels=source_labels, target_labels=target_labels,
                            )
                            prs = report.recall_at_k["per_relation_strict"]
                            n_sub, n_sup = gold["<"], gold[">"]
                            r = {
                                "sub": {k: prs["subclass"].get(k, 0.0) for k in (10, 20)},
                                "sup": {k: prs["superclass"].get(k, 0.0) for k in (10, 20)},
                            }
                            hits = {
                                "sub": {k: round(r["sub"][k] * n_sub) for k in (10, 20)},
                                "sup": {k: round(r["sup"][k] * n_sup) for k in (10, 20)},
                            }

                            config_dump = {
                                "git_sha": sha, "model": model, "model_alias": alias,
                                "lora": lora_tag, "lora_adapter": adapter,
                                "A": a_label, "B": b_label, "description": desc_method,
                                "template_id": template_id, "dataset": dataset,
                                "top_k_max": args.top_k_max, "seed": args.seed,
                                "matcher": "MatcherAsymmetricRetrieval", "fusion": False,
                                "wandb_group": group, "run_name": run_name,
                            }
                            (out_dir / "config.json").write_text(json.dumps(config_dump, indent=2))
                            (out_dir / "metrics.json").write_text(json.dumps({
                                "recall_at_k": report.recall_at_k, "mrr": report.mrr,
                                "gold": {"subclass": n_sub, "superclass": n_sup, "equivalence": gold["="]},
                                "r10_r20": r, "hits_at_k": hits,
                                "n_reference_total": report.n_reference_total,
                                "n_reference_after_filter": report.n_reference_after_filter,
                                "runtime_seconds": t_elapsed,
                            }, indent=2))
                            _write_predictions_tsv(out_dir / "predictions.tsv", predictions)

                            logger.info(
                                "[%s] R@20 sub=%.4f (%d/%d) sup=%.4f (%d/%d) | R@10 sub=%.4f sup=%.4f | %.1fs",
                                run_name, r["sub"][20], hits["sub"][20], n_sub,
                                r["sup"][20], hits["sup"][20], n_sup,
                                r["sub"][10], r["sup"][10], t_elapsed,
                            )
                            if n_sup > 0 and hits["sup"][20] == 0:
                                failures.append(run_name)
                                logger.error("SUSPECT: %s has '>' gold (n=%d) but superclass hits@20=0", run_name, n_sup)

                            if wandb is not None:
                                wrun = wandb.init(
                                    project=project, group=group, reinit=True, name=run_name,
                                    tags=["phase:main-ablation", "axis:AxB", f"A:{a_label}", f"B:{b_label}",
                                          f"model:{alias}", f"lora:{lora_tag}", f"dataset:{dataset}"],
                                    config=config_dump,
                                )
                                wandb.log({
                                    "per_relation_strict/subclass/R@10": r["sub"][10],
                                    "per_relation_strict/subclass/R@20": r["sub"][20],
                                    "per_relation_strict/superclass/R@10": r["sup"][10],
                                    "per_relation_strict/superclass/R@20": r["sup"][20],
                                    "gold/n_subclass": n_sub, "gold/n_superclass": n_sup,
                                    "hits/subclass@20": hits["sub"][20], "hits/superclass@20": hits["sup"][20],
                                })
                                wrun.finish()
                            n_done += 1
            finally:
                if getattr(matcher, "_embedder", None) is not None:
                    matcher._embedder = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    logger.info("Ablation done in %.1fs. runs=%d  suspect_runs=%d",
                time.perf_counter() - sweep_t0, n_done, len(failures))
    if failures:
        logger.error("Runs with '>' gold but 0 superclass hits@20: %s", failures)
        sys.exit(2)


if __name__ == "__main__":
    main()
