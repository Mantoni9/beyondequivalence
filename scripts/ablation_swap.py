"""
ablation_swap.py — Stage-1 swapped-retrieval ablation runner (superclass
recall ceiling fix). Frozen verbalization/template throughout: A=path_context,
B=sub_b_pin (T2), MatcherAsymmetricRetrieval, top-50, seed 42.

One run per (config, dataset) produces ALL FOUR passes by invoking the
UNMODIFIED frozen matcher twice (swap_retrieval.run_all_passes):
  match(S, T) -> s_broader ('<'), s_narrower ('>')
  match(T, S) -> t_broader, t_narrower — re-oriented to canonical
                 (source, target) orientation: (t, s', '<') -> (s', t, '>').
The ablation variants are offline pass subsets (swap_retrieval.VARIANTS):
  baseline = s_broader + s_narrower             (the frozen d11c97e pipeline)
  v_sym    = s_broader + t_broader              (PRIMARY: both directions fan-in)
  v_3pass  = s_broader + s_narrower + t_broader (amendment 2026-06-12: keeps
             the s_narrower cross-rescue that v_sym structurally loses)
  v_union  = all four                           (recall ceiling)

Grid: {qwen3-noLoRA (primary), nemo+LoRA (robustness side-run — does NOT
reopen the model freeze)} x 6 datasets = 12 runs.

Per run dir (results/swap_{alias}_{lora}_{dataset}_{sha}/):
  config.json, metrics.json, passes.tsv (all four passes, canonical
  orientation, per-(pass, query) ranks — schema contract in the TSV header).

metrics.json per run:
  - variants.{baseline,v_sym,v_union}: pair coverage per relation (<, >, =)
    at budget K in {5,10,20,50} + candidate-pair volume per K.
  - per_directed_query: recall@K/MRR with '<' gold ranked in s-broader lists
    and '>' gold in t-broader lists (definition in evaluation_recall).
  - legacy_per_relation_strict: compute_recall_at_k on the baseline subset —
    direct comparability with the d11c97e tables.
  - volume: |S|, |T|, per-pass pair counts and pairwise pass overlaps @20.
  - provenance_crosstab_at_20: which pass combination found each gold pair,
    by gold relation. LOG ONLY — no claims derived here.
  - identity_check: the s-side passes must reproduce the d11c97e
    predictions.tsv for the same (model, lora, dataset) exactly (6-dp scores).

Exit codes (checked in this priority order, all AFTER every run has written
its artifacts): 3 = identity check mismatch OR skipped while enabled (s-side
passes not verified against d11c97e — comparability broken/unproven);
2 = a run with '>' gold got v_sym superclass coverage@20 of 0 (swap path
silently broke); 4 = pooled guard violation (a swap variant's pooled
subclass- or equivalence-coverage@20 dropped > 0.02 vs the same run's
pooled baseline — the pre-registered guard). NOTE: a v_sym '<'-guard trip
is expected-possible per the amendment (it quantifies the s_narrower
cross-rescue) — it is a FINDING to report, not a code failure; artifacts
are always fully persisted before any gate decides the exit code.
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
from itertools import combinations
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
from Correspondence import Correspondence
from tracks.zenodo_loader import load_subdataset
from evaluation_recall import (
    _normalize_relation,
    compute_pair_coverage,
    compute_per_directed_query_recall,
    compute_recall_at_k,
)
from MatcherAsymmetricRetrieval import MatcherAsymmetricRetrieval
from prompt import get_subb_asym_templates
from subB_pinned_config import SUBB_PIN_ASYM
from swap_retrieval import (
    VARIANTS,
    PassRow,
    assemble_variant,
    candidate_pairs_at_budget,
    run_all_passes,
    write_passes_tsv,
)

logger = logging.getLogger("ablation_swap")

DEFAULT_DATASETS = (
    "mouse-human", "g1-web", "g2-diseases", "g3-text", "g5-groceries", "g7-literature",
)

# config name -> (model alias for _resolve_model, lora_on)
CONFIG_GRID = {
    "qwen3-noLoRA": ("qwen3-embedding-8b", False),       # PRIMARY (frozen Stage-1 model)
    "nemo+LoRA":    ("llama-embed-nemotron-8b", True),   # robustness side-run only
    "sbert-noLoRA": ("sbert", False),                    # local smoke / plumbing check
}

BUDGET_KS = (5, 10, 20, 50)
EVAL_KS = (1, 5, 10, 20, 50)
VOLUME_K = 20

# Pre-registered guards: pooled coverage@20 of the swap variants may not drop
# more than this vs the pooled baseline of the SAME run, for '<' and '='.
# Superclass is the PRIMARY outcome, never a guard. v_3pass registered by
# pre-amendment 2026-06-12, same guards/thresholds as the other variants.
GUARD_MAX_DROP = 0.02
GUARD_RELATIONS = ("subclass", "equivalence")
GUARD_VARIANTS = ("v_sym", "v_3pass", "v_union")


def _pool_coverage(acc: dict, cfg_name: str, variant: str, cov_at_k: dict) -> None:
    """Accumulate {covered, n} per (config, variant, relation) across datasets —
    pooled coverage is sum(covered)/sum(n), micro-pooled like the d11c97e tables."""
    for rel in ("subclass", "superclass", "equivalence"):
        bucket = acc.setdefault((cfg_name, variant, rel), {"covered": 0, "n": 0})
        bucket["covered"] += cov_at_k[rel]["covered"]
        bucket["n"] += cov_at_k[rel]["n"]


def _guard_violations(acc: dict, configs: list[str]) -> list[str]:
    """Pre-registered guard: pooled '<'/'=' coverage@20 drop > GUARD_MAX_DROP
    vs baseline. Returns human-readable violation strings (empty = all pass)."""
    out: list[str] = []
    for cfg_name in configs:
        for variant in GUARD_VARIANTS:
            for rel in GUARD_RELATIONS:
                base = acc.get((cfg_name, "baseline", rel))
                var = acc.get((cfg_name, variant, rel))
                if not base or not var or not base["n"] or not var["n"]:
                    continue
                drop = base["covered"] / base["n"] - var["covered"] / var["n"]
                if drop > GUARD_MAX_DROP:
                    out.append(
                        f"{cfg_name}/{variant}/{rel}: pooled coverage@{VOLUME_K} "
                        f"drop {drop:.4f} > {GUARD_MAX_DROP} "
                        f"(baseline {base['covered']}/{base['n']}, "
                        f"variant {var['covered']}/{var['n']})")
    return out


def _query_lists(rows: list[PassRow], *, retrieved: str) -> dict[str, list[str]]:
    """Per-query ranked URI lists of ONE pass (rank order, ties already settled)."""
    by_query: dict[str, list[str]] = {}
    for r in sorted(rows, key=lambda r: (r.query_uri, r.rank)):
        by_query.setdefault(r.query_uri, []).append(
            r.source_uri if retrieved == "source" else r.target_uri,
        )
    return by_query


def _volume_stats(passes: dict[str, list[PassRow]], n_source_classes: int,
                  n_target_classes: int, k: int = VOLUME_K) -> dict:
    per_pass_pairs = {pid: candidate_pairs_at_budget(rows, k) for pid, rows in passes.items()}
    return {
        "n_source_classes": n_source_classes,
        "n_target_classes": n_target_classes,
        f"pairs_at_{k}_per_pass": {pid: len(p) for pid, p in sorted(per_pass_pairs.items())},
        f"pass_pair_overlap_at_{k}": {
            f"{a}+{b}": len(per_pass_pairs[a] & per_pass_pairs[b])
            for a, b in combinations(sorted(per_pass_pairs), 2)
        },
    }


def _provenance_crosstab(passes: dict[str, list[PassRow]], reference: Alignment,
                         k: int = VOLUME_K) -> dict:
    """Which pass combination found each gold pair at budget k, by gold relation.
    Log only — the no-direction-signal question is settled post-hoc, not here."""
    pair_to_passes: dict[tuple[str, str], set[str]] = {}
    for pass_id, rows in passes.items():
        for r in rows:
            if r.rank <= k:
                pair_to_passes.setdefault((r.source_uri, r.target_uri), set()).add(pass_id)
    crosstab: dict[str, dict[str, int]] = {}
    for cor in reference:
        norm = _normalize_relation(cor.relation)
        if norm is None:
            continue
        found = pair_to_passes.get((cor.source, cor.target))
        bucket = "+".join(sorted(found)) if found else "none"
        crosstab.setdefault(bucket, {"<": 0, ">": 0, "=": 0})[norm] += 1
    return crosstab


def _identity_check(passes: dict[str, list[PassRow]], ablbi_dir: Path,
                    legacy_prs: dict | None = None) -> dict:
    """Two-layer comparability gate against the frozen d11c97e run dir:
    (1) the s-side passes must reproduce its predictions.tsv exactly (same
    row set at that file's 6-dp score convention); (2) the recomputed legacy
    per_relation_strict recalls must equal its stored metrics.json values
    exactly — this second layer also catches sub-1e-6 score drift that
    rank-flips inside 6-dp tie clusters, which layer (1) cannot see."""
    ablbi_tsv = ablbi_dir / "predictions.tsv"
    if not ablbi_tsv.is_file():
        return {"status": "skipped", "reason": f"not found: {ablbi_tsv}"}
    old_rows: set[tuple[str, str, str, str]] = set()
    old_scores: dict[tuple[str, str, str], float] = {}
    with ablbi_tsv.open(encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            s, t, rel, sc = line.rstrip("\n").split("\t")
            old_rows.add((s, t, rel, sc))
            old_scores[(s, t, rel)] = float(sc)
    new_rows = {
        (r.source_uri, r.target_uri, r.relation, f"{r.score:.6f}")
        for r in assemble_variant(passes, "baseline")
    }
    rows_ok = new_rows == old_rows

    metrics_check = "absent"
    metric_max_abs_diff = None
    ablbi_metrics = ablbi_dir / "metrics.json"
    if legacy_prs is not None and ablbi_metrics.is_file():
        stored = json.loads(ablbi_metrics.read_text())["recall_at_k"]["per_relation_strict"]
        metric_max_abs_diff = 0.0
        for rel in ("subclass", "superclass"):
            for k_str, stored_val in stored.get(rel, {}).items():
                ours = legacy_prs.get(rel, {}).get(int(k_str))
                if ours is not None:
                    metric_max_abs_diff = max(metric_max_abs_diff, abs(ours - stored_val))
        metrics_check = "ok" if metric_max_abs_diff == 0.0 else "mismatch"

    if rows_ok and metrics_check in ("ok", "absent"):
        return {"status": "ok", "n_rows": len(new_rows), "metrics_check": metrics_check}

    new_scores = {(r.source_uri, r.target_uri, r.relation): r.score
                  for r in assemble_variant(passes, "baseline")}
    common = set(old_scores) & set(new_scores)
    max_delta = max((abs(old_scores[key] - new_scores[key]) for key in common), default=None)
    return {
        "status": "mismatch",
        "rows_ok": rows_ok,
        "metrics_check": metrics_check,
        "metric_max_abs_diff": metric_max_abs_diff,
        "n_old": len(old_rows), "n_new": len(new_rows),
        "n_only_old": len({r[:3] for r in old_rows} - set(new_scores)),
        "n_only_new": len(set(new_scores) - set(old_scores)),
        "n_common_keys": len(common),
        "max_score_delta_on_common": max_delta,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Swapped-retrieval ablation (all four passes per run).")
    p.add_argument("--configs", nargs="+", choices=sorted(CONFIG_GRID),
                   default=["qwen3-noLoRA", "nemo+LoRA"])
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--lora-adapter-qwen3", default="lora_adapters/qwen3_subsumption_lora_extracted")
    p.add_argument("--lora-adapter-nemo", default="lora_adapters/nemo_subsumption_lora_extracted")
    p.add_argument("--top-k-max", type=int, default=50)
    p.add_argument("--kg-format", default="turtle")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--identity-sha", default="d11c97e",
                   help="SHA suffix of the ablbi_* run dirs used for the identity check.")
    p.add_argument("--no-identity-check", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--results-root", default="results")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    )
    if args.top_k_max < max(BUDGET_KS):
        sys.exit(f"ERROR: --top-k-max {args.top_k_max} < max budget K {max(BUDGET_KS)} — "
                 f"@{max(BUDGET_KS)} numbers would be computed on truncated lists.")
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    sha, dirty, dirty_paths = _git_sha_and_dirty()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    group = args.wandb_group or f"swap_ablation_{ts}_{sha}"

    adapters = {
        "qwen3-embedding-8b": args.lora_adapter_qwen3,
        "llama-embed-nemotron-8b": args.lora_adapter_nemo,
    }
    for cfg_name in args.configs:
        model, lora_on = CONFIG_GRID[cfg_name]
        if lora_on and not Path(adapters[model]).is_dir():
            sys.exit(f"ERROR: LoRA adapter dir for {model} not found: {adapters[model]}")

    _set_seeds(args.seed)
    device = _detect_device()
    template_id = SUBB_PIN_ASYM[1]
    broader_instr, narrower_instr = get_subb_asym_templates(template_id)

    logger.info("=" * 84)
    logger.info("SWAPPED-RETRIEVAL ABLATION  group=%s", group)
    logger.info("git SHA=%s dirty=%s  device=%s  seed=%d", sha, dirty, device, args.seed)
    if dirty:
        logger.warning("Working tree DIRTY:\n%s", dirty_paths)
    logger.info("frozen levers: A=path_context  B=sub_b_pin(%s)  top_k=%d", template_id, args.top_k_max)
    logger.info("passes per run: s_broader s_narrower t_broader t_narrower (2 matcher.match calls)")
    logger.info("variants (offline subsets): baseline | v_sym | v_union")
    logger.info("configs=%s  datasets=%s", args.configs, args.datasets)
    logger.info("=" * 84)

    wandb = None
    if args.wandb:
        import wandb as _wandb
        wandb = _wandb
    project = args.wandb_project or "beyondequivalence-retrieval-stage1"

    n_done = 0
    suspect_runs: list[str] = []
    identity_failures: list[str] = []
    identity_skipped: list[str] = []
    pooled_acc: dict = {}
    sweep_t0 = time.perf_counter()

    for cfg_name in args.configs:
        model, lora_on = CONFIG_GRID[cfg_name]
        resolved = _resolve_model(model)
        alias = _alias_for_naming(model)
        lora_tag = "lora-on" if lora_on else "lora-off"
        adapter = adapters.get(model) if lora_on else None

        matcher = MatcherAsymmetricRetrieval(
            model=resolved,
            broader_query_instruction=broader_instr,
            narrower_query_instruction=narrower_instr,
            document_instruction="",
            description="description_path_context",
            top_k=args.top_k_max,
            kg_format=args.kg_format,
        )
        matcher._ensure_embedder()
        if lora_on:
            logger.info("[%s] attaching LoRA: %s", cfg_name, adapter)
            _attach_lora(matcher, adapter)

        try:
            for dataset in args.datasets:
                _set_seeds(args.seed)
                src_path, tgt_path, ref_path = load_subdataset(dataset)
                kg_source, _ = _load_kg_with_labels(src_path)
                kg_target, _ = _load_kg_with_labels(tgt_path)
                reference = Alignment(str(ref_path))
                gold = _gold_direction_counts(reference)

                run_name = f"swap_{alias}_{lora_tag}_{dataset}_{sha}"
                out_dir = results_root / run_name
                out_dir.mkdir(parents=True, exist_ok=True)

                t0 = time.perf_counter()
                passes = run_all_passes(matcher, kg_source, kg_target)
                t_match = time.perf_counter() - t0

                # --- variant metrics: pair coverage + volume per budget K ---
                variants: dict[str, dict] = {}
                for variant in VARIANTS:
                    rows = assemble_variant(passes, variant)
                    cov, n_pairs = {}, {}
                    for k in BUDGET_KS:
                        pair_set = candidate_pairs_at_budget(rows, k)
                        cov[k] = compute_pair_coverage(reference, pair_set)
                        n_pairs[k] = len(pair_set)
                    variants[variant] = {"pair_coverage": cov, "pairs_at_budget": n_pairs}
                    _pool_coverage(pooled_acc, cfg_name, variant, cov[VOLUME_K])

                # --- per_directed_query (pass-level, variant-independent) ---
                pdq = compute_per_directed_query_recall(
                    reference,
                    _query_lists(passes["s_broader"], retrieved="target"),
                    _query_lists(passes["t_broader"], retrieved="source"),
                    k_values=EVAL_KS,
                )

                # --- legacy per-source modes on the baseline subset ---
                # NOTE: compute_recall_at_k's strict/lax modes internally rank
                # s_broader + s_narrower jointly by raw score (cross-pass score
                # sort) — they are intentionally DISCARDED below; only the
                # within-pass per_relation_strict values are persisted.
                legacy_alignment = Alignment()
                for r in assemble_variant(passes, "baseline"):
                    legacy_alignment.add(
                        Correspondence(r.source_uri, r.target_uri, r.relation, r.score))
                legacy = compute_recall_at_k(reference, legacy_alignment, k_values=EVAL_KS)

                volume = _volume_stats(passes, len(kg_source.get_classes()),
                                       len(kg_target.get_classes()))
                crosstab = _provenance_crosstab(passes, reference)

                identity = {"status": "disabled"}
                if not args.no_identity_check:
                    ablbi_dir = (results_root /
                                 f"ablbi_{alias}_{lora_tag}_A-path_context_B-sub_b_pin_"
                                 f"{dataset}_{args.identity_sha}")
                    identity = _identity_check(
                        passes, ablbi_dir, legacy.recall_at_k["per_relation_strict"])
                    if identity["status"] == "mismatch":
                        identity_failures.append(run_name)
                        logger.error("IDENTITY MISMATCH [%s]: %s", run_name, identity)
                    elif identity["status"] == "skipped":
                        identity_skipped.append(run_name)
                        logger.error("IDENTITY SKIPPED [%s]: %s — comparability "
                                     "unverified (use --no-identity-check to allow)",
                                     run_name, identity.get("reason"))

                config_dump = {
                    "git_sha": sha, "model": model, "model_alias": alias,
                    "config": cfg_name, "lora": lora_tag, "lora_adapter": adapter,
                    "A": "path_context", "B": "sub_b_pin",
                    "description": "description_path_context", "template_id": template_id,
                    "dataset": dataset, "top_k_max": args.top_k_max, "seed": args.seed,
                    "matcher": "MatcherAsymmetricRetrieval (direct + transposed)",
                    "passes": sorted(passes), "wandb_group": group, "run_name": run_name,
                }
                (out_dir / "config.json").write_text(json.dumps(config_dump, indent=2))
                (out_dir / "metrics.json").write_text(json.dumps({
                    "gold": {"subclass": gold["<"], "superclass": gold[">"],
                             "equivalence": gold["="]},
                    "variants": variants,
                    "per_directed_query": {k: v for k, v in pdq.items() if k != "k_values"},
                    "legacy_per_relation_strict":
                        legacy.recall_at_k["per_relation_strict"],
                    "legacy_mrr_per_relation_strict": legacy.mrr["per_relation_strict"],
                    "volume": volume,
                    "provenance_crosstab_at_20": crosstab,
                    "identity_check": identity,
                    "n_reference_total": legacy.n_reference_total,
                    "n_reference_after_filter": legacy.n_reference_after_filter,
                    "runtime_seconds": {"all_passes": t_match,
                                        "total": time.perf_counter() - t0},
                }, indent=2))
                write_passes_tsv(out_dir / "passes.tsv",
                                 [r for rows in passes.values() for r in rows])

                cov_sym = variants["v_sym"]["pair_coverage"][VOLUME_K]
                cov_base = variants["baseline"]["pair_coverage"][VOLUME_K]
                logger.info(
                    "[%s] >cov@20 base=%.4f v_sym=%.4f | <cov@20 base=%.4f v_sym=%.4f | "
                    "=cov@20 v_sym=%.4f | pairs@20 base=%d v_sym=%d | identity=%s | %.1fs",
                    run_name,
                    cov_base["superclass"]["coverage"] or 0.0,
                    cov_sym["superclass"]["coverage"] or 0.0,
                    cov_base["subclass"]["coverage"] or 0.0,
                    cov_sym["subclass"]["coverage"] or 0.0,
                    cov_sym["equivalence"]["coverage"] or 0.0,
                    variants["baseline"]["pairs_at_budget"][VOLUME_K],
                    variants["v_sym"]["pairs_at_budget"][VOLUME_K],
                    identity["status"], t_match,
                )
                if gold[">"] > 0 and cov_sym["superclass"]["covered"] == 0:
                    suspect_runs.append(run_name)
                    logger.error("SUSPECT: %s has '>' gold (n=%d) but v_sym superclass "
                                 "coverage@20 = 0", run_name, gold[">"])

                if wandb is not None:
                    wrun = wandb.init(
                        project=project, group=group, reinit=True, name=run_name,
                        tags=["phase:swap-ablation", f"config:{cfg_name}", f"model:{alias}",
                              f"lora:{lora_tag}", f"dataset:{dataset}"],
                        config=config_dump,
                    )
                    log_payload = {
                        "per_directed_query/subclass/R@20": pdq["recall_at_k"]["subclass"][20],
                        "per_directed_query/superclass/R@20": pdq["recall_at_k"]["superclass"][20],
                        "legacy_prs/subclass/R@20": legacy.recall_at_k["per_relation_strict"]["subclass"][20],
                        "legacy_prs/superclass/R@20": legacy.recall_at_k["per_relation_strict"]["superclass"][20],
                        "gold/n_subclass": gold["<"], "gold/n_superclass": gold[">"],
                        "gold/n_equivalence": gold["="],
                        # 1 ok / 0 mismatch / -1 skipped / -2 disabled
                        "identity_check_ok": {"ok": 1, "mismatch": 0, "skipped": -1,
                                              "disabled": -2}[identity["status"]],
                    }
                    for variant in VARIANTS:
                        for rel in ("subclass", "superclass", "equivalence"):
                            val = variants[variant]["pair_coverage"][VOLUME_K][rel]["coverage"]
                            if val is not None:
                                log_payload[f"coverage_{variant}/{rel}@{VOLUME_K}"] = val
                        log_payload[f"volume/{variant}_pairs@{VOLUME_K}"] = \
                            variants[variant]["pairs_at_budget"][VOLUME_K]
                    wandb.log(log_payload)
                    wrun.finish()
                n_done += 1
        finally:
            if getattr(matcher, "_embedder", None) is not None:
                matcher._embedder = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- pooled summary (micro-pooled across datasets, per config x variant) ---
    pooled_summary: dict[str, dict[str, dict[str, dict]]] = {}
    for (cfg_name, variant, rel), bucket in sorted(pooled_acc.items()):
        cov = (bucket["covered"] / bucket["n"]) if bucket["n"] else None
        pooled_summary.setdefault(cfg_name, {}).setdefault(variant, {})[rel] = {
            **bucket, "coverage": cov,
        }
        logger.info("POOLED coverage@%d  %s/%s/%s = %s  (%d/%d)", VOLUME_K,
                    cfg_name, variant, rel,
                    f"{cov:.4f}" if cov is not None else "n/a",
                    bucket["covered"], bucket["n"])
    summary_path = results_root / f"{group}_pooled_summary.json"
    summary_path.write_text(json.dumps(pooled_summary, indent=2))
    logger.info("pooled summary written: %s", summary_path)

    guard_violations = _guard_violations(pooled_acc, args.configs)
    logger.info("Swap ablation done in %.1fs. runs=%d  suspect=%d  identity_failures=%d  "
                "identity_skipped=%d  guard_violations=%d",
                time.perf_counter() - sweep_t0, n_done, len(suspect_runs),
                len(identity_failures), len(identity_skipped), len(guard_violations))
    if identity_failures or identity_skipped:
        if identity_failures:
            logger.error("Runs FAILING the d11c97e identity check: %s", identity_failures)
        if identity_skipped:
            logger.error("Runs with SKIPPED identity check (artifacts missing while the "
                         "check was enabled): %s", identity_skipped)
        sys.exit(3)
    if suspect_runs:
        logger.error("Runs with '>' gold but v_sym coverage@20 = 0: %s", suspect_runs)
        sys.exit(2)
    if guard_violations:
        for v in guard_violations:
            logger.error("GUARD VIOLATION: %s", v)
        sys.exit(4)


if __name__ == "__main__":
    main()
