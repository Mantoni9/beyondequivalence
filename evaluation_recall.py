"""
Recall@K and MRR evaluation for the BeyondEquivalence subsumption study.

Lives next to the existing Evaluation.py but does NOT touch it: the equivalence
pipeline keeps using Evaluation.evaluate (P/R/F1 over (s,t,relation) triples),
and the subsumption runner uses compute_recall_at_k from this module.

Three scoring modes are computed unconditionally for every run:
  - strict:              hit iff target appears in top-K of the source's full
                         (cross-relation) ranking AND predicted relation matches
                         gold.
  - lax:                 hit iff target appears in top-K of the source's full
                         ranking (relation ignored).
  - per_relation_strict: hit iff target appears in top-K of the source's
                         relation-restricted sub-ranking. For a reference
                         (s, t, '<'), only '<'-predictions for s are ranked and
                         t's position is checked against K. For (s, t, '>'),
                         analogously over '>'-predictions. This avoids the
                         score-comparability assumption between asymmetric runs
                         (broader vs. narrower may have systematically different
                         score distributions, in which case one run dominates the
                         joint top-K and the other falls out).

The strict and lax modes are reported per relation type (equivalence / subclass /
superclass) and aggregated as 'all'. The per_relation_strict mode is reported
ONLY for subclass and superclass — equivalence is omitted because there is no
relation-specific sub-list for '=' in the asymmetric setting, and 'all' is
omitted because the per-relation sub-rankings are not meaningfully aggregable.

In the symmetric setting (all predictions carry '='), per_relation_strict for
subclass / superclass is structurally 0 and matches strict for those relations.

MRR is reported in the same breakdown for all three modes.

Reference Alignments may contain Unicode relation strings (≡, ≥, ≤, ⊑, ⊒).
They are normalised to {'=', '>', '<'}. Anything not in RELATION_NORMALIZATION
(notably ≃ overlap and ⊘ disjoint) is dropped and counted in the report.

Predictions are expected to use ASCII relations '=', '<', or '>' — that is what
MatcherEmbeddingRetrieval / MatcherAsymmetricRetrieval emit.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from Alignment import Alignment

logger = logging.getLogger(__name__)

# Reference relation strings -> ASCII canonical form. Anything not here is dropped
# and counted in dropped_relations_breakdown.
#
# NOTE on what is *not* in this table — verified empirically on Apr 2026 against
# all STROMA/TaSeR sub-datasets (g1-web, g2-diseases, g3-text, g5-groceries,
# g7-literature) plus product-classification (gpc-unspsc, etim-eclass, eclass-gpc)
# in docs/relations_per_dataset.md:
#
#   PartOf  — methodically in scope per the thesis proposal (asymmetric meronymy
#             relation alongside ⊑/⊒). Empirically present in only 3 mappings
#             across all 5 STROMA/TaSeR cases (n=1 in g7, n=2 in g1-web), which
#             is too sparse for statistically meaningful per-relation Recall@K.
#             The exclusion is data-driven (n ≤ 3), NOT methodology-driven.
#             Added to RELATION_NORMALIZATION once a Reference with sufficient
#             PartOf annotations becomes available.
#
#   ~       — OAEI ASCII shorthand for ≃ (Overlap / approximate match). Dominates
#             the product-classification datasets (~85-90% of mappings on
#             gpc-unspsc and eclass-gpc) but is absent from STROMA/TaSeR.
#             Stage 3 of the thesis (product-classification) must decide before
#             its first run whether to add a fourth recall breakdown for Overlap
#             or to evaluate only on the {=,<,>} subset.
#
#   HasA, Related — small counts (single digits) on g1-web, g2-diseases,
#             g5-groceries; same data-sparsity rationale as PartOf.
RELATION_NORMALIZATION: dict[str, str] = {
    "=":  "=",
    "≡":  "=",
    ">":  ">",
    "≥":  ">",
    "⊒":  ">",
    "<":  "<",
    "≤":  "<",
    "⊑":  "<",
}

# ASCII relation → human-friendly label used in metric names.
RELATION_LABELS: dict[str, str] = {
    "=": "equivalence",
    "<": "subclass",     # source is more specific than target (s ⊑ t)
    ">": "superclass",   # source is more general than target (s ⊒ t)
}

DEFAULT_K_VALUES: tuple[int, ...] = (1, 5, 10, 20)


def _normalize_relation(rel: str) -> Optional[str]:
    if not rel:
        return None
    return RELATION_NORMALIZATION.get(rel.strip())


@dataclass
class RecallReport:
    # {mode: {relation_label: {k: recall}}}
    recall_at_k: dict
    # {mode: {relation_label: mrr}}
    mrr: dict

    n_reference_total: int
    n_reference_after_filter: int
    dropped_relations_count: int
    dropped_relations_breakdown: dict[str, int]

    # Long-format rows for the per-source W&B table / TSV. One dict per prediction.
    per_source_rows: list[dict]

    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    relation_labels_present: tuple[str, ...] = field(default_factory=tuple)

    def to_wandb_metrics(self) -> dict:
        """Flatten the nested dicts into wandb-friendly slash-separated keys."""
        out: dict = {}
        for mode, by_rel in self.recall_at_k.items():
            for rel_label, by_k in by_rel.items():
                for k, v in by_k.items():
                    out[f"recall_at_k_{mode}/{rel_label}/k={k}"] = v
        for mode, by_rel in self.mrr.items():
            for rel_label, v in by_rel.items():
                out[f"mrr_{mode}/{rel_label}"] = v
        out["dropped_relations_count"] = self.dropped_relations_count
        for raw_rel, count in self.dropped_relations_breakdown.items():
            out[f"dropped_relations/{raw_rel}"] = count
        out["n_reference_total"] = self.n_reference_total
        out["n_reference_after_filter"] = self.n_reference_after_filter
        return out


def compute_recall_at_k(
    reference: Alignment,
    predictions: Alignment,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    source_labels: Optional[dict[str, str]] = None,
    target_labels: Optional[dict[str, str]] = None,
) -> RecallReport:
    """Compute strict + lax Recall@K and MRR, broken down by ASCII relation type.

    Predictions are ranked per source by (-confidence, target, relation) — confidence
    descending, then deterministic tie-break by target URI then relation. Ranks are
    1-based. K-window is per-source; if a source has fewer than K predictions, the
    smaller list is used as-is (no padding).
    """
    k_values = tuple(k_values)
    source_labels = source_labels or {}
    target_labels = target_labels or {}

    # 1) Normalise reference and bin by source.
    reference_by_source: dict[str, list[tuple[str, str, str]]] = {}
    dropped: Counter = Counter()
    n_total = 0
    n_kept = 0
    for cor in reference:
        n_total += 1
        norm = _normalize_relation(cor.relation)
        if norm is None:
            raw = cor.relation.strip() if cor.relation else ""
            dropped[raw] += 1
            continue
        n_kept += 1
        reference_by_source.setdefault(cor.source, []).append(
            (cor.target, norm, cor.relation.strip()),
        )

    if dropped:
        logger.warning(
            "Dropped %d reference correspondence(s) with non-evaluable relations: %s",
            sum(dropped.values()), dict(dropped),
        )

    # 2) Bin predictions by source and rank deterministically. Two indices are
    #    built: the full per-source ranking (used by strict + lax) and a
    #    per-source-per-relation ranking (used by per_relation_strict).
    ranked_by_source: dict[str, list] = {}
    ranked_by_src_rel: dict[tuple[str, str], list] = {}
    for cor in predictions:
        ranked_by_source.setdefault(cor.source, []).append(cor)
        ranked_by_src_rel.setdefault((cor.source, cor.relation), []).append(cor)
    for lst in ranked_by_source.values():
        lst.sort(key=lambda c: (-c.confidence, c.target, c.relation))
    for lst in ranked_by_src_rel.values():
        lst.sort(key=lambda c: (-c.confidence, c.target, c.relation))

    # 3) Compute per-relation stats.
    #    strict / lax: aggregated into the "all" bucket and into per-label buckets.
    #    per_relation_strict: ONLY into per-label buckets, and only for subclass /
    #    superclass — see module docstring for why equivalence and "all" are
    #    omitted.
    per_label_stats: dict[str, dict[str, dict]] = {}
    PER_RELATION_LABELS = ("subclass", "superclass")

    def _ensure(label: str) -> None:
        if label not in per_label_stats:
            per_label_stats[label] = {
                "strict": {"hits_at_k": {k: 0 for k in k_values}, "rrs": [], "n": 0},
                "lax":    {"hits_at_k": {k: 0 for k in k_values}, "rrs": [], "n": 0},
            }
            if label in PER_RELATION_LABELS:
                per_label_stats[label]["per_relation_strict"] = {
                    "hits_at_k": {k: 0 for k in k_values}, "rrs": [], "n": 0,
                }

    _ensure("all")
    for rel_label in RELATION_LABELS.values():
        _ensure(rel_label)

    first_hit_strict_by_src: dict[str, Optional[int]] = {}
    first_hit_lax_by_src:    dict[str, Optional[int]] = {}

    for src, refs in reference_by_source.items():
        ranked = ranked_by_source.get(src, [])

        # Full per-source position lookups (for strict and lax).
        target_positions_lax: dict[str, int] = {}
        target_rel_positions_strict: dict[tuple[str, str], int] = {}
        for rank_zero, cor in enumerate(ranked):
            rank = rank_zero + 1
            target_positions_lax.setdefault(cor.target, rank)
            target_rel_positions_strict.setdefault((cor.target, cor.relation), rank)

        # Per-source-per-relation position lookups (for per_relation_strict).
        # pos_per_rel[rel_norm][target_uri] = 1-based rank in the rel-restricted
        # sub-list for this source.
        pos_per_rel: dict[str, dict[str, int]] = {}
        for rel_norm in ("=", "<", ">"):
            sub = ranked_by_src_rel.get((src, rel_norm), [])
            d: dict[str, int] = {}
            for rank_zero, cor in enumerate(sub):
                d.setdefault(cor.target, rank_zero + 1)
            pos_per_rel[rel_norm] = d

        src_first_strict: Optional[int] = None
        src_first_lax:    Optional[int] = None

        for tgt, rel_norm, _raw_rel in refs:
            label = RELATION_LABELS[rel_norm]

            # lax
            rank_lax = target_positions_lax.get(tgt)
            for bucket in (label, "all"):
                stats = per_label_stats[bucket]["lax"]
                stats["n"] += 1
                if rank_lax is not None:
                    stats["rrs"].append(1.0 / rank_lax)
                    for k in k_values:
                        if rank_lax <= k:
                            stats["hits_at_k"][k] += 1
                else:
                    stats["rrs"].append(0.0)
            if rank_lax is not None:
                src_first_lax = rank_lax if src_first_lax is None else min(src_first_lax, rank_lax)

            # strict
            rank_strict = target_rel_positions_strict.get((tgt, rel_norm))
            for bucket in (label, "all"):
                stats = per_label_stats[bucket]["strict"]
                stats["n"] += 1
                if rank_strict is not None:
                    stats["rrs"].append(1.0 / rank_strict)
                    for k in k_values:
                        if rank_strict <= k:
                            stats["hits_at_k"][k] += 1
                else:
                    stats["rrs"].append(0.0)
            if rank_strict is not None:
                src_first_strict = rank_strict if src_first_strict is None else min(src_first_strict, rank_strict)

            # per_relation_strict — only for subclass / superclass refs.
            # No "all" aggregation, no equivalence bucket.
            if label in PER_RELATION_LABELS:
                rank_pr = pos_per_rel.get(rel_norm, {}).get(tgt)
                stats = per_label_stats[label]["per_relation_strict"]
                stats["n"] += 1
                if rank_pr is not None:
                    stats["rrs"].append(1.0 / rank_pr)
                    for k in k_values:
                        if rank_pr <= k:
                            stats["hits_at_k"][k] += 1
                else:
                    stats["rrs"].append(0.0)

        first_hit_strict_by_src[src] = src_first_strict
        first_hit_lax_by_src[src]    = src_first_lax

    # 4) Roll up to recall_at_k + mrr dicts. per_relation_strict only carries
    #    subclass + superclass entries; the dict shape stays consistent so
    #    downstream code (W&B flattening) can iterate uniformly.
    recall_at_k: dict = {"strict": {}, "lax": {}, "per_relation_strict": {}}
    mrr:         dict = {"strict": {}, "lax": {}, "per_relation_strict": {}}
    for label, modes in per_label_stats.items():
        for mode in ("strict", "lax", "per_relation_strict"):
            if mode not in modes:
                continue
            n = modes[mode]["n"]
            recall_at_k[mode][label] = {
                k: (modes[mode]["hits_at_k"][k] / n) if n > 0 else 0.0 for k in k_values
            }
            mrr[mode][label] = (sum(modes[mode]["rrs"]) / n) if n > 0 else 0.0

    # 5) Long-format per-source rows. One row per ranked prediction.
    gold_pairs_lax: set[tuple[str, str]] = set()
    gold_st_to_relset: dict[tuple[str, str], set[str]] = {}
    for src, refs in reference_by_source.items():
        for tgt, rel_norm, _raw in refs:
            gold_pairs_lax.add((src, tgt))
            gold_st_to_relset.setdefault((src, tgt), set()).add(rel_norm)

    per_source_rows: list[dict] = []
    for src, ranked in ranked_by_source.items():
        for rank_zero, cor in enumerate(ranked):
            rank = rank_zero + 1
            gold_rels = gold_st_to_relset.get((cor.source, cor.target), set())
            if not gold_rels:
                gold_relation = ""
            elif len(gold_rels) == 1:
                gold_relation = next(iter(gold_rels))
            else:
                gold_relation = "/".join(sorted(gold_rels))
            per_source_rows.append({
                "source_uri": cor.source,
                "source_label": source_labels.get(cor.source, ""),
                "rank": rank,
                "target_uri": cor.target,
                "target_label": target_labels.get(cor.target, ""),
                "score": float(cor.confidence),
                "predicted_relation": cor.relation,
                "gold_relation": gold_relation,
                "is_correct_strict": cor.relation in gold_rels,
                "is_correct_lax": (cor.source, cor.target) in gold_pairs_lax,
                "first_hit_at_k_strict": first_hit_strict_by_src.get(cor.source),
                "first_hit_at_k_lax":    first_hit_lax_by_src.get(cor.source),
            })

    relation_labels_present = tuple(sorted(
        label for label in per_label_stats
        if label != "all" and per_label_stats[label]["lax"]["n"] > 0
    ))

    return RecallReport(
        recall_at_k=recall_at_k,
        mrr=mrr,
        n_reference_total=n_total,
        n_reference_after_filter=n_kept,
        dropped_relations_count=sum(dropped.values()),
        dropped_relations_breakdown=dict(dropped),
        per_source_rows=per_source_rows,
        k_values=k_values,
        relation_labels_present=relation_labels_present,
    )
