"""
Multi-class evaluation for the Stage-2 relation classifier.

Sits next to Evaluation.py (binary equivalence P/R/F1) and
evaluation_recall.py (Stage-1 Recall@K / MRR). Computes the artefacts
the Stage-2 smoke needs:

  - 4x4 confusion matrix over {=, <, >, none}
  - Per-class precision / recall / F1 for the three primary classes
    {=, <, >}, plus macro- and micro-F1 across those three.
  - Direction accuracy: of the gold subclass/superclass refs where the
    reranker predicted EITHER direction (<,< or <,> or >,< or >,>), how
    often was the direction correct?

Conventions
-----------
- Gold relations are normalised via ``evaluation_recall._normalize_relation``;
  anything not in ``RELATION_NORMALIZATION`` is dropped and counted (same as
  Stage-1 eval).
- Predicted relations are folded to the 4-class display space {=, <, >, none}
  by ``_fold_pred_relation``: 'partof', '', or any unexpected string maps to
  'none'. The 4x4 matrix axes therefore share the same label set.
- 'partof' is excluded from the displayed matrix per the data-sparsity
  rationale in evaluation_recall.py:62-68 (n <= 3 across all STROMA/TaSeR
  cases). Its raw count is still tracked under ``predicted_partof_count``
  so a future Stage-3 product-classification run can re-introduce it.

Universe of (s, t) pairs scored
-------------------------------
- If ``candidate_pairs`` is provided (the deduplicated Stage-1 candidate set
  fed to the reranker), the universe is ``candidate_pairs ∪ gold_pairs``.
  This separates 'reranker said none' (in candidates, not in predictions)
  from 'Stage-1 miss' (gold pair never reached the reranker) — the latter
  count is reported as ``n_gold_not_in_candidates``.
- If ``candidate_pairs`` is None, the universe is ``pred_pairs ∪ gold_pairs``;
  the report cannot distinguish the two miss categories.

For the smoke run, pass candidate_pairs so the report is interpretable.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from Alignment import Alignment
from evaluation_recall import _normalize_relation


DISPLAY_LABELS:    tuple[str, ...] = ("=", "<", ">", "none")
PRIMARY_CLASSES:   tuple[str, ...] = ("=", "<", ">")
PARTOF_LABEL = "partof"


def _fold_pred_relation(rel: str) -> str:
    """Map a Correspondence.relation emitted by MatcherSubsumptionReranker
    onto the 4-class display space.

    Returns one of {'=', '<', '>', 'none'}. 'partof' folds to 'none'.
    """
    rel = (rel or "").strip()
    if rel in ("=", "<", ">"):
        return rel
    return "none"


@dataclass
class MultiClassReport:
    # confusion[gold_label][pred_label] = count, over DISPLAY_LABELS x DISPLAY_LABELS.
    confusion: dict[str, dict[str, int]]
    # per_class[label] = {precision, recall, f1, tp, fp, fn, support}
    per_class: dict[str, dict[str, float]]

    macro_f1:        float
    micro_precision: float
    micro_recall:    float
    micro_f1:        float

    # Of the gold subclass/superclass refs that the reranker predicted as
    # one of {<, >}, what fraction got the direction right?
    direction_accuracy: Optional[float]
    direction_correct:  int
    direction_swapped:  int

    n_universe:                  int
    n_candidates_total:          int
    n_reference_total:           int
    n_reference_after_filter:    int
    dropped_relations_count:     int
    dropped_relations_breakdown: dict[str, int]
    n_gold_not_in_candidates:    int       # Stage-1 misses
    predicted_partof_count:      int       # folded to 'none' in the displayed matrix
    predicted_partof_pairs:      list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "confusion":          self.confusion,
            "per_class":          self.per_class,
            "macro_f1":           self.macro_f1,
            "micro_precision":    self.micro_precision,
            "micro_recall":       self.micro_recall,
            "micro_f1":           self.micro_f1,
            "direction_accuracy": self.direction_accuracy,
            "direction_correct":  self.direction_correct,
            "direction_swapped":  self.direction_swapped,
            "n_universe":                  self.n_universe,
            "n_candidates_total":          self.n_candidates_total,
            "n_reference_total":           self.n_reference_total,
            "n_reference_after_filter":    self.n_reference_after_filter,
            "dropped_relations_count":     self.dropped_relations_count,
            "dropped_relations_breakdown": self.dropped_relations_breakdown,
            "n_gold_not_in_candidates":    self.n_gold_not_in_candidates,
            "predicted_partof_count":      self.predicted_partof_count,
        }


def compute_multiclass_metrics(
    reference: Alignment,
    predictions: Alignment,
    candidate_pairs: Optional[set[tuple[str, str]]] = None,
) -> MultiClassReport:
    """Compute the 4x4 confusion matrix and derived metrics.

    Parameters
    ----------
    reference:        gold Alignment (relations normalised by
                      ``evaluation_recall._normalize_relation``).
    predictions:      Stage-2 reranker output (Correspondences with
                      relation in {'=', '<', '>', 'partof'}).
    candidate_pairs:  optional deduplicated (source, target) candidate set
                      fed to the reranker. Used to separate "reranker said
                      none" from "Stage-1 miss" in the report.
    """
    # --- 1. Normalise gold; bin by (source, target). --------------------
    gold: dict[tuple[str, str], str] = {}
    dropped: Counter = Counter()
    n_ref_total = 0
    n_ref_kept = 0
    for cor in reference:
        n_ref_total += 1
        norm = _normalize_relation(cor.relation)
        if norm is None:
            raw = (cor.relation or "").strip()
            dropped[raw] += 1
            continue
        # If gold contains duplicate (s,t) with different rels, keep the
        # first; eval logic below treats one relation per pair.
        gold.setdefault((cor.source, cor.target), norm)
        n_ref_kept += 1

    # --- 2. Predictions: fold to {=, <, >, none}; collect partof. -------
    pred: dict[tuple[str, str], str] = {}
    partof_pairs: list[tuple[str, str]] = []
    for cor in predictions:
        key = (cor.source, cor.target)
        raw = (cor.relation or "").strip()
        if raw == PARTOF_LABEL:
            partof_pairs.append(key)
        pred[key] = _fold_pred_relation(raw)

    # --- 3. Build the universe of pairs to score. -----------------------
    if candidate_pairs is not None:
        universe: set[tuple[str, str]] = set(candidate_pairs) | set(gold.keys())
        n_candidates_total = len(candidate_pairs)
    else:
        universe = set(pred.keys()) | set(gold.keys())
        n_candidates_total = len(pred)

    n_gold_not_in_candidates = 0
    if candidate_pairs is not None:
        for key in gold:
            if key not in candidate_pairs:
                n_gold_not_in_candidates += 1

    # --- 4. 4x4 confusion matrix. ---------------------------------------
    cm: dict[str, dict[str, int]] = {
        g: {p: 0 for p in DISPLAY_LABELS} for g in DISPLAY_LABELS
    }
    for key in universe:
        g_label = gold.get(key, "none")
        p_label = pred.get(key, "none")
        cm[g_label][p_label] += 1

    # --- 5. Per-class P/R/F1 over PRIMARY_CLASSES. ----------------------
    per_class: dict[str, dict[str, float]] = {}
    for c in PRIMARY_CLASSES:
        tp = cm[c][c]
        fp = sum(cm[g][c] for g in DISPLAY_LABELS if g != c)
        fn = sum(cm[c][p] for p in DISPLAY_LABELS if p != c)
        support = sum(cm[c][p] for p in DISPLAY_LABELS)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[c] = {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "support": support,
        }

    macro_f1 = sum(per_class[c]["f1"] for c in PRIMARY_CLASSES) / len(PRIMARY_CLASSES)

    tp_sum = sum(per_class[c]["tp"] for c in PRIMARY_CLASSES)
    fp_sum = sum(per_class[c]["fp"] for c in PRIMARY_CLASSES)
    fn_sum = sum(per_class[c]["fn"] for c in PRIMARY_CLASSES)
    micro_precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    micro_recall    = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )

    # --- 6. Direction accuracy: <-> swap rate among directional preds. --
    direction_correct = cm["<"]["<"] + cm[">"][">"]
    direction_swapped = cm["<"][">"] + cm[">"]["<"]
    denom = direction_correct + direction_swapped
    direction_accuracy: Optional[float] = (
        direction_correct / denom if denom > 0 else None
    )

    return MultiClassReport(
        confusion=cm,
        per_class=per_class,
        macro_f1=macro_f1,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        direction_accuracy=direction_accuracy,
        direction_correct=direction_correct,
        direction_swapped=direction_swapped,
        n_universe=len(universe),
        n_candidates_total=n_candidates_total,
        n_reference_total=n_ref_total,
        n_reference_after_filter=n_ref_kept,
        dropped_relations_count=sum(dropped.values()),
        dropped_relations_breakdown=dict(dropped),
        n_gold_not_in_candidates=n_gold_not_in_candidates,
        predicted_partof_count=len(partof_pairs),
        predicted_partof_pairs=partof_pairs,
    )


def format_confusion_matrix_tsv(cm: dict[str, dict[str, int]]) -> str:
    """Return a TSV string with header gold\\pred + columns in DISPLAY_LABELS order."""
    header = ["gold\\pred"] + list(DISPLAY_LABELS)
    rows = [header]
    for g in DISPLAY_LABELS:
        row = [g] + [str(cm[g][p]) for p in DISPLAY_LABELS]
        rows.append(row)
    return "\n".join("\t".join(r) for r in rows) + "\n"
