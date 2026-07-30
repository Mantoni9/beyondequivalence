"""
closure_credit.py — pure hierarchy-credit primitives for the Closure re-scoring
(P3 / thesis 4.1.3). No I/O, stdlib only, unit-testable.

Credit rule (Kiritchenko-style, variant (i), per Antonio's 2026-07-07 spec):
  - SAME-LABEL only. A '<'-prediction (e1,t') credits a '<'-gold (e1,t) iff the
    predicted target t' is an ANCESTOR-or-self of the gold target t in the
    TARGET ontology (t ⊑ t'). A '>'-prediction credits a '>'-gold iff t' is a
    DESCENDANT-or-self of t. '=' and 'none' are untouched (strict).
  - Only the target side varies; the source e1 is fixed.
  - EXISTENTIAL set semantics, no 1:1 matching: a prediction is precision-
    credited if ∃ a gold pair it coarsens; a gold pair is recall-credited if
    ∃ a prediction that coarsens it (one prediction may recall-cover many gold).
  - Closure is the REFLEXIVE-transitive subClassOf hull ⇒ exact hits are the
    t'=t special case, so credited ⊇ strict (strict is credit on an edgeless
    hierarchy). This gives the acceptance property used by the driver's
    identity check (strict == a24e146).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def build_closure(subclass_edges: Iterable[tuple[str, str]]):
    """subclass_edges: (child, parent) rdfs:subClassOf pairs (parent = more
    general). Returns (ancestors, descendants), each dict class -> frozenset
    INCLUDING self (reflexive-transitive). ancestors[c] = c + all superclasses;
    descendants[c] = c + all subclasses."""
    parents: dict[str, set] = defaultdict(set)   # child -> direct parents
    children: dict[str, set] = defaultdict(set)  # parent -> direct children
    nodes: set = set()
    for child, parent in subclass_edges:
        parents[child].add(parent)
        children[parent].add(child)
        nodes.add(child)
        nodes.add(parent)

    def _reach(start: str, adj: dict) -> frozenset:
        seen = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return frozenset(seen)

    ancestors = {n: _reach(n, parents) for n in nodes}
    descendants = {n: _reach(n, children) for n in nodes}
    return ancestors, descendants


def _credit_set(gold_t: str, closure: dict) -> frozenset:
    """Target classes that credit a gold target `gold_t`: ancestors-or-self
    (rel '<') or descendants-or-self (rel '>'). If gold_t is absent from the
    ontology graph (isolated), only the exact target credits it."""
    return closure.get(gold_t, frozenset({gold_t}))


def credited_direction(preds: dict, gold: dict, closure: dict, rel: str,
                       universe: Iterable[tuple[str, str]]) -> dict:
    """Credited vs strict P/R/F1 for one directional class rel in {'<','>'}.

    preds/gold: dict (source, target) -> relation. `closure` = ancestors for
    rel='<' or descendants for rel='>' (reflexive). `universe`: the (s,t) pairs
    to score (conditional or e2e — caller decides). Returns strict + credited
    counts/P/R/F1, the '<'-FP resolution numbers (K1), and the set of FP pairs
    that flip to credited (byproduct #5)."""
    U = set(universe)
    pred_pairs = [(s, t) for (s, t) in U if preds.get((s, t)) == rel]
    gold_pairs = [(s, t) for (s, t) in U if gold.get((s, t)) == rel]

    gold_t_by_src: dict[str, set] = defaultdict(set)
    for (s, t) in gold_pairs:
        gold_t_by_src[s].add(t)
    pred_t_by_src: dict[str, set] = defaultdict(set)
    for (s, t) in pred_pairs:
        pred_t_by_src[s].add(t)

    # Precision side: a predicted (s, tp) is credited-correct iff some gold
    # (s, tg) of the same rel is coarsened by tp (tp ∈ credit_set(tg)).
    credited_preds: set = set()
    strict_preds: set = set()
    for (s, tp) in pred_pairs:
        if gold.get((s, tp)) == rel:                 # exact strict hit
            strict_preds.add((s, tp))
            credited_preds.add((s, tp))
            continue
        for tg in gold_t_by_src.get(s, ()):
            if tp in _credit_set(tg, closure):
                credited_preds.add((s, tp))
                break

    # Recall side (existential): a gold (s, tg) is credited-recalled iff some
    # predicted (s, tp) of the same rel coarsens it.
    credited_gold: set = set()
    strict_gold: set = set()
    for (s, tg) in gold_pairs:
        if preds.get((s, tg)) == rel:
            strict_gold.add((s, tg))
            credited_gold.add((s, tg))
            continue
        cset = _credit_set(tg, closure)
        for tp in pred_t_by_src.get(s, ()):
            if tp in cset:
                credited_gold.add((s, tg))
                break

    n_pred, n_gold = len(pred_pairs), len(gold_pairs)

    def _prf(tp_p: int, tp_r: int) -> tuple[float, float, float]:
        p = tp_p / n_pred if n_pred else 0.0
        r = tp_r / n_gold if n_gold else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    sP, sR, sF = _prf(len(strict_preds), len(strict_gold))
    cP, cR, cF = _prf(len(credited_preds), len(credited_gold))
    strict_fp = n_pred - len(strict_preds)
    credited_fp = n_pred - len(credited_preds)
    resolved = strict_fp - credited_fp
    return {
        "rel": rel,
        "n_pred": n_pred,
        "n_gold": n_gold,
        "strict":   {"precision": sP, "recall": sR, "f1": sF,
                     "tp_prec": len(strict_preds), "tp_rec": len(strict_gold)},
        "credited": {"precision": cP, "recall": cR, "f1": cF,
                     "tp_prec": len(credited_preds), "tp_rec": len(credited_gold)},
        "strict_fp": strict_fp,
        "credited_fp": credited_fp,
        "fp_resolved": resolved,
        "fp_resolved_frac": (resolved / strict_fp) if strict_fp else 0.0,
        "flip_pairs": sorted(credited_preds - strict_preds),
    }
