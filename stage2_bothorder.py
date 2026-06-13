"""
stage2_bothorder.py — Stufe-B Both-Order-Voting: order-invariant directional
classification for the Stage-2 reranker.

Motivation (Stufe A, H-position, clean): Llama's subclass-prior is POSITIONAL
— it labels the first-presented concept as the subclass, content-independent.
This is the Reversal Curse (Berglund et al. 2023): models that learn "A is B"
fail to infer "B is A". The remedy here is ORDER-INVARIANCE: query each pair
in both argument orders and reconcile, so the positional bias cancels instead
of flipping sign.

Inference (compute once): for each candidate (s, t), query the LLM in
  order AB = present (s, t)   → canonical label is as-emitted
  order BA = present (t, s)   → canonical label is the A2-inverted label
The BA inversion reuses prompt.relation_for_canonical_pair (Stufe-A, tested
exactly-once). Both raw labels and both answer-span logprobs are persisted;
the three reconciliation variants are OFFLINE recombinations of this one
double-order run.

Reconciliation truth tables (canonical predictions pred_AB, pred_BA, each in
{<, >, =, none}; partof / none / parse_fail fold to 'none'):

  Derivation (verified on the Stufe-A named-26, 24/26): a purely positional
  model gives the SAME raw label ('subclass') in both frames → canonical
  (<,>) DISAGREEMENT. A content-tracking model flips its raw label with the
  swap → canonical (<,<) AGREEMENT. Therefore canonical agreement = the
  model tracked content (trust the direction); canonical directional
  disagreement = same raw label both ways = X⊑Y ∧ X⊒Y = mutual subsumption.

  cell                         B1 (abstain)   B3 (symmetry)   B2 (confidence)
  (<,<) / (>,>)  agreement     that dir       that dir        that dir
  (=,=)          agreement     =              =               =
  (none,none)    agreement     none           none            none
  (<,>) / (>,<)  dir-conflict  none           =               tie-break*
  (<,=)/(>,=)/(=,<)/(=,>)
                 dir-vs-equiv  none           none            tie-break*
  (dir,none) / (none,dir)      none           none            tie-break*
  (=,none) / (none,=)          none           none            tie-break*
  * B2: on any disagreement the frame with the higher answer-span mean
    logprob wins (emit that frame's canonical label); ties → AB.

B1 and B3 differ ONLY in the (<,>)/(>,<) directional-conflict cell
(none vs '='). The dir-vs-equiv cells are a named registered row: neither a
clean agreement nor a clean directional disagreement → none (B1/B3) /
tie-break (B2). The named-26's 2 outliers live exactly here (canonical (<,=)).
"""

from __future__ import annotations

import math
from typing import Optional

from Alignment import Alignment
from prompt import parse_relation_label
from MatcherSubsumptionReranker import relation_for_canonical_pair

RECON_VARIANTS = ("B1", "B2", "B3")

# canonical relations after folding the non-scored labels to 'none'.
_DIR = ("<", ">")


def canonical_for_order(raw_label: str, *, order: str) -> str:
    """Map a parsed raw label to the canonical (s, t) relation for the given
    presentation order. AB = as-emitted; BA = A2-inverted (reuse the tested
    exactly-once inversion). partof / none / parse_fail fold to 'none'."""
    if order == "AB":
        rel = relation_for_canonical_pair(raw_label, swapped=False)
    elif order == "BA":
        rel = relation_for_canonical_pair(raw_label, swapped=True)
    else:
        raise ValueError(f"order must be 'AB' or 'BA', got {order!r}")
    return rel if rel in ("<", ">", "=") else "none"


def reconcile(ab: str, ba: str, *, variant: str,
              ab_lp: Optional[float] = None, ba_lp: Optional[float] = None) -> str:
    """Reconcile the two canonical predictions per the registered truth table."""
    if variant == "B1":
        return ab if ab == ba else "none"
    if variant == "B3":
        if ab == ba:
            return ab
        if {ab, ba} == {"<", ">"}:
            return "="
        return "none"
    if variant == "B2":
        if ab == ba:
            return ab
        if ab_lp is None or ba_lp is None:
            raise ValueError("B2 requires ab_lp and ba_lp on disagreement")
        return ab if ab_lp >= ba_lp else ba
    raise ValueError(f"unknown variant {variant!r}")


def answer_span_logprob(tokens: list[str], logprobs: list[float]) -> float:
    """Mean logprob of the answer span = the first line under answer-first
    (the 'Relation: <label>' line). Tokens up to and INCLUDING the first
    token containing a newline; if none, all tokens. Empty → -inf."""
    if not tokens or not logprobs:
        return -math.inf
    span: list[float] = []
    for tok, lp in zip(tokens, logprobs):
        span.append(lp)
        if "\n" in tok:
            break
    return sum(span) / len(span)


def run_both_orders(
    llm,
    kg_source,
    kg_target,
    candidates: Alignment,
    *,
    prompt_id: str,
    description: str,
    kg_format: str = "turtle",
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> list[dict]:
    """Query each deduplicated (s, t) candidate in both orders; return one
    record per pair with both raw labels, both canonical predictions, and
    both answer-span logprobs. No reconciliation here — that is offline."""
    from MatcherSubsumptionReranker import MatcherSubsumptionReranker

    helper = MatcherSubsumptionReranker(
        llm=llm, prompt_id=prompt_id, description=description,
        kg_format=kg_format, max_new_tokens=max_new_tokens, batch_size=batch_size,
    )
    pairs = MatcherSubsumptionReranker._dedup_alignment(candidates)

    # Verbalization is a function of (kg, concept) only — same string in
    # either slot (A2 identity guard). Cache per concept.
    s_text: dict[str, str] = {}
    t_text: dict[str, str] = {}
    for src, tgt, _ev in pairs:
        if src not in s_text:
            s_text[src] = helper._get_entity_text(kg_source, src)
        if tgt not in t_text:
            t_text[tgt] = helper._get_entity_text(kg_target, tgt)

    ab_prompts, ba_prompts = [], []
    for src, tgt, _ev in pairs:
        # AB: present (s, t). BA: present (t, s) — slots filled with the
        # SAME per-concept verbalizations, only positions swap.
        ab_prompts.append(helper._build_prompt(src, tgt, s_text[src], t_text[tgt]))
        ba_prompts.append(helper._build_prompt(tgt, src, t_text[tgt], s_text[src]))

    ab_res = _score(llm, ab_prompts, max_new_tokens, batch_size)
    ba_res = _score(llm, ba_prompts, max_new_tokens, batch_size)

    records: list[dict] = []
    for (src, tgt, ev), ab, ba in zip(pairs, ab_res, ba_res):
        ab_raw = parse_relation_label(ab.get("text", "") or "")
        ba_raw = parse_relation_label(ba.get("text", "") or "")
        records.append({
            "source": src, "target": tgt,
            "stage1_relations": ev["stage1_relations"],
            "ab_raw": ab_raw, "ba_raw": ba_raw,
            "ab_canonical": canonical_for_order(ab_raw, order="AB"),
            "ba_canonical": canonical_for_order(ba_raw, order="BA"),
            "ab_span_logprob": _span(ab),
            "ba_span_logprob": _span(ba),
            "ab_text": (ab.get("text", "") or "").replace("\n", "  "),
            "ba_text": (ba.get("text", "") or "").replace("\n", "  "),
        })
    return records


def _score(llm, prompts, max_new_tokens, batch_size):
    out = []
    for start in range(0, len(prompts), batch_size):
        out.extend(llm.get_text_completion_with_logprobs(
            prompts[start:start + batch_size], max_new_tokens=max_new_tokens))
    return out


def _span(res: dict) -> Optional[float]:
    tokens = res.get("tokens")
    logprobs = res.get("token_logprobs")
    if not tokens or not logprobs:
        return None
    val = answer_span_logprob(tokens, logprobs)
    return None if val == -math.inf else val
