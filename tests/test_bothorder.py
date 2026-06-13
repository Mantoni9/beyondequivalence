"""Stufe-B Both-Order-Voting: reconciliation truth tables (B1/B2/B3),
answer-span logprob extraction, and the double-order inference path
(FakeLLM). Correctness-critical, GPU-free.

Derivation grounding (verified on the Stufe-A named-26):
a purely positional model says "first-presented = subclass". On a true X⊏Y
it gives raw 'subclass' in BOTH frames → canonical (<,>) DISAGREEMENT. A
content-tracking model flips its raw label with the swap → same canonical
direction in both frames → (<,<) AGREEMENT. Hence:
  (<,<)/(>,>)  canonical agreement = content-consistent  → real direction
  (<,>)/(>,<)  canonical disagreement = same raw both ways = X⊑Y ∧ X⊒Y
               = mutual subsumption                         → '=' (B3) / none (B1)
"""

from __future__ import annotations

import math
import pytest

from Alignment import Alignment
from Correspondence import Correspondence
from stage2_bothorder import (
    RECON_VARIANTS,
    answer_span_logprob,
    canonical_for_order,
    reconcile,
    run_both_orders,
)

DIR = ("<", ">", "=", "none")


# ---------------------------------------------- canonical_for_order (reuse A2)

def test_canonical_for_order_ab_keeps_label():
    assert canonical_for_order("subclass", order="AB") == "<"
    assert canonical_for_order("superclass", order="AB") == ">"
    assert canonical_for_order("equivalent", order="AB") == "="


def test_canonical_for_order_ba_inverts_once():
    assert canonical_for_order("subclass", order="BA") == ">"
    assert canonical_for_order("superclass", order="BA") == "<"
    assert canonical_for_order("equivalent", order="BA") == "="


def test_canonical_for_order_drops_fold_to_none():
    for lbl in ("none", "partof", "parse_fail"):
        assert canonical_for_order(lbl, order="AB") == "none"
        assert canonical_for_order(lbl, order="BA") == "none"


# ----------------------------------------------------------------- B1 abstain

def test_b1_agreement_emits_including_equivalence():
    assert reconcile("<", "<", variant="B1") == "<"
    assert reconcile(">", ">", variant="B1") == ">"
    assert reconcile("=", "=", variant="B1") == "="   # = passes through (A1 lesson)
    assert reconcile("none", "none", variant="B1") == "none"


def test_b1_any_disagreement_abstains_to_none():
    assert reconcile("<", ">", variant="B1") == "none"
    assert reconcile(">", "<", variant="B1") == "none"
    assert reconcile("<", "=", variant="B1") == "none"
    assert reconcile("=", ">", variant="B1") == "none"
    assert reconcile("<", "none", variant="B1") == "none"
    assert reconcile("=", "none", variant="B1") == "none"


# -------------------------------------------------------- B3 symmetry-grounded

def test_b3_canonical_agreement_is_real_direction():
    # content-consistent (model flipped raw label with the swap) → trust direction
    assert reconcile("<", "<", variant="B3") == "<"
    assert reconcile(">", ">", variant="B3") == ">"


def test_b3_canonical_disagreement_is_equivalence():
    # same raw label both frames = X⊑Y ∧ X⊒Y = mutual subsumption = '='
    assert reconcile("<", ">", variant="B3") == "="
    assert reconcile(">", "<", variant="B3") == "="


def test_b3_double_equivalence_and_conflicts():
    assert reconcile("=", "=", variant="B3") == "="
    assert reconcile("<", "=", variant="B3") == "none"   # dir vs equiv conflict
    assert reconcile("=", ">", variant="B3") == "none"
    assert reconcile("<", "none", variant="B3") == "none"
    assert reconcile("=", "none", variant="B3") == "none"
    assert reconcile("none", "none", variant="B3") == "none"


def test_b1_and_b3_differ_only_in_the_disagreement_cell():
    for ab in DIR:
        for ba in DIR:
            r1 = reconcile(ab, ba, variant="B1")
            r3 = reconcile(ab, ba, variant="B3")
            if {ab, ba} == {"<", ">"}:
                assert r1 == "none" and r3 == "="
            else:
                assert r1 == r3


# ----------------------------------------------------- B2 confidence tie-break

def test_b2_agreement_emits_agreed_label():
    assert reconcile("<", "<", variant="B2", ab_lp=-0.1, ba_lp=-0.9) == "<"
    assert reconcile("=", "=", variant="B2", ab_lp=-0.1, ba_lp=-0.9) == "="


def test_b2_disagreement_higher_logprob_frame_wins():
    # AB more confident (less negative) -> AB's canonical label
    assert reconcile("<", ">", variant="B2", ab_lp=-0.2, ba_lp=-1.0) == "<"
    # BA more confident -> BA's canonical label
    assert reconcile("<", ">", variant="B2", ab_lp=-1.0, ba_lp=-0.2) == ">"
    # a frame's label may be 'none' if that frame won
    assert reconcile("<", "none", variant="B2", ab_lp=-2.0, ba_lp=-0.1) == "none"


def test_b2_tie_is_deterministic_toward_ab():
    assert reconcile("<", ">", variant="B2", ab_lp=-0.5, ba_lp=-0.5) == "<"


def test_b2_direction_meets_equivalence_is_tiebroken():
    # the named-26's 2 outliers are canonical (<,=); B2 treats it as any
    # disagreement → higher-logprob frame wins.
    assert reconcile("<", "=", variant="B2", ab_lp=-0.2, ba_lp=-1.0) == "<"
    assert reconcile("<", "=", variant="B2", ab_lp=-1.0, ba_lp=-0.2) == "="


def test_direction_meets_equivalence_row_explicit_all_variants():
    # registered named row: dir-vs-equiv is neither agreement nor a clean
    # directional disagreement. B1/B3 → none; B2 → tie-break.
    for ab, ba in (("<", "="), (">", "="), ("=", "<"), ("=", ">")):
        assert reconcile(ab, ba, variant="B1") == "none"
        assert reconcile(ab, ba, variant="B3") == "none"
        assert reconcile(ab, ba, variant="B2", ab_lp=-0.1, ba_lp=-9.0) in (ab, ba)


def test_b2_requires_logprobs():
    with pytest.raises(ValueError):
        reconcile("<", ">", variant="B2")


def test_recon_variants_registry():
    assert set(RECON_VARIANTS) == {"B1", "B2", "B3"}


# --------------------------------------------------- answer_span_logprob

def test_answer_span_is_first_line_mean():
    # "Relation: superclass\n<justification>" — span = first line tokens.
    tokens = ["Relation", ":", " superclass", "\n", "Because", " ..."]
    logprobs = [-0.1, -0.2, -0.3, -0.05, -2.0, -3.0]
    span = answer_span_logprob(tokens, logprobs)
    assert span == pytest.approx(sum([-0.1, -0.2, -0.3, -0.05]) / 4)


def test_answer_span_handles_label_glued_to_newline():
    tokens = ["Relation", ": ", "subclass\n", "rest"]
    logprobs = [-0.1, -0.2, -0.4, -5.0]
    span = answer_span_logprob(tokens, logprobs)
    assert span == pytest.approx(sum([-0.1, -0.2, -0.4]) / 3)


def test_answer_span_single_line_uses_all():
    tokens = ["Relation", ": ", "none"]
    logprobs = [-0.1, -0.2, -0.6]
    assert answer_span_logprob(tokens, logprobs) == pytest.approx(-0.9 / 3)


def test_answer_span_empty_is_neginf():
    assert answer_span_logprob([], []) == -math.inf


# --------------------------------------------------- double-order inference

SOURCE_TTL = """\
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <http://example.org/src/> .
ex:Dog a owl:Class ; rdfs:label "dog" .
"""
TARGET_TTL = """\
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ey:   <http://example.org/tgt/> .
ey:Canine a owl:Class ; rdfs:label "canine" .
"""
S_URI, T_URI = "http://example.org/src/Dog", "http://example.org/tgt/Canine"


class _OrderAwareLLM:
    """Returns a label that depends on which concept is in the SOURCE slot —
    emulates a purely positional 'first = subclass' model."""

    def __init__(self):
        self.seen = []

    def get_text_completion_with_logprobs(self, prompts, max_new_tokens, **kwargs):
        out = []
        for p in prompts:
            text = str(p)
            self.seen.append(text)
            # positional model: always 'subclass' about the presented source
            out.append({"text": "Relation: subclass\n",
                        "tokens": ["Relation", ": ", "subclass", "\n"],
                        "token_logprobs": [-0.1, -0.1, -0.2, -0.05],
                        "sum_logprob": -0.45, "n_tokens": 4})
        return out


@pytest.fixture(scope="module")
def toy_kgs(tmp_path_factory):
    from run_subsumption_experiment import _load_kg_with_labels
    d = tmp_path_factory.mktemp("toy_bothorder")
    (d / "s.ttl").write_text(SOURCE_TTL, encoding="utf-8")
    (d / "t.ttl").write_text(TARGET_TTL, encoding="utf-8")
    return _load_kg_with_labels(d / "s.ttl")[0], _load_kg_with_labels(d / "t.ttl")[0]


def test_run_both_orders_positional_model_yields_canonical_disagreement(toy_kgs):
    kg_source, kg_target = toy_kgs
    cands = Alignment()
    cands.add(Correspondence(S_URI, T_URI, "<", 0.9))
    records = run_both_orders(_OrderAwareLLM(), kg_source, kg_target, cands,
                              prompt_id="d_subs_v2", description="description_path_context")
    assert len(records) == 1
    r = records[0]
    # positional 'subclass' in both frames → AB '<', BA '>'
    assert r["ab_canonical"] == "<"
    assert r["ba_canonical"] == ">"
    assert r["ab_raw"] == "subclass" and r["ba_raw"] == "subclass"
    # B3 → '=', B1 → 'none', and span logprobs present
    assert reconcile(r["ab_canonical"], r["ba_canonical"], variant="B3") == "="
    assert reconcile(r["ab_canonical"], r["ba_canonical"], variant="B1") == "none"
    assert r["ab_span_logprob"] is not None and r["ba_span_logprob"] is not None
    assert r["source"] == S_URI and r["target"] == T_URI  # canonical orientation


def test_run_both_orders_presents_both_slot_orders(toy_kgs):
    kg_source, kg_target = toy_kgs
    cands = Alignment()
    cands.add(Correspondence(S_URI, T_URI, "<", 0.9))
    llm = _OrderAwareLLM()
    run_both_orders(llm, kg_source, kg_target, cands,
                    prompt_id="d_subs_v2", description="description_path_context")
    assert len(llm.seen) == 2
    ab, ba = llm.seen
    assert f"Source entity: <{S_URI}>" in ab and f"Target entity: <{T_URI}>" in ab
    assert f"Source entity: <{T_URI}>" in ba and f"Target entity: <{S_URI}>" in ba
