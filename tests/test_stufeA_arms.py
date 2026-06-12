"""Stufe-A arm correctness tests (pre-registered controls, 2026-06-12):

A1 (d_subs_v4b): byte-identical to d_subs_v2 EXCEPT the registered changes —
label ORDER flipped at both (and only) occurrences, with the directional
definitions' trailing padding equalized (the padding control: v2's 4-vs-2
space column alignment is a latent tokenizer-level asymmetry the order flip
must not carry along).

A2 (--swap-pair-presentation): manipulates ONLY which concept sits in which
slot; directional labels are inverted exactly once at parse time when mapping
back to canonical (s, t); verbalization of a concept is identical regardless
of slot (verbalization-identity guard).
"""

from __future__ import annotations

import pytest

from Alignment import Alignment
from Correspondence import Correspondence
from prompt import RERANKING_PROMPTS
from MatcherSubsumptionReranker import (
    MatcherSubsumptionReranker,
    relation_for_canonical_pair,
)

V2 = RERANKING_PROMPTS["d_subs_v2"]
V4B = RERANKING_PROMPTS["d_subs_v4b"]

SUB_DEF = "subclass    source is a more specific kind of target (source ⊑ target)"
SUP_DEF = "superclass  source is a more general kind of target (source ⊒ target)"
SUB_DEF_EQ = "subclass  source is a more specific kind of target (source ⊑ target)"
SUP_DEF_EQ = "superclass  source is a more general kind of target (source ⊒ target)"


# ------------------------------------------------------- A1: v4b string diff

def test_v4b_is_v2_with_exactly_the_registered_transform():
    """Programmatic diff: applying the registered transform to v2 must yield
    v4b byte-exactly — order flip at both occurrences + padding equalization,
    nothing else."""
    transformed = V2.replace(
        f"\n  {SUB_DEF}\n  {SUP_DEF}",
        f"\n  {SUP_DEF_EQ}\n  {SUB_DEF_EQ}",
    ).replace(
        "one of: subclass, superclass, equivalent,",
        "one of: superclass, subclass, equivalent,",
    )
    assert transformed != V2, "transform must change something"
    assert transformed == V4B


def test_v4b_line_level_diff_is_exactly_three_lines():
    v2_lines = V2.split("\n")
    v4b_lines = V4B.split("\n")
    assert len(v2_lines) == len(v4b_lines)
    diff = [(a, b) for a, b in zip(v2_lines, v4b_lines) if a != b]
    assert len(diff) == 3  # two definition lines + the closing enumeration
    assert diff[0][0].strip().startswith("subclass")
    assert diff[0][1].strip().startswith("superclass")
    assert diff[1][0].strip().startswith("superclass")
    assert diff[1][1].strip().startswith("subclass")
    assert "one of: subclass, superclass" in diff[2][0]
    assert "one of: superclass, subclass" in diff[2][1]


def test_v4b_padding_control_equalized():
    """Both directional definition lines carry the SAME trailing-space count
    after the label (two spaces) — the padding is held constant so A1 tests
    label order only."""
    assert "\n  superclass  source is a more general" in V4B
    assert "\n  subclass  source is a more specific" in V4B
    assert "subclass    source" not in V4B          # the v2 4-space padding is gone
    # Non-directional lines stay byte-identical to v2.
    for line in ("equivalent  source and target denote the same concept",
                 "partof      source is a part of target",
                 "none        none of the above applies"):
        assert line in V2 and line in V4B


def test_v2_equalized_padding_sanity_diff():
    """Offline zero-cost sanity (registered): equalizing v2's padding changes
    NOTHING but whitespace — confirms the padding is the only non-semantic
    asymmetry between the two directional definitions in v2."""
    v2_eq = V2.replace(f"  {SUB_DEF}", f"  {SUB_DEF_EQ}")
    assert v2_eq != V2
    # The diff is exactly two removed space characters.
    assert len(V2) - len(v2_eq) == 2
    assert v2_eq.replace(" ", "") == V2.replace(" ", "")


def test_v4b_is_zero_shot():
    assert "Example" not in V4B
    assert V4B.count("Relation:") == 1  # only the answer-format instruction


# ------------------------------------- A2: parse-time inversion exactly once

def test_relation_inversion_table():
    assert relation_for_canonical_pair("subclass", swapped=False) == "<"
    assert relation_for_canonical_pair("superclass", swapped=False) == ">"
    assert relation_for_canonical_pair("subclass", swapped=True) == ">"
    assert relation_for_canonical_pair("superclass", swapped=True) == "<"
    for label in ("equivalent", "partof", "none", "parse_fail"):
        assert relation_for_canonical_pair(label, swapped=True) == \
            relation_for_canonical_pair(label, swapped=False)


class _FakeLLM:
    """Returns a fixed completion; records every prompt it sees."""

    def __init__(self, completion: str = "Relation: subclass"):
        self.completion = completion
        self.seen_prompts: list = []

    def get_text_completion_with_logprobs(self, prompts, max_new_tokens):
        self.seen_prompts.extend(prompts)
        return [{"text": self.completion, "token_logprobs": [-0.1],
                 "sum_logprob": -0.1, "n_tokens": 1} for _ in prompts]


SOURCE_TTL = """\
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <http://example.org/src/> .

ex:Animal a owl:Class ; rdfs:label "animal" .
ex:Dog    a owl:Class ; rdfs:label "dog" ; rdfs:subClassOf ex:Animal .
"""

TARGET_TTL = """\
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ey:   <http://example.org/tgt/> .

ey:Organism a owl:Class ; rdfs:label "organism" .
ey:Canine   a owl:Class ; rdfs:label "canine" ; rdfs:subClassOf ey:Organism .
"""

S_URI = "http://example.org/src/Dog"
T_URI = "http://example.org/tgt/Canine"


@pytest.fixture(scope="module")
def toy_kgs(tmp_path_factory):
    from run_subsumption_experiment import _load_kg_with_labels
    d = tmp_path_factory.mktemp("toy_kgs_stage2")
    src = d / "source.ttl"
    tgt = d / "target.ttl"
    src.write_text(SOURCE_TTL, encoding="utf-8")
    tgt.write_text(TARGET_TTL, encoding="utf-8")
    return _load_kg_with_labels(src)[0], _load_kg_with_labels(tgt)[0]


def _run_reranker(kgs, *, swap: bool, completion: str = "Relation: subclass"):
    kg_source, kg_target = kgs
    llm = _FakeLLM(completion)
    reranker = MatcherSubsumptionReranker(
        llm=llm, prompt_id="d_subs_v2", description="description_path_context",
        swap_pair_presentation=swap,
    )
    candidates = Alignment()
    candidates.add(Correspondence(S_URI, T_URI, "<", 0.9))
    out = reranker.match(kg_source, kg_target, candidates, parameters={})
    return out, reranker, llm


def test_swap_inverts_directional_label_exactly_once_full_path(toy_kgs):
    """Full parse path: model says 'subclass' about the PRESENTED (t, s) pair
    -> canonical (s, t) relation must be '>' (inverted once, not twice)."""
    out, reranker, _ = _run_reranker(toy_kgs, swap=True)
    [cor] = list(out)
    assert (cor.source, cor.target) == (S_URI, T_URI)  # canonical orientation kept
    assert cor.relation == ">"
    [detail] = reranker.last_run_details
    assert detail["parsed_canonical"] == "subclass"     # presented frame
    assert detail["predicted_relation"] == ">"          # canonical frame
    assert detail["pair_presentation"] == "swapped"


def test_no_swap_keeps_directional_label(toy_kgs):
    out, reranker, _ = _run_reranker(toy_kgs, swap=False)
    [cor] = list(out)
    assert cor.relation == "<"
    assert reranker.last_run_details[0]["pair_presentation"] == "canonical"


def test_swap_does_not_touch_equivalent(toy_kgs):
    out, _, _ = _run_reranker(toy_kgs, swap=True, completion="Relation: equivalent")
    [cor] = list(out)
    assert cor.relation == "="


def test_swap_fills_slots_with_swapped_concepts(toy_kgs):
    """Prompt text is byte-identical v2; only the slot CONTENTS swap: the
    source slots carry the t-concept (URI + its verbalization), the target
    slots the s-concept."""
    _, _, llm = _run_reranker(toy_kgs, swap=True)
    [prompt] = llm.seen_prompts
    text = str(prompt)
    assert f"Source entity: <{T_URI}>" in text
    assert f"Target entity: <{S_URI}>" in text


def test_verbalization_identity_across_slots(toy_kgs):
    """A2 verbalization-identity guard (registered): the path_context
    verbalization of a concept is the same string whether it lands in the
    source slot (swapped run) or the target slot (canonical run)."""
    kg_source, kg_target = toy_kgs
    _, reranker_canon, llm_canon = _run_reranker(toy_kgs, swap=False)
    _, _, llm_swap = _run_reranker(toy_kgs, swap=True)

    t_verbalization = reranker_canon._get_entity_text(kg_target, T_URI)
    s_verbalization = reranker_canon._get_entity_text(kg_source, S_URI)
    assert t_verbalization and s_verbalization

    canon_text = str(llm_canon.seen_prompts[0])
    swap_text = str(llm_swap.seen_prompts[0])
    # Same strings appear in both runs — only their slot changes.
    assert t_verbalization in canon_text and t_verbalization in swap_text
    assert s_verbalization in canon_text and s_verbalization in swap_text
    # Slot placement: in the swapped prompt the t-verbalization sits in the
    # source block (before the Target header), in the canonical prompt in the
    # target block (after it).
    assert swap_text.index(t_verbalization) < swap_text.index("Target entity:")
    assert canon_text.index(t_verbalization) > canon_text.index("Target entity:")
