"""Mirror-exactness acceptance test (binding point 1 of the swap ablation):
the swapped t-side passes must be byte-identical to the frozen
MatcherAsymmetricRetrieval invoked on the transposed ontology pair —
same encode path, same instruction roles, same scores — modulo re-orientation.

Uses a tiny synthetic ontology pair and all-MiniLM-L6-v2 (CPU-friendly).
"""

from __future__ import annotations

import pytest

from Alignment import Alignment

SOURCE_TTL = """\
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex:   <http://example.org/src/> .

ex:Animal a owl:Class ; rdfs:label "animal" .
ex:Dog    a owl:Class ; rdfs:label "dog" ; rdfs:subClassOf ex:Animal .
ex:Poodle a owl:Class ; rdfs:label "poodle" ; rdfs:subClassOf ex:Dog .
"""

TARGET_TTL = """\
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ey:   <http://example.org/tgt/> .

ey:Organism a owl:Class ; rdfs:label "organism" .
ey:Canine   a owl:Class ; rdfs:label "canine" ; rdfs:subClassOf ey:Organism .
ey:Pet      a owl:Class ; rdfs:label "pet" ; rdfs:subClassOf ey:Organism .
"""


@pytest.fixture(scope="module")
def toy_kgs(tmp_path_factory):
    from run_subsumption_experiment import _load_kg_with_labels

    d = tmp_path_factory.mktemp("toy_kgs")
    src = d / "source.ttl"
    tgt = d / "target.ttl"
    src.write_text(SOURCE_TTL, encoding="utf-8")
    tgt.write_text(TARGET_TTL, encoding="utf-8")
    kg_source, _ = _load_kg_with_labels(src)
    kg_target, _ = _load_kg_with_labels(tgt)
    return kg_source, kg_target


@pytest.fixture(scope="module")
def frozen_style_matcher():
    from MatcherAsymmetricRetrieval import MatcherAsymmetricRetrieval
    from prompt import get_subb_asym_templates

    broader, narrower = get_subb_asym_templates("T2")
    return MatcherAsymmetricRetrieval(
        model="sentence-transformers/all-MiniLM-L6-v2",
        broader_query_instruction=broader,
        narrower_query_instruction=narrower,
        document_instruction="",
        description="description_path_context",
        top_k=3,
        kg_format="turtle",
    )


def test_swapped_passes_mirror_transposed_matcher_exactly(toy_kgs, frozen_style_matcher):
    """run_swapped_passes == matcher.match on (T, S), modulo re-orientation."""
    from swap_retrieval import run_swapped_passes

    kg_source, kg_target = toy_kgs

    # Reference invocation: the existing frozen matcher on the transposed task.
    raw = frozen_style_matcher.match(kg_target, kg_source, Alignment(), {})
    raw_broader = {(c.source, c.target): float(c.confidence)
                   for c in raw if c.relation == "<"}
    raw_narrower = {(c.source, c.target): float(c.confidence)
                    for c in raw if c.relation == ">"}
    assert raw_broader and raw_narrower, "toy run must produce both passes"

    # Wrapper invocation (its own match() call -> also guards determinism).
    passes = run_swapped_passes(frozen_style_matcher, kg_source, kg_target)

    got_broader = {(r.query_uri, r.source_uri): r.score for r in passes["t_broader"]}
    got_narrower = {(r.query_uri, r.source_uri): r.score for r in passes["t_narrower"]}
    assert got_broader == raw_broader
    assert got_narrower == raw_narrower

    # Canonical orientation: source_uri ALWAYS from the source ontology,
    # query_uri ALWAYS the t-concept, relation flipped.
    for r in passes["t_broader"] + passes["t_narrower"]:
        assert r.source_uri.startswith("http://example.org/src/")
        assert r.target_uri.startswith("http://example.org/tgt/")
        assert r.query_uri == r.target_uri
    assert {r.relation for r in passes["t_broader"]} == {">"}
    assert {r.relation for r in passes["t_narrower"]} == {"<"}


def test_run_all_passes_source_side_identical_to_direct_match(toy_kgs, frozen_style_matcher):
    """The s-side passes of run_all_passes are the frozen matcher's own output."""
    from swap_retrieval import run_all_passes

    kg_source, kg_target = toy_kgs
    direct = frozen_style_matcher.match(kg_source, kg_target, Alignment(), {})
    direct_rows = {(c.source, c.target, c.relation, float(c.confidence)) for c in direct}

    passes = run_all_passes(frozen_style_matcher, kg_source, kg_target)
    assert set(passes) == {"s_broader", "s_narrower", "t_broader", "t_narrower"}
    got_rows = {(r.source_uri, r.target_uri, r.relation, r.score)
                for r in passes["s_broader"] + passes["s_narrower"]}
    assert got_rows == direct_rows
