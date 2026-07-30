"""
fewshot_exemplars.py — E15 few-shot exemplar builder (thesis 3.3.2 / K4).

Exemplars are sourced from a HELD-OUT track (default g1-web), evaluation-disjoint
from the E15 eval sets (g5, g7, g3). g1-web gold carries native '<','>','='
(29/26/275), so NO swap-derivation is needed — every arm is sourced from the
matching native pool. Selection is deterministic (seeded, sorted pools).

Arms:
  A0  zero-shot (empty block; the run uses plain d_subs_v2 — no work here)
  A1  N=1: one '<' exemplar (replicates the pilot claim)
  A2  balanced-3: one each of '<','>','='
  A3  balanced-6: two each
  A4  mirrored-6: balanced-6 where each exemplar ALSO appears swapped
      (source<->target, label inverted) — an anti-position-prior control.

Each exemplar's source/target entity is verbalized with the SAME description
method as the eval pairs (description_path_context), so exemplars and queries
are format-identical.
"""

from __future__ import annotations

import random

# arm -> (per-relation exemplar count, mirrored?)
ARM_SPECS: dict[str, tuple[dict[str, int], bool]] = {
    "A0": ({}, False),
    "A1": ({"<": 1}, False),
    "A2": ({"<": 1, ">": 1, "=": 1}, False),
    "A3": ({"<": 2, ">": 2, "=": 2}, False),
    "A4": ({"<": 2, ">": 2, "=": 2}, True),
}

_REL_TO_LABEL = {"<": "subclass", ">": "superclass", "=": "equivalent"}
_INVERT = {"<": ">", ">": "<", "=": "="}


def select_exemplars(pools: dict, arm: str, seed: int) -> list[dict]:
    """Pure, deterministic exemplar selection. NO I/O — unit-testable.

    pools: {rel: [(source_uri, target_uri), ...]} from the exemplar track's gold.
    Returns exemplar specs in prompt order. Each spec carries the ORIGIN ontology
    per slot ('src' = exemplar-track source ontology, 'tgt' = its target ontology)
    so the caller verbalizes each entity with the right graph even after mirroring.
    Base exemplars keep gold orientation (source in 'src', target in 'tgt');
    mirrored ones swap slots and invert the label. Raises if a pool is too small."""
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown arm {arm!r}; known={sorted(ARM_SPECS)}")
    label_mix, mirrored = ARM_SPECS[arm]
    if not label_mix:
        return []
    rng = random.Random(seed)
    base: list[dict] = []
    for rel in sorted(label_mix):                      # deterministic: '<','=','>'
        n = label_mix[rel]
        pool = sorted(pools.get(rel, []))              # deterministic base order
        if len(pool) < n:
            raise ValueError(
                f"exemplar pool '{rel}' has {len(pool)} pairs < {n} needed for {arm}")
        for (s, t) in rng.sample(pool, n):
            base.append({"rel": rel, "source": s, "target": t,
                         "source_onto": "src", "target_onto": "tgt",
                         "mirrored": False})
    specs = list(base)
    if mirrored:
        for e in base:
            specs.append({"rel": _INVERT[e["rel"]],
                          "source": e["target"], "target": e["source"],
                          "source_onto": "tgt", "target_onto": "src",
                          "mirrored": True})
    return specs


def _entity_text(kg, uri: str, description: str, kg_format: str) -> str:
    from rdflib.term import URIRef
    from RDFGraphWrapper import RDFGraphWrapper
    result = getattr(kg, description)(URIRef(uri))
    if isinstance(result, str):
        return result
    return RDFGraphWrapper.serialize(result, format=kg_format)


def _render(idx: int, src_uri: str, src_text: str, tgt_uri: str, tgt_text: str,
            rel: str) -> str:
    return (f"\n--- Example {idx} ---"
            f"\nSource entity: <{src_uri}>"
            f"\nSource knowledge graph:\n{src_text}"
            f"\n\nTarget entity: <{tgt_uri}>"
            f"\nTarget knowledge graph:\n{tgt_text}"
            f"\nRelation: {_REL_TO_LABEL[rel]}")


def build_fewshot_block(arm: str, exemplar_track: str, description: str,
                        kg_format: str, seed: int) -> tuple[str, list[dict]]:
    """Return (prompt_block, manifest). manifest records every exemplar's
    provenance (uris, relation, mirrored flag) for config.json. Empty for A0."""
    label_mix, _ = ARM_SPECS.get(arm, ({}, False))
    if not label_mix:
        return "", []

    from Alignment import Alignment
    from RDFGraphWrapper import RDFGraphWrapper
    from evaluation_recall import _normalize_relation
    from tracks.zenodo_loader import load_subdataset

    src_path, tgt_path, ref_path = load_subdataset(exemplar_track)
    pools: dict[str, list] = {"<": [], ">": [], "=": []}
    for c in Alignment(str(ref_path)):
        rel = _normalize_relation(c.relation)
        if rel in pools:
            pools[rel].append((c.source, c.target))

    specs = select_exemplars(pools, arm, seed)
    kg = {"src": RDFGraphWrapper(str(src_path)), "tgt": RDFGraphWrapper(str(tgt_path))}

    blocks, manifest = [], []
    for i, e in enumerate(specs, start=1):
        s_text = _entity_text(kg[e["source_onto"]], e["source"], description, kg_format)
        t_text = _entity_text(kg[e["target_onto"]], e["target"], description, kg_format)
        blocks.append(_render(i, e["source"], s_text, e["target"], t_text, e["rel"]))
        manifest.append({"idx": i, "source": e["source"], "target": e["target"],
                         "relation": e["rel"], "mirrored": e["mirrored"]})
    return "\n".join(blocks), manifest
