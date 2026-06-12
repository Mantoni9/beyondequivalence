"""
swap_retrieval.py — pass model + re-orientation for the Stage-1
swapped-retrieval ablation (fixing the superclass recall ceiling).

The frozen MatcherAsymmetricRetrieval queries from one ontology side with the
broader/narrower instruction prefixes (query side only — the passage side is
encoded without any prefix) and emits correspondences keyed
(query_concept, retrieved_concept). The swapped t-side passes are produced by
LITERALLY invoking that frozen matcher on the transposed ontology pair
(kg_target as the querying side) and re-orienting its output — same encode
path, same instruction roles, same verbalization. Nothing of the frozen
matcher is reimplemented or modified here.

Canonical orientation invariant: every PassRow carries
(source_uri = source-ontology concept, target_uri = target-ontology concept,
relation = canonical direction hint). A t-side broader hit (t, s', '<')
[t is more specific than s'] is re-oriented to (s', t, '>').

Score/rank semantics (binding): scores are comparable ONLY within one
(pass_id, query_uri) list — the four passes use different query encodings
with systematically different score distributions (the original rationale
for per_relation_strict). `rank` is the 1-based position within the
originating (pass_id, query_uri) list, ordered by (score desc,
retrieved-URI asc). No code may sort across passes by raw score; budget
capping is per (query_uri, pass), never per source_uri — a per-source cap
on t-side candidates would re-introduce the fan-out ceiling this ablation
exists to remove.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from Alignment import Alignment


@dataclass(frozen=True)
class PassSpec:
    query_side: str          # ontology side that issues the embedding queries
    instruction: str         # 'broader' | 'narrower' query instruction
    canonical_relation: str  # direction hint in canonical (source, target) orientation


PASS_SPECS: dict[str, PassSpec] = {
    "s_broader":  PassSpec("source", "broader",  "<"),
    "s_narrower": PassSpec("source", "narrower", ">"),
    "t_broader":  PassSpec("target", "broader",  ">"),
    "t_narrower": PassSpec("target", "narrower", "<"),
}

# Ablation variants as pass subsets. One GPU run produces all four passes;
# the variants (and the frozen baseline) are offline recombinations.
VARIANTS: dict[str, tuple[str, ...]] = {
    "baseline": ("s_broader", "s_narrower"),
    "v_sym":    ("s_broader", "t_broader"),
    "v_union":  ("s_broader", "s_narrower", "t_broader", "t_narrower"),
}


@dataclass(frozen=True, order=True)
class PassRow:
    source_uri: str   # ALWAYS a source-ontology concept
    target_uri: str   # ALWAYS a target-ontology concept
    relation: str     # canonical direction hint: '<' (s ⊑ t) or '>' (s ⊒ t)
    score: float
    pass_id: str
    query_uri: str    # the concept that issued the embedding query
    rank: int         # 1-based position within this (pass_id, query_uri) list


def passes_from_alignment(alignment: Alignment, *, query_side: str) -> dict[str, list[PassRow]]:
    """Split a MatcherAsymmetricRetrieval output into canonical per-pass rows.

    query_side='source': output of match(kg_source, kg_target) — orientation
    kept as-is. query_side='target': output of match(kg_target, kg_source) —
    correspondences are keyed (t, s') and are re-oriented; the relation flips
    ('<' -> '>' and vice versa) because the matcher stamped it relative to
    its querying side.
    """
    if query_side == "source":
        pass_by_rel = {"<": "s_broader", ">": "s_narrower"}
    elif query_side == "target":
        pass_by_rel = {"<": "t_broader", ">": "t_narrower"}
    else:
        raise ValueError(f"query_side must be 'source' or 'target', got {query_side!r}")

    by_pass_query: dict[str, dict[str, list[tuple[str, float]]]] = {
        pass_id: {} for pass_id in pass_by_rel.values()
    }
    for cor in alignment:
        pass_id = pass_by_rel.get(cor.relation)
        if pass_id is None:
            raise ValueError(f"unexpected relation {cor.relation!r} in matcher output")
        by_pass_query[pass_id].setdefault(cor.source, []).append(
            (cor.target, float(cor.confidence)),
        )

    out: dict[str, list[PassRow]] = {pass_id: [] for pass_id in pass_by_rel.values()}
    for pass_id, by_query in by_pass_query.items():
        spec = PASS_SPECS[pass_id]
        for query_uri, entries in by_query.items():
            entries.sort(key=lambda e: (-e[1], e[0]))
            for rank_zero, (retrieved_uri, score) in enumerate(entries):
                if spec.query_side == "source":
                    src, tgt = query_uri, retrieved_uri
                else:
                    src, tgt = retrieved_uri, query_uri
                out[pass_id].append(PassRow(
                    src, tgt, spec.canonical_relation, score,
                    pass_id, query_uri, rank_zero + 1,
                ))
    return out


def run_swapped_passes(
    matcher,
    kg_source,
    kg_target,
    *,
    parameters: Optional[dict] = None,
) -> dict[str, list[PassRow]]:
    """Run the frozen matcher on the TRANSPOSED task (target concepts become
    the instruction-prefixed queries, source concepts the prefix-free passage
    index) and re-orient. Returns {'t_broader': [...], 't_narrower': [...]}."""
    raw = matcher.match(kg_target, kg_source, Alignment(), parameters or {})
    return passes_from_alignment(raw, query_side="target")


def run_all_passes(
    matcher,
    kg_source,
    kg_target,
    *,
    parameters: Optional[dict] = None,
) -> dict[str, list[PassRow]]:
    """All four passes from two invocations of the unmodified frozen matcher."""
    direct = matcher.match(kg_source, kg_target, Alignment(), parameters or {})
    passes = passes_from_alignment(direct, query_side="source")
    passes.update(run_swapped_passes(matcher, kg_source, kg_target, parameters=parameters))
    return passes


def assemble_variant(passes: dict[str, list[PassRow]], variant: str) -> list[PassRow]:
    """Concatenate the variant's pass lists. Raises KeyError on unknown variant.
    Rows stay tagged by pass — this is NOT a joint ranking."""
    rows: list[PassRow] = []
    for pass_id in VARIANTS[variant]:
        rows.extend(passes.get(pass_id, []))
    return rows


def candidate_pairs_at_budget(rows: Iterable[PassRow], k: int) -> set[tuple[str, str]]:
    """Candidate pair set at budget K: every (query, pass) list capped at K,
    pairs collapsed across passes (direction hints ignored)."""
    return {(r.source_uri, r.target_uri) for r in rows if r.rank <= k}


def candidate_triples(rows: Iterable[PassRow]) -> set[tuple[str, str, str]]:
    """Canonical (source, target, relation) set — cross-pass collisions such as
    s_narrower (s, t, '>') vs re-oriented t_broader (s, t, '>') collapse here."""
    return {(r.source_uri, r.target_uri, r.relation) for r in rows}


TSV_COLUMNS = ("source_uri", "target_uri", "relation", "score", "pass_id", "query_uri", "rank")

TSV_HEADER_COMMENTS = (
    "# passes.tsv — swapped-retrieval ablation pass artifact (one row per ranked hit per pass).",
    "# Canonical orientation: source_uri is ALWAYS a source-ontology concept, target_uri ALWAYS",
    "# a target-ontology concept; relation is the canonical direction hint ('<' s⊑t, '>' s⊒t).",
    "# pass_id ∈ {s_broader, s_narrower, t_broader, t_narrower}; query_uri is the concept that",
    "# issued the embedding query; rank is the 1-based position within the (pass_id, query_uri)",
    "# list ordered by (score desc, retrieved-URI asc).",
    "# Scores are comparable ONLY within one (pass_id, query_uri) list — never sort across",
    "# passes by raw score.",
    "# rank — not score — is authoritative for list order: scores are stored 6-dp rounded",
    "# while ranks were assigned from full precision.",
    "# Future Stage-2 loader semantics: cap top-K per (query_uri, pass) — i.e. per directed",
    "# query — NOT per source_uri; a per-source cap would re-introduce the fan-out ceiling.",
    "# After capping, dedup to unique (source_uri, target_uri) pairs — cross-pass canonical",
    "# collisions such as s_narrower (s,t,'>') vs re-oriented t_broader (s,t,'>') are",
    "# intentional provenance, not distinct candidates.",
)


def write_passes_tsv(path, rows: Iterable[PassRow], comment_lines=TSV_HEADER_COMMENTS) -> None:
    ordered = sorted(rows, key=lambda r: (r.pass_id, r.query_uri, r.rank))
    with Path(path).open("w", encoding="utf-8") as f:
        for line in comment_lines:
            f.write(line + "\n")
        f.write("\t".join(TSV_COLUMNS) + "\n")
        for r in ordered:
            f.write(f"{r.source_uri}\t{r.target_uri}\t{r.relation}\t{r.score:.6f}"
                    f"\t{r.pass_id}\t{r.query_uri}\t{r.rank}\n")


def read_passes_tsv(path) -> list[PassRow]:
    rows: list[PassRow] = []
    header_seen = False
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line:
                continue
            if not header_seen:
                if line.split("\t") != list(TSV_COLUMNS):
                    raise ValueError(f"{path}: unexpected header {line!r}")
                header_seen = True
                continue
            s, t, rel, sc, pass_id, query_uri, rank = line.split("\t")
            rows.append(PassRow(s, t, rel, float(sc), pass_id, query_uri, int(rank)))
    return rows
