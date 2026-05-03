"""
MatcherBidirectionalConsolidation — Hebel C: bidirectional subsumption
retrieval consolidated via Reciprocal Rank Fusion (Cormack et al. 2009,
SIGIR).

Scope (per Sub-B exit decision, see THESIS_NOTES.md):
  - Direction is fixed to '<' (Source ⊂ Target). Both passes emit
    Correspondence(s, t, '<') and the fusion happens over (s, t)-pairs.
  - The mirror-image direction '>' is a separate experiment, deferred.
  - C=off means: ONE forward pass only — broader-instruction anchored at
    the source side, output relation '<'. Confidence is the RRF score with
    a single summand, so C=off and C=on report scores in the same
    numerical range and are directly comparable.

C=on adds an inverse pass: narrower-instruction anchored at the target
side, retrieving sources for each target. The inverse-pass result for
target t with sources [S1, S2, ...] is mapped onto the same (s, t, '<')
schlüsselraum as the forward pass — that is methodologically the point:
both passes seek the same ground-truth subsumption relation from
opposite directions, with complementary error patterns.

Cost model:
  - Both passes share the document-side encoding direction implicitly
    (forward target-side and inverse source-side use the same empty
    document_instruction). To keep this simple we still encode each side
    twice when fusion=True (source with broader and as document; target
    as document and with narrower) — the four encodings are needed
    because the prompt prefixes differ. An optimisation pass that reuses
    the document-side encoding across forward/inverse would save one
    encoding per match() call but is not required for correctness.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import torch
from sentence_transformers import util

from MatcherBase import MatcherBase
from RDFGraphWrapper import RDFGraphWrapper
from Alignment import Alignment
from Correspondence import Correspondence
from MatcherEmbeddingRetrieval import _sync, _verify_loader_kwargs_applied, _truncation_stats
from prompt import build_instruct_query_prompt, get_loader_kwargs

logger = logging.getLogger(__name__)


# RRF k=60 per Cormack et al. (SIGIR 2009), the industry standard for rank
# fusion. Smaller k weighs the head of each list more heavily; 60 is the
# documented default that the original paper validated against TREC runs.
RRF_K_DEFAULT: int = 60


def reciprocal_rank_fusion(
    forward_ranks: dict[tuple[int, int], int],
    inverse_ranks: dict[tuple[int, int], int],
    k: int = RRF_K_DEFAULT,
) -> dict[tuple[int, int], float]:
    """Fuse two ranked-list dicts indexed by (source_idx, target_idx).

    forward_ranks[(s, t)] = 1-based position of t in source s's forward
    ranking (= broader-pass anchored at source).
    inverse_ranks[(s, t)] = 1-based position of s in target t's inverse
    ranking (= narrower-pass anchored at target).

    Pairs that appear in only one list contribute one summand. RRF score
    formula: score(s, t) = Σ_pass 1 / (k + rank_pass(s, t)).
    """
    pairs = set(forward_ranks) | set(inverse_ranks)
    out: dict[tuple[int, int], float] = {}
    for pair in pairs:
        score = 0.0
        if pair in forward_ranks:
            score += 1.0 / (k + forward_ranks[pair])
        if pair in inverse_ranks:
            score += 1.0 / (k + inverse_ranks[pair])
        out[pair] = score
    return out


class MatcherBidirectionalConsolidation(MatcherBase):
    """
    Bidirectional subsumption retrieval with optional RRF fusion.

    Parameters
    ----------
    model : str
        SentenceTransformer model id or local path.
    broader_query_instruction : str
        Instruction prepended to the source-side encoding in the forward
        pass (= "Pass 1 — broader anchored at source").
    narrower_query_instruction : str
        Instruction prepended to the target-side encoding in the inverse
        pass (= "Pass 2 — narrower anchored at target"). Only used when
        fusion=True.
    document_instruction : str, default ""
        Instruction prepended to the document-side encoding in BOTH passes.
        Forward pass uses it on the target side; inverse pass uses it on
        the source side. Empty per the BeyondEquivalence Stage-1 spec.
    fusion : bool, default True
        True  = Pass 1 + Pass 2 + RRF fusion. This is "C=on".
        False = Pass 1 only. Confidence is still RRF-shaped (a single
                summand 1/(k + rank)) so the score column stays in the same
                numerical range as fusion=True. This is "C=off".
    rrf_k : int, default 60
        Rank-fusion constant. Lower k = more weight on the head of each
        list. 60 is the Cormack et al. (SIGIR 2009) default.
    description : str, default "description_one_gen"
        RDFGraphWrapper method name used to serialise each class to text.
    top_k : int, default 20
        Top-K per source after fusion. Each pass internally also retrieves
        top_k candidates so the fusion sees the same head depth from both
        directions.
    kg_format : str, default "turtle"
        Serialisation format passed to RDFGraphWrapper.serialize for
        Graph-shaped descriptions. String-shaped descriptions
        (description_path_context) pass through unchanged.
    """

    def __init__(
        self,
        model: str,
        *,
        broader_query_instruction: str,
        narrower_query_instruction: str,
        document_instruction: str = "",
        fusion: bool = True,
        rrf_k: int = RRF_K_DEFAULT,
        description: str = "description_one_gen",
        top_k: int = 20,
        kg_format: str = "turtle",
    ):
        super().__init__()
        self.model = model
        self.broader_query_instruction = broader_query_instruction
        self.narrower_query_instruction = narrower_query_instruction
        self.document_instruction = document_instruction
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.description = description
        self.top_k = top_k
        self.kg_format = kg_format

        self._embedder = None
        self.last_run_metrics: dict[str, Any] = {}

    def _ensure_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            loader_kwargs = get_loader_kwargs(self.model)
            logger.info(
                "Loading SentenceTransformer model='%s' loader_kwargs=%s",
                self.model, loader_kwargs,
            )
            self._embedder = SentenceTransformer(self.model, trust_remote_code=True, **loader_kwargs)
            _verify_loader_kwargs_applied(self._embedder, self.model, loader_kwargs)

    def _serialize(self, kg: RDFGraphWrapper, classes: list) -> list[str]:
        method = getattr(kg, self.description)
        return [RDFGraphWrapper.serialize(method(cls), format=self.kg_format) for cls in classes]

    def _encode(self, texts: list[str], instruction: str, *, role: str) -> torch.Tensor:
        prompt: Optional[str] = build_instruct_query_prompt(instruction) or None
        trunc = _truncation_stats(self._embedder, prompt, texts)
        self.last_run_metrics[f"tokens_truncated/{role}/count"] = trunc["count"]
        self.last_run_metrics[f"tokens_truncated/{role}/max"]   = trunc["max"]
        self.last_run_metrics[f"tokens_truncated/{role}/limit"] = trunc["limit"]
        if trunc["count"] > 0:
            logger.warning(
                "Truncation: %d/%d texts (%s side) exceed max_seq_length=%d (max=%d).",
                trunc["count"], len(texts), role, trunc["limit"], trunc["max"],
            )
        embeddings = self._embedder.encode(
            texts, prompt=prompt, convert_to_tensor=True, show_progress_bar=False,
        )
        return util.normalize_embeddings(embeddings)

    def match(
        self,
        kg_source: RDFGraphWrapper,
        kg_target: RDFGraphWrapper,
        input_alignment: Alignment,
        parameters: dict[str, Any] = None,
    ) -> Alignment:
        self._ensure_embedder()
        # Reset per-run metrics so a long-lived matcher doesn't leak previous
        # iteration's truncation/encoding numbers into the next run.
        self.last_run_metrics = {}

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()

        source_elements = sorted(kg_source.get_classes(), key=str)
        target_elements = sorted(kg_target.get_classes(), key=str)
        logger.info("Source classes: %d, target classes: %d", len(source_elements), len(target_elements))

        source_texts = self._serialize(kg_source, source_elements)
        target_texts = self._serialize(kg_target, target_elements)

        # ── Pass 1 — forward (broader-anchored at source). ────────────────
        # forward_ranks[(s_idx, t_idx)] = 1-based rank of t in s's top-K.
        _sync(); t0 = time.perf_counter()
        src_emb_broader = self._encode(source_texts, self.broader_query_instruction, role="source_broader")
        _sync(); t_src_broader = time.perf_counter() - t0

        _sync(); t0 = time.perf_counter()
        tgt_emb_doc = self._encode(target_texts, self.document_instruction, role="target")
        _sync(); t_tgt_doc = time.perf_counter() - t0

        forward_hits = util.semantic_search(
            src_emb_broader, tgt_emb_doc, top_k=self.top_k, score_function=util.dot_score,
        )
        _sync()

        forward_ranks: dict[tuple[int, int], int] = {}
        for s_idx, hits in enumerate(forward_hits):
            for rank_zero, h in enumerate(hits):
                forward_ranks[(s_idx, int(h["corpus_id"]))] = rank_zero + 1

        # ── Pass 2 — inverse (narrower-anchored at target). C=on only. ────
        inverse_ranks: dict[tuple[int, int], int] = {}
        t_tgt_narrower = t_src_doc = 0.0
        if self.fusion:
            _sync(); t0 = time.perf_counter()
            tgt_emb_narrower = self._encode(target_texts, self.narrower_query_instruction, role="target_narrower")
            _sync(); t_tgt_narrower = time.perf_counter() - t0

            _sync(); t0 = time.perf_counter()
            src_emb_doc = self._encode(source_texts, self.document_instruction, role="source")
            _sync(); t_src_doc = time.perf_counter() - t0

            inverse_hits = util.semantic_search(
                tgt_emb_narrower, src_emb_doc, top_k=self.top_k, score_function=util.dot_score,
            )
            _sync()

            for t_idx, hits in enumerate(inverse_hits):
                for rank_zero, h in enumerate(hits):
                    s_idx = int(h["corpus_id"])
                    # Map onto the same (s, t, '<') schlüsselraum as Pass 1.
                    inverse_ranks[(s_idx, t_idx)] = rank_zero + 1

        # ── RRF fusion. Single-summand when fusion=False; same range. ─────
        rrf_scores = reciprocal_rank_fusion(forward_ranks, inverse_ranks, k=self.rrf_k)

        # Top-K per source after fusion.
        by_source: dict[int, list[tuple[int, float]]] = {}
        for (s_idx, t_idx), score in rrf_scores.items():
            by_source.setdefault(s_idx, []).append((t_idx, score))

        alignment = Alignment()
        n_pairs_total = len(rrf_scores)
        n_pairs_emitted = 0
        for s_idx in range(len(source_elements)):
            items = by_source.get(s_idx, [])
            items.sort(key=lambda x: (-x[1], x[0]))
            for t_idx, score in items[: self.top_k]:
                alignment.add(Correspondence(
                    str(source_elements[s_idx]),
                    str(target_elements[t_idx]),
                    "<",
                    float(score),
                ))
                n_pairs_emitted += 1

        peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if cuda_available else None
        emb_dim = int(src_emb_broader.shape[1])
        n_overlap = sum(1 for p in forward_ranks if p in inverse_ranks)
        self.last_run_metrics.update({
            "n_source_classes": len(source_elements),
            "n_target_classes": len(target_elements),
            "encode_source_broader_seconds": t_src_broader,
            "encode_target_doc_seconds":      t_tgt_doc,
            "encode_target_narrower_seconds": t_tgt_narrower,
            "encode_source_doc_seconds":      t_src_doc,
            "fusion_enabled": self.fusion,
            "rrf_k": self.rrf_k,
            "n_pairs_forward":  len(forward_ranks),
            "n_pairs_inverse":  len(inverse_ranks),
            "n_pairs_overlap":  n_overlap,
            "n_pairs_total":    n_pairs_total,
            "n_pairs_emitted":  n_pairs_emitted,
            "embedding_dim": emb_dim,
            "gpu_peak_memory_gb": peak_gb,
        })
        logger.info(
            "fusion=%s n_forward=%d n_inverse=%d overlap=%d total_pairs=%d emitted=%d alignment_size=%d",
            self.fusion, len(forward_ranks), len(inverse_ranks), n_overlap,
            n_pairs_total, n_pairs_emitted, len(alignment),
        )
        return alignment

    def __str__(self):
        model_short = self.model.split("/")[-1]
        return (
            f"MatcherBidirectionalConsolidation#{model_short}"
            f"#{self.description}#fusion={int(self.fusion)}#rrf_k={self.rrf_k}"
            f"#k={self.top_k}"
        )


# ── Inline unit test — run via:  python MatcherBidirectionalConsolidation.py ──
if __name__ == "__main__":
    # Verify the RRF math against hand-computed values, no model needed.
    forward = {
        ("a", "x"): 1,   # source 'a' ranks 'x' first
        ("a", "y"): 2,
        ("b", "x"): 1,
    }
    inverse = {
        ("a", "x"): 2,   # target 'x' ranks 'a' second
        ("b", "z"): 1,   # target 'z' ranks 'b' first; 'z' was not in forward
    }
    scores = reciprocal_rank_fusion(forward, inverse, k=60)
    expected = {
        ("a", "x"): 1.0/61 + 1.0/62,   # ≈ 0.03253
        ("a", "y"): 1.0/62,            # ≈ 0.01613
        ("b", "x"): 1.0/61,            # ≈ 0.01639
        ("b", "z"): 1.0/61,            # ≈ 0.01639
    }
    print("RRF unit test (k=60):")
    ok = True
    for pair, exp in expected.items():
        got = scores[pair]
        match = abs(got - exp) < 1e-9
        ok = ok and match
        print(f"  {pair}: got={got:.6f}  expected={exp:.6f}  {'OK' if match else 'FAIL'}")
    if not ok:
        raise SystemExit(1)
    # Range sanity: with k=60 and ranks in [1, 20], a single-pass score is
    # in [1/80, 1/61] = [0.0125, 0.01639]; a two-pass full overlap with both
    # ranks at 1 gives 2/61 ≈ 0.0328. The thesis-spec range "0.01 to 0.03"
    # therefore brackets the single-summand and equal-double-summand cases.
    assert 1.0 / (60 + 20) < min(scores.values())
    assert max(scores.values()) <= 2.0 / 61
    print("Range sanity OK.")
