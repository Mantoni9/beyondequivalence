"""
MatcherAsymmetricRetrieval — two-run asymmetric embedding retrieval for the
BeyondEquivalence subsumption study.

A single matcher instance executes two retrieval passes against a shared model
and a shared document-side encoding:

  - "broader"  run: source encoded with broader_query_instruction, output
                    relation '<' (source ⊑ target — target is broader / more
                    general than source).
  - "narrower" run: source encoded with narrower_query_instruction, output
                    relation '>' (source ⊒ target — target is narrower / more
                    specific than source).

The two top-K result lists are unioned into a single Alignment. Both
predictions for the same (source, target) pair coexist when both runs return
that target — they live as separate Correspondences keyed by relation, which
is what evaluation_recall.compute_recall_at_k expects (one row per ranked
prediction in the per-source long-format table).

Cost model — verified by design, not measured: each match() invocation does
ONE document encoding (shared) plus TWO source encodings. Two naive
MatcherEmbeddingRetrieval runs would do TWO of each — i.e. this wrapper saves
one full target encoding pass, which on Qwen3-8B / NV-Embed-v2 is the
dominant cost for the smaller g7-style sub-datasets.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from sentence_transformers import util

from MatcherBase import MatcherBase
from RDFGraphWrapper import RDFGraphWrapper
from Alignment import Alignment
from Correspondence import Correspondence
from MatcherEmbeddingRetrieval import _sync
from prompt import format_instruction, infer_model_family

logger = logging.getLogger(__name__)


class MatcherAsymmetricRetrieval(MatcherBase):
    def __init__(
        self,
        model: str,
        *,
        broader_query_instruction: str,
        narrower_query_instruction: str,
        document_instruction: str = "",
        model_family: str | None = None,
        description: str = "description_one_gen",
        top_k: int = 20,
        kg_format: str = "turtle",
    ):
        super().__init__()
        self.model = model
        self.model_family = model_family if model_family is not None else infer_model_family(model)
        self.broader_query_instruction = broader_query_instruction
        self.narrower_query_instruction = narrower_query_instruction
        self.document_instruction = document_instruction
        self.description = description
        self.top_k = top_k
        self.kg_format = kg_format

        self._embedder = None
        self.last_run_metrics: dict[str, Any] = {}

    def _ensure_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer model='%s' (family=%s)", self.model, self.model_family)
            self._embedder = SentenceTransformer(self.model, trust_remote_code=True)

    def _serialize(self, kg: RDFGraphWrapper, classes: list) -> list[str]:
        method = getattr(kg, self.description)
        return [RDFGraphWrapper.serialize(method(cls), format=self.kg_format) for cls in classes]

    def _encode(self, texts: list[str], instruction: str) -> torch.Tensor:
        formatted = [format_instruction(self.model_family, instruction, t) for t in texts]
        embeddings = self._embedder.encode(formatted, convert_to_tensor=True, show_progress_bar=False)
        return util.normalize_embeddings(embeddings)

    def match(
        self,
        kg_source: RDFGraphWrapper,
        kg_target: RDFGraphWrapper,
        input_alignment: Alignment,
        parameters: dict[str, Any] = None,
    ) -> Alignment:
        self._ensure_embedder()

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()

        source_elements = sorted(kg_source.get_classes(), key=str)
        target_elements = sorted(kg_target.get_classes(), key=str)
        logger.info("Source classes: %d, target classes: %d", len(source_elements), len(target_elements))

        source_texts = self._serialize(kg_source, source_elements)
        target_texts = self._serialize(kg_target, target_elements)

        # Document side: encoded ONCE, reused across both runs.
        _sync()
        t0 = time.perf_counter()
        target_emb = self._encode(target_texts, self.document_instruction)
        _sync()
        t_target = time.perf_counter() - t0

        # The two asymmetric runs.
        # output_relation '<' = source ⊑ target (target is broader); '>' = source ⊒ target.
        runs = [
            ("broader",  self.broader_query_instruction,  "<"),
            ("narrower", self.narrower_query_instruction, ">"),
        ]

        alignment = Alignment()
        per_run_metrics: dict[str, dict[str, float]] = {}

        for run_name, query_instr, output_rel in runs:
            _sync()
            t0 = time.perf_counter()
            source_emb = self._encode(source_texts, query_instr)
            _sync()
            t_src = time.perf_counter() - t0

            hits = util.semantic_search(
                source_emb, target_emb, top_k=self.top_k, score_function=util.dot_score,
            )
            _sync()

            for src_hits, src_cls in zip(hits, source_elements):
                for h in src_hits:
                    alignment.add(Correspondence(
                        str(src_cls),
                        str(target_elements[h["corpus_id"]]),
                        output_rel,
                        float(h["score"]),
                    ))

            per_run_metrics[run_name] = {
                "encode_source_seconds": t_src,
                "source_vecs_per_sec": (len(source_elements) / t_src) if t_src > 0 else 0.0,
                "output_relation": output_rel,
            }
            logger.info(
                "Run '%s' (rel=%s): encoded source in %.2fs (%.0f vec/s)",
                run_name, output_rel, t_src, per_run_metrics[run_name]["source_vecs_per_sec"],
            )

        peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if cuda_available else None
        emb_dim = int(target_emb.shape[1])
        self.last_run_metrics = {
            "n_source_classes": len(source_elements),
            "n_target_classes": len(target_elements),
            "encode_target_seconds": t_target,
            "target_vecs_per_sec": (len(target_elements) / t_target) if t_target > 0 else 0.0,
            "per_run": per_run_metrics,
            "embedding_dim": emb_dim,
            "gpu_peak_memory_gb": peak_gb,
        }
        logger.info(
            "Encoded target in %.2fs (%.0f vec/s); dim=%d; alignment size=%d",
            t_target, self.last_run_metrics["target_vecs_per_sec"], emb_dim, len(alignment),
        )
        return alignment

    def __str__(self):
        model_short = self.model.split("/")[-1]
        return (
            f"MatcherAsymmetricRetrieval#{model_short}#fam={self.model_family}"
            f"#{self.description}#di={1 if self.document_instruction else 0}"
            f"#k={self.top_k}"
        )
