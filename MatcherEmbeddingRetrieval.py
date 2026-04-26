"""
MatcherEmbeddingRetrieval — pure embedding-based candidate retrieval for the
BeyondEquivalence subsumption study.

Differences vs. the existing MatcherCandidateGen (intentional):
  - No MatcherSimple lexical mixin appended at the end.
  - No automatic both_directions reruns; this matcher does exactly one
    source -> target retrieval pass.
  - Output relation is configurable (default '=') so the asymmetric wrapper
    can stamp '<' / '>' onto its two runs.
  - Instructions are passed as TEXT (not prompt-id), formatted per model family
    via prompt.format_instruction; the matcher does not consult EMBEDDING_PROMPTS.
  - SentenceTransformer is loaded lazily at first match() call so this module
    is importable on a MacBook without the heavy model weights present.

The output Alignment contains, for each source class, top-K target candidates
with scores as confidences (NOT binarised). Use Alignment.find_by_source(src)
to recover the per-source ranking; the runner relies on this for Recall@K.
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
from prompt import format_instruction, get_loader_kwargs, infer_model_family

logger = logging.getLogger(__name__)


def _sync() -> None:
    """Block until queued accelerator work is done, so wall-clock timings are honest.

    SentenceTransformer.encode(..., convert_to_tensor=True) returns asynchronously on
    CUDA/MPS — without a sync, time.perf_counter() captures kernel-submission, not
    real encoding time, and torch.cuda.max_memory_allocated() can underreport peak.
    No-op on CPU.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


class MatcherEmbeddingRetrieval(MatcherBase):
    def __init__(
        self,
        model: str,
        *,
        model_family: str | None = None,
        description: str = "description_one_gen",
        query_instruction: str = "",
        document_instruction: str = "",
        output_relation: str = "=",
        top_k: int = 20,
        kg_format: str = "turtle",
    ):
        super().__init__()
        self.model = model
        # Family resolution: explicit override wins; otherwise infer (with WARNING on unknown).
        self.model_family = model_family if model_family is not None else infer_model_family(model)
        self.description = description
        self.query_instruction = query_instruction
        self.document_instruction = document_instruction
        self.output_relation = output_relation
        self.top_k = top_k
        self.kg_format = kg_format

        self._embedder = None
        # Filled in by match(); the runner reads this for W&B side-metric logging.
        self.last_run_metrics: dict[str, Any] = {}

    def _ensure_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            loader_kwargs = get_loader_kwargs(self.model_family)
            logger.info(
                "Loading SentenceTransformer model='%s' (family=%s) loader_kwargs=%s",
                self.model, self.model_family, loader_kwargs,
            )
            self._embedder = SentenceTransformer(self.model, trust_remote_code=True, **loader_kwargs)

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

        # Sort by URI string for run-to-run determinism (encoding-time variance, not correctness).
        source_elements = sorted(kg_source.get_classes(), key=str)
        target_elements = sorted(kg_target.get_classes(), key=str)
        logger.info("Source classes: %d, target classes: %d", len(source_elements), len(target_elements))

        source_texts = self._serialize(kg_source, source_elements)
        target_texts = self._serialize(kg_target, target_elements)

        _sync()
        t0 = time.perf_counter()
        source_emb = self._encode(source_texts, self.query_instruction)
        _sync()
        t_src = time.perf_counter() - t0

        t0 = time.perf_counter()
        target_emb = self._encode(target_texts, self.document_instruction)
        _sync()
        t_tgt = time.perf_counter() - t0

        hits = util.semantic_search(
            source_emb, target_emb, top_k=self.top_k, score_function=util.dot_score,
        )
        _sync()

        alignment = Alignment()
        for src_hits, src_cls in zip(hits, source_elements):
            for h in src_hits:
                alignment.add(Correspondence(
                    str(src_cls),
                    str(target_elements[h["corpus_id"]]),
                    self.output_relation,
                    float(h["score"]),
                ))

        peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if cuda_available else None
        emb_dim = int(source_emb.shape[1])
        self.last_run_metrics = {
            "n_source_classes": len(source_elements),
            "n_target_classes": len(target_elements),
            "encode_source_seconds": t_src,
            "encode_target_seconds": t_tgt,
            "source_vecs_per_sec": (len(source_elements) / t_src) if t_src > 0 else 0.0,
            "target_vecs_per_sec": (len(target_elements) / t_tgt) if t_tgt > 0 else 0.0,
            "embedding_dim": emb_dim,
            "gpu_peak_memory_gb": peak_gb,
        }
        logger.info(
            "Encoded source in %.2fs (%.0f vec/s), target in %.2fs (%.0f vec/s); dim=%d",
            t_src, self.last_run_metrics["source_vecs_per_sec"],
            t_tgt, self.last_run_metrics["target_vecs_per_sec"], emb_dim,
        )
        return alignment

    def __str__(self):
        model_short = self.model.split("/")[-1]
        return (
            f"MatcherEmbeddingRetrieval#{model_short}#fam={self.model_family}"
            f"#{self.description}#qi={1 if self.query_instruction else 0}"
            f"#di={1 if self.document_instruction else 0}"
            f"#rel={self.output_relation}#k={self.top_k}"
        )
