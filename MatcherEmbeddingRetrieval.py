"""
MatcherEmbeddingRetrieval — pure embedding-based candidate retrieval for the
BeyondEquivalence subsumption study.

Differences vs. the existing MatcherCandidateGen (intentional):
  - No MatcherSimple lexical mixin appended at the end.
  - No automatic both_directions reruns; this matcher does exactly one
    source -> target retrieval pass.
  - Output relation is configurable (default '=') so the asymmetric wrapper
    can stamp '<' / '>' onto its two runs.
  - Instructions are passed as TEXT and applied via SentenceTransformer's
    native `encode(prompt=...)` parameter — the library handles pooling and
    normalisation, including last-token pool + left-padding on Nemotron / Qwen3.
    No per-family wrapping helper is involved (Option C, decided 2026-04-29).
  - SentenceTransformer is loaded lazily at first match() call so this module
    is importable on a MacBook without the heavy model weights present.

The output Alignment contains, for each source class, top-K target candidates
with scores as confidences (NOT binarised). Use Alignment.find_by_source(src)
to recover the per-source ranking; the runner relies on this for Recall@K.
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
from prompt import build_instruct_query_prompt, get_loader_kwargs

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


def _truncation_stats(embedder, prompt: Optional[str], texts: list[str]) -> dict:
    """Pre-encode tokenizer pass — counts how many texts exceed max_seq_length
    once the prompt prefix is prepended. Used for the Sub-B description-ablation
    so we can distinguish a recall drop on description_three_gen from a
    truncation artefact.
    """
    tokenizer = embedder.tokenizer
    max_len = getattr(embedder, "max_seq_length", None) or getattr(tokenizer, "model_max_length", 0)
    full_texts = [(prompt or "") + t for t in texts]
    enc = tokenizer(full_texts, truncation=False, padding=False,
                    return_length=True, add_special_tokens=True)
    lengths = enc.get("length") or [len(ids) for ids in enc["input_ids"]]
    n_truncated = sum(1 for L in lengths if L > max_len)
    return {
        "count": int(n_truncated),
        "max":   int(max(lengths)) if lengths else 0,
        "limit": int(max_len),
    }


def _verify_loader_kwargs_applied(embedder, model_id: str, loader_kwargs: dict) -> None:
    """Hard-fail if a declared tokenizer pin (padding_side) didn't survive load.

    Pragmatic check, scoped to what we actually depend on: Nemotron and Qwen3
    require padding_side="left" for last-token pooling. If `loader_kwargs`
    declares it but the tokenizer reports something else, the embeddings would
    be silently wrong. Fail loud at load time, never trust ST defaults.
    """
    declared = loader_kwargs.get("tokenizer_kwargs", {}).get("padding_side")
    if declared is None:
        return
    actual = getattr(getattr(embedder, "tokenizer", None), "padding_side", None)
    if actual != declared:
        raise RuntimeError(
            f"SentenceTransformer for model='{model_id}' loaded with "
            f"tokenizer.padding_side={actual!r}, but the model loader_kwargs "
            f"declared padding_side={declared!r}. Last-token pooling expects "
            f"the declared value; aborting to avoid silently wrong embeddings."
        )


class MatcherEmbeddingRetrieval(MatcherBase):
    def __init__(
        self,
        model: str,
        *,
        description: str = "description_one_gen",
        query_instruction: str = "",
        document_instruction: str = "",
        output_relation: str = "=",
        top_k: int = 20,
        kg_format: str = "turtle",
    ):
        super().__init__()
        self.model = model
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
            loader_kwargs = get_loader_kwargs(self.model)
            logger.info(
                "Loading SentenceTransformer model='%s' loader_kwargs=%s",
                self.model, loader_kwargs,
            )
            self._embedder = SentenceTransformer(self.model, trust_remote_code=True, **loader_kwargs)
            _verify_loader_kwargs_applied(self._embedder, self.model, loader_kwargs)

    def release(self) -> None:
        """Free the embedder reference and the underlying GPU memory.

        Idempotent. Subsequent match() calls reload the model lazily via
        _ensure_embedder. Sweep drivers MUST call this on a matcher whose
        embedder would otherwise be co-resident in VRAM with another
        matcher's embedder of the same or different model — without it,
        two 8B models in a 47-GB A6000 trigger a CUDA-OOM in
        SentenceTransformer.__init__ on the second matcher's load
        (observed 2026-05-04 in the A-sweep nemo asym crashes).
        """
        if self._embedder is not None:
            logger.info("Releasing embedder for model='%s'", self.model)
            del self._embedder
            self._embedder = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _serialize(self, kg: RDFGraphWrapper, classes: list) -> list[str]:
        method = getattr(kg, self.description)
        return [RDFGraphWrapper.serialize(method(cls), format=self.kg_format) for cls in classes]

    def _encode(self, texts: list[str], instruction: str, *, role: str = "source") -> torch.Tensor:
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

        # Sort by URI string for run-to-run determinism (encoding-time variance, not correctness).
        source_elements = sorted(kg_source.get_classes(), key=str)
        target_elements = sorted(kg_target.get_classes(), key=str)
        logger.info("Source classes: %d, target classes: %d", len(source_elements), len(target_elements))

        source_texts = self._serialize(kg_source, source_elements)
        target_texts = self._serialize(kg_target, target_elements)

        _sync()
        t0 = time.perf_counter()
        source_emb = self._encode(source_texts, self.query_instruction, role="source")
        _sync()
        t_src = time.perf_counter() - t0

        t0 = time.perf_counter()
        target_emb = self._encode(target_texts, self.document_instruction, role="target")
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
        # Merge with existing truncation entries set by _encode.
        self.last_run_metrics.update({
            "n_source_classes": len(source_elements),
            "n_target_classes": len(target_elements),
            "encode_source_seconds": t_src,
            "encode_target_seconds": t_tgt,
            "source_vecs_per_sec": (len(source_elements) / t_src) if t_src > 0 else 0.0,
            "target_vecs_per_sec": (len(target_elements) / t_tgt) if t_tgt > 0 else 0.0,
            "embedding_dim": emb_dim,
            "gpu_peak_memory_gb": peak_gb,
        })
        logger.info(
            "Encoded source in %.2fs (%.0f vec/s), target in %.2fs (%.0f vec/s); dim=%d",
            t_src, self.last_run_metrics["source_vecs_per_sec"],
            t_tgt, self.last_run_metrics["target_vecs_per_sec"], emb_dim,
        )
        return alignment

    def __str__(self):
        model_short = self.model.split("/")[-1]
        return (
            f"MatcherEmbeddingRetrieval#{model_short}"
            f"#{self.description}#qi={1 if self.query_instruction else 0}"
            f"#di={1 if self.document_instruction else 0}"
            f"#rel={self.output_relation}#k={self.top_k}"
        )
