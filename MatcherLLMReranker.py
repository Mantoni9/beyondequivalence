import logging
from typing import Any, List

from rdflib.term import URIRef

from Alignment import Alignment
from Correspondence import Correspondence
from LLMHuggingFace import LLMHuggingFace
from MatcherBase import MatcherBase
from RDFGraphWrapper import RDFGraphWrapper
from prompt import Prompt, get_reranking_prompt

logger = logging.getLogger(__name__)


class MatcherLLMReranker(MatcherBase):
    """Reranks candidates from *input_alignment* using an LLM.

    For each correspondence in *input_alignment* a prompt is built from the
    KG sub-graphs of both entities.  ``LLMHuggingFace.get_confidence_first_token``
    is used to obtain a yes/no confidence score.  Only correspondences whose
    score meets *threshold* are kept; the original confidence is replaced by the
    LLM score.

    Parameters
    ----------
    llm:
        A ready-to-use :class:`LLMHuggingFace` instance.
    prompt_id:
        Key into ``RERANKING_PROMPTS`` (or a composite ``system_user`` key).
        Defaults to ``"d"``.  The selected prompt must contain the placeholders
        ``{source_url}``, ``{target_url}``, ``{source_kg}``, and ``{target_kg}``.
    description:
        Name of the :class:`RDFGraphWrapper` method used to extract the
        sub-graph for each entity.  Defaults to ``"description_one_gen"``.
    kg_format:
        Serialization format passed to :meth:`RDFGraphWrapper.serialize`.
        Defaults to ``"turtle"``.
    threshold:
        Minimum LLM confidence required to keep a correspondence.
        Defaults to ``0.5``.
    batch_size:
        Number of prompts forwarded to the LLM in a single call.
        Reduce this if you run out of GPU memory.  Defaults to ``8``.
    """

    def __init__(
        self,
        llm: LLMHuggingFace,
        prompt_id: str = "d",
        description: str = "description_one_gen",
        kg_format: str = "turtle",
        threshold: float = 0.5,
        batch_size: int = 8,
    ):
        self.llm = llm
        self.prompt_template: Prompt = get_reranking_prompt(prompt_id)
        self.prompt_id = prompt_id
        self.description = description
        self.kg_format = kg_format
        self.threshold = threshold
        self.batch_size = batch_size

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self,
        source_uri: str,
        target_uri: str,
        source_kg_text: str,
        target_kg_text: str,
    ) -> Prompt:
        return self.prompt_template.format(
            source_url=source_uri,
            target_url=target_uri,
            source_kg=source_kg_text,
            target_kg=target_kg_text,
        )

    def _get_entity_text(self, kg: RDFGraphWrapper, uri: str) -> str:
        method = getattr(kg, self.description)
        subgraph = method(URIRef(uri))
        return RDFGraphWrapper.serialize(subgraph, format=self.kg_format)

    def _score_in_batches(self, prompts: List[Prompt]) -> List[float]:
        scores: List[float] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            batch_scores = self.llm.get_confidence_first_token(batch)
            scores.extend(batch_scores)
        return scores

    # ------------------------------------------------------------------ #
    #  MatcherBase interface                                               #
    # ------------------------------------------------------------------ #

    def match(
        self,
        kg_source: RDFGraphWrapper,
        kg_target: RDFGraphWrapper,
        input_alignment: Alignment,
        parameters: dict[str, Any] = None,
    ) -> Alignment:
        correspondences = list(input_alignment)
        logger.info(
            f"MatcherLLMReranker: scoring {len(correspondences)} candidates "
            f"(prompt={self.prompt_id}, description={self.description}, "
            f"threshold={self.threshold})"
        )

        prompts: List[Prompt] = []
        for corr in correspondences:
            source_text = self._get_entity_text(kg_source, corr.source)
            target_text = self._get_entity_text(kg_target, corr.target)
            prompts.append(
                self._build_prompt(corr.source, corr.target, source_text, target_text)
            )

        scores = self._score_in_batches(prompts)

        output = Alignment()
        kept = 0
        for corr, score in zip(correspondences, scores):
            if score >= self.threshold:
                output.add(Correspondence(corr.source, corr.target, corr.relation, score))
                kept += 1

        logger.info(
            f"MatcherLLMReranker: kept {kept}/{len(correspondences)} correspondences "
            f"above threshold {self.threshold}"
        )
        return output

    def __str__(self) -> str:
        return (
            f"MatcherLLMReranker#p{self.prompt_id}#d{self.description}"
            f"#t{self.threshold}#b{self.batch_size}"
        )
