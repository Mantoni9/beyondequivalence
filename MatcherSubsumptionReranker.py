"""
MatcherSubsumptionReranker — Stage-2 multi-class relation classifier.

Distinct from MatcherLLMReranker (binary equivalence P(yes)). For each
candidate (source, target) pair, the LLM emits a label from
{subclass, superclass, equivalent, partof, none}; the argmax-by-generation
becomes the predicted relation. 'none' is a full fifth class — it is the
"drop this candidate" signal, NOT a confidence-threshold artefact.

Extraction path: text generation + parse (see prompt.parse_relation_label).
The first-token logit comparison used by MatcherLLMReranker is unfair for
reasoner models (gpt-oss, Gemma-4-thinking) that emit chain-of-thought
before the answer; a uniform generation path is the only way to compare
reasoners and non-reasoners on equal terms.

Dedup contract: Stage-1 may emit two correspondences per (source, target)
pair (one from the broader '<' pass, one from the narrower '>' pass). This
matcher deduplicates to unique (source, target) pairs before reranking —
the Stage-1 direction is the reason the pair is a candidate, not an input
to the Stage-2 decision. Stage 2 resolves direction independently.

Output: one Correspondence per kept (source, target) with
  - relation:   RELATION_LABEL_TO_RELATION[canonical_label]
                  (one of {'=', '<', '>', 'partof'}; 'partof' stays literal
                   so evaluation_multiclass can fold it into 'none' for the
                   displayed 4x4 confusion matrix.)
  - confidence: exp(mean token_logprob) of the generated response, in [0, 1].
                Smoke-time proxy; refine to label-token sub-sum later.

Side channel: ``self.last_run_details`` holds one dict per input candidate
(including dropped ones), populated by the runner into predictions.tsv.
"""
from __future__ import annotations

import logging
import math
from typing import Any, List

from rdflib.term import URIRef

from Alignment import Alignment
from Correspondence import Correspondence
from MatcherBase import MatcherBase
from RDFGraphWrapper import RDFGraphWrapper
from prompt import (
    Prompt,
    RELATION_LABEL_TO_RELATION,
    get_reranking_prompt,
    parse_relation_label,
)

logger = logging.getLogger(__name__)


def relation_for_canonical_pair(canonical_label: str, swapped: bool) -> str:
    """Map a parsed canonical label to the relation of the CANONICAL (s, t)
    pair. With swap_pair_presentation the model judged the PRESENTED pair
    (t in the source slots, s in the target slots), so the two directional
    labels invert EXACTLY ONCE here — '='/'partof'/'none'/'parse_fail' are
    symmetric resp. drops and stay unchanged (Stufe-A arm A2, registered
    2026-06-12)."""
    relation = RELATION_LABEL_TO_RELATION.get(canonical_label, "")
    if swapped and relation in ("<", ">"):
        relation = ">" if relation == "<" else "<"
    return relation


class MatcherSubsumptionReranker(MatcherBase):
    """LLM-based multi-class relation classifier over a deduplicated
    Stage-1 candidate set.
    """

    def __init__(
        self,
        llm,                                # duck-typed: LLMOpenAI or LLMHuggingFace
        prompt_id: str = "d_subs",
        description: str = "description_one_gen",
        kg_format: str = "turtle",
        max_new_tokens: int = 256,
        threshold: float = 0.0,             # optional confidence cutoff, applied
                                            # AFTER the 'none' filter — never the
                                            # 'none' mechanism. Keep at 0.0 for smoke.
        batch_size: int = 8,
        swap_pair_presentation: bool = False,
                                            # Stufe-A arm A2: fill the prompt
                                            # slots with (target, source) instead
                                            # of (source, target); directional
                                            # labels invert exactly once at parse
                                            # time (relation_for_canonical_pair).
                                            # Prompt text and verbalizations are
                                            # untouched.
        temperature: float = 0.0,           # Stage-2 matrix decoding: reasoners
        top_p: float | None = None,         # model-recommended (temp>0),
                                            # non-reasoners temp=0 (default).
    ):
        self.llm = llm
        self.prompt_template: Prompt = get_reranking_prompt(prompt_id)
        self.prompt_id = prompt_id
        self.description = description
        self.kg_format = kg_format
        self.max_new_tokens = max_new_tokens
        self.threshold = threshold
        self.batch_size = batch_size
        self.swap_pair_presentation = swap_pair_presentation
        self.temperature = temperature
        self.top_p = top_p

        # Filled by match(); the runner reads this for predictions.tsv.
        self.last_run_details: List[dict] = []

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dedup_alignment(input_alignment: Alignment) -> list[tuple[str, str, dict]]:
        """Collapse Stage-1 (s, t) duplicates. Returns
        [(source, target, evidence_dict), ...] in deterministic order.

        For each unique (source, target), evidence_dict records what Stage-1
        produced so we can log it in predictions.tsv:
          - stage1_relations:    comma-separated, sorted (e.g. "<,>")
          - stage1_max_confidence: max confidence across the duplicates
        """
        grouped: dict[tuple[str, str], dict] = {}
        for cor in input_alignment:
            key = (cor.source, cor.target)
            if key not in grouped:
                grouped[key] = {
                    "stage1_relations":     [cor.relation],
                    "stage1_max_confidence": float(cor.confidence),
                }
            else:
                grouped[key]["stage1_relations"].append(cor.relation)
                grouped[key]["stage1_max_confidence"] = max(
                    grouped[key]["stage1_max_confidence"], float(cor.confidence),
                )

        out = []
        for (src, tgt), ev in grouped.items():
            ev["stage1_relations"] = ",".join(sorted(set(ev["stage1_relations"])))
            out.append((src, tgt, ev))
        # Deterministic order.
        out.sort(key=lambda r: (r[0], r[1]))
        return out

    def _get_entity_text(self, kg: RDFGraphWrapper, uri: str) -> str:
        """Serialize the entity's sub-graph. Supports both Graph-returning
        description methods (description_one_gen etc.) and str-returning ones
        (description_path_context — already a verbalisation, no serialize)."""
        method = getattr(kg, self.description)
        result = method(URIRef(uri))
        if isinstance(result, str):
            return result
        return RDFGraphWrapper.serialize(result, format=self.kg_format)

    def _build_prompt(
        self, source_uri: str, target_uri: str,
        source_kg_text: str, target_kg_text: str,
    ) -> Prompt:
        return self.prompt_template.format(
            source_url=source_uri,
            target_url=target_uri,
            source_kg=source_kg_text,
            target_kg=target_kg_text,
        )

    def _score_in_batches(self, prompts: List[Prompt]) -> List[dict]:
        """Run the LLM in chunks of ``batch_size``. Each item is the
        per-prompt dict returned by ``get_text_completion_with_logprobs``.
        """
        results: List[dict] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            batch_results = self.llm.get_text_completion_with_logprobs(
                batch, max_new_tokens=self.max_new_tokens,
                temperature=self.temperature, top_p=self.top_p,
            )
            results.extend(batch_results)
        return results

    @staticmethod
    def _confidence_from_logprobs(token_logprobs: list[float]) -> float:
        """exp(mean logprob): geometric-mean per-token probability, in [0, 1].

        Smoke-time proxy: it answers "how predictable was each token on
        average?" rather than "how confident is the model in the label
        specifically". For sharper signal, point at the label-token sub-sum
        once parsing locates the label token range in the generation."""
        if not token_logprobs:
            return 0.0
        return float(math.exp(sum(token_logprobs) / len(token_logprobs)))

    # ------------------------------------------------------------------ #
    #  MatcherBase interface                                              #
    # ------------------------------------------------------------------ #

    def match(
        self,
        kg_source: RDFGraphWrapper,
        kg_target: RDFGraphWrapper,
        input_alignment: Alignment,
        parameters: dict[str, Any] = None,
    ) -> Alignment:
        candidates = self._dedup_alignment(input_alignment)
        logger.info(
            "MatcherSubsumptionReranker: input=%d corr, deduped=%d unique (s,t) pairs "
            "(prompt=%s, description=%s, max_new_tokens=%d, threshold=%.3f)",
            len(input_alignment), len(candidates),
            self.prompt_id, self.description, self.max_new_tokens, self.threshold,
        )

        prompts: List[Prompt] = []
        for src, tgt, _ev in candidates:
            # Verbalization is a function of (kg, concept) ONLY — identical
            # string regardless of which slot it lands in (A2 identity guard).
            source_text = self._get_entity_text(kg_source, src)
            target_text = self._get_entity_text(kg_target, tgt)
            if self.swap_pair_presentation:
                prompts.append(self._build_prompt(tgt, src, target_text, source_text))
            else:
                prompts.append(self._build_prompt(src, tgt, source_text, target_text))

        results = self._score_in_batches(prompts)

        output = Alignment()
        self.last_run_details = []
        n_kept = n_none = n_parse_fail = n_below_threshold = n_partof = 0
        per_class_counts = {"subclass": 0, "superclass": 0, "equivalent": 0,
                            "partof": 0, "none": 0, "parse_fail": 0}

        for (src, tgt, ev), res in zip(candidates, results):
            text = res.get("text", "") or ""
            token_logprobs = res.get("token_logprobs", []) or []
            sum_lp = float(res.get("sum_logprob", 0.0))
            n_tok = int(res.get("n_tokens", 0))

            canonical = parse_relation_label(text)
            per_class_counts[canonical] = per_class_counts.get(canonical, 0) + 1

            confidence = self._confidence_from_logprobs(token_logprobs)
            # The ONLY place the A2 inversion is applied (exactly once).
            relation = relation_for_canonical_pair(canonical, self.swap_pair_presentation)

            kept = False
            drop_reason = ""
            if canonical == "parse_fail":
                # No "Relation: <label>" anchor found. Distinct from an explicit
                # "Relation: none" reply — both drop, but we track separately so
                # a high parse_fail rate surfaces a format-compliance regression
                # directly instead of getting silently lumped under 'none'.
                n_parse_fail += 1
                drop_reason = "parse_fail"
            elif canonical == "none":
                n_none += 1
                drop_reason = "none"
            elif confidence < self.threshold:
                n_below_threshold += 1
                drop_reason = f"threshold({confidence:.4f}<{self.threshold:.3f})"
            else:
                # 'partof' is emitted but the eval module folds it into 'none'.
                # We still keep it in the output Alignment so the runner can
                # surface its rate in metrics.json.
                output.add(Correspondence(src, tgt, relation, confidence))
                kept = True
                n_kept += 1
                if canonical == "partof":
                    n_partof += 1

            self.last_run_details.append({
                "source": src,
                "target": tgt,
                "pair_presentation":     ("swapped" if self.swap_pair_presentation
                                          else "canonical"),
                "stage1_relations":      ev["stage1_relations"],
                "stage1_max_confidence": ev["stage1_max_confidence"],
                "raw_response":          text,
                "parsed_canonical":      canonical,
                "predicted_relation":    relation,
                "confidence":            confidence,
                "sum_logprob":           sum_lp,
                "n_tokens":              n_tok,
                "kept":                  kept,
                "drop_reason":           drop_reason,
            })

        logger.info(
            "MatcherSubsumptionReranker: kept=%d  none=%d  parse_fail=%d  "
            "below_threshold=%d  partof_kept=%d  (per-class %s)",
            n_kept, n_none, n_parse_fail, n_below_threshold, n_partof,
            per_class_counts,
        )
        if n_parse_fail > 0:
            pct = 100 * n_parse_fail / len(candidates)
            level = logger.warning if pct > 5.0 else logger.info
            level(
                "MatcherSubsumptionReranker: parse_fail rate = %d/%d (%.1f%%). "
                "High values indicate the LLM is not emitting the required "
                "'Relation: <label>' anchor — check max_new_tokens, prompt "
                "format compliance, and whether truncation is occurring.",
                n_parse_fail, len(candidates), pct,
            )
        return output

    def __str__(self) -> str:
        return (
            f"MatcherSubsumptionReranker#p{self.prompt_id}#d{self.description}"
            f"#mnt{self.max_new_tokens}#t{self.threshold}#b{self.batch_size}"
            + ("#SWAPPED-PRESENTATION" if self.swap_pair_presentation else "")
        )
