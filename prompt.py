import json
import logging
from typing import List, Dict, Optional, Any
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class Prompt:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []

    def system(self, content: str) -> Self:
        self.messages.append({"role": "system", "content": content})
        return self

    def user(self, content: str) -> Self:
        self.messages.append({"role": "user", "content": content})
        return self

    def assistant(self, content: str, tool_calls: Optional[list] = None) -> Self:
        if tool_calls is not None:
            self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        else:
            self.messages.append({"role": "assistant", "content": content})
        return self

    def tool(self, content: str, tool_call_id: str) -> Self:
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})
        return self

    def to_messages(self) -> List[Dict[str, Any]]:
        return self.messages

    def to_text(self) -> str:
        """Return all message contents joined as a single string."""
        return "\n".join(msg["content"] for msg in self.messages if msg.get("content"))

    def format(self, **kwargs) -> "Prompt":
        """Return a new Prompt with all message contents formatted using kwargs."""
        formatted = Prompt()
        for msg in self.messages:
            formatted.messages.append({
                key: value.format(**kwargs) if isinstance(value, str) else value
                for key, value in msg.items()
            })
        return formatted

    def has_placeholder(self, *names: str) -> bool:
        """Return True if any message content contains at least one of the given {placeholder} names."""
        targets = [f'{{{name}}}' for name in names]
        return any(
            target in (msg.get('content') or '')
            for msg in self.messages
            for target in targets
        )
    
    def write_to_file(self, path: str) -> None:
        """Write the prompt messages as pretty-printed JSON to a file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, indent=2, ensure_ascii=False)

    def __str__(self) -> str:
        return "\n".join(f"{msg['role']}: {msg.get('content', '')}" for msg in self.messages)


#### EMBEDDING PROMPTS ####

EMBEDDING_PROMPTS = {
    "one": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{text}",
    "two": "Instruct: Given a entity description, retrieve relevant other entities which are most similar\nQuery:{text}",
    "three": "Instruct: Given a entity description in turtle, retrieve relevant other entities which are most similar\nQuery:{text}",
    "four": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {uri}\n{text}",
    "five": "Instruct: Given a web search query for this URI {uri}, retrieve relevant passages that answer the query\nQuery:{text}",
}


#### RERANKING PROMPTS ####

RERANKING_PROMPTS = {
    "a": (
        'Given the following two knowledge graphs, decide whether <{source_url}> and <{target_url}> describe the same entity.'
        ' Answer with a JSON object with a single key "match" and a boolean value true or false. Only output the JSON object.'
        '\n\nSource:\n{source_kg}\n\nTarget:\n{target_kg}\nAnswer:'
    ),
    "b": (
        'Given the following two knowledge graphs, decide whether two entities describe the same real world entity.'
        ' Answer with a JSON object with a single key "match" and a boolean value true or false. Only output the JSON object.'
        '\n\nSource knowledge graph:\n{source_kg}\n\nTarget knowledge graph:\n{target_kg}'
        '\nSource URL: {source_url}\nTarget URL: {target_url}\nAnswer:'
    ),
    "c": (
        "Classify if the two concepts <{source_url}> and <{target_url}> are the same."
        "\n\nSource knowledge graph:\n{source_kg}\n\nTarget knowledge graph:\n{target_kg}"
        "\nAnswer with 'yes' or 'no':"
    ),
    "d": (
        "You are an expert in ontology matching. Given the following two knowledge graphs,"
        " determine if the entities <{source_url}> and <{target_url}> refer to the same real world entity."
        "\n\nSource knowledge graph:\n{source_kg}\n\nTarget knowledge graph:\n{target_kg}"
        "\nAnswer with 'yes' or 'no':"
    ),
    "e": (
        'You are an expert in ontology matching. Given the following two knowledge graphs, decide whether <{source_url}>'
        ' and <{target_url}> describe the same entity.'
        ' Answer with a JSON object with a single key "match" and a boolean value true or false. Only output the JSON object.'
        '\n\nSource knowledge graph:\n{source_kg}\n\nTarget knowledge graph:\n{target_kg}\nAnswer:'
    ),
    # Stage-2 multi-class relation classifier. Placeholders identical to "d".
    # The prompt forces a "Relation: <label>" line as the LAST line so reasoners
    # (gpt-oss, Gemma-4-thinking) can emit chain-of-thought beforehand and the
    # parser still anchors on the final line. Labels match RELATION_LABEL_SYNONYMS.
    #
    # DEPRECATED (kept for reproducibility of job 255391, 2026-06-02).
    # The "Think step by step if needed" clause was too permissive for
    # non-reasoners: Llama-3.3-70B-AWQ took it as a CoT cue and produced
    # ~250-token reasoning that hit max_new_tokens=256 BEFORE emitting the
    # "Relation: <label>" anchor. 1039/1040 responses lacked the anchor and
    # fell into the (now-removed) freeform synonym-scan fallback in the
    # parser, producing a structural >-bias driven by reasoning-order and
    # truncation, not by the model itself. Use d_subs_v2 instead.
    "d_subs": (
        "You are an expert in ontology matching. Determine the precise"
        " semantic relation between two entities from different ontologies."
        "\n\nSource entity: <{source_url}>"
        "\nSource knowledge graph:\n{source_kg}"
        "\n\nTarget entity: <{target_url}>"
        "\nTarget knowledge graph:\n{target_kg}"
        "\n\nChoose exactly one label that describes how the source relates to the target:"
        "\n- subclass:   source is a more specific kind of target (source ⊑ target)"
        "\n- superclass: source is a more general kind of target (source ⊒ target)"
        "\n- equivalent: source and target denote the same concept"
        "\n- partof:     source is a part of target (mereological, not taxonomic)"
        "\n- none:       none of the above applies"
        "\n\nThink step by step if needed, then end your response with EXACTLY this line:"
        "\nRelation: <label>"
        "\n\nwhere <label> is one of: subclass, superclass, equivalent, partof, none."
    ),
    # ANSWER-FIRST + SYMMETRISED + BALANCED FEW-SHOT variants (v3a / v3b).
    #
    # v2 was answer-first but otherwise word-for-word the unsymmetric default.
    # Job 255471 (g7-literature, Llama-3.3-70B-AWQ, 2026-06-02) showed a strong
    # subclass-default: 26 of 52 gold-`>` flipped to `<` (Gold-`>`->pred-`<`
    # flip-rate 26/45 = 57.8% of directional predictions). v2's prompt is
    # structurally symmetric in syntax but carries three un-neutralised
    # asymmetries that could each contribute to the bias:
    #   1. Listen-position: subclass is mentioned FIRST in the label list and
    #      again FIRST in the closing "one of: subclass, superclass, ..." line.
    #      Primacy-bias on multi-choice labels is documented at ~5-10 pp.
    #   2. Lexical frequency: 'subclass' >> 'superclass' in English pre-training
    #      corpora; the prompt doesn't fight this.
    #   3. Symbol frequency: '⊑' >> '⊒' in formal-logic / type-theory texts.
    #   Plus: no demonstrations.
    #
    # v3a / v3b BOTH symmetrise the wording and add a 3-shot balanced demo:
    # the SAME concept pair (Novel/Book) is shown in BOTH directions so the
    # model treats the directional choice as a primary axis to reason on,
    # not as a default. v3a keeps subclass FIRST in the list and the first
    # demo example; v3b puts superclass first in both. Running both controls
    # for position-bias vs. true model prior:
    #   - flip-rate unchanged in v3a AND v3b -> true model prior (Llama default
    #     mode falls back to subclass regardless of prompt steering -> the
    #     four-model comparison is the next lever; reasoners are the next
    #     candidate hypothesis).
    #   - flip-rate drops in v3a AND v3b -> prompt-engineering is sufficient
    #     (reasoning not required for direction resolution).
    #   - flip-rate drops only in v3b -> listen-position dominated.
    #
    # CAVEAT: v3 changes TWO things vs. v2 (symmetric wording AND few-shot).
    # If both v3a/v3b drop the flip-rate, the experiment cannot separate
    # which of the two interventions did it. For the strategic question
    # ('prompt-engineering correctable?') this is fine — it gives us the
    # answer we need. For a clean attribution decomposition a fourth run
    # (symmetric without few-shot) would be needed; deferred unless the
    # decomposition matters downstream.
    #
    # The few-shot examples use literary concepts (Novel/Book/Author/Writer),
    # which fits g7-literature naturally. For multi-dataset use later the
    # examples must be either domain-generic or per-dataset; this would
    # otherwise introduce a new confound across datasets.
    "d_subs_v2": (
        "You are an expert in ontology matching. Determine the precise"
        " semantic relation between two entities from different ontologies."
        "\n\nSource entity: <{source_url}>"
        "\nSource knowledge graph:\n{source_kg}"
        "\n\nTarget entity: <{target_url}>"
        "\nTarget knowledge graph:\n{target_kg}"
        "\n\nValid labels:"
        "\n  subclass    source is a more specific kind of target (source ⊑ target)"
        "\n  superclass  source is a more general kind of target (source ⊒ target)"
        "\n  equivalent  source and target denote the same concept"
        "\n  partof      source is a part of target (mereological, not taxonomic)"
        "\n  none        none of the above applies"
        "\n\nYour response MUST start with EXACTLY this line and nothing else"
        " on it:"
        "\nRelation: <label>"
        "\n\nReplace <label> with one of: subclass, superclass, equivalent,"
        " partof, none. A short justification MAY follow on the next lines,"
        " but the very first line of your response must be the answer."
    ),
    # v3a — subclass-first, symmetric wording, balanced 3-shot.
    # The Novel/Book pair appears in both directions to anti-bias the model;
    # Author/Writer provides an equivalent example. Listen-positions in the
    # label list and the closing "one of:" line both keep subclass first.
    "d_subs_v3a": (
        "You are an expert in ontology matching. Determine the precise"
        " semantic relation between two entities from different ontologies."
        "\n\nValid labels (apply with equal weight):"
        "\n  subclass    source is a more specific kind of target  (source ⊑ target)"
        "\n  superclass  source is a more general kind of target   (source ⊒ target)"
        "\n  equivalent  source and target denote the same concept"
        "\n  partof      source is a part of target (mereological, not taxonomic)"
        "\n  none        none of the above applies"
        "\n\nExamples:"
        "\n--- Example 1 ---"
        "\nSource entity: <example:Novel>"
        "\nSource knowledge graph:"
        "\n<example:Novel> a owl:Class ; rdfs:label \"Novel\" ."
        "\nTarget entity: <example:Book>"
        "\nTarget knowledge graph:"
        "\n<example:Book> a owl:Class ; rdfs:label \"Book\" ."
        "\nRelation: subclass"
        "\n--- Example 2 ---"
        "\nSource entity: <example:Book>"
        "\nSource knowledge graph:"
        "\n<example:Book> a owl:Class ; rdfs:label \"Book\" ."
        "\nTarget entity: <example:Novel>"
        "\nTarget knowledge graph:"
        "\n<example:Novel> a owl:Class ; rdfs:label \"Novel\" ."
        "\nRelation: superclass"
        "\n--- Example 3 ---"
        "\nSource entity: <example:Author>"
        "\nSource knowledge graph:"
        "\n<example:Author> a owl:Class ; rdfs:label \"Author\" ."
        "\nTarget entity: <example:Writer>"
        "\nTarget knowledge graph:"
        "\n<example:Writer> a owl:Class ; rdfs:label \"Writer\" ."
        "\nRelation: equivalent"
        "\n\n--- Now the actual task ---"
        "\nSource entity: <{source_url}>"
        "\nSource knowledge graph:\n{source_kg}"
        "\nTarget entity: <{target_url}>"
        "\nTarget knowledge graph:\n{target_kg}"
        "\n\nYour response MUST start with EXACTLY this line and nothing else"
        " on it:"
        "\nRelation: <label>"
        "\n\nReplace <label> with one of: subclass, superclass, equivalent,"
        " partof, none. A short justification MAY follow on the next lines,"
        " but the very first line of your response must be the answer."
    ),
    # v3b — superclass-first, otherwise identical to v3a. Identical wording,
    # identical few-shot pairs, only the surface order in the label list AND
    # in the first demonstration AND in the closing "one of:" line is swapped.
    "d_subs_v3b": (
        "You are an expert in ontology matching. Determine the precise"
        " semantic relation between two entities from different ontologies."
        "\n\nValid labels (apply with equal weight):"
        "\n  superclass  source is a more general kind of target   (source ⊒ target)"
        "\n  subclass    source is a more specific kind of target  (source ⊑ target)"
        "\n  equivalent  source and target denote the same concept"
        "\n  partof      source is a part of target (mereological, not taxonomic)"
        "\n  none        none of the above applies"
        "\n\nExamples:"
        "\n--- Example 1 ---"
        "\nSource entity: <example:Book>"
        "\nSource knowledge graph:"
        "\n<example:Book> a owl:Class ; rdfs:label \"Book\" ."
        "\nTarget entity: <example:Novel>"
        "\nTarget knowledge graph:"
        "\n<example:Novel> a owl:Class ; rdfs:label \"Novel\" ."
        "\nRelation: superclass"
        "\n--- Example 2 ---"
        "\nSource entity: <example:Novel>"
        "\nSource knowledge graph:"
        "\n<example:Novel> a owl:Class ; rdfs:label \"Novel\" ."
        "\nTarget entity: <example:Book>"
        "\nTarget knowledge graph:"
        "\n<example:Book> a owl:Class ; rdfs:label \"Book\" ."
        "\nRelation: subclass"
        "\n--- Example 3 ---"
        "\nSource entity: <example:Author>"
        "\nSource knowledge graph:"
        "\n<example:Author> a owl:Class ; rdfs:label \"Author\" ."
        "\nTarget entity: <example:Writer>"
        "\nTarget knowledge graph:"
        "\n<example:Writer> a owl:Class ; rdfs:label \"Writer\" ."
        "\nRelation: equivalent"
        "\n\n--- Now the actual task ---"
        "\nSource entity: <{source_url}>"
        "\nSource knowledge graph:\n{source_kg}"
        "\nTarget entity: <{target_url}>"
        "\nTarget knowledge graph:\n{target_kg}"
        "\n\nYour response MUST start with EXACTLY this line and nothing else"
        " on it:"
        "\nRelation: <label>"
        "\n\nReplace <label> with one of: superclass, subclass, equivalent,"
        " partof, none. A short justification MAY follow on the next lines,"
        " but the very first line of your response must be the answer."
    ),
    # v4b — label-ORDER flip with padding held constant (Stufe-A arm A1;
    # registered 2026-06-12, docs/stage2_stufeA_registration.md).
    # Byte-identical to d_subs_v2 EXCEPT:
    #   1. the two directional definition lines swap order (superclass first),
    #   2. the closing enumeration starts with superclass,
    #   3. padding control: BOTH directional labels carry the SAME
    #      trailing-space count (two). v2 column-aligns with 4-vs-2 spaces —
    #      a latent tokenizer-level asymmetry (leading whitespace folds into
    #      tokens) that the order flip would otherwise carry along and
    #      confound the attribution.
    # Zero-shot like v2. There is NO d_subs_v4a: "v2 wording, subclass-first,
    # zero-shot" is definitionally d_subs_v2 itself (verified Phase-0 audit) —
    # Run 255471 / R0 are the subclass-first arm.
    "d_subs_v4b": (
        "You are an expert in ontology matching. Determine the precise"
        " semantic relation between two entities from different ontologies."
        "\n\nSource entity: <{source_url}>"
        "\nSource knowledge graph:\n{source_kg}"
        "\n\nTarget entity: <{target_url}>"
        "\nTarget knowledge graph:\n{target_kg}"
        "\n\nValid labels:"
        "\n  superclass  source is a more general kind of target (source ⊒ target)"
        "\n  subclass  source is a more specific kind of target (source ⊑ target)"
        "\n  equivalent  source and target denote the same concept"
        "\n  partof      source is a part of target (mereological, not taxonomic)"
        "\n  none        none of the above applies"
        "\n\nYour response MUST start with EXACTLY this line and nothing else"
        " on it:"
        "\nRelation: <label>"
        "\n\nReplace <label> with one of: superclass, subclass, equivalent,"
        " partof, none. A short justification MAY follow on the next lines,"
        " but the very first line of your response must be the answer."
    ),
    # d_subs_v2_fs — E15 few-shot. BYTE-IDENTICAL to d_subs_v2 except one
    # {exemplars} slot after the intro sentence. The block is built at runtime
    # from held-out g1-web gold (fewshot_exemplars.build_fewshot_block) and passed
    # as a .format() VALUE, so its contents are never re-scanned for placeholders.
    # With exemplars="" it degrades to d_subs_v2, so A0 stays on plain d_subs_v2
    # and remains byte-identical to the matrix zero-shot cell.
    "d_subs_v2_fs": (
        "You are an expert in ontology matching. Determine the precise"
        " semantic relation between two entities from different ontologies."
        "\n{exemplars}"
        "\n\nSource entity: <{source_url}>"
        "\nSource knowledge graph:\n{source_kg}"
        "\n\nTarget entity: <{target_url}>"
        "\nTarget knowledge graph:\n{target_kg}"
        "\n\nValid labels:"
        "\n  subclass    source is a more specific kind of target (source ⊑ target)"
        "\n  superclass  source is a more general kind of target (source ⊒ target)"
        "\n  equivalent  source and target denote the same concept"
        "\n  partof      source is a part of target (mereological, not taxonomic)"
        "\n  none        none of the above applies"
        "\n\nYour response MUST start with EXACTLY this line and nothing else"
        " on it:"
        "\nRelation: <label>"
        "\n\nReplace <label> with one of: subclass, superclass, equivalent,"
        " partof, none. A short justification MAY follow on the next lines,"
        " but the very first line of your response must be the answer."
    ),
}


#### MULTI-CLASS RELATION LABELS (Stage-2 reranker) ####

# Canonical multi-class labels emitted by parse_relation_label. The reranker
# maps these to ASCII relation chars via RELATION_LABEL_TO_RELATION.
#
# "parse_fail" is reserved for: the response had no "Relation: <label>" anchor
# at all. It is distinct from an explicit "Relation: none" reply (which means
# "the model chose none"). The reranker drops both, but the runner reports
# the parse_fail rate separately so a high rate immediately surfaces a
# format-compliance regression instead of being silently lumped into 'none'
# via a synonym-scan heuristic.
RELATION_LABEL_CANONICAL: tuple[str, ...] = (
    "subclass", "superclass", "equivalent", "partof", "none", "parse_fail",
)

# Canonical label -> tuple of accepted synonyms (all lowercase, no punctuation).
# Synonyms are matched after lowercasing + stripping a small set of trailing
# punctuation; whitespace inside multi-word synonyms is collapsed.
RELATION_LABEL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "subclass":   ("subclass", "sub-class", "sub class", "subclassof",
                   "narrower", "more specific", "is-a", "isa",
                   "⊑", "≤"),
    "superclass": ("superclass", "super-class", "super class", "superclassof",
                   "broader", "more general", "subsumes",
                   "⊒", "≥"),
    "equivalent": ("equivalent", "equivalence", "equal", "same",
                   "sameas", "same-as", "same as", "=", "≡"),
    "partof":     ("partof", "part of", "part-of", "part", "meronym",
                   "part_of"),
    "none":       ("none", "no relation", "no-relation", "unrelated",
                   "n/a", "na", "other", "no", "irrelevant"),
}

# Stage-2 canonical label -> output Correspondence.relation string.
# 'partof' stays as a literal — evaluation_multiclass folds it into 'none'
# in the displayed 4x4 confusion matrix (per the data-sparsity rationale
# in evaluation_recall.py:62-68). 'none' and 'parse_fail' yield the empty
# string; the reranker uses that as the "drop" sentinel and records the
# distinction in last_run_details so the runner can surface both rates.
RELATION_LABEL_TO_RELATION: dict[str, str] = {
    "subclass":   "<",
    "superclass": ">",
    "equivalent": "=",
    "partof":     "partof",
    "none":       "",
    "parse_fail": "",
}


import re as _re

# Pre-compiled patterns. The "Relation: <label>" anchor is the primary path;
# the freeform fallback scans the whole text for any synonym occurrence.
_RELATION_LINE_RE = _re.compile(r"relation\s*[:\-]\s*([^\n\r]+)", _re.IGNORECASE)
# Strip wrapping punctuation/brackets/quotes from the captured label.
_LABEL_STRIP_RE = _re.compile(r"^[\s\*\.\,\;\:\'\"\`\(\)\[\]\{\}<>]+|[\s\*\.\,\;\:\'\"\`\(\)\[\]\{\}<>]+$")


def _canonical_from_token(token: str) -> str | None:
    """Match a normalised token against the synonym table. Returns canonical
    label or None. Token is expected to be lowercased and punctuation-stripped.
    """
    token = " ".join(token.split())  # collapse whitespace
    if not token:
        return None
    for canonical, synonyms in RELATION_LABEL_SYNONYMS.items():
        for syn in synonyms:
            if token == syn:
                return canonical
    # Substring fallback: handle e.g. "subclass." after strip, or "the
    # relation is subclass" — only used by the freeform scan below.
    return None


def parse_relation_label(text: str) -> str:
    """Extract one of RELATION_LABEL_CANONICAL from a model completion.

    Strict parser — anchored matches ONLY. Verified empirically on the
    2026-06-02 g7-literature smoke (job 255391) that a synonym-scan
    fallback over the full response produces dangerous false positives:
    when responses are truncated mid-reasoning (1039 / 1040 calls hit
    max_new_tokens=256 before emitting the anchor line), the last
    synonym in the response is purely a function of reasoning-order
    and truncation-point, not of the model's actual decision. That
    surfaced as a phantom >-bias in the smoke. We removed the fallback.

    Strategy (in order):
      1. Find the LAST "Relation: <label>" / "Relation - <label>" match
         (case-insensitive), strip wrapping punctuation, match against
         RELATION_LABEL_SYNONYMS. If the captured token (or its first
         whitespace-delimited sub-token) maps to a canonical label, return it.
      2. No match -> return "parse_fail". This is DISTINCT from the model
         emitting "Relation: none" — the latter returns "none". The
         reranker drops both, but reports the rates separately in
         last_run_details so a format-compliance regression is visible
         in metrics.json rather than silently lumped into "none".
    """
    if not text:
        return "parse_fail"

    # 1. Anchored "Relation: <label>" matches — take the last one.
    matches = list(_RELATION_LINE_RE.finditer(text))
    if matches:
        captured = matches[-1].group(1).strip().lower()
        captured = _LABEL_STRIP_RE.sub("", captured).strip()
        # Try direct equality first.
        canonical = _canonical_from_token(captured)
        if canonical is not None:
            return canonical
        # Then try the first whitespace-delimited token (label might be
        # followed by an explanation on the same line).
        first_token = captured.split()[0] if captured else ""
        first_token = _LABEL_STRIP_RE.sub("", first_token).strip()
        canonical = _canonical_from_token(first_token)
        if canonical is not None:
            return canonical
        # Anchor present but captured text didn't resolve to a known label
        # (e.g. "Relation: maybe?"): treat as parse_fail rather than guessing.
        return "parse_fail"

    # 2. No anchor at all -> parse_fail. No synonym-scan fallback by design.
    return "parse_fail"


#### SYSTEM PROMPTS ####

SYSTEM_PROMPTS = {
    "sone": "You are a helpful assistant that helps to decide whether two knowledge graph fragments describe the same entity.",
    "stwo": "You are a perfect ontology matching system that can decide if two entities belong to the same real world entity based on their descriptions.",
}


def _build_prompt(prompt_id: str, user_prompts: dict[str, str]) -> Prompt:
    """Build an unformatted Prompt from a prompt_id.

    Resolution order:
      1. Direct lookup: if the full ``prompt_id`` (lowercased) is a key in
         ``user_prompts``, use it as the user prompt with no system prompt.
         This lets registry keys legitimately contain underscores (e.g.
         the Stage-2 multi-class key ``d_subs``) without colliding with the
         composite-id convention below.
      2. Composite ``system_user`` form: everything before the first
         underscore is the system prompt (looked up in ``SYSTEM_PROMPTS``
         if it matches a key, used as-is otherwise). Everything after is
         the user text (looked up in ``user_prompts`` if it matches a key,
         used as-is otherwise).
      3. No-underscore form: the id is the user prompt key (or text), no
         system prompt.
    """
    if prompt_id is None or prompt_id == "":
        raise ValueError("Prompt id cannot be None or empty")

    # 1. Direct registry hit wins — keeps underscore-containing keys usable.
    direct = user_prompts.get(prompt_id.lower())
    if direct is not None:
        prompt = Prompt()
        prompt.user(direct)
        return prompt

    system_text = None
    user_key = prompt_id
    if '_' in prompt_id:
        prefix, user_key = prompt_id.split('_', 1)
        system_text = SYSTEM_PROMPTS.get(prefix.lower(), prefix)

    user_text = user_prompts.get(user_key.lower())
    if user_text is None:
        logger.info("using prompt id as the user prompt directly")
        user_text = user_key

    prompt = Prompt()
    if system_text:
        prompt.system(system_text)
    prompt.user(user_text)
    return prompt


#### SPARQL AGENT PROMPTS ####

SPARQL_AGENT_PROMPTS = {
    "sa": (
        "You are an expert in ontology matching. Determine if"
        " <{source_url}> from the source ontology and <{target_url}> from the"
        " target ontology refer to the same real-world concept.\n\n"
        "You have access to SPARQL query tools for both ontologies."
        " Explore labels, descriptions, class hierarchies, and properties"
        " before making your decision."
    ),
    "sb": (
        "You are an expert in ontology matching. Given a source entity"
        " <{source_url}>, determine which of the following candidate target"
        " entities it matches: {candidate_urls}\n\n"
        "Use the SPARQL query tools to explore both ontologies."
    ),
}


def get_embedding_prompt(prompt_id: str) -> Prompt:
    return _build_prompt(prompt_id, EMBEDDING_PROMPTS)


def get_reranking_prompt(prompt_id: str) -> Prompt:
    return _build_prompt(prompt_id, RERANKING_PROMPTS)


def get_sparql_agent_prompt(prompt_id: str) -> Prompt:
    return _build_prompt(prompt_id, SPARQL_AGENT_PROMPTS)


#### SUBSUMPTION INSTRUCTIONS ####

# Versioned instruction texts for the BeyondEquivalence retrieval study.
# Iteration policy: never edit an existing _vN entry — add a new _vN+1 next to it.
# The instruction text lands in the W&B run config, so older runs stay reconstructible.
SUBSUMPTION_INSTRUCTIONS: dict[str, str] = {
    # Symmetric: same instruction for query and document side
    "sym_v1": "Given a concept description, retrieve concept descriptions that are semantically related",

    # Asymmetric: query side encodes the direction; document side stays symmetric (empty)
    "asym_broader_v1":  "Given a concept description, retrieve more general / broader concept descriptions",
    "asym_narrower_v1": "Given a concept description, retrieve more specific / narrower concept descriptions",

    # Explicit "no instruction" marker — handy for the document side in asymmetric runs
    "none": "",

    # ── Sub-B description-ablation templates (5 sym × 5 asym pair). The _v1
    # entries above stay frozen. Tag in the comment names the specificity
    # axis along which the template differs from v1.
    #
    # S1 — minimal generic, baseline
    "sym_S1": "Given a class description, retrieve semantically equivalent classes",
    # S2 — ontology-context explicit
    "sym_S2": "Given an ontology class description, retrieve equivalent or closely related classes from another ontology",
    # S3 — equivalence vocabulary (synonym / same-as)
    "sym_S3": "Given a class, retrieve classes that refer to the same concept — i.e. its synonyms, equivalents, or same-as classes",
    # S4 — taxonomy-explicit, long
    "sym_S4": "Given a category from a taxonomy, retrieve other categories that represent the same concept, including equivalent terms, synonyms, or alternative labels for the same entity",
    # S5 — example-driven (few-shot)
    "sym_S5": 'Given a class, retrieve classes that refer to the same concept — for example, given "car" retrieve "automobile" or "motor vehicle"',

    # T1 — minimal asymmetric, baseline
    "asym_broader_T1":  "Given a class, retrieve its broader parent classes",
    "asym_narrower_T1": "Given a class, retrieve its narrower child classes",
    # T2 — ontology-context explicit
    "asym_broader_T2":  "Given an ontology class description, retrieve broader (more general) parent classes from another ontology",
    "asym_narrower_T2": "Given an ontology class description, retrieve narrower (more specific) child classes from another ontology",
    # T3 — lexical hierarchy vocabulary (is-a / hypernym, hyponym)
    "asym_broader_T3":  "Given a class, retrieve classes that this class is-a — i.e. its superclasses or hypernyms",
    "asym_narrower_T3": "Given a class, retrieve classes that are-a this class — i.e. its subclasses or hyponyms",
    # T4 — taxonomy-explicit, long
    "asym_broader_T4":  "Given a category from a hierarchical taxonomy, retrieve broader categories that subsume it (the parent or ancestor concepts)",
    "asym_narrower_T4": "Given a category from a hierarchical taxonomy, retrieve narrower categories that it subsumes (the child or descendant concepts)",
    # T5 — example-driven (few-shot)
    "asym_broader_T5":  'Given a class, retrieve its broader parent classes — for example, given "dog" retrieve "animal" or "mammal"',
    "asym_narrower_T5": 'Given a class, retrieve its narrower child classes — for example, given "animal" retrieve "dog" or "cat"',
}


# Sub-B template-id sets used by the sweep runner. Single source of truth so
# the SLURM scripts and the runner can't drift apart.
SUBB_SYM_TEMPLATE_IDS:           tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5")
SUBB_ASYM_TEMPLATE_IDS:          tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5")
SUBB_DESCRIPTION_METHODS:        tuple[str, ...] = (
    "description_text",
    "description_basic",
    "description_one_gen",
    "description_two_gen",
    "description_three_gen",
)


def get_subb_sym_template(template_id: str) -> str:
    """Resolve a Sub-B sym template id (e.g. 'S2') to its instruction string."""
    return get_subsumption_instruction(f"sym_{template_id}")


def get_subb_asym_templates(template_id: str) -> tuple[str, str]:
    """Resolve a Sub-B asym template id (e.g. 'T2') to (broader, narrower)."""
    return (
        get_subsumption_instruction(f"asym_broader_{template_id}"),
        get_subsumption_instruction(f"asym_narrower_{template_id}"),
    )


def get_subsumption_instruction(prompt_id: str | None) -> str:
    """Resolve a SUBSUMPTION_INSTRUCTIONS id to its instruction text.

    Empty/None input or 'none' return the empty string. Unknown ids raise KeyError
    so typos surface immediately at run-start.
    """
    if prompt_id is None or prompt_id == "":
        return ""
    if prompt_id not in SUBSUMPTION_INSTRUCTIONS:
        raise KeyError(
            f"Unknown subsumption instruction id '{prompt_id}'. "
            f"Available: {sorted(SUBSUMPTION_INSTRUCTIONS.keys())}"
        )
    return SUBSUMPTION_INSTRUCTIONS[prompt_id]


#### EMBEDDING-PROMPT WRAPPING ####

# All currently used instruction-aware embedding models (Qwen3-Embedding,
# llama-embed-nemotron-8b) share the "Instruct: {instruction}\nQuery: " prefix
# convention per their HuggingFace model cards. This is passed verbatim to
# SentenceTransformer.encode(..., prompt=...) so the library handles pooling
# and normalisation natively (left-padding + last-token pool on Nemotron).
#
# Empty instruction => empty prompt prefix; the matcher passes prompt=None to
# encode in that case (document side in asymmetric runs, sbert).

INSTRUCT_QUERY_PREFIX = "Instruct: {instruction}\nQuery: "


def build_instruct_query_prompt(instruction: str) -> str:
    """Build the Instruct/Query prefix for SentenceTransformer.encode(prompt=...).

    Returns "" for an empty instruction so the caller can collapse to prompt=None.
    """
    if not instruction:
        return ""
    return INSTRUCT_QUERY_PREFIX.format(instruction=instruction)


# SentenceTransformer constructor kwargs needed for instruction-aware embedding
# models in this study. trust_remote_code=True is set unconditionally by the
# matcher; the kwargs below are the per-model pins documented on the HF model
# cards and required for correct pooling/dtype:
#
# nvidia/llama-embed-nemotron-8b (model card):
#   - tokenizer padding_side="left" — last-token pool over the latent-attention
#     pooler assumes left-padding; right-padding silently produces the wrong
#     pooled vector.
#   - attn_implementation="eager" — the bidirectional-attention custom code is
#     only verified against the eager kernel; FA2 path is silently broken.
#   - torch_dtype="bfloat16" — matches released weight dtype; avoids fp32
#     upcast on load.
#
# Qwen/Qwen3-Embedding-8B (model card):
#   - tokenizer padding_side="left" — same last-token pooling assumption.
#
# sentence-transformers/all-MiniLM-L6-v2 (sbert): mean-pool, right-padding —
# defaults are correct, no kwargs needed.
_MODEL_LOADER_KWARGS: list[tuple[str, dict[str, dict]]] = [
    ("llama-embed-nemotron", {
        "model_kwargs":     {"attn_implementation": "eager", "torch_dtype": "bfloat16"},
        "tokenizer_kwargs": {"padding_side": "left"},
    }),
    ("llama-nemotron-embed", {
        "model_kwargs":     {"attn_implementation": "eager", "torch_dtype": "bfloat16"},
        "tokenizer_kwargs": {"padding_side": "left"},
    }),
    ("qwen3-embedding", {
        "tokenizer_kwargs": {"padding_side": "left"},
    }),
    ("qwen3-emb", {
        "tokenizer_kwargs": {"padding_side": "left"},
    }),
]


def get_loader_kwargs(model_id_or_path: str) -> dict[str, dict]:
    """SentenceTransformer constructor kwargs for the given model id / path.

    Substring-matched against `_MODEL_LOADER_KWARGS` (lower-cased); first match
    wins. Returns {} for unmatched models (sbert and unknowns rely on ST
    defaults). The caller spreads the result with **kwargs into
    SentenceTransformer(...).
    """
    haystack = (model_id_or_path or "").lower()
    for needle, kw in _MODEL_LOADER_KWARGS:
        if needle in haystack:
            return kw
    return {}