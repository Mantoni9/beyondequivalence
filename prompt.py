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
    # ANSWER-FIRST variant. The anchor is the first line of the response, any
    # justification follows. This decouples the answer from any reasoning the
    # model wants to produce: even if the model continues with 1000 tokens of
    # justification afterwards, the parser already has the label and the rest
    # is cheap (or could be cut by lowering max_new_tokens). Fair across
    # reasoner + non-reasoner models — the structural difference between
    # "model thought long" and "model thought short" doesn't break the parse.
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