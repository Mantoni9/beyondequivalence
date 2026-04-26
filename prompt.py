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
}


#### SYSTEM PROMPTS ####

SYSTEM_PROMPTS = {
    "sone": "You are a helpful assistant that helps to decide whether two knowledge graph fragments describe the same entity.",
    "stwo": "You are a perfect ontology matching system that can decide if two entities belong to the same real world entity based on their descriptions.",
}


def _build_prompt(prompt_id: str, user_prompts: dict[str, str]) -> Prompt:
    """Build an unformatted Prompt from a prompt_id.

    Everything before the first underscore is the system prompt
    (looked up in SYSTEM_PROMPTS if it matches a key, used as-is otherwise).
    Everything after is the user text (looked up in user_prompts if it
    matches a key, used as-is otherwise).
    If there is no underscore, there is no system prompt.
    """
    if prompt_id is None or prompt_id == "":
        raise ValueError("Prompt id cannot be None or empty")

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
}


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


#### MODEL-FAMILY-AWARE INSTRUCTION FORMATTING ####

# All currently used instruction-aware embedding models share the
# "Instruct: {instruction}\nQuery: {text}" wrapping convention (per Qwen3-Embedding,
# NV-Embed-v2, and e5-mistral-7b-instruct model cards). The per-family indirection
# exists so a future model with a different convention can be slotted in without
# changing call sites.

def _wrap_instruct_query(instruction: str, text: str) -> str:
    return f"Instruct: {instruction}\nQuery: {text}"


def _wrap_naive_concat(instruction: str, text: str) -> str:
    return f"{instruction}\n{text}"


_FAMILY_FORMATTERS = {
    "qwen3-embedding": _wrap_instruct_query,
    "nv-embed":        _wrap_instruct_query,
    "e5-mistral":      _wrap_instruct_query,
    "sbert":           _wrap_naive_concat,
    "auto":            _wrap_naive_concat,
}

# Substring-based family inference. Lower-cased haystack; first matching needle wins.
# Order matters when substrings overlap (e.g. "qwen3-embedding" before "qwen3").
_FAMILY_INFERENCE_RULES: list[tuple[str, str]] = [
    ("qwen3-embedding",        "qwen3-embedding"),
    ("qwen3-emb",              "qwen3-embedding"),
    ("nv-embed",               "nv-embed"),
    ("e5-mistral",             "e5-mistral"),
    ("all-minilm",             "sbert"),
    ("minilm",                 "sbert"),
    ("sbert-mini",             "sbert"),
    ("sentence-transformers/", "sbert"),
]


def infer_model_family(model_id_or_path: str) -> str:
    """Infer the embedding-model family from an HF id or local path.

    Returns one of the keys in _FAMILY_FORMATTERS. Falls back to 'auto' with a
    WARNING — never info/debug — so an unrecognised model is visible at run-start
    instead of silently producing numbers from naive instruction-concat formatting.
    """
    haystack = (model_id_or_path or "").lower()
    for needle, family in _FAMILY_INFERENCE_RULES:
        if needle in haystack:
            return family
    logger.warning(
        "Could not infer embedding-model family for '%s' — falling back to 'auto' "
        "(naive instruction-concat formatting). Pass model_family= explicitly "
        "('qwen3-embedding' / 'nv-embed' / 'e5-mistral' / 'sbert') if this is wrong.",
        model_id_or_path,
    )
    return "auto"


def format_instruction(model_family: str, instruction: str, text: str) -> str:
    """Wrap `text` with `instruction` according to the model family's convention.

    Empty instruction is a pass-through (returns text unchanged) for every family.
    Unknown family raises KeyError; the explicit fallback family is 'auto'.
    """
    if not instruction:
        return text
    formatter = _FAMILY_FORMATTERS.get(model_family)
    if formatter is None:
        raise KeyError(
            f"Unknown model_family '{model_family}'. "
            f"Available: {sorted(_FAMILY_FORMATTERS.keys())}"
        )
    return formatter(instruction, text)