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