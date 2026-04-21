import logging
from abc import ABC, abstractmethod
from typing import List, Sequence, Set

import regex

from llm_tool import Tool
from prompt import Prompt

logger = logging.getLogger(__name__)


MODEL_REVISIONS = {
    # Meta Llama 3 families
    'meta-llama/Meta-Llama-3-8B': '8cde5ca',
    'meta-llama/Meta-Llama-3-8B-Instruct': '8afb486',

    'meta-llama/Meta-Llama-3-70B': 'c824948',
    'meta-llama/Meta-Llama-3-70B-Instruct': '50fd307',

    # Meta Llama 3.1 families
    'meta-llama/Llama-3.1-8B': 'd04e592',
    'meta-llama/Llama-3.1-8B-Instruct': '0e9e39f',

    'meta-llama/Llama-3.1-70B': '349b2dd',
    'meta-llama/Llama-3.1-70B-Instruct': '1605565',

    # Meta Llama 3.3 families
    'meta-llama/Llama-3.3-70B-Instruct': '6f6073b',
}


def get_model_revision(model_name, default_value="main"):
    revision = MODEL_REVISIONS.get(model_name)
    if revision is None:
        logging.warning(f"Model revision for model {model_name} not found. Using default value {default_value}")
        return default_value
    return revision


class LLMBase(ABC):
    """Abstract base class defining the interface for LLM wrappers."""

    # ------------------------------------------------------------------ #
    #  Tokenizer & token-set helpers                                      #
    # ------------------------------------------------------------------ #

    def _init_tokenizer(self):
        """Initialize tokenizer.

        Tries tiktoken first, then falls back to HuggingFace AutoTokenizer.
        Subclasses that already have a tokenizer from their engine should set
        ``self.tokenizer`` directly instead of calling this method.
        """
        try:
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except (KeyError, ImportError):
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True,
            )

    def _initialize_positive_negative_tokens(self):
        """Scan the full vocabulary and populate general and specific
        positive/negative token sets.

        Requires ``self.tokenizer`` (with ``.decode()``) to be set
        before calling.

        Sets the following instance attributes (general pattern uses
        ``true/yes`` and ``false/no`` bounded by non-letter characters;
        specific pattern uses ``yes`` / ``no`` bounded by punctuation or
        whitespace)::

            # General
            self.positive_token_ids   : List[int]
            self.negative_token_ids   : List[int]
            self.positive_tokens      : Set[str]
            self.negative_tokens      : Set[str]

            # Specific
            self.specific_positive_token_ids : List[int]
            self.specific_negative_token_ids : List[int]
            self.specific_positive_tokens    : Set[str]
            self.specific_negative_tokens    : Set[str]
        """
        general_pos_pat = regex.compile(
            r'(^|[^\p{L}])(true|yes)([^\p{L}]|$)', regex.IGNORECASE,
        )
        general_neg_pat = regex.compile(
            r'(^|[^\p{L}])(false|no)([^\p{L}]|$)', regex.IGNORECASE,
        )
        specific_pos_pat = regex.compile(
            r'(^|[\.,:\s])yes([\.,:\s]|$)', regex.IGNORECASE,
        )
        specific_neg_pat = regex.compile(
            r'(^|[\.,:\s])no([\.,:\s]|$)', regex.IGNORECASE,
        )

        self.positive_token_ids: List[int] = []
        self.negative_token_ids: List[int] = []
        self.positive_tokens: Set[str] = set()
        self.negative_tokens: Set[str] = set()

        self.specific_positive_token_ids: List[int] = []
        self.specific_negative_token_ids: List[int] = []
        self.specific_positive_tokens: Set[str] = set()
        self.specific_negative_tokens: Set[str] = set()

        vocab_size = getattr(self.tokenizer, "n_vocab", None) or getattr(self.tokenizer, "vocab_size", None)
        if vocab_size is None:
            raise AttributeError(
                f"Cannot determine vocab size: {type(self.tokenizer).__name__} "
                f"has neither 'n_vocab' nor 'vocab_size'"
            )

        for i in range(vocab_size):
            try:
                decoded = self.tokenizer.decode([i])
                if general_pos_pat.search(decoded):
                    self.positive_token_ids.append(i)
                    self.positive_tokens.add(decoded)
                if general_neg_pat.search(decoded):
                    self.negative_token_ids.append(i)
                    self.negative_tokens.add(decoded)
                if specific_pos_pat.search(decoded):
                    self.specific_positive_token_ids.append(i)
                    self.specific_positive_tokens.add(decoded)
                if specific_neg_pat.search(decoded):
                    self.specific_negative_token_ids.append(i)
                    self.specific_negative_tokens.add(decoded)
            except Exception:
                pass

        logger.info(
            f"Token sets for {self.model_name}: "
            f"{len(self.positive_token_ids)} general positive, "
            f"{len(self.negative_token_ids)} general negative, "
            f"{len(self.specific_positive_token_ids)} specific positive, "
            f"{len(self.specific_negative_token_ids)} specific negative"
        )
        print(
            f"[Token init] {self.model_name}: "
            f"{len(self.positive_token_ids)} general positive, "
            f"{len(self.negative_token_ids)} general negative, "
            f"{len(self.specific_positive_token_ids)} specific positive, "
            f"{len(self.specific_negative_token_ids)} specific negative",
            flush=True,
        )

        # --- Fallback: direct encoding if regex scan found nothing ----------
        # LLaMA 3.x (and other SentencePiece/tiktoken) tokenizers may produce
        # tokens with byte-fallback prefixes that don't match the regex.  In
        # that case, encode the canonical yes/no strings directly.
        if not self.positive_token_ids:
            for word in ["yes", "Yes", "YES", " yes", " Yes", "true", "True", "TRUE"]:
                try:
                    ids = self.tokenizer.encode(word, add_special_tokens=False)
                    for tid in ids:
                        if tid not in self.positive_token_ids:
                            self.positive_token_ids.append(tid)
                            self.positive_tokens.add(word)
                except Exception:
                    pass
            logger.warning(
                f"Regex scan found no general positive tokens for {self.model_name}. "
                f"Direct-encode fallback: {len(self.positive_token_ids)} token(s) found."
            )
            print(
                f"[Token init WARNING] No positive tokens via regex — "
                f"direct-encode fallback: {self.positive_token_ids}",
                flush=True,
            )

        if not self.negative_token_ids:
            for word in ["no", "No", "NO", " no", " No", "false", "False", "FALSE"]:
                try:
                    ids = self.tokenizer.encode(word, add_special_tokens=False)
                    for tid in ids:
                        if tid not in self.negative_token_ids:
                            self.negative_token_ids.append(tid)
                            self.negative_tokens.add(word)
                except Exception:
                    pass
            logger.warning(
                f"Regex scan found no general negative tokens for {self.model_name}. "
                f"Direct-encode fallback: {len(self.negative_token_ids)} token(s) found."
            )
            print(
                f"[Token init WARNING] No negative tokens via regex — "
                f"direct-encode fallback: {self.negative_token_ids}",
                flush=True,
            )

        # Apply the same fallback for specific token sets (used by LLMOpenAI).
        if not self.specific_positive_token_ids:
            for word in ["yes", "Yes", " yes", " Yes"]:
                try:
                    ids = self.tokenizer.encode(word, add_special_tokens=False)
                    for tid in ids:
                        if tid not in self.specific_positive_token_ids:
                            self.specific_positive_token_ids.append(tid)
                            self.specific_positive_tokens.add(word)
                except Exception:
                    pass

        if not self.specific_negative_token_ids:
            for word in ["no", "No", " no", " No"]:
                try:
                    ids = self.tokenizer.encode(word, add_special_tokens=False)
                    for tid in ids:
                        if tid not in self.specific_negative_token_ids:
                            self.specific_negative_token_ids.append(tid)
                            self.specific_negative_tokens.add(word)
                except Exception:
                    pass

        # --- Hard failure if still empty after fallback ----------------------
        if not self.positive_token_ids:
            raise ValueError(
                f"No positive tokens (yes/true) found for model '{self.model_name}' "
                f"after both regex scan and direct-encode fallback. "
                f"Verify the tokenizer is loaded correctly."
            )
        if not self.negative_token_ids:
            raise ValueError(
                f"No negative tokens (no/false) found for model '{self.model_name}' "
                f"after both regex scan and direct-encode fallback. "
                f"Verify the tokenizer is loaded correctly."
            )

    def count_prompt_tokens(self, prompts: List[Prompt]) -> List[int]:
        """Return the number of tokens per prompt including chat-format overhead."""
        conversations = [p.to_messages() for p in prompts]
        if hasattr(self.tokenizer, "apply_chat_template"):
            token_lists = self.tokenizer.apply_chat_template(conversations)
            return [len(toks) for toks in token_lists]
        # tiktoken has no chat template; use OpenAI's documented overhead:
        # 3 tokens per message (<|im_start|>{role}\n … <|im_end|>\n)
        # + 3 tokens to prime the assistant reply
        TOKENS_PER_MESSAGE = 3
        TOKENS_REPLY_PRIMING = 3
        counts = []
        for messages in conversations:
            total = TOKENS_REPLY_PRIMING
            for msg in messages:
                total += TOKENS_PER_MESSAGE
                total += len(self.tokenizer.encode(msg["role"]))
                total += len(self.tokenizer.encode(msg.get("content") or ""))
            counts.append(total)
        return counts

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                 #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_text_completion(self, prompts: List[Prompt], max_new_tokens: int = 512) -> List[str]:
        ...

    @abstractmethod
    def get_confidence_first_token(self, prompts: List[Prompt]) -> List[float]:
        ...

    @abstractmethod
    def get_confidence_with_tools(
        self,
        prompts: List[Prompt],
        tools: Sequence[Tool],
        max_iterations: int = 10,
    ) -> List[float]:
        """For each prompt, run an agentic tool-calling loop, then extract
        first-token yes/no confidence.  Returns one float per prompt.

        Each :class:`~llm_tool.Tool` supplies its OpenAI function schema,
        validates arguments via Pydantic on :meth:`~llm_tool.Tool.invoke`, and
        runs its implementation.
        """
        ...
