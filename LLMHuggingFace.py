import logging
from typing import List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from LLMBase import LLMBase
from llm_tool import Tool
from prompt import Prompt

logger = logging.getLogger(__name__)


class LLMHuggingFace(LLMBase):
    """HuggingFace Transformers backend for local LLaMA 3 / similar models.

    Parameters
    ----------
    model_path:
        Local path to a HuggingFace-compatible model directory.
    device_map:
        Passed to ``from_pretrained``; ``'auto'`` distributes layers
        across all visible GPUs.
    dtype:
        Weight dtype; ``torch.bfloat16`` is recommended for LLaMA 3.
    """

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_name = model_path  # used by LLMBase token-set helpers

        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._initialize_positive_negative_tokens()

        logger.info(
            f"Loading model from {model_path} "
            f"(device_map={device_map}, dtype={dtype})"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.model.eval()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _first_device(self) -> torch.device:
        """Return the device of the first model parameter (used as input device)."""
        return next(self.model.parameters()).device

    def _apply_template(self, prompt: Prompt) -> torch.Tensor:
        """Tokenize *prompt* with the model's chat template and move to device."""
        input_ids = self.tokenizer.apply_chat_template(
            prompt.to_messages(),
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return input_ids.to(self._first_device())

    # ------------------------------------------------------------------ #
    #  LLMBase interface                                                  #
    # ------------------------------------------------------------------ #

    def get_text_completion(
        self, prompts: List[Prompt], max_new_tokens: int = 512
    ) -> List[str]:
        completions: List[str] = []
        for prompt in prompts:
            try:
                input_ids = self._apply_template(prompt)
                n_input = input_ids.shape[1]
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                new_tokens = output_ids[0, n_input:]
                completions.append(
                    self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                )
            except Exception as e:
                logger.error(f"Error in get_text_completion: {e}")
                completions.append("")
        return completions

    def get_confidence_first_token(self, prompts: List[Prompt]) -> List[float]:
        """Return P(yes) / (P(yes) + P(no)) from the next-token logits.

        Uses a single forward pass (no generation) to obtain the full
        next-token distribution, then aggregates probabilities over all
        vocabulary tokens that match the positive / negative patterns
        defined by :meth:`LLMBase._initialize_positive_negative_tokens`.
        """
        scores: List[float] = []
        for prompt in prompts:
            try:
                input_ids = self._apply_template(prompt)
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids)
                # logits: (1, seq_len, vocab_size) → last position
                next_token_logits = outputs.logits[0, -1, :]  # (vocab_size,)
                probs = torch.softmax(next_token_logits, dim=-1).cpu()

                yes_prob = max(
                    (probs[tid].item() for tid in self.positive_token_ids),
                    default=0.0,
                )
                no_prob = max(
                    (probs[tid].item() for tid in self.negative_token_ids),
                    default=0.0,
                )

                total = yes_prob + no_prob
                scores.append(yes_prob / total if total > 0 else 0.5)
            except Exception as e:
                logger.error(f"Error in get_confidence_first_token: {e}")
                scores.append(0.5)
        return scores

    def get_confidence_with_tools(
        self,
        prompts: List[Prompt],
        tools: Sequence[Tool],
        max_iterations: int = 10,
    ) -> List[float]:
        raise NotImplementedError(
            "Tool calling is not supported for the local HuggingFace backend. "
            "Use LLMOpenAI with a vLLM / compatible endpoint instead."
        )
