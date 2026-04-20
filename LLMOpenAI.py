from typing import Any, Dict, List, Optional, Sequence, Set

from openai import OpenAI
from openai.types.chat import ChatCompletion

from LLMBase import LLMBase
from llm_tool import Tool
from prompt import Prompt
import math
import os
import logging
import tempfile
import time
import json

logger = logging.getLogger(__name__)


class LLMOpenAI(LLMBase):
    """
    A wrapper around OpenAI API providing utilities for generation
    and confidence estimation for binary (yes/no) style outputs.
    Uses synchronous per-request calls.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_poll_interval: Optional[float] = None,
    ):
        self.model_name = model_name
        self.batch_poll_interval = batch_poll_interval
        self._init_tokenizer()
        self._initialize_positive_negative_tokens()

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    def _chat_completions_batched(self, prompts: List[Prompt], **kwargs) -> List[ChatCompletion]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i, p in enumerate(prompts):
                req = {
                    "custom_id": f"req-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model_name,
                        "messages": p.to_messages(),
                        **kwargs,
                    }
                }
                f.write(json.dumps(req) + "\n")
            jsonl_path = f.name
        try:
            uploaded = self.client.files.create(
                file=open(jsonl_path, "rb"),
                purpose="batch",
            )
        finally:
            os.remove(jsonl_path)

        batch = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        logger.info(f"Batch {batch.id} created – waiting for completion …")

        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            time.sleep(self.batch_poll_interval)
            batch = self.client.batches.retrieve(batch.id)
            logger.info(f"Batch {batch.id} status: {batch.status}")

        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch.id} ended with status: {batch.status}")

        result_file = self.client.files.content(batch.output_file_id)
        results_by_id = {}
        for line in result_file.text.splitlines():
            obj = json.loads(line)
            results_by_id[obj["custom_id"]] = ChatCompletion.model_validate(obj["response"]["body"])

        missing = [f"req-{i}" for i in range(len(prompts)) if f"req-{i}" not in results_by_id]
        if missing:
            errors = {}
            if batch.error_file_id:
                error_file = self.client.files.content(batch.error_file_id)
                for line in error_file.text.splitlines():
                    err = json.loads(line)
                    if err["custom_id"] in missing:
                        errors[err["custom_id"]] = err.get("error", err.get("response", {}))
            raise RuntimeError(f"Batch {batch.id} missing results for: {missing}, errors: {errors}")

        return [results_by_id[f"req-{i}"] for i in range(len(prompts))]

    def _chat_completions_synchronous(self, prompts: List[Prompt], **kwargs) -> List[ChatCompletion]:
        completions: List[ChatCompletion] = []
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt.to_messages(),
                **kwargs,
            )
            completions.append(response)
        return completions

    def _chat_completions(self, prompts: List[Prompt], **kwargs) -> List[ChatCompletion]:
        if self.batch_poll_interval is not None:
            return self._chat_completions_batched(prompts, **kwargs)
        return self._chat_completions_synchronous(prompts, **kwargs)

    # ------------------------------------------------------------------ #
    #  Public methods                                                    #
    # ------------------------------------------------------------------ #

    def get_text_completion(self, prompts: List[Prompt], max_new_tokens: int = 512) -> List[str]:
        responses = self._chat_completions(prompts, max_tokens=max_new_tokens, temperature=0.0)
        completions: List[str] = []
        for response in responses:
            try:
                completions.append(response.choices[0].message.content or "")
            except Exception as e:
                logger.error(f"Error generating completion: {e}")
                completions.append("")
        return completions

    def get_confidence_first_token(self, prompts: List[Prompt]) -> List[float]:
        """Return P(yes) / (P(yes) + P(no)) derived from first-token logprobs."""
        responses = self._chat_completions(
            prompts, max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=20,
        )
        scores: List[float] = []
        for response in responses:
            try:
                top_lps = response.choices[0].logprobs.content[0].top_logprobs

                yes_prob = max(
                    (math.exp(lp.logprob) for lp in top_lps if lp.token in self.positive_tokens),
                    default=0.0,
                )
                no_prob = max(
                    (math.exp(lp.logprob) for lp in top_lps if lp.token in self.negative_tokens),
                    default=0.0,
                )

                total = yes_prob + no_prob
                scores.append(yes_prob / total if total > 0 else 0.5)
            except Exception as e:
                logger.error(f"Error computing confidence: {e}")
                scores.append(0.5)
        return scores

    def get_confidence_with_tools(
        self,
        prompts: List[Prompt],
        tools: Sequence[Tool],
        max_iterations: int = 10,
        final_user_message: str = "Based on your exploration, answer with 'yes' or 'no':",
    ) -> List[float]:
        """Batched tool exploration: each round calls the API for all conversations
        that still need a model turn; after tool results, those stay in the next
        batch.  Ends with one batched yes/no logprob call for all prompts."""
        if not prompts:
            return []
        tools_by_name = {t.name: t for t in tools}
        openai_tools = [t.openai_function_dict() for t in tools]
        n = len(prompts)
        exploration_done = [False] * n

        for _ in range(max_iterations):
            active_idx = [i for i in range(n) if not exploration_done[i]]
            if not active_idx:
                break
            batch = [prompts[i] for i in active_idx]
            responses = self._chat_completions(batch, tools=openai_tools, temperature=0.0)
            for i, response in zip(active_idx, responses):
                choice = response.choices[0]

                tcs = choice.message.tool_calls
                if tcs:
                    prompts[i].assistant(
                        choice.message.content or "",
                        tool_calls=[tc.model_dump() for tc in tcs],
                    )

                    for tc in tcs:
                        fn_name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments)
                            if not isinstance(args, dict):
                                args = {}
                        except json.JSONDecodeError:
                            args = {}
                        tool = tools_by_name.get(fn_name)
                        if tool is None:
                            result_text = f"Error: unknown tool '{fn_name}'"
                        else:
                            result_text = tool.invoke(args)
                        prompts[i].tool(result_text, tc.id)
                elif choice.message.content:
                    prompts[i].assistant(choice.message.content or "")
                    exploration_done[i] = True

        for p in prompts:
            p.user(final_user_message)
        
        return self.get_confidence_first_token(prompts)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    openai_wrapper = LLMOpenAI(api_key=os.getenv("OPENAI_API_KEY"), batch_poll_interval=10)
    prompts = [Prompt().user("Is the following sentence true: 'The capital of France is Paris.'? Answer only yes or no.")]
    #print(f"confidence(yes): {openai_wrapper.get_confidence(prompts)}")
    print(f"text: {openai_wrapper.get_text_completion(prompts)}")
