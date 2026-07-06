from concurrent.futures import ThreadPoolExecutor
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
    and confidence estimation.

    Concurrency model
    -----------------
    Per-request calls are dispatched through ``_chat_completions`` which
    chooses one of three paths:

      - ``batch_poll_interval`` set         → OpenAI Batch API
                                              (file upload + poll, 24h window)
      - ``max_concurrency`` > 1, len > 1    → ThreadPoolExecutor with
                                              ``max_concurrency`` workers.
                                              Order is preserved via index
                                              tagging; per-request failures
                                              return ``None`` so a single
                                              bad call doesn't kill the batch.
      - otherwise                           → sequential synchronous loop

    The thread-pool path exists because vLLM continuous batching schedules
    concurrent requests far more efficiently than vLLM's queue handling of
    serial calls. The 2026-06-02 smoke run (job 255327) measured ~60 s per
    sequential call on Llama-3.3-70B-AWQ + 2× A40 with ``--enforce-eager``
    (decode at ~4 tok/s × ``max_new_tokens=256``). With ``max_concurrency=16``
    we expect the equivalent wall time to drop ~10-16× since vLLM has KV
    cache headroom for that many concurrent sequences.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_poll_interval: Optional[float] = None,
        max_concurrency: int = 16,
        extra_body: Optional[dict] = None,
    ):
        self.model_name = model_name
        self.batch_poll_interval = batch_poll_interval
        self.max_concurrency = max(1, int(max_concurrency))
        # Backend-specific request fields (e.g. gpt-oss reasoning_effort, or
        # chat_template_kwargs={"enable_thinking": False} for hybrid reasoners).
        # Rides every non-batch chat.completions.create() via extra_body.
        self.extra_body = dict(extra_body) if extra_body else {}
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

    def _chat_completions_parallel(
        self, prompts: List[Prompt], **kwargs,
    ) -> List[Optional[ChatCompletion]]:
        """ThreadPoolExecutor over ``self.max_concurrency`` workers.

        Input order is preserved in the returned list. A per-call exception
        is logged and replaced by ``None`` in that slot so the surrounding
        batch survives one bad request — downstream consumers must handle
        ``None`` (see ``get_text_completion`` / ``get_text_completion_with_logprobs``).

        ``httpx`` (the OpenAI client's transport) releases the GIL during
        network I/O, so threads scale linearly for this workload.
        """
        n = len(prompts)
        results: List[Optional[ChatCompletion]] = [None] * n

        def _call(idx_prompt):
            idx, prompt = idx_prompt
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt.to_messages(),
                    **kwargs,
                )
                return idx, resp
            except Exception as e:
                logger.error(
                    "Parallel chat-completion failed for prompt %d/%d: %s",
                    idx + 1, n, e,
                )
                return idx, None

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            for idx, resp in ex.map(_call, list(enumerate(prompts))):
                results[idx] = resp
        return results

    def _chat_completions(
        self, prompts: List[Prompt], **kwargs,
    ) -> List[Optional[ChatCompletion]]:
        if self.batch_poll_interval is not None:
            return self._chat_completions_batched(prompts, **kwargs)
        # Inject backend-specific fields on the live (vLLM) paths only. The
        # batched builder spreads **kwargs straight into the request body, where
        # a nested extra_body would be wrong — but that path is unused here.
        if self.extra_body:
            kwargs.setdefault("extra_body", {}).update(self.extra_body)
        if self.max_concurrency > 1 and len(prompts) > 1:
            return self._chat_completions_parallel(prompts, **kwargs)
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

    def get_text_completion_with_logprobs(
        self, prompts: List[Prompt], max_new_tokens: int = 256,
        temperature: float = 0.0, top_p: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Text completion with per-token logprobs.

        Decoding: ``temperature``/``top_p`` are configurable for the Stage-2
        matrix (reasoners run at model-recommended temp>0; non-reasoners at
        temp=0). Default temp=0.0 preserves the prior greedy behaviour.

        Primary extraction path for Stage-2 multi-class relation classification:
        reasoner models (gpt-oss, Gemma-4-thinking) emit chain-of-thought before
        the answer, so first-token logit-comparison is structurally unfair.
        Generation + parse is uniform across reasoner and non-reasoner models.

        Returns one dict per prompt with::

            {
                "text":           full generated text (str),
                "tokens":         per-token string of the chosen token (list[str]),
                "token_logprobs": per-token logprob of the chosen token (list[float]),
                "sum_logprob":    sum of token_logprobs (joint log-prob of the
                                  greedy completion),
                "n_tokens":       len(token_logprobs),
            }

        ``tokens`` is aligned 1:1 with ``token_logprobs`` so downstream code
        can isolate the answer span (Stufe-B B2 answer-span mean-logprob).
        On error per prompt: text="" and empty / zero numeric fields.
        """
        extra = {"top_p": top_p} if top_p is not None else {}
        responses = self._chat_completions(
            prompts, max_tokens=max_new_tokens, temperature=temperature,
            logprobs=True, **extra,
        )
        out: List[Dict[str, Any]] = []
        for response in responses:
            try:
                text = response.choices[0].message.content or ""
                lp_obj = getattr(response.choices[0], "logprobs", None)
                content = getattr(lp_obj, "content", None) if lp_obj is not None else None
                if content:
                    pairs = [(getattr(t, "token", ""), float(t.logprob))
                             for t in content if t.logprob is not None]
                    tokens = [tok for tok, _ in pairs]
                    token_logprobs = [lp for _, lp in pairs]
                else:
                    tokens, token_logprobs = [], []
                out.append({
                    "text":           text,
                    "tokens":         tokens,
                    "token_logprobs": token_logprobs,
                    "sum_logprob":    float(sum(token_logprobs)),
                    "n_tokens":       len(token_logprobs),
                })
            except Exception as e:
                logger.error(f"Error in get_text_completion_with_logprobs: {e}")
                out.append({
                    "text": "", "tokens": [], "token_logprobs": [],
                    "sum_logprob": 0.0, "n_tokens": 0,
                })
        return out

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
