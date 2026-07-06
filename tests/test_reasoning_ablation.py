"""D9 reasoning-ablation plumbing: the pure flags->extra_body map, and that
LLMOpenAI rides extra_body on the live (parallel/sync) chat paths only — never
on the (unused) batched path where it would nest wrongly into the request body."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_stage2_experiment import _reasoning_extra_body
from LLMOpenAI import LLMOpenAI


# ---------------------------------------------------- pure flags -> extra_body

def test_extra_body_empty_by_default():
    assert _reasoning_extra_body() == {}
    assert _reasoning_extra_body(None, False) == {}


def test_extra_body_reasoning_effort_only():
    assert _reasoning_extra_body(reasoning_effort="low") == {"reasoning_effort": "low"}


def test_extra_body_disable_thinking_only():
    assert _reasoning_extra_body(disable_thinking=True) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_extra_body_both_flags():
    assert _reasoning_extra_body("high", True) == {
        "reasoning_effort": "high",
        "chat_template_kwargs": {"enable_thinking": False},
    }


# ------------------------------------------------- LLMOpenAI injection point

def _bare_llm(extra_body):
    """LLMOpenAI without the network/tokenizer __init__ — just the fields
    _chat_completions dispatches on."""
    o = object.__new__(LLMOpenAI)
    o.batch_poll_interval = None
    o.max_concurrency = 16
    o.extra_body = dict(extra_body) if extra_body else {}
    return o


def test_injects_extra_body_on_parallel_path():
    o = _bare_llm({"reasoning_effort": "low"})
    seen = {}
    o._chat_completions_parallel = lambda prompts, **kw: seen.update(kw) or []
    o._chat_completions(["p1", "p2"])  # len>1 -> parallel path
    assert seen.get("extra_body") == {"reasoning_effort": "low"}


def test_injects_extra_body_on_synchronous_path():
    o = _bare_llm({"chat_template_kwargs": {"enable_thinking": False}})
    seen = {}
    o._chat_completions_synchronous = lambda prompts, **kw: seen.update(kw) or []
    o._chat_completions(["only-one"])  # len==1 -> synchronous path
    assert seen.get("extra_body") == {"chat_template_kwargs": {"enable_thinking": False}}


def test_no_extra_body_key_when_empty():
    o = _bare_llm({})
    seen = {}
    o._chat_completions_parallel = lambda prompts, **kw: seen.update(kw) or []
    o._chat_completions(["p1", "p2"])
    assert "extra_body" not in seen


def test_extra_body_preserves_existing_caller_kwargs():
    o = _bare_llm({"reasoning_effort": "low"})
    seen = {}
    o._chat_completions_parallel = lambda prompts, **kw: seen.update(kw) or []
    o._chat_completions(["p1", "p2"], max_tokens=1, temperature=0.0)
    assert seen["max_tokens"] == 1 and seen["temperature"] == 0.0
    assert seen["extra_body"] == {"reasoning_effort": "low"}
