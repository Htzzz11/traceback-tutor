"""Load a model (optionally with a LoRA adapter) and generate one response with mlx-lm.

Shared by the eval runner and the CLI so both use the same prompt construction and
the same decoding settings. Decoding is greedy (temperature 0): runs are
reproducible and every system is compared under identical settings.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

GENERATION_SETTINGS = {"sampler": "greedy", "temperature": 0.0}


@dataclass
class Model:
    model: Any  # mlx nn.Module
    tokenizer: Any  # mlx_lm TokenizerWrapper
    name: str  # hf id or local path
    adapter: str | None


@dataclass
class Generation:
    text: str
    n_tokens: int  # generated tokens (mlx-lm counts the end-of-sequence token too)
    seconds: float  # wall clock for the whole call: prompt processing + decoding
    finish_reason: str  # "stop" (model ended) or "length" (hit max_tokens: truncated)


# What eval and the CLI depend on, so tests can pass a fake instead of a real model.
GenerateFn = Callable[[list[dict[str, str]]], Generation]


def load(model: str, adapter: str | None = None) -> Model:
    """Load an hf id or local path; apply the LoRA adapter directory if given."""
    import mlx_lm  # imported here so tests and `--help` don't pay for loading mlx

    m, tokenizer = mlx_lm.load(model, adapter_path=adapter)
    return Model(model=m, tokenizer=tokenizer, name=model, adapter=adapter)


def generate(model: Model, messages: list[dict[str, str]], max_tokens: int = 300) -> Generation:
    """Greedy-decode one assistant turn for `messages` (from prompt.build_messages)."""
    import mlx_lm

    prompt = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    text = ""
    start = time.perf_counter()
    # stream_generate with no sampler uses mlx-lm's greedy (argmax) sampler.
    for chunk in mlx_lm.stream_generate(
        model.model, model.tokenizer, prompt, max_tokens=max_tokens
    ):
        text += chunk.text
    seconds = time.perf_counter() - start
    return Generation(
        text=text,
        n_tokens=chunk.generation_tokens,
        seconds=seconds,
        finish_reason=chunk.finish_reason,
    )


def make_generate_fn(model: Model, max_tokens: int = 300) -> GenerateFn:
    return lambda messages: generate(model, messages, max_tokens=max_tokens)
