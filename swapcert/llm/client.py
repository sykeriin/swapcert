"""LLM access: LiteLLM wrapper with caching, budget caps and a token ledger.

Budget is $0, so the ledger counts calls and tokens and enforces `max_calls`, not dollars.
Local models are reached through an OpenAI-compatible server (Ollama / llama.cpp), e.g.
    LLM(model="openai/qwen3:4b", api_base="http://localhost:11434/v1", api_key="local")
Tests inject `backend=` so nothing here needs a network or GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .cache import Cache, key_for

Backend = Callable[[str, list[dict], dict], dict]  # (model, messages, params) -> {"text","in","out"}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Ledger:
    calls: int = 0
    cached: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    by_model: dict[str, int] = field(default_factory=dict)


class LLM:
    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        cache: Cache | None = None,
        max_calls: int | None = None,
        backend: Backend | None = None,
    ):
        self.model, self.api_base, self.api_key = model, api_base, api_key
        self.cache = cache or Cache()
        self.max_calls = max_calls
        self.ledger = Ledger()
        self._backend = backend or self._litellm

    def complete(self, messages: list[dict], temperature: float = 0.0, seed: int = 0, **params: Any) -> str:
        params = {"temperature": temperature, "seed": seed, **params}
        k = key_for(model=self.model, messages=messages, params=params)
        hit = self.cache.get(k)
        if hit is not None:
            self.ledger.cached += 1
            return hit["text"]
        if self.max_calls is not None and self.ledger.calls >= self.max_calls:
            raise BudgetExceeded(f"max_calls={self.max_calls} reached")
        out = self._backend(self.model, messages, params)
        self.ledger.calls += 1
        self.ledger.tokens_in += out.get("in", 0)
        self.ledger.tokens_out += out.get("out", 0)
        self.ledger.by_model[self.model] = self.ledger.by_model.get(self.model, 0) + 1
        self.cache.put(k, out)
        return out["text"]

    def _litellm(self, model: str, messages: list[dict], params: dict) -> dict:
        try:
            import litellm  # optional dependency: pip install 'swapcert[llm]'
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("install swapcert[llm] to call real models") from e
        r = litellm.completion(model=model, messages=messages, api_base=self.api_base, api_key=self.api_key, **params)
        u = r.usage
        return {"text": r.choices[0].message.content or "", "in": u.prompt_tokens, "out": u.completion_tokens}
