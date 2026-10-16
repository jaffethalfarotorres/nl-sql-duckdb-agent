"""Groq (free tier) through the official ``groq`` SDK."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, ClassVar

from nl_sql_agent.envelopes import ModelTier
from nl_sql_agent.providers.base import (
    MAX_TOKENS,
    LLMBackend,
    chat_messages,
    import_sdk,
    models_from_env,
    require_env,
)


class GroqBackend(LLMBackend):
    name = "groq"
    default_models: ClassVar[dict[ModelTier, str]] = {
        "small": "llama-3.1-8b-instant",
        "large": "llama-3.3-70b-versatile",
    }

    def __init__(self, client: Any, models: Mapping[ModelTier, str] | None = None) -> None:
        super().__init__(models)
        self._client = client

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> GroqBackend:
        env = os.environ if env is None else env
        api_key = require_env(env, "GROQ_API_KEY", backend=cls.name)
        sdk = import_sdk("groq", extra="groq")
        return cls(sdk.Groq(api_key=api_key), models=models_from_env(env))

    def _complete(self, prompt: str, *, system: str, model: str) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=chat_messages(system, prompt),
            temperature=0,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content
