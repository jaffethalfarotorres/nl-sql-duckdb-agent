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

# Transport retries, not answer retries: the SDK waits the server's retry-after on a 429
# or 5xx and sends the same request again. The free tier allows 8,000 tokens per minute
# per model, which a full eval run exceeds; the SDK default of 2 retries gave up mid-run.
MAX_RETRIES = 6


class GroqBackend(LLMBackend):
    name = "groq"
    default_models: ClassVar[dict[ModelTier, str]] = {
        "small": "openai/gpt-oss-20b",
        "large": "openai/gpt-oss-120b",
    }

    def __init__(self, client: Any, models: Mapping[ModelTier, str] | None = None) -> None:
        super().__init__(models)
        self._client = client

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> GroqBackend:
        env = os.environ if env is None else env
        api_key = require_env(env, "GROQ_API_KEY", backend=cls.name)
        sdk = import_sdk("groq", extra="groq")
        client = sdk.Groq(api_key=api_key, max_retries=MAX_RETRIES)
        return cls(client, models=models_from_env(env))

    def _complete(self, prompt: str, *, system: str, model: str) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=chat_messages(system, prompt),
            temperature=0,
            max_tokens=MAX_TOKENS,
        )
        choice = response.choices[0]
        self._reject_truncated(getattr(choice, "finish_reason", None))
        return choice.message.content
