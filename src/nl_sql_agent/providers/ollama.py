"""Local models through the ``ollama`` SDK. No credentials, no cost."""

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
)

DEFAULT_HOST = "http://localhost:11434"


class OllamaBackend(LLMBackend):
    name = "ollama"
    default_models: ClassVar[dict[ModelTier, str]] = {
        "small": "llama3.2:3b",
        "large": "llama3.1:8b",
    }

    def __init__(self, client: Any, models: Mapping[ModelTier, str] | None = None) -> None:
        super().__init__(models)
        self._client = client

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> OllamaBackend:
        env = os.environ if env is None else env
        sdk = import_sdk("ollama", extra="ollama")
        host = env.get("OLLAMA_HOST", DEFAULT_HOST)
        return cls(sdk.Client(host=host), models=models_from_env(env))

    def _complete(self, prompt: str, *, system: str, model: str) -> str:
        response = self._client.chat(
            model=model,
            messages=chat_messages(system, prompt),
            options={"temperature": 0, "num_predict": MAX_TOKENS},
        )
        return response["message"]["content"]
