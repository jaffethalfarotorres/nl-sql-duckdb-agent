"""Hugging Face Inference Providers through ``huggingface_hub.InferenceClient``."""

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


class HuggingFaceBackend(LLMBackend):
    name = "huggingface"
    default_models: ClassVar[dict[ModelTier, str]] = {
        "small": "Qwen/Qwen2.5-7B-Instruct",
        "large": "meta-llama/Llama-3.3-70B-Instruct",
    }

    def __init__(self, client: Any, models: Mapping[ModelTier, str] | None = None) -> None:
        super().__init__(models)
        self._client = client

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> HuggingFaceBackend:
        env = os.environ if env is None else env
        token = require_env(env, "HF_TOKEN", backend=cls.name)
        sdk = import_sdk("huggingface_hub", extra="huggingface")
        return cls(sdk.InferenceClient(token=token), models=models_from_env(env))

    def _complete(self, prompt: str, *, system: str, model: str) -> str:
        # No temperature here: some inference providers reject temperature=0.
        response = self._client.chat_completion(
            chat_messages(system, prompt), model=model, max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content
