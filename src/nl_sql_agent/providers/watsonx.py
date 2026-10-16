"""IBM watsonx.ai (Lite plan works) through the official ``ibm-watsonx-ai`` SDK.

``ModelInference`` validates the model against the service when it is built, so
models are created lazily, one per tier, and reused.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
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

DEFAULT_URL = "https://us-south.ml.cloud.ibm.com"

ModelFactory = Callable[[str], Any]


class WatsonxBackend(LLMBackend):
    name = "watsonx"
    default_models: ClassVar[dict[ModelTier, str]] = {
        "small": "ibm/granite-3-8b-instruct",
        "large": "meta-llama/llama-3-3-70b-instruct",
    }

    def __init__(
        self, model_factory: ModelFactory, models: Mapping[ModelTier, str] | None = None
    ) -> None:
        super().__init__(models)
        self._factory = model_factory
        self._models_by_id: dict[str, Any] = {}

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> WatsonxBackend:
        env = os.environ if env is None else env
        api_key = require_env(env, "WATSONX_APIKEY", backend=cls.name)
        project_id = require_env(env, "WATSONX_PROJECT_ID", backend=cls.name)
        sdk = import_sdk("ibm_watsonx_ai", extra="watsonx")
        foundation_models = import_sdk("ibm_watsonx_ai.foundation_models", extra="watsonx")
        credentials = sdk.Credentials(url=env.get("WATSONX_URL", DEFAULT_URL), api_key=api_key)

        def factory(model_id: str) -> Any:
            return foundation_models.ModelInference(
                model_id=model_id, credentials=credentials, project_id=project_id
            )

        return cls(factory, models=models_from_env(env))

    def _complete(self, prompt: str, *, system: str, model: str) -> str:
        if model not in self._models_by_id:
            self._models_by_id[model] = self._factory(model)
        response = self._models_by_id[model].chat(
            messages=chat_messages(system, prompt),
            params={"temperature": 0, "max_tokens": MAX_TOKENS},
        )
        return response["choices"][0]["message"]["content"]
