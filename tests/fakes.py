"""Credential-free stand-ins for each provider SDK client.

Each fake mimics only the call shape the matching backend uses (verified against
the SDK source) and records the keyword arguments it received.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from nl_sql_agent.providers import (
    GroqBackend,
    HuggingFaceBackend,
    LLMBackend,
    MockBackend,
    OllamaBackend,
    WatsonxBackend,
)


def _openai_shape(text: str | None, finish_reason: str = "stop") -> SimpleNamespace:
    message = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


class FakeGroqClient:
    """groq.Groq: client.chat.completions.create(...) -> object with choices."""

    def __init__(
        self,
        text: str | None = "",
        error: Exception | None = None,
        finish_reason: str = "stop",
    ) -> None:
        self.kwargs: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._text, self._error, self._finish_reason = text, error, finish_reason

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.kwargs.append(kwargs)
        if self._error:
            raise self._error
        return _openai_shape(self._text, self._finish_reason)


class FakeHFClient:
    """huggingface_hub.InferenceClient: client.chat_completion(messages, ...)."""

    def __init__(self, text: str = "", finish_reason: str = "stop") -> None:
        self.kwargs: list[dict[str, Any]] = []
        self._text, self._finish_reason = text, finish_reason

    def chat_completion(self, messages: list[dict[str, str]], **kwargs: Any) -> SimpleNamespace:
        self.kwargs.append({"messages": messages, **kwargs})
        return _openai_shape(self._text, self._finish_reason)


class FakeOllamaClient:
    """ollama.Client: client.chat(model=..., messages=...) -> subscriptable response."""

    def __init__(self, text: str = "", done_reason: str = "stop") -> None:
        self.kwargs: list[dict[str, Any]] = []
        self._text, self._done_reason = text, done_reason

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs.append(kwargs)
        message = {"role": "assistant", "content": self._text}
        return {"message": message, "done_reason": self._done_reason}


class FakeWatsonxModel:
    """ibm_watsonx_ai ModelInference: model.chat(messages, params) -> dict."""

    def __init__(self, text: str, finish_reason: str = "stop") -> None:
        self.kwargs: list[dict[str, Any]] = []
        self._text, self._finish_reason = text, finish_reason

    def chat(self, messages: list[dict[str, str]], params: dict[str, Any]) -> dict[str, Any]:
        self.kwargs.append({"messages": messages, "params": params})
        message = {"role": "assistant", "content": self._text}
        return {"choices": [{"message": message, "finish_reason": self._finish_reason}]}


class FakeWatsonxFactory:
    """Stands in for building a ModelInference per model id."""

    def __init__(self, text: str = "", finish_reason: str = "stop") -> None:
        self.created: dict[str, FakeWatsonxModel] = {}
        self._text, self._finish_reason = text, finish_reason

    def __call__(self, model_id: str) -> FakeWatsonxModel:
        self.created[model_id] = FakeWatsonxModel(self._text, self._finish_reason)
        return self.created[model_id]


def all_backends(text: str, stop: str = "stop") -> dict[str, LLMBackend]:
    """Every backend, each wired to a fake client that answers ``text`` and reports why it
    stopped (``"length"``: cut at the token cap). The mock has no stop reason."""
    return {
        "mock": MockBackend(lambda call: text),
        "watsonx": WatsonxBackend(FakeWatsonxFactory(text, stop)),
        "groq": GroqBackend(FakeGroqClient(text, finish_reason=stop)),
        "huggingface": HuggingFaceBackend(FakeHFClient(text, stop)),
        "ollama": OllamaBackend(FakeOllamaClient(text, stop)),
    }
