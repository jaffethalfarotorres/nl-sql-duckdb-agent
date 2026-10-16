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


def _openai_shape(text: str | None) -> SimpleNamespace:
    message = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroqClient:
    """groq.Groq: client.chat.completions.create(...) -> object with choices."""

    def __init__(self, text: str | None = "", error: Exception | None = None) -> None:
        self.kwargs: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._text, self._error = text, error

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.kwargs.append(kwargs)
        if self._error:
            raise self._error
        return _openai_shape(self._text)


class FakeHFClient:
    """huggingface_hub.InferenceClient: client.chat_completion(messages, ...)."""

    def __init__(self, text: str = "") -> None:
        self.kwargs: list[dict[str, Any]] = []
        self._text = text

    def chat_completion(self, messages: list[dict[str, str]], **kwargs: Any) -> SimpleNamespace:
        self.kwargs.append({"messages": messages, **kwargs})
        return _openai_shape(self._text)


class FakeOllamaClient:
    """ollama.Client: client.chat(model=..., messages=...) -> subscriptable response."""

    def __init__(self, text: str = "") -> None:
        self.kwargs: list[dict[str, Any]] = []
        self._text = text

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs.append(kwargs)
        return {"message": {"role": "assistant", "content": self._text}}


class FakeWatsonxModel:
    """ibm_watsonx_ai ModelInference: model.chat(messages, params) -> dict."""

    def __init__(self, text: str) -> None:
        self.kwargs: list[dict[str, Any]] = []
        self._text = text

    def chat(self, messages: list[dict[str, str]], params: dict[str, Any]) -> dict[str, Any]:
        self.kwargs.append({"messages": messages, "params": params})
        return {"choices": [{"message": {"role": "assistant", "content": self._text}}]}


class FakeWatsonxFactory:
    """Stands in for building a ModelInference per model id."""

    def __init__(self, text: str = "") -> None:
        self.created: dict[str, FakeWatsonxModel] = {}
        self._text = text

    def __call__(self, model_id: str) -> FakeWatsonxModel:
        self.created[model_id] = FakeWatsonxModel(self._text)
        return self.created[model_id]


def all_backends(text: str) -> dict[str, LLMBackend]:
    """Every backend, each wired to a fake client that answers ``text``."""
    return {
        "mock": MockBackend(lambda call: text),
        "watsonx": WatsonxBackend(FakeWatsonxFactory(text)),
        "groq": GroqBackend(FakeGroqClient(text)),
        "huggingface": HuggingFaceBackend(FakeHFClient(text)),
        "ollama": OllamaBackend(FakeOllamaClient(text)),
    }
