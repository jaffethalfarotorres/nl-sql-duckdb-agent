"""The one contract every LLM provider implements.

``LLMBackend.complete`` owns the parts that must behave identically across
providers: tier-to-model resolution, error mapping and response validation.
A concrete provider only implements ``_complete`` (one SDK call) and
``from_env`` (credentials and client construction). Adding a provider means
copying an existing one, not inventing a new pattern.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import ModuleType
from typing import ClassVar

from nl_sql_agent.envelopes import ModelTier

MAX_TOKENS = 512  # a SQL query needs more room than a one-line routing verdict


class BackendError(RuntimeError):
    """A provider call failed: network, auth, quota or a malformed response."""


class BackendUnavailable(BackendError):
    """The provider cannot be used here: SDK not installed or credentials missing."""


class LLMBackend(ABC):
    """Text-in, text-out chat completion behind a provider-neutral interface."""

    name: ClassVar[str]
    default_models: ClassVar[Mapping[ModelTier, str]]

    def __init__(self, models: Mapping[ModelTier, str] | None = None) -> None:
        self.models: dict[ModelTier, str] = {**self.default_models, **(models or {})}

    def complete(self, prompt: str, *, system: str, tier: ModelTier = "small") -> str:
        """Return the model's reply. Any provider failure surfaces as ``BackendError``."""
        model = self.models[tier]
        try:
            text = self._complete(prompt, system=system, model=model)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        if not isinstance(text, str):
            raise BackendError(f"{self.name}: empty or non-text response")
        return text

    @abstractmethod
    def _complete(self, prompt: str, *, system: str, model: str) -> str:
        """Make exactly one provider call and return the raw reply text."""


def chat_messages(system: str, prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def require_env(env: Mapping[str, str], key: str, *, backend: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise BackendUnavailable(f"{backend}: set {key} (see .env.example)")
    return value


def import_sdk(module: str, *, extra: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        hint = f"pip install 'nl-sql-duckdb-agent[{extra}]'"
        raise BackendUnavailable(f"{module} is not installed: {hint}") from exc


def models_from_env(env: Mapping[str, str]) -> dict[ModelTier, str]:
    """Optional per-tier model overrides shared by every backend."""
    overrides: dict[ModelTier, str] = {}
    if small := env.get("NLSQL_MODEL_SMALL"):
        overrides["small"] = small
    if large := env.get("NLSQL_MODEL_LARGE"):
        overrides["large"] = large
    return overrides
