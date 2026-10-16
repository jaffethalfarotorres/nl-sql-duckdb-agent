"""Deterministic backend for tests: scripted replies, recorded calls, simulated outages."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar

from nl_sql_agent.envelopes import ModelTier
from nl_sql_agent.providers.base import BackendError, LLMBackend


@dataclass(frozen=True)
class LLMCall:
    prompt: str
    system: str
    model: str


Responder = Callable[[LLMCall], str]


class MockBackend(LLMBackend):
    name = "mock"
    default_models: ClassVar[dict[ModelTier, str]] = {"small": "mock-small", "large": "mock-large"}

    def __init__(
        self, responses: Iterable[str] | Responder = (), *, fail_with: str | None = None
    ) -> None:
        super().__init__()
        self.calls: list[LLMCall] = []
        self._fail_with = fail_with
        self._responder = responses if callable(responses) else None
        self._script = iter(()) if callable(responses) else iter(responses)

    def _complete(self, prompt: str, *, system: str, model: str) -> str:
        call = LLMCall(prompt=prompt, system=system, model=model)
        self.calls.append(call)
        if self._fail_with:
            raise BackendError(f"mock: {self._fail_with}")
        if self._responder:
            return self._responder(call)
        try:
            return next(self._script)
        except StopIteration:
            raise BackendError("mock: no scripted response left") from None
