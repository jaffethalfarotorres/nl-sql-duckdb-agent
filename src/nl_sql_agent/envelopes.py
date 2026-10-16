"""Typed envelopes exchanged between the defense scan, the planner, the validator,
the executor and the agent.

Every hand-off is a frozen Pydantic model that rejects unknown fields, so any
caller (a UI, another agent, a test) can rely on its shape. ``AgentResult``
checks its own invariants on construction. The one that matters most: a result
can carry rows only if the exact SQL that produced them passed validation, so
"the model's SQL ran unchecked" is a state the type cannot represent.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

SCHEMA_VERSION = "1.0"

ModelTier = Literal["small", "large"]


class _Envelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _recompute_derived_fields(cls, data: Any) -> Any:
        # Computed fields (``ok``, ``row_count``) are written to JSON for readers but
        # never trusted on input: they are dropped here and recomputed from the data.
        if isinstance(data, dict) and cls.model_computed_fields:
            return {k: v for k, v in data.items() if k not in cls.model_computed_fields}
        return data


# --- prompt defense ------------------------------------------------------------------


class Finding(_Envelope):
    """One prompt-defense rule that matched the question."""

    rule: str
    excerpt: str


class DefenseReport(_Envelope):
    """Outcome of the prompt-defense scan that runs before the model sees anything."""

    blocked: bool
    findings: tuple[Finding, ...] = ()


# --- planner -------------------------------------------------------------------------


class SQLProposal(_Envelope):
    """What the model proposed. ``sql=None`` means it declined to write a query."""

    sql: str | None
    explanation: str = ""
    parse: str
    """How the reply was read: clean_json, markdown_fenced, prose_wrapped or direct."""
    model_tier: ModelTier | None = None


# --- validator -----------------------------------------------------------------------


class Layer(StrEnum):
    """Which validator layer raised a violation."""

    LEXER = "lexer"
    ENGINE = "engine"


class Violation(_Envelope):
    """One reason the validator refused a statement."""

    layer: Layer
    rule: str
    detail: str = ""


class ValidationReport(_Envelope):
    """The validator's verdict on one SQL string."""

    sql: str
    violations: tuple[Violation, ...] = ()
    tables: tuple[str, ...] = ()
    """Base tables the statement reads, as resolved by the engine layer."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def rules(self) -> set[str]:
        return {violation.rule for violation in self.violations}


# --- executor ------------------------------------------------------------------------


class QueryResult(_Envelope):
    """Rows returned by the read-only executor, capped at its row limit."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool = False
    elapsed_ms: float = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def row_count(self) -> int:
        return len(self.rows)

    @model_validator(mode="after")
    def _rows_match_columns(self) -> Self:
        width = len(self.columns)
        if any(len(row) != width for row in self.rows):
            raise ValueError(f"every row must have {width} values, one per columns entry")
        return self


# --- agent ---------------------------------------------------------------------------


class AgentStatus(StrEnum):
    COMPLETED = "completed"
    BLOCKED = "blocked"
    """The prompt-defense scan stopped the question before any model call."""
    REJECTED = "rejected"
    """The validator refused the SQL. Nothing was executed."""
    UNANSWERABLE = "unanswerable"
    """The model declined to write SQL (data not in the schema, or a request to change it)."""
    SQL_ERROR = "sql_error"
    """Validated SQL failed in the engine or hit the time limit."""
    LLM_ERROR = "llm_error"
    """The model gave no usable reply."""


_ERROR_STATUSES = {AgentStatus.SQL_ERROR, AgentStatus.LLM_ERROR}


class AgentResult(_Envelope):
    """Final output of one agent run. Its invariants are checked on construction."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    status: AgentStatus
    question: str
    defense: DefenseReport
    proposal: SQLProposal | None = None
    validation: ValidationReport | None = None
    result: QueryResult | None = None
    error: str | None = None
    trace: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        status, proposal, validation = self.status, self.proposal, self.validation
        blocked = status is AgentStatus.BLOCKED
        if blocked != self.defense.blocked:
            raise ValueError("status 'blocked' must match the defense report")
        if blocked and (proposal or validation or self.result):
            raise ValueError("a blocked question never reaches the model or the database")
        if validation is not None and (proposal is None or proposal.sql != validation.sql):
            raise ValueError("the validator must have checked exactly the proposed SQL")
        if self.result is not None and (validation is None or not validation.ok):
            raise ValueError("rows require SQL that passed validation")
        if (status is AgentStatus.COMPLETED) != (self.result is not None):
            raise ValueError("status 'completed' is exactly the case with a query result")
        if status is AgentStatus.REJECTED and (validation is None or validation.ok):
            raise ValueError("status 'rejected' requires a failed validation report")
        if status is AgentStatus.UNANSWERABLE and (proposal is None or proposal.sql):
            raise ValueError("status 'unanswerable' means the model proposed no SQL")
        if status is AgentStatus.SQL_ERROR and (validation is None or not validation.ok):
            raise ValueError("status 'sql_error' is for SQL that passed validation")
        if status is AgentStatus.LLM_ERROR and proposal is not None:
            raise ValueError("status 'llm_error' means there is no usable proposal")
        if (status in _ERROR_STATUSES) != (self.error is not None):
            raise ValueError("an error message is required exactly for error statuses")
        return self
