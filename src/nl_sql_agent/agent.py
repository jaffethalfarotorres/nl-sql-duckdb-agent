"""The agent: defense -> plan -> validate -> execute, with one informed repair.

The model proposes SQL; code decides whether it runs:

* the question must pass the prompt-defense scan before the model sees it;
* every proposal, including a repaired one, goes through the full validator;
* only validated SQL reaches the read-only executor;
* a policy rejection is final. Letting the model retry against the validator
  would turn the guard into an oracle that an attacker can probe;
* an engine error the model can fix (unknown column, syntax) gets exactly one
  informed retry on the large tier, with the failed query and the error in the
  prompt. Timeouts and refusals are not retried.

Every run returns an ``AgentResult`` whose validators reject contradictory
states, plus a trace of each step that fired.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Self

from nl_sql_agent.dataset import PUBLIC_TABLES
from nl_sql_agent.defense import DEFAULT_MAX_CHARS, scan
from nl_sql_agent.envelopes import (
    AgentResult,
    AgentStatus,
    DefenseReport,
    ModelTier,
    QueryResult,
    SQLProposal,
    ValidationReport,
)
from nl_sql_agent.executor import DEFAULT_MAX_ROWS, DEFAULT_TIMEOUT_S, Executor, QueryError
from nl_sql_agent.planner import PlanError, Planner, RepairContext
from nl_sql_agent.providers.base import LLMBackend
from nl_sql_agent.validator import SQLValidator

_REPAIRABLE_RULES = {"parse_error"}
_TRACE_CHARS = 120

Attempt = tuple[ValidationReport, QueryResult | None, QueryError | None]


class Agent:
    def __init__(
        self,
        validator: SQLValidator,
        executor: Executor,
        planner: Planner | None = None,
        *,
        tier: ModelTier = "small",
        repair_tier: ModelTier = "large",
        max_question_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self.validator = validator
        self.executor = executor
        self.planner = planner
        self.tier = tier
        self.repair_tier = repair_tier
        self.max_question_chars = max_question_chars

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.validator.close()
        self.executor.close()

    def ask(self, question: str) -> AgentResult:
        """Answer a natural-language question. Requires an LLM backend."""
        if self.planner is None:
            raise ValueError("asking a question needs an LLM backend; use run_sql() without one")
        defense = scan(question, max_chars=self.max_question_chars)
        if defense.blocked:
            rules = ", ".join(finding.rule for finding in defense.findings)
            return AgentResult(
                status=AgentStatus.BLOCKED,
                question=question,
                defense=defense,
                trace=(f"defense: blocked ({rules})",),
            )
        trace = ["defense: clean"]

        try:
            proposal = self.planner.propose(question, tier=self.tier)
        except PlanError as exc:
            return self._llm_error(question, defense, trace, exc)
        trace.append(f"plan: {proposal.model_tier} tier, {proposal.parse}")
        if proposal.sql is None:
            return self._unanswerable(question, defense, trace, proposal)

        attempt = self._attempt(proposal, trace)
        if (error := self._repairable(attempt)) is not None:
            trace.append(f"repair: one informed retry on the {self.repair_tier} tier")
            repair = RepairContext(sql=proposal.sql, error=error)
            try:
                proposal = self.planner.propose(question, tier=self.repair_tier, repair=repair)
            except PlanError as exc:
                return self._llm_error(question, defense, trace, exc)
            trace.append(f"plan: {proposal.model_tier} tier, {proposal.parse}")
            if proposal.sql is None:
                return self._unanswerable(question, defense, trace, proposal)
            attempt = self._attempt(proposal, trace)
        return self._finish(question, defense, proposal, attempt, trace)

    def run_sql(self, sql: str) -> AgentResult:
        """Validate and run SQL directly, with no model in the loop and no repair."""
        defense = DefenseReport(blocked=False)
        trace = ["defense: skipped (direct SQL, no model in the loop)"]
        proposal = SQLProposal(sql=sql, parse="direct")
        attempt = self._attempt(proposal, trace)
        return self._finish(sql, defense, proposal, attempt, trace)

    # --- steps -----------------------------------------------------------------------

    def _attempt(self, proposal: SQLProposal, trace: list[str]) -> Attempt:
        validation = self.validator.validate(proposal.sql or "")
        if not validation.ok:
            rules = ", ".join(sorted(validation.rules))
            trace.append(f"validate: rejected ({rules})")
            return validation, None, None
        trace.append(f"validate: pass (reads {', '.join(validation.tables) or 'no table'})")
        try:
            result = self.executor.run(validation)
        except QueryError as exc:
            trace.append(f"execute: {exc.kind}: {exc}"[:_TRACE_CHARS])
            return validation, None, exc
        note = ", truncated" if result.truncated else ""
        trace.append(f"execute: {result.row_count} rows in {result.elapsed_ms:.1f} ms{note}")
        return validation, result, None

    @staticmethod
    def _repairable(attempt: Attempt) -> str | None:
        """The error to show the model if one informed retry could help, else None."""
        validation, _, failure = attempt
        if failure is not None:
            return str(failure) if failure.fixable else None
        if validation.violations and validation.rules <= _REPAIRABLE_RULES:
            return "; ".join(violation.detail for violation in validation.violations)
        return None

    @staticmethod
    def _finish(
        question: str,
        defense: DefenseReport,
        proposal: SQLProposal,
        attempt: Attempt,
        trace: list[str],
    ) -> AgentResult:
        validation, result, failure = attempt
        if not validation.ok:
            status, error = AgentStatus.REJECTED, None
        elif failure is not None:
            status, error = AgentStatus.SQL_ERROR, f"{failure.kind}: {failure}"
        else:
            status, error = AgentStatus.COMPLETED, None
        return AgentResult(
            status=status,
            question=question,
            defense=defense,
            proposal=proposal,
            validation=validation,
            result=result,
            error=error,
            trace=tuple(trace),
        )

    @staticmethod
    def _unanswerable(
        question: str, defense: DefenseReport, trace: list[str], proposal: SQLProposal
    ) -> AgentResult:
        trace.append(f"plan: model declined ({proposal.explanation})"[:_TRACE_CHARS])
        return AgentResult(
            status=AgentStatus.UNANSWERABLE,
            question=question,
            defense=defense,
            proposal=proposal,
            trace=tuple(trace),
        )

    @staticmethod
    def _llm_error(
        question: str, defense: DefenseReport, trace: list[str], exc: PlanError
    ) -> AgentResult:
        trace.append(f"plan: unusable reply ({exc.diagnosis})")
        return AgentResult(
            status=AgentStatus.LLM_ERROR,
            question=question,
            defense=defense,
            error=str(exc),
            trace=tuple(trace),
        )


def build_agent(
    db_path: Path,
    backend: LLMBackend | None = None,
    *,
    allowed_tables: Iterable[str] = PUBLIC_TABLES,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Agent:
    """Wire the agent for one database. Without a backend, only ``run_sql`` works."""
    allowed = tuple(allowed_tables)
    executor = Executor(db_path, max_rows=max_rows, timeout_s=timeout_s)
    hidden = [table for table in executor.table_names() if table not in allowed]
    validator = SQLValidator(allowed, denied_names=hidden)
    planner = Planner(backend, executor.describe(allowed)) if backend is not None else None
    return Agent(validator, executor, planner)
