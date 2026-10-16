"""Planner: asks the model for one SQL query and reads its reply defensively.

The model's only job is to propose SQL. It never executes anything, and its
reply is read with the same discipline as the Phase 1 router: the JSON is
validated with Pydantic, deterministic format slips (markdown fences,
surrounding prose, a ```sql fence inside the value) are repaired, and anything
else is diagnosed and raised as ``PlanError``. The planner never re-sends the
same prompt hoping for a better answer.

The only second call the system makes is the agent's single *informed* repair
after an engine error, and that prompt carries the failed query and the error.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ValidationError

from nl_sql_agent.defense import build_system_prompt, wrap_untrusted
from nl_sql_agent.envelopes import ModelTier, SQLProposal
from nl_sql_agent.providers.base import BackendError, LLMBackend

PLANNER_TASK = """\
Write one DuckDB SQL query that answers the user's question, using only this schema:

{schema}

Rules:
- Exactly one read-only SELECT statement (a WITH clause is fine). Never change data.
- Use only the tables and columns above, and follow the business rules in the comments.
- If the schema cannot answer the question, or the question asks to change data,
  set "sql" to null and say why in the explanation.
Reply with only a JSON object:
{{"sql": "<query or null>", "explanation": "<max 25 words>"}}"""

REPAIR_INSTRUCTION = (
    "The previous query failed in the database. "
    "Reply with a corrected query in the same JSON format."
)

_SQL_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?(?P<body>.*?)\n?\s*```$", re.DOTALL)


class Diagnosis(StrEnum):
    """How a model reply was read, or why it could not be used."""

    OK = "clean_json"
    MARKDOWN_FENCED = "markdown_fenced"
    PROSE_WRAPPED = "prose_wrapped"
    NOT_JSON = "not_json"
    SCHEMA_VIOLATION = "schema_violation"
    BACKEND_ERROR = "backend_error"


class PlanError(RuntimeError):
    """The model gave no usable proposal."""

    def __init__(self, diagnosis: Diagnosis, detail: str = "") -> None:
        super().__init__(f"{diagnosis}: {detail}".rstrip(": "))
        self.diagnosis = diagnosis


@dataclass(frozen=True)
class RepairContext:
    """What failed last time, shown to the model as untrusted data."""

    sql: str
    error: str


class _Reply(BaseModel):
    sql: str | None
    explanation: str = ""


def _extract_json(raw: str) -> tuple[Diagnosis, object]:
    """Find the JSON payload in a reply and say how much repair it needed."""
    stripped = raw.strip()
    try:
        return Diagnosis.OK, json.loads(stripped)
    except json.JSONDecodeError:
        pass
    repaired = Diagnosis.MARKDOWN_FENCED if "```" in stripped else Diagnosis.PROSE_WRAPPED
    decoder = json.JSONDecoder()
    for start in (match.start() for match in re.finditer(r"\{", stripped)):
        try:
            payload, _ = decoder.raw_decode(stripped, start)
        except json.JSONDecodeError:
            continue
        return repaired, payload
    return Diagnosis.NOT_JSON, None


def _clean_sql(sql: str | None) -> str | None:
    if sql is None or not sql.strip():
        return None
    sql = sql.strip()
    if fenced := _SQL_FENCE.match(sql):
        sql = fenced.group("body").strip()
    return sql or None


class Planner:
    def __init__(self, backend: LLMBackend, schema: str) -> None:
        self.backend = backend
        self.system_prompt = build_system_prompt(PLANNER_TASK.format(schema=schema))

    def propose(
        self,
        question: str,
        *,
        tier: ModelTier = "small",
        repair: RepairContext | None = None,
    ) -> SQLProposal:
        prompt = wrap_untrusted(question)
        if repair is not None:
            failed = f"previous query:\n{repair.sql}\n\ndatabase error:\n{repair.error}"
            prompt = f"{prompt}\n\n{REPAIR_INSTRUCTION}\n{wrap_untrusted(failed)}"
        try:
            raw = self.backend.complete(prompt, system=self.system_prompt, tier=tier)
        except BackendError as exc:
            raise PlanError(Diagnosis.BACKEND_ERROR, str(exc)) from exc

        diagnosis, payload = _extract_json(raw)
        if diagnosis is Diagnosis.NOT_JSON:
            raise PlanError(diagnosis, raw[:80])
        try:
            reply = _Reply.model_validate(payload)
        except ValidationError as exc:
            raise PlanError(Diagnosis.SCHEMA_VIOLATION, str(exc).splitlines()[0]) from exc
        return SQLProposal(
            sql=_clean_sql(reply.sql),
            explanation=reply.explanation.strip(),
            parse=diagnosis,
            model_tier=tier,
        )
