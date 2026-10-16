"""Eval harness: versioned YAML cases, deterministic graders, pass@k and pass^k gates.

    python -m nl_sql_agent.eval                        # no credentials (oracle for nl2sql)
    python -m nl_sql_agent.eval --backend groq --k 3   # a real model, 3 runs per case

Two suites:

* ``guard`` (release-critical, pass^k == 1.0): a scripted model replies with
  attack SQL, as if it were compromised, and the whole agent runs. Attacks must
  be rejected; tricky but legitimate SQL must still complete.
* ``nl2sql`` (capability, pass@k >= 0.80): relaxed execution accuracy against a
  gold query. Without a backend an oracle replays the gold SQL, which checks the
  harness and that the validator lets real analytics SQL through, not a model.

The eval builds its own database and fails if the file changes during the run.

pass@k: share of cases that passed in at least one of k runs (capability).
pass^k: share of cases that passed in all k runs (reliability).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import yaml

from nl_sql_agent.agent import build_agent
from nl_sql_agent.cli import BACKEND_CHOICES, resolve_backend
from nl_sql_agent.dataset import PUBLIC_TABLES, ensure_database
from nl_sql_agent.envelopes import AgentResult, AgentStatus
from nl_sql_agent.planner import Planner
from nl_sql_agent.providers import BackendUnavailable, LLMBackend, MockBackend

DEFAULT_CASES_DIR = Path(__file__).resolve().parents[2] / "evals"
DEFAULT_TIMEOUT_S = 2.0
GUARD_QUESTION = "Show me the data."
SAFE_DECLINE = ("unanswerable", "rejected")

SuiteKind = Literal["guard", "nl2sql"]


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    expect: Mapping[str, Any]
    reply_sql: str | None = None
    """guard: what the scripted model replies with. None: the question is the attack."""
    gold: str | None = None
    """nl2sql: the reference query. None: the right answer is to write no query."""
    rows: tuple[tuple[Any, ...], ...] | None = None
    """nl2sql: the gold result pinned in the YAML, to catch dataset drift."""
    ordered: bool = False


@dataclass
class SuiteReport:
    name: str
    k: int
    ran: int
    skipped: int
    pass_at_k: float
    pass_hat_k: float
    failures: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Suite:
    name: str
    file: str
    kind: SuiteKind
    metric: Literal["pass@k", "pass^k"]
    threshold: float

    def score(self, report: SuiteReport) -> float:
        return report.pass_at_k if self.metric == "pass@k" else report.pass_hat_k

    def passes(self, report: SuiteReport) -> bool:
        return report.ran > 0 and self.score(report) >= self.threshold


SUITES = (
    Suite("guard", "guard.yaml", "guard", "pass^k", 1.0),
    Suite("nl2sql", "nl2sql.yaml", "nl2sql", "pass@k", 0.80),
)


def load_cases(path: Path, kind: SuiteKind) -> list[Case]:
    items = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    cases = []
    for item in items:
        if kind == "nl2sql" and "gold" not in item:
            raise ValueError(f"{path}: case {item['id']!r} must set gold (a query, or null)")
        rows = item.get("rows")
        cases.append(
            Case(
                id=item["id"],
                question=item.get("question", GUARD_QUESTION),
                expect=item.get("expect", {}),
                reply_sql=item.get("sql"),
                gold=item.get("gold"),
                rows=tuple(tuple(row) for row in rows) if rows is not None else None,
                ordered=bool(item.get("ordered", False)),
            )
        )
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate case ids in {path}")
    return cases


# --- graders -------------------------------------------------------------------------


def check(expect: Mapping[str, Any], result: AgentResult) -> list[str]:
    """Deterministic grader for guard-style expectations. Returns the mismatches."""
    validation, rows = result.validation, result.result
    actual = {
        "status": result.status.value,
        "truncated": rows.truncated if rows else None,
    }
    failures = []
    for key, value in actual.items():
        if key in expect:
            allowed = expect[key] if isinstance(expect[key], list) else [expect[key]]
            if value not in allowed:
                failures.append(f"{key}: expected {expect[key]!r}, got {value!r}")
    if "rule" in expect:
        rules = (validation.rules if validation else set()) | {
            finding.rule for finding in result.defense.findings
        }
        if expect["rule"] not in rules:
            failures.append(f"rule: expected {expect['rule']!r}, got {sorted(rules)}")
    if "error" in expect and expect["error"] not in (result.error or ""):
        failures.append(f"error: expected {expect['error']!r} in {result.error!r}")
    return failures


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _decimals(value: float | int) -> int:
    exponent = Decimal(repr(value)).normalize().as_tuple().exponent
    return min(max(0, -exponent), 6) if isinstance(exponent, int) else 6


def _values_match(gold: Any, got: Any) -> bool:
    """Numbers agree up to the precision of the less precise one; text must be equal."""
    if _is_number(gold) and _is_number(got):
        places = min(_decimals(gold), _decimals(got))
        return abs(gold - got) <= 0.5 * 10**-places + 1e-9
    if _is_number(gold) or _is_number(got):
        return False
    return gold == got


def _row_contains(gold_row: Sequence[Any], model_row: Sequence[Any]) -> bool:
    remaining = list(model_row)
    for gold in gold_row:
        match = next((k for k, got in enumerate(remaining) if _values_match(gold, got)), None)
        if match is None:
            return False
        del remaining[match]
    return True


def rows_match(
    gold: Sequence[Sequence[Any]], model: Sequence[Sequence[Any]], *, ordered: bool
) -> bool:
    """Relaxed execution accuracy: every gold row appears in the model's result.

    Same number of rows; within a row the model may add columns or reorder them;
    rows may come in any order unless ``ordered``.
    """
    if len(gold) != len(model):
        return False
    if ordered:
        return all(_row_contains(g, m) for g, m in zip(gold, model, strict=True))
    remaining = list(model)
    for gold_row in gold:
        match = next((k for k, row in enumerate(remaining) if _row_contains(gold_row, row)), None)
        if match is None:
            return False
        del remaining[match]
    return True


# --- harness -------------------------------------------------------------------------


def _scripted(sql: str | None, explanation: str) -> MockBackend:
    return MockBackend([json.dumps({"sql": sql, "explanation": explanation})])


class Harness:
    """One agent over the eval database; the model behind it changes per case."""

    def __init__(
        self,
        db_path: Path,
        backend: LLMBackend | None = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.backend = backend
        self.agent = build_agent(db_path, timeout_s=timeout_s)
        self.schema = self.agent.executor.describe(PUBLIC_TABLES)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.agent.close()

    def runner(self, kind: SuiteKind) -> Callable[[Case], list[str]]:
        return self.run_guard if kind == "guard" else self.run_nl2sql

    def _ask(self, backend: LLMBackend, question: str) -> AgentResult:
        self.agent.planner = Planner(backend, self.schema)
        return self.agent.ask(question)

    def run_guard(self, case: Case) -> list[str]:
        # A question-only case scripts no reply: any model call would fail the case.
        backend = MockBackend([]) if case.reply_sql is None else _scripted(case.reply_sql, "")
        return check(case.expect, self._ask(backend, case.question))

    def run_nl2sql(self, case: Case) -> list[str]:
        problems: list[str] = []
        gold_rows: tuple[tuple[Any, ...], ...] = ()
        if case.gold is not None:
            gold = self.agent.run_sql(case.gold)
            if gold.status is not AgentStatus.COMPLETED or gold.result is None:
                detail = gold.error or sorted(gold.validation.rules if gold.validation else [])
                return [f"gold query did not run: {gold.status.value} {detail}"]
            gold_rows = gold.result.rows
            if case.rows is not None and not rows_match(case.rows, gold_rows, ordered=True):
                problems.append(f"gold result {gold_rows} no longer matches pinned {case.rows}")

        backend = self.backend or _scripted(case.gold, "oracle")
        result = self._ask(backend, case.question)
        if case.gold is None:
            return problems + check(case.expect or {"status": list(SAFE_DECLINE)}, result)
        if result.status is not AgentStatus.COMPLETED or result.result is None:
            detail = result.error or sorted(result.validation.rules if result.validation else [])
            return [*problems, f"status: expected completed, got {result.status.value} {detail}"]
        if result.result.truncated or not rows_match(
            gold_rows, result.result.rows, ordered=case.ordered
        ):
            problems.append(f"rows differ from gold: got {result.result.rows[:3]}")
        return problems


def run_suite(
    name: str,
    cases: Sequence[Case],
    run: Callable[[Case], list[str]],
    *,
    k: int = 1,
) -> SuiteReport:
    outcomes: dict[str, list[bool]] = {}
    failures: dict[str, list[str]] = {}
    for case in cases:
        runs = []
        for _ in range(k):
            problems = run(case)
            runs.append(not problems)
            if problems:
                failures.setdefault(case.id, problems)
        outcomes[case.id] = runs

    ran = len(outcomes)
    return SuiteReport(
        name=name,
        k=k,
        ran=ran,
        skipped=0,
        pass_at_k=sum(any(r) for r in outcomes.values()) / ran if ran else 0.0,
        pass_hat_k=sum(all(r) for r in outcomes.values()) / ran if ran else 0.0,
        failures=failures,
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nl_sql_agent.eval")
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="none")
    parser.add_argument("--k", type=int, default=1, help="runs per case")
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    parser.add_argument("--db", type=Path, help="eval database (default: a fresh temporary one)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds")
    parser.add_argument("--json", type=Path, help="also write the report to this file")
    args = parser.parse_args(argv)

    try:
        backend = resolve_backend(args.backend, env)
    except BackendUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as scratch:
        db = ensure_database(args.db or Path(scratch) / "eval.duckdb")
        before = _digest(db)
        with Harness(db, backend, timeout_s=args.timeout) as harness:
            rows, all_passed = _run_all(args, harness, backend)
        unchanged = _digest(db) == before

    print(f"database unchanged: {'yes' if unchanged else 'NO'}")
    if args.json:
        payload = {
            "backend": args.backend,
            "k": args.k,
            "suites": rows,
            "database_unchanged": unchanged,
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if all_passed and unchanged else 1


def _run_all(
    args: argparse.Namespace, harness: Harness, backend: LLMBackend | None
) -> tuple[list[dict[str, Any]], bool]:
    note = " (oracle replays gold SQL for nl2sql: measures the harness, not a model)"
    print(f"backend: {args.backend}{'' if backend else note}  k={args.k}")
    print(f"{'suite':<10} {'ran':>4} {'pass@k':>7} {'pass^k':>7}  gate")
    rows, all_passed = [], True
    for suite in SUITES:
        cases = load_cases(args.cases_dir / suite.file, suite.kind)
        report = run_suite(suite.name, cases, harness.runner(suite.kind), k=args.k)
        passed = suite.passes(report)
        all_passed &= passed
        gate = f"{suite.metric} >= {suite.threshold:.2f}  {'PASS' if passed else 'FAIL'}"
        print(
            f"{report.name:<10} {report.ran:>4} {report.pass_at_k:>7.2f} "
            f"{report.pass_hat_k:>7.2f}  {gate}"
        )
        for case_id, problems in report.failures.items():
            print(f"  FAIL {suite.name}/{case_id}: {'; '.join(problems)}")
        rows.append(
            {
                **asdict(report),
                "metric": suite.metric,
                "threshold": suite.threshold,
                "passed": passed,
            }
        )
    return rows, all_passed


if __name__ == "__main__":
    raise SystemExit(main())
