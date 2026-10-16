"""Read-only executor: the last line of defense, independent of the validator.

Even if the validator were bypassed or buggy, the executor:

* refuses any ``ValidationReport`` that did not pass;
* parses the SQL again and executes exactly one parsed ``SELECT`` statement object,
  never the raw string, so a stacked second statement has nothing to run on;
* runs it on a connection opened with ``read_only=True``, no file or network
  access, no extension loading, and a locked configuration that a query cannot
  ``SET`` back open;
* stops queries that exceed a time limit and caps the rows it returns.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Iterator
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any, Self
from uuid import UUID

import duckdb

from nl_sql_agent.envelopes import QueryResult, ValidationReport

DEFAULT_MAX_ROWS = 200
DEFAULT_TIMEOUT_S = 10.0
_MESSAGE_CHARS = 300

READ_ONLY_CONFIG = {
    "enable_external_access": False,
    "autoinstall_known_extensions": False,
    "autoload_known_extensions": False,
    "lock_configuration": True,
}

# Errors a model can fix by rewriting its query; anything else is not worth a retry.
_FIXABLE = (
    duckdb.ParserException,
    duckdb.BinderException,
    duckdb.CatalogException,
    duckdb.ConversionException,
)


class QueryError(RuntimeError):
    """The engine refused or failed a validated query.

    ``kind`` is ``engine``, ``timeout`` or ``refused`` (the executor's own gate).
    ``fixable`` tells the agent whether one informed retry by the model could help.
    """

    def __init__(self, kind: str, message: str, *, fixable: bool = False) -> None:
        super().__init__(message)
        self.kind, self.fixable = kind, fixable


def open_read_only(path: Path) -> duckdb.DuckDBPyConnection:
    """The locked-down connection every query runs on."""
    return duckdb.connect(str(path), read_only=True, config=READ_ONLY_CONFIG)


def _short(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0][:_MESSAGE_CHARS] if text else type(exc).__name__


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime | date | clock_time):
        return value.isoformat()
    if isinstance(value, timedelta | UUID):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


class Executor:
    """Runs validated SQL on a read-only DuckDB file. Not thread-safe: one query at a time."""

    def __init__(
        self,
        db_path: Path,
        *,
        max_rows: int = DEFAULT_MAX_ROWS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self._con = open_read_only(Path(db_path))

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
        self._con.close()

    def run(self, report: ValidationReport) -> QueryResult:
        if not report.ok:
            raise ValueError("refusing to run SQL that failed validation")

        cursor = self._con.cursor()
        timer = threading.Timer(self.timeout_s, cursor.interrupt)
        try:
            try:
                statements = cursor.extract_statements(report.sql)
            except duckdb.Error as exc:
                raise QueryError("engine", _short(exc), fixable=True) from exc
            if len(statements) != 1:
                raise QueryError(
                    "refused", f"expected exactly one statement, got {len(statements)}"
                )
            if statements[0].type != duckdb.StatementType.SELECT:
                raise QueryError("refused", f"only SELECT runs here, got {statements[0].type.name}")

            start = time.perf_counter()
            timer.start()
            cursor.execute(statements[0])
            rows = cursor.fetchmany(self.max_rows + 1)
            elapsed_ms = (time.perf_counter() - start) * 1000
            columns = tuple(column[0] for column in cursor.description or ())
        except duckdb.InterruptException as exc:
            raise QueryError(
                "timeout", f"query exceeded the {self.timeout_s:g}s time limit"
            ) from exc
        except duckdb.Error as exc:
            raise QueryError("engine", _short(exc), fixable=isinstance(exc, _FIXABLE)) from exc
        finally:
            timer.cancel()
            cursor.close()

        return QueryResult(
            columns=columns,
            rows=tuple(tuple(_jsonable(value) for value in row) for row in rows[: self.max_rows]),
            truncated=len(rows) > self.max_rows,
            elapsed_ms=round(elapsed_ms, 3),
        )

    def table_names(self) -> tuple[str, ...]:
        """Every base table in the database, allowlisted or not."""
        rows = self._con.execute("SELECT table_name FROM duckdb_tables() ORDER BY 1").fetchall()
        return tuple(name for (name,) in rows)

    def describe(self, tables: Iterable[str]) -> str:
        """Schema text for the prompt, generated from the catalog and its comments."""
        wanted = list(tables)
        found = dict(
            self._con.execute(
                "SELECT table_name, comment FROM duckdb_tables() WHERE table_name = ANY(?)",
                [wanted],
            ).fetchall()
        )
        if missing := [table for table in wanted if table not in found]:
            raise ValueError(f"unknown tables: {', '.join(missing)}")
        columns = self._con.execute(
            "SELECT table_name, column_name, data_type, comment FROM duckdb_columns() "
            "WHERE table_name = ANY(?) ORDER BY table_name, column_index",
            [wanted],
        ).fetchall()
        return "\n".join(self._describe_lines(wanted, found, columns))

    @staticmethod
    def _describe_lines(
        tables: list[str], comments: dict[str, str | None], columns: list[tuple]
    ) -> Iterator[str]:
        for table in tables:
            yield f"TABLE {table} -- {comments[table] or ''}".rstrip(" -")
            for owner, name, data_type, comment in columns:
                if owner == table:
                    yield f"  {name} {data_type} -- {comment or ''}".rstrip(" -")
