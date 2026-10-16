"""Every attack class against every layer, each layer tested in isolation.

The matrix below is the source of truth for the README table. It is asserted,
not just printed: if a layer gets weaker or stronger, this test says so.
"""

import hashlib

import duckdb
import pytest

from nl_sql_agent.dataset import PUBLIC_TABLES
from nl_sql_agent.envelopes import ValidationReport
from nl_sql_agent.executor import Executor, QueryError, open_read_only
from nl_sql_agent.validator import SQLValidator

# attack, sql, then whether each layer stops it on its own:
# lexer, engine (DuckDB parser + AST allowlist), executor gate, read-only connection
MATRIX = [
    ("stacked write", "SELECT 1; DROP TABLE orders", 1, 1, 1, 1),
    ("stacked read", "SELECT 1; SELECT * FROM payroll", 1, 1, 1, 0),
    ("DDL", "DROP TABLE orders", 1, 1, 1, 1),
    ("DML", "UPDATE orders SET status = 'paid'", 1, 1, 1, 1),
    ("temp table", "CREATE TEMP TABLE t AS SELECT 1", 1, 1, 1, 0),
    ("literal smuggling", "SELECT E'\\''; DROP TABLE orders; --'", 1, 1, 1, 1),
    ("file read", "SELECT * FROM read_text('pyproject.toml')", 1, 1, 0, 1),
    ("file write", "COPY orders TO 'leak.csv'", 1, 1, 1, 1),
    ("extension", "INSTALL httpfs", 1, 1, 1, 1),
    ("settings", "SET enable_external_access = true", 1, 1, 1, 1),
    ("hidden table", "SELECT * FROM payroll", 1, 1, 0, 0),
    ("hidden table via query()", "SELECT * FROM query('SELECT * FROM payroll')", 1, 1, 0, 0),
    (
        "hidden table via CTE scope",
        "SELECT * FROM (WITH payroll AS (SELECT 1) SELECT * FROM payroll) t, payroll",
        1, 1, 0, 0,
    ),
    ("catalog discovery", "SELECT * FROM information_schema.tables", 1, 1, 0, 0),
    ("metadata subquery", "SELECT * FROM (SUMMARIZE payroll)", 1, 1, 0, 0),
    ("PRAGMA", "PRAGMA database_list", 1, 1, 0, 0),
]  # fmt: skip


@pytest.fixture(scope="module")
def layers(db_path):
    validator = SQLValidator(PUBLIC_TABLES, denied_names=["payroll"])
    executor = Executor(db_path)

    def lexer(sql: str) -> bool:
        return bool(validator.lexical_check(sql))

    def engine(sql: str) -> bool:
        return bool(validator.engine_check(sql)[0])

    def executor_gate(sql: str) -> bool:
        try:
            executor.run(ValidationReport(sql=sql))  # pretend the validator was bypassed
        except QueryError as exc:
            return exc.kind == "refused"
        return False

    def connection(sql: str) -> bool:
        cursor = executor._con.cursor()  # same locked connection, raw string, no gate
        try:
            cursor.execute(sql).fetchall()
        except duckdb.Error:
            return True
        finally:
            cursor.close()
        return False

    yield (lexer, engine, executor_gate, connection)
    validator.close()
    executor.close()


@pytest.mark.parametrize(
    ("attack", "sql", "expected"),
    [(row[0], row[1], row[2:]) for row in MATRIX],
    ids=[row[0] for row in MATRIX],
)
def test_each_layer_stops_exactly_what_the_matrix_says(layers, attack, sql, expected):
    observed = tuple(int(layer(sql)) for layer in layers)
    assert observed == expected, f"{attack}: lexer, engine, gate, connection = {observed}"


def test_every_attack_class_is_stopped_by_at_least_two_independent_layers():
    assert all(sum(row[2:]) >= 2 for row in MATRIX)


def test_the_database_file_is_unchanged_after_the_whole_matrix(db_path, layers):
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    for _, sql, *_ in MATRIX:
        for layer in layers:
            layer(sql)
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before


def test_the_raw_connection_alone_refuses_writes(db_path):
    with open_read_only(db_path) as con:
        with pytest.raises(duckdb.Error, match="read-only"):
            con.execute("DELETE FROM orders")
        assert con.execute("SELECT count(*) FROM orders").fetchone() == (400,)
