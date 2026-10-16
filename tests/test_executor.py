import hashlib
import time

import pytest

from nl_sql_agent.dataset import PUBLIC_TABLES
from nl_sql_agent.envelopes import Layer, ValidationReport, Violation
from nl_sql_agent.executor import Executor, QueryError


@pytest.fixture(scope="module")
def executor(db_path):
    with Executor(db_path, max_rows=50, timeout_s=5.0) as ex:
        yield ex


def unchecked(sql: str) -> ValidationReport:
    """A report that claims to pass without the validator ever seeing the SQL.

    Used only to prove the executor's own defenses hold if the validator were
    bypassed or buggy.
    """
    return ValidationReport(sql=sql)


def file_digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runs_a_validated_select_and_returns_json_friendly_values(executor):
    result = executor.run(
        unchecked(
            "SELECT name, unit_price, DATE '2025-03-01' AS d FROM products WHERE product_id = 1"
        )
    )
    assert result.columns == ("name", "unit_price", "d")
    assert result.rows == (("Standing Desk", 549.0, "2025-03-01"),)
    assert not result.truncated
    assert result.elapsed_ms >= 0


def test_refuses_sql_that_failed_validation(executor):
    failed = ValidationReport(
        sql="SELECT * FROM payroll",
        violations=(Violation(layer=Layer.ENGINE, rule="table_not_allowed", detail="payroll"),),
    )
    with pytest.raises(ValueError, match="validation"):
        executor.run(failed)


def test_caps_rows_and_says_so(executor):
    result = executor.run(unchecked("SELECT order_id FROM orders ORDER BY order_id"))
    assert result.row_count == 50
    assert result.truncated
    assert result.rows[0] == (1,)


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("SELECT 1 AS a; SELECT 2 AS b", "exactly one"),
        ("SELECT 1; DROP TABLE orders", "exactly one"),
        ("CREATE TEMP TABLE t AS SELECT 1", "SELECT"),
        ("DROP TABLE orders", "SELECT"),
    ],
)
def test_runs_only_one_parsed_select_statement(executor, sql, message):
    with pytest.raises(QueryError, match=message) as info:
        executor.run(unchecked(sql))
    assert not info.value.fixable


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_text('pyproject.toml')",
        "SELECT * FROM read_csv('https://example.com/data.csv')",
        "SELECT * FROM glob('*')",
    ],
)
def test_connection_has_no_file_or_network_access(executor, sql):
    with pytest.raises(QueryError, match="disabled") as info:
        executor.run(unchecked(sql))
    assert not info.value.fixable


def test_engine_errors_the_model_can_fix_are_marked_fixable_and_kept_short(executor):
    with pytest.raises(QueryError) as info:
        executor.run(unchecked("SELECT revenue FROM orders"))
    assert info.value.fixable
    assert info.value.kind == "engine"
    assert "revenue" in str(info.value)
    assert "\n" not in str(info.value)
    assert len(str(info.value)) <= 300


def test_long_queries_are_interrupted_and_the_executor_recovers(db_path):
    heavy = (
        "SELECT sum(a.order_id * b.order_id * c.order_id) "
        "FROM orders a, orders b, orders c, orders d"
    )
    with Executor(db_path, timeout_s=0.3) as ex:
        start = time.perf_counter()
        with pytest.raises(QueryError, match="time limit") as info:
            ex.run(unchecked(heavy))
        assert info.value.kind == "timeout"
        assert time.perf_counter() - start < 5
        assert ex.run(unchecked("SELECT count(*) FROM orders")).rows == ((400,),)


def test_describe_documents_the_requested_tables_from_the_catalog(executor):
    schema = executor.describe(PUBLIC_TABLES)
    assert "TABLE orders -- One row per order placed in 2025." in schema
    assert "  status VARCHAR -- One of 'paid', 'refunded', 'pending'." in schema
    assert "  unit_price DECIMAL(10,2)" in schema
    assert "payroll" not in schema


def test_describe_rejects_unknown_tables(executor):
    with pytest.raises(ValueError, match="nope"):
        executor.describe(["orders", "nope"])


def test_the_database_file_is_byte_identical_after_attacks(db_path):
    before = file_digest(db_path)
    with Executor(db_path) as ex:
        for sql in [
            "DROP TABLE orders",
            "DELETE FROM orders",
            "UPDATE orders SET status = 'paid'",
            "INSERT INTO orders VALUES (999, 1, DATE '2025-01-01', 'paid', 'web')",
            "CREATE TABLE stolen AS SELECT * FROM payroll",
            "ALTER TABLE orders ADD COLUMN x INTEGER",
            "COPY orders TO 'leak.csv'",
            "ATTACH 'other.duckdb' AS other",
            "SET enable_external_access = true",
            "CHECKPOINT",
        ]:
            with pytest.raises(QueryError):
                ex.run(unchecked(sql))
    assert file_digest(db_path) == before
