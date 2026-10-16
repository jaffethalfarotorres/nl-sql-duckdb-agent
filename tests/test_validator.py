import random

import duckdb
import pytest

from nl_sql_agent.dataset import PUBLIC_TABLES
from nl_sql_agent.envelopes import Layer
from nl_sql_agent.validator import SQLValidator, TokenKind, tokenize


@pytest.fixture(scope="module")
def validator():
    v = SQLValidator(PUBLIC_TABLES)
    yield v
    v.close()


def rules(violations) -> set[str]:
    return {v.rule for v in violations}


# --- tokenizer -----------------------------------------------------------------------


def kinds(sql):
    return [(t.kind, t.text) for t in tokenize(sql)]


def test_tokenizer_skips_comments_and_whitespace():
    assert kinds("SELECT 1 -- note ; DROP\n/* ; DELETE */ ;") == [
        (TokenKind.WORD, "SELECT"),
        (TokenKind.NUMBER, "1"),
        (TokenKind.SEMICOLON, ";"),
    ]


def test_tokenizer_reads_doubled_quotes_inside_strings_and_identifiers():
    assert kinds("SELECT 'O''Brien; DROP' AS \"a\"\"b\"") == [
        (TokenKind.WORD, "SELECT"),
        (TokenKind.STRING, "O'Brien; DROP"),
        (TokenKind.WORD, "AS"),
        (TokenKind.QUOTED, 'a"b'),
    ]


def test_tokenizer_treats_backslash_as_a_plain_character_like_duckdb():
    # DuckDB standard strings have no backslash escapes: 'C:\' is a complete literal.
    assert kinds("SELECT 'C:\\' , 1")[1] == (TokenKind.STRING, "C:\\")


def test_line_comment_ends_at_newline():
    assert TokenKind.SEMICOLON in [t.kind for t in tokenize("SELECT 1 -- c\n; DROP TABLE x")]


# --- lexer layer ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "rule"),
    [
        ("", "empty"),
        ("  -- only a comment\n", "empty"),
        ("SELECT 1; DROP TABLE orders", "multiple_statements"),
        ("SELECT 1;DROP TABLE orders", "multiple_statements"),
        ("SELECT 1; SELECT * FROM payroll", "multiple_statements"),
        ("SELECT 1 -- c\n; DELETE FROM orders", "multiple_statements"),
        ("DROP TABLE orders", "not_a_select"),
        ("FROM orders", "not_a_select"),
        ("EXPLAIN ANALYZE SELECT 1", "not_a_select"),
        ("PRAGMA database_list", "not_a_select"),
        ("WITH x AS (SELECT 1) DELETE FROM orders", "denied_keyword"),
        ("SELECT * FROM orders WHERE 1 = 1 AND DROP", "denied_keyword"),
        ("COPY (SELECT * FROM orders) TO 'leak.csv'", "denied_keyword"),
        ("SELECT * FROM read_csv('/etc/passwd')", "denied_function"),
        ("SELECT * FROM READ_TEXT ('C:/Windows/win.ini')", "denied_function"),
        ("SELECT * FROM \"read_csv\"('x')", "denied_function"),
        ("SELECT * FROM read_csv /* gap */ ('x')", "denied_function"),
        ("SELECT * FROM query('SELECT * FROM payroll')", "denied_function"),
        ("SELECT * FROM glob('*')", "denied_function"),
        ("SELECT * FROM duckdb_tables()", "denied_function"),
        ("SELECT * FROM (SUMMARIZE payroll)", "denied_keyword"),
        ("SELECT * FROM (DESCRIBE payroll)", "denied_keyword"),
        ("SELECT * FROM (SHOW TABLES)", "denied_keyword"),
        ("SELECT 'unterminated", "unterminated_literal"),
        ('SELECT "unterminated', "unterminated_literal"),
        ("SELECT 1 /* never closed", "unterminated_literal"),
        ("SELECT 1 /* /* */ ; DROP TABLE orders; */", "nested_comment"),
        ("SELECT E'\\''; DROP TABLE orders; --'", "exotic_literal"),
        ("SELECT $$;$$; DROP TABLE orders", "exotic_literal"),
        ("SELECT $1", "exotic_literal"),
        ("SELECT 1\x00; DROP TABLE orders", "control_character"),
        ("SELECT 1 \u200b", "control_character"),
    ],
)
def test_lexer_rejects(validator, sql, rule):
    violations = validator.lexical_check(sql)
    assert rule in rules(violations)
    assert all(v.layer is Layer.LEXER for v in violations)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM pg_catalog.pg_class",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM duckdb_tables",
        "SELECT * FROM pg_tables",
    ],
)
def test_lexer_refuses_system_catalog_names(validator, sql):
    assert "denied_identifier" in rules(validator.lexical_check(sql))


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM payroll",
        'SELECT * FROM "PAYROLL"',
        "SELECT * FROM main.payroll",
        "WITH payroll AS (SELECT 1 AS x) SELECT * FROM payroll",
    ],
)
def test_lexer_refuses_identifiers_that_name_a_known_hidden_table(sql):
    v = SQLValidator(PUBLIC_TABLES, denied_names=["Payroll"])
    try:
        assert "denied_identifier" in rules(v.lexical_check(sql))
    finally:
        v.close()


def test_a_hidden_table_name_inside_a_string_is_just_data():
    v = SQLValidator(PUBLIC_TABLES, denied_names=["payroll"])
    try:
        assert v.validate("SELECT 'payroll' AS word").ok
    finally:
        v.close()


def test_lexer_reports_each_denied_keyword_once(validator):
    violations = validator.lexical_check("SELECT 1 FROM t WHERE drop AND DROP AND insert")
    assert sorted(v.detail for v in violations if v.rule == "denied_keyword") == [
        "DROP",
        "INSERT",
    ]


def test_oversized_sql_is_rejected_before_tokenizing():
    v = SQLValidator(PUBLIC_TABLES, max_chars=20)
    try:
        assert v.validate("SELECT " + "1 + " * 10 + "1").rules == {"oversized"}
    finally:
        v.close()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 'pending; DROP TABLE orders' AS note",
        "SELECT count(*) FROM orders -- ; DROP TABLE orders",
        "SELECT count(*) FROM orders /* DELETE FROM orders */",
        'SELECT status AS "drop" FROM orders',
        "SELECT 'O''Brien; DELETE' AS s",
        "SELECT replace(company, 'Labs', 'Lab') FROM customers",
        "SELECT count(*) FROM orders;",
        "(SELECT 1) UNION (SELECT 2)",
        "with x as (select 1) select * from x",
        "SELECT EXTRACT(YEAR FROM order_date) AS y, count(*) FROM orders GROUP BY y",
        "SELECT 'C:\\' AS path",
    ],
)
def test_lexer_accepts_tricky_but_benign_sql(validator, sql):
    assert validator.lexical_check(sql) == []


# --- engine layer, on its own --------------------------------------------------------
# Each case is something the lexer would also catch; calling engine_check directly
# proves the engine layer stops it by itself.


@pytest.mark.parametrize(
    ("sql", "rule"),
    [
        ("SELECT 1; DROP TABLE orders", "multiple_statements"),
        ("SELECT 1; SELECT * FROM orders", "multiple_statements"),
        ("SELECT E'\\''; DROP TABLE orders; --'", "multiple_statements"),
        ("DROP TABLE orders", "not_a_select"),
        ("CREATE TEMP TABLE t AS SELECT 1", "not_a_select"),
        ("PRAGMA database_list", "not_a_select"),
        ("ATTACH 'other.duckdb' AS other", "not_a_select"),
        ("SELEC 1", "parse_error"),
        ("WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x", "parse_error"),
        ("SELECT * FROM read_csv('/etc/passwd')", "table_function"),
        ("SELECT * FROM query('SELECT * FROM payroll')", "table_function"),
        ("SELECT * FROM range(10)", "table_function"),
        ("SELECT * FROM payroll", "table_not_allowed"),
        ("SELECT * FROM main.payroll", "table_not_allowed"),
        ('SELECT * FROM "PAYROLL"', "table_not_allowed"),
        ("SELECT * FROM shop.main.orders", "table_not_allowed"),
        ("SELECT * FROM information_schema.tables", "table_not_allowed"),
        ("SELECT * FROM pg_catalog.pg_class", "table_not_allowed"),
        ("SELECT * FROM sqlite_master", "table_not_allowed"),
        ("SELECT * FROM duckdb_tables", "table_not_allowed"),
        (
            "SELECT * FROM orders WHERE customer_id IN (SELECT employee_id FROM payroll)",
            "table_not_allowed",
        ),
        ("SELECT (SELECT max(annual_salary) FROM payroll) AS top FROM orders", "table_not_allowed"),
        ("SELECT order_id FROM orders UNION SELECT employee_id FROM payroll", "table_not_allowed"),
        ("SELECT * FROM (SUMMARIZE payroll)", "unsupported_ref"),
        ("SELECT * FROM (DESCRIBE payroll)", "unsupported_ref"),
        ("SELECT * FROM (SHOW TABLES)", "unsupported_ref"),
        ("SELECT * FROM (FROM payroll)", "table_not_allowed"),
        (
            "SELECT * FROM payroll PIVOT (sum(annual_salary) FOR team IN ('sales'))",
            "table_not_allowed",
        ),
    ],
)
def test_engine_layer_rejects_on_its_own(validator, sql, rule):
    violations, _ = validator.engine_check(sql)
    assert rule in rules(violations)
    assert all(v.layer is Layer.ENGINE for v in violations)


@pytest.mark.parametrize(
    "sql",
    [
        # A CTE defined inside a subquery does not hide the real table outside it.
        "SELECT * FROM (WITH payroll AS (SELECT 1 AS x) SELECT * FROM payroll) t, payroll",
        # A CTE cannot see a sibling defined after it: this payroll is the real table.
        "WITH a AS (SELECT * FROM payroll), payroll AS (SELECT 1 AS x) SELECT * FROM a",
        # Nor can a non-recursive CTE see itself.
        "WITH payroll AS (SELECT * FROM payroll) SELECT * FROM payroll",
    ],
)
def test_cte_scoping_cannot_smuggle_a_hidden_table(validator, sql):
    violations, _ = validator.engine_check(sql)
    assert "table_not_allowed" in rules(violations)


@pytest.mark.parametrize(
    "sql",
    [
        "WITH payroll AS (SELECT 1 AS x) SELECT * FROM payroll",
        "WITH a AS (SELECT * FROM orders), b AS (SELECT * FROM a) SELECT * FROM b",
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) "
        "SELECT * FROM r",
        "SELECT * FROM (WITH o AS (SELECT * FROM orders) SELECT * FROM o) t",
        "SELECT * FROM orders USING SAMPLE 10%",
        "SELECT * FROM (VALUES (1), (2)) v(x)",
        "SELECT * FROM orders PIVOT (count(*) FOR channel IN ('web', 'partner'))",
    ],
)
def test_engine_layer_accepts_allowed_reads(validator, sql):
    assert validator.engine_check(sql)[0] == []


def test_engine_layer_reports_the_tables_it_resolved(validator):
    _, tables = validator.engine_check(
        "WITH t AS (SELECT * FROM orders) "
        "SELECT * FROM t JOIN main.customers c USING (customer_id) "
        "JOIN ORDER_ITEMS USING (order_id)"
    )
    assert tables == ("main.customers", "order_items", "orders")


def test_without_an_allowlist_any_table_may_be_read_but_table_functions_may_not():
    v = SQLValidator(allowed_tables=None)
    try:
        assert v.engine_check("SELECT * FROM payroll")[0] == []
        assert "table_function" in rules(v.engine_check("SELECT * FROM range(3)")[0])
    finally:
        v.close()


# --- both layers ---------------------------------------------------------------------


def test_validate_stops_at_the_lexer_so_the_engine_never_parses_rejected_input(validator):
    report = validator.validate("SELECT 1; DROP TABLE orders")
    assert not report.ok
    assert {v.layer for v in report.violations} == {Layer.LEXER}


def test_validate_passes_a_realistic_analytics_query(validator):
    sql = """
        WITH revenue AS (
            SELECT p.category, sum(i.quantity * i.unit_price) AS total
            FROM order_items i
            JOIN orders o USING (order_id)
            JOIN products p USING (product_id)
            WHERE o.status = 'paid' AND EXTRACT(MONTH FROM o.order_date) <= 6
            GROUP BY p.category
        )
        SELECT category, round(total, 2) AS total,
               rank() OVER (ORDER BY total DESC) AS position
        FROM revenue ORDER BY position;
    """
    report = validator.validate(sql)
    assert report.ok, report.violations
    assert report.tables == ("order_items", "orders", "products")
    assert report.sql == sql


def test_validate_runs_the_engine_layer_when_the_lexer_is_satisfied(validator):
    report = validator.validate("SELECT * FROM payroll")
    assert report.rules == {"table_not_allowed"}
    assert report.violations[0].layer is Layer.ENGINE


# --- differential fuzzing against DuckDB's own parser --------------------------------

HEADS = [
    "SELECT 1", "SELECT 'a'", "SELECT order_id FROM orders",
    "WITH a AS (SELECT 1) SELECT * FROM a", "DROP TABLE orders", "",
]  # fmt: skip
NOISE = [
    "", " ", "\n", "-- c", "-- c\n", "/* c */", "/* ; */", "/* /* */ */", "/* /* */", "'",
    "';'", "''", "E'\\'", "E'\\';'", "e'", "\\'", "$$", "$$;$$", "$t$;$t$", '"', '";"',
    "--'\n", "/*'*/",
]  # fmt: skip
SEPARATORS = ["", ";", "; ", ";\n", ";;"]
TAILS = ["", "DROP TABLE orders", "SELECT 2", "DELETE FROM orders", "PRAGMA version", "x"]


def fuzz_case(rng: random.Random) -> str:
    """head + noise + separator + noise + tail: the noise is where literal and comment
    boundaries decide whether the separator and the tail are code or data."""
    parts = (HEADS, NOISE, NOISE, SEPARATORS, NOISE, TAILS, NOISE)
    return "".join(rng.choice(options) for options in parts)


def test_lexer_never_accepts_what_duckdb_parses_as_several_statements():
    """Seeded differential fuzz against DuckDB's parser: whenever DuckDB sees more than
    one statement (or a non-SELECT), the lexer must have rejected the input on its own."""
    rng = random.Random(2026)
    parser = duckdb.connect(":memory:")
    v = SQLValidator(PUBLIC_TABLES)
    disagreements, dangerous = [], 0
    try:
        for _ in range(20_000):
            sql = fuzz_case(rng)
            try:
                statements = parser.extract_statements(sql)
            except duckdb.Error:
                continue
            if len(statements) == 1 and statements[0].type == duckdb.StatementType.SELECT:
                continue
            dangerous += 1
            if not v.lexical_check(sql):
                disagreements.append(sql)
    finally:
        v.close()
        parser.close()
    assert dangerous > 300, "the fuzzer must keep exercising the dangerous region"
    assert disagreements == []


def test_engine_layer_reports_a_statement_with_only_comments_as_empty(validator):
    assert rules(validator.engine_check("-- nothing here")[0]) == {"empty"}
