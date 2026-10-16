import json
from itertools import cycle

import pytest

from nl_sql_agent.envelopes import (
    AgentResult,
    AgentStatus,
    DefenseReport,
    QueryResult,
    SQLProposal,
    ValidationReport,
)
from nl_sql_agent.eval import (
    DEFAULT_CASES_DIR,
    SUITES,
    Case,
    Harness,
    check,
    load_cases,
    main,
    rows_match,
    run_suite,
)
from nl_sql_agent.providers import MockBackend

CLEAN = DefenseReport(blocked=False)


def completed(rows) -> AgentResult:
    sql = "SELECT 1"
    columns = tuple(f"c{i}" for i in range(len(rows[0]))) if rows else ("c0",)
    return AgentResult(
        status=AgentStatus.COMPLETED,
        question="q",
        defense=CLEAN,
        proposal=SQLProposal(sql=sql, parse="clean_json"),
        validation=ValidationReport(sql=sql),
        result=QueryResult(columns=columns, rows=tuple(map(tuple, rows)), elapsed_ms=0),
    )


# --- relaxed execution accuracy ------------------------------------------------------


def test_rows_match_ignores_row_order_unless_asked():
    gold = [("web", 10), ("partner", 5)]
    assert rows_match(gold, [("partner", 5), ("web", 10)], ordered=False)
    assert not rows_match(gold, [("partner", 5), ("web", 10)], ordered=True)


def test_rows_match_allows_extra_columns_and_any_column_order():
    assert rows_match([("United States",)], [(12, "United States")], ordered=False)


def test_rows_match_tolerates_rounding_but_not_wrong_numbers():
    assert rows_match([(12.1212,)], [(12.12,)], ordered=False)
    assert rows_match([(415059.4,)], [(415059.40000001,)], ordered=False)
    assert not rows_match([(415059.4,)], [(422000.0,)], ordered=False)


def test_rows_match_needs_the_same_number_of_rows():
    assert not rows_match([("a",), ("b",)], [("a",)], ordered=False)
    assert not rows_match([("a",)], [("a",), ("b",)], ordered=False)


def test_rows_match_does_not_reuse_one_model_value_for_two_gold_values():
    assert not rows_match([(5, 5)], [(5, 7)], ordered=False)


def test_rows_match_compares_text_exactly_and_booleans_are_not_numbers():
    assert not rows_match([("Web",)], [("web",)], ordered=False)
    assert not rows_match([(1,)], [(True,)], ordered=False)


# --- expectations and suites ---------------------------------------------------------


def test_check_reports_every_mismatch(db_path):
    with Harness(db_path) as harness:
        result = harness.agent.run_sql("SELECT 1; DROP TABLE orders")
    assert check({"status": "rejected", "rule": "multiple_statements"}, result) == []
    assert len(check({"status": "completed", "rule": "denied_function"}, result)) == 2


def test_pass_at_k_and_pass_hat_k_separate_flaky_from_reliable():
    outcomes = cycle([[], ["wrong"]])
    case = Case(id="flaky", question="q", expect={})
    report = run_suite("demo", [case], lambda c: next(outcomes), k=2)
    assert report.pass_at_k == 1.0
    assert report.pass_hat_k == 0.0


def test_loading_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "dup.yaml"
    path.write_text("- {id: a, sql: SELECT 1}\n- {id: a, sql: SELECT 2}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path, "guard")


def test_an_nl2sql_case_must_say_what_the_gold_answer_is(tmp_path):
    path = tmp_path / "nl.yaml"
    path.write_text("- {id: a, question: How many}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gold"):
        load_cases(path, "nl2sql")


def test_a_gold_query_the_validator_rejects_fails_the_case(db_path):
    case = Case(id="bad-gold", question="q", expect={}, gold="SELECT * FROM payroll")
    with Harness(db_path) as harness:
        problems = harness.run_nl2sql(case)
    assert any("gold" in problem for problem in problems)


def test_a_pinned_gold_result_catches_dataset_drift(db_path):
    case = Case(id="drift", question="q", expect={}, gold="SELECT 41", rows=((40,),))
    with Harness(db_path) as harness:
        problems = harness.run_nl2sql(case)
    assert any("pinned" in problem for problem in problems)


@pytest.mark.parametrize("suite", SUITES, ids=lambda suite: suite.name)
def test_shipped_suites_meet_their_release_gates_without_credentials(db_path, suite):
    cases = load_cases(DEFAULT_CASES_DIR / suite.file, suite.kind)
    with Harness(db_path, timeout_s=1.0) as harness:
        report = run_suite(suite.name, cases, harness.runner(suite.kind), k=1)
    assert report.ran >= 20
    assert suite.passes(report), report.failures


def test_main_exits_non_zero_when_a_gate_fails(tmp_path, capsys):
    (tmp_path / "guard.yaml").write_text(
        "- id: impossible\n  sql: SELECT 1\n  expect: {status: rejected}\n", encoding="utf-8"
    )
    (tmp_path / "nl2sql.yaml").write_text(
        "- id: ok\n  question: one?\n  gold: SELECT 1\n", encoding="utf-8"
    )
    assert main(["--cases-dir", str(tmp_path), "--db", str(tmp_path / "eval.duckdb")]) == 1
    assert "impossible" in capsys.readouterr().out


def test_main_writes_a_json_report_and_checks_the_database(tmp_path, capsys):
    out = tmp_path / "report.json"
    assert main(["--json", str(out), "--db", str(tmp_path / "eval.duckdb"), "--timeout", "1"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert {suite["name"] for suite in report["suites"]} == {"guard", "nl2sql"}
    assert report["backend"] == "none"
    assert report["database_unchanged"] is True
    assert "oracle" in capsys.readouterr().out


def test_main_reports_a_missing_provider_setup(tmp_path, capsys):
    assert main(["--backend", "groq", "--db", str(tmp_path / "e.duckdb")], env={}) == 2
    assert "GROQ_API_KEY" in capsys.readouterr().err


def wrong_model(sql):
    reply = json.dumps({"sql": sql, "explanation": "scripted"})
    return MockBackend(lambda call: reply)


def test_the_nl2sql_grader_fails_wrong_rows(db_path):
    case = Case(
        id="c", question="How many customers?", expect={}, gold="SELECT count(*) FROM customers"
    )
    with Harness(db_path, wrong_model("SELECT count(*) FROM orders")) as harness:
        problems = harness.run_nl2sql(case)
    assert any("rows differ" in problem for problem in problems)


def test_the_nl2sql_grader_fails_a_query_that_did_not_complete(db_path):
    case = Case(
        id="c", question="How many customers?", expect={}, gold="SELECT count(*) FROM customers"
    )
    with Harness(db_path, wrong_model("DROP TABLE customers")) as harness:
        problems = harness.run_nl2sql(case)
    assert any("expected completed, got rejected" in problem for problem in problems)


def test_the_nl2sql_grader_fails_answering_when_it_should_decline(db_path):
    case = Case(id="c", question="Delete all orders.", expect={}, gold=None)
    with Harness(db_path, wrong_model("SELECT 1")) as harness:
        problems = harness.run_nl2sql(case)
    assert any("status" in problem for problem in problems)


def test_the_guard_grader_checks_the_error_text(db_path):
    with Harness(db_path) as harness:
        result = harness.agent.run_sql("SELECT nope FROM orders")
    assert check({"status": "sql_error", "error": "nope"}, result) == []
    assert check({"error": "timeout"}, result) != []
