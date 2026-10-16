import io
import json

import pytest

from nl_sql_agent import cli
from nl_sql_agent.cli import main
from nl_sql_agent.providers import MockBackend


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "shop.duckdb")


def run(capsys, *argv, env=None):
    code = main(list(argv), env=env)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_sql_mode_runs_the_guard_and_prints_the_result_as_json(capsys, db):
    code, out, _ = run(capsys, "--db", db, "--sql", "SELECT count(*) AS n FROM customers")
    assert code == 0
    payload = json.loads(out)
    assert payload["status"] == "completed"
    assert payload["result"]["rows"] == [[40]]


def test_sql_mode_shows_why_an_attack_was_rejected(capsys, db):
    _, out, _ = run(capsys, "--db", db, "--sql", "SELECT 1; DROP TABLE orders")
    payload = json.loads(out)
    assert payload["status"] == "rejected"
    assert {v["rule"] for v in payload["validation"]["violations"]} >= {"multiple_statements"}


def test_the_database_is_built_on_first_use_and_can_be_rebuilt(capsys, tmp_path):
    path = tmp_path / "fresh.duckdb"
    run(capsys, "--db", str(path), "--sql", "SELECT 1")
    assert path.exists()
    stamp = path.stat().st_mtime_ns
    run(capsys, "--db", str(path), "--rebuild-db", "--sql", "SELECT 1")
    assert path.stat().st_mtime_ns != stamp


def test_schema_prints_what_the_model_sees_and_hides_the_rest(capsys, db):
    code, out, _ = run(capsys, "--db", db, "--schema")
    assert code == 0
    assert "TABLE order_items" in out
    assert "payroll" not in out


def test_a_question_without_a_backend_is_a_usage_error(capsys, db):
    code, out, err = run(capsys, "--db", db, "How many customers?")
    assert code == 2
    assert "--sql" in err
    assert out == ""


def test_a_missing_provider_setup_is_reported_without_a_traceback(capsys, db):
    code, _, err = run(capsys, "--db", db, "--backend", "groq", "hi", env={})
    assert code == 2
    assert "GROQ_API_KEY" in err


def test_a_question_goes_through_the_model_and_can_come_from_stdin(capsys, db, monkeypatch):
    reply = json.dumps({"sql": "SELECT count(*) FROM products", "explanation": "count"})
    monkeypatch.setattr(cli, "resolve_backend", lambda name, env: MockBackend([reply]))
    monkeypatch.setattr("sys.stdin", io.StringIO("How many products?"))
    code, out, _ = run(capsys, "--db", db, "--backend", "groq")
    assert code == 0
    payload = json.loads(out)
    assert payload["question"] == "How many products?"
    assert payload["result"]["rows"] == [[15]]
