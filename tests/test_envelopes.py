import json
from datetime import date

import pytest
from pydantic import ValidationError

from nl_sql_agent.envelopes import (
    SCHEMA_VERSION,
    AgentResult,
    AgentStatus,
    DefenseReport,
    Finding,
    Layer,
    QueryResult,
    SQLProposal,
    ValidationReport,
    Violation,
)

SQL = "SELECT count(*) FROM orders"
CLEAN = DefenseReport(blocked=False)
BLOCKED = DefenseReport(blocked=True, findings=(Finding(rule="r", excerpt="x"),))
PROPOSAL = SQLProposal(sql=SQL, explanation="count orders", parse="clean_json")
PASSED = ValidationReport(sql=SQL, tables=("orders",))
FAILED = ValidationReport(
    sql=SQL, violations=(Violation(layer=Layer.LEXER, rule="denied_keyword", detail="DROP"),)
)
ROWS = QueryResult(columns=("n",), rows=((400,),), elapsed_ms=1.5)


def completed(**changes) -> AgentResult:
    fields = {
        "status": AgentStatus.COMPLETED,
        "question": "How many orders?",
        "defense": CLEAN,
        "proposal": PROPOSAL,
        "validation": PASSED,
        "result": ROWS,
    }
    return AgentResult(**{**fields, **changes})


def test_validation_report_is_ok_only_without_violations():
    assert PASSED.ok
    assert not FAILED.ok
    assert FAILED.rules == {"denied_keyword"}


def test_ok_and_row_count_are_part_of_the_json_contract():
    payload = json.loads(completed().model_dump_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["validation"]["ok"] is True
    assert payload["result"]["row_count"] == 1


def test_agent_result_round_trips_through_json():
    result = completed()
    assert AgentResult.model_validate_json(result.model_dump_json()) == result


def test_a_tampered_ok_flag_in_json_is_recomputed_not_trusted():
    payload = json.loads(FAILED.model_dump_json())
    payload["ok"] = True
    assert not ValidationReport.model_validate(payload).ok


def test_envelopes_are_immutable_and_reject_unknown_fields():
    with pytest.raises(ValidationError):
        PASSED.sql = "DROP TABLE orders"
    with pytest.raises(ValidationError):
        SQLProposal(sql=SQL, parse="clean_json", surprise=True)


def test_query_result_rows_must_match_the_columns():
    with pytest.raises(ValidationError, match="columns"):
        QueryResult(columns=("a", "b"), rows=((1,),), elapsed_ms=0)


def test_query_result_serializes_dates_for_json():
    result = QueryResult(columns=("d",), rows=((date(2025, 1, 2),),), elapsed_ms=0)
    assert json.loads(result.model_dump_json())["rows"] == [["2025-01-02"]]


def test_completed_result_needs_rows():
    with pytest.raises(ValidationError, match="completed"):
        completed(result=None)


def test_rows_can_never_come_from_sql_that_failed_validation():
    with pytest.raises(ValidationError, match="validation"):
        completed(validation=FAILED)


def test_rows_can_never_come_from_unvalidated_sql():
    with pytest.raises(ValidationError, match="validation"):
        completed(validation=None)


def test_validator_must_have_checked_exactly_the_proposed_sql():
    other = ValidationReport(sql="SELECT 1")
    with pytest.raises(ValidationError, match="proposed"):
        completed(validation=other)


def test_blocked_question_never_reaches_the_model():
    with pytest.raises(ValidationError, match="blocked"):
        AgentResult(status=AgentStatus.BLOCKED, question="q", defense=BLOCKED, proposal=PROPOSAL)
    ok = AgentResult(status=AgentStatus.BLOCKED, question="q", defense=BLOCKED)
    assert ok.proposal is None


def test_blocked_status_must_match_the_defense_report():
    with pytest.raises(ValidationError, match="blocked"):
        AgentResult(status=AgentStatus.BLOCKED, question="q", defense=CLEAN)
    with pytest.raises(ValidationError, match="blocked"):
        completed(defense=BLOCKED)


def test_rejected_means_the_validator_refused():
    rejected = completed(status=AgentStatus.REJECTED, validation=FAILED, result=None)
    assert rejected.validation is FAILED
    with pytest.raises(ValidationError, match="rejected"):
        completed(status=AgentStatus.REJECTED, result=None)


def test_unanswerable_means_the_model_wrote_no_sql():
    declined = SQLProposal(sql=None, explanation="no such data", parse="clean_json")
    result = AgentResult(
        status=AgentStatus.UNANSWERABLE, question="q", defense=CLEAN, proposal=declined
    )
    assert result.proposal.sql is None
    with pytest.raises(ValidationError, match="unanswerable"):
        AgentResult(status=AgentStatus.UNANSWERABLE, question="q", defense=CLEAN, proposal=PROPOSAL)


def test_sql_error_needs_validated_sql_and_an_error_message():
    failed = completed(status=AgentStatus.SQL_ERROR, result=None, error="timeout")
    assert failed.error == "timeout"
    with pytest.raises(ValidationError, match="error"):
        completed(status=AgentStatus.SQL_ERROR, result=None)
    with pytest.raises(ValidationError, match="sql_error"):
        completed(status=AgentStatus.SQL_ERROR, validation=FAILED, result=None, error="x")


def test_llm_error_has_no_proposal_and_explains_why():
    result = AgentResult(
        status=AgentStatus.LLM_ERROR, question="q", defense=CLEAN, error="not_json"
    )
    assert result.error == "not_json"
    with pytest.raises(ValidationError, match="error"):
        AgentResult(status=AgentStatus.LLM_ERROR, question="q", defense=CLEAN)


def test_error_message_only_on_error_statuses():
    with pytest.raises(ValidationError, match="error"):
        completed(error="surprise")


def test_llm_error_cannot_carry_a_proposal():
    with pytest.raises(ValidationError, match="llm_error"):
        AgentResult(
            status=AgentStatus.LLM_ERROR,
            question="q",
            defense=CLEAN,
            proposal=SQLProposal(sql=None, parse="clean_json"),
            error="x",
        )
