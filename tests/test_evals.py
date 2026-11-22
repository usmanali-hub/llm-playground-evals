from llm_playground.eval_cases import CASES
from llm_playground.evals import EvalCase, run_case, run_suite

PERSON_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}


def test_run_case_passes_when_extraction_succeeds_and_was_expected_to():
    case = EvalCase("ok", "Jane is 34.", PERSON_SCHEMA, ['{"name": "Jane", "age": 34}'])
    result = run_case(case)
    assert result.passed is True
    assert result.attempts == 1


def test_run_case_fails_when_extraction_unexpectedly_succeeds():
    # expect_valid=False but the scripted reply is actually valid: the case itself is wrong, and run_case must say so
    # rather than reporting a pass just because extraction didn't raise.
    case = EvalCase("surprise_pass", "Jane is 34.", PERSON_SCHEMA, ['{"name": "Jane", "age": 34}'], expect_valid=False)
    result = run_case(case)
    assert result.passed is False
    assert "succeeded" in result.detail


def test_run_case_fails_when_extraction_unexpectedly_fails():
    case = EvalCase("surprise_fail", "Jane is 34.", PERSON_SCHEMA, ["not json"], max_repairs=0)
    result = run_case(case)
    assert result.passed is False
    assert "expected valid output" in result.detail


def test_run_case_passes_when_failure_was_expected():
    case = EvalCase("expected_fail", "Jane is 34.", PERSON_SCHEMA, ["not json"], expect_valid=False, max_repairs=0)
    result = run_case(case)
    assert result.passed is True
    assert "as expected" in result.detail


def test_run_suite_aggregates_results_and_reports_a_clean_summary():
    cases = [
        EvalCase("a", "x", PERSON_SCHEMA, ['{"name": "x", "age": 1}']),
        EvalCase("b", "x", PERSON_SCHEMA, ["not json"], expect_valid=False, max_repairs=0),
    ]
    report = run_suite(cases)
    assert report.passed == 2
    assert report.failed_results == ()
    summary = report.summary()
    assert "PASS  a" in summary
    assert "PASS  b" in summary
    assert "2/2 cases passed" in summary


def test_run_suite_reports_failures_distinctly_from_passes():
    cases = [
        EvalCase("good", "x", PERSON_SCHEMA, ['{"name": "x", "age": 1}']),
        EvalCase("bad", "x", PERSON_SCHEMA, ["not json"], max_repairs=0),  # expected valid, will actually fail
    ]
    report = run_suite(cases)
    assert report.passed == 1
    assert [r.name for r in report.failed_results] == ["bad"]
    assert "FAIL  bad" in report.summary()


def test_the_shipped_eval_suite_has_30_cases_and_all_pass():
    assert len(CASES) == 30
    report = run_suite(CASES)
    assert report.failed_results == (), report.summary()
    assert report.passed == 30


def test_the_shipped_eval_suite_has_unique_case_names():
    names = [case.name for case in CASES]
    assert len(names) == len(set(names))
