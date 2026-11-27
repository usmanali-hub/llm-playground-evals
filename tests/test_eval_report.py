"""The committed eval_report.json is a shipped artifact (what `make report` produces), not generated at test time —
this just proves it hasn't gone stale: a change to a case, a check, or the extraction pipeline that would change the
report must also update the committed file, the same drift check ml-end-to-end's model card gets."""

import json
from pathlib import Path

from llm_playground.eval_cases import CASES
from llm_playground.evals import run_suite

REPORT_PATH = Path(__file__).resolve().parents[1] / "eval_report.json"


def test_committed_eval_report_matches_a_fresh_run_of_the_suite() -> None:
    committed = json.loads(REPORT_PATH.read_text())
    fresh = run_suite(CASES).to_dict()
    assert committed == fresh, "eval_report.json is stale; regenerate it with `make report`"
