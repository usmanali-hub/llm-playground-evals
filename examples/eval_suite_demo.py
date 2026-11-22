"""Run the 30-case eval suite and print a pass/fail report. Same thing `python -m llm_playground eval` does; see
`llm_playground.eval_cases` for the cases and `llm_playground.evals` for what a "pass" here does and does not prove.
"""

from __future__ import annotations

from llm_playground.eval_cases import CASES
from llm_playground.evals import run_suite

if __name__ == "__main__":
    report = run_suite(CASES)
    print(report.summary())
    raise SystemExit(0 if not report.failed_results else 1)
