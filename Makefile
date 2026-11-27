# From a fresh clone to a chat reply, a structured extraction, and a scored eval report:  make demo
PY ?= python

.PHONY: install demo test report
install:
	$(PY) -m pip install -e ".[dev]"

demo:
	$(PY) -m llm_playground chat --provider fake --system "Answer briefly." --stream "What is a token?"
	$(PY) examples/repair_demo.py
	$(PY) -m llm_playground eval

test:
	$(PY) -m pytest -q

report:
	$(PY) -m llm_playground eval --json > eval_report.json
