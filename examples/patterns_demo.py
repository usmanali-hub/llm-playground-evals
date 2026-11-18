"""Runs the three prompt patterns through the comparison harness with a hand-scripted offline "model".

The model's answers are written in advance (it gets the arithmetic wrong when it answers directly and right when it
shows work), so the table below is a property of the script, NOT a measurement of any real model. It shows what the
harness reports: correct count, unparsed replies, and the extra output tokens that reasoning costs.
"""

from llm_playground import FakeProvider
from llm_playground.prompts import (
    Case,
    Example,
    Pattern,
    compare_patterns,
    cot_prompt,
    few_shot_prompt,
    format_reports,
    parse_final_answer,
    parse_plain_answer,
    zero_shot_prompt,
)

TASK = "Add the numbers."
CASES = [Case("17 + 25", "42"), Case("38 + 46", "84"), Case("59 + 27", "86")]
SHOTS = [Example("2 + 3", "5"), Example("10 + 11", "21")]
WORK = {"17 + 25": "17 + 25 = 17 + 20 + 5 = 42", "38 + 46": "38 + 46 = 38 + 40 + 6 = 84", "59 + 27": "59 + 27 = 86"}
GUESS = {"17 + 25": "42", "38 + 46": "74", "59 + 27": "86"}  # one wrong guess, written by hand


def scripted_model(request) -> str:  # type: ignore[no-untyped-def]
    prompt = request.messages[-1].content
    question = next(q for q in WORK if q in prompt)
    if "step by step" in prompt:
        return f"{WORK[question]}\nAnswer: {WORK[question].split('= ')[-1]}"
    return GUESS[question]


patterns = [
    Pattern("zero-shot", lambda q: zero_shot_prompt(TASK, q), parse_plain_answer),
    Pattern("few-shot", lambda q: few_shot_prompt(TASK, SHOTS, q), parse_plain_answer),
    Pattern("chain-of-thought", lambda q: cot_prompt(TASK, q), parse_final_answer),
]
print(format_reports(compare_patterns(FakeProvider(reply=scripted_model), "fake", patterns, CASES)))
print("(scripted answers: this shows the harness, not how a real model behaves)")
