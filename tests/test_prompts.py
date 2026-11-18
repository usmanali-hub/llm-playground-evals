import pytest

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

TASK = "Classify the sentiment as positive or negative."
SHOTS = [Example("I loved it", "positive"), Example("Awful", "negative")]


def test_few_shot_prompt_layout_is_exact() -> None:
    shots = "Input: I loved it\nOutput: positive\n\nInput: Awful\nOutput: negative"
    assert few_shot_prompt(TASK, SHOTS, "Fine I guess") == f"{TASK}\n\n{shots}\n\nInput: Fine I guess\nOutput:"


def test_few_shot_needs_examples() -> None:
    with pytest.raises(ValueError, match="at least one example"):
        few_shot_prompt(TASK, [], "x")


def test_zero_shot_and_cot_prompts() -> None:
    assert zero_shot_prompt(TASK, "x").endswith("Input: x\nOutput:")
    cot = cot_prompt("Do arithmetic.", "2+2?")
    assert "step by step" in cot and "'Answer: <result>'" in cot and "Question: 2+2?" in cot


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2+2 is 4.\nAnswer: 4", "4"),
        ("The word answer: appears early.\nAnswer: yes.", "yes"),  # last marker wins, trailing period dropped
        ("ANSWER:   42  ", "42"),
        ("I think it is four", None),
        ("Answer:", None),
    ],
)
def test_parse_final_answer(text: str, expected: str | None) -> None:
    assert parse_final_answer(text) == expected


def test_parse_plain_answer_takes_first_nonempty_line() -> None:
    assert parse_plain_answer("\n  positive.\nbecause...") == "positive"
    assert parse_plain_answer("  \n") is None


def scripted(replies: dict[str, str]) -> FakeProvider:
    """A fake model that answers by looking for a key in the prompt."""

    def reply(request) -> str:  # type: ignore[no-untyped-def]
        prompt = request.messages[-1].content
        return next(text for key, text in replies.items() if key in prompt)

    return FakeProvider(reply=reply)


def test_compare_patterns_scores_parses_and_counts_tokens() -> None:
    cases = [Case("2+2?", "4"), Case("3+3?", "6"), Case("9-1?", "8")]
    provider = scripted({"2+2?": "Answer: 4", "3+3?": "Answer: 7", "9-1?": "no marker here"})
    pattern = Pattern("cot", lambda q: cot_prompt("Math.", q), parse_final_answer)
    (report,) = compare_patterns(provider, "fake", [pattern], cases)
    assert [r.correct for r in report.results] == [True, False, False]
    assert report.unparsed == 1 and report.accuracy == pytest.approx(1 / 3) and report.output_tokens > 0
    assert "cot" in format_reports([report]) and "1/3" in format_reports([report])


def test_each_case_runs_in_a_fresh_conversation() -> None:
    provider = scripted({"a": "Answer: 1", "b": "Answer: 1"})
    pattern = Pattern("p", lambda q: q, parse_final_answer)
    compare_patterns(provider, "fake", [pattern], [Case("a", "1"), Case("b", "1")])
    assert all(len(r.messages) == 1 for r in provider.requests)


def test_matching_ignores_case_and_no_cases_is_an_error() -> None:
    provider = scripted({"q": "Positive"})
    pattern = Pattern("p", lambda q: q, parse_plain_answer)
    (report,) = compare_patterns(provider, "fake", [pattern], [Case("q", "positive")])
    assert report.results[0].correct
    with pytest.raises(ValueError, match="no cases"):
        compare_patterns(provider, "fake", [pattern], [])
