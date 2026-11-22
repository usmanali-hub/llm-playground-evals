"""Eval sets: named, repeatable test cases that exercise `extract_structured` against scripted provider replies,
so a suite runs the same way every time, offline, with no network call.

**Honest scope, stated plainly**: each case's "model" reply is a hand-written string played back by `FakeProvider`
via `scripted_reply`, not a live model's output. This is a regression suite for the extraction-and-validation
pipeline itself — schema checks, the repair loop, value-level checks, native tool mode — and for the prompts that
drive it, not a benchmark of how well any real model follows instructions. Running the same `EvalCase.schema` and
`checks` against a real provider's replies (swap `FakeProvider` for `AnthropicProvider`/`OpenAICompatProvider`, and
drop the scripted ``replies``) would turn the same suite into a genuine model eval; that swap is not done here
because this sandbox holds no provider API key.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from llm_playground.client import Chat
from llm_playground.extract import StructuredOutputError, extract_structured
from llm_playground.providers import FakeProvider, scripted_reply

Check = Callable[[Any], list[str]]


@dataclass(frozen=True)
class EvalCase:
    """One test case: a source text, a schema, and the exact replies a scripted "model" gives back.

    ``replies`` are consumed in order, one per `extract_structured` attempt (the first reply, then one per repair);
    the last reply repeats if more attempts happen than replies given. ``expect_valid`` says whether extraction
    should end up succeeding — a case can legitimately expect failure, to prove the pipeline rejects bad output
    rather than silently accepting it.
    """

    name: str
    text: str
    schema: Any
    replies: Sequence[str]
    expect_valid: bool = True
    checks: Sequence[Check] = ()
    native: bool = False
    max_repairs: int = 2
    instruction: str = "Extract the requested information from the text below."


@dataclass(frozen=True)
class EvalResult:
    name: str
    passed: bool
    detail: str
    attempts: int = 0


def run_case(case: EvalCase) -> EvalResult:
    """Run one `EvalCase` to completion and say whether its outcome matched ``expect_valid``."""
    provider = FakeProvider(scripted_reply(list(case.replies)))
    chat = Chat(provider, model="fake-eval")
    try:
        result = extract_structured(
            chat,
            case.text,
            case.schema,
            instruction=case.instruction,
            max_repairs=case.max_repairs,
            native=case.native,
            checks=case.checks,
        )
    except StructuredOutputError as err:
        if case.expect_valid:
            return EvalResult(case.name, False, f"expected valid output, extraction failed: {err}")
        return EvalResult(case.name, True, "failed to validate, as expected", attempts=err.attempts)
    if not case.expect_valid:
        return EvalResult(
            case.name, False, f"expected extraction to fail, but it succeeded in {result.attempts} attempt(s)"
        )
    return EvalResult(case.name, True, f"valid in {result.attempts} attempt(s)", attempts=result.attempts)


@dataclass(frozen=True)
class EvalReport:
    results: tuple[EvalResult, ...] = ()

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_results(self) -> tuple[EvalResult, ...]:
        return tuple(r for r in self.results if not r.passed)

    def summary(self) -> str:
        lines = [f"{'PASS' if r.passed else 'FAIL'}  {r.name} — {r.detail}" for r in self.results]
        lines.append(f"\n{self.passed}/{len(self.results)} cases passed")
        return "\n".join(lines)


def run_suite(cases: Sequence[EvalCase]) -> EvalReport:
    return EvalReport(tuple(run_case(c) for c in cases))
