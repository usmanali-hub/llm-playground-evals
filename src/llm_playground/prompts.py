"""Prompt patterns (zero-shot, few-shot, chain of thought) and a small harness to compare them on labelled cases.

The builders are pure functions, so the exact text sent to a model is testable. The harness runs any provider; with the
offline ``FakeProvider`` it only proves the harness works. It says nothing about how real models respond to a pattern:
that needs a real model, many cases, and repeated runs, because one run of one model on a few cases is mostly noise.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from llm_playground.client import Chat
from llm_playground.providers import Provider

ANSWER_MARKER = "Answer:"


@dataclass(frozen=True)
class Example:
    input: str
    output: str


@dataclass(frozen=True)
class Case:
    question: str
    expected: str


def zero_shot_prompt(task: str, question: str) -> str:
    return f"{task}\n\nInput: {question}\nOutput:"


def few_shot_prompt(task: str, examples: Sequence[Example], question: str) -> str:
    """Show worked examples in the exact format the answer should take, then the real input.

    Examples teach format and label vocabulary more reliably than they teach reasoning, and their order and balance can
    bias the answer (a run of identical labels pushes the model toward that label), so keep them varied.
    """
    if not examples:
        raise ValueError("few-shot needs at least one example; use zero_shot_prompt for none")
    shots = "\n\n".join(f"Input: {e.input}\nOutput: {e.output}" for e in examples)
    return f"{task}\n\n{shots}\n\nInput: {question}\nOutput:"


def cot_prompt(task: str, question: str) -> str:
    """Ask for reasoning first and a clearly marked final answer last, so the answer can be parsed reliably."""
    return (
        f"{task}\n\nQuestion: {question}\n\n"
        f"Think through it step by step, then give the final result on its own last line as '{ANSWER_MARKER} <result>'."
    )


def parse_final_answer(text: str) -> str | None:
    """The text after the *last* 'Answer:' marker (reasoning may mention the word earlier), or None if absent."""
    found = re.findall(rf"{re.escape(ANSWER_MARKER)}[ \t]*(.+)", text, flags=re.IGNORECASE)
    return found[-1].strip().rstrip(".") if found else None


def parse_plain_answer(text: str) -> str | None:
    """The first non-empty line, for patterns that ask for the answer and nothing else."""
    for line in text.splitlines():
        if line.strip():
            return line.strip().rstrip(".")
    return None


@dataclass(frozen=True)
class Pattern:
    name: str
    build: Callable[[str], str]
    parse: Callable[[str], str | None]


@dataclass(frozen=True)
class CaseResult:
    question: str
    expected: str
    got: str | None
    correct: bool
    output_tokens: int


@dataclass(frozen=True)
class PatternReport:
    pattern: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return sum(r.correct for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def unparsed(self) -> int:
        """Replies where no answer could be read; counted wrong, but reported apart because the cause differs."""
        return sum(r.got is None for r in self.results)

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results)


def _same(got: str | None, expected: str) -> bool:
    return got is not None and got.strip().lower() == expected.strip().lower()


def compare_patterns(
    provider: Provider, model: str, patterns: Sequence[Pattern], cases: Sequence[Case], system: str | None = None
) -> list[PatternReport]:
    """Run every pattern on every case, each in a fresh conversation, and score by exact match (case-insensitive).

    Reports accuracy, replies that could not be parsed, and output tokens (chain of thought costs more of them).
    """
    if not cases:
        raise ValueError("no cases to evaluate")
    reports: list[PatternReport] = []
    for pattern in patterns:
        report = PatternReport(pattern.name)
        for case in cases:
            chat = Chat(provider, model, system=system)
            reply = chat.send(pattern.build(case.question))
            got = pattern.parse(reply.text)
            report.results.append(
                CaseResult(case.question, case.expected, got, _same(got, case.expected), reply.usage.output_tokens)
            )
        reports.append(report)
    return reports


def format_reports(reports: Sequence[PatternReport]) -> str:
    rows = ["pattern        correct  unparsed  output tokens"]
    for r in reports:
        right = sum(x.correct for x in r.results)
        rows.append(f"{r.pattern:<14} {right:>3}/{len(r.results):<4} {r.unparsed:>8}  {r.output_tokens:>13}")
    return "\n".join(rows)
