"""Output guardrails: checks run on a model's reply before it reaches the caller, with a repair loop (ask the
model to try again, listing what was wrong) mirroring `extract_structured`'s approach to schema validation —
except these checks apply to any plain-text reply, not just one that is supposed to be JSON.

A guardrail is a regex/length/keyword-level heuristic, not a safety classifier: `no_pii_patterns` catches an email
address or a US-style phone number shaped string, not every way PII can appear in text, and `no_banned_terms` is a
literal substring match. These are the same class of check as `llm_playground.checks` (grounded_in_text,
totals_match) applied to free text instead of structured data, not a claim that passing every guardrail here means
a reply is actually safe.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from llm_playground.client import Chat
from llm_playground.types import ChatResult

Guardrail = Callable[[str], list[str]]  # returns violation messages; empty means the text passed

MAX_VIOLATIONS_SHOWN = 10

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_US_PHONE = re.compile(r"(?<!\d)(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")
_CREDIT_CARD_LIKE = re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)")


def max_length(limit: int, unit: str = "characters") -> Guardrail:
    """A reply over ``limit`` units (characters, or words if ``unit="words"``) fails."""

    def check(text: str) -> list[str]:
        size = len(text) if unit == "characters" else len(text.split())
        if size > limit:
            return [f"reply is {size} {unit} long, over the {limit} {unit} limit"]
        return []

    return check


def min_length(minimum: int, unit: str = "characters") -> Guardrail:
    """A reply under ``minimum`` units fails — catches a suspiciously empty or truncated-looking reply."""

    def check(text: str) -> list[str]:
        size = len(text) if unit == "characters" else len(text.split())
        if size < minimum:
            return [f"reply is only {size} {unit} long, under the {minimum} {unit} minimum"]
        return []

    return check


def no_banned_terms(terms: Sequence[str]) -> Guardrail:
    """A reply containing any of ``terms`` (case-insensitive substring match) fails. A literal match, not a
    semantic one: it will not catch a paraphrase, and it will flag a term appearing inside an unrelated word."""
    lowered = [t.lower() for t in terms]

    def check(text: str) -> list[str]:
        found = [t for t in lowered if t in text.lower()]
        return [f"reply contains the banned term {t!r}" for t in found]

    return check


def no_pii_patterns() -> Guardrail:
    """A reply containing something shaped like an email address, a US-style phone number, or a 13-16 digit
    run (credit-card-like) fails. Shape-based, not identity-based: it cannot tell a real SSN from a made-up example
    number in the same shape, and it will miss PII in a shape it does not recognize (an address, a non-US phone)."""

    def check(text: str) -> list[str]:
        violations: list[str] = []
        if _EMAIL.search(text):
            violations.append("reply appears to contain an email address")
        if _US_PHONE.search(text):
            violations.append("reply appears to contain a US-style phone number")
        if _CREDIT_CARD_LIKE.search(text):
            violations.append("reply appears to contain a 13-16 digit number (credit-card-shaped)")
        return violations

    return check


def must_match(pattern: str, description: str) -> Guardrail:
    """A reply that does NOT match ``pattern`` (searched, not anchored) fails — the inverse of the other checks,
    for requiring a format rather than forbidding content (e.g. "must start with a greeting")."""
    compiled = re.compile(pattern)

    def check(text: str) -> list[str]:
        return [] if compiled.search(text) else [f"reply does not satisfy: {description}"]

    return check


def run_guardrails(text: str, guardrails: Sequence[Guardrail]) -> list[str]:
    """Run every guardrail against ``text`` and return all violations found, in guardrail order."""
    violations: list[str] = []
    for guardrail in guardrails:
        violations.extend(guardrail(text))
    return violations


class GuardrailError(RuntimeError):
    """No reply passed every guardrail within the allowed attempts."""

    def __init__(self, message: str, violations: list[str], text: str, attempts: int) -> None:
        super().__init__(message)
        self.violations = violations
        self.text = text
        self.attempts = attempts


@dataclass(frozen=True)
class GuardedResult:
    text: str
    attempts: int  # 1 means the first reply already passed every guardrail
    violations_by_attempt: list[list[str]] = field(default_factory=list)  # what failed on each rejected reply


def _repair_prompt(violations: list[str]) -> str:
    shown = violations[:MAX_VIOLATIONS_SHOWN]
    more = f"\n- ...and {len(violations) - len(shown)} more" if len(violations) > len(shown) else ""
    body = "\n".join(f"- {v}" for v in shown)
    return f"That reply failed these checks:\n{body}{more}\nReply again, fixing them."


def guarded_send(chat: Chat, prompt: str, guardrails: Sequence[Guardrail], max_retries: int = 2) -> GuardedResult:
    """Send ``prompt`` and run ``guardrails`` on the reply; if any fail, ask the model to try again with the
    violations listed, up to ``1 + max_retries`` attempts total — the same repair-loop shape `extract_structured`
    uses for schema errors, applied to plain text instead of JSON. Raises `GuardrailError` if no attempt passes.
    """
    rejected: list[list[str]] = []
    result: ChatResult | None = None
    next_prompt = prompt
    for attempt in range(1, max_retries + 2):
        result = chat.send(next_prompt)
        violations = run_guardrails(result.text, guardrails)
        if not violations:
            return GuardedResult(result.text, attempt, rejected)
        rejected.append(violations)
        next_prompt = _repair_prompt(violations)
    raise GuardrailError(
        f"no reply passed every guardrail after {max_retries + 1} attempts; last violations: {rejected[-1][:3]}",
        rejected[-1],
        result.text if result is not None else "",
        max_retries + 1,
    )
