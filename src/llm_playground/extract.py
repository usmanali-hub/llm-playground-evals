"""Structured extraction: ask for JSON that matches a schema, validate it, and ask again with the errors if it does not.

This is the portable approach (prompt, parse, validate, repair) and works with any provider. Providers also offer
native modes (forced tool calls, JSON mode); those are not implemented here. Validation proves the *shape* is right,
not that the values are true: a model can return a perfectly valid object with a wrong date.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from llm_playground.client import Chat
from llm_playground.schema import check_schema, validate
from llm_playground.types import Usage

MAX_ERRORS_SHOWN = 10
_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)


class ExtractionError(ValueError):
    """No JSON value could be read from the text."""


class StructuredOutputError(RuntimeError):
    """The model never produced valid output within the allowed attempts."""

    def __init__(self, message: str, errors: list[str], raw: str, attempts: int) -> None:
        super().__init__(message)
        self.errors = errors
        self.raw = raw
        self.attempts = attempts


@dataclass(frozen=True)
class StructuredResult:
    data: Any
    attempts: int  # 1 means the first reply was already valid
    errors_by_attempt: list[list[str]] = field(default_factory=list)  # what was wrong with each rejected reply
    usage: Usage = Usage()
    raw: str = ""


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


_DECODER = json.JSONDecoder(parse_constant=_reject_constant)


def extract_json(text: str) -> Any:
    """Read one JSON value from a model reply: bare JSON, a fenced code block, or JSON inside prose.

    Tries the whole text, then fenced blocks, then the first ``{`` or ``[`` that starts a valid object or array. If a
    reply contains several JSON values the first one wins. NaN and Infinity are rejected. Raises ExtractionError.
    """
    stripped = text.strip()
    try:
        return _DECODER.decode(stripped)
    except ValueError:
        pass
    for block in _FENCE.findall(text):
        try:
            return _DECODER.decode(block.strip())
        except ValueError:
            continue
    for match in re.finditer(r"[{\[]", text):
        try:
            value, _ = _DECODER.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(value, (dict, list)):
            return value
    raise ExtractionError("no JSON object or array found in the reply")


def build_prompt(text: str, schema: Any, instruction: str) -> str:
    return (
        f"{instruction}\n\n"
        "Return only a JSON value that matches this JSON Schema. No explanation and no code fence.\n\n"
        f"Schema:\n{json.dumps(schema, indent=2)}\n\nText:\n{text}"
    )


def repair_prompt(errors: list[str], truncated: bool) -> str:
    shown = errors[:MAX_ERRORS_SHOWN]
    more = f"\n- ...and {len(errors) - len(shown)} more" if len(errors) > len(shown) else ""
    cut = "Your reply was cut off at the token limit; keep it shorter. " if truncated else ""
    return (
        f"{cut}That reply did not match the schema:\n" + "\n".join(f"- {e}" for e in shown) + more
        + "\nReturn only the corrected JSON."
    )  # fmt: skip


def extract_structured(
    chat: Chat,
    text: str,
    schema: Any,
    instruction: str = "Extract the requested information from the text below.",
    max_repairs: int = 2,
) -> StructuredResult:
    """Extract data from ``text`` as JSON matching ``schema``, asking the model to fix invalid replies.

    Makes up to ``1 + max_repairs`` calls, each repair continuing the same conversation with the errors listed.
    Raises ``SchemaError`` for a bad schema before any call, and ``StructuredOutputError`` if no reply validates.
    """
    check_schema(schema)
    before = chat.usage
    prompt = build_prompt(text, schema, instruction)
    rejected: list[list[str]] = []
    raw = ""
    for attempt in range(1, max_repairs + 2):
        raw = chat.send(prompt).text
        errors: list[str]
        try:
            data = extract_json(raw)
            errors = validate(data, schema)
        except ExtractionError as err:
            errors = [f"could not read JSON: {err}"]
        if not errors:
            usage = Usage(
                chat.usage.input_tokens - before.input_tokens, chat.usage.output_tokens - before.output_tokens
            )
            return StructuredResult(data, attempt, rejected, usage, raw)
        rejected.append(errors)
        stop = chat.last_result.stop_reason if chat.last_result else None
        prompt = repair_prompt(errors, truncated=stop in ("max_tokens", "length"))
    raise StructuredOutputError(
        f"no valid output after {max_repairs + 1} attempts; last errors: {rejected[-1][:3]}",
        rejected[-1],
        raw,
        max_repairs + 1,
    )
