"""Value-level checks for extracted data. A schema says what shape the data has; these ask whether it is plausible.

``grounded_in_text`` catches the most common silent failure of extraction, a value the model made up that is not in
the source. It cannot tell whether a value that *is* in the text was put in the right field.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any

Check = Callable[[Any], list[str]]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _walk(data: Any, parts: list[str], path: str) -> Iterator[tuple[str, Any]]:
    """Yield (path, value) for a dotted path where ``[]`` after a name means every item ("items[].name")."""
    if not parts:
        yield path, data
        return
    head, rest = parts[0], parts[1:]
    each = head.endswith("[]")
    key = head[:-2] if each else head
    if not isinstance(data, dict) or key not in data:
        return  # a missing field is the schema's job to report, not this check's
    value = data[key]
    if each:
        if isinstance(value, list):
            for i, item in enumerate(value):
                yield from _walk(item, rest, f"{path}.{key}[{i}]")
    else:
        yield from _walk(value, rest, f"{path}.{key}")


def number_forms(value: float) -> set[str]:
    """The ways a number is commonly written in text: 12.5 -> '12.5', '12.50'; 1000 -> '1000', '1,000'."""
    if float(value).is_integer():
        n = int(value)
        return {str(n), f"{n:,}", f"{n}.0", f"{n}.00", f"{n:,}.00"}
    return {f"{value:g}", f"{value:.2f}", f"{value:,.2f}"}


def grounded_in_text(source: str, paths: Sequence[str]) -> Check:
    """A check that every string or number at the given paths appears in ``source``.

    Paths are dotted names, with ``[]`` for every item of a list: ``"vendor"``, ``"items[].name"``. Strings match
    case-insensitively with whitespace collapsed; numbers match any common way of writing them (12.5, 12.50, 1,000)
    and are not matched inside a longer number (12 does not appear in 12.50 or 1012).
    """
    haystack = _normalize(source)
    number_text = source.replace(" ", "")

    def check(data: Any) -> list[str]:
        problems: list[str] = []
        for path in paths:
            for where, value in _walk(data, path.split("."), "$"):
                if isinstance(value, bool) or value is None:
                    continue
                if isinstance(value, str):
                    if _normalize(value) not in haystack:
                        problems.append(f"{where}: {value!r} does not appear in the text")
                elif isinstance(value, (int, float)):
                    forms = sorted(number_forms(float(value)), key=len, reverse=True)
                    pattern = "(?<![\\d.,])(?:" + "|".join(re.escape(f) for f in forms) + ")(?![\\d]|[.,]\\d)"
                    if not re.search(pattern, number_text):
                        problems.append(f"{where}: {value} does not appear in the text")
        return problems

    return check


def totals_match(items_path: str, amount_field: str, total_field: str, tolerance: float = 0.01) -> Check:
    """A check that the sum of ``amount_field`` over the list at ``items_path`` equals ``total_field``."""

    def check(data: Any) -> list[str]:
        items = data.get(items_path) if isinstance(data, dict) else None
        if not isinstance(items, list) or not isinstance(data.get(total_field), (int, float)):
            return []
        amounts: list[float] = []
        for item in items:
            amount = item.get(amount_field) if isinstance(item, dict) else None
            if not isinstance(amount, (int, float)) or isinstance(amount, bool):
                return []  # the schema reports a malformed item; this check only compares clean numbers
            amounts.append(float(amount))
        total = sum(amounts)
        if abs(total - data[total_field]) > tolerance:
            return [
                f"$.{total_field}: {data[total_field]} does not equal the sum of "
                f"{items_path}[].{amount_field} ({total:g})"
            ]
        return []

    return check
