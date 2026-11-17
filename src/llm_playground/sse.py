"""Server-sent events parsing, following the WHATWG rules the LLM streaming APIs use."""

from __future__ import annotations

from collections.abc import Iterable, Iterator


def parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Yield ``(event, data)`` for each event in an iterable of text lines.

    ``event`` defaults to ``"message"``. Several ``data:`` lines in one event are joined with a newline. Lines
    starting with ``:`` are comments. A trailing event without a final blank line is still delivered. Lines may or
    may not end in a newline; one leading space after the colon is dropped.
    """
    event = "message"
    data: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield event, "\n".join(data)
