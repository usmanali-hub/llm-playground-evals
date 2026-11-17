"""Recorded-shape payloads and a transport that serves them, so provider code runs with no network."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

ANTHROPIC_STREAM = [
    'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","usage":{"input_tokens":25,"output_tokens":1}}}\n\n',
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
    'event: ping\ndata: {"type": "ping"}\n\n',
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}\n\n',
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}\n\n',
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}\n\n',
    'event: message_stop\ndata: {"type":"message_stop"}\n\n',
]

OPENAI_STREAM = [
    'data: {"choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
    'data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}\n\n',
    'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n\n',
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
    'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":2}}\n\n',
    "data: [DONE]\n\n",
]


class FakeTransport:
    def __init__(self, json_reply: dict[str, Any] | None = None, stream_lines: list[str] | None = None) -> None:
        self.json_reply = json_reply or {}
        self.stream_lines = stream_lines or []
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def post_json(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((url, headers, json.loads(json.dumps(body))))
        return self.json_reply

    def post_stream(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> Iterator[str]:
        self.calls.append((url, headers, json.loads(json.dumps(body))))
        for chunk in self.stream_lines:
            yield from chunk.splitlines(keepends=True)
