"""Providers and the urllib transport against a real HTTP server on localhost that replays recorded-shape payloads.

This exercises the actual request, header, status, Retry-After and streaming code, not the live provider services."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from llm_playground import (
    AnthropicProvider,
    Chat,
    ChatRequest,
    Message,
    OpenAICompatProvider,
    ProviderError,
    RetryingProvider,
)
from tests.fakes import ANTHROPIC_STREAM, OPENAI_STREAM

REQUEST = ChatRequest("m", (Message("user", "hi"),), system="Be brief.")
KEY = "sk-live-looking-secret-0123456789"


class Handler(BaseHTTPRequestHandler):
    script: list[tuple[int, dict[str, str], bytes]] = []
    seen: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("content-length", 0))
        Handler.seen.append(
            {
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": json.loads(self.rfile.read(length)),
            }
        )
        status, headers, body = Handler.script.pop(0) if Handler.script else (500, {}, b"script exhausted")
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # silence request logging
        pass


@pytest.fixture
def server() -> Iterator[str]:
    Handler.script, Handler.seen = [], []
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1/messages"
    httpd.shutdown()
    httpd.server_close()


def sse(chunks: list[str]) -> bytes:
    return "".join(chunks).encode()


def test_complete_sends_the_documented_request_and_parses_the_reply(server: str) -> None:
    reply = {
        "model": "m",
        "content": [{"type": "text", "text": "Hello"}],
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
    Handler.script = [(200, {"content-type": "application/json"}, json.dumps(reply).encode())]
    result = AnthropicProvider(api_key=KEY, url=server).complete(REQUEST)
    assert result.text == "Hello" and result.usage.total == 4
    sent = Handler.seen[0]
    assert sent["headers"]["x-api-key"] == KEY and sent["headers"]["anthropic-version"]
    assert sent["body"]["system"] == "Be brief." and sent["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_streaming_over_http_returns_the_text_and_usage(server: str) -> None:
    Handler.script = [(200, {"content-type": "text/event-stream"}, sse(ANTHROPIC_STREAM))]
    chat = Chat(AnthropicProvider(api_key=KEY, url=server), "m")
    assert "".join(chat.stream("hi")) == "Hello"
    assert chat.usage.input_tokens == 25 and chat.usage.output_tokens == 15


def test_openai_style_streaming_over_http(server: str) -> None:
    Handler.script = [(200, {"content-type": "text/event-stream"}, sse(OPENAI_STREAM))]
    chat = Chat(OpenAICompatProvider(api_key=KEY, url=server), "m")
    assert "".join(chat.stream("hi")) == "Hello" and chat.usage.total == 11


def test_http_errors_carry_status_retry_after_and_never_the_key(server: str) -> None:
    body = f"invalid key {KEY}".encode()
    Handler.script = [(429, {"retry-after": "2"}, body)]
    with pytest.raises(ProviderError) as info:
        AnthropicProvider(api_key=KEY, url=server).complete(REQUEST)
    err = info.value
    assert err.status == 429 and err.retryable and err.retry_after == 2.0
    assert KEY not in str(err) and "[redacted]" in str(err)


def test_a_400_is_not_retryable(server: str) -> None:
    Handler.script = [(400, {}, b'{"error":"bad request"}')]
    with pytest.raises(ProviderError) as info:
        AnthropicProvider(api_key=KEY, url=server).complete(REQUEST)
    assert info.value.status == 400 and not info.value.retryable


def test_retry_wrapper_recovers_from_an_overloaded_server(server: str) -> None:
    ok = {
        "model": "m",
        "content": [{"type": "text", "text": "Hello"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    Handler.script = [(529, {}, b"overloaded"), (529, {}, b"overloaded"), (200, {}, json.dumps(ok).encode())]
    sleeps: list[float] = []
    provider = RetryingProvider(AnthropicProvider(api_key=KEY, url=server), sleep=sleeps.append)
    assert provider.complete(REQUEST).text == "Hello" and len(sleeps) == 2 and len(Handler.seen) == 3


def test_an_unreachable_server_is_a_retryable_provider_error() -> None:
    provider = AnthropicProvider(api_key=KEY, url="http://127.0.0.1:1/v1/messages")  # nothing listens on port 1
    with pytest.raises(ProviderError) as info:
        provider.complete(REQUEST)
    assert info.value.retryable and KEY not in str(info.value)
