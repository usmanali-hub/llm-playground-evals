"""Providers: Anthropic Messages API, OpenAI-compatible chat completions (also Ollama, vLLM, llama.cpp servers),
and an offline fake. Parsing is split into pure functions so it can be tested against recorded payloads."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Protocol

from llm_playground.sse import parse_sse
from llm_playground.transport import Transport, UrllibTransport
from llm_playground.types import ChatRequest, ChatResult, ProviderError, StreamEvent, Usage


class Provider(Protocol):
    def complete(self, request: ChatRequest) -> ChatResult: ...

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]: ...


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _load(data: str) -> dict[str, Any]:
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as err:
        raise ProviderError(f"stream sent invalid JSON: {data[:80]!r}") from err
    if not isinstance(parsed, dict):
        raise ProviderError("stream sent JSON that is not an object")
    return parsed


# ---- Anthropic ------------------------------------------------------------------------------------------------------

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


def anthropic_body(request: ChatRequest, stream: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
    }
    if request.system:
        body["system"] = request.system
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if stream:
        body["stream"] = True
    return body


def parse_anthropic_response(data: dict[str, Any]) -> ChatResult:
    blocks = data.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
    usage = data.get("usage") or {}
    return ChatResult(
        text=text,
        usage=Usage(_int(usage.get("input_tokens")), _int(usage.get("output_tokens"))),
        stop_reason=data.get("stop_reason"),
        model=str(data.get("model", "")),
    )


def anthropic_events(lines: Iterable[str]) -> Iterator[StreamEvent]:
    """Turn the Messages API event stream into text, usage and stop events; raise if it ends early or errors."""
    input_tokens = output_tokens = 0
    stop_reason: str | None = None
    for event, data in parse_sse(lines):
        payload = _load(data)
        if event == "error":
            error = payload.get("error") or {}
            kind = str(error.get("type", "unknown"))
            raise ProviderError(
                f"stream error: {kind}: {error.get('message', '')}"[:300],
                retryable=kind in {"overloaded_error", "rate_limit_error", "api_error"},
            )
        if event == "message_start":
            input_tokens = _int(((payload.get("message") or {}).get("usage") or {}).get("input_tokens"))
        elif event == "content_block_delta":
            delta = payload.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                yield StreamEvent("text", text=str(delta["text"]))
        elif event == "message_delta":
            output_tokens = _int((payload.get("usage") or {}).get("output_tokens"))
            stop_reason = (payload.get("delta") or {}).get("stop_reason") or stop_reason
        elif event == "message_stop":
            yield StreamEvent("usage", usage=Usage(input_tokens, output_tokens))
            yield StreamEvent("stop", stop_reason=stop_reason)
            return
    raise ProviderError("stream ended before message_stop (connection dropped?)", retryable=True)


class AnthropicProvider:
    def __init__(
        self, api_key: str | None = None, transport: Transport | None = None, url: str = ANTHROPIC_URL
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self._headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
        self._transport = transport or UrllibTransport()
        self._url = url

    def complete(self, request: ChatRequest) -> ChatResult:
        return parse_anthropic_response(
            self._transport.post_json(self._url, self._headers, anthropic_body(request, False))
        )

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        lines = self._transport.post_stream(self._url, self._headers, anthropic_body(request, True))
        yield from anthropic_events(lines)


# ---- OpenAI-compatible ----------------------------------------------------------------------------------------------

OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def openai_body(request: ChatRequest, stream: bool) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.extend({"role": m.role, "content": m.content} for m in request.messages)
    body: dict[str, Any] = {"model": request.model, "max_tokens": request.max_tokens, "messages": messages}
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def parse_openai_response(data: dict[str, Any]) -> ChatResult:
    choices = data.get("choices") or []
    first = choices[0] if choices else {}
    usage = data.get("usage") or {}
    return ChatResult(
        text=str((first.get("message") or {}).get("content") or ""),
        usage=Usage(_int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens"))),
        stop_reason=first.get("finish_reason"),
        model=str(data.get("model", "")),
    )


def openai_events(lines: Iterable[str]) -> Iterator[StreamEvent]:
    usage = Usage()
    stop_reason: str | None = None
    for _event, data in parse_sse(lines):
        if data.strip() == "[DONE]":
            yield StreamEvent("usage", usage=usage)
            yield StreamEvent("stop", stop_reason=stop_reason)
            return
        payload = _load(data)
        if "error" in payload:
            error = payload["error"]
            message = error.get("message", "") if isinstance(error, dict) else str(error)
            raise ProviderError(f"stream error: {message}"[:300])
        for choice in payload.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if content:
                yield StreamEvent("text", text=str(content))
            stop_reason = choice.get("finish_reason") or stop_reason
        if payload.get("usage"):
            u = payload["usage"]
            usage = Usage(_int(u.get("prompt_tokens")), _int(u.get("completion_tokens")))
    raise ProviderError("stream ended before [DONE] (connection dropped?)", retryable=True)


class OpenAICompatProvider:
    """OpenAI chat completions, or any server that speaks it (Ollama, vLLM, llama.cpp) via ``url``."""

    def __init__(
        self,
        api_key: str | None = None,
        transport: Transport | None = None,
        url: str = OPENAI_URL,
        require_key: bool = True,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if require_key and not key:
            raise ProviderError("OPENAI_API_KEY is not set")
        self._headers = {"authorization": f"Bearer {key}"} if key else {}
        self._transport = transport or UrllibTransport()
        self._url = url

    def complete(self, request: ChatRequest) -> ChatResult:
        return parse_openai_response(self._transport.post_json(self._url, self._headers, openai_body(request, False)))

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        lines = self._transport.post_stream(self._url, self._headers, openai_body(request, True))
        yield from openai_events(lines)


# ---- Offline fake ---------------------------------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough size only (about 4 characters per token). The fake uses it; real providers report exact counts."""
    return math.ceil(len(text) / 4) if text else 0


def default_reply(request: ChatRequest) -> str:
    last = request.messages[-1].content if request.messages else ""
    style = f" [system prompt: {request.system[:40]}]" if request.system else ""
    return f"(offline fake model){style} You said: {last}"


class FakeProvider:
    """Deterministic provider for tests and offline demos. It is not a language model."""

    def __init__(
        self, reply: Callable[[ChatRequest], str] = default_reply, fail_after_chunks: int | None = None
    ) -> None:
        self._reply = reply
        self._fail_after = fail_after_chunks
        self.requests: list[ChatRequest] = []

    def _usage(self, request: ChatRequest, text: str) -> Usage:
        prompt = (request.system or "") + "".join(m.content for m in request.messages)
        return Usage(estimate_tokens(prompt), estimate_tokens(text))

    def complete(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        text = self._reply(request)
        return ChatResult(text, self._usage(request, text), "end_turn", "fake")

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        self.requests.append(request)
        text = self._reply(request)
        pieces = [w + " " for w in text.split(" ")]
        pieces[-1] = pieces[-1].rstrip(" ")
        for index, piece in enumerate(pieces):
            if self._fail_after is not None and index >= self._fail_after:
                raise ProviderError("simulated connection drop", retryable=True)
            yield StreamEvent("text", text=piece)
        yield StreamEvent("usage", usage=self._usage(request, text))
        yield StreamEvent("stop", stop_reason="end_turn")
