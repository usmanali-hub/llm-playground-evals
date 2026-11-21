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
from llm_playground.types import ChatRequest, ChatResult, ProviderError, StreamEvent, ToolCall, Usage


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
        if request.cache_system:
            # A content-block list lets one block carry cache_control; a plain string cannot.
            body["system"] = [{"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}]
        else:
            body["system"] = request.system
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.tool:
        tool_def: dict[str, Any] = {
            "name": request.tool.name,
            "description": request.tool.description,
            "input_schema": request.tool.schema,
        }
        if request.cache_system:
            tool_def["cache_control"] = {"type": "ephemeral"}
        body["tools"] = [tool_def]
        body["tool_choice"] = {"type": "tool", "name": request.tool.name}
    elif request.tools:
        defs = [{"name": t.name, "description": t.description, "input_schema": t.schema} for t in request.tools]
        if request.cache_system:
            defs[-1]["cache_control"] = {"type": "ephemeral"}
        body["tools"] = defs
        body["tool_choice"] = {"type": "auto"}
    if stream:
        body["stream"] = True
    return body


def parse_anthropic_response(data: dict[str, Any]) -> ChatResult:
    blocks = data.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
    usage = data.get("usage") or {}
    tool_input = next((b.get("input") for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"), None)
    tool_calls = tuple(
        ToolCall(str(b.get("id", "")), str(b.get("name", "")), b.get("input"))
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "tool_use"
    )
    return ChatResult(
        text=text,
        usage=Usage(
            _int(usage.get("input_tokens")),
            _int(usage.get("output_tokens")),
            _int(usage.get("cache_creation_input_tokens")),
            _int(usage.get("cache_read_input_tokens")),
        ),
        stop_reason=data.get("stop_reason"),
        model=str(data.get("model", "")),
        tool_input=tool_input,
        tool_calls=tool_calls,
    )


def anthropic_events(lines: Iterable[str]) -> Iterator[StreamEvent]:
    """Turn the Messages API event stream into text, usage and stop events; raise if it ends early or errors."""
    input_tokens = output_tokens = cache_write = cache_read = 0
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
            start_usage = (payload.get("message") or {}).get("usage") or {}
            input_tokens = _int(start_usage.get("input_tokens"))
            cache_write = _int(start_usage.get("cache_creation_input_tokens"))
            cache_read = _int(start_usage.get("cache_read_input_tokens"))
        elif event == "content_block_delta":
            delta = payload.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                yield StreamEvent("text", text=str(delta["text"]))
        elif event == "message_delta":
            output_tokens = _int((payload.get("usage") or {}).get("output_tokens"))
            stop_reason = (payload.get("delta") or {}).get("stop_reason") or stop_reason
        elif event == "message_stop":
            yield StreamEvent("usage", usage=Usage(input_tokens, output_tokens, cache_write, cache_read))
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
        if request.tool or request.tools:
            raise ProviderError("streaming a tool-calling request is not supported; use complete()")
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
    if request.tool:
        function = {
            "name": request.tool.name,
            "description": request.tool.description,
            "parameters": request.tool.schema,
        }
        body["tools"] = [{"type": "function", "function": function}]
        body["tool_choice"] = {"type": "function", "function": {"name": request.tool.name}}
    elif request.tools:
        body["tools"] = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema}}
            for t in request.tools
        ]
        body["tool_choice"] = "auto"
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def parse_openai_response(data: dict[str, Any]) -> ChatResult:
    choices = data.get("choices") or []
    first = choices[0] if choices else {}
    usage = data.get("usage") or {}
    message = first.get("message") or {}
    text = str(message.get("content") or "")
    tool_input: Any = None
    calls = message.get("tool_calls") or []
    tool_calls: tuple[ToolCall, ...] = ()
    if calls:
        arguments = str((calls[0].get("function") or {}).get("arguments") or "")
        try:
            tool_input = json.loads(arguments)
        except json.JSONDecodeError:
            text = arguments  # keep the raw text so the caller can still try to read or repair it
        parsed_calls = []
        for call in calls:
            function = call.get("function") or {}
            try:
                arguments_value = json.loads(str(function.get("arguments") or ""))
            except json.JSONDecodeError:
                arguments_value = None
            parsed_calls.append(ToolCall(str(call.get("id", "")), str(function.get("name", "")), arguments_value))
        tool_calls = tuple(parsed_calls)
    return ChatResult(
        text=text,
        usage=Usage(_int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens"))),
        stop_reason=first.get("finish_reason"),
        model=str(data.get("model", "")),
        tool_input=tool_input,
        tool_calls=tool_calls,
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
        if request.tool or request.tools:
            raise ProviderError("streaming a tool-calling request is not supported; use complete()")
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


def scripted_reply(replies: list[str]) -> Callable[[ChatRequest], str]:
    """A reply function for FakeProvider that returns the given replies in order (the last one repeats)."""
    state = {"i": 0}

    def reply(request: ChatRequest) -> str:
        text = replies[min(state["i"], len(replies) - 1)]
        state["i"] += 1
        return text

    return reply


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
        if request.tool:  # imitate a native tool call: arguments that parse as JSON arrive as ``tool_input``
            try:
                return ChatResult("", self._usage(request, text), "tool_use", "fake", tool_input=json.loads(text))
            except json.JSONDecodeError:
                pass
        if request.tools:
            # A reply of {"call": "<tool name>", "arguments": {...}} picks a tool, like "auto" tool_choice would;
            # any other reply (including one that fails to parse) means the fake chose not to call anything.
            try:
                chosen = json.loads(text)
                if isinstance(chosen, dict) and chosen.get("call") in {t.name for t in request.tools}:
                    call = ToolCall("call_1", str(chosen["call"]), chosen.get("arguments"))
                    return ChatResult("", self._usage(request, text), "tool_use", "fake", tool_calls=(call,))
            except json.JSONDecodeError:
                pass
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
