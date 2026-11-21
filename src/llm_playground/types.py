"""Plain data types shared by every provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["user", "assistant"]


class ProviderError(RuntimeError):
    """A provider call failed. The message never contains credentials."""

    def __init__(
        self,
        message: str,
        status: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable  # worth trying again (overload, rate limit, dropped connection), not a bad request
        self.retry_after = retry_after  # seconds the server asked us to wait, if it said


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Usage:
    """Token counts as reported by the provider (not estimated, except by the offline fake).

    ``cache_write_tokens`` and ``cache_read_tokens`` are reported separately only by providers that support prompt
    caching (currently Anthropic). A cache write usually costs more than a plain input token and a cache read much
    less; both are already counted once in ``input_tokens``, so adding them in again would double-count.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ToolSpec:
    """A function the model is forced to call; its arguments are the structured output (``schema`` is a JSON Schema)."""

    name: str
    description: str
    schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """One function the model asked to invoke, with an id so its result can be matched back to this call."""

    id: str
    name: str
    arguments: Any


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple[Message, ...]
    system: str | None = None
    max_tokens: int = 1024
    temperature: float | None = None
    tool: ToolSpec | None = None  # a single tool the model is FORCED to call (used by extract_structured's native mode)
    tools: tuple[ToolSpec, ...] = ()  # several tools the model may freely choose among (or call none)
    cache_system: bool = False  # mark the system prompt (and tool schema, if any) as a cache breakpoint

    def __post_init__(self) -> None:
        if self.tool is not None and self.tools:
            raise ValueError("pass either `tool` (forced) or `tools` (free choice), not both")


@dataclass(frozen=True)
class ChatResult:
    text: str
    usage: Usage
    stop_reason: str | None
    model: str
    tool_input: Any = None  # the arguments of a forced tool call, already parsed, when the request had a tool
    tool_calls: tuple[ToolCall, ...] = ()  # the tools the model chose to call, when the request offered `tools`


@dataclass(frozen=True)
class StreamEvent:
    """One piece of a streamed reply: text, then (once each) usage and a stop reason."""

    kind: Literal["text", "usage", "stop"]
    text: str = ""
    usage: Usage | None = None
    stop_reason: str | None = None
