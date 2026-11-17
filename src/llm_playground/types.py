"""Plain data types shared by every provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    """Token counts as reported by the provider (not estimated, except by the offline fake)."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple[Message, ...]
    system: str | None = None
    max_tokens: int = 1024
    temperature: float | None = None


@dataclass(frozen=True)
class ChatResult:
    text: str
    usage: Usage
    stop_reason: str | None
    model: str


@dataclass(frozen=True)
class StreamEvent:
    """One piece of a streamed reply: text, then (once each) usage and a stop reason."""

    kind: Literal["text", "usage", "stop"]
    text: str = ""
    usage: Usage | None = None
    stop_reason: str | None = None
