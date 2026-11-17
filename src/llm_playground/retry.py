"""Retry with exponential backoff and jitter around any provider."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator

from llm_playground.providers import Provider
from llm_playground.types import ChatRequest, ChatResult, ProviderError, StreamEvent


class RetryingProvider:
    """Retry retryable failures (overload, rate limits, dropped connections), never bad requests.

    The wait before attempt ``n`` is a random value up to ``min(max_delay, base_delay * 2**n)`` ("full jitter", so many
    clients do not retry in lockstep). If the server sent ``Retry-After`` that is used instead when it is longer.
    A stream is retried only if it failed before yielding any text: after that, retrying would repeat output the caller
    has already seen, so the error is raised instead.
    """

    def __init__(
        self,
        provider: Provider,
        max_attempts: int = 4,
        base_delay: float = 0.5,
        max_delay: float = 20.0,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._provider = provider
        self._max_attempts = max_attempts
        self._base = base_delay
        self._max = max_delay
        self._sleep = sleep
        self._rng = rng or random.Random()
        self.retries = 0

    def _wait(self, attempt: int, err: ProviderError) -> None:
        delay = self._rng.uniform(0, min(self._max, self._base * 2**attempt))
        if err.retry_after is not None:
            delay = max(delay, min(err.retry_after, self._max))
        self.retries += 1
        self._sleep(delay)

    def complete(self, request: ChatRequest) -> ChatResult:
        for attempt in range(self._max_attempts):
            try:
                return self._provider.complete(request)
            except ProviderError as err:
                if not err.retryable or attempt == self._max_attempts - 1:
                    raise
                self._wait(attempt, err)
        raise AssertionError("unreachable")  # pragma: no cover

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        for attempt in range(self._max_attempts):
            yielded = False
            try:
                for event in self._provider.stream(request):
                    yielded = yielded or event.kind == "text"
                    yield event
                return
            except ProviderError as err:
                if yielded or not err.retryable or attempt == self._max_attempts - 1:
                    raise
                self._wait(attempt, err)
