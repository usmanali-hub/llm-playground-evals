import random
from collections.abc import Iterator

import pytest

from llm_playground import ChatRequest, ChatResult, Message, ProviderError, RetryingProvider, StreamEvent, Usage
from llm_playground.transport import RETRYABLE_STATUS, parse_retry_after

REQUEST = ChatRequest("m", (Message("user", "hi"),))
OK = ChatResult("fine", Usage(1, 1), "end_turn", "m")


class Scripted:
    """Raises the queued errors one call at a time, then succeeds."""

    def __init__(self, errors: list[ProviderError], text_before_error: bool = False) -> None:
        self.errors = errors
        self.calls = 0
        self.text_before_error = text_before_error

    def complete(self, request: ChatRequest) -> ChatResult:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return OK

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        self.calls += 1
        if self.errors:
            if self.text_before_error:
                yield StreamEvent("text", text="partial ")
            raise self.errors.pop(0)
        yield StreamEvent("text", text="fine")
        yield StreamEvent("usage", usage=Usage(1, 1))
        yield StreamEvent("stop", stop_reason="end_turn")


def overloaded() -> ProviderError:
    return ProviderError("HTTP 529", status=529, retryable=True)


def make(provider: Scripted, **kwargs: float) -> tuple[RetryingProvider, list[float]]:
    sleeps: list[float] = []
    return RetryingProvider(provider, sleep=sleeps.append, rng=random.Random(0), **kwargs), sleeps  # type: ignore[arg-type]


def test_retries_a_retryable_error_then_succeeds() -> None:
    provider = Scripted([overloaded(), overloaded()])
    retrying, sleeps = make(provider)
    assert retrying.complete(REQUEST) == OK
    assert provider.calls == 3 and len(sleeps) == 2 and retrying.retries == 2


def test_does_not_retry_a_bad_request() -> None:
    provider = Scripted([ProviderError("HTTP 400: bad", status=400, retryable=False)])
    retrying, sleeps = make(provider)
    with pytest.raises(ProviderError, match="400"):
        retrying.complete(REQUEST)
    assert provider.calls == 1 and sleeps == []


def test_gives_up_after_max_attempts_and_raises_the_last_error() -> None:
    provider = Scripted([overloaded() for _ in range(10)])
    retrying, sleeps = make(provider, max_attempts=3)
    with pytest.raises(ProviderError, match="529"):
        retrying.complete(REQUEST)
    assert provider.calls == 3 and len(sleeps) == 2


def test_backoff_is_jittered_and_capped_by_an_exponential_ceiling() -> None:
    provider = Scripted([overloaded() for _ in range(5)])
    retrying, sleeps = make(provider, max_attempts=6, base_delay=1.0, max_delay=5.0)
    retrying.complete(REQUEST)
    ceilings = [1.0, 2.0, 4.0, 5.0, 5.0]
    assert all(0 <= s <= c for s, c in zip(sleeps, ceilings, strict=True)) and len(set(sleeps)) > 1


def test_retry_after_from_the_server_is_a_floor_for_the_wait() -> None:
    provider = Scripted([ProviderError("HTTP 429", status=429, retryable=True, retry_after=7.0)])
    retrying, sleeps = make(provider, base_delay=0.1)
    retrying.complete(REQUEST)
    assert sleeps == [7.0]


def test_retry_after_is_capped_at_max_delay() -> None:
    provider = Scripted([ProviderError("HTTP 429", status=429, retryable=True, retry_after=900.0)])
    retrying, sleeps = make(provider, max_delay=20.0)
    retrying.complete(REQUEST)
    assert sleeps == [20.0]


def test_a_stream_that_fails_before_any_text_is_retried() -> None:
    provider = Scripted([overloaded()])
    retrying, _ = make(provider)
    events = list(retrying.stream(REQUEST))
    assert "".join(e.text for e in events if e.kind == "text") == "fine" and provider.calls == 2


def test_a_stream_that_fails_after_text_is_not_retried_so_output_is_never_repeated() -> None:
    provider = Scripted([overloaded()], text_before_error=True)
    retrying, sleeps = make(provider)
    seen: list[str] = []
    with pytest.raises(ProviderError):
        for event in retrying.stream(REQUEST):
            seen.append(event.text)
    assert seen == ["partial "] and provider.calls == 1 and sleeps == []


def test_max_attempts_must_be_positive() -> None:
    with pytest.raises(ValueError):
        RetryingProvider(Scripted([]), max_attempts=0)


def test_retry_after_header_parsing() -> None:
    assert parse_retry_after("5") == 5.0 and parse_retry_after("0.5") == 0.5
    assert parse_retry_after(None) is None and parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
    assert parse_retry_after("-3") is None and parse_retry_after("99999") is None


def test_retryable_statuses_cover_overload_and_rate_limits_but_not_client_errors() -> None:
    assert {429, 500, 503, 529} <= RETRYABLE_STATUS and not ({400, 401, 403, 404} & RETRYABLE_STATUS)
