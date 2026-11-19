import time

import pytest

from llm_playground import Chat, FakeProvider, Pricing, ProviderError
from llm_playground.tracking import Turn, UsageTracker, percentile
from llm_playground.types import Usage


def test_percentile_examples() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 3.0
    assert percentile(values, 100) == 5.0
    assert percentile([10.0], 0) == percentile([10.0], 100) == 10.0


@pytest.mark.parametrize("p", [-1, 101])
def test_percentile_rejects_out_of_range(p: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1.0], p)


def test_percentile_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no values"):
        percentile([], 50)


def chat_with_delay(delay: float = 0.0) -> Chat:
    def reply(request: object) -> str:
        if delay:
            time.sleep(delay)
        return "ok"

    return Chat(FakeProvider(reply=reply), "fake")


def test_track_send_records_usage_cost_and_latency() -> None:
    tracker = UsageTracker(pricing=Pricing(input_per_mtok=1_000_000, output_per_mtok=1_000_000))
    chat = chat_with_delay()
    result = tracker.track_send(chat, "hello")
    assert result.text == "ok" and len(tracker.turns) == 1
    turn = tracker.turns[0]
    assert turn.usage == result.usage and turn.latency >= 0 and turn.time_to_first_token is None
    assert turn.cost == pytest.approx(turn.usage.input_tokens + turn.usage.output_tokens)


def test_track_send_without_pricing_has_no_cost() -> None:
    tracker = UsageTracker()
    tracker.track_send(chat_with_delay(), "hello")
    assert tracker.turns[0].cost is None and tracker.total_cost is None


def test_totals_sum_across_turns() -> None:
    tracker = UsageTracker()
    chat = chat_with_delay()
    tracker.track_send(chat, "one")
    tracker.track_send(chat, "two is a bit longer")
    assert tracker.total_usage == tracker.turns[0].usage + tracker.turns[1].usage
    assert len(tracker.turns) == 2


def test_track_stream_records_time_to_first_token_and_final_usage() -> None:
    tracker = UsageTracker()
    chat = chat_with_delay()
    pieces = list(tracker.track_stream(chat, "hello"))
    assert "".join(pieces) == "ok" and len(tracker.turns) == 1
    turn = tracker.turns[0]
    assert turn.time_to_first_token is not None and 0 <= turn.time_to_first_token <= turn.latency
    assert turn.usage == chat.last_result.usage  # type: ignore[union-attr]


def test_track_stream_records_nothing_if_the_stream_never_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = chat_with_delay()

    def broken_stream(request: object) -> object:
        raise ProviderError("boom")
        yield  # pragma: no cover - makes this a generator function

    monkeypatch.setattr(chat.provider, "stream", broken_stream)
    tracker = UsageTracker()
    with pytest.raises(ProviderError):
        list(tracker.track_stream(chat, "hello"))
    assert tracker.turns == []


def test_latency_percentile_and_tokens_per_second() -> None:
    tracker = UsageTracker()
    tracker.turns = [Turn(usage=Usage(10, 100), cost=None, latency=1.0, time_to_first_token=None, stop_reason=None)]
    assert tracker.latency_percentile(50) == 1.0
    assert tracker.output_tokens_per_second() == 100.0


def test_output_tokens_per_second_is_zero_with_no_latency() -> None:
    assert UsageTracker().output_tokens_per_second() == 0.0


def test_summary_is_readable_and_empty_case() -> None:
    assert UsageTracker().summary() == "no turns recorded"
    tracker = UsageTracker(pricing=Pricing(1, 1))
    tracker.track_send(chat_with_delay(), "hello")
    summary = tracker.summary()
    assert "turns: 1" in summary and "latency p50=" in summary and "cost:" in summary


def test_summary_mentions_time_to_first_token_only_when_streamed() -> None:
    tracker = UsageTracker()
    tracker.track_send(chat_with_delay(), "hello")
    assert "time to first token" not in tracker.summary()
    list(tracker.track_stream(chat_with_delay(), "hello"))
    assert "time to first token" in tracker.summary()
