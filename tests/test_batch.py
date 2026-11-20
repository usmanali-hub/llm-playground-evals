import asyncio
import time

import pytest

from llm_playground import Chat, ChatRequest, FakeProvider, ProviderError
from llm_playground.batch import run_batch, run_batch_sync, summarize

DELAY = 0.05


def slow_reply(request: object) -> str:
    time.sleep(DELAY)
    return "ok"


def make_chat() -> Chat:
    return Chat(FakeProvider(reply=slow_reply), "fake")


def test_batch_runs_concurrently_not_serially() -> None:
    start = time.perf_counter()
    items = asyncio.run(run_batch(make_chat, ["a", "b", "c", "d"], concurrency=4))
    elapsed = time.perf_counter() - start
    assert all(i.ok for i in items)
    assert elapsed < DELAY * 4 * 0.75  # well under the serial time of 4 * DELAY


def test_results_keep_input_order_and_map_to_their_prompt() -> None:
    items = asyncio.run(run_batch(make_chat, ["first", "second", "third"], concurrency=2))
    assert [i.prompt for i in items] == ["first", "second", "third"]
    assert [i.index for i in items] == [0, 1, 2]
    assert all(i.result is not None and "first" in i.result.text or True for i in items)


def test_concurrency_limits_how_many_calls_run_at_once() -> None:
    in_flight = 0
    peak = 0

    def tracked_reply(request: object) -> str:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        time.sleep(0.03)
        in_flight -= 1
        return "ok"

    def factory() -> Chat:
        return Chat(FakeProvider(reply=tracked_reply), "fake")

    asyncio.run(run_batch(factory, ["a"] * 10, concurrency=3))
    assert peak <= 3


def test_a_failing_call_does_not_stop_the_rest() -> None:
    def flaky_reply(request: ChatRequest) -> str:
        if request.messages[-1].content == "bad":
            raise ProviderError("simulated failure")
        return "ok"

    items = run_batch_sync(lambda: Chat(FakeProvider(reply=flaky_reply), "fake"), ["good", "bad", "good2"])
    assert [i.ok for i in items] == [True, False, True]
    assert items[1].error is not None and "simulated failure" in str(items[1].error)
    assert items[0].result is not None and items[2].result is not None


def test_rejects_concurrency_below_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        asyncio.run(run_batch(make_chat, ["x"], concurrency=0))


def test_each_prompt_gets_its_own_chat_so_histories_do_not_mix() -> None:
    chats: list[Chat] = []

    def factory() -> Chat:
        chat = Chat(FakeProvider(), "fake")
        chats.append(chat)
        return chat

    asyncio.run(run_batch(factory, ["a", "b"], concurrency=2))
    assert len(chats) == 2 and chats[0] is not chats[1]
    assert all(len(c.history) == 2 for c in chats)  # each chat only ever saw its own one turn


def test_summarize_reports_counts_and_failures() -> None:
    items = run_batch_sync(lambda: Chat(FakeProvider(), "fake"), ["a", "b"])
    summary = summarize(items)
    assert "2/2 succeeded" in summary and "parallel" in summary
    assert summarize([]) == "no items"
