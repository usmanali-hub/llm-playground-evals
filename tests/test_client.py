import pytest

from llm_playground import Chat, FakeProvider, Pricing, ProviderError, Usage
from llm_playground.types import ChatRequest, Message


def test_history_grows_in_pairs_and_the_next_request_carries_it() -> None:
    provider = FakeProvider()
    chat = Chat(provider, "fake", system="Be brief.")
    chat.send("one")
    chat.send("two")
    assert [m.role for m in chat.history] == ["user", "assistant", "user", "assistant"]
    last = provider.requests[-1]
    assert last.system == "Be brief." and [m.content for m in last.messages if m.role == "user"] == [
        "one",
        "two",
    ]


def test_usage_is_totalled_across_turns_and_priced() -> None:
    chat = Chat(FakeProvider(), "fake", pricing=Pricing(3.0, 15.0))
    first = chat.send("hello").usage
    second = chat.send("again").usage
    assert chat.usage == first + second
    expected = (chat.usage.input_tokens * 3.0 + chat.usage.output_tokens * 15.0) / 1_000_000
    assert chat.cost == pytest.approx(expected)
    assert Chat(FakeProvider(), "fake").cost is None


def test_input_tokens_grow_each_turn_because_history_is_resent() -> None:
    chat = Chat(FakeProvider(), "fake")
    inputs = [chat.send("a fairly ordinary question number " + str(i)).usage.input_tokens for i in range(4)]
    assert inputs == sorted(inputs) and inputs[-1] > inputs[0]


def test_stream_yields_pieces_and_commits_once_finished() -> None:
    chat = Chat(FakeProvider(), "fake")
    text = "".join(chat.stream("hello there"))
    assert chat.history[-1] == Message("assistant", text)
    assert chat.last_result is not None and chat.last_result.usage.total > 0


def test_a_dropped_stream_leaves_history_and_usage_unchanged() -> None:
    chat = Chat(FakeProvider(fail_after_chunks=3), "fake")
    with pytest.raises(ProviderError, match="simulated"):
        list(chat.stream("hello there, how are you today my friend"))
    assert chat.history == [] and chat.usage == Usage()


def test_abandoning_a_stream_early_commits_nothing() -> None:
    chat = Chat(FakeProvider(), "fake")
    stream = chat.stream("hello there, how are you today my friend")
    next(stream)
    stream.close()
    assert chat.history == [] and chat.usage == Usage()


def test_failed_send_keeps_roles_alternating_for_the_retry() -> None:
    def boom(request: ChatRequest) -> str:
        raise ProviderError("down")

    chat = Chat(FakeProvider(reply=boom), "fake")
    with pytest.raises(ProviderError):
        chat.send("hi")
    assert chat.history == []


def test_empty_message_and_empty_reply_are_rejected() -> None:
    chat = Chat(FakeProvider(reply=lambda r: "   "), "fake")
    with pytest.raises(ValueError, match="empty"):
        chat.send("   ")
    with pytest.raises(ProviderError, match="empty reply"):
        chat.send("hi")
    assert chat.history == []


def test_negative_prices_are_rejected() -> None:
    with pytest.raises(ValueError):
        Pricing(-1.0, 1.0)


def test_cache_tokens_default_to_the_plain_input_price() -> None:
    pricing = Pricing(input_per_mtok=3.0, output_per_mtok=15.0)
    usage = Usage(input_tokens=100, output_tokens=10, cache_write_tokens=40, cache_read_tokens=20)
    assert pricing.cost(usage) == pytest.approx((100 * 3.0 + 10 * 15.0) / 1_000_000)


def test_cache_prices_reprice_just_the_cache_portion() -> None:
    pricing = Pricing(input_per_mtok=3.0, output_per_mtok=15.0, cache_write_per_mtok=3.75, cache_read_per_mtok=0.3)
    # 100 input tokens total: 40 were a cache write, 20 a cache read, the remaining 40 plain input
    usage = Usage(input_tokens=100, output_tokens=10, cache_write_tokens=40, cache_read_tokens=20)
    expected = (40 * 3.0 + 40 * 3.75 + 20 * 0.3 + 10 * 15.0) / 1_000_000
    assert pricing.cost(usage) == pytest.approx(expected)


def test_negative_cache_prices_are_rejected() -> None:
    with pytest.raises(ValueError):
        Pricing(1.0, 1.0, cache_write_per_mtok=-1.0)
    with pytest.raises(ValueError):
        Pricing(1.0, 1.0, cache_read_per_mtok=-1.0)


def test_cache_system_flows_into_the_request_sent_to_the_provider() -> None:
    provider = FakeProvider()
    chat = Chat(provider, "fake", system="Be brief.", cache_system=True)
    chat.send("hi")
    assert provider.requests[0].cache_system is True
    assert Chat(provider, "fake").cache_system is False
