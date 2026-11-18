import pytest

from llm_playground.providers import (
    AnthropicProvider,
    FakeProvider,
    OpenAICompatProvider,
    anthropic_body,
    estimate_tokens,
    openai_body,
)
from llm_playground.transport import scrub
from llm_playground.types import ChatRequest, Message, ProviderError, ToolSpec, Usage
from tests.fakes import ANTHROPIC_STREAM, OPENAI_STREAM, FakeTransport

REQUEST = ChatRequest("some-model", (Message("user", "Hi"),), system="Be brief.", max_tokens=50, temperature=0.2)


def test_anthropic_body_puts_system_at_top_level_and_omits_unset_fields() -> None:
    body = anthropic_body(REQUEST, stream=False)
    assert body["system"] == "Be brief." and body["messages"] == [{"role": "user", "content": "Hi"}]
    assert body["temperature"] == 0.2 and "stream" not in body
    bare = anthropic_body(ChatRequest("m", (Message("user", "x"),)), stream=True)
    assert "system" not in bare and "temperature" not in bare and bare["stream"] is True


def test_openai_body_puts_system_first_and_asks_for_stream_usage() -> None:
    body = openai_body(REQUEST, stream=True)
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}
    assert body["stream_options"] == {"include_usage": True}
    assert "stream_options" not in openai_body(REQUEST, stream=False)


def test_anthropic_complete_parses_text_blocks_usage_and_stop_reason() -> None:
    reply = {
        "model": "some-model",
        "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " there"}],
        "usage": {"input_tokens": 12, "output_tokens": 4},
        "stop_reason": "end_turn",
    }
    transport = FakeTransport(json_reply=reply)
    result = AnthropicProvider(api_key="sk-test-key-12345", transport=transport).complete(REQUEST)
    assert (result.text, result.usage, result.stop_reason) == ("Hello there", Usage(12, 4), "end_turn")
    url, headers, _ = transport.calls[0]
    assert url.endswith("/v1/messages") and headers["x-api-key"] == "sk-test-key-12345"
    assert headers["anthropic-version"]


def test_anthropic_stream_yields_text_then_usage_then_stop() -> None:
    provider = AnthropicProvider(api_key="sk-test-key-12345", transport=FakeTransport(stream_lines=ANTHROPIC_STREAM))
    events = list(provider.stream(REQUEST))
    assert "".join(e.text for e in events if e.kind == "text") == "Hello"
    assert events[-2].usage == Usage(25, 15) and events[-1].stop_reason == "end_turn"


def test_anthropic_stream_that_drops_before_message_stop_raises() -> None:
    provider = AnthropicProvider(
        api_key="sk-test-key-12345", transport=FakeTransport(stream_lines=ANTHROPIC_STREAM[:5])
    )
    with pytest.raises(ProviderError, match="message_stop"):
        list(provider.stream(REQUEST))


def test_anthropic_stream_error_event_raises_with_the_message() -> None:
    lines = ['event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n']
    provider = AnthropicProvider(api_key="sk-test-key-12345", transport=FakeTransport(stream_lines=lines))
    with pytest.raises(ProviderError, match="overloaded_error: Overloaded"):
        list(provider.stream(REQUEST))


def test_openai_complete_and_stream() -> None:
    reply = {
        "model": "m",
        "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 2},
    }
    provider = OpenAICompatProvider(api_key="sk-test-key-12345", transport=FakeTransport(json_reply=reply))
    assert provider.complete(REQUEST).usage == Usage(9, 2)
    streamed = OpenAICompatProvider(api_key="sk-test-key-12345", transport=FakeTransport(stream_lines=OPENAI_STREAM))
    events = list(streamed.stream(REQUEST))
    assert "".join(e.text for e in events if e.kind == "text") == "Hello"
    assert events[-2].usage == Usage(9, 2) and events[-1].stop_reason == "stop"


def test_openai_stream_without_done_raises() -> None:
    provider = OpenAICompatProvider(api_key="k" * 10, transport=FakeTransport(stream_lines=OPENAI_STREAM[:3]))
    with pytest.raises(ProviderError, match="DONE"):
        list(provider.stream(REQUEST))


def test_invalid_json_in_stream_is_a_provider_error() -> None:
    provider = OpenAICompatProvider(api_key="k" * 10, transport=FakeTransport(stream_lines=["data: {not json\n\n"]))
    with pytest.raises(ProviderError, match="invalid JSON"):
        list(provider.stream(REQUEST))


def test_missing_keys_fail_before_any_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider()
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAICompatProvider()
    OpenAICompatProvider(require_key=False, url="http://localhost:11434/v1/chat/completions")  # local servers need none


def test_scrub_removes_credentials_and_truncates() -> None:
    headers = {"x-api-key": "sk-secret-value-123"}
    cleaned = scrub("bad key sk-secret-value-123 rejected " + "x" * 1000, headers)
    assert "sk-secret-value-123" not in cleaned and "[redacted]" in cleaned and len(cleaned) <= 300


def test_fake_provider_is_deterministic_and_streams_the_same_text() -> None:
    fake = FakeProvider()
    request = ChatRequest("fake", (Message("user", "hello world"),), system="Be brief.")
    full = fake.complete(request)
    streamed = "".join(e.text for e in fake.stream(request) if e.kind == "text")
    assert streamed == full.text == fake.complete(request).text
    assert full.usage.input_tokens == estimate_tokens("Be brief.hello world")


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0 and estimate_tokens("abcd") == 1 and estimate_tokens("abcde") == 2


def test_overload_error_events_are_retryable_but_other_stream_errors_are_not() -> None:
    def failing(kind: str) -> ProviderError:
        lines = [f'event: error\ndata: {{"type":"error","error":{{"type":"{kind}","message":"x"}}}}\n\n']
        provider = AnthropicProvider(api_key="sk-test-key-12345", transport=FakeTransport(stream_lines=lines))
        with pytest.raises(ProviderError) as info:
            list(provider.stream(REQUEST))
        return info.value

    assert failing("overloaded_error").retryable and failing("rate_limit_error").retryable
    assert not failing("invalid_request_error").retryable


def test_a_dropped_stream_is_marked_retryable() -> None:
    provider = AnthropicProvider(
        api_key="sk-test-key-12345", transport=FakeTransport(stream_lines=ANTHROPIC_STREAM[:5])
    )
    with pytest.raises(ProviderError) as info:
        list(provider.stream(REQUEST))
    assert info.value.retryable


TOOL = ToolSpec("record", "Record it.", {"type": "object", "properties": {"n": {"type": "integer"}}})
TOOL_REQUEST = ChatRequest("m", (Message("user", "hi"),), tool=TOOL)


def test_anthropic_tool_request_and_response() -> None:
    body = anthropic_body(TOOL_REQUEST, stream=False)
    assert body["tools"] == [{"name": "record", "description": "Record it.", "input_schema": TOOL.schema}]
    assert body["tool_choice"] == {"type": "tool", "name": "record"}
    reply = {
        "content": [{"type": "tool_use", "name": "record", "input": {"n": 4}}],
        "usage": {},
        "stop_reason": "tool_use",
    }
    result = AnthropicProvider(api_key="sk-test-key-12345", transport=FakeTransport(json_reply=reply)).complete(
        TOOL_REQUEST
    )
    assert result.tool_input == {"n": 4} and result.text == ""


def test_openai_tool_request_and_response() -> None:
    body = openai_body(TOOL_REQUEST, stream=False)
    assert body["tools"][0]["function"] == {"name": "record", "description": "Record it.", "parameters": TOOL.schema}
    assert body["tool_choice"] == {"type": "function", "function": {"name": "record"}}
    call = {"function": {"name": "record", "arguments": '{"n": 4}'}}
    reply = {"choices": [{"message": {"content": None, "tool_calls": [call]}, "finish_reason": "tool_calls"}]}
    result = OpenAICompatProvider(api_key="k" * 12, transport=FakeTransport(json_reply=reply)).complete(TOOL_REQUEST)
    assert result.tool_input == {"n": 4}


def test_openai_non_json_arguments_come_back_as_text() -> None:
    call = {"function": {"name": "record", "arguments": "{n: 4"}}
    reply = {"choices": [{"message": {"content": None, "tool_calls": [call]}, "finish_reason": "length"}]}
    result = OpenAICompatProvider(api_key="k" * 12, transport=FakeTransport(json_reply=reply)).complete(TOOL_REQUEST)
    assert result.tool_input is None and result.text == "{n: 4"


@pytest.mark.parametrize("cls", [AnthropicProvider, OpenAICompatProvider])
def test_streaming_a_forced_tool_call_is_refused(cls: type) -> None:
    provider = cls(api_key="k" * 12, transport=FakeTransport())
    with pytest.raises(ProviderError, match="streaming a forced tool call"):
        list(provider.stream(TOOL_REQUEST))
