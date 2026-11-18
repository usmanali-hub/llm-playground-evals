import pytest

from llm_playground import (
    Chat,
    ExtractionError,
    FakeProvider,
    SchemaError,
    StructuredOutputError,
    extract_json,
    extract_structured,
    scripted_reply,
)
from llm_playground.extract import build_prompt, repair_prompt
from llm_playground.types import ChatRequest

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
    "required": ["name", "age"],
    "additionalProperties": False,
}
GOOD = '{"name": "Aisha", "age": 31}'


def chat_with(*replies: str) -> tuple[Chat, FakeProvider]:
    provider = FakeProvider(reply=scripted_reply(list(replies)))
    return Chat(provider, "fake"), provider


@pytest.mark.parametrize(
    "reply",
    [
        GOOD,
        f"  \n{GOOD}\n ",
        f"```json\n{GOOD}\n```",
        f"```\n{GOOD}\n```",
        f"Sure! Here is the JSON you asked for:\n\n```json\n{GOOD}\n```\nLet me know if you need more.",
        f"The person is described as {GOOD} in the text.",
    ],
)
def test_extract_json_finds_json_in_common_reply_shapes(reply: str) -> None:
    assert extract_json(reply) == {"name": "Aisha", "age": 31}


def test_extract_json_skips_braces_that_are_not_json_and_takes_the_first_value() -> None:
    assert extract_json('Use {curly} braces. Then {"a": 1} and {"b": 2}') == {"a": 1}
    assert extract_json("values: [1, 2, 3]") == [1, 2, 3]


@pytest.mark.parametrize(
    "reply", ["", "no json here", '{"a": 1', "{'a': 1}", "{a: 1}", '{"x": NaN}', '{"x": Infinity}']
)
def test_extract_json_rejects_non_json(reply: str) -> None:
    with pytest.raises(ExtractionError):
        extract_json(reply)


def test_a_valid_first_reply_costs_one_call() -> None:
    chat, provider = chat_with(GOOD)
    result = extract_structured(chat, "Aisha is 31.", SCHEMA)
    assert (result.data, result.attempts, result.errors_by_attempt) == ({"name": "Aisha", "age": 31}, 1, [])
    assert len(provider.requests) == 1 and result.usage == chat.usage
    prompt = provider.requests[0].messages[0].content
    assert "Aisha is 31." in prompt and '"additionalProperties": false' in prompt


def test_an_invalid_reply_is_repaired_with_the_errors_in_the_follow_up() -> None:
    chat, provider = chat_with('{"name": "Aisha", "age": "31"}', GOOD)
    result = extract_structured(chat, "Aisha is 31.", SCHEMA)
    assert result.attempts == 2 and result.data["age"] == 31
    assert result.errors_by_attempt == [["$.age: expected integer, got string"]]
    follow_up = provider.requests[1].messages[-1].content
    assert "$.age: expected integer, got string" in follow_up and "Return only the corrected JSON" in follow_up
    assert len(provider.requests[1].messages) == 3  # the repair continues the same conversation


def test_unparseable_replies_are_repaired_too() -> None:
    chat, _ = chat_with("I cannot do that.", GOOD)
    result = extract_structured(chat, "x", SCHEMA)
    assert result.attempts == 2 and result.errors_by_attempt[0][0].startswith("could not read JSON")


def test_gives_up_after_the_allowed_repairs_with_the_last_errors_and_raw_reply() -> None:
    chat, provider = chat_with('{"name": 5}')
    with pytest.raises(StructuredOutputError) as info:
        extract_structured(chat, "x", SCHEMA, max_repairs=2)
    assert info.value.attempts == 3 and len(provider.requests) == 3 and info.value.raw == '{"name": 5}'
    assert "$.name: expected string, got integer" in info.value.errors


def test_zero_repairs_means_a_single_attempt() -> None:
    chat, provider = chat_with("{}")
    with pytest.raises(StructuredOutputError):
        extract_structured(chat, "x", SCHEMA, max_repairs=0)
    assert len(provider.requests) == 1


def test_a_bad_schema_fails_before_any_model_call() -> None:
    chat, provider = chat_with(GOOD)
    with pytest.raises(SchemaError):
        extract_structured(chat, "x", {"oneOf": []})
    assert provider.requests == []


def test_usage_counts_every_attempt_including_repairs() -> None:
    chat, _ = chat_with('{"name": 5}', '{"name": "A"}', GOOD)
    result = extract_structured(chat, "text", SCHEMA)
    assert result.attempts == 3 and result.usage == chat.usage and result.usage.input_tokens > 0


def test_usage_is_only_this_extraction_when_the_chat_was_used_before() -> None:
    chat, _ = chat_with("hello there", GOOD)
    chat.send("warm-up")
    before = chat.usage
    result = extract_structured(chat, "x", SCHEMA)
    assert result.usage.total == chat.usage.total - before.total


def test_a_reply_cut_off_at_the_token_limit_gets_a_shorter_please() -> None:
    class Truncating(FakeProvider):
        def complete(self, request: ChatRequest):  # type: ignore[no-untyped-def]
            result = super().complete(request)
            if len(self.requests) == 1:
                return type(result)(result.text, result.usage, "max_tokens", result.model)
            return result

    provider = Truncating(reply=scripted_reply(['{"name": "Ais', GOOD]))
    result = extract_structured(Chat(provider, "fake"), "x", SCHEMA)
    assert result.attempts == 2 and "cut off at the token limit" in provider.requests[1].messages[-1].content


def test_long_error_lists_are_truncated_in_the_repair_prompt() -> None:
    errors = [f"$.f{i}: bad" for i in range(25)]
    prompt = repair_prompt(errors, truncated=False)
    assert prompt.count("\n- ") == 11 and "...and 15 more" in prompt


def test_build_prompt_includes_instruction_schema_and_text() -> None:
    prompt = build_prompt("the text", {"type": "string"}, "Do the thing.")
    assert prompt.startswith("Do the thing.") and '"type": "string"' in prompt and prompt.endswith("the text")


def test_value_checks_send_their_problems_back_to_the_model() -> None:
    def adult(data: object) -> list[str]:
        return [] if isinstance(data, dict) and data["age"] >= 18 else ["$.age: must be an adult"]

    chat, provider = chat_with('{"name": "Sam", "age": 9}', GOOD)
    result = extract_structured(chat, "x", SCHEMA, checks=[adult])
    assert result.attempts == 2 and result.data["age"] == 31
    assert "must be an adult" in provider.requests[1].messages[-1].content


def record(calls: list[object], data: object) -> list[str]:
    calls.append(data)
    return []


def test_checks_do_not_run_when_the_schema_already_fails() -> None:
    calls: list[object] = []
    chat, _ = chat_with('{"name": "Sam"}', GOOD)
    extract_structured(chat, "x", SCHEMA, checks=[lambda d: record(calls, d)])
    assert calls == [{"name": "Aisha", "age": 31}]


def test_native_mode_forces_the_tool_and_uses_its_arguments_as_data() -> None:
    chat, provider = chat_with(GOOD)
    result = extract_structured(chat, "Aisha is 31", SCHEMA, native=True, tool_name="record_person")
    tool = provider.requests[0].tool
    assert result.data == {"name": "Aisha", "age": 31} and tool is not None
    assert tool.name == "record_person" and tool.schema == SCHEMA
    assert "Call the record_person tool" in provider.requests[0].messages[-1].content


def test_native_mode_still_validates_and_repairs() -> None:
    chat, _ = chat_with('{"name": "Aisha", "age": -1}', GOOD)
    assert extract_structured(chat, "x", SCHEMA, native=True).attempts == 2


def test_native_mode_needs_an_object_schema() -> None:
    chat, _ = chat_with("[]")
    with pytest.raises(ValueError, match="type 'object'"):
        extract_structured(chat, "x", {"type": "array"}, native=True)
