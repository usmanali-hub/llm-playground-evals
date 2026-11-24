import json

import pytest

from llm_playground import Chat, FakeProvider, ToolSpec
from llm_playground.tools import AgentStep, ToolError, ToolRegistry, run_agent_loop, run_agent_step

WEATHER = ToolSpec("get_weather", "Look up the weather for a city.", {"type": "object", "properties": {"city": {}}})
CALC = ToolSpec("add", "Add two numbers.", {"type": "object", "properties": {"a": {}, "b": {}}})


def registry_with_weather_and_add() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WEATHER, lambda city: f"sunny in {city}")
    registry.register(CALC, lambda a, b: a + b)
    return registry


def scripted_choice(choice: dict[str, object] | None):  # type: ignore[no-untyped-def]
    def reply(request):  # type: ignore[no-untyped-def]
        if request.tools:
            return json.dumps(choice) if choice else "no tool needed, here's a plain answer"
        return "the tool's result has been noted"

    return reply


def test_model_calling_no_tool_returns_a_plain_reply() -> None:
    chat = Chat(FakeProvider(reply=scripted_choice(None)), "fake")
    step = run_agent_step(chat, registry_with_weather_and_add(), "what's 2+2 colloquially?")
    assert step.called is None and step.follow_up is None
    assert "plain answer" in step.reply.text


def test_model_calling_a_tool_runs_it_and_sends_the_result_back() -> None:
    chat = Chat(
        FakeProvider(reply=scripted_choice({"call": "get_weather", "arguments": {"city": "Kuwait City"}})), "fake"
    )
    step = run_agent_step(chat, registry_with_weather_and_add(), "weather in Kuwait City?")
    assert step.called is not None and step.called.name == "get_weather"
    assert step.tool_result == "sunny in Kuwait City" and step.tool_error is None
    assert step.follow_up is not None and "noted" in step.follow_up.text


def test_registry_rejects_a_duplicate_name() -> None:
    registry = ToolRegistry()
    registry.register(WEATHER, lambda city: city)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(WEATHER, lambda city: city)


def test_dispatch_raises_tool_error_for_an_unregistered_name() -> None:
    from llm_playground.types import ToolCall

    registry = ToolRegistry()
    with pytest.raises(ToolError, match="not a registered tool"):
        registry.dispatch(ToolCall("1", "ghost", {}))


def test_dispatch_wraps_a_signature_mismatch_as_tool_error() -> None:
    from llm_playground.types import ToolCall

    registry = ToolRegistry()
    registry.register(CALC, lambda a, b: a + b)
    with pytest.raises(ToolError, match="does not match its function's signature"):
        registry.dispatch(ToolCall("1", "add", {"a": 1}))


def test_dispatch_wraps_the_tool_functions_own_exception() -> None:
    from llm_playground.types import ToolCall

    registry = ToolRegistry()

    def boom(x: int) -> int:
        raise ZeroDivisionError("nope")

    registry.register(ToolSpec("boom", "fails", {"type": "object", "properties": {"x": {}}}), boom)
    with pytest.raises(ToolError, match="ZeroDivisionError: nope"):
        registry.dispatch(ToolCall("1", "boom", {"x": 1}))


def test_a_failing_tool_still_gets_a_follow_up_reporting_the_error() -> None:
    def boom(city: str) -> str:
        raise RuntimeError("service down")

    registry = ToolRegistry()
    registry.register(WEATHER, boom)
    chat = Chat(FakeProvider(reply=scripted_choice({"call": "get_weather", "arguments": {"city": "X"}})), "fake")
    step = run_agent_step(chat, registry, "weather?")
    assert step.tool_error is not None and "service down" in step.tool_error
    assert step.follow_up is not None and "tool failed" not in step.reply.text  # the error went to the model, not us


def test_run_agent_step_reports_a_call_to_an_unoffered_tool_without_raising() -> None:
    # The fake "model" names a tool that was never in the registry it was offered (not possible for a real
    # provider, but worth defending against a provider bug or a mismatched registry/offer).
    def names_an_unoffered_tool(request):  # type: ignore[no-untyped-def]
        return '{"call": "ghost_tool", "arguments": {}}'

    registry = ToolRegistry()
    registry.register(CALC, lambda a, b: a + b)
    chat = Chat(FakeProvider(reply=names_an_unoffered_tool), "fake")
    step = run_agent_step(chat, registry, "do something")
    assert step.called is None and step.tool_error is None  # the fake never offered ghost_tool, so it wasn't parsed


def test_agent_step_is_frozen_and_reports_no_tool_called_as_none() -> None:
    step = AgentStep(reply=None, called=None)  # type: ignore[arg-type]
    assert step.called is None and step.tool_result is None and step.follow_up is None


def test_run_agent_loop_dispatches_every_tool_call_in_one_round() -> None:
    def picks_both_tools(request):  # type: ignore[no-untyped-def]
        if any(m.role == "tool" for m in request.messages):
            return "weather and sum reported"
        return json.dumps(
            {
                "calls": [
                    {"call": "get_weather", "arguments": {"city": "Kuwait City"}},
                    {"call": "add", "arguments": {"a": 2, "b": 3}},
                ]
            }
        )

    chat = Chat(FakeProvider(reply=picks_both_tools), "fake")
    turns = run_agent_loop(chat, registry_with_weather_and_add(), "weather and sum?")
    assert len(turns) == 2
    first, last = turns
    assert len(first.calls) == 2
    assert first.results == ("sunny in Kuwait City", 5)
    assert first.errors == (None, None)
    assert last.calls == () and last.reply.text == "weather and sum reported"


def test_run_agent_loop_chains_rounds_until_a_plain_reply() -> None:
    rounds = {"n": 0}

    def chains_then_answers(request):  # type: ignore[no-untyped-def]
        if any(m.role == "tool" for m in request.messages):
            rounds["n"] += 1
            if rounds["n"] < 2:
                return json.dumps({"call": "add", "arguments": {"a": 1, "b": 1}})
            return "final answer: 2"
        return json.dumps({"call": "add", "arguments": {"a": 1, "b": 1}})

    chat = Chat(FakeProvider(reply=chains_then_answers), "fake")
    turns = run_agent_loop(chat, registry_with_weather_and_add(), "chain a few calls", max_rounds=5)
    assert len(turns) == 3  # two tool rounds, then the final plain-text turn
    assert all(t.calls for t in turns[:2])
    assert turns[-1].calls == () and turns[-1].reply.text == "final answer: 2"


def test_run_agent_loop_reports_still_calling_tools_when_max_rounds_is_hit() -> None:
    def always_calls_a_tool(request):  # type: ignore[no-untyped-def]
        return json.dumps({"call": "add", "arguments": {"a": 1, "b": 1}})

    chat = Chat(FakeProvider(reply=always_calls_a_tool), "fake")
    turns = run_agent_loop(chat, registry_with_weather_and_add(), "never stops", max_rounds=2)
    assert len(turns) == 3  # 2 dispatched rounds + 1 final turn reporting the still-pending call
    assert turns[-1].calls and turns[-1].results == () and turns[-1].errors == ()


def test_run_agent_loop_records_a_failing_tool_without_stopping_the_round() -> None:
    def boom(city: str) -> str:
        raise RuntimeError("service down")

    broken = ToolRegistry()
    broken.register(WEATHER, boom)
    broken.register(CALC, lambda a, b: a + b)

    def picks_both_tools(request):  # type: ignore[no-untyped-def]
        if any(m.role == "tool" for m in request.messages):
            return "done"
        return json.dumps(
            {
                "calls": [
                    {"call": "get_weather", "arguments": {"city": "X"}},
                    {"call": "add", "arguments": {"a": 2, "b": 3}},
                ]
            }
        )

    chat = Chat(FakeProvider(reply=picks_both_tools), "fake")
    turns = run_agent_loop(chat, broken, "weather and sum?")
    first = turns[0]
    assert first.errors[0] is not None and "service down" in first.errors[0]
    assert first.results[0] is None
    assert first.errors[1] is None and first.results[1] == 5
