"""A model choosing among two tools (get_weather, add), one call run and its result sent back as the next turn.

Scripted: the fake "model" always picks get_weather for a weather question, and plain-answers otherwise. This shows
the registry/dispatch/follow-up plumbing, not a real model's judgement about when to call a tool.
"""

import json

from llm_playground import Chat, FakeProvider, ToolSpec
from llm_playground.tools import ToolRegistry, run_agent_step

WEATHER = ToolSpec(
    "get_weather", "Look up today's weather for a city.", {"type": "object", "properties": {"city": {"type": "string"}}}
)
ADD = ToolSpec(
    "add", "Add two numbers.", {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}
)

registry = ToolRegistry()
registry.register(WEATHER, lambda city: f"72F and sunny in {city}")
registry.register(ADD, lambda a, b: a + b)


def scripted_model(request: object) -> str:  # type: ignore[no-untyped-def]
    prompt = request.messages[-1].content
    if request.tools and "weather" in prompt.lower():
        return json.dumps({"call": "get_weather", "arguments": {"city": "Kuwait City"}})
    if request.tools:
        return "I can answer that directly without a tool."
    return f"Noted — {prompt}"


chat = Chat(FakeProvider(reply=scripted_model), "fake")
step = run_agent_step(chat, registry, "What's the weather in Kuwait City?")
print(f"model called: {step.called.name if step.called else None}")
print(f"tool result: {step.tool_result!r}")
print(f"follow-up reply: {step.follow_up.text if step.follow_up else None!r}")

no_tool_step = run_agent_step(chat, registry, "What is the capital of Kuwait?")
print(f"\nno-tool case — model called: {no_tool_step.called}, reply: {no_tool_step.reply.text!r}")
print("(scripted, not a real model's decision about when a tool is needed)")
