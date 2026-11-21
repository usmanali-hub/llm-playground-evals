"""Function calling: let the model choose among several tools, run the one it picked, and continue the conversation
with the result.

Scope, stated plainly: this sends the tools with ``tool_choice="auto"`` (the model may call one, or answer in
plain text instead), dispatches the single call it returns through a `ToolRegistry`, and feeds the result back as an
ordinary next turn of `Chat` (``chat.send(f"The {name} tool returned: ...")``). What it does NOT do: thread a
structured, id-matched tool-result message the way the real Anthropic/OpenAI APIs expect (a dedicated content block
tied to the call's id), or handle a model that calls more than one tool in the same turn — `ChatResult.tool_calls`
can hold several, but `run_agent_step` only acts on the first. Those are the natural next layer, not built here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llm_playground.client import Chat
from llm_playground.types import ChatResult, ToolCall, ToolSpec


class ToolError(RuntimeError):
    """A registered tool raised, or the model named a tool that is not registered."""


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    func: Callable[..., Any]  # called as func(**arguments); arguments is whatever the model sent for the tool's schema


@dataclass
class ToolRegistry:
    """Tools the model may be offered, keyed by name."""

    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, spec: ToolSpec, func: Callable[..., Any]) -> None:
        if spec.name in self.tools:
            raise ValueError(f"a tool named {spec.name!r} is already registered")
        self.tools[spec.name] = Tool(spec, func)

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(t.spec for t in self.tools.values())

    def dispatch(self, call: ToolCall) -> Any:
        """Run the tool ``call`` named, with its arguments as keyword arguments. Raises `ToolError`, never the
        tool's own exception type, so a caller can catch one thing regardless of what a tool does internally."""
        tool = self.tools.get(call.name)
        if tool is None:
            raise ToolError(f"model called {call.name!r}, which is not a registered tool")
        arguments = call.arguments if isinstance(call.arguments, dict) else {}
        try:
            return tool.func(**arguments)
        except TypeError as err:
            raise ToolError(f"{call.name}({arguments}) does not match its function's signature: {err}") from err
        except Exception as err:  # the tool's own failure, not ours; always reported as a ToolError
            raise ToolError(f"{call.name}({arguments}) raised {err.__class__.__name__}: {err}") from err


@dataclass(frozen=True)
class AgentStep:
    """What happened on one turn of `run_agent_step`: either a plain reply, or a tool call and its outcome."""

    reply: ChatResult
    called: ToolCall | None
    tool_result: Any = None
    tool_error: str | None = None
    follow_up: ChatResult | None = None  # the model's reply after seeing the tool's result, if a tool was called


def run_agent_step(chat: Chat, registry: ToolRegistry, prompt: str) -> AgentStep:
    """Send ``prompt`` with every tool in ``registry`` on offer; if the model calls one, run it and send the model
    the result as the next turn. A tool that raises does not stop the loop: its error is reported to the model as
    the "result", the same as a real API's failed function call would be, and also returned on `AgentStep`."""
    reply = chat.send(prompt, tools=registry.specs)
    if not reply.tool_calls:
        return AgentStep(reply, called=None)
    call = reply.tool_calls[0]
    try:
        result = registry.dispatch(call)
        follow_up = chat.send(f"The {call.name} tool returned: {result!r}")
        return AgentStep(reply, called=call, tool_result=result, follow_up=follow_up)
    except ToolError as err:
        follow_up = chat.send(f"The {call.name} tool failed: {err}")
        return AgentStep(reply, called=call, tool_error=str(err), follow_up=follow_up)
