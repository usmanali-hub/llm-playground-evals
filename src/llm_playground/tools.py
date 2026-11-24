"""Function calling: let the model choose among several tools, run the one(s) it picked, and continue the
conversation with each result as a structured, id-matched tool message (`Chat.send_tool_results`) — the same
message shape a real Anthropic/OpenAI call expects back, not a plain next user turn standing in for it.

Two entry points, in increasing order of capability:

* `run_agent_step` sends the tools with ``tool_choice="auto"`` and dispatches only the model's first tool call
  (`ChatResult.tool_calls` can hold several; this acts on `[0]`), with a single follow-up round that offers no
  tools, so the model cannot chain a second call off the first. Kept for the simplest case and for existing code.
* `run_agent_loop` handles both of those: every tool call in a round (not just the first) is dispatched and
  threaded back together via `Chat.send_tool_results`, and the follow-up offers the same tools again, so the model
  may chain further rounds, up to ``max_rounds``.
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
    the result through `Chat.send_tool_result`, a structured message tied to that call's id. A tool that raises
    does not stop the loop: its error is reported to the model as the "result", the same as a real API's failed
    function call would be, and also returned on `AgentStep`."""
    reply = chat.send(prompt, tools=registry.specs)
    if not reply.tool_calls:
        return AgentStep(reply, called=None)
    call = reply.tool_calls[0]
    try:
        result = registry.dispatch(call)
        follow_up = chat.send_tool_result(call.id, repr(result))
        return AgentStep(reply, called=call, tool_result=result, follow_up=follow_up)
    except ToolError as err:
        follow_up = chat.send_tool_result(call.id, f"error: {err}")
        return AgentStep(reply, called=call, tool_error=str(err), follow_up=follow_up)


@dataclass(frozen=True)
class AgentTurn:
    """One round of `run_agent_loop`: the model's reply and, if it called any tools this round, each call's outcome
    (aligned by position with ``calls``). ``results[i]`` is the tool's return value, or ``None`` if ``errors[i]`` is
    set instead (a `ToolError` message); both are ``None`` for a call that was never dispatched (see below).

    The loop's last `AgentTurn` is the model's final plain-text reply (``calls`` empty), *unless* the model was
    still calling tools when ``max_rounds`` was reached: then the last turn's ``calls`` holds those un-dispatched
    calls with empty ``results``/``errors``, so a caller can tell "the model wanted more tool rounds than we
    allowed" apart from "the model finished".
    """

    reply: ChatResult
    calls: tuple[ToolCall, ...]
    results: tuple[Any, ...]
    errors: tuple[str | None, ...]


def run_agent_loop(chat: Chat, registry: ToolRegistry, prompt: str, max_rounds: int = 4) -> list[AgentTurn]:
    """Like `run_agent_step`, but handles every tool call the model makes in a round (not just the first) and lets
    the model chain further rounds of tool calls (not just one), stopping as soon as a round's reply calls no tool
    or ``max_rounds`` is reached.

    A tool that raises (`ToolError`) does not stop the loop: its message is sent back to the model as that call's
    "result", the same as a real API's failed function call would be, and is also recorded in the returned
    `AgentTurn.errors`.
    """
    turns: list[AgentTurn] = []
    reply = chat.send(prompt, tools=registry.specs)
    for _ in range(max_rounds):
        if not reply.tool_calls:
            turns.append(AgentTurn(reply, (), (), ()))
            return turns
        results: list[tuple[str, str]] = []
        values: list[Any] = []
        errors: list[str | None] = []
        for call in reply.tool_calls:
            try:
                value = registry.dispatch(call)
                values.append(value)
                errors.append(None)
                results.append((call.id, repr(value)))
            except ToolError as err:
                values.append(None)
                errors.append(str(err))
                results.append((call.id, f"error: {err}"))
        turns.append(AgentTurn(reply, reply.tool_calls, tuple(values), tuple(errors)))
        reply = chat.send_tool_results(results, tools=registry.specs)
    turns.append(AgentTurn(reply, reply.tool_calls, (), ()))  # max_rounds reached; these calls were never dispatched
    return turns
