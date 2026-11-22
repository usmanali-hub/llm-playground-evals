"""A conversation that keeps its history, totals its token usage and stays consistent when a call fails."""

from __future__ import annotations

import json
from collections.abc import Generator

from llm_playground.pricing import Pricing
from llm_playground.providers import Provider
from llm_playground.types import ChatRequest, ChatResult, Message, ProviderError, ToolSpec, Usage


class Chat:
    def __init__(
        self,
        provider: Provider,
        model: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        pricing: Pricing | None = None,
        cache_system: bool = False,
    ) -> None:
        self.provider = provider
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.pricing = pricing
        self.cache_system = cache_system
        self.history: list[Message] = []
        self.usage = Usage()
        self.last_result: ChatResult | None = None

    def _request(self, text: str, tool: ToolSpec | None = None, tools: tuple[ToolSpec, ...] = ()) -> ChatRequest:
        if not text.strip():
            raise ValueError("message is empty")
        messages = (*self.history, Message("user", text))
        return ChatRequest(
            self.model,
            messages,
            self.system,
            self.max_tokens,
            self.temperature,
            tool,
            tools,
            cache_system=self.cache_system,
        )

    def _commit(self, text: str, result: ChatResult) -> None:
        # History only grows with complete user/assistant pairs, so roles keep alternating after a failure.
        # A freely-chosen tool call (ChatResult.tool_calls) is kept as a proper assistant message carrying
        # tool_calls, matching the structured shape a real API expects, so send_tool_result can thread a
        # correctly id-matched result back to it. A forced tool call (the native-extraction `tool` field) has no
        # id to match against, so its arguments are still kept as JSON text, same as before.
        if result.tool_calls:
            self.history += [Message("user", text), Message("assistant", result.text, tool_calls=result.tool_calls)]
            self.usage += result.usage
            self.last_result = result
            return
        if result.text.strip():
            reply = result.text
        elif result.tool_input is not None:
            reply = json.dumps(result.tool_input)
        else:
            reply = ""
        if not reply.strip():
            raise ProviderError("the model returned an empty reply")
        self.history += [Message("user", text), Message("assistant", reply)]
        self.usage += result.usage
        self.last_result = result

    def send(self, text: str, tool: ToolSpec | None = None, tools: tuple[ToolSpec, ...] = ()) -> ChatResult:
        result = self.provider.complete(self._request(text, tool, tools))
        self._commit(text, result)
        return result

    def send_tool_result(self, tool_call_id: str, content: str) -> ChatResult:
        """Continue the conversation after a tool call, with its result threaded as a structured, id-matched
        message (the shape a real API expects), not a plain next user turn.

        Must follow a `send(..., tools=...)` call whose result had `tool_calls`; that call's assistant message
        (carrying the matching `tool_calls`) must still be the last thing in history. No new tools are offered on
        this follow-up call, so the model's reply here is always plain text, never another tool call.
        """
        if not self.history or self.history[-1].role != "assistant" or not self.history[-1].tool_calls:
            raise ValueError("send_tool_result must follow a send(...) call whose result had tool_calls")
        self.history.append(Message("tool", content, tool_call_id=tool_call_id))
        request = ChatRequest(
            self.model,
            tuple(self.history),
            self.system,
            self.max_tokens,
            self.temperature,
            cache_system=self.cache_system,
        )
        result = self.provider.complete(request)
        if not result.text.strip():
            raise ProviderError("the model returned an empty reply")
        self.history.append(Message("assistant", result.text))
        self.usage += result.usage
        self.last_result = result
        return result

    def stream(self, text: str) -> Generator[str, None, None]:
        """Yield the reply piece by piece. Nothing is added to the history unless the stream finishes normally."""
        request = self._request(text)
        pieces: list[str] = []
        usage = Usage()
        stop_reason: str | None = None
        for event in self.provider.stream(request):
            if event.kind == "text":
                pieces.append(event.text)
                yield event.text
            elif event.kind == "usage" and event.usage is not None:
                usage = event.usage
            elif event.kind == "stop":
                stop_reason = event.stop_reason
        self._commit(text, ChatResult("".join(pieces), usage, stop_reason, self.model))

    @property
    def cost(self) -> float | None:
        return self.pricing.cost(self.usage) if self.pricing else None
