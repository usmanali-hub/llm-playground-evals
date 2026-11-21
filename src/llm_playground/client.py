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
        # A forced tool call has no text; its arguments (or a freely-chosen tool call) are kept in the history as
        # JSON text so roles stay plain, even though that is not the structured shape a real API would send back.
        if result.text.strip():
            reply = result.text
        elif result.tool_input is not None:
            reply = json.dumps(result.tool_input)
        elif result.tool_calls:
            reply = json.dumps([{"name": c.name, "arguments": c.arguments} for c in result.tool_calls])
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
