"""Token, cost and latency tracking across a run of calls, with a stdlib-only percentile summary.

A tracker wraps a `Chat` rather than changing it, so a `Chat` works tracked or not. Latency is wall-clock time
around the actual provider call; against `FakeProvider` these numbers measure Python overhead, not a model, so
treat the demo's figures as a sanity check of the plumbing, not a benchmark (the usual caveat in this repo: nothing
here is measured against a live provider).
"""

from __future__ import annotations

import math
import time
from collections.abc import Generator
from dataclasses import dataclass, field

from llm_playground.client import Chat
from llm_playground.pricing import Pricing
from llm_playground.types import ChatResult, ToolSpec, Usage


@dataclass(frozen=True)
class Turn:
    usage: Usage
    cost: float | None
    latency: float  # seconds, start of the call to the last byte (or full reply for a non-streamed call)
    time_to_first_token: float | None  # seconds to the first streamed piece; None for a non-streamed call
    stop_reason: str | None


def percentile(values: list[float], p: float) -> float:
    """The p-th percentile (0-100) of ``values`` by nearest-rank; raises on an empty list."""
    if not (0 <= p <= 100):
        raise ValueError("p must be between 0 and 100")
    if not values:
        raise ValueError("no values to take a percentile of")
    ordered = sorted(values)
    # nearest-rank: the smallest value at or above the p-th percent of the (sorted) data
    rank = max(1, math.ceil(round(p / 100 * len(ordered), 9)))
    return ordered[min(rank, len(ordered)) - 1]


@dataclass
class UsageTracker:
    """Records a `Turn` per call and reports totals, cost and latency percentiles across them."""

    pricing: Pricing | None = None
    turns: list[Turn] = field(default_factory=list)

    def track_send(self, chat: Chat, text: str, tool: ToolSpec | None = None) -> ChatResult:
        start = time.perf_counter()
        result = chat.send(text, tool=tool)
        self._record(result, time.perf_counter() - start, time_to_first_token=None)
        return result

    def track_stream(self, chat: Chat, text: str) -> Generator[str, None, None]:
        """Like ``Chat.stream``, but also records latency and time to first token."""
        start = time.perf_counter()
        first: float | None = None
        for piece in chat.stream(text):
            if first is None:
                first = time.perf_counter() - start
            yield piece
        result = chat.last_result
        if result is None:  # the stream raised before any text committed; nothing to record
            return
        self._record(result, time.perf_counter() - start, time_to_first_token=first)

    def _record(self, result: ChatResult, latency: float, time_to_first_token: float | None) -> None:
        cost = self.pricing.cost(result.usage) if self.pricing else None
        self.turns.append(Turn(result.usage, cost, latency, time_to_first_token, result.stop_reason))

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for turn in self.turns:
            total += turn.usage
        return total

    @property
    def total_cost(self) -> float | None:
        if self.pricing is None:
            return None
        return sum(turn.cost or 0.0 for turn in self.turns)

    def latency_percentile(self, p: float) -> float:
        return percentile([t.latency for t in self.turns], p)

    def output_tokens_per_second(self) -> float:
        """Aggregate output tokens over the aggregate latency they took; 0.0 if there is no latency to divide by."""
        total_seconds = sum(t.latency for t in self.turns)
        if total_seconds <= 0:
            return 0.0
        return sum(t.usage.output_tokens for t in self.turns) / total_seconds

    def summary(self) -> str:
        if not self.turns:
            return "no turns recorded"
        usage, n = self.total_usage, len(self.turns)
        lines = [
            f"turns: {n}",
            f"tokens: in={usage.input_tokens} out={usage.output_tokens} total={usage.total}",
            f"latency p50={self.latency_percentile(50):.4f}s p95={self.latency_percentile(95):.4f}s "
            f"max={self.latency_percentile(100):.4f}s",
            f"output tokens/sec (aggregate): {self.output_tokens_per_second():.1f}",
        ]
        ttft = [t.time_to_first_token for t in self.turns if t.time_to_first_token is not None]
        if ttft:
            lines.append(f"time to first token: mean={sum(ttft) / len(ttft):.4f}s over {len(ttft)} streamed turn(s)")
        if self.total_cost is not None:
            lines.append(f"cost: {self.total_cost:.6f}")
        return "\n".join(lines)
