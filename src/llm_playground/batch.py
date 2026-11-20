"""Run many Chat calls concurrently with asyncio, bounded by a semaphore, over the stdlib-only sync providers.

Providers here use urllib (blocking), so real concurrency comes from running each call in a worker thread via
`asyncio.to_thread` rather than an async HTTP client; this still gets wall-clock overlap for I/O-bound calls (a real
provider spends most of its time waiting on the network) without adding an async HTTP dependency. One failing call
does not stop the batch: each result is either the `ChatResult` or the exception that call raised, kept in input
order, so correlating prompts to outcomes does not need sorting afterward.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from llm_playground.client import Chat
from llm_playground.types import ChatResult, ProviderError, ToolSpec


@dataclass(frozen=True)
class BatchItem:
    index: int
    prompt: str
    result: ChatResult | None
    error: BaseException | None
    latency: float

    @property
    def ok(self) -> bool:
        return self.error is None


async def run_batch(
    chat_factory: Callable[[], Chat],
    prompts: Sequence[str],
    concurrency: int = 5,
    tool: ToolSpec | None = None,
) -> list[BatchItem]:
    """Send every prompt in its own `Chat` (so histories don't mix), at most `concurrency` calls in flight.

    `chat_factory` builds a fresh `Chat` per prompt; share one only if the provider itself is safe to call from
    several threads at once (the bundled providers are, since each call opens its own connection).
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int, prompt: str) -> BatchItem:
        async with semaphore:
            start = time.perf_counter()
            try:
                result = await asyncio.to_thread(chat_factory().send, prompt, tool)
                return BatchItem(index, prompt, result, None, time.perf_counter() - start)
            except (ProviderError, ValueError) as err:
                return BatchItem(index, prompt, None, err, time.perf_counter() - start)

    items = await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
    return list(items)


def run_batch_sync(
    chat_factory: Callable[[], Chat], prompts: Sequence[str], concurrency: int = 5, tool: ToolSpec | None = None
) -> list[BatchItem]:
    """Synchronous convenience wrapper for callers (like the CLI) that are not already in an event loop."""
    return asyncio.run(run_batch(chat_factory, prompts, concurrency, tool))


def summarize(items: Sequence[BatchItem]) -> str:
    if not items:
        return "no items"
    ok = [i for i in items if i.ok]
    failed = [i for i in items if not i.ok]
    lines = [f"{len(ok)}/{len(items)} succeeded"]
    if ok:
        total_latency = sum(i.latency for i in ok)
        lines.append(f"wall time of the batch would be ~{max(i.latency for i in ok):.4f}s if fully parallel")
        lines.append(f"serial time would have been ~{total_latency:.4f}s (sum of each call's own latency)")
    for item in failed:
        lines.append(f"  [{item.index}] {item.prompt[:40]!r}: {item.error}")
    return "\n".join(lines)
