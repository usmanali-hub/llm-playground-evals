"""Two more pieces that were missing from the LLM phase: concurrent batching and Pydantic-validated extraction.

Both run against the offline fake provider, so this demonstrates the plumbing (concurrency actually overlaps,
failures don't stop the batch, a Pydantic-only validator still triggers a repair), not real provider throughput
or a real model's extraction accuracy.
"""

import time

from pydantic import BaseModel, Field

from llm_playground import Chat, FakeProvider, scripted_reply
from llm_playground.batch import run_batch_sync, summarize
from llm_playground.pydantic_models import extract_model


def slow_reply(request: object) -> str:
    time.sleep(0.05)
    return "noted."


print("--- concurrent batch ---")
start = time.perf_counter()
items = run_batch_sync(
    lambda: Chat(FakeProvider(reply=slow_reply), "fake"), [f"prompt {i}" for i in range(8)], concurrency=4
)
print(f"wall clock: {time.perf_counter() - start:.3f}s for 8 calls at concurrency 4 (vs ~0.4s if run one at a time)")
print(summarize(items))


class Invoice(BaseModel):
    vendor: str
    total: float = Field(ge=0.01)  # exclusiveMinimum (gt=) is not a supported schema keyword here


print("\n--- Pydantic-validated extraction ---")
scripted = scripted_reply(['{"vendor": "Acme", "total": -5}', '{"vendor": "Acme", "total": 42.5}'])
chat = Chat(FakeProvider(reply=scripted), "fake")
invoice, result = extract_model(chat, "Acme billed us 42.50.", Invoice)
print(f"{invoice!r} after {result.attempts} attempt(s); first attempt's problems: {result.errors_by_attempt[0]}")
print("(scripted replies; neither demo measures a real model or a real provider's concurrency)")
