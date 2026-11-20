"""Shows the shape of a cache-aware request and how Pricing separates cache-write and cache-read cost.

No live call is made (the fake provider does not actually cache anything); this demonstrates the request body
Anthropic would receive with --cache-system and how to read the usage it would report back.
"""

from llm_playground import Chat, FakeProvider, Pricing, Usage
from llm_playground.providers import anthropic_body

chat = Chat(FakeProvider(), "fake", system="You are a careful, concise assistant." * 50, cache_system=True)
request = chat._request("Summarise the attached document.")  # noqa: SLF001 - inspecting the request shape, not sending it
body = anthropic_body(request, stream=False)
print("system block carries a cache breakpoint:", body["system"][0]["cache_control"])

# A realistic usage report after a cache has been written once and read on later turns:
first_turn = Usage(input_tokens=2000, output_tokens=50, cache_write_tokens=1900, cache_read_tokens=0)
later_turn = Usage(input_tokens=2000, output_tokens=50, cache_write_tokens=0, cache_read_tokens=1900)
pricing = Pricing(input_per_mtok=3.0, output_per_mtok=15.0, cache_write_per_mtok=3.75, cache_read_per_mtok=0.3)
print(f"first turn (cache write):  cost={pricing.cost(first_turn):.6f}")
print(f"later turn (cache read):   cost={pricing.cost(later_turn):.6f}")
print("(illustrative numbers; nothing here was measured against a live provider)")
