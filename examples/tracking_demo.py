"""Runs a short burst of calls through the offline fake provider with UsageTracker, then prints the summary.

Latency here measures Python/dataclass overhead around the fake provider, not a model: a real measurement needs a
real provider and many more calls. This only shows that the plumbing (percentiles, cost, time to first token)
works end to end.
"""

from llm_playground import Chat, FakeProvider, Pricing, UsageTracker

tracker = UsageTracker(pricing=Pricing(input_per_mtok=3.0, output_per_mtok=15.0))
chat = Chat(FakeProvider(), "fake")

for prompt in ["What is a token?", "Explain top-p sampling in one sentence.", "Name three uses of embeddings."]:
    tracker.track_send(chat, prompt)

chat2 = Chat(FakeProvider(), "fake")
list(tracker.track_stream(chat2, "Stream this reply piece by piece."))

print(tracker.summary())
print("(fake-provider latencies measure Python overhead, not any real model's speed)")
