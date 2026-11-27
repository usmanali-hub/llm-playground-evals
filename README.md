# llm-playground-evals

A small, tested layer for talking to chat LLMs: system prompts, multi-turn history, streaming, retries, structured
JSON extraction checked against a schema, token and cost accounting, and a deterministic offline provider so the whole
thing can be tested and demonstrated without an API key.
It has no runtime dependencies (standard library only) and is the base for later experiments in this repo
(structured output, prompt patterns, evaluations).

```bash
pip install -e ".[dev]"
python -m llm_playground chat --provider fake --system "Answer briefly." --stream "What is a token?"
# (offline fake model) [system prompt: Answer briefly.] You said: What is a token?
# tokens: in=8 out=20 total=28 (estimated by the offline fake, about 4 characters per token)
```

The fake is not a language model: it echoes your message so you can see how the plumbing behaves. For a real model:

```bash
export ANTHROPIC_API_KEY=...    # read from the environment, never written to disk or logs
python -m llm_playground chat --provider anthropic --model <model-name> --system "Answer briefly." --stream "What is a token?" \
    --price-in <your $/M input> --price-out <your $/M output>
```

`--provider openai` speaks the OpenAI chat-completions format, and `--provider local` speaks the same format to a local
server (Ollama by default at `localhost:11434`, or any vLLM / llama.cpp server via `--url`) and needs no key.
Model names and prices are never built in: they change, so you pass the ones you use.

## See it work (one command)

```bash
make install   # once
make demo      # a streamed chat reply, a structured-extraction repair demo, and the 30-case eval report
make test      # the full test suite
```

`eval_report.json` at the repo root is the same report `make demo`'s eval step prints, committed so it can be read
without running anything; `make report` regenerates it, and a test (`tests/test_eval_report.py`) fails CI if the
committed file and a fresh run of the suite disagree, so it cannot go stale silently.

## What's in this repo, at a glance

| Piece | What it does | Section |
| --- | --- | --- |
| `Chat` / providers | Multi-turn history, streaming, retries, token/cost accounting, offline fake provider | below |
| `extract_structured` | JSON Schema-checked extraction with a parse/validate/repair loop, native tool mode | [Structured output](#structured-output-json-that-matches-a-schema) |
| `PromptLibrary` | Versioned prompts as files, not strings in code | [Prompt versioning](#prompt-versioning-in-files) |
| `run_agent_step` / `run_agent_loop` | Multi-tool function calling, single-call or multi-round | [Function/tool calling](#functiontool-calling-a-model-choosing-among-several-tools) |
| `UsageTracker` | Per-call latency, time-to-first-token, cost | [Tokens, cost and latency](#tokens-cost-and-latency-tracking) |
| `run_batch` / `extract_model` | Concurrent batching; Pydantic-validated extraction | [Concurrent batching](#concurrent-batching), [Pydantic](#pydantic-validated-extraction) |
| `guarded_send` | Plain-text output guardrails with a repair loop | [Output guardrails](#output-guardrails-with-a-repair-loop) |
| `run_suite` / `gate` | A scored, CI-gatable eval suite over the extraction pipeline | [Eval sets](#eval-sets) |

## Prompt versioning in files

Prompts live as files under `prompts/<name>/v<N>.prompt` instead of strings embedded in code: a small frontmatter
block (`description`, `variables`) plus a body that references each variable as `{{var}}`. `PromptLibrary.load(dir)`
discovers every version of every prompt, `library.get(name)` returns the highest version number (numerically, so
`v10` beats `v2`), and `library.get(name, version=1)` pins an older one. `PromptVersion.render(**values)` fills in
the template and raises `PromptError` naming exactly which variables are missing or weren't declared.

* A new version is a new file (`v2.prompt`), not an edit to `v1.prompt` — nothing is overwritten, so a caller that
  pinned `version=1` keeps working, and the file itself (its `description`) records *why* a new version exists
  (see `prompts/summarize/v2.prompt`: added after v1 produced inconsistent lengths).
* This only parses and renders files. Pairing a rendered prompt with `Chat`/`extract_structured` and recording which
  name/version was used for a given call (a natural fit with `UsageTracker`) is left to the caller.
* `prompt.unused_variables()` / `prompt.undeclared_placeholders()` catch a frontmatter/body mismatch (a declared
  variable the body never uses, or a `{{typo}}` the frontmatter never declared) without raising, since a template
  author may want to check these without attempting a render.

## Function/tool calling: a model choosing among several tools

`ToolSpec`s can now be offered as a free choice (`Chat.send(prompt, tools=(spec_a, spec_b))`, `tool_choice="auto"`
for both providers) instead of only the single forced tool `extract_structured`'s native mode uses. `tools.py`
wraps that into two entry points, in increasing order of capability:

* `run_agent_step(chat, registry, prompt)` dispatches only the model's first tool call and allows one follow-up
  round with no tools offered (so the model cannot chain a second call). The simplest case; kept as-is.
* `run_agent_loop(chat, registry, prompt, max_rounds=4)` dispatches **every** tool call the model makes in a round
  (`ChatResult.tool_calls` can hold several — a model calling more than one tool in the same turn is handled, not
  just `[0]`) and threads them all back together via `Chat.send_tool_results([(id, content), ...])`, offering the
  same tools again so the model may chain further rounds, up to `max_rounds`. It returns a list of `AgentTurn`s;
  the last one is the final plain-text reply, unless the model was still calling tools when `max_rounds` was
  reached — that case is reported distinctly (`AgentTurn.calls` non-empty with empty `results`/`errors`), so a
  caller can tell "the model finished" apart from "we cut it off".

Either way, each tool call's result is threaded back as a structured, id-matched message (an Anthropic `tool_result`
content block, or an OpenAI `role: "tool"` message with `tool_call_id`) — the same shape a real API expects back,
not a plain next user turn standing in for it.

* A tool that raises is caught (`ToolError`) and its message is sent back to the model as the "result", the same
  shape a real failed function call would take, instead of crashing the loop.
* `examples/tool_calling_demo.py` is a scripted fake that always picks the same tool for a weather question; it
  shows the registry/dispatch/follow-up plumbing, not any model's judgement about when a tool is needed.

## Prompt patterns: zero-shot, few-shot, chain of thought

`prompts.py` has pure builders (`zero_shot_prompt`, `few_shot_prompt`, `cot_prompt`), answer parsers
(`parse_final_answer` takes the text after the *last* `Answer:`; `parse_plain_answer` the first line) and
`compare_patterns(provider, model, patterns, cases)`, which runs each pattern on each case in a fresh conversation and
reports correct count, replies that could not be parsed, and output tokens.

* Few-shot examples mostly teach format and labels; their order and balance can bias the answer. Chain of thought asks
  for reasoning before a marked final answer, which costs more output tokens and only sometimes helps.
* The tests check the exact prompt text, parsing edge cases and scoring. `examples/patterns_demo.py` uses a hand-scripted
  model whose answers were chosen in advance, so its table is **not evidence about any real model**. Judging a pattern
  needs a real model, many cases and repeated runs.

## Tokens, cost and latency tracking

`tracking.py`'s `UsageTracker` wraps `Chat.send`/`Chat.stream` (`track_send`/`track_stream`) and records, per call,
token usage, cost (if you pass `Pricing`), wall-clock latency, and for streaming calls the time to the first piece.
`--track` on the CLI prints a summary: totals, cost, p50/p95/max latency, aggregate output tokens/sec, and mean
time-to-first-token across any streamed calls.

* Percentiles use a plain stdlib nearest-rank calculation (`percentile`), not numpy.
* Latency is wall-clock time around the actual provider call. Against `FakeProvider` (`examples/tracking_demo.py`)
  that measures Python/dataclass overhead, not a model's speed; a real latency or throughput number needs a real
  provider and many calls, not three offline ones.

## Prompt caching (Anthropic)

`--cache-system` (or `Chat(..., cache_system=True)`) marks the system prompt, and any forced tool's schema, as a
cache breakpoint (`cache_control: {"type": "ephemeral"}`) in the Anthropic request body. `Usage` separately reports
`cache_write_tokens` and `cache_read_tokens` from the response (`cache_creation_input_tokens` /
`cache_read_input_tokens`); both are already counted once in `input_tokens`. `Pricing` accepts
`cache_write_per_mtok` / `cache_read_per_mtok` to re-price just that portion (a cache write usually costs more than
a plain input token, a cache read much less); leaving them unset prices cached tokens as plain input, which is a
safe but pessimistic default.

* Only useful when the same system prompt (and tool) is reused across calls — a one-off prompt gains nothing from a
  breakpoint, and a short system prompt rarely clears the provider's minimum cacheable size.
* OpenAI-compatible endpoints cache automatically server-side with no request change; there is nothing to set here,
  and the response usage would need a different field (`prompt_tokens_details.cached_tokens`) to report it, which
  is not wired up yet.
* `examples/caching_demo.py` shows the request shape and a pricing comparison with made-up usage numbers; no cache
  was actually written or read, because that requires a live Anthropic account.

## Concurrent batching

`batch.py`'s `run_batch`/`run_batch_sync` send many prompts through `asyncio`, bounded by a `concurrency` limit;
since every provider here is a blocking `urllib` call, concurrency comes from `asyncio.to_thread` (a worker-thread
pool), which still overlaps real network waiting time. Each prompt gets its own `Chat` from a `chat_factory`, so
histories never mix, and a failing call returns a `BatchItem` with its exception instead of stopping the batch; a
test asserts concurrency is actually bounded (peak concurrent calls tracked with a counter), not just "looks async".

## Pydantic-validated extraction

`pydantic_models.py`'s `extract_model(chat, text, Model)` derives the JSON Schema from `Model.model_json_schema()`
(this repo's validator already understands the shapes Pydantic emits: `$defs`/`$ref`, `anyOf` for `Optional`,
numeric bounds) and adds a Pydantic-level check after it, so a `@field_validator` or cross-field rule the schema
subset cannot express still sends a repair prompt instead of crashing. It is optional (`pip install ".[pydantic]"`)
and kept out of the stdlib-only core; import it directly rather than through the package's top-level `__init__`.

* **Known gap**: `Field(gt=...)`/`Field(lt=...)` (and a few other Pydantic/JSON-Schema keywords not in this
  validator's supported set) produce a schema `check_schema` refuses outright, before any model call — use `ge`/`le`
  instead, or validate that part only at the Pydantic level.

## Structured output: JSON that matches a schema

```python
from llm_playground import Chat, extract_structured

result = extract_structured(chat, "Invoice from Acme Tools: 3 bolts, total 12.50 USD.", schema, max_repairs=2)
result.data  # a dict that has been validated against the schema
result.attempts  # 1 means the first reply was already valid
```

```bash
python -m llm_playground extract "Invoice from Acme Tools: 3 bolts, total 12.50 USD." \
    --schema examples/invoice_schema.json --provider anthropic --model <model-name>
python examples/repair_demo.py     # offline: a scripted model makes three mistakes, the loop fixes them
```

How it works: the prompt carries the schema and the text; the reply is parsed (bare JSON, a fenced block, or JSON inside
prose; `NaN` and `Infinity` are rejected); it is validated; if it is invalid, the errors (with their JSON paths, for
example `$.items[0].qty: 0 is below the minimum 1`) go back to the model in the same conversation and it is asked for
corrected JSON, up to `max_repairs` times. A reply cut off at the token limit gets a "keep it shorter" hint. Usage
counts every attempt, so repairs show up in the cost.

* The validator (`schema.py`) implements a documented subset of JSON Schema and **raises on any keyword it does not
  implement** instead of silently ignoring a constraint. It supports `oneOf`, `allOf`, local `$ref`/`$defs` (cycles are
  detected), `prefixItems` and `uniqueItems` besides the basics. Not supported (and refused): remote `$ref`, `not`,
  `if/then/else`, `format`, `patternProperties`, `dependentRequired`.
* Validation proves the *shape* is right, not that the values are true: a model can return a valid object with the wrong
  total. Pass `checks` for value rules: `grounded_in_text(source, paths)` requires extracted strings and numbers to
  appear in the source text (it cannot tell whether a value that is present sits in the right field), and
  `totals_match(...)` compares line items with a total. Their problems go back to the model like schema errors
  (`--ground PATH` on the CLI).
* `native=True` (`--native`) forces a tool call whose arguments are the data, for Anthropic and OpenAI-compatible
  request shapes; the object schema must have `type: object`. The result is still validated and repaired. Streaming a
  forced tool call is refused. The request and response shapes are tested against recorded-shape payloads and a
  localhost server, not a live provider.
* `examples/repair_demo.py` uses hand-written replies that look like common mistakes. It shows how the harness reacts;
  it says nothing about how often a real model makes them.

## Output guardrails with a repair loop

`llm_playground.guardrails` checks a reply against plain-text rules — `max_length`/`min_length`, `no_banned_terms`,
`no_pii_patterns` (email/phone/credit-card-shaped), `must_match` (require a pattern instead of forbidding one) — and
`guarded_send(chat, prompt, guardrails, max_retries=2)` re-prompts the model with the violations listed when a
reply fails any of them, the same repair-loop shape `extract_structured` uses for schema errors, applied to free
text instead of JSON. It raises `GuardrailError` if no attempt ever passes.

```bash
python examples/guardrails_demo.py
# accepted after 2 attempt(s): ...through the in-app contact form for help.
# rejected reply's violations: ['reply appears to contain an email address']
```

* **Scope, stated plainly**: every guardrail here is a regex/length/keyword heuristic, the same class of check as
  `llm_playground.checks`'s `grounded_in_text`/`totals_match` applied to free text — not a safety classifier.
  `no_pii_patterns` catches a shape, not an identity (it cannot tell a real SSN from a made-up example number in
  the same shape, and misses PII in a shape it doesn't recognize), and `no_banned_terms` is a literal substring
  match a paraphrase slips past. Passing every guardrail here is not a claim that a reply is actually safe.
* This complements `RetryingProvider` (below), not replaces it: that retries a *failed call* (overload, rate limit,
  dropped connection); `guarded_send` retries a *reply the provider returned successfully* that failed a content
  check.

## Eval sets

`llm_playground.evals` runs named `EvalCase`s — a source text, a JSON Schema, and a scripted "model" reply — through
`extract_structured` and checks whether the outcome (valid data, or a rejection) matched what the case expected.
`python -m llm_playground eval` runs the 30 cases in `llm_playground.eval_cases` and prints a pass/fail report; they
cover basic schema shapes (objects, arrays, enums, `oneOf`/`allOf`, local `$ref`, tuples, `uniqueItems`), the
repair loop (one and two rounds, native tool-call mode repairing a bad value), the value-level checks
(`grounded_in_text` catching a fabricated field and `totals_match` catching a bad sum, both then repaired), and
cases that are *supposed* to end in failure — exhausted repairs on garbage, a persistent schema violation, a check
that never passes — to prove the pipeline actually rejects bad output instead of waving it through.

* **Scope, stated plainly**: every reply in the suite is a hand-written string played back by `FakeProvider`, not a
  live model's output. This is a regression suite for the extraction pipeline and the prompts that drive it — schema
  validation, the repair loop, the value-level checks, native tool mode — not a benchmark of how well any real model
  follows instructions. The same `EvalCase.schema` and `checks` would make a genuine model eval if run against
  `AnthropicProvider`/`OpenAICompatProvider` instead of the scripted replies; that swap needs a live provider key,
  which this sandbox does not have.

### Automated, scored, and gated for CI

`EvalReport` carries a numeric `pass_rate` (an empty report scores 0.0, not 1.0 — it proves nothing, and must not
look like a clean pass to anything reading that number) and `to_dict()`, the same data as `summary()` but
machine-readable, for a CI step to save as a build artifact or diff against a prior run. `gate(report,
min_pass_rate=1.0)` is the actual "automated" part: a threshold a pipeline can check without a human reading the
summary first.

```bash
python -m llm_playground eval                       # text summary, exits 1 if any case failed
python -m llm_playground eval --json                 # the same report as JSON, for a CI artifact
python -m llm_playground eval --min-pass-rate 0.9    # looser gate: exit 0 if at least 90% of cases passed
```

## How it is put together

```mermaid
flowchart LR
    CLI[CLI / your code] --> Chat[Chat: history, usage totals, cost]
    Chat --> P{Provider}
    P --> A[Anthropic Messages API]
    P --> O[OpenAI-compatible: OpenAI, Ollama, vLLM, llama.cpp]
    P --> F[Offline fake]
    A --> R[Retry: backoff, jitter, Retry-After]
    O --> R
    R --> T[Transport: urllib, the only network code]
    T --> S[SSE parser -> text / usage / stop events]
```

* **`Chat`** keeps the conversation. A turn is added to the history only when the reply finishes, so a failed call or an
  abandoned stream leaves roles alternating and the retry clean. Usage is summed across turns; because every turn resends
  the history, input tokens grow each turn (a test shows it).
* **Providers** build the request (Anthropic puts `system` at the top level, OpenAI makes it the first message) and parse
  the response with pure functions, so parsing is tested against recorded-shape payloads instead of the network.
* **Streaming** goes through one small server-sent-events parser. A stream that ends before the final event
  (`message_stop` or `[DONE]`) raises instead of returning a truncated answer, and provider error events become errors.
* **Retries** (`RetryingProvider`, on by default in the CLI with `--retries`) wait with exponential backoff and full jitter, honour
  a server's `Retry-After`, and retry only errors marked retryable (overload, rate limits, 5xx, dropped connections), never
  a bad request. A stream is retried only if it failed before producing any text, so output is never repeated.
* **Errors** never contain credentials: header values are scrubbed from error text and it is truncated.

## What is and is not verified

* 250 tests cover the SSE parser, both request builders, response and stream parsing (dropped streams, error events,
  invalid JSON), history and usage handling, cost arithmetic, missing keys, retry and backoff rules, the schema validator,
  extraction and repair, the eval harness and its 30-case suite, and the CLI. CI runs
  ruff, strict mypy, the tests and the offline demo.
* **HTTP tests use a real server on localhost.** `tests/test_http_end_to_end.py` sends the actual urllib requests to a local
  server that replays recorded-shape payloads, so the request headers and body, status handling, `Retry-After`, streaming
  over a socket, key redaction, and recovery from an overloaded server run for real, just not against a provider.
* **Nothing here has been run against a live provider.** The payload shapes in `tests/fakes.py` follow the providers'
  documented formats; they are not captures from live calls, so a provider changing its format would not be caught
  until you run it with a real key.
* Token counts from the offline fake are estimates (about 4 characters per token). Real providers report exact counts,
  which is what `Usage` records.
* Not built yet: measured local-model serving (vLLM/Ollama/llama.cpp with real weights — unavailable in this sandbox).

## Layout

```
src/llm_playground/
  types.py       Message, Usage, ChatRequest, ChatResult, StreamEvent, ProviderError
  sse.py         server-sent-events parser
  transport.py   urllib HTTP, error scrubbing (the only network code)
  providers.py   Anthropic, OpenAI-compatible, offline fake; pure parsing functions
  pricing.py     cost from token counts and your prices
  retry.py       RetryingProvider: backoff, jitter, Retry-After
  schema.py      JSON Schema subset validator (refuses unsupported keywords)
  extract.py     extract_json, extract_structured: parse, validate, repair
  client.py      Chat: history, usage totals, failure-safe streaming
examples/        invoice schema and an offline repair demo
tests/           parser, provider, retry, client, CLI and localhost HTTP tests
```
