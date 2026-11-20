"""Command line: send one prompt to a provider and print the reply with token usage (and cost if priced)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from llm_playground.checks import grounded_in_text
from llm_playground.client import Chat
from llm_playground.extract import StructuredOutputError, extract_structured
from llm_playground.pricing import Pricing
from llm_playground.providers import AnthropicProvider, FakeProvider, OpenAICompatProvider, Provider
from llm_playground.retry import RetryingProvider
from llm_playground.schema import SchemaError
from llm_playground.tracking import UsageTracker
from llm_playground.types import ProviderError

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"


def build_provider(name: str, url: str | None) -> Provider:
    if name == "fake":
        return FakeProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAICompatProvider(url=url) if url else OpenAICompatProvider()
    return OpenAICompatProvider(url=url or OLLAMA_URL, require_key=False)  # "local": Ollama, vLLM or llama.cpp


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm_playground", description=__doc__)
    parser.add_argument("command", choices=["chat", "extract"])
    parser.add_argument("prompt", help="the message to send (chat) or the text to extract from (extract)")
    parser.add_argument("--provider", choices=["fake", "anthropic", "openai", "local"], default="fake")
    parser.add_argument("--model", help="model name; required for every provider except fake")
    parser.add_argument("--system", help="system prompt")
    parser.add_argument("--stream", action="store_true", help="print the reply as it arrives")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--url", help="endpoint for openai-compatible servers (default: Ollama for --provider local)")
    parser.add_argument("--schema", help="extract: path to a JSON Schema file the reply must match")
    parser.add_argument(
        "--repairs", type=int, default=2, help="extract: times to ask the model to fix an invalid reply"
    )
    parser.add_argument("--native", action="store_true", help="extract: force a tool call instead of asking for JSON")
    parser.add_argument(
        "--ground",
        action="append",
        default=[],
        metavar="PATH",
        help="extract: require the value at this path (items[].name for lists) to be in the text; repeatable",
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="extra attempts for overload, rate limit or dropped connection"
    )
    parser.add_argument("--price-in", type=float, help="your price per million input tokens")
    parser.add_argument("--price-out", type=float, help="your price per million output tokens")
    parser.add_argument("--track", action="store_true", help="print latency and throughput alongside usage")
    parser.add_argument(
        "--cache-system", action="store_true", help="mark the system prompt (and tool schema) as cacheable (Anthropic)"
    )
    args = parser.parse_args(argv)

    if args.command == "extract" and not args.schema:
        parser.error("extract needs --schema")
    if args.provider != "fake" and not args.model:
        parser.error("--model is required unless --provider fake")
    if (args.price_in is None) != (args.price_out is None):
        parser.error("give both --price-in and --price-out, or neither")
    pricing = Pricing(args.price_in, args.price_out) if args.price_in is not None else None
    tracker = UsageTracker(pricing=pricing) if args.track else None
    try:
        chat = Chat(
            RetryingProvider(build_provider(args.provider, args.url), max_attempts=args.retries + 1),
            args.model or "fake",
            system=args.system,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            pricing=pricing,
            cache_system=args.cache_system,
        )
        if args.command == "extract":
            with open(args.schema, encoding="utf-8") as handle:
                schema = json.load(handle)
            checks = [grounded_in_text(args.prompt, args.ground)] if args.ground else []
            found = extract_structured(
                chat, args.prompt, schema, max_repairs=args.repairs, native=args.native, checks=checks
            )
            print(json.dumps(found.data, indent=2, ensure_ascii=False))
            print(f"attempts: {found.attempts} (1 means the first reply was already valid)")
        elif args.stream:
            pieces = tracker.track_stream(chat, args.prompt) if tracker else chat.stream(args.prompt)
            for piece in pieces:
                print(piece, end="", flush=True)
            print()
        else:
            if tracker:
                print(tracker.track_send(chat, args.prompt).text)
            else:
                print(chat.send(args.prompt).text)
    except (ProviderError, StructuredOutputError, SchemaError, OSError, json.JSONDecodeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    line = f"tokens: in={chat.usage.input_tokens} out={chat.usage.output_tokens} total={chat.usage.total}"
    if chat.cost is not None:
        line += f" cost={chat.cost:.6f}"
    if args.provider == "fake":
        line += " (estimated by the offline fake, about 4 characters per token)"
    print(line)
    if tracker:
        print(tracker.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
