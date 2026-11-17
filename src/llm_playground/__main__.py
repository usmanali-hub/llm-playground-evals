"""Command line: send one prompt to a provider and print the reply with token usage (and cost if priced)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from llm_playground.client import Chat
from llm_playground.pricing import Pricing
from llm_playground.providers import AnthropicProvider, FakeProvider, OpenAICompatProvider, Provider
from llm_playground.retry import RetryingProvider
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
    parser.add_argument("command", choices=["chat"])
    parser.add_argument("prompt")
    parser.add_argument("--provider", choices=["fake", "anthropic", "openai", "local"], default="fake")
    parser.add_argument("--model", help="model name; required for every provider except fake")
    parser.add_argument("--system", help="system prompt")
    parser.add_argument("--stream", action="store_true", help="print the reply as it arrives")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--url", help="endpoint for openai-compatible servers (default: Ollama for --provider local)")
    parser.add_argument(
        "--retries", type=int, default=3, help="extra attempts for overload, rate limit or dropped connection"
    )
    parser.add_argument("--price-in", type=float, help="your price per million input tokens")
    parser.add_argument("--price-out", type=float, help="your price per million output tokens")
    args = parser.parse_args(argv)

    if args.provider != "fake" and not args.model:
        parser.error("--model is required unless --provider fake")
    if (args.price_in is None) != (args.price_out is None):
        parser.error("give both --price-in and --price-out, or neither")
    pricing = Pricing(args.price_in, args.price_out) if args.price_in is not None else None
    try:
        chat = Chat(
            RetryingProvider(build_provider(args.provider, args.url), max_attempts=args.retries + 1),
            args.model or "fake",
            system=args.system,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            pricing=pricing,
        )
        if args.stream:
            for piece in chat.stream(args.prompt):
                print(piece, end="", flush=True)
            print()
        else:
            print(chat.send(args.prompt).text)
    except ProviderError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    line = f"tokens: in={chat.usage.input_tokens} out={chat.usage.output_tokens} total={chat.usage.total}"
    if chat.cost is not None:
        line += f" cost={chat.cost:.6f}"
    if args.provider == "fake":
        line += " (estimated by the offline fake, about 4 characters per token)"
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
