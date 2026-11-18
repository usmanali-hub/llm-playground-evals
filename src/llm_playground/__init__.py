"""A small LLM client layer: chat, system prompts, streaming, token and cost accounting."""

from llm_playground.checks import grounded_in_text, totals_match
from llm_playground.client import Chat
from llm_playground.extract import (
    ExtractionError,
    StructuredOutputError,
    StructuredResult,
    extract_json,
    extract_structured,
)
from llm_playground.pricing import Pricing
from llm_playground.providers import AnthropicProvider, FakeProvider, OpenAICompatProvider, scripted_reply
from llm_playground.retry import RetryingProvider
from llm_playground.schema import SchemaError, validate
from llm_playground.types import ChatRequest, ChatResult, Message, ProviderError, StreamEvent, ToolSpec, Usage

__all__ = [
    "AnthropicProvider",
    "Chat",
    "ChatRequest",
    "ChatResult",
    "ExtractionError",
    "FakeProvider",
    "Message",
    "OpenAICompatProvider",
    "Pricing",
    "ProviderError",
    "RetryingProvider",
    "SchemaError",
    "StreamEvent",
    "StructuredOutputError",
    "StructuredResult",
    "ToolSpec",
    "Usage",
    "extract_json",
    "extract_structured",
    "grounded_in_text",
    "scripted_reply",
    "totals_match",
    "validate",
]
