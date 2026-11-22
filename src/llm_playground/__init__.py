"""A small LLM client layer: chat, system prompts, streaming, token and cost accounting."""

from llm_playground.checks import grounded_in_text, totals_match
from llm_playground.client import Chat
from llm_playground.evals import EvalCase, EvalReport, EvalResult, run_case, run_suite
from llm_playground.extract import (
    ExtractionError,
    StructuredOutputError,
    StructuredResult,
    extract_json,
    extract_structured,
)
from llm_playground.pricing import Pricing
from llm_playground.promptfile import PromptError, PromptLibrary, PromptVersion, parse_prompt_file
from llm_playground.providers import AnthropicProvider, FakeProvider, OpenAICompatProvider, scripted_reply
from llm_playground.retry import RetryingProvider
from llm_playground.schema import SchemaError, validate
from llm_playground.tools import AgentStep, Tool, ToolError, ToolRegistry, run_agent_step
from llm_playground.tracking import Turn, UsageTracker, percentile
from llm_playground.types import ChatRequest, ChatResult, Message, ProviderError, StreamEvent, ToolCall, ToolSpec, Usage

__all__ = [
    "AgentStep",
    "AnthropicProvider",
    "Chat",
    "ChatRequest",
    "ChatResult",
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "ExtractionError",
    "FakeProvider",
    "Message",
    "OpenAICompatProvider",
    "Pricing",
    "PromptError",
    "PromptLibrary",
    "PromptVersion",
    "ProviderError",
    "RetryingProvider",
    "SchemaError",
    "Tool",
    "ToolCall",
    "ToolError",
    "ToolRegistry",
    "StreamEvent",
    "StructuredOutputError",
    "StructuredResult",
    "Turn",
    "ToolSpec",
    "UsageTracker",
    "Usage",
    "extract_json",
    "extract_structured",
    "grounded_in_text",
    "parse_prompt_file",
    "run_agent_step",
    "run_case",
    "run_suite",
    "percentile",
    "scripted_reply",
    "totals_match",
    "validate",
]
