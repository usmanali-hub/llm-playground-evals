"""Extract into a Pydantic model instead of a raw JSON Schema.

``model.model_json_schema()`` already produces schemas this repo's validator understands (properties, $defs/$ref,
anyOf for ``Optional``, numeric bounds), so this is a thin wrapper around ``extract_structured``: it derives the
schema, adds a check that re-validates with Pydantic itself (catching anything the JSON Schema subset cannot
express, such as a custom ``@field_validator``), and returns a parsed model instance alongside the usual result.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from llm_playground.client import Chat
from llm_playground.extract import Check, StructuredResult, extract_structured

ModelT = TypeVar("ModelT", bound=BaseModel)


def _pydantic_check(model: type[BaseModel]) -> Check:
    def check(data: object) -> list[str]:
        try:
            model.model_validate(data)
            return []
        except ValidationError as err:
            problems = []
            for error in err.errors():
                where = "$" + "".join(f".{p}" if isinstance(p, str) else f"[{p}]" for p in error["loc"])
                problems.append(f"{where}: {error['msg']}")
            return problems

    return check


def extract_model(
    chat: Chat,
    text: str,
    model: type[ModelT],
    instruction: str = "Extract the requested information from the text below.",
    max_repairs: int = 2,
    native: bool = False,
    checks: Sequence[Check] = (),
) -> tuple[ModelT, StructuredResult]:
    """Extract data matching ``model``'s schema and return a validated instance plus the raw `StructuredResult`.

    A Pydantic-level check runs after the JSON Schema check, so a model with validation the schema subset cannot
    express (a cross-field ``@model_validator``, a regex via ``constr``) still gets repair prompts, not a crash.
    """
    result = extract_structured(
        chat,
        text,
        model.model_json_schema(),
        instruction=instruction,
        max_repairs=max_repairs,
        native=native,
        checks=(*checks, _pydantic_check(model)),
        tool_name=model.__name__.lower(),
    )
    return model.model_validate(result.data), result
