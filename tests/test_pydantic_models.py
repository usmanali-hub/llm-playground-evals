import pytest
from pydantic import BaseModel, Field, field_validator

from llm_playground import Chat, FakeProvider, StructuredOutputError, scripted_reply
from llm_playground.pydantic_models import extract_model


class Item(BaseModel):
    name: str
    qty: int = Field(ge=1)


class Invoice(BaseModel):
    vendor: str
    items: list[Item]
    total: float


GOOD = '{"vendor": "Acme", "items": [{"name": "bolt", "qty": 3}], "total": 12.5}'


def chat_with(*replies: str) -> Chat:
    return Chat(FakeProvider(reply=scripted_reply(list(replies))), "fake")


def test_extract_model_returns_a_validated_instance() -> None:
    model, result = extract_model(chat_with(GOOD), "an invoice", Invoice)
    assert isinstance(model, Invoice) and model.vendor == "Acme" and model.items[0].qty == 3
    assert result.attempts == 1


def test_schema_violations_trigger_a_repair() -> None:
    bad = '{"vendor": "Acme", "items": [{"name": "bolt", "qty": 0}], "total": 12.5}'
    model, result = extract_model(chat_with(bad, GOOD), "x", Invoice)
    assert model.items[0].qty == 3 and result.attempts == 2
    assert "below the minimum" in result.errors_by_attempt[0][0]


def test_pydantic_only_validation_also_triggers_a_repair() -> None:
    class Even(BaseModel):
        n: int

        @field_validator("n")
        @classmethod
        def must_be_even(cls, value: int) -> int:
            if value % 2:
                raise ValueError("must be even")
            return value

    odd = '{"n": 3}'  # passes the plain JSON Schema (just "type: integer"), fails the pydantic validator
    even = '{"n": 4}'
    model, result = extract_model(chat_with(odd, even), "x", Even)
    assert model.n == 4 and result.attempts == 2
    assert "must be even" in result.errors_by_attempt[0][0]


def test_gives_up_after_max_repairs_like_extract_structured() -> None:
    with pytest.raises(StructuredOutputError):
        extract_model(chat_with('{"n": 1}', '{"n": 3}', '{"n": 5}'), "x", Invoice, max_repairs=1)


def test_native_mode_works_with_a_pydantic_schema() -> None:
    model, result = extract_model(chat_with(GOOD), "x", Invoice, native=True)
    assert model.vendor == "Acme" and result.attempts == 1


def test_extra_checks_run_alongside_the_pydantic_check() -> None:
    def no_zero_total(data: object) -> list[str]:
        return [] if isinstance(data, dict) and data.get("total") else ["$.total: must not be zero"]

    zero = '{"vendor": "Acme", "items": [{"name": "bolt", "qty": 1}], "total": 0}'
    model, result = extract_model(chat_with(zero, GOOD), "x", Invoice, checks=[no_zero_total])
    assert model.total == 12.5 and result.attempts == 2
