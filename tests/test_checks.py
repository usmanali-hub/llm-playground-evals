import pytest

from llm_playground.checks import grounded_in_text, number_forms, totals_match

TEXT = "Invoice from  Acme   Corp\nWidget x3 at 12.50 each. Total: 1,037.50 due 2026-11-30."


def test_strings_match_case_insensitively_with_whitespace_collapsed() -> None:
    assert grounded_in_text(TEXT, ["vendor"])({"vendor": "acme corp"}) == []


def test_a_string_not_in_the_text_is_reported_with_its_path() -> None:
    problems = grounded_in_text(TEXT, ["vendor"])({"vendor": "Globex"})
    assert problems == ["$.vendor: 'Globex' does not appear in the text"]


@pytest.mark.parametrize("value", [12.5, 12.50, 1037.5, 3])
def test_numbers_match_the_ways_they_are_written(value: float) -> None:
    assert grounded_in_text(TEXT, ["n"])({"n": value}) == []


@pytest.mark.parametrize(("source", "value"), [("price 12.50", 12), ("room 1012", 12), ("pi is 3.14159", 3.14)])
def test_numbers_are_not_matched_inside_longer_numbers(source: str, value: float) -> None:
    assert grounded_in_text(source, ["n"])({"n": value}) != []


def test_wildcards_missing_fields_and_non_text_values() -> None:
    check = grounded_in_text(TEXT, ["items[].name", "absent", "flag"])
    data = {"items": [{"name": "widget"}, {"name": "gadget"}], "flag": True}
    assert check(data) == ["$.items[1].name: 'gadget' does not appear in the text"]


def test_number_forms() -> None:
    assert {"12.5", "12.50"} <= number_forms(12.5)
    assert {"1000", "1,000", "1000.00"} <= number_forms(1000)


@pytest.mark.parametrize(
    ("data", "ok"),
    [
        ({"items": [{"amount": 12.5}, {"amount": 1}], "total": 13.5}, True),
        ({"items": [{"amount": 12.5}], "total": 13.0}, False),
        ({"items": [{"amount": "x"}], "total": 1}, True),  # malformed item: the schema reports it, not this check
    ],
)
def test_totals_match(data: dict[str, object], ok: bool) -> None:
    assert (totals_match("items", "amount", "total")(data) == []) is ok


def test_totals_message() -> None:
    msg = totals_match("items", "amount", "total")({"items": [{"amount": 12.5}], "total": 13.0})
    assert msg == ["$.total: 13.0 does not equal the sum of items[].amount (12.5)"]
