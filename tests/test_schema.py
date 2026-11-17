import pytest

from llm_playground import SchemaError, validate

INVOICE = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string", "minLength": 1},
        "total": {"type": "number", "minimum": 0},
        "currency": {"enum": ["USD", "EUR", "PKR"]},
        "items": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "qty": {"type": "integer", "minimum": 1}},
                "required": ["name", "qty"],
                "additionalProperties": False,
            },
        },
        "paid": {"type": ["boolean", "null"]},
    },
    "required": ["vendor", "total", "currency", "items"],
}
GOOD = {"vendor": "Acme", "total": 12.5, "currency": "USD", "items": [{"name": "bolt", "qty": 3}], "paid": None}


def test_a_valid_document_has_no_errors() -> None:
    assert validate(GOOD, INVOICE) == []


def test_errors_name_the_path_of_each_problem() -> None:
    bad = {"vendor": "", "total": "12.5", "currency": "GBP", "items": [{"name": "bolt", "qty": 0, "color": "red"}, 7]}
    errors = validate(bad, INVOICE)
    assert "$.vendor: shorter than 1 characters" in errors
    assert "$.total: expected number, got string" in errors
    assert any(e.startswith("$.currency: must be one of") for e in errors)
    assert "$.items[0].qty: 0 is below the minimum 1" in errors
    assert "$.items[0]: unexpected field 'color'" in errors
    assert "$.items[1]: expected object, got integer" in errors


def test_missing_required_fields_are_reported_each() -> None:
    assert validate({}, INVOICE) == [
        f"$: missing required field {n!r}" for n in ("vendor", "total", "currency", "items")
    ]


def test_booleans_are_not_numbers_and_integral_floats_are_integers() -> None:
    assert validate(True, {"type": "integer"}) == ["$: expected integer, got boolean"]
    assert validate(True, {"type": "number"}) == ["$: expected number, got boolean"]
    assert validate(3.0, {"type": "integer"}) == []
    assert validate(3.5, {"type": "integer"}) == ["$: expected integer, got number"]
    assert validate(3, {"type": "number"}) == []


def test_enum_and_const_do_not_confuse_true_with_one() -> None:
    assert validate(True, {"enum": [1]}) != []
    assert validate(1, {"enum": [True]}) != []
    assert validate(1, {"const": 1}) == [] and validate(True, {"const": 1}) != []


def test_type_lists_any_of_and_pattern() -> None:
    assert validate(None, {"type": ["string", "null"]}) == []
    assert validate(5, {"type": ["string", "null"]}) == ["$: expected string or null, got integer"]
    schema = {"anyOf": [{"type": "string"}, {"type": "integer", "minimum": 10}]}
    assert validate("x", schema) == [] and validate(12, schema) == [] and validate(3, schema) != []
    assert validate("2026-11-18", {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}) == []
    assert validate("18/11/2026", {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}) != []


def test_additional_properties_can_be_a_schema() -> None:
    schema = {"type": "object", "properties": {"id": {"type": "integer"}}, "additionalProperties": {"type": "string"}}
    assert validate({"id": 1, "note": "ok"}, schema) == []
    assert validate({"id": 1, "note": 5}, schema) == ["$.note: expected string, got integer"]


def test_array_bounds() -> None:
    schema = {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "integer"}}
    assert validate([1], schema) == ["$: has 1 items, needs at least 2"]
    assert validate([1, 2, 3, 4], schema) == ["$: has 4 items, allows at most 3"]


def test_unsupported_keywords_are_refused_not_ignored() -> None:
    with pytest.raises(SchemaError, match="unsupported schema keyword 'oneOf'"):
        validate(1, {"oneOf": [{"type": "integer"}]})
    with pytest.raises(SchemaError, match="unsupported schema keyword '\\$ref'"):
        validate({}, {"properties": {"a": {"$ref": "#/defs/a"}}})


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "datetime"},
        {"type": []},
        {"required": "name"},
        {"pattern": "("},
        {"minItems": "2"},
        {"minimum": True},
        5,
        {"anyOf": []},
    ],
)
def test_malformed_schemas_raise_schema_error(schema: object) -> None:
    with pytest.raises(SchemaError):
        validate({}, schema)


def test_true_and_false_schemas() -> None:
    assert validate({"anything": 1}, True) == []
    assert validate(1, False) == ["$: no value is allowed here"]
