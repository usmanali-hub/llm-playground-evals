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
    with pytest.raises(SchemaError, match="unsupported schema keyword 'not'"):
        validate(1, {"not": {"type": "integer"}})
    with pytest.raises(SchemaError, match="unsupported schema keyword 'format'"):
        validate("a@b.c", {"type": "string", "format": "email"})
    with pytest.raises(SchemaError, match="only local references"):
        validate({}, {"properties": {"a": {"$ref": "https://example.com/a.json"}}})


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


def test_one_of_needs_exactly_one_match() -> None:
    schema = {"oneOf": [{"type": "integer"}, {"type": "number", "minimum": 0.5}]}
    assert validate("x", schema) == ["$: matches 0 of the 2 allowed shapes, needs exactly one"]
    assert validate(0, schema) == []  # integer only
    assert validate(1.5, schema) == []  # number only
    assert validate(3, schema) == ["$: matches 2 of the 2 allowed shapes, needs exactly one"]  # integer and number


def test_all_of_collects_every_error() -> None:
    schema = {"allOf": [{"type": "object", "required": ["a"]}, {"type": "object", "required": ["b"]}]}
    assert validate({"a": 1, "b": 2}, schema) == []
    assert validate({}, schema) == ["$: missing required field 'a'", "$: missing required field 'b'"]


def test_local_refs_to_defs_and_definitions_are_resolved_with_the_path_of_the_use() -> None:
    address = {
        "type": "object",
        "properties": {"zip": {"type": "string", "pattern": "^[0-9]{5}$"}},
        "required": ["zip"],
    }
    schema = {
        "type": "object",
        "$defs": {"address": address},
        "properties": {"home": {"$ref": "#/$defs/address"}, "work": {"$ref": "#/$defs/address"}},
    }
    assert validate({"home": {"zip": "12345"}, "work": {"zip": "1"}}, schema) == [
        "$.work.zip: does not match pattern '^[0-9]{5}$'"
    ]
    legacy = {"definitions": {"n": {"type": "integer"}}, "type": "array", "items": {"$ref": "#/definitions/n"}}
    assert validate([1, "x"], legacy) == ["$[1]: expected integer, got string"]


def test_recursive_schemas_validate_trees_of_any_finite_depth() -> None:
    tree = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "children": {"type": "array", "items": {"$ref": "#"}}},
        "required": ["name"],
    }
    good = {"name": "a", "children": [{"name": "b", "children": [{"name": "c"}]}]}
    assert validate(good, tree) == []
    bad = {"name": "a", "children": [{"name": "b", "children": [{"name": 3}]}]}
    assert validate(bad, tree) == ["$.children[0].children[0].name: expected string, got integer"]


def test_refs_that_point_nowhere_or_loop_forever_are_schema_errors() -> None:
    with pytest.raises(SchemaError, match="does not point to anything"):
        validate(1, {"$ref": "#/$defs/missing"})
    with pytest.raises(SchemaError, match="too deeply"):
        validate(1, {"$ref": "#"})


def test_ref_with_sibling_keywords_applies_both() -> None:
    schema = {"$defs": {"n": {"type": "integer"}}, "$ref": "#/$defs/n", "minimum": 10}
    assert validate(12, schema) == [] and validate(3, schema) == ["$: 3 is below the minimum 10"]
    assert validate("x", schema) != []


def test_prefix_items_validate_positions_and_items_covers_the_rest() -> None:
    point = {
        "type": "array",
        "prefixItems": [{"type": "number"}, {"type": "number"}, {"enum": ["N", "S"]}],
        "items": {"type": "string"},
    }
    assert validate([1.5, 2, "N"], point) == []
    assert validate([1.5, 2, "N", "note"], point) == []
    assert validate([1.5, "x", "E", 5], point) == [
        "$[1]: expected number, got string",
        "$[2]: must be one of ['N', 'S'], got 'E'",
        "$[3]: expected string, got integer",
    ]


def test_unique_items_treats_one_and_one_point_zero_as_equal_but_one_and_true_as_different() -> None:
    schema = {"type": "array", "uniqueItems": True}
    assert validate([1, 2, 3], schema) == []
    assert validate([1, True, "1"], schema) == []
    assert validate([1, 1.0], schema) == ["$[1]: duplicate of an earlier item"]
    assert validate([{"a": [1, 2]}, {"a": [1, 2]}], schema) == ["$[1]: duplicate of an earlier item"]
    assert validate([{"a": 1, "b": 2}, {"b": 2, "a": 1}], schema) == ["$[1]: duplicate of an earlier item"]


@pytest.mark.parametrize(
    "schema", [{"oneOf": []}, {"allOf": "x"}, {"prefixItems": []}, {"uniqueItems": "yes"}, {"$ref": 5}]
)
def test_new_keywords_are_checked_for_malformed_values(schema: object) -> None:
    with pytest.raises(SchemaError):
        validate({}, schema)
