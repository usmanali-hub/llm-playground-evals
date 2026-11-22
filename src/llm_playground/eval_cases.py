"""30 eval cases for the extraction-and-validation pipeline (see `llm_playground.evals` for what this does and does
not prove). Run with ``python -m llm_playground eval`` or ``python -m llm_playground.eval_cases``.

Grouped by what they exercise: basic schema shapes, repair-loop behavior, value-level checks (grounded_in_text,
totals_match), native tool-call mode, and deliberately-invalid replies that should fail rather than be waved through.
"""

from __future__ import annotations

from llm_playground.checks import grounded_in_text, totals_match
from llm_playground.evals import EvalCase

INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "total": {"type": "number"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "price": {"type": "number"}},
                "required": ["name", "price"],
            },
        },
    },
    "required": ["vendor", "total", "items"],
}

INVOICE_TEXT = "Acme Supplies invoice: 2 Widgets at 12.50 each and 1 Gadget at 25.00. Total due: 50.00."

PERSON_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
    "required": ["name", "age"],
}

CASES: list[EvalCase] = [
    # --- basic schema shapes, first reply already valid ---
    EvalCase("object_required_fields", "Jane Doe is 34.", PERSON_SCHEMA, ['{"name": "Jane Doe", "age": 34}']),
    EvalCase(
        "nested_array_of_objects",
        INVOICE_TEXT,
        INVOICE_SCHEMA,
        [
            '{"vendor": "Acme Supplies", "total": 50.0, '
            '"items": [{"name": "Widget", "price": 12.5}, {"name": "Gadget", "price": 25.0}]}'
        ],
    ),
    EvalCase(
        "string_enum",
        "The ticket is marked urgent.",
        {
            "type": "object",
            "properties": {"priority": {"enum": ["low", "medium", "high", "urgent"]}},
            "required": ["priority"],
        },
        ['{"priority": "urgent"}'],
    ),
    EvalCase(
        "boolean_and_null",
        "The order has not shipped; no tracking number yet.",
        {
            "type": "object",
            "properties": {"shipped": {"type": "boolean"}, "tracking": {"type": ["string", "null"]}},
            "required": ["shipped", "tracking"],
        },
        ['{"shipped": false, "tracking": null}'],
    ),
    EvalCase(
        "array_of_strings",
        "Tags: urgent, billing, escalated.",
        {
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["tags"],
        },
        ['{"tags": ["urgent", "billing", "escalated"]}'],
    ),
    EvalCase(
        "min_max_items",
        "Pick exactly two winners: Alice and Bob.",
        {
            "type": "object",
            "properties": {"winners": {"type": "array", "minItems": 2, "maxItems": 2}},
            "required": ["winners"],
        },
        ['{"winners": ["Alice", "Bob"]}'],
    ),
    EvalCase(
        "one_of_shape",
        "Payment: credit card ending 4242.",
        {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {"method": {"const": "card"}, "last4": {"type": "string"}},
                    "required": ["method", "last4"],
                },
                {"type": "object", "properties": {"method": {"const": "cash"}}, "required": ["method"]},
            ]
        },
        ['{"method": "card", "last4": "4242"}'],
    ),
    EvalCase(
        "all_of_shape",
        "A contact: name Sam, email sam@example.com.",
        {
            "allOf": [
                {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]},
            ]
        },
        ['{"name": "Sam", "email": "sam@example.com"}'],
    ),
    EvalCase(
        "local_ref_and_defs",
        "Shipping to 221B Baker Street, London.",
        {
            "$defs": {
                "address": {
                    "type": "object",
                    "properties": {"line1": {"type": "string"}, "city": {"type": "string"}},
                    "required": ["line1", "city"],
                }
            },
            "type": "object",
            "properties": {"ship_to": {"$ref": "#/$defs/address"}},
            "required": ["ship_to"],
        },
        ['{"ship_to": {"line1": "221B Baker Street", "city": "London"}}'],
    ),
    EvalCase(
        "prefix_items_tuple",
        "A 3D point: 1, 2, 3.",
        {"type": "array", "prefixItems": [{"type": "number"}, {"type": "number"}, {"type": "number"}]},
        ["[1, 2, 3]"],
    ),
    EvalCase(
        "unique_items",
        "Three distinct IDs: 7, 9, 11.",
        {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
        ["[7, 9, 11]"],
    ),
    # --- the reply is fenced / has surrounding prose, extract_json must still find it ---
    EvalCase(
        "fenced_code_block_reply",
        "Jane Doe is 34.",
        PERSON_SCHEMA,
        ['Sure, here you go:\n```json\n{"name": "Jane Doe", "age": 34}\n```'],
    ),
    EvalCase(
        "prose_wrapped_reply",
        "Jane Doe is 34.",
        PERSON_SCHEMA,
        ['The extracted record is {"name": "Jane Doe", "age": 34} as requested.'],
    ),
    # --- repair loop: first reply is wrong, second reply fixes it ---
    EvalCase(
        "repair_missing_field",
        "Jane Doe is 34.",
        PERSON_SCHEMA,
        ['{"name": "Jane Doe"}', '{"name": "Jane Doe", "age": 34}'],
    ),
    EvalCase(
        "repair_wrong_type",
        "Jane Doe is 34.",
        PERSON_SCHEMA,
        ['{"name": "Jane Doe", "age": "34"}', '{"name": "Jane Doe", "age": 34}'],
    ),
    EvalCase(
        "repair_negative_violates_minimum",
        "Jane Doe is -5.",
        PERSON_SCHEMA,
        ['{"name": "Jane Doe", "age": -5}', '{"name": "Jane Doe", "age": 0}'],
    ),
    EvalCase(
        "repair_two_rounds",
        "Jane Doe is 34.",
        PERSON_SCHEMA,
        ["not json at all", '{"name": "Jane Doe"}', '{"name": "Jane Doe", "age": 34}'],
        max_repairs=2,
    ),
    EvalCase(
        "repair_enum_value_fixed",
        "Marked urgent.",
        {
            "type": "object",
            "properties": {"priority": {"enum": ["low", "medium", "high", "urgent"]}},
            "required": ["priority"],
        },
        ['{"priority": "URGENT"}', '{"priority": "urgent"}'],
    ),
    # --- value-level checks: grounded_in_text ---
    EvalCase(
        "grounded_in_text_pass",
        INVOICE_TEXT,
        INVOICE_SCHEMA,
        [
            '{"vendor": "Acme Supplies", "total": 50.0, '
            '"items": [{"name": "Widget", "price": 12.5}, {"name": "Gadget", "price": 25.0}]}'
        ],
        checks=[grounded_in_text(INVOICE_TEXT, ["vendor", "items[].name"])],
    ),
    EvalCase(
        "grounded_in_text_catches_fabrication_then_repairs",
        INVOICE_TEXT,
        INVOICE_SCHEMA,
        [
            '{"vendor": "Acme Supplies", "total": 50.0, '
            '"items": [{"name": "Thingamajig", "price": 12.5}, {"name": "Gadget", "price": 25.0}]}',
            '{"vendor": "Acme Supplies", "total": 50.0, '
            '"items": [{"name": "Widget", "price": 12.5}, {"name": "Gadget", "price": 25.0}]}',
        ],
        checks=[grounded_in_text(INVOICE_TEXT, ["vendor", "items[].name"])],
    ),
    EvalCase(
        "grounded_in_text_number_forms",
        "Total: 1,000 units at 12.50 each.",
        {
            "type": "object",
            "properties": {"total": {"type": "integer"}, "price": {"type": "number"}},
            "required": ["total", "price"],
        },
        ['{"total": 1000, "price": 12.5}'],
        checks=[grounded_in_text("Total: 1,000 units at 12.50 each.", ["total", "price"])],
    ),
    # --- value-level checks: totals_match ---
    EvalCase(
        "totals_match_pass",
        INVOICE_TEXT,
        INVOICE_SCHEMA,
        [
            '{"vendor": "Acme Supplies", "total": 50.0, '
            '"items": [{"name": "Widget", "price": 12.5}, {"name": "Gadget", "price": 25.0}, '
            '{"name": "Widget", "price": 12.5}]}'
        ],
        checks=[totals_match("items", "price", "total")],
    ),
    EvalCase(
        "totals_match_catches_arithmetic_error_then_repairs",
        INVOICE_TEXT,
        INVOICE_SCHEMA,
        [
            '{"vendor": "Acme Supplies", "total": 99.0, '
            '"items": [{"name": "Widget", "price": 12.5}, {"name": "Gadget", "price": 25.0}]}',
            '{"vendor": "Acme Supplies", "total": 37.5, '
            '"items": [{"name": "Widget", "price": 12.5}, {"name": "Gadget", "price": 25.0}]}',
        ],
        checks=[totals_match("items", "price", "total")],
    ),
    # --- native tool-call mode ---
    EvalCase(
        "native_tool_call_basic",
        "Jane Doe is 34.",
        PERSON_SCHEMA,
        ['{"name": "Jane Doe", "age": 34}'],
        native=True,
    ),
    EvalCase(
        "native_tool_call_with_check",
        INVOICE_TEXT,
        INVOICE_SCHEMA,
        [
            '{"vendor": "Acme Supplies", "total": 50.0, '
            '"items": [{"name": "Widget", "price": 12.5}, {"name": "Gadget", "price": 25.0}]}'
        ],
        checks=[grounded_in_text(INVOICE_TEXT, ["vendor"])],
        native=True,
    ),
    EvalCase(
        "native_tool_call_repairs_bad_value",
        "Jane Doe is 34.",
        PERSON_SCHEMA,
        ['{"name": "Jane Doe", "age": "thirty-four"}', '{"name": "Jane Doe", "age": 34}'],
        native=True,
    ),
    # --- cases that should legitimately fail: proves the pipeline rejects bad output rather than accepting it ---
    EvalCase(
        "exhausts_repairs_on_persistent_garbage",
        "Jane Doe is 34.",
        PERSON_SCHEMA,
        ["not json", "still not json", "nope"],
        expect_valid=False,
        max_repairs=2,
    ),
    EvalCase(
        "exhausts_repairs_on_persistent_schema_violation",
        "Jane Doe is minus five.",
        PERSON_SCHEMA,
        ['{"name": "Jane Doe", "age": -1}', '{"name": "Jane Doe", "age": -2}', '{"name": "Jane Doe", "age": -3}'],
        expect_valid=False,
        max_repairs=2,
    ),
    EvalCase(
        "exhausts_repairs_when_check_never_passes",
        INVOICE_TEXT,
        INVOICE_SCHEMA,
        [
            '{"vendor": "Acme Supplies", "total": 50.0, "items": [{"name": "Ghost Item", "price": 50.0}]}',
            '{"vendor": "Acme Supplies", "total": 50.0, "items": [{"name": "Phantom Good", "price": 50.0}]}',
            '{"vendor": "Acme Supplies", "total": 50.0, "items": [{"name": "Imaginary Thing", "price": 50.0}]}',
        ],
        checks=[grounded_in_text(INVOICE_TEXT, ["items[].name"])],
        expect_valid=False,
        max_repairs=2,
    ),
    EvalCase(
        "missing_required_field_never_supplied",
        "No age given, just a name.",
        PERSON_SCHEMA,
        ['{"name": "Jane Doe"}', '{"name": "Jane Doe"}', '{"name": "Jane Doe"}'],
        expect_valid=False,
        max_repairs=2,
    ),
]
