"""A small JSON Schema validator for checking model output.

It supports the subset that matters for extraction and refuses (raises ``SchemaError``) on any keyword it does not
implement, so an unsupported constraint is never silently ignored. Supported: type (one name or a list), enum, const,
properties, required, additionalProperties (false or a schema), items (one schema), minItems, maxItems, minLength,
maxLength, minimum, maximum, pattern, anyOf, plus the annotation keywords title, description, default and examples.
"""

from __future__ import annotations

import re
from typing import Any

ANNOTATIONS = {"title", "description", "default", "examples"}
KEYWORDS = ANNOTATIONS | {
    "type", "enum", "const", "properties", "required", "additionalProperties", "items",
    "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum", "pattern", "anyOf",
}  # fmt: skip
TYPES = {"string", "integer", "number", "boolean", "null", "array", "object"}


class SchemaError(ValueError):
    """The schema itself is invalid or uses a keyword this validator does not implement."""


def json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def check_schema(schema: Any, path: str = "$") -> None:
    """Raise SchemaError if the schema is malformed or uses an unsupported keyword."""
    if schema is True or schema is False:
        return
    if not isinstance(schema, dict):
        raise SchemaError(f"{path}: a schema must be an object, got {json_type(schema)}")
    for key in schema:
        if key not in KEYWORDS:
            raise SchemaError(f"{path}: unsupported schema keyword {key!r}")
    declared = schema.get("type")
    if declared is not None:
        names = declared if isinstance(declared, list) else [declared]
        if not names or not all(isinstance(n, str) and n in TYPES for n in names):
            raise SchemaError(f"{path}: invalid type {declared!r}")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise SchemaError(f"{path}: enum must be a list")
    if "required" in schema and not (
        isinstance(schema["required"], list) and all(isinstance(r, str) for r in schema["required"])
    ):
        raise SchemaError(f"{path}: required must be a list of strings")
    for key in ("properties",):
        if key in schema:
            if not isinstance(schema[key], dict):
                raise SchemaError(f"{path}: properties must be an object")
            for name, sub in schema[key].items():
                check_schema(sub, f"{path}.{name}")
    if "items" in schema:
        check_schema(schema["items"], f"{path}[]")
    if "additionalProperties" in schema and schema["additionalProperties"] not in (True, False):
        check_schema(schema["additionalProperties"], f"{path}.*")
    if "anyOf" in schema:
        if not isinstance(schema["anyOf"], list) or not schema["anyOf"]:
            raise SchemaError(f"{path}: anyOf must be a non-empty list")
        for i, sub in enumerate(schema["anyOf"]):
            check_schema(sub, f"{path}.anyOf[{i}]")
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except (re.error, TypeError) as err:
            raise SchemaError(f"{path}: invalid pattern: {err}") from err
    for key in ("minItems", "maxItems", "minLength", "maxLength"):
        if key in schema and not (isinstance(schema[key], int) and not isinstance(schema[key], bool)):
            raise SchemaError(f"{path}: {key} must be an integer")
    for key in ("minimum", "maximum"):
        if key in schema and (isinstance(schema[key], bool) or not isinstance(schema[key], (int, float))):
            raise SchemaError(f"{path}: {key} must be a number")


def _matches_type(value: Any, name: str) -> bool:
    actual = json_type(value)
    if name == "number":
        return actual in ("integer", "number")
    if name == "integer":
        return actual == "integer" or (actual == "number" and float(value).is_integer())
    return actual == name


def validate(value: Any, schema: Any) -> list[str]:
    """Return a list of human-readable errors (empty means valid). Raises SchemaError for a bad schema."""
    check_schema(schema)
    return _validate(value, schema, "$")


def _validate(value: Any, schema: Any, path: str) -> list[str]:
    if schema is True:
        return []
    if schema is False:
        return [f"{path}: no value is allowed here"]
    errors: list[str] = []
    declared = schema.get("type")
    if declared is not None:
        names = declared if isinstance(declared, list) else [declared]
        if not any(_matches_type(value, n) for n in names):
            return [f"{path}: expected {' or '.join(names)}, got {json_type(value)}"]
    if "const" in schema and not (schema["const"] == value and json_type(schema["const"]) == json_type(value)):
        errors.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and not any(v == value and json_type(v) == json_type(value) for v in schema["enum"]):
        errors.append(f"{path}: must be one of {schema['enum']!r}, got {value!r}")
    if "anyOf" in schema and not any(not _validate(value, sub, path) for sub in schema["anyOf"]):
        errors.append(f"{path}: does not match any of the {len(schema['anyOf'])} allowed shapes")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match pattern {schema['pattern']!r}")
    if json_type(value) in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} is above the maximum {schema['maximum']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: has {len(value)} items, needs at least {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: has {len(value)} items, allows at most {schema['maxItems']}")
        if "items" in schema:
            for i, item in enumerate(value):
                errors += _validate(item, schema["items"], f"{path}[{i}]")
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required field {name!r}")
        properties = schema.get("properties", {})
        for name, sub in properties.items():
            if name in value:
                errors += _validate(value[name], sub, f"{path}.{name}")
        extra = schema.get("additionalProperties", True)
        for name in value:
            if name in properties:
                continue
            if extra is False:
                errors.append(f"{path}: unexpected field {name!r}")
            elif extra is not True:
                errors += _validate(value[name], extra, f"{path}.{name}")
    return errors
