"""A small JSON Schema validator for checking model output.

It supports the subset that matters for extraction and refuses (raises ``SchemaError``) on any keyword it does not
implement, so an unsupported constraint is never silently ignored. Supported: type (one name or a list), enum, const,
properties, required, additionalProperties (false or a schema), items, prefixItems, uniqueItems, minItems, maxItems,
minLength, maxLength, minimum, maximum, pattern, anyOf, oneOf, allOf, local ``$ref`` ("#" or "#/path/to/subschema")
with ``$defs`` / ``definitions``, plus the annotation keywords title, description, default and examples.
Not supported: remote ``$ref``, ``not``, ``if/then/else``, ``format``, ``patternProperties`` and ``dependentRequired``.
"""

from __future__ import annotations

import json
import re
from typing import Any

ANNOTATIONS = {"title", "description", "default", "examples"}
KEYWORDS = ANNOTATIONS | {
    "type", "enum", "const", "properties", "required", "additionalProperties", "items",
    "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum", "pattern", "anyOf",
    "oneOf", "allOf", "$ref", "$defs", "definitions", "prefixItems", "uniqueItems",
}  # fmt: skip
MAX_REF_DEPTH = 200
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


def resolve_ref(root: Any, ref: str, path: str = "$") -> Any:
    """Follow a local ``$ref`` ("#" or a JSON pointer such as "#/$defs/address") inside the root schema."""
    if not isinstance(ref, str) or not (ref == "#" or ref.startswith("#/")):
        raise SchemaError(f"{path}: only local references starting with '#' are supported, got {ref!r}")
    node = root
    if ref == "#":
        return node
    for part in ref[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and key in node:
            node = node[key]
        elif isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
        else:
            raise SchemaError(f"{path}: reference {ref!r} does not point to anything in the schema")
    return node


def check_schema(schema: Any, path: str = "$", root: Any = None) -> None:
    """Raise SchemaError if the schema is malformed or uses an unsupported keyword."""
    root = schema if root is None else root
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
    for key in ("properties", "$defs", "definitions"):
        if key in schema:
            if not isinstance(schema[key], dict):
                raise SchemaError(f"{path}: {key} must be an object")
            for name, sub in schema[key].items():
                check_schema(sub, f"{path}.{name}", root)
    if "items" in schema:
        check_schema(schema["items"], f"{path}[]", root)
    if "prefixItems" in schema:
        if not isinstance(schema["prefixItems"], list) or not schema["prefixItems"]:
            raise SchemaError(f"{path}: prefixItems must be a non-empty list of schemas")
        for i, sub in enumerate(schema["prefixItems"]):
            check_schema(sub, f"{path}[{i}]", root)
    if "uniqueItems" in schema and not isinstance(schema["uniqueItems"], bool):
        raise SchemaError(f"{path}: uniqueItems must be true or false")
    if "additionalProperties" in schema and schema["additionalProperties"] not in (True, False):
        check_schema(schema["additionalProperties"], f"{path}.*", root)
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            if not isinstance(schema[key], list) or not schema[key]:
                raise SchemaError(f"{path}: {key} must be a non-empty list")
            for i, sub in enumerate(schema[key]):
                check_schema(sub, f"{path}.{key}[{i}]", root)
    if "$ref" in schema:
        resolve_ref(root, schema["$ref"], path)
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
    return _validate(value, schema, "$", schema, 0)


def _canonical(value: Any) -> str:
    """A string that is equal for equal JSON values (1 and 1.0 are the same number; 1 and true are not)."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, list):
        return "[" + ",".join(_canonical(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{json.dumps(k)}:{_canonical(v)}" for k, v in sorted(value.items())) + "}"
    return json.dumps(value)


def _validate(value: Any, schema: Any, path: str, root: Any, depth: int) -> list[str]:
    if depth > MAX_REF_DEPTH:
        raise SchemaError(f"{path}: schema references nest too deeply (a $ref cycle?)")
    if schema is True:
        return []
    if schema is False:
        return [f"{path}: no value is allowed here"]
    errors: list[str] = []
    if "$ref" in schema:
        errors += _validate(value, resolve_ref(root, schema["$ref"], path), path, root, depth + 1)
    declared = schema.get("type")
    if declared is not None:
        names = declared if isinstance(declared, list) else [declared]
        if not any(_matches_type(value, n) for n in names):
            return [f"{path}: expected {' or '.join(names)}, got {json_type(value)}"]
    if "const" in schema and not (schema["const"] == value and json_type(schema["const"]) == json_type(value)):
        errors.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and not any(v == value and json_type(v) == json_type(value) for v in schema["enum"]):
        errors.append(f"{path}: must be one of {schema['enum']!r}, got {value!r}")
    if "anyOf" in schema and not any(not _validate(value, sub, path, root, depth + 1) for sub in schema["anyOf"]):
        errors.append(f"{path}: does not match any of the {len(schema['anyOf'])} allowed shapes")
    if "oneOf" in schema:
        matching = sum(1 for sub in schema["oneOf"] if not _validate(value, sub, path, root, depth + 1))
        if matching != 1:
            errors.append(f"{path}: matches {matching} of the {len(schema['oneOf'])} allowed shapes, needs exactly one")
    for sub in schema.get("allOf", []):
        errors += _validate(value, sub, path, root, depth + 1)
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
        prefix = schema.get("prefixItems", [])
        for i, item in enumerate(value[: len(prefix)]):
            errors += _validate(item, prefix[i], f"{path}[{i}]", root, depth + 1)
        if "items" in schema:
            for i, item in enumerate(value):
                if i >= len(prefix):
                    errors += _validate(item, schema["items"], f"{path}[{i}]", root, depth + 1)
        if schema.get("uniqueItems"):
            seen: set[str] = set()
            for i, item in enumerate(value):
                key = _canonical(item)
                if key in seen:
                    errors.append(f"{path}[{i}]: duplicate of an earlier item")
                seen.add(key)
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required field {name!r}")
        properties = schema.get("properties", {})
        for name, sub in properties.items():
            if name in value:
                errors += _validate(value[name], sub, f"{path}.{name}", root, depth + 1)
        extra = schema.get("additionalProperties", True)
        for name in value:
            if name in properties:
                continue
            if extra is False:
                errors.append(f"{path}: unexpected field {name!r}")
            elif extra is not True:
                errors += _validate(value[name], extra, f"{path}.{name}", root, depth + 1)
    return errors
