"""Validación del subconjunto de JSON Schema usado por el bootstrap."""

from __future__ import annotations

import re
from typing import Any

from .bootstrap import PINNED_BOOTSTRAP_BUNDLE_ID, VerifiedBootstrap
from .errors import SchemaValidationError


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "null": value is None,
    }.get(expected, False)


def _json_type(value: Any) -> str:
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
    return f"non-json:{type(value).__name__}"


def _json_equal(left: Any, right: Any) -> bool:
    if _json_type(left) != _json_type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    return left == right


def _validate_json_object_keys(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError(
                    f"{path}: clave JSON de tipo {type(key).__name__}; se requiere string"
                )
            _validate_json_object_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_object_keys(item, f"{path}[{index}]")


def _validate(value: Any, schema: dict, path: str = "$") -> None:
    # JSON objects admit exclusivamente nombres string. Esto debe comprobarse
    # antes de const/enum y de las operaciones de membership de dict de Python.
    _validate_json_object_keys(value, path)
    if "const" in schema and not _json_equal(value, schema["const"]):
        raise SchemaValidationError(f"{path}: se esperaba {schema['const']!r}")
    if "enum" in schema and not any(
        _json_equal(value, candidate) for candidate in schema["enum"]
    ):
        raise SchemaValidationError(f"{path}: valor fuera del enum")

    expected = schema.get("type")
    if expected is not None:
        alternatives = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(value, kind) for kind in alternatives):
            raise SchemaValidationError(
                f"{path}: tipo {type(value).__name__} no permitido; esperado {alternatives}"
            )

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise SchemaValidationError(f"{path}: string demasiado corto")
        pattern = schema.get("pattern")
        if pattern is not None and re.fullmatch(pattern, value) is None:
            raise SchemaValidationError(f"{path}: no satisface el patrón {pattern}")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise SchemaValidationError(f"{path}: faltan campos {sorted(missing)}")
        if schema.get("additionalProperties") is False:
            unknown = set(value) - properties.keys()
            if unknown:
                raise SchemaValidationError(f"{path}: campos desconocidos {sorted(unknown)}")
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key], f"{path}.{key}")

    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{index}]")


def validate_legacy_rule(rule: Any, bootstrap: VerifiedBootstrap) -> None:
    _validate(rule, bootstrap.legacy_rule_schema)
    fields = bootstrap.field_classes.get("fields")
    if not isinstance(fields, dict):
        raise SchemaValidationError("field-classes.fields no es objeto")
    schema_fields = set(bootstrap.legacy_rule_schema.get("properties", {}))
    if set(fields) != schema_fields:
        raise SchemaValidationError("schema y field classification no cubren los mismos campos")


def validate_shadow_manifest(manifest: Any, bootstrap: VerifiedBootstrap) -> None:
    _validate(manifest, bootstrap.manifest_schema)
    if manifest["bootstrap_bundle_id"] != PINNED_BOOTSTRAP_BUNDLE_ID:
        raise SchemaValidationError("manifest apunta a un bootstrap no pinneado")
