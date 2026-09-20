"""Canonical projections and domain-separated hashes for Block 6."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Mapping
from policy.authority_ledger.canonical import domain_hash
from policy.canonicalization.canonical_json import canonical_json_bytes

def utc_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("UTC datetime requerido")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def canonical_value(value: Any) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float) or isinstance(value, bytes):
        raise ValueError("float/bytes prohibidos en contexto de ejecución")
    if isinstance(value, (tuple, list)):
        return [canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("claves de contexto deben ser strings")
        return {key: canonical_value(value[key]) for key in sorted(value)}
    raise ValueError("tipo de contexto no canónico")

def execution_request_hash(projection: Mapping[str, Any]) -> str:
    return domain_hash("JAX-EXECUTION-REQUEST", "1.0", projection)

def execution_authorization_hash(projection: Mapping[str, Any]) -> str:
    return domain_hash("JAX-EXECUTION-AUTHORIZATION", "1.0", projection)

def execution_record_hash(projection: Mapping[str, Any]) -> str:
    return domain_hash("JAX-GOVERNED-EXECUTION-RECORD", "1.0", projection)

def parameters_hash(prompt: str, context: Mapping[str, Any]) -> str:
    return domain_hash("JAX-EXECUTION-PARAMETERS", "1.0", {"prompt": prompt, "context": canonical_value(context)})

def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value)
