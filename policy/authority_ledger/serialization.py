"""Strict persistence codecs for immutable ledger records."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .errors import AuthorityEventValidationError
from .models import (
    AuthorityEvent, AuthorityEventIntent, AuthorityEventType, AuthorityLedgerGenesis,
    OverlayPayload, OverlayScope, OverlayType, _RATIFICATION_STORAGE_SEAL,
)


_INTENT_FIELDS = {
    "event_type", "actor_id", "evidence_refs", "policy_corpus_hash",
    "static_policy_view_projection", "ratification_event_id", "overlay", "overlay_id",
}
_EVENT_FIELDS = {
    "event_id", "sequence", "previous_event_hash", "intent",
    "recorded_at_utc", "signature", "event_hash",
}
_VARIANT_PAYLOAD_FIELDS = {
    AuthorityEventType.RATIFICATION_GRANTED: {"policy_corpus_hash", "static_policy_view_projection"},
    AuthorityEventType.RATIFICATION_REVOKED: {"ratification_event_id"},
    AuthorityEventType.ACTIVATION_GRANTED: {"ratification_event_id"},
    AuthorityEventType.ACTIVATION_DEACTIVATED: set(),
    AuthorityEventType.OVERLAY_ISSUED: {"overlay"},
    AuthorityEventType.OVERLAY_REVOKED: {"overlay_id"},
}


def intent_projection(intent: AuthorityEventIntent) -> dict:
    """Return only the signed public fields; internal provenance seals never persist."""
    return intent.canonical_projection()


def event_projection(event: AuthorityEvent) -> dict:
    return {
        "event_id": event.event_id,
        "sequence": event.sequence,
        "previous_event_hash": event.previous_event_hash,
        "intent": intent_projection(event.intent),
        "recorded_at_utc": event.recorded_at_utc.isoformat(),
        "signature": event.signature,
        "event_hash": event.event_hash,
    }


def _closed_object(value, fields: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise AuthorityEventValidationError(f"{name} inválido: campos ausentes o desconocidos")
    return value


def _array(value, name: str) -> tuple:
    if not isinstance(value, list):
        raise AuthorityEventValidationError(f"{name} debe ser array")
    return tuple(value)


def _overlay_from_projection(value):
    if value is None:
        return None
    fields = {
        "overlay_id", "overlay_type", "policy_corpus_hash", "scope",
        "valid_from_utc", "valid_until_utc", "target_rule_ids", "exception_code",
        "delegate_actor_id", "delegate_actor_kind", "delegated_scope",
        "may_subdelegate", "delegated_action_ids", "interpretation_code", "target_overlay_id",
    }
    _closed_object(value, fields, "overlay")

    def scope(data):
        if data is None:
            return None
        _closed_object(data, {"subjects", "actions", "conditions_all"}, "overlay.scope")
        return OverlayScope(
            _array(data["subjects"], "overlay.scope.subjects"),
            _array(data["actions"], "overlay.scope.actions"),
            _array(data["conditions_all"], "overlay.scope.conditions_all"),
        )

    return OverlayPayload(
        overlay_id=value["overlay_id"],
        overlay_type=OverlayType(value["overlay_type"]),
        policy_corpus_hash=value["policy_corpus_hash"],
        scope=scope(value["scope"]),
        valid_from_utc=datetime.fromisoformat(value["valid_from_utc"]),
        valid_until_utc=datetime.fromisoformat(value["valid_until_utc"]) if value["valid_until_utc"] is not None else None,
        target_rule_ids=_array(value["target_rule_ids"], "overlay.target_rule_ids"),
        exception_code=value["exception_code"],
        delegate_actor_id=value["delegate_actor_id"],
        delegate_actor_kind=value["delegate_actor_kind"],
        delegated_scope=scope(value["delegated_scope"]),
        may_subdelegate=value["may_subdelegate"],
        delegated_action_ids=_array(value["delegated_action_ids"], "overlay.delegated_action_ids"),
        interpretation_code=value["interpretation_code"],
        target_overlay_id=value["target_overlay_id"],
    )


def intent_from_projection(value) -> AuthorityEventIntent:
    _closed_object(value, _INTENT_FIELDS, "canonical_intent")
    try:
        event_type = AuthorityEventType(value["event_type"])
        allowed_payload = _VARIANT_PAYLOAD_FIELDS[event_type]
        actual_payload = {name for name in allowed_payload if value[name] is not None}
        other_payload = {
            "policy_corpus_hash", "static_policy_view_projection", "ratification_event_id",
            "overlay", "overlay_id",
        } - allowed_payload
        if actual_payload != allowed_payload or any(value[name] is not None for name in other_payload):
            raise AuthorityEventValidationError(f"payload inválido para {event_type.value}")
        seal = _RATIFICATION_STORAGE_SEAL if event_type is AuthorityEventType.RATIFICATION_GRANTED else None
        return AuthorityEventIntent(
            event_type=event_type,
            actor_id=value["actor_id"],
            evidence_refs=_array(value["evidence_refs"], "evidence_refs"),
            policy_corpus_hash=value["policy_corpus_hash"],
            static_policy_view_projection=value["static_policy_view_projection"],
            ratification_event_id=value["ratification_event_id"],
            overlay=_overlay_from_projection(value["overlay"]),
            overlay_id=value["overlay_id"],
            _ratification_snapshot_seal=seal,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorityEventValidationError("canonical_intent inválido") from exc


def genesis_from_projection(value):
    if isinstance(value, bytes): value = value.decode("utf-8")
    if isinstance(value, str): value = json.loads(value)
    if not isinstance(value, dict): raise TypeError("canonical_genesis inválido")
    return AuthorityLedgerGenesis(**value)


def event_from_storage_row(row):
    if len(row) != 11:
        raise AuthorityEventValidationError("fila authority_event inválida")
    (sequence, event_id, event_type, actor_id, stored_intent, value,
     stored_evidence, stored_previous_hash, stored_hash, stored_signature,
     stored_recorded_at) = row
    if stored_intent is None:
        raise AuthorityEventValidationError("canonical_intent ausente")
    value = _json_value(value, "canonical_event")
    _closed_object(value, _EVENT_FIELDS, "canonical_event")
    try:
        if value["event_id"] != event_id or value["sequence"] != sequence:
            raise AuthorityEventValidationError("fila no coincide con canonical_event")
        intent = intent_from_projection(value["intent"])
        event = AuthorityEvent(
            event_id, sequence, value["previous_event_hash"], intent,
            datetime.fromisoformat(value["recorded_at_utc"]), value["signature"], value["event_hash"],
        )
        intent_value = _json_value(stored_intent, "canonical_intent")
        if intent_value != intent_projection(event.intent):
            raise AuthorityEventValidationError("canonical_intent no coincide con canonical_event")
        evidence = _json_value(stored_evidence, "evidence_refs")
        if not isinstance(evidence, list) or evidence != list(event.intent.evidence_refs):
            raise AuthorityEventValidationError("evidence_refs no coincide con canonical_event")
        stored_time = stored_recorded_at
        if not isinstance(stored_time, datetime):
            raise AuthorityEventValidationError("recorded_at_utc de storage inválido")
        if stored_time.tzinfo is None or stored_time.utcoffset() is None:
            stored_time = stored_time.replace(tzinfo=timezone.utc)
        else:
            stored_time = stored_time.astimezone(timezone.utc)
        if (
            event_type != event.intent.event_type.value
            or actor_id != event.intent.actor_id
            or stored_previous_hash != event.previous_event_hash
            or stored_hash != event.event_hash
            or stored_signature != event.signature
            or stored_time != event.recorded_at_utc
        ):
            raise AuthorityEventValidationError("columnas authority_event no coinciden con canonical_event")
        return event
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorityEventValidationError("canonical_event inválido") from exc


def _json_value(value, field: str):
    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            value = json.loads(value)
        return value
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorityEventValidationError(f"{field} JSON inválido") from exc
