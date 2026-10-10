"""Closed RulePermit values; only the authority store may seal a returned permit."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import unicodedata
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from policy.authority_ledger.canonical import canonical_bytes, domain_hash
from policy.authority_ledger.errors import AuthorityEventValidationError
from policy.authority_ledger.ids import uuid7_text

from .errors import RuleAuthorityError


_SEAL = object()
_DOMAIN = "JAX-FARO-RULE-PERMIT"
_VERSION = "1"
_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9._:-]{0,127}\Z")
_VERSION_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}\Z")
_RULE_PATH = re.compile(r"policy/faro/[a-z][a-z0-9-]{0,63}\.yaml\Z")
_FIELDS = frozenset({
    "permit_id", "request_id", "request_hash", "rule_id", "rule_path",
    "rule_blob_oid", "rule_content_hash", "policy_revision", "policy_tree_oid",
    "policy_snapshot_hash", "ratification_event_id", "authority_ledger_checkpoint",
    "stop_checkpoint", "capability_id", "capability_version", "capability_class",
    "capability_limits", "issued_at_utc", "expires_at_utc",
})


def _identifier(value: object, field_name: str) -> str:
    if (type(value) is not str or not value or unicodedata.normalize("NFC", value) != value
            or not _IDENTIFIER.fullmatch(value)):
        raise AuthorityEventValidationError(f"{field_name} inválido")
    return value


def _utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise AuthorityEventValidationError(f"{field_name} requiere zona horaria")
    return value.astimezone(timezone.utc)


def _freeze_json(value: Any, field_name: str) -> Any:
    if value is None or type(value) in (str, bool, int):
        if type(value) is str:
            return unicodedata.normalize("NFC", value)
        if type(value) is int and abs(value) > 2**53:
            raise AuthorityEventValidationError(f"{field_name}: entero fuera de rango")
        return value
    if type(value) is float:
        raise AuthorityEventValidationError(f"{field_name}: float no admitido")
    if type(value) in (list, tuple):
        return tuple(_freeze_json(item, field_name) for item in value)
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise AuthorityEventValidationError(f"{field_name}: clave JSON inválida")
            normalized = unicodedata.normalize("NFC", key)
            if normalized in frozen:
                raise AuthorityEventValidationError(f"{field_name}: claves NFC duplicadas")
            frozen[normalized] = _freeze_json(item, field_name)
        return MappingProxyType(frozen)
    raise AuthorityEventValidationError(f"{field_name}: valor fuera de JSON cerrado")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return value


@dataclass(frozen=True)
class RulePermitDraft:
    """Untrusted candidate assembled by evaluation; it cannot authorize consume."""

    permit_id: str
    request_id: str
    request_hash: str
    rule_id: str
    rule_path: str
    rule_blob_oid: str
    rule_content_hash: str
    policy_revision: str
    policy_tree_oid: str
    policy_snapshot_hash: str
    ratification_event_id: str
    authority_ledger_checkpoint: Mapping[str, Any]
    stop_checkpoint: Mapping[str, Any]
    capability_id: str
    capability_version: str
    capability_class: str
    capability_limits: Mapping[str, Any]
    issued_at_utc: datetime
    expires_at_utc: datetime
    _payload: Mapping[str, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("permit_id", "request_id", "ratification_event_id"):
            try:
                uuid7_text(getattr(self, name), name)
            except Exception as exc:
                raise AuthorityEventValidationError(f"{name} inválido") from exc
        _identifier(self.rule_id, "rule_id")
        _identifier(self.capability_id, "capability_id")
        if (type(self.capability_version) is not str or not self.capability_version
                or unicodedata.normalize("NFC", self.capability_version) != self.capability_version
                or not _VERSION_TOKEN.fullmatch(self.capability_version)):
            raise AuthorityEventValidationError("capability_version inválida")
        if type(self.capability_class) is not str or self.capability_class not in {
            "REVERSIBLE", "OBLIGATING"
        }:
            raise AuthorityEventValidationError("capability_class fuera del enum cerrado")
        if type(self.rule_path) is not str or not _RULE_PATH.fullmatch(self.rule_path):
            raise AuthorityEventValidationError("rule_path fuera de policy/faro/<nombre>.yaml")
        for name in ("request_hash", "rule_content_hash", "policy_snapshot_hash"):
            if type(getattr(self, name)) is not str or not _SHA256.fullmatch(getattr(self, name)):
                raise AuthorityEventValidationError(f"{name} inválido")
        git_oids = (self.rule_blob_oid, self.policy_revision, self.policy_tree_oid)
        if any(type(value) is not str or not _OID.fullmatch(value) for value in git_oids):
            raise AuthorityEventValidationError("OID Git de RulePermit inválido")
        if len({len(value) for value in git_oids}) != 1:
            raise AuthorityEventValidationError("OID Git mezclan SHA-1/SHA-256")
        issued = _utc(self.issued_at_utc, "issued_at_utc")
        expires = _utc(self.expires_at_utc, "expires_at_utc")
        if expires <= issued:
            raise AuthorityEventValidationError("expires_at_utc debe ser posterior a issued_at_utc")
        if not isinstance(self.authority_ledger_checkpoint, Mapping):
            raise AuthorityEventValidationError("authority_ledger_checkpoint debe ser objeto")
        if not isinstance(self.stop_checkpoint, Mapping):
            raise AuthorityEventValidationError("stop_checkpoint debe ser objeto")
        if not isinstance(self.capability_limits, Mapping):
            raise AuthorityEventValidationError("capability_limits debe ser objeto")
        payload = {
            "permit_id": self.permit_id,
            "request_id": self.request_id,
            "request_hash": self.request_hash,
            "rule_id": self.rule_id,
            "rule_path": self.rule_path,
            "rule_blob_oid": self.rule_blob_oid,
            "rule_content_hash": self.rule_content_hash,
            "policy_revision": self.policy_revision,
            "policy_tree_oid": self.policy_tree_oid,
            "policy_snapshot_hash": self.policy_snapshot_hash,
            "ratification_event_id": self.ratification_event_id,
            "authority_ledger_checkpoint": _freeze_json(self.authority_ledger_checkpoint,
                                                         "authority_ledger_checkpoint"),
            "stop_checkpoint": _freeze_json(self.stop_checkpoint, "stop_checkpoint"),
            "capability_id": self.capability_id,
            "capability_version": self.capability_version,
            "capability_class": self.capability_class,
            "capability_limits": _freeze_json(self.capability_limits, "capability_limits"),
            "issued_at_utc": issued,
            "expires_at_utc": expires,
        }
        object.__setattr__(self, "_payload", MappingProxyType(payload))

    def projection(self) -> dict[str, Any]:
        return _plain(self._payload)


class RulePermit:
    """Store-sealed permit. Drafts or decoded caller values are never trusted."""

    __slots__ = ("_payload", "_seal")

    def __init__(self, payload: Mapping[str, Any], *, _seal: object = None) -> None:
        if _seal is not _SEAL:
            raise RuleAuthorityError("RulePermit solo lo entrega el store autoritativo")
        frozen = _freeze_json(payload, "RulePermit")
        if not isinstance(frozen, Mapping) or frozenset(frozen) != _FIELDS | {"permit_hash"}:
            raise RuleAuthorityError("RulePermit: proyección cerrada inválida")
        expected = _permit_hash({key: value for key, value in frozen.items()
                                 if key != "permit_hash"})
        if frozen["permit_hash"] != expected:
            raise RuleAuthorityError("RulePermit permit_hash no coincide")
        object.__setattr__(self, "_payload", frozen)
        object.__setattr__(self, "_seal", _SEAL)

    def __getattr__(self, name: str) -> Any:
        if name in self._payload:
            value = self._payload[name]
            if name in {"issued_at_utc", "expires_at_utc"} and type(value) is str:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            return value
        raise AttributeError(name)

    def __setattr__(self, *_args) -> None:
        raise RuleAuthorityError("RulePermit es inmutable")

    def __delattr__(self, *_args) -> None:
        raise RuleAuthorityError("RulePermit es inmutable")

    def projection(self) -> Mapping[str, Any]:
        return self._payload

    def _is_store_sealed(self) -> bool:
        return self._seal is _SEAL


def _permit_hash(projection: Mapping[str, Any]) -> str:
    return domain_hash(_DOMAIN, _VERSION, _plain(projection))


def _trusted_permit(value: RulePermitDraft | Mapping[str, Any]) -> RulePermit:
    """Internal adapter boundary used only after durable store verification."""
    if type(value) is RulePermitDraft:
        payload = value.projection()
    elif isinstance(value, Mapping):
        payload = dict(value)
        stored_hash = payload.pop("permit_hash", None)
        if frozenset(payload) != _FIELDS:
            raise RuleAuthorityError("RulePermit: proyección incompleta o con campos extra")
        for field_name in ("issued_at_utc", "expires_at_utc"):
            if type(payload[field_name]) is str:
                try:
                    payload[field_name] = datetime.fromisoformat(
                        payload[field_name].replace("Z", "+00:00")
                    )
                except ValueError as exc:
                    raise RuleAuthorityError(f"RulePermit {field_name} inválido") from exc
        draft = RulePermitDraft(**payload)
        payload = draft.projection()
        computed = _permit_hash(payload)
        if stored_hash is not None and stored_hash != computed:
            raise RuleAuthorityError("RulePermit permit_hash no coincide")
    else:
        raise RuleAuthorityError("store requiere RulePermitDraft o proyección cerrada")
    payload["permit_hash"] = _permit_hash(payload)
    return RulePermit(payload, _seal=_SEAL)


__all__ = ["RulePermit", "RulePermitDraft"]
