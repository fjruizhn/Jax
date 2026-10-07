"""Transactional MariaDB adapter for the RuleDecisionStore protocol."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from types import MappingProxyType
from typing import Callable

from policy.authority_ledger.canonical import canonical_bytes, domain_hash
from policy.authority_ledger.errors import AuthorityStateError

from .errors import RuleAuthorityStorageError
from .models import RuleDecision, RuleDecisionStatus, RuleEvaluationRequest


_LOG = logging.getLogger(__name__)
_DECISION_DOMAIN = "JAX-FARO-RULE-AUTHORITY-DECISION"
_DECISION_VERSION = "1"
_CATALOG_DOMAIN = "JAX-FARO-RULE-CATALOG"
_CATALOG_VERSION = "1"
_DECISION_FIELDS = frozenset({
    "request_id", "request_hash", "request_catalog_hash", "status",
    "required_rule_id", "reason_code", "decided_at_utc",
    "catalogo",
})


def _catalog_projection(catalog):
    if catalog is None:
        return None
    if not isinstance(catalog, MappingProxyType) or any(
        not isinstance(key, str)
        or not isinstance(values, tuple)
        or not values
        or any(not isinstance(value, str) for value in values)
        for key, values in catalog.items()
    ):
        raise RuleAuthorityStorageError("catalogo de decisión inválido")
    return {key: list(values) for key, values in catalog.items()}


def _catalog_hash(catalog) -> str:
    return domain_hash(_CATALOG_DOMAIN, _CATALOG_VERSION, _catalog_projection(catalog))


def _decision_projection(decision: RuleDecision, request_catalog_hash: str) -> dict[str, object]:
    return {
        "request_id": decision.request_id,
        "request_hash": decision.request_hash,
        "request_catalog_hash": request_catalog_hash,
        "status": decision.status.value,
        "required_rule_id": decision.required_rule_id,
        "reason_code": decision.reason_code,
        "decided_at_utc": decision.decided_at_utc.isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z"),
        "catalogo": _catalog_projection(decision.catalogo),
    }


def _record_hash(sequence: int, request_id: str, previous_hash: str | None,
                 projection: dict[str, object]) -> str:
    return domain_hash(_DECISION_DOMAIN, _DECISION_VERSION, {
        "sequence": sequence,
        "record_type": "DECISION",
        "record_id": request_id,
        "previous_record_hash": previous_hash,
        "record": projection,
    })


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class MariaDBRuleDecisionStore:
    """Durable append-only implementation of the step-4 decision-store protocol.

    The connection factory must return a fresh MariaDB connection. A PERMIT is
    rejected until the next implementation step can persist its RulePermit in the
    same transaction.
    """

    def __init__(self, connection_factory: Callable[[], object]) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory debe ser invocable")
        self._connect = connection_factory

    def get(self, request: RuleEvaluationRequest) -> RuleDecision | None:
        if not isinstance(request, RuleEvaluationRequest):
            raise TypeError("get requiere RuleEvaluationRequest para ligar request_hash")
        connection = None
        try:
            connection = self._connect()
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT request_hash,request_catalog_hash,required_rule_id,status,reason_code,decided_at_utc,"
                    "canonical_decision,previous_record_hash,record_hash,audit_sequence "
                    "FROM rule_decisions WHERE request_id=%s",
                    (request.request_id,),
                )
                row = cursor.fetchone()
            if (row is None or row[0] != request.request_hash
                    or row[1] != _catalog_hash(request.catalogo)):
                return None
            return self._decode_row(request.request_id, row, request.catalogo)
        except RuleAuthorityStorageError:
            raise
        except Exception as exc:
            raise RuleAuthorityStorageError("no se pudo leer Rule Authority") from exc
        finally:
            if connection is not None:
                connection.close()

    def record(self, request: RuleEvaluationRequest, decision: RuleDecision) -> RuleDecision:
        if not isinstance(request, RuleEvaluationRequest) or not isinstance(decision, RuleDecision):
            raise TypeError("record requiere RuleEvaluationRequest y RuleDecision")
        if (decision.request_id != request.request_id
                or decision.request_hash != request.request_hash
                or decision.required_rule_id != request.rule_id):
            raise AuthorityStateError("decisión no corresponde a la solicitud")
        if decision.status is RuleDecisionStatus.PERMIT:
            raise RuleAuthorityStorageError(
                "PERMIT requiere insertar RulePermit en la misma transacción (paso 6)"
            )

        request_catalog_hash = _catalog_hash(request.catalogo)
        projection = _decision_projection(decision, request_catalog_hash)
        if (projection["catalogo"] is not None
                and projection["catalogo"] != _catalog_projection(request.catalogo)):
            raise AuthorityStateError("catalogo de decisión no corresponde al pin de la solicitud")
        canonical = canonical_bytes(projection)
        connection = None
        try:
            connection = self._connect()
            connection.begin()
            with connection.cursor() as cursor:
                # Acquire the singleton first. This serializes writers before
                # any missing-key/gap lock can be held while waiting for it.
                cursor.execute(
                    "SELECT audit_sequence,head_hash FROM rule_authority_audit_head "
                    "WHERE singleton=1 FOR UPDATE"
                )
                head = cursor.fetchone()
                if head is None or type(head[0]) is not int or head[0] < 0:
                    raise RuleAuthorityStorageError("audit head ausente o inválido")
                existing = self._select_request(cursor, request.request_id)
                if existing is not None:
                    if existing[1] != request_catalog_hash:
                        raise AuthorityStateError("request_id reutilizado con otro catálogo")
                    stored = self._decode_row(request.request_id, existing, request.catalogo)
                    if stored.request_hash != request.request_hash:
                        raise AuthorityStateError("request_id reutilizado con otro hash")
                    connection.commit()
                    return stored
                sequence = head[0] + 1
                previous_hash = head[1]
                record_hash = _record_hash(sequence, request.request_id, previous_hash, projection)
                cursor.execute(
                    "INSERT INTO rule_decisions "
                    "(request_id,request_hash,request_catalog_hash,required_rule_id,status,"
                    "reason_code,decided_at_utc,canonical_decision,previous_record_hash,"
                    "record_hash,audit_sequence) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        request.request_id, request.request_hash, request_catalog_hash,
                        decision.required_rule_id, decision.status.value, decision.reason_code,
                        decision.decided_at_utc.astimezone(timezone.utc).replace(tzinfo=None),
                        canonical, previous_hash, record_hash, sequence,
                    ),
                )
                cursor.execute(
                    "UPDATE rule_authority_audit_head SET audit_sequence=%s,head_hash=%s,"
                    "head_kind='DECISION',head_key=%s WHERE singleton=1 AND audit_sequence=%s",
                    (sequence, record_hash, request.request_id, sequence - 1),
                )
                if cursor.rowcount != 1:
                    raise RuleAuthorityStorageError("audit head cambió durante la transacción")
            connection.commit()
            return decision
        except (AuthorityStateError, RuleAuthorityStorageError):
            if connection is not None:
                connection.rollback()
            raise
        except Exception as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:  # fail-soft: el rollback fallido se registra y el error original se relanza como RuleAuthorityStorageError; la conexión se cierra y MariaDB revierte sola la transacción abierta
                    _LOG.exception("rollback de Rule Authority falló; se relanza el error original")
            raise RuleAuthorityStorageError("no se pudo persistir la decisión") from exc
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _select_request(cursor, request_id: str):
        cursor.execute(
            "SELECT request_hash,request_catalog_hash,required_rule_id,status,reason_code,decided_at_utc,"
            "canonical_decision,previous_record_hash,record_hash,audit_sequence "
            "FROM rule_decisions WHERE request_id=%s FOR UPDATE",
            (request_id,),
        )
        return cursor.fetchone()

    @staticmethod
    def _decode_row(request_id: str, row, expected_catalog=None) -> RuleDecision:
        (
            request_hash, request_catalog_hash, required_rule_id, status, reason, decided_at, payload_bytes,
            previous_hash, stored_hash, sequence,
        ) = row
        try:
            payload = json.loads(bytes(payload_bytes).decode("utf-8"))
            if not isinstance(payload, dict) or frozenset(payload) != _DECISION_FIELDS:
                raise ValueError("proyección canónica cerrada inválida")
            if canonical_bytes(payload) != bytes(payload_bytes):
                raise ValueError("proyección no canónica")
            if payload["request_id"] != request_id:
                raise ValueError("request_id distinto en proyección")
            if (payload["request_hash"] != request_hash
                    or payload["request_catalog_hash"] != request_catalog_hash
                    or payload["required_rule_id"] != required_rule_id
                    or payload["status"] != status
                    or payload["reason_code"] != reason):
                raise ValueError("columnas distintas de la proyección")
            if status == RuleDecisionStatus.PERMIT.value:
                raise ValueError("PERMIT está cerrado hasta el adapter atómico del paso 6")
            if request_catalog_hash != _catalog_hash(expected_catalog):
                raise ValueError("catálogo del request distinto de la proyección")
            projected_time = datetime.fromisoformat(
                payload["decided_at_utc"].replace("Z", "+00:00")
            )
            if _as_utc(decided_at) != projected_time.astimezone(timezone.utc):
                raise ValueError("decided_at_utc distinto de la proyección")
            if type(sequence) is not int or sequence < 1:
                raise ValueError("audit_sequence inválida")
            if _record_hash(sequence, request_id, previous_hash, payload) != stored_hash:
                raise ValueError("hash de decisión no coincide")
            return RuleDecision(
                request_id=request_id,
                request_hash=request_hash,
                status=RuleDecisionStatus(status),
                required_rule_id=required_rule_id,
                reason_code=reason,
                decided_at_utc=projected_time,
                catalogo=_decode_catalog(payload["catalogo"], expected_catalog),
            )
        except Exception as exc:
            raise RuleAuthorityStorageError("fila de decisión inválida") from exc


def _decode_catalog(value, expected_catalog):
    if value is None:
        return None
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not isinstance(values, list)
        or any(not isinstance(item, str) for item in values)
        for key, values in value.items()
    ):
        raise ValueError("catálogo de decisión inválido")
    decoded = MappingProxyType({key: tuple(values) for key, values in value.items()})
    if value != _catalog_projection(expected_catalog):
        raise ValueError("catalogo no corresponde al pin de la solicitud")
    return decoded
