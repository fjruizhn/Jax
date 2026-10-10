"""Transactional MariaDB adapter for the RuleDecisionStore protocol."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Callable

from jax.faro.catalogo_topes import CatalogoTopes
from policy.authority_ledger.canonical import canonical_bytes, domain_hash
from policy.authority_ledger.errors import AuthorityStateError

from .errors import RuleAuthorityStorageError
from .models import RuleDecision, RuleDecisionStatus, RuleEvaluationRequest
from .permit import (RulePermit, RulePermitConsumption, RulePermitConsumptionDraft,
                     RulePermitDraft, _EvaluatedPermit, _trusted_consumption,
                     _trusted_permit)
from .providers import AlmacenCheckpointsBloqueable


_LOG = logging.getLogger(__name__)
_DECISION_DOMAIN = "JAX-FARO-RULE-AUTHORITY-DECISION"
_DECISION_VERSION = "1"
_CATALOG_DOMAIN = "JAX-FARO-RULE-CATALOG"
_CATALOG_VERSION = "1"
_DECISION_FIELDS = frozenset({
    "request_id", "request_hash", "request_catalog_hash", "status",
    "required_rule_id", "reason_code", "decided_at_utc",
})
_PERMIT_DOMAIN = "JAX-FARO-RULE-AUTHORITY-PERMIT-RECORD"
_PERMIT_VERSION = "1"
_CONSUMPTION_RECORD_DOMAIN = "JAX-FARO-RULE-AUTHORITY-CONSUMPTION-RECORD"
_CONSUMPTION_RECORD_VERSION = "1"
_PERMIT_FIELDS = frozenset({
    "permit_id", "request_id", "request_hash", "rule_id", "rule_path",
    "rule_blob_oid", "rule_content_hash", "policy_revision", "policy_tree_oid",
    "policy_snapshot_hash", "ratification_event_id", "authority_ledger_checkpoint",
    "stop_checkpoint", "capability_id", "capability_version", "capability_class",
    "capability_limits", "issued_at_utc", "expires_at_utc", "permit_hash",
})


class RuleAuthorityCommittedCheckpointError(RuleAuthorityStorageError):
    """DB committed, but its required external Rule Authority anchor failed.

    The operation is deliberately not reported as rolled back.  The next
    writer compares the durable checkpoint head with the DB head and fails
    closed until an operator reconciles the two stores.
    """


class RuleAuthorityCommittedCleanupError(RuleAuthorityStorageError):
    """DB and external checkpoint committed, but local cleanup then failed.

    This outcome is explicitly durable and anchored.  It is distinct from a
    failed checkpoint publication so callers never infer an anchor mismatch
    from a connection/cursor close error that occurred after confirmation.
    """


def _catalog_projection(catalog):
    """Proyección canónica del catálogo SELLADO del pin (clases + OID del pin)."""
    if type(catalog) is not CatalogoTopes:
        raise RuleAuthorityStorageError("catalogo de solicitud inválido")
    return {
        "oid_pin": catalog.oid_pin,
        "clases": {key: list(values) for key, values in catalog.items()},
    }


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


def _permit_record_hash(sequence: int, permit_id: str, previous_hash: str,
                        projection: dict[str, object]) -> str:
    return domain_hash(_PERMIT_DOMAIN, _PERMIT_VERSION, {
        "sequence": sequence,
        "record_type": "PERMIT",
        "record_id": permit_id,
        "previous_record_hash": previous_hash,
        "record": projection,
    })


def _consumption_record_hash(sequence: int, permit_id: str, previous_hash: str | None,
                             projection: dict[str, object]) -> str:
    """Hash-chain entry for the single append-only consumption leaf."""
    return domain_hash(_CONSUMPTION_RECORD_DOMAIN, _CONSUMPTION_RECORD_VERSION, {
        "sequence": sequence,
        "record_type": "CONSUMPTION",
        "record_id": permit_id,
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

    def __init__(self, connection_factory: Callable[[], object], *, checkpoint_store=None) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory debe ser invocable")
        if checkpoint_store is not None and not isinstance(
            checkpoint_store, AlmacenCheckpointsBloqueable
        ):
            raise TypeError("checkpoint_store no implementa el contrato durable bloqueable")
        self._connect = connection_factory
        self._checkpoint_store = checkpoint_store

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

        if self._checkpoint_store is not None:
            with self._checkpoint_store.locked():
                return self._record_transaction(request, decision)
        return self._record_transaction(request, decision)

    def _record_transaction(self, request: RuleEvaluationRequest,
                            decision: RuleDecision) -> RuleDecision:
        request_catalog_hash = _catalog_hash(request.catalogo)
        projection = _decision_projection(decision, request_catalog_hash)
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
                checkpoint_head = None
                if self._checkpoint_store is not None:
                    from .trusted_checkpoint import RULE_AUDIT_GENESIS_HEAD

                    checkpoint_head = self._checkpoint_store.head_actual()
                    expected_head = (
                        RULE_AUDIT_GENESIS_HEAD if head[0] == 0 and head[1] is None
                        else head[1]
                    )
                    if (type(expected_head) is not str
                            or checkpoint_head != expected_head
                            or self._checkpoint_store.confirmar(checkpoint_head) is not True):
                        raise RuleAuthorityStorageError(
                            "checkpoint externo no coincide con el audit head DB"
                        )
                existing = self._select_request(cursor, request.request_id)
                if existing is not None:
                    if existing[1] != request_catalog_hash:
                        raise AuthorityStateError("request_id reutilizado con otro catálogo")
                    stored = self._decode_row(request.request_id, existing, request.catalogo)
                    if stored.request_hash != request.request_hash:
                        raise AuthorityStateError("request_id reutilizado con otro hash")
                    if (self._checkpoint_store is not None
                            and self._checkpoint_store.confirmar(existing[8]) is not True):
                        raise RuleAuthorityStorageError(
                            "checkpoint externo no contiene la decisión idempotente"
                        )
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
            if self._checkpoint_store is not None:
                try:
                    self._checkpoint_store.publicar(record_hash, anterior=checkpoint_head)
                    if self._checkpoint_store.confirmar(record_hash) is not True:
                        raise RuleAuthorityStorageError(
                            "checkpoint externo no confirma el head recién publicado"
                        )
                except Exception as exc:
                    raise RuleAuthorityStorageError(
                        "decisión confirmada en DB pero sin checkpoint externo confirmado"
                    ) from exc
            return decision
        except (AuthorityStateError, RuleAuthorityStorageError) as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except Exception as rollback_exc:  # fail-soft: el rollback fallido se registra y sale como RuleAuthorityStorageError (nunca se traga ni deja escapar un error crudo); la conexión se cierra y MariaDB revierte sola la transacción abierta
                    _LOG.exception("rollback de Rule Authority falló; se relanza como error tipado")
                    raise RuleAuthorityStorageError("no se pudo persistir la decisión") from exc
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

    def record_permit(self, request: RuleEvaluationRequest, decision: RuleDecision,
                      evaluated_permit: _EvaluatedPermit) -> RulePermit:
        """Atomically insert PERMIT + RulePermit, then anchor the committed head.

        No caller receives a store-sealed permit unless the database transaction
        committed and the external checkpoint log confirms the exact permit record.
        """
        if self._checkpoint_store is None:
            raise RuleAuthorityStorageError("PERMIT requiere checkpoint externo durable")
        if (type(request) is not RuleEvaluationRequest or type(decision) is not RuleDecision
                or type(evaluated_permit) is not _EvaluatedPermit
                or not evaluated_permit._is_kernel_sealed()):
            raise TypeError("record_permit requiere una evaluación sellada del kernel")
        permit_draft = evaluated_permit._draft
        if type(permit_draft) is not RulePermitDraft:
            raise TypeError("evaluación sellada no contiene RulePermitDraft")
        if (decision.status is not RuleDecisionStatus.PERMIT
                or decision.reason_code is not None
                or decision.request_id != request.request_id
                or decision.request_hash != request.request_hash
                or decision.required_rule_id != request.rule_id):
            raise AuthorityStateError("decisión PERMIT no corresponde a la solicitud")
        if (permit_draft.request_id != request.request_id
                or permit_draft.request_hash != request.request_hash
                or permit_draft.rule_id != request.rule_id
                or permit_draft.issued_at_utc.astimezone(timezone.utc)
                != decision.decided_at_utc.astimezone(timezone.utc)):
            raise AuthorityStateError("RulePermit no corresponde a la decisión/request")

        permit = _trusted_permit(permit_draft)
        permit_projection = dict(permit.projection())
        if frozenset(permit_projection) != _PERMIT_FIELDS:
            raise RuleAuthorityStorageError("proyección RulePermit no coincide con el contrato de persistencia")
        permit_payload = canonical_bytes(permit_projection)
        request_catalog_hash = _catalog_hash(request.catalogo)
        decision_projection = _decision_projection(decision, request_catalog_hash)
        canonical_decision = canonical_bytes(decision_projection)
        connection = None
        try:
            with self._checkpoint_store.locked():
                connection = self._connect()
                connection.begin()
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT audit_sequence,head_hash FROM rule_authority_audit_head "
                        "WHERE singleton=1 FOR UPDATE"
                    )
                    head = cursor.fetchone()
                    if head is None or type(head[0]) is not int or head[0] < 0:
                        raise RuleAuthorityStorageError("audit head ausente o inválido")
                    from .trusted_checkpoint import RULE_AUDIT_GENESIS_HEAD

                    checkpoint_head = self._checkpoint_store.head_actual()
                    expected_head = (RULE_AUDIT_GENESIS_HEAD if head[0] == 0 and head[1] is None
                                     else head[1])
                    if (type(expected_head) is not str or checkpoint_head != expected_head
                            or self._checkpoint_store.confirmar(checkpoint_head) is not True):
                        raise RuleAuthorityStorageError(
                            "checkpoint externo no coincide con el audit head DB"
                        )

                    existing = self._select_request(cursor, request.request_id)
                    if existing is not None:
                        if (existing[1] != request_catalog_hash
                                or existing[0] != request.request_hash
                                or existing[2] != request.rule_id):
                            raise AuthorityStateError("request_id reutilizado con otro hash/catálogo/regla")
                        if existing[3] != RuleDecisionStatus.PERMIT.value:
                            raise AuthorityStateError("request_id ya tiene decisión distinta de PERMIT")
                        stored_decision = self._decode_row(
                            request.request_id, existing, request.catalogo, allow_permit=True
                        )
                        permit_row = self._select_permit(cursor, request.request_id)
                        if permit_row is None:
                            raise RuleAuthorityStorageError("PERMIT existente sin RulePermit durable")
                        stored_permit = self._decode_permit_row(permit_row)
                        if dict(stored_permit.projection()) != permit_projection:
                            raise AuthorityStateError("request_id reintentado con RulePermit distinto")
                        if self._checkpoint_store.confirmar(permit_row[23]) is not True:
                            raise RuleAuthorityStorageError("checkpoint no contiene RulePermit idempotente")
                        connection.commit()
                        return stored_permit

                    decision_sequence = head[0] + 1
                    decision_previous = head[1]
                    decision_hash = _record_hash(
                        decision_sequence, request.request_id, decision_previous,
                        decision_projection,
                    )
                    cursor.execute(
                        "INSERT INTO rule_decisions "
                        "(request_id,request_hash,request_catalog_hash,required_rule_id,status,"
                        "reason_code,decided_at_utc,canonical_decision,previous_record_hash,"
                        "record_hash,audit_sequence) VALUES (%s,%s,%s,%s,'PERMIT',NULL,%s,%s,%s,%s,%s)",
                        (request.request_id, request.request_hash, request_catalog_hash,
                         request.rule_id, decision.decided_at_utc.astimezone(timezone.utc).replace(tzinfo=None),
                         canonical_decision, decision_previous, decision_hash, decision_sequence),
                    )
                    permit_sequence = decision_sequence + 1
                    permit_hash = _permit_record_hash(
                        permit_sequence, permit.permit_id, decision_hash, permit_projection,
                    )
                    cursor.execute(
                        "INSERT INTO rule_permits "
                        "(permit_id,request_id,request_hash,rule_id,rule_path,rule_blob_oid,"
                        "rule_content_hash,policy_revision,policy_tree_oid,policy_snapshot_hash,"
                        "ratification_event_id,authority_ledger_checkpoint,stop_checkpoint,"
                        "capability_id,capability_version,capability_class,capability_limits,"
                        "issued_at_utc,expires_at_utc,permit_hash,canonical_payload,"
                        "previous_record_hash,record_hash,audit_sequence) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (permit.permit_id, request.request_id, request.request_hash,
                         permit.rule_id, permit.rule_path, permit.rule_blob_oid,
                         permit.rule_content_hash, permit.policy_revision, permit.policy_tree_oid,
                         permit.policy_snapshot_hash, permit.ratification_event_id,
                         canonical_bytes(permit.authority_ledger_checkpoint),
                         canonical_bytes(permit.stop_checkpoint), permit.capability_id,
                         permit.capability_version, permit.capability_class,
                         canonical_bytes(permit.capability_limits),
                         decision.decided_at_utc.astimezone(timezone.utc).replace(tzinfo=None),
                         permit.expires_at_utc.astimezone(timezone.utc).replace(tzinfo=None),
                         permit.permit_hash, permit_payload, decision_hash, permit_hash,
                         permit_sequence),
                    )
                    cursor.execute(
                        "UPDATE rule_authority_audit_head SET audit_sequence=%s,head_hash=%s,"
                        "head_kind='PERMIT',head_key=%s WHERE singleton=1 AND audit_sequence=%s",
                        (permit_sequence, permit_hash, permit.permit_id, head[0]),
                    )
                    if cursor.rowcount != 1:
                        raise RuleAuthorityStorageError("audit head cambió durante la transacción PERMIT")
                connection.commit()
                try:
                    self._checkpoint_store.publicar(permit_hash, anterior=checkpoint_head)
                    if self._checkpoint_store.confirmar(permit_hash) is not True:
                        raise RuleAuthorityStorageError("checkpoint no confirma el RulePermit recién publicado")
                except Exception as exc:
                    raise RuleAuthorityStorageError(
                        "PERMIT confirmado en DB pero sin checkpoint externo confirmado"
                    ) from exc
                return permit
        except (AuthorityStateError, RuleAuthorityStorageError):
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    _LOG.exception("rollback de Rule Authority PERMIT falló")
            raise
        except Exception as exc:
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    _LOG.exception("rollback de Rule Authority PERMIT falló")
            raise RuleAuthorityStorageError("no se pudo persistir PERMIT + RulePermit") from exc
        finally:
            if connection is not None:
                connection.close()

    def consume_permit(self, transaction_owner, consumption_draft: RulePermitConsumptionDraft
                       ) -> RulePermitConsumption:
        """Append the sole consumption leaf inside the verified shared transaction.

        The caller supplies the owner only through the typed Block-4 fence
        contract.  The owner validates permit → Block-4-head ordering; this
        store owns the Rule Authority audit row, canonical payload and chain
        update, so a caller cannot commit a permit lock with arbitrary SQL.
        """
        if type(consumption_draft) is not RulePermitConsumptionDraft:
            raise TypeError("consume_permit requiere RulePermitConsumptionDraft exacto")
        try:
            return transaction_owner.consume_rule_permit(self, consumption_draft)
        except (AuthorityStateError, RuleAuthorityStorageError):
            raise
        except Exception as exc:
            raise RuleAuthorityStorageError("no se pudo consumir RulePermit atómicamente") from exc

    def _finalize_consumption_in_transaction(self, transaction_owner, consumption_draft
                                             ) -> RulePermitConsumption:
        """Persist, commit, and anchor one consumption while both fences hold.

        This is intentionally the only completion route.  A caller cannot get
        a consumption object after an SQL append but before the transaction and
        the external Rule Authority checkpoint agree on its record hash.
        """
        from policy.authority_ledger.storage import MariaDBAuthorityLedgerTransactionOwner

        if type(transaction_owner) is not MariaDBAuthorityLedgerTransactionOwner:
            raise AuthorityStateError("consumo requiere transaction owner MariaDB sellado")
        if transaction_owner._rule_authority_decision_store is not self:
            raise AuthorityStateError("transaction owner no pertenece a este decision-store")
        if (not transaction_owner.active or not transaction_owner._verified
                or not transaction_owner._permit_locked
                or transaction_owner._locked_permit_identity
                != (consumption_draft.permit_id, consumption_draft.request_hash)):
            raise AuthorityStateError("consumo no coincide con el permit bloqueado/verificado")
        locked_permit = transaction_owner._locked_permit
        if (type(locked_permit) is not RulePermit
                or locked_permit._is_store_sealed() is not True
                or locked_permit.permit_id != consumption_draft.permit_id
                or locked_permit.request_hash != consumption_draft.request_hash):
            raise AuthorityStateError("consumo requiere RulePermit bloqueado, completo y sellado")
        if self._checkpoint_store is None:
            raise RuleAuthorityStorageError("consumo requiere checkpoint Rule Authority durable")

        consumption = _trusted_consumption(consumption_draft)
        projection = dict(consumption.projection())
        canonical_payload = canonical_bytes(projection)
        try:
            audit_sequence, previous_hash = transaction_owner.lock_rule_authority_audit_head_for_update()
            from .trusted_checkpoint import RULE_AUDIT_GENESIS_HEAD

            checkpoint_head = self._checkpoint_store.head_actual()
            expected_head = (
                RULE_AUDIT_GENESIS_HEAD if audit_sequence == 0 and previous_hash is None
                else previous_hash
            )
            if (type(expected_head) is not str or checkpoint_head != expected_head
                    or self._checkpoint_store.confirmar(checkpoint_head) is not True):
                raise RuleAuthorityStorageError(
                    "checkpoint externo no coincide con el audit head DB antes del consumo"
                )
            sequence = audit_sequence + 1
            cursor = transaction_owner._cursor
            cursor.execute(
                "SELECT permit_id,request_hash,consumed_at_utc,canonical_payload,consumption_hash,"
                "previous_record_hash,record_hash,audit_sequence "
                "FROM jax_rule_authority.rule_permit_consumptions "
                "WHERE permit_id=%s FOR UPDATE",
                (consumption.permit_id,),
            )
            existing = cursor.fetchone()
            if existing is not None:
                stored, stored_record_hash, stored_sequence = self._decode_consumption_row(existing)
                # The public idempotency identity is exactly permit_id plus
                # request_hash. A client retry naturally has a fresh local
                # timestamp, so it must return the durable consumption rather
                # than treating that timestamp as a conflicting identity.
                if (stored.permit_id != locked_permit.permit_id
                        or stored.request_hash != locked_permit.request_hash
                        or stored.permit_id != consumption_draft.permit_id
                        or stored.request_hash != consumption_draft.request_hash):
                    raise AuthorityStateError("permit ya consumido con identidad distinta")
                if stored_sequence > audit_sequence:
                    raise RuleAuthorityStorageError("consumo almacenado está delante del audit head")
                if (stored_sequence == audit_sequence
                        and stored_record_hash != previous_hash):
                    raise RuleAuthorityStorageError(
                        "audit head no referencia el consumo idempotente"
                    )
                if self._checkpoint_store.confirmar(stored_record_hash) is not True:
                    raise RuleAuthorityStorageError(
                        "checkpoint externo no confirma el consumo idempotente"
                    )
                transaction_owner._mark_consumption_persisted()
                close_error = transaction_owner._commit_consumption_for_store()
                if close_error is not None:
                    raise RuleAuthorityCommittedCleanupError(
                        "consumo confirmado y anclado pero el cierre posterior al commit falló"
                    ) from close_error
                return stored

            record_hash = _consumption_record_hash(
                sequence, consumption.permit_id, previous_hash, projection
            )
            cursor.execute(
                "INSERT INTO jax_rule_authority.rule_permit_consumptions "
                "(permit_id,request_hash,consumed_at_utc,canonical_payload,consumption_hash,"
                "previous_record_hash,record_hash,audit_sequence) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    consumption.permit_id, consumption.request_hash,
                    consumption.consumed_at_utc.astimezone(timezone.utc).replace(tzinfo=None),
                    canonical_payload, consumption.consumption_hash, previous_hash,
                    record_hash, sequence,
                ),
            )
            cursor.execute(
                "UPDATE jax_rule_authority.rule_authority_audit_head "
                "SET audit_sequence=%s,head_hash=%s,head_kind='CONSUMPTION',head_key=%s "
                "WHERE singleton=1 AND audit_sequence=%s",
                (sequence, record_hash, consumption.permit_id, audit_sequence),
            )
            if cursor.rowcount != 1:
                raise RuleAuthorityStorageError(
                    "audit head cambió durante el consumo RulePermit"
                )
            transaction_owner._mark_consumption_persisted()
            close_error = transaction_owner._commit_consumption_for_store()
            try:
                self._checkpoint_store.publicar(record_hash, anterior=checkpoint_head)
                if self._checkpoint_store.confirmar(record_hash) is not True:
                    raise RuleAuthorityStorageError(
                        "checkpoint externo no confirma el consumo recién publicado"
                    )
            except Exception as exc:
                raise RuleAuthorityCommittedCheckpointError(
                    "consumo confirmado en DB pero sin checkpoint externo confirmado"
                ) from exc
            if close_error is not None:
                raise RuleAuthorityCommittedCleanupError(
                    "consumo confirmado y anclado pero el cierre posterior al commit falló"
                ) from close_error
            return consumption
        except (AuthorityStateError, RuleAuthorityStorageError):
            raise
        except Exception as exc:
            raise RuleAuthorityStorageError("no se pudo persistir consumo RulePermit") from exc

    @staticmethod
    def _decode_consumption_row(row) -> tuple[RulePermitConsumption, str, int]:
        (permit_id, request_hash, consumed_at, payload_bytes, consumption_hash,
         previous_hash, record_hash, sequence) = row
        try:
            raw = bytes(payload_bytes)
            payload = json.loads(raw.decode("utf-8"))
            if type(payload) is not dict:
                raise ValueError("proyección de consumo inválida")
            if canonical_bytes(payload) != raw:
                raise ValueError("proyección de consumo no canónica")
            if (payload.get("permit_id") != permit_id
                    or payload.get("request_hash") != request_hash
                    or payload.get("consumption_hash") != consumption_hash):
                raise ValueError("columnas de consumo difieren de la proyección")
            consumption = _trusted_consumption(payload)
            projected_time = datetime.fromisoformat(
                payload["consumed_at_utc"].replace("Z", "+00:00")
            )
            if _as_utc(consumed_at) != projected_time.astimezone(timezone.utc):
                raise ValueError("tiempo SQL de consumo no coincide")
            if type(sequence) is not int or sequence < 1:
                raise ValueError("secuencia de consumo inválida")
            if _consumption_record_hash(sequence, permit_id, previous_hash, payload) != record_hash:
                raise ValueError("record hash de consumo no coincide")
            return consumption, record_hash, sequence
        except Exception as exc:
            raise RuleAuthorityStorageError("fila RulePermitConsumption inválida") from exc

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
    def _select_permit(cursor, request_id: str):
        cursor.execute(
            "SELECT permit_id,request_id,request_hash,decision_status,rule_id,rule_path,"
            "rule_blob_oid,rule_content_hash,policy_revision,policy_tree_oid,"
            "policy_snapshot_hash,ratification_event_id,authority_ledger_checkpoint,"
            "stop_checkpoint,capability_id,capability_version,capability_class,"
            "capability_limits,issued_at_utc,expires_at_utc,permit_hash,canonical_payload,"
            "previous_record_hash,record_hash,audit_sequence FROM rule_permits "
            "WHERE request_id=%s FOR UPDATE",
            (request_id,),
        )
        return cursor.fetchone()

    @staticmethod
    def _decode_row(request_id: str, row, expected_catalog=None, *, allow_permit: bool = False) -> RuleDecision:
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
            if status == RuleDecisionStatus.PERMIT.value and not allow_permit:
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
            )
        except Exception as exc:
            raise RuleAuthorityStorageError("fila de decisión inválida") from exc

    @staticmethod
    def _decode_permit_row(row) -> RulePermit:
        (permit_id, request_id, request_hash, decision_status, rule_id, rule_path,
         rule_blob_oid, rule_content_hash, policy_revision, policy_tree_oid,
         policy_snapshot_hash, ratification_event_id, authority_checkpoint,
         stop_checkpoint, capability_id, capability_version, capability_class,
         capability_limits, issued_at, expires_at, stored_permit_hash,
         payload_bytes, previous_hash, stored_record_hash, sequence) = row
        try:
            raw = bytes(payload_bytes)
            payload = json.loads(raw.decode("utf-8"))
            if type(payload) is not dict or frozenset(payload) != _PERMIT_FIELDS:
                raise ValueError("proyección RulePermit cerrada inválida")
            if canonical_bytes(payload) != raw:
                raise ValueError("proyección RulePermit no canónica")
            columns = {
                "permit_id": permit_id,
                "request_id": request_id,
                "request_hash": request_hash,
                "rule_id": rule_id,
                "rule_path": rule_path,
                "rule_blob_oid": rule_blob_oid,
                "rule_content_hash": rule_content_hash,
                "policy_revision": policy_revision,
                "policy_tree_oid": policy_tree_oid,
                "policy_snapshot_hash": policy_snapshot_hash,
                "ratification_event_id": ratification_event_id,
                "capability_id": capability_id,
                "capability_version": capability_version,
                "capability_class": capability_class,
                "permit_hash": stored_permit_hash,
            }
            if decision_status != RuleDecisionStatus.PERMIT.value:
                raise ValueError("RulePermit no enlaza una decisión PERMIT")
            if any(payload[name] != value for name, value in columns.items()):
                raise ValueError("columnas RulePermit difieren de la proyección")
            if (canonical_bytes(payload["authority_ledger_checkpoint"])
                    != bytes(authority_checkpoint)
                    or canonical_bytes(payload["stop_checkpoint"]) != bytes(stop_checkpoint)
                    or canonical_bytes(payload["capability_limits"]) != bytes(capability_limits)):
                raise ValueError("checkpoints/límites no coinciden con RulePermit")
            issued = datetime.fromisoformat(payload["issued_at_utc"].replace("Z", "+00:00"))
            expires = datetime.fromisoformat(payload["expires_at_utc"].replace("Z", "+00:00"))
            if (_as_utc(issued_at) != issued.astimezone(timezone.utc)
                    or _as_utc(expires_at) != expires.astimezone(timezone.utc)):
                raise ValueError("intervalo SQL no coincide con RulePermit")
            if (type(sequence) is not int or sequence < 2
                    or _permit_record_hash(sequence, permit_id, previous_hash, payload)
                    != stored_record_hash):
                raise ValueError("hash/secuencia del registro RulePermit inválido")
            return _trusted_permit(payload)
        except Exception as exc:
            raise RuleAuthorityStorageError("fila RulePermit inválida") from exc
