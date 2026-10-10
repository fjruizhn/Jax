"""Transactional MariaDB adapter for the RuleDecisionStore protocol."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Callable

from jax.faro.catalogo_topes import CatalogoTopes
from policy.authority_ledger.canonical import canonical_bytes, domain_hash
from policy.authority_ledger.errors import AuthorityStateError

from .errors import RuleAuthorityStorageError
from .models import RuleDecision, RuleDecisionStatus, RuleEvaluationRequest
from .permit import (RulePermit, RulePermitConsumption, RulePermitConsumptionDraft,
                     RulePermitDraft, _EvaluatedPermit, _trusted_consumption,
                     _trusted_permit)
from .providers import AlmacenCheckpointAuditoriaBloqueable


_DECISION_DOMAIN = "JAX-FARO-RULE-AUTHORITY-DECISION"
_DECISION_VERSION = "1"
_CATALOG_DOMAIN = "JAX-FARO-RULE-CATALOG"
_CATALOG_VERSION = "1"
_DECISION_FIELDS_V1 = frozenset({
    "request_id", "request_hash", "request_catalog_hash", "status",
    "required_rule_id", "reason_code", "decided_at_utc",
})
_DECISION_FIELDS = _DECISION_FIELDS_V1 | {"request"}
_PERMIT_DOMAIN = "JAX-FARO-RULE-AUTHORITY-PERMIT-RECORD"
_PERMIT_VERSION = "1"
_CONSUMPTION_RECORD_DOMAIN = "JAX-FARO-RULE-AUTHORITY-CONSUMPTION-RECORD"
_CONSUMPTION_RECORD_VERSION = "1"
_PERMIT_FIELDS = frozenset({
    "permit_id", "request_id", "request_hash", "rule_id", "rule_path",
    "rule_blob_oid", "rule_content_hash", "policy_revision", "policy_tree_oid",
    "policy_snapshot_hash", "ratification_event_id", "authority_ledger_checkpoint",
    "stop_checkpoint", "capability_id", "policy_repository_id",
    "capability_contract_id", "capability_version", "capability_class",
    "capability_limits", "issued_at_utc", "expires_at_utc", "permit_hash",
})
_PERMIT_LEGACY_FIELDS = _PERMIT_FIELDS - {"policy_repository_id", "capability_contract_id"}


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


def _decision_projection(decision: RuleDecision, request_catalog_hash: str,
                         request: RuleEvaluationRequest | None = None) -> dict[str, object]:
    projection = {
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
    if request is not None:
        projection["request"] = request.canonical_projection()
    return projection


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
            checkpoint_store, AlmacenCheckpointAuditoriaBloqueable
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

    def recover_durable_evaluation(self, request: RuleEvaluationRequest
                                   ) -> tuple[RuleDecision, RulePermit | None] | None:
        """Read an idempotent result under the RA checkpoint fence only.

        This precedes mutable providers and Block 4 for retries.  A miss is
        merely a miss; initial evaluation must later use the shared owner.
        """
        if type(request) is not RuleEvaluationRequest:
            raise TypeError("recuperación requiere RuleEvaluationRequest exacta")
        if self._checkpoint_store is None:
            raise RuleAuthorityStorageError("recuperación requiere checkpoint Rule Authority durable")
        connection = None
        try:
            with self._checkpoint_store.locked():
                connection = self._connect()
                connection.begin()
                with connection.cursor() as cursor:
                    cursor.execute("SELECT audit_sequence,head_hash FROM rule_authority_audit_head WHERE singleton=1 FOR UPDATE")
                    head = cursor.fetchone()
                    from .trusted_checkpoint import RULE_AUDIT_GENESIS_HEAD
                    if (not isinstance(head, tuple) or type(head[0]) is not int):
                        raise RuleAuthorityStorageError("audit head ausente o inválido")
                    expected = RULE_AUDIT_GENESIS_HEAD if head[0] == 0 and head[1] is None else head[1]
                    if (type(expected) is not str or self._checkpoint_store.head_actual() != expected
                            or self._checkpoint_store.confirmar(expected) is not True):
                        raise RuleAuthorityStorageError("checkpoint externo no coincide con audit head DB")
                    row = self._select_request(cursor, request.request_id)
                    if row is None:
                        connection.commit()
                        return None
                    if (row[0] != request.request_hash or row[1] != _catalog_hash(request.catalogo)
                            or row[2] != request.rule_id):
                        raise AuthorityStateError("request_id reutilizado con otro hash/catálogo/regla")
                    decision = self._decode_row(request.request_id, row, request.catalogo, allow_permit=True)
                    permit = None
                    if decision.status is RuleDecisionStatus.PERMIT:
                        permit_row = self._select_permit(cursor, request.request_id)
                        if permit_row is None:
                            raise RuleAuthorityStorageError("PERMIT recuperado sin RulePermit")
                        permit = self._decode_permit_row(permit_row)
                        if self._checkpoint_store.confirmar(permit_row[-2]) is not True:
                            raise RuleAuthorityStorageError("checkpoint no contiene RulePermit recuperado")
                    elif self._checkpoint_store.confirmar(row[8]) is not True:
                        raise RuleAuthorityStorageError("checkpoint no contiene decisión recuperada")
                connection.commit()
                return decision, permit
        except (AuthorityStateError, RuleAuthorityStorageError):
            if connection is not None:
                connection.rollback()
            raise
        except Exception as exc:
            if connection is not None:
                connection.rollback()
            raise RuleAuthorityStorageError("no se pudo recuperar evaluación durable") from exc
        finally:
            if connection is not None:
                connection.close()

    def record(self, request: RuleEvaluationRequest, decision: RuleDecision) -> RuleDecision:
        """Closed legacy API; evaluations must share Block-4 verification."""
        raise RuleAuthorityStorageError(
            "record standalone está cerrado: use record_evaluation_in_transaction con Block 4"
        )

    def record_permit(self, request: RuleEvaluationRequest, decision: RuleDecision,
                      evaluated_permit: _EvaluatedPermit) -> RulePermit:
        """Closed legacy API; production PERMIT writes require the shared Block-4 owner."""
        raise RuleAuthorityStorageError(
            "record_permit standalone está cerrado: use record_evaluation_in_transaction con Block 4"
        )

    def recover_evaluation_in_transaction(
        self, transaction_owner, request: RuleEvaluationRequest
    ) -> tuple[RuleDecision, RulePermit | None] | None:
        """Recover an exact durable outcome before reading mutable providers.

        This is deliberately called after Block 4 verification and while the
        two checkpoint fences are held.  A hit is a complete idempotent result;
        callers must not re-evaluate the current pin, STOP or clock first.
        A miss leaves the Rule Authority audit row locked for the subsequent
        append on the same EVALUATION owner.
        """
        self._require_evaluation_owner(transaction_owner)
        if type(request) is not RuleEvaluationRequest:
            raise TypeError("recuperación requiere RuleEvaluationRequest exacta")
        request_catalog_hash = _catalog_hash(request.catalogo)
        self._checked_audit_head_for_evaluation(transaction_owner)
        cursor = transaction_owner._cursor
        existing = self._select_request(cursor, request.request_id)
        if existing is None:
            return None
        if (existing[0] != request.request_hash or existing[1] != request_catalog_hash
                or existing[2] != request.rule_id):
            raise AuthorityStateError("request_id reutilizado con otro hash/catálogo/regla")
        stored_decision = self._decode_row(
            request.request_id, existing, request.catalogo, allow_permit=True
        )
        if self._checkpoint_store.confirmar(existing[8]) is not True:
            raise RuleAuthorityStorageError(
                "checkpoint externo no contiene la decisión idempotente")
        if stored_decision.status is not RuleDecisionStatus.PERMIT:
            return stored_decision, None
        permit_row = self._select_permit(cursor, request.request_id)
        if permit_row is None:
            raise RuleAuthorityStorageError("PERMIT existente sin RulePermit durable")
        stored_permit = self._decode_permit_row(permit_row)
        if self._checkpoint_store.confirmar(permit_row[-2]) is not True:
            raise RuleAuthorityStorageError(
                "checkpoint externo no contiene RulePermit idempotente")
        return stored_decision, stored_permit

    def _require_evaluation_owner(self, transaction_owner) -> None:
        from policy.authority_ledger.storage import (
            AuthorityLedgerTransactionPurpose,
            MariaDBAuthorityLedgerTransactionOwner,
        )

        if type(transaction_owner) is not MariaDBAuthorityLedgerTransactionOwner:
            raise AuthorityStateError("evaluación requiere transaction owner MariaDB sellado")
        if transaction_owner._rule_authority_decision_store is not self:
            raise AuthorityStateError("transaction owner no pertenece a este decision-store")
        if (not transaction_owner.active or not transaction_owner._verified
                or transaction_owner.purpose is not AuthorityLedgerTransactionPurpose.EVALUATION
                or transaction_owner._permit_locked):
            raise AuthorityStateError("evaluación requiere owner EVALUATION verificado sin permit lock")
        if self._checkpoint_store is None:
            raise RuleAuthorityStorageError("evaluación requiere checkpoint Rule Authority durable")

    def _checked_audit_head_for_evaluation(self, transaction_owner) -> tuple[int, str | None, str]:
        from .trusted_checkpoint import RULE_AUDIT_GENESIS_HEAD

        if transaction_owner._audit_head_locked:
            head = transaction_owner._locked_audit_head
            if (not isinstance(head, tuple) or len(head) != 2
                    or type(head[0]) is not int):
                raise AuthorityStateError("owner EVALUATION perdió audit head bloqueado")
            audit_sequence, previous_hash = head
        else:
            audit_sequence, previous_hash = transaction_owner.lock_rule_authority_audit_head_for_update()
        checkpoint_head = self._checkpoint_store.head_actual()
        expected_head = (
            RULE_AUDIT_GENESIS_HEAD if audit_sequence == 0 and previous_hash is None
            else previous_hash
        )
        if (type(expected_head) is not str or checkpoint_head != expected_head
                or self._checkpoint_store.confirmar(checkpoint_head) is not True):
            raise RuleAuthorityStorageError(
                "checkpoint externo no coincide con el audit head DB antes de evaluar")
        return audit_sequence, previous_hash, checkpoint_head

    def record_evaluation_in_transaction(
        self,
        transaction_owner,
        request: RuleEvaluationRequest,
        decision: RuleDecision,
        evaluated_permit: _EvaluatedPermit | None,
    ) -> tuple[RuleDecision, RulePermit | None]:
        """Append an evaluated decision under the already-verified shared owner.

        The caller obtains ``transaction_owner`` only from Block 4's checkpoint
        fence.  Thus the Block 4 replay, the Rule Authority audit-row lock, the
        decision (and, for PERMIT, its permit) and the database commit form one
        linearized operation.  The external Rule Authority checkpoint is
        published only after the DB commit and while both checkpoint fences are
        still held; a publication failure is a durable fail-closed outcome.
        """
        self._require_evaluation_owner(transaction_owner)
        if type(request) is not RuleEvaluationRequest or type(decision) is not RuleDecision:
            raise TypeError("evaluación requiere RuleEvaluationRequest y RuleDecision exactas")
        if (decision.request_id != request.request_id
                or decision.request_hash != request.request_hash
                or decision.required_rule_id != request.rule_id):
            raise AuthorityStateError("decisión no corresponde a la solicitud")

        request_catalog_hash = _catalog_hash(request.catalogo)
        decision_projection = _decision_projection(decision, request_catalog_hash, request)
        canonical_decision = canonical_bytes(decision_projection)

        try:
            audit_sequence, previous_hash, checkpoint_head = (
                self._checked_audit_head_for_evaluation(transaction_owner)
            )

            cursor = transaction_owner._cursor
            existing = self._select_request(cursor, request.request_id)
            if existing is not None:
                if (existing[0] != request.request_hash or existing[1] != request_catalog_hash
                        or existing[2] != request.rule_id):
                    raise AuthorityStateError("request_id reutilizado con otro hash/catálogo/regla")
                stored_decision = self._decode_row(
                    request.request_id, existing, request.catalogo, allow_permit=True
                )
                if self._checkpoint_store.confirmar(existing[8]) is not True:
                    raise RuleAuthorityStorageError(
                        "checkpoint externo no contiene la decisión idempotente")
                if stored_decision.status is RuleDecisionStatus.PERMIT:
                    permit_row = self._select_permit(cursor, request.request_id)
                    if permit_row is None:
                        raise RuleAuthorityStorageError("PERMIT existente sin RulePermit durable")
                    stored_permit = self._decode_permit_row(permit_row)
                    if self._checkpoint_store.confirmar(permit_row[-2]) is not True:
                        raise RuleAuthorityStorageError(
                            "checkpoint externo no contiene RulePermit idempotente")
                    permit = stored_permit
                else:
                    permit = None
                transaction_owner._mark_evaluation_persisted()
                close_error = transaction_owner._commit_evaluation_for_store()
                if close_error is not None:
                    raise RuleAuthorityCommittedCleanupError(
                        "evaluación confirmada y anclada pero el cierre posterior al commit falló"
                    ) from close_error
                return stored_decision, permit

            permit, permit_projection, permit_payload = self._validated_evaluation_permit(
                request, decision, evaluated_permit
            )

            decision_sequence = audit_sequence + 1
            decision_hash = _record_hash(
                decision_sequence, request.request_id, previous_hash, decision_projection
            )
            cursor.execute(
                "INSERT INTO rule_decisions "
                "(request_id,request_hash,request_catalog_hash,required_rule_id,status,"
                "reason_code,decided_at_utc,canonical_decision,previous_record_hash,"
                "record_hash,audit_sequence) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (request.request_id, request.request_hash, request_catalog_hash,
                 request.rule_id, decision.status.value, decision.reason_code,
                 decision.decided_at_utc.astimezone(timezone.utc).replace(tzinfo=None),
                 canonical_decision, previous_hash, decision_hash, decision_sequence),
            )

            head_hash = decision_hash
            head_sequence = decision_sequence
            head_kind = "DECISION"
            head_key = request.request_id
            if permit is not None:
                assert permit_projection is not None and permit_payload is not None
                permit_sequence = decision_sequence + 1
                permit_record_hash = _permit_record_hash(
                    permit_sequence, permit.permit_id, decision_hash, permit_projection
                )
                cursor.execute(
                    "INSERT INTO rule_permits "
                    "(permit_id,request_id,request_hash,rule_id,rule_path,rule_blob_oid,"
                    "rule_content_hash,policy_revision,policy_tree_oid,policy_snapshot_hash,"
                    "ratification_event_id,authority_ledger_checkpoint,stop_checkpoint,"
                    "capability_id,policy_repository_id,capability_contract_id,"
                    "capability_version,capability_class,capability_limits,"
                    "issued_at_utc,expires_at_utc,permit_hash,canonical_payload,"
                    "previous_record_hash,record_hash,audit_sequence) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (permit.permit_id, request.request_id, request.request_hash,
                     permit.rule_id, permit.rule_path, permit.rule_blob_oid,
                     permit.rule_content_hash, permit.policy_revision, permit.policy_tree_oid,
                     permit.policy_snapshot_hash, permit.ratification_event_id,
                     canonical_bytes(permit.authority_ledger_checkpoint),
                     canonical_bytes(permit.stop_checkpoint), permit.capability_id,
                     permit.policy_repository_id, permit.capability_contract_id,
                     permit.capability_version, permit.capability_class,
                     canonical_bytes(permit.capability_limits),
                     decision.decided_at_utc.astimezone(timezone.utc).replace(tzinfo=None),
                     permit.expires_at_utc.astimezone(timezone.utc).replace(tzinfo=None),
                     permit.permit_hash, permit_payload, decision_hash, permit_record_hash,
                     permit_sequence),
                )
                head_hash = permit_record_hash
                head_sequence = permit_sequence
                head_kind = "PERMIT"
                head_key = permit.permit_id

            cursor.execute(
                "UPDATE rule_authority_audit_head SET audit_sequence=%s,head_hash=%s,"
                "head_kind=%s,head_key=%s WHERE singleton=1 AND audit_sequence=%s",
                (head_sequence, head_hash, head_kind, head_key, audit_sequence),
            )
            if cursor.rowcount != 1:
                raise RuleAuthorityStorageError("audit head cambió durante la evaluación")
            transaction_owner._mark_evaluation_persisted()
            close_error = transaction_owner._commit_evaluation_for_store()
            try:
                self._checkpoint_store.publicar(head_hash, anterior=checkpoint_head)
                if self._checkpoint_store.confirmar(head_hash) is not True:
                    raise RuleAuthorityStorageError(
                        "checkpoint externo no confirma la evaluación recién publicada")
            except Exception as exc:
                raise RuleAuthorityCommittedCheckpointError(
                    "evaluación confirmada en DB pero sin checkpoint externo confirmado"
                ) from exc
            if close_error is not None:
                raise RuleAuthorityCommittedCleanupError(
                    "evaluación confirmada y anclada pero el cierre posterior al commit falló"
                ) from close_error
            return decision, permit
        except (AuthorityStateError, RuleAuthorityStorageError):
            raise
        except Exception as exc:
            raise RuleAuthorityStorageError("no se pudo persistir evaluación Rule Authority") from exc

    @staticmethod
    def _validated_evaluation_permit(
        request: RuleEvaluationRequest,
        decision: RuleDecision,
        evaluated_permit: _EvaluatedPermit | None,
    ) -> tuple[RulePermit | None, dict[str, object] | None, bytes | None]:
        if decision.status is not RuleDecisionStatus.PERMIT:
            if evaluated_permit is not None:
                raise AuthorityStateError("decisión negativa no admite RulePermit")
            return None, None, None
        if (type(evaluated_permit) is not _EvaluatedPermit
                or not evaluated_permit._is_kernel_sealed()):
            raise AuthorityStateError("PERMIT requiere evaluación sellada del kernel")
        draft = evaluated_permit._draft
        if type(draft) is not RulePermitDraft:
            raise AuthorityStateError("evaluación sellada no contiene RulePermitDraft")
        if (draft.request_id != request.request_id
                or draft.request_hash != request.request_hash
                or draft.rule_id != request.rule_id
                or draft.issued_at_utc.astimezone(timezone.utc)
                != decision.decided_at_utc.astimezone(timezone.utc)):
            raise AuthorityStateError("RulePermit no corresponde a la decisión/request")
        permit = _trusted_permit(draft)
        projection = dict(permit.projection())
        if frozenset(projection) != _PERMIT_FIELDS:
            raise RuleAuthorityStorageError(
                "proyección RulePermit no coincide con el contrato de persistencia")
        return permit, projection, canonical_bytes(projection)

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

    def load_consumption_request_in_transaction(self, transaction_owner, catalog):
        """Load the canonical request only after permit and Block-4 locks are held."""
        from policy.authority_ledger.storage import (
            AuthorityLedgerTransactionPurpose,
            MariaDBAuthorityLedgerTransactionOwner,
        )

        if (type(transaction_owner) is not MariaDBAuthorityLedgerTransactionOwner
                or transaction_owner._rule_authority_decision_store is not self
                or not transaction_owner.active or not transaction_owner._verified
                or transaction_owner.purpose is not AuthorityLedgerTransactionPurpose.CONSUMPTION
                or not transaction_owner._permit_locked):
            raise AuthorityStateError(
                "carga de solicitud requiere permit bloqueado y Block 4 verificado"
            )
        permit = transaction_owner._locked_permit
        if type(permit) is not RulePermit or not permit._is_store_sealed():
            raise AuthorityStateError("permit bloqueado no está sellado por Rule Authority")
        cursor = transaction_owner._cursor
        row = self._select_request(cursor, permit.request_id)
        if row is None:
            raise RuleAuthorityStorageError("RulePermit sin decisión durable")
        decision = self._decode_row(permit.request_id, row, catalog, allow_permit=True)
        if (decision.status is not RuleDecisionStatus.PERMIT
                or decision.request_hash != permit.request_hash
                or decision.required_rule_id != permit.rule_id):
            raise RuleAuthorityStorageError("RulePermit no coincide con decisión PERMIT")
        try:
            decision_payload = json.loads(bytes(row[6]).decode("utf-8"))
            request_projection = decision_payload["request"]
        except Exception as exc:
            raise RuleAuthorityStorageError(
                "decisión durable carece de proyección completa de solicitud"
            ) from exc
        if type(request_projection) is not dict:
            raise RuleAuthorityStorageError("proyección durable de solicitud inválida")

        permit_row = self._select_permit(cursor, permit.request_id)
        if permit_row is None:
            raise RuleAuthorityStorageError("decisión PERMIT sin fila RulePermit")
        persisted_permit = self._decode_permit_row(permit_row)
        if persisted_permit.projection() != permit.projection():
            raise RuleAuthorityStorageError("permit bloqueado difiere de fila durable")
        if (self._checkpoint_store is None
                or self._checkpoint_store.confirmar(row[8]) is not True
                or self._checkpoint_store.confirmar(permit_row[-2]) is not True):
            raise RuleAuthorityStorageError(
                "checkpoint externo no contiene la decisión y RulePermit bloqueados"
            )
        return request_projection

    def recover_consumption(self, permit_id: str, request_hash: str,
                            consumption_request_id: str) -> RulePermitConsumption | None:
        """Return only the exact durable attempt; a different retry ID is denied."""
        if (type(permit_id) is not str or type(request_hash) is not str
                or type(consumption_request_id) is not str):
            raise TypeError("recuperación de consumo requiere identidades string exactas")
        if self._checkpoint_store is None:
            raise RuleAuthorityStorageError("recuperación de consumo requiere checkpoint durable")
        connection = None
        try:
            with self._checkpoint_store.locked():
                connection = self._connect()
                connection.begin()
                with connection.cursor() as cursor:
                    cursor.execute("SELECT audit_sequence,head_hash FROM rule_authority_audit_head WHERE singleton=1 FOR UPDATE")
                    head = cursor.fetchone()
                    from .trusted_checkpoint import RULE_AUDIT_GENESIS_HEAD
                    expected = (RULE_AUDIT_GENESIS_HEAD if head and head[0] == 0 and head[1] is None
                                else (head[1] if head else None))
                    if (type(expected) is not str or self._checkpoint_store.head_actual() != expected
                            or self._checkpoint_store.confirmar(expected) is not True):
                        raise RuleAuthorityStorageError("checkpoint externo no coincide con audit head DB")
                    cursor.execute(
                        "SELECT permit_id,request_hash,consumed_at_utc,canonical_payload,consumption_hash,"
                        "previous_record_hash,record_hash,audit_sequence,consumption_request_id "
                        "FROM rule_permit_consumptions WHERE permit_id=%s FOR UPDATE", (permit_id,))
                    row = cursor.fetchone()
                    if row is None:
                        connection.commit()
                        return None
                    consumption, record_hash, _sequence = self._decode_consumption_row(row)
                    if consumption.request_hash != request_hash:
                        raise AuthorityStateError("permit ya consumido con otro request_hash")
                    if consumption.consumption_request_id != consumption_request_id:
                        raise AuthorityStateError("permit ya consumido con consumption_request_id distinto")
                    if self._checkpoint_store.confirmar(record_hash) is not True:
                        raise RuleAuthorityStorageError("checkpoint no contiene consumo recuperado")
                connection.commit()
                return consumption
        except (AuthorityStateError, RuleAuthorityStorageError):
            if connection is not None:
                connection.rollback()
            raise
        except Exception as exc:
            if connection is not None:
                connection.rollback()
            raise RuleAuthorityStorageError("no se pudo recuperar consumo durable") from exc
        finally:
            if connection is not None:
                connection.close()

    def recover_consumption_in_transaction(self, transaction_owner, request_hash: str,
                                           consumption_request_id: str
                                           ) -> RulePermitConsumption | None:
        """Recover the exact retry after its permit row is locked.

        A competing consume can commit between the initial durable lookup and
        this owner's ``FOR UPDATE`` permit lock.  Its exact caller retry must
        return before current authority is read; a different request hash or
        caller retry ID remains a closed denial.
        """
        from policy.authority_ledger.storage import (
            AuthorityLedgerTransactionPurpose,
            MariaDBAuthorityLedgerTransactionOwner,
        )

        if type(transaction_owner) is not MariaDBAuthorityLedgerTransactionOwner:
            raise AuthorityStateError("recuperación de consumo requiere transaction owner MariaDB sellado")
        if transaction_owner._rule_authority_decision_store is not self:
            raise AuthorityStateError("transaction owner no pertenece a este decision-store")
        if (not transaction_owner.active
                or not transaction_owner._verified
                or transaction_owner.purpose is not AuthorityLedgerTransactionPurpose.CONSUMPTION
                or not transaction_owner._permit_locked):
            raise AuthorityStateError(
                "recuperación de consumo requiere permit bloqueado y Block 4 verificado"
            )
        locked_permit = transaction_owner._locked_permit
        if (type(locked_permit) is not RulePermit or not locked_permit._is_store_sealed()
                or locked_permit.permit_id != transaction_owner._locked_permit_identity[0]
                or locked_permit.request_hash != transaction_owner._locked_permit_identity[1]):
            raise AuthorityStateError("permit bloqueado no está sellado para recuperación")
        if type(request_hash) is not str or type(consumption_request_id) is not str:
            raise TypeError("recuperación de consumo requiere identidades string exactas")
        if transaction_owner._audit_head_locked:
            locked_head = transaction_owner._locked_audit_head
            if (type(locked_head) is not tuple or len(locked_head) != 2
                    or type(locked_head[0]) is not int):
                raise AuthorityStateError("owner perdió audit head bloqueado durante recuperación")
            audit_sequence, db_head = locked_head
        else:
            audit_sequence, db_head = transaction_owner.lock_rule_authority_audit_head_for_update()
        from .trusted_checkpoint import RULE_AUDIT_GENESIS_HEAD

        expected_head = (
            RULE_AUDIT_GENESIS_HEAD if audit_sequence == 0 and db_head is None else db_head
        )
        checkpoint_head = self._checkpoint_store.head_actual() if self._checkpoint_store else None
        if (type(expected_head) is not str or checkpoint_head != expected_head
                or self._checkpoint_store.confirmar(checkpoint_head) is not True):
            raise RuleAuthorityStorageError(
                "checkpoint externo no coincide con el audit head DB durante recuperación"
            )
        cursor = transaction_owner._cursor
        cursor.execute(
            "SELECT permit_id,request_hash,consumed_at_utc,canonical_payload,consumption_hash,"
            "previous_record_hash,record_hash,audit_sequence,consumption_request_id "
            "FROM jax_rule_authority.rule_permit_consumptions WHERE permit_id=%s FOR UPDATE",
            (locked_permit.permit_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        consumption, record_hash, _sequence = self._decode_consumption_row(row)
        if (consumption.request_hash != request_hash
                or consumption.request_hash != locked_permit.request_hash):
            raise AuthorityStateError("permit ya consumido con otro request_hash")
        if consumption.consumption_request_id != consumption_request_id:
            raise AuthorityStateError("permit ya consumido con consumption_request_id distinto")
        if self._checkpoint_store.confirmar(record_hash) is not True:
            raise RuleAuthorityStorageError("checkpoint no contiene consumo recuperado")
        return consumption

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
        if locked_permit._has_consumption_bindings() is not True:
            raise AuthorityStateError(
                "RulePermit histórico sin binding de repositorio/contrato no es consumible"
            )
        if self._checkpoint_store is None:
            raise RuleAuthorityStorageError("consumo requiere checkpoint Rule Authority durable")

        consumption = _trusted_consumption(consumption_draft)
        projection = dict(consumption.projection())
        canonical_payload = canonical_bytes(projection)
        try:
            if transaction_owner._audit_head_locked:
                locked_head = transaction_owner._locked_audit_head
                if (type(locked_head) is not tuple or len(locked_head) != 2
                        or type(locked_head[0]) is not int):
                    raise AuthorityStateError("owner perdió audit head bloqueado durante consumo")
                audit_sequence, previous_hash = locked_head
            else:
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
                "previous_record_hash,record_hash,audit_sequence,consumption_request_id "
                "FROM jax_rule_authority.rule_permit_consumptions "
                "WHERE permit_id=%s FOR UPDATE",
                (consumption.permit_id,),
            )
            existing = cursor.fetchone()
            if existing is not None:
                stored, stored_record_hash, stored_sequence = self._decode_consumption_row(existing)
                # A retry is recoverable only with the durable caller attempt ID.
                if (stored.permit_id != locked_permit.permit_id
                        or stored.request_hash != locked_permit.request_hash
                        or stored.permit_id != consumption_draft.permit_id
                        or stored.request_hash != consumption_draft.request_hash
                        or stored.consumption_request_id != consumption_draft.consumption_request_id):
                    raise AuthorityStateError("permit ya consumido con consumption_request_id distinto")
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
                "(permit_id,consumption_request_id,request_hash,consumed_at_utc,canonical_payload,"
                "consumption_hash,previous_record_hash,record_hash,audit_sequence) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    consumption.permit_id, consumption.consumption_request_id,
                    consumption.request_hash,
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
         previous_hash, record_hash, sequence, consumption_request_id) = row
        try:
            raw = bytes(payload_bytes)
            payload = json.loads(raw.decode("utf-8"))
            if type(payload) is not dict:
                raise ValueError("proyección de consumo inválida")
            if canonical_bytes(payload) != raw:
                raise ValueError("proyección de consumo no canónica")
            if (payload.get("permit_id") != permit_id
                    or payload.get("request_hash") != request_hash
                    or payload.get("consumption_hash") != consumption_hash
                    or payload.get("consumption_request_id") != consumption_request_id):
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
            "stop_checkpoint,capability_id,policy_repository_id,capability_contract_id,"
            "capability_version,capability_class,"
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
            if (not isinstance(payload, dict)
                    or frozenset(payload) not in (_DECISION_FIELDS_V1, _DECISION_FIELDS)):
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
            if frozenset(payload) == _DECISION_FIELDS:
                if expected_catalog is None:
                    raise ValueError("decisión v2 requiere catálogo sellado para verificarla")
                request_projection = payload["request"]
                if type(request_projection) is not dict:
                    raise ValueError("proyección completa de solicitud inválida")
                request_values = dict(request_projection)
                if request_values.pop("catalogo_oid", None) != expected_catalog.oid_pin:
                    raise ValueError("proyección completa requiere otro catálogo")
                request_values["catalogo"] = expected_catalog
                rehydrated_request = RuleEvaluationRequest(**request_values)
                if rehydrated_request.canonical_projection() != request_projection:
                    raise ValueError("proyección completa de solicitud no coincide")
                if rehydrated_request.request_hash != request_hash:
                    raise ValueError("request_hash no coincide con la proyección persistida")
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
         stop_checkpoint, capability_id, policy_repository_id, capability_contract_id,
         capability_version, capability_class,
         capability_limits, issued_at, expires_at, stored_permit_hash,
         payload_bytes, previous_hash, stored_record_hash, sequence) = row
        try:
            raw = bytes(payload_bytes)
            payload = json.loads(raw.decode("utf-8"))
            if (type(payload) is not dict
                    or frozenset(payload) not in (_PERMIT_FIELDS, _PERMIT_LEGACY_FIELDS)):
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
            if frozenset(payload) == _PERMIT_FIELDS:
                columns["policy_repository_id"] = policy_repository_id
                columns["capability_contract_id"] = capability_contract_id
            elif policy_repository_id is not None or capability_contract_id is not None:
                raise ValueError("RulePermit histórico tiene bindings SQL parciales")
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
