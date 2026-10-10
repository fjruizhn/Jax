"""Fail-closed Rule Authority evaluation boundary.

Durable recovery is intentionally first: retrying an already committed
``request_id`` returns its sealed prior result without consulting mutable
providers again. New permits are emitted only after the trusted snapshot,
ratification, scope, individual limits and provider leases all agree.
"""
from __future__ import annotations

from datetime import timedelta
import uuid

from policy.authority_ledger.errors import AuthorityEventValidationError, AuthorityStateError
from policy.authority_ledger.ids import uuid7_text
from policy.authority_ledger.storage import AuthorityLedgerTransactionPurpose
from jax.faro.git_objetos import es_ancestro

from .errors import RuleAuthorityStorageError
from .models import (RuleDecision, RuleDecisionStatus, RuleEvaluationOutcome,
                     RuleEvaluationRequest, limites_de)
from .permit import (RulePermitConsumptionDraft, RulePermitDraft,
                     RulePermitConsumption, _permit_after_kernel_evaluation)
from .providers import (ClasificacionDesconocida, FormaLimites, leases_de_emision)
from .snapshot import load_trusted_policy_snapshot
from .storage import MariaDBRuleDecisionStore


class RuleAuthorityKernel:
    """Coordinate only a verified Block-4/Rule-Authority transaction.

    Dependencies remain explicit so evaluation cannot take caller supplied
    pin, STOP, clock or capability classification. They are not read by
    durable retry recovery.
    """

    def __init__(self, *, active_policy_pin_provider, authority_ledger_store,
                 trusted_root, trusted_checkpoint_store, stop_provider,
                 trusted_clock, capability_classification_provider,
                 rule_authority_store: MariaDBRuleDecisionStore,
                 ttl_max_seconds: int = 3600) -> None:
        if type(rule_authority_store) is not MariaDBRuleDecisionStore:
            raise TypeError("kernel requiere MariaDBRuleDecisionStore canónico")
        if type(ttl_max_seconds) is not int or ttl_max_seconds <= 0:
            raise ValueError("ttl_max_seconds debe ser entero positivo")
        self._pin_provider = active_policy_pin_provider
        self._ledger_store = authority_ledger_store
        self._trusted_root = trusted_root
        self._trusted_checkpoint_store = trusted_checkpoint_store
        self._stop_provider = stop_provider
        self._clock = trusted_clock
        self._classification_provider = capability_classification_provider
        self._store = rule_authority_store
        self._ttl_max_seconds = ttl_max_seconds

    def evaluate(self, request: RuleEvaluationRequest) -> RuleEvaluationOutcome:
        """Recover a durable evaluation; fail closed for an initial request.

        An exact retry is recovered under the Rule Authority checkpoint fence
        before live providers are read. A miss is rechecked after Block 4
        verification inside the shared owner before any new evaluation, so a
        concurrent first request cannot create a second decision or permit.
        """
        if type(request) is not RuleEvaluationRequest:
            raise TypeError("evaluate requiere RuleEvaluationRequest exacta")
        recovered = self._store.recover_durable_evaluation(request)
        if recovered is not None:
            return RuleEvaluationOutcome(*recovered)
        with leases_de_emision(
            pin=self._pin_provider, stop=self._stop_provider, reloj=self._clock,
            clasificacion=self._classification_provider,
        ) as vistas:
            pin_activo = vistas.pin.valor
            snapshot = load_trusted_policy_snapshot(
                pin_activo.repositorio.repository_path, pin_activo.pin
            )
            with self._ledger_store.checkpoint_fence(self._trusted_checkpoint_store) as fence:
                with fence.transaction_owner(
                    self._store, purpose=AuthorityLedgerTransactionPurpose.EVALUATION
                ) as owner:
                    state = fence.verify_for_update(owner, self._trusted_root)
                    recovered = self._store.recover_evaluation_in_transaction(owner, request)
                    if recovered is not None:
                        decision, _permit = recovered
                        decision, permit = self._store.record_evaluation_in_transaction(
                            owner, request, decision, None
                        )
                        return RuleEvaluationOutcome(decision, permit)
                    decision, evaluated_permit = self._initial_evaluation(
                        request, snapshot, state, vistas, pin_activo.repositorio.repository_path
                    )
                    decision, permit = self._store.record_evaluation_in_transaction(
                        owner, request, decision, evaluated_permit
                    )
                    return RuleEvaluationOutcome(decision, permit)

    def consume(self, permit_id: str, expected_request_hash: str,
                consumption_request_id: str) -> RulePermitConsumption:
        """Consume one permit after revalidating all live authority under leases.

        An exact durable retry is returned first: it describes an already
        committed consume and cannot authorize another effect. A new consume
        holds pin/classification/STOP leases through the shared Block-4 and
        Rule-Authority transaction and its external audit checkpoint.
        """
        try:
            uuid7_text(permit_id, "permit_id")
            uuid7_text(consumption_request_id, "consumption_request_id")
        except Exception as exc:
            raise AuthorityStateError("identidad de consumo inválida") from exc
        if (type(expected_request_hash) is not str
                or len(expected_request_hash) != 71
                or not expected_request_hash.startswith("sha256:")
                or any(ch not in "0123456789abcdef" for ch in expected_request_hash[7:])):
            raise AuthorityStateError("expected_request_hash inválido")

        recovered = self._store.recover_consumption(
            permit_id, expected_request_hash, consumption_request_id
        )
        if recovered is not None:
            if (type(recovered) is not RulePermitConsumption
                    or not recovered._is_store_sealed()):
                raise RuleAuthorityStorageError(
                    "recuperación de consumo no devolvió objeto sellado del store"
                )
            return recovered

        with leases_de_emision(
            pin=self._pin_provider, stop=self._stop_provider, reloj=self._clock,
            clasificacion=self._classification_provider,
        ) as vistas:
            pin_activo = vistas.pin.valor
            snapshot = load_trusted_policy_snapshot(
                pin_activo.repositorio.repository_path, pin_activo.pin
            )
            with self._ledger_store.checkpoint_fence(self._trusted_checkpoint_store) as fence:
                with fence.transaction_owner(
                    self._store, purpose=AuthorityLedgerTransactionPurpose.CONSUMPTION
                ) as owner:
                    permit = owner.lock_permit_for_consumption(
                        permit_id, expected_request_hash
                    )
                    state = fence.verify_for_update(owner, self._trusted_root)
                    projection = self._store.load_consumption_request_in_transaction(
                        owner, snapshot.catalogo
                    )
                    request = self._request_from_projection(
                        projection, snapshot.catalogo, expected_request_hash
                    )
                    self._validate_consumption_context(
                        permit=permit, request=request, snapshot=snapshot,
                        state=state, vistas=vistas,
                        repository_path=pin_activo.repositorio.repository_path,
                    )
                    draft = RulePermitConsumptionDraft(
                        permit_id=permit_id,
                        consumption_request_id=consumption_request_id,
                        request_hash=expected_request_hash,
                        consumed_at_utc=vistas.ahora_utc,
                    )
                    consumption = self._store.consume_permit(owner, draft)
                    if (type(consumption) is not RulePermitConsumption
                            or not consumption._is_store_sealed()):
                        raise RuleAuthorityStorageError(
                            "append de consumo no devolvió objeto sellado del store"
                        )
                    return consumption

    def _validate_consumption_context(self, *, permit, request, snapshot, state,
                                      vistas, repository_path) -> None:
        from .permit import RulePermit

        if (type(permit) is not RulePermit or not permit._is_store_sealed()
                or not permit._has_consumption_bindings()):
            raise AuthorityStateError("RulePermit no es actual o carece de bindings de consumo")
        now = vistas.ahora_utc
        if (permit.request_id != request.request_id
                or permit.request_hash != request.request_hash
                or permit.rule_id != request.rule_id
                or permit.capability_id != request.capability
                or permit.policy_repository_id != snapshot.repositorio
                or permit.policy_repository_id != vistas.pin.valor.repositorio.repository_id):
            raise AuthorityStateError("RulePermit no coincide con su solicitud/repositorio")
        if now < permit.issued_at_utc or now >= permit.expires_at_utc:
            raise AuthorityStateError("RulePermit fuera de vigencia")
        if vistas.stop.valor.activo:
            raise AuthorityStateError("STOP activo durante consumo RulePermit")

        sealed = next((entry for entry in snapshot.reglas
                       if entry.regla.rule_id == permit.rule_id), None)
        if (sealed is None or sealed.ruta != permit.rule_path
                or sealed.blob_oid != permit.rule_blob_oid
                or sealed.content_hash != permit.rule_content_hash):
            raise AuthorityStateError("regla actual no coincide con RulePermit")
        rule = sealed.regla
        if request.rule_hash is not None:
            try:
                current_rule_hash = limites_de(rule, snapshot.catalogo).rule_hash
            except AuthorityEventValidationError as exc:
                raise AuthorityStateError(
                    "regla actual sin límites no puede consumir RulePermit"
                ) from exc
            if request.rule_hash != current_rule_hash:
                raise AuthorityStateError("request rule_hash cambió desde la evaluación")
        if es_ancestro(repository_path, permit.policy_revision, snapshot.commit) is not True:
            raise AuthorityStateError("pin actual no desciende de la revisión del RulePermit")

        grant_event = state.latest_unrevoked_rule_ratification(permit.rule_id)
        if grant_event is None or grant_event.event_id != permit.ratification_event_id:
            raise AuthorityStateError("ratificación del RulePermit ya no es la vigente")
        grant = grant_event.intent.rule_ratification
        if (grant is None
                or not self._grant_matches(grant, sealed, snapshot, now, repository_path)):
            raise AuthorityStateError("ratificación vigente no valida la misma regla")
        if (now < rule.validity.not_before_utc
                or (rule.validity.not_after_utc is not None
                    and now >= rule.validity.not_after_utc)):
            raise AuthorityStateError("regla fuera de vigencia al consumir RulePermit")

        try:
            contract = vistas.clasificacion.valor.contrato_de(request.capability)
        except ClasificacionDesconocida as exc:
            raise AuthorityStateError("capability sin contrato vigente") from exc
        if (contract.identidad != permit.capability_contract_id
                or contract.version != permit.capability_version
                or contract.clase.value != permit.capability_class
                or contract.clase.value != rule.action_class
                or request.subject not in rule.scope.subjects
                or request.capability not in rule.scope.capabilities
                or request.objective not in rule.scope.objectives):
            raise AuthorityStateError("contrato de capability cambió desde la emisión")
        limits, reason = self._validate_request_limits(request, rule, contract)
        if reason is not None or limits != permit.capability_limits:
            raise AuthorityStateError("límites de RulePermit ya no coinciden con la regla/contrato")

    @staticmethod
    def _deny(request: RuleEvaluationRequest, now, reason: str) -> RuleDecision:
        return RuleDecision(request.request_id, request.request_hash, RuleDecisionStatus.DENY,
                            request.rule_id, reason, now)

    def _initial_evaluation(self, request, snapshot, state, vistas, repository_path):
        now = vistas.ahora_utc
        if vistas.stop.valor.activo:
            return self._deny(request, now, "STOP_ACTIVE"), None
        sealed = next((entry for entry in snapshot.reglas if entry.regla.rule_id == request.rule_id), None)
        if sealed is None:
            return (RuleDecision(request.request_id, request.request_hash,
                                 RuleDecisionStatus.MISSING_RULE, request.rule_id,
                                 "RULE_NOT_FOUND", now), None)
        rule = sealed.regla
        if request.rule_hash is not None:
            try:
                current_rule_hash = limites_de(rule, snapshot.catalogo).rule_hash
            except AuthorityEventValidationError:
                return self._deny(request, now, "AUTHORITY_INVALID"), None
            if request.rule_hash != current_rule_hash:
                return self._deny(request, now, "RULE_CHANGED"), None
        grant_event = state.latest_unrevoked_rule_ratification(request.rule_id)
        if grant_event is None:
            return self._deny(request, now, "AUTHORITY_INVALID"), None
        grant = grant_event.intent.rule_ratification
        if grant is None or not self._grant_matches(grant, sealed, snapshot, now, repository_path):
            return self._deny(request, now, "AUTHORITY_INVALID"), None
        if now < rule.validity.not_before_utc or (rule.validity.not_after_utc is not None and now >= rule.validity.not_after_utc):
            return self._deny(request, now, "RULE_EXPIRED"), None
        try:
            contract = vistas.clasificacion.valor.contrato_de(request.capability)
        except ClasificacionDesconocida:
            return self._deny(request, now, "CAPABILITY_MISMATCH"), None
        if (contract.clase.value != rule.action_class or request.subject not in rule.scope.subjects
                or request.capability not in rule.scope.capabilities
                or request.objective not in rule.scope.objectives):
            reason = ("CAPABILITY_MISMATCH" if contract.clase.value != rule.action_class
                      else "OUT_OF_SCOPE")
            return self._deny(request, now, reason), None
        capability_limits, reason = self._validate_request_limits(request, rule, contract)
        if reason is not None:
            return self._deny(request, now, reason), None
        ttl_seconds = min(rule.permit.ttl_seconds, self._ttl_max_seconds)
        expires_at = now + timedelta(seconds=ttl_seconds)
        if rule.validity.not_after_utc is not None:
            expires_at = min(expires_at, rule.validity.not_after_utc)
        if grant.valid_until_utc is not None:
            expires_at = min(expires_at, grant.valid_until_utc)
        if expires_at <= now:
            return self._deny(request, now, "RULE_EXPIRED"), None
        self._clock.marcar_emision(now)
        creator = getattr(uuid, "uuid7", None)
        if creator is None:
            raise RuleAuthorityStorageError("runtime no provee UUIDv7 para RulePermit")
        draft = RulePermitDraft(
            permit_id=str(creator()), request_id=request.request_id,
            request_hash=request.request_hash, rule_id=rule.rule_id,
            rule_path=sealed.ruta, rule_blob_oid=sealed.blob_oid,
            rule_content_hash=sealed.content_hash,
            policy_revision=snapshot.commit, policy_tree_oid=snapshot.policy_tree_oid,
            policy_snapshot_hash=snapshot.snapshot_hash,
            ratification_event_id=grant_event.event_id,
            authority_ledger_checkpoint=state.checkpoint.projection(),
            stop_checkpoint={"version": vistas.stop.version.numero,
                             "fingerprint": vistas.stop.valor.huella},
            capability_id=request.capability,
            policy_repository_id=snapshot.repositorio,
            capability_contract_id=contract.identidad,
            capability_version=contract.version, capability_class=contract.clase.value,
            capability_limits=capability_limits, issued_at_utc=now,
            expires_at_utc=expires_at,
        )
        decision = RuleDecision(request.request_id, request.request_hash,
                                RuleDecisionStatus.PERMIT, request.rule_id, None, now)
        return decision, _permit_after_kernel_evaluation(draft)

    @staticmethod
    def _validate_request_limits(request, rule, contract):
        """Return the exact capability limits projection, or a stable denial.

        F1.1 has no schema for matching arbitrary arguments and no durable
        accumulator for frequency/tope. Therefore those requests cannot
        authorize an effect. ``resource_id`` remains bound by request_hash and
        the durable decision projection.
        """
        if request.arguments:
            return None, "OUT_OF_SCOPE"
        obligation = rule.obligation_limits
        if obligation.frequency is not None or rule.tope is not None:
            return None, "AUTHORITY_INVALID"
        if obligation.quantity is None and obligation.amount is None:
            return None, "AUTHORITY_INVALID"

        if obligation.quantity is not None:
            quantity = obligation.quantity
            if (contract.forma_limites is not FormaLimites.CANTIDAD
                    or contract.unidad != quantity.unit
                    or request.quantity is None
                    or request.quantity_unit != quantity.unit
                    or request.quantity > quantity.max
                    or request.amount is not None):
                return None, "OUT_OF_SCOPE"
            return {"form": "CANTIDAD", "unit": quantity.unit, "max": quantity.max}, None

        if obligation.amount is not None:
            amount = obligation.amount
            if (contract.forma_limites is not FormaLimites.MONTO
                    or contract.moneda != amount.currency
                    or request.amount is None
                    or request.currency is None
                    or request.currency.upper() != amount.currency
                    or request.amount > amount.max
                    or request.quantity is not None):
                return None, "OUT_OF_SCOPE"
            return {"form": "MONTO", "currency": amount.currency, "max": amount.max}, None

        return None, "OUT_OF_SCOPE"

    @staticmethod
    def _request_from_projection(projection, catalog, expected_request_hash):
        """Rebuild a request only from its closed durable v2 projection."""
        from policy.authority_ledger.errors import AuthorityEventValidationError

        fields = frozenset({
            "request_id", "rule_id", "subject", "capability", "objective",
            "resource_id", "arguments", "quantity", "quantity_unit", "amount",
            "currency", "catalogo_oid", "rule_hash",
        })
        if type(projection) is not dict or frozenset(projection) != fields:
            raise RuleAuthorityStorageError("proyección durable de solicitud fuera de esquema")
        if (type(projection["catalogo_oid"]) is not str
                or projection["catalogo_oid"] != getattr(catalog, "oid_pin", None)):
            raise RuleAuthorityStorageError("proyección durable requiere otro catálogo")
        values = {key: value for key, value in projection.items() if key != "catalogo_oid"}
        values["catalogo"] = catalog
        try:
            request = RuleEvaluationRequest(**values)
        except AuthorityEventValidationError as exc:
            raise RuleAuthorityStorageError("proyección durable de solicitud inválida") from exc
        if request.request_hash != expected_request_hash:
            raise RuleAuthorityStorageError("request_hash no coincide con la proyección durable")
        return request

    @staticmethod
    def _grant_matches(grant, sealed, snapshot, now, repository_path) -> bool:
        if (grant.rule_id != sealed.regla.rule_id or grant.rule_path != sealed.ruta
                or grant.rule_blob_oid != sealed.blob_oid
                or grant.rule_content_hash != sealed.content_hash
                or now < grant.valid_from_utc
                or (grant.valid_until_utc is not None and now >= grant.valid_until_utc)):
            return False
        return es_ancestro(repository_path, grant.ratified_policy_revision, snapshot.commit) is True


__all__ = ["RuleAuthorityKernel"]
