from __future__ import annotations

import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock
from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from policy.canonicalization.strict_yaml import load_strict_yaml
from policy.rule_authority.kernel import RuleAuthorityKernel
from policy.rule_authority.models import (
    RuleDecisionStatus, RuleEvaluationRequest, _hash_de_regla,
)
from policy.rule_authority.storage import RuleAuthorityStorageError
from policy.rule_authority.permit import (
    RulePermitConsumptionDraft, RulePermitDraft, _trusted_consumption, _trusted_permit,
)
from policy.rule_authority.storage import MariaDBRuleDecisionStore
from policy.authority_ledger.errors import AuthorityStateError
from policy.authority_ledger.storage import AuthorityLedgerTransactionPurpose
from policy.rule_authority.providers import (
    ClaseCapability, ContratoCapability, EstadoStop, FormaLimites, PinActivo,
    RepositorioConfiable,
)
from policy.rule_authority.schema import validar_regla
from policy.rule_authority.snapshot import TrustedPolicyPin
from tests.policy.catalogo_pin import catalogo_del_pin
from tests.policy.proveedores_dobles import (
    ClasificacionFija, PinFijo, RelojDeterminista, StopFijo,
)

FIXTURES = Path(__file__).parent / "fixtures" / "faro_rules"
CLASES = {"monto_dinero": ["hnl", "usd"],
          "actos_externos": ["mensajes", "correos", "publicaciones", "compras", "pagos"],
          "frecuencia": ["por_hora", "por_dia"], "duracion": ["segundos"],
          "tokens_costo": ["tokens", "usd"]}
CATALOG = catalogo_del_pin(json.dumps({"version": 1, "decision": "kernel tests",
                                      "clases": CLASES}).encode())


def _request(**updates):
    fields = dict(request_id="0199f8a1-8c00-7000-8000-000000000031",
                  rule_id="ejemplo-regla", subject="actor:ejemplo",
                  capability="mail.send", objective="objetivo-ejemplo",
                  resource_id="mail:1", arguments={}, catalogo=CATALOG)
    fields.update(updates)
    return RuleEvaluationRequest(**fields)


def _rule(name: str, **updates):
    data = copy.deepcopy(load_strict_yaml((FIXTURES / name).read_text()))
    if "quantity" in updates:
        data["obligation_limits"]["quantity"] = updates["quantity"]
    return validar_regla(data, catalogo=CATALOG)


def test_reversible_rule_without_any_limit_fails_closed():
    rule = _rule("regla-ejemplo.yaml")
    contract = ContratoCapability(identidad="mail.send.contract.v1", version="1",
                                  clase=ClaseCapability.REVERSIBLE, unidad=None, moneda=None,
                                  forma_limites=FormaLimites.NINGUNA)

    limits, reason = RuleAuthorityKernel._validate_request_limits(_request(), rule, contract)

    assert limits is None
    assert reason == "AUTHORITY_INVALID"


def test_request_exceeding_rule_quantity_is_denied():
    rule = _rule("regla-ejemplo.yaml",
                 quantity={"unit": "mensajes", "max": 3})
    contract = ContratoCapability(identidad="mail.send.contract.v1", version="1",
                                  clase=ClaseCapability.REVERSIBLE, unidad="mensajes", moneda=None,
                                  forma_limites=FormaLimites.CANTIDAD)

    limits, reason = RuleAuthorityKernel._validate_request_limits(
        _request(quantity=4, quantity_unit="mensajes"), rule, contract
    )

    assert limits is None
    assert reason == "OUT_OF_SCOPE"


def test_request_arguments_without_closed_rule_matcher_are_denied():
    rule = _rule("regla-ejemplo.yaml")
    contract = ContratoCapability(identidad="mail.send.contract.v1", version="1",
                                  clase=ClaseCapability.REVERSIBLE, unidad=None, moneda=None,
                                  forma_limites=FormaLimites.NINGUNA)

    limits, reason = RuleAuthorityKernel._validate_request_limits(
        _request(arguments={"recipient": "client@example.test"}), rule, contract
    )

    assert limits is None
    assert reason == "OUT_OF_SCOPE"


def test_frequency_rule_fails_closed_without_transactional_counter():
    rule = _rule("regla-ejemplo-tope.yaml",
                 quantity={"unit": "mensajes", "max": 3})
    contract = ContratoCapability(identidad="mail.send.contract.v1", version="1",
                                  clase=ClaseCapability.OBLIGATING, unidad="mensajes", moneda=None,
                                  forma_limites=FormaLimites.CANTIDAD)

    limits, reason = RuleAuthorityKernel._validate_request_limits(
        _request(quantity=1, quantity_unit="mensajes"), rule, contract
    )

    assert limits is None
    assert reason == "AUTHORITY_INVALID"


def test_unbounded_rule_with_request_hash_returns_stable_authority_deny(monkeypatch):
    now = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
    rule = _rule("regla-ejemplo.yaml")
    request = _request(capability="CAPABILITY_ID", rule_hash=_hash_de_regla(rule))
    sealed = SimpleNamespace(regla=rule, ruta="policy/faro/ejemplo.yaml",
                             blob_oid="a" * 40, content_hash="sha256:" + "c" * 64)
    snapshot = SimpleNamespace(commit="b" * 40, repositorio="jax", reglas=(sealed,),
                               catalogo=CATALOG)
    ratification = SimpleNamespace(valid_from_utc=now.replace(year=now.year - 1),
                                   valid_until_utc=None)
    grant_event = SimpleNamespace(event_id="0199f8a1-8c00-7000-8000-000000000061",
                                  intent=SimpleNamespace(rule_ratification=ratification))
    state = SimpleNamespace(
        latest_unrevoked_rule_ratification=lambda _rule_id: grant_event,
    )
    contract = ContratoCapability(
        identidad="mail.send.contract.v1", version="1",
        clase=ClaseCapability.REVERSIBLE, unidad=None, moneda=None,
        forma_limites=FormaLimites.NINGUNA,
    )
    views = SimpleNamespace(
        ahora_utc=now,
        stop=SimpleNamespace(valor=SimpleNamespace(activo=False)),
        clasificacion=SimpleNamespace(valor=SimpleNamespace(
            contrato_de=lambda _capability: contract)),
    )
    monkeypatch.setattr(RuleAuthorityKernel, "_grant_matches", staticmethod(lambda *_args: True))
    kernel = object.__new__(RuleAuthorityKernel)

    decision, permit = kernel._initial_evaluation(
        request, snapshot, state, views, Path("/tmp"),
    )

    assert decision.status is RuleDecisionStatus.DENY
    assert decision.reason_code == "AUTHORITY_INVALID"
    assert permit is None


def test_durable_request_projection_reconstructs_exactly_under_the_current_catalog():
    original = _request()

    recovered = RuleAuthorityKernel._request_from_projection(
        original.canonical_projection(), CATALOG, original.request_hash
    )

    assert recovered.canonical_projection() == original.canonical_projection()
    assert recovered.request_hash == original.request_hash


def test_durable_request_projection_rejects_catalog_or_hash_drift():
    original = _request()
    projection = original.canonical_projection()
    projection["arguments"] = {"unbound": True}

    with pytest.raises(RuleAuthorityStorageError, match="request_hash"):
        RuleAuthorityKernel._request_from_projection(
            projection, CATALOG, original.request_hash
        )
    other_catalog = catalogo_del_pin(json.dumps({"version": 1, "decision": "other",
                                                 "clases": CLASES}).encode())
    with pytest.raises(RuleAuthorityStorageError, match="catálogo"):
        RuleAuthorityKernel._request_from_projection(
            original.canonical_projection(), other_catalog, original.request_hash
        )


def test_consume_recovers_exact_durable_retry_before_mutable_providers(monkeypatch):
    store = MariaDBRuleDecisionStore(lambda: None)
    expected = _trusted_consumption(RulePermitConsumptionDraft(
        permit_id="0199f8a1-8c00-7000-8000-000000000041",
        consumption_request_id="0199f8a1-8c00-7000-8000-000000000042",
        request_hash="sha256:" + "a" * 64,
        consumed_at_utc=datetime(2026, 10, 10, tzinfo=timezone.utc),
    ))
    recover = Mock(return_value=expected)
    monkeypatch.setattr(store, "recover_consumption", recover)
    kernel = RuleAuthorityKernel(
        active_policy_pin_provider=object(), authority_ledger_store=object(),
        trusted_root=object(), trusted_checkpoint_store=object(), stop_provider=object(),
        trusted_clock=object(), capability_classification_provider=object(),
        rule_authority_store=store,
    )

    result = kernel.consume(
        expected.permit_id, expected.request_hash, expected.consumption_request_id
    )

    assert result is expected
    assert result._is_store_sealed()
    recover.assert_called_once_with(
        expected.permit_id, expected.request_hash, expected.consumption_request_id
    )


def test_consume_durable_retry_rejects_different_consumption_id(monkeypatch):
    store = MariaDBRuleDecisionStore(lambda: None)
    recover = Mock(side_effect=AuthorityStateError(
        "permit ya consumido con consumption_request_id distinto"
    ))
    monkeypatch.setattr(store, "recover_consumption", recover)
    kernel = RuleAuthorityKernel(
        active_policy_pin_provider=object(), authority_ledger_store=object(),
        trusted_root=object(), trusted_checkpoint_store=object(), stop_provider=object(),
        trusted_clock=object(), capability_classification_provider=object(),
        rule_authority_store=store,
    )

    with pytest.raises(AuthorityStateError, match="consumption_request_id distinto"):
        kernel.consume("0199f8a1-8c00-7000-8000-000000000051",
                       "sha256:" + "b" * 64,
                       "0199f8a1-8c00-7000-8000-000000000052")


def test_consume_rejects_unsealed_store_recovery(monkeypatch):
    store = MariaDBRuleDecisionStore(lambda: None)
    monkeypatch.setattr(store, "recover_consumption", Mock(return_value=object()))
    kernel = RuleAuthorityKernel(
        active_policy_pin_provider=object(), authority_ledger_store=object(),
        trusted_root=object(), trusted_checkpoint_store=object(), stop_provider=object(),
        trusted_clock=object(), capability_classification_provider=object(),
        rule_authority_store=store,
    )

    with pytest.raises(RuleAuthorityStorageError, match="sellado"):
        kernel.consume("0199f8a1-8c00-7000-8000-000000000071",
                       "sha256:" + "c" * 64,
                       "0199f8a1-8c00-7000-8000-000000000072")


def _consumption_context(now, *, unbounded=False):
    if unbounded:
        rule = _rule("regla-ejemplo.yaml")
        request = _request(capability="CAPABILITY_ID", objective="objetivo-ejemplo",
                           rule_hash=_hash_de_regla(rule))
        capability_limits = {"form": "NINGUNA"}
        contract_unit = None
        limit_form = FormaLimites.NINGUNA
    else:
        request = _request(capability="CAPABILITY_ID", objective="objetivo-ejemplo",
                           quantity=1, quantity_unit="mensajes")
        rule = _rule("regla-ejemplo.yaml",
                     quantity={"unit": "mensajes", "max": 2})
        capability_limits = {"form": "CANTIDAD", "unit": "mensajes", "max": 2}
        contract_unit = "mensajes"
        limit_form = FormaLimites.CANTIDAD
    sealed = SimpleNamespace(regla=rule, ruta="policy/faro/ejemplo.yaml",
                              blob_oid="a" * 40, content_hash="sha256:" + "c" * 64)
    revision = "b" * 40
    snapshot = SimpleNamespace(commit=revision, repositorio="jax", reglas=(sealed,),
                               catalogo=CATALOG)
    ratification_id = "0199f8a1-8c00-7000-8000-000000000061"
    permit = _trusted_permit(RulePermitDraft(
        permit_id="0199f8a1-8c00-7000-8000-000000000062",
        request_id=request.request_id, request_hash=request.request_hash,
        rule_id=rule.rule_id, rule_path=sealed.ruta, rule_blob_oid=sealed.blob_oid,
        rule_content_hash=sealed.content_hash, policy_revision=revision,
        policy_tree_oid="d" * 40, policy_snapshot_hash="sha256:" + "e" * 64,
        ratification_event_id=ratification_id,
        authority_ledger_checkpoint={"sequence": 1, "head_event_hash": "f" * 64},
        stop_checkpoint={"version": 1, "fingerprint": "sha256:" + "1" * 64},
        capability_id=request.capability, policy_repository_id="jax",
        capability_contract_id="mail.send.contract.v1", capability_version="1",
        capability_class="REVERSIBLE",
        capability_limits=capability_limits,
        issued_at_utc=now.replace(microsecond=0),
        expires_at_utc=now.replace(microsecond=0).replace(second=now.second + 30),
    ))
    contract = ContratoCapability(
        identidad="mail.send.contract.v1", version="1",
        clase=ClaseCapability.REVERSIBLE, unidad=contract_unit, moneda=None,
        forma_limites=limit_form,
    )
    grant = SimpleNamespace(valid_from_utc=now.replace(year=now.year - 1),
                            valid_until_utc=None)
    grant_event = SimpleNamespace(event_id=ratification_id,
                                  intent=SimpleNamespace(rule_ratification=grant))
    state = SimpleNamespace(latest_unrevoked_rule_ratification=lambda _rule_id: grant_event)
    views = SimpleNamespace(
        ahora_utc=now,
        stop=SimpleNamespace(valor=SimpleNamespace(activo=False)),
        pin=SimpleNamespace(valor=SimpleNamespace(
            repositorio=SimpleNamespace(repository_id="jax"))),
        clasificacion=SimpleNamespace(valor=SimpleNamespace(
            contrato_de=lambda _capability: contract)),
    )
    return permit, request, snapshot, state, views, revision


def _kernel_consume_harness(tmp_path, monkeypatch, *, changed_provider=None):
    now = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
    permit, request, snapshot, state, views, revision = _consumption_context(now)
    pin_id = "other" if changed_provider == "pin" else "jax"
    pin_value = TrustedPolicyPin(pin_id, revision, "d" * 40, "refs/heads/main")
    pin = PinFijo(pin_value, "refs/heads/main", repository_path=tmp_path)
    stop = StopFijo(activo=changed_provider == "stop", huella="sha256:stop")
    contract = ContratoCapability(
        identidad=("mail.send.contract.v2" if changed_provider == "classification"
                   else "mail.send.contract.v1"),
        version=("2" if changed_provider == "classification" else "1"),
        clase=ClaseCapability.REVERSIBLE, unidad="mensajes", moneda=None,
        forma_limites=FormaLimites.CANTIDAD,
    )
    classification = ClasificacionFija({"CAPABILITY_ID": contract})

    class Owner:
        def lock_permit_for_consumption(self, permit_id, request_hash):
            assert permit_id == permit.permit_id
            assert request_hash == permit.request_hash
            return permit

    owner = Owner()

    class Fence:
        @contextmanager
        def transaction_owner(self, _store, *, purpose):
            assert purpose is AuthorityLedgerTransactionPurpose.CONSUMPTION
            yield owner

        def verify_for_update(self, _owner, _trusted_root):
            return state

    fence = Fence()

    class LedgerStore:
        @contextmanager
        def checkpoint_fence(self, _checkpoint):
            yield fence

    decision_store = MariaDBRuleDecisionStore(lambda: None)
    monkeypatch.setattr(decision_store, "recover_consumption", Mock(return_value=None))
    monkeypatch.setattr(
        decision_store, "load_consumption_request_in_transaction",
        lambda _owner, _catalog: request.canonical_projection(),
    )
    monkeypatch.setattr(
        decision_store, "consume_permit",
        lambda _owner, draft: _trusted_consumption(draft),
    )
    monkeypatch.setattr(
        "policy.rule_authority.kernel.load_trusted_policy_snapshot",
        lambda _path, _pin: snapshot,
    )
    monkeypatch.setattr(RuleAuthorityKernel, "_grant_matches", staticmethod(lambda *_args: True))
    monkeypatch.setattr("policy.rule_authority.kernel.es_ancestro", lambda *_args: True)
    kernel = RuleAuthorityKernel(
        active_policy_pin_provider=pin, authority_ledger_store=LedgerStore(),
        trusted_root=object(), trusted_checkpoint_store=object(), stop_provider=stop,
        trusted_clock=RelojDeterminista(now),
        capability_classification_provider=classification,
        rule_authority_store=decision_store,
    )
    providers = {"pin": pin, "stop": stop, "classification": classification}
    return kernel, providers


def _mutate_provider(kind, providers, tmp_path):
    provider = providers[kind]
    if kind == "stop":
        with provider.lease_exclusivo():
            provider._reemplazar(EstadoStop(activo=True, huella="sha256:stop-updated"))
    elif kind == "classification":
        provider.republicar(
            "CAPABILITY_ID",
            ContratoCapability(
                identidad="mail.send.contract.v2", version="2",
                clase=ClaseCapability.REVERSIBLE, unidad="mensajes", moneda=None,
                forma_limites=FormaLimites.CANTIDAD,
            ),
        )
    else:
        proc = "refs/heads/main"
        new_pin = TrustedPolicyPin("other", "c" * 40, "d" * 40, proc)
        binding = RepositorioConfiable(
            repository_id="other", repository_path=tmp_path, procedencia=proc,
        )
        with provider.lease_exclusivo():
            provider._reemplazar(PinActivo(new_pin, binding))


@pytest.mark.parametrize(
    "kind,reason",
    [("stop", "STOP activo"),
     ("pin", "solicitud/repositorio"),
     ("classification", "contrato de capability")],
)
def test_consume_denies_provider_change_that_wins_before_its_lease(tmp_path, monkeypatch,
                                                                    kind, reason):
    kernel, _providers = _kernel_consume_harness(
        tmp_path, monkeypatch, changed_provider=kind
    )

    with pytest.raises(AuthorityStateError, match=reason):
        kernel.consume(
            "0199f8a1-8c00-7000-8000-000000000062",
            _request(capability="CAPABILITY_ID", objective="objetivo-ejemplo",
                     quantity=1, quantity_unit="mensajes").request_hash,
            "0199f8a1-8c00-7000-8000-000000000063",
        )


@pytest.mark.parametrize("kind", ["stop", "pin", "classification"])
def test_consume_holds_provider_lease_until_durable_append(tmp_path, monkeypatch, kind):
    kernel, providers = _kernel_consume_harness(tmp_path, monkeypatch)
    validated = threading.Event()
    release_validation = threading.Event()
    append_started = threading.Event()
    release_append = threading.Event()
    writer_started = threading.Event()
    original_validate = kernel._validate_consumption_context
    original_append = kernel._store.consume_permit

    def pause_after_validation(**kwargs):
        result = original_validate(**kwargs)
        validated.set()
        if not release_validation.wait(timeout=5):
            raise TimeoutError("la prueba no liberó la validación pausada")
        return result

    def pause_during_append(owner, draft):
        append_started.set()
        if not release_append.wait(timeout=5):
            raise TimeoutError("la prueba no liberó el append pausado")
        return original_append(owner, draft)

    monkeypatch.setattr(kernel, "_validate_consumption_context", pause_after_validation)
    monkeypatch.setattr(kernel._store, "consume_permit", pause_during_append)
    provider = providers[kind]

    def update_provider():
        writer_started.set()
        _mutate_provider(kind, providers, tmp_path)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            consume_future = pool.submit(
                kernel.consume,
                "0199f8a1-8c00-7000-8000-000000000062",
                _request(capability="CAPABILITY_ID", objective="objetivo-ejemplo",
                         quantity=1, quantity_unit="mensajes").request_hash,
                "0199f8a1-8c00-7000-8000-000000000064",
            )
            assert validated.wait(timeout=5)
            writer_future = pool.submit(update_provider)
            assert writer_started.wait(timeout=5)
            deadline = time.monotonic() + 5
            while provider.esperando() == 0 and time.monotonic() < deadline:
                threading.Event().wait(0.001)
            assert provider.esperando() > 0
            assert not writer_future.done()
            release_validation.set()
            assert append_started.wait(timeout=5)
            assert provider.esperando() > 0
            assert not writer_future.done()
            release_append.set()
            consumption = consume_future.result(timeout=5)
            writer_future.result(timeout=5)
        assert consumption._is_store_sealed()
    finally:
        release_validation.set()
        release_append.set()


def test_consume_context_accepts_exact_current_rule_grant_scope_and_contract(monkeypatch):
    now = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
    permit, request, snapshot, state, views, _revision = _consumption_context(now)
    monkeypatch.setattr(RuleAuthorityKernel, "_grant_matches", staticmethod(lambda *_args: True))
    monkeypatch.setattr("policy.rule_authority.kernel.es_ancestro", lambda *_args: True)
    kernel = object.__new__(RuleAuthorityKernel)

    assert kernel._validate_consumption_context(
        permit=permit, request=request, snapshot=snapshot, state=state, vistas=views,
        repository_path=Path("/tmp"),
    ) is None


def test_consume_context_types_unbounded_rule_hash_as_authority_error(monkeypatch):
    now = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
    permit, request, snapshot, state, views, _revision = _consumption_context(
        now, unbounded=True
    )
    monkeypatch.setattr(RuleAuthorityKernel, "_grant_matches", staticmethod(lambda *_args: True))
    monkeypatch.setattr("policy.rule_authority.kernel.es_ancestro", lambda *_args: True)
    kernel = object.__new__(RuleAuthorityKernel)

    with pytest.raises(AuthorityStateError, match="sin límites"):
        kernel._validate_consumption_context(
            permit=permit, request=request, snapshot=snapshot, state=state, vistas=views,
            repository_path=Path("/tmp"),
        )


@pytest.mark.parametrize("change,reason", [
    ("stop", "STOP activo"),
    ("expired", "fuera de vigencia"),
    ("grant", "ratificación.*vigente"),
    ("contract", "contrato de capability"),
])
def test_consume_context_denies_changed_live_authority(monkeypatch, change, reason):
    now = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
    permit, request, snapshot, state, views, _revision = _consumption_context(now)
    monkeypatch.setattr(RuleAuthorityKernel, "_grant_matches", staticmethod(lambda *_args: True))
    monkeypatch.setattr("policy.rule_authority.kernel.es_ancestro", lambda *_args: True)
    if change == "stop":
        views.stop.valor.activo = True
    elif change == "expired":
        views.ahora_utc = now.replace(minute=1)
    elif change == "grant":
        state.latest_unrevoked_rule_ratification = lambda _rule_id: None
    elif change == "contract":
        views.clasificacion.valor.contrato_de = lambda capability: ContratoCapability(
            identidad="mail.send.contract.v2", version="2",
            clase=ClaseCapability.REVERSIBLE, unidad=None, moneda=None,
            forma_limites=FormaLimites.NINGUNA,
        )
    kernel = object.__new__(RuleAuthorityKernel)

    with pytest.raises(AuthorityStateError, match=reason):
        kernel._validate_consumption_context(
            permit=permit, request=request, snapshot=snapshot, state=state, vistas=views,
            repository_path=Path("/tmp"),
        )
