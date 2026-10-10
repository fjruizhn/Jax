"""Modelos de solicitud/evaluacion/decision de Faro F1.1 (paso 4).

Una sola fuente de verdad para los topes: el CATALOGO SELLADO del pin (#370,
``CatalogoTopes``) y el modelo de la regla (#370, ``ReglaValidada`` /
``LimitesObligatorios``). Aqui no hay taxonomia propia: los limites se DERIVAN
con ``limites_de`` y el catalogo se exige por tipo exacto.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

import pytest

from jax.faro.catalogo_topes import CatalogoTopes, CatalogoTopesInvalido
from policy.authority_ledger.errors import AuthorityEventValidationError, AuthorityStateError
from policy.canonicalization.strict_yaml import load_strict_yaml
from policy.rule_authority.models import (
    RuleDecision,
    RuleDecisionStatus,
    RuleEvaluation,
    RuleEvaluationRequest,
    RuleLimits,
    limites_de,
)
from policy.rule_authority.schema import (
    MAX_CANTIDAD,
    Cantidad,
    Frecuencia,
    LimitesObligatorios,
    Monto,
    Tope,
    validar_regla,
)
from policy.rule_authority.store import InMemoryRuleDecisionStore
from policy.rule_authority.providers import leases_de_emision
from policy.rule_authority.snapshot import TrustedPolicyPin
from tests.policy.catalogo_pin import catalogo_del_pin
from tests.policy.proveedores_dobles import (
    CheckpointsEnMemoria,
    ClasificacionFija,
    PinFijo,
    RelojDeterminista,
    StopFijo,
)

FIXTURES = Path(__file__).parent / "fixtures" / "faro_rules"
_CLASES = {"monto_dinero": ["hnl", "usd"],
           "actos_externos": ["mensajes", "correos", "publicaciones", "compras", "pagos"],
           "frecuencia": ["por_hora", "por_dia"], "duracion": ["segundos"],
           "tokens_costo": ["tokens", "usd"]}


def _bytes_catalogo(decision: str) -> bytes:
    return json.dumps({"version": 1, "decision": decision, "clases": _CLASES}).encode()


CATALOG = catalogo_del_pin(_bytes_catalogo("test fixture"))
# Otro catalogo SELLADO: mismas clases, otro blob en el pin -> otro OID.
OTRO_CATALOG = catalogo_del_pin(_bytes_catalogo("otro fixture"))

NOW = datetime(2026, 10, 6, tzinfo=timezone.utc)
FORBIDDEN = ("connections", "conexiones", "concurrency", "concurrencia", "workers", "threads",
             "hilos", "processes", "procesos", "agents", "agentes", "conexiones_db", "num_hilos")


class AlwaysEqualStr(str):
    """str que dice ser igual a todo: ``in`` sobre una tupla de unidades lo acepta."""
    def __eq__(self, otro):
        return True

    __hash__ = str.__hash__


def _subclase_forjada(clases=None):
    """Subclase de CatalogoTopes con ``__slots__``: salta el testigo del constructor."""
    class Forjado(CatalogoTopes):
        __slots__ = ()

    objeto = Forjado.__new__(Forjado)
    object.__setattr__(objeto, "_clases", MappingProxyType(
        {k: tuple(v) for k, v in (clases or _CLASES).items()}))
    object.__setattr__(objeto, "_oid_pin", "a" * 40)
    return objeto


def _a_mano(clases=None):
    return MappingProxyType({k: tuple(v) for k, v in (clases or _CLASES).items()})


def request(**overrides):
    values = {
        "request_id": "0199f8a1-8c00-7000-8000-000000000001",
        "rule_id": "ejemplo-tope-mensajes",
        "subject": "actor:fernando",
        "capability": "mail.send",
        "objective": "notify-client",
        "resource_id": "message:invoice-42",
        "arguments": {"recipient": "client@example.test", "body": "Invoice ready"},
        "catalogo": CATALOG,
        "quantity": 1,
        "quantity_unit": "mensajes",
        "amount": None,
    }
    values.update(overrides)
    return RuleEvaluationRequest(**values)


def _datos(fixture="regla-ejemplo-tope.yaml"):
    return load_strict_yaml((FIXTURES / fixture).read_text())


def _regla_obligating(**cambios):
    """Una regla valida de #370 con limites de actos; `cambios` pisa obligation_limits/tope."""
    datos = copy.deepcopy(_datos())
    datos["obligation_limits"]["quantity"]["unit"] = "mensajes"
    for clave, valor in cambios.items():
        if clave == "tope":
            datos["tope"] = valor
        else:
            datos["obligation_limits"][clave] = valor
    return validar_regla(datos, catalogo=CATALOG)


def _a_mano_regla(**limites):
    """ReglaValidada de #370 armada a mano (no tiene sello): sus limites se re-validan."""
    base = _regla_obligating()
    tope = limites.pop("tope", base.tope)
    nuevos = dataclasses.replace(base.obligation_limits, **limites)
    return dataclasses.replace(base, obligation_limits=nuevos, tope=tope)


def decision(**overrides):
    values = dict(request_id="0199f8a1-8c00-7000-8000-000000000001",
                  request_hash="sha256:" + "a" * 64, status=RuleDecisionStatus.DENY,
                  required_rule_id="rule-one", reason_code="STOP_ACTIVE", decided_at_utc=NOW)
    values.update(overrides)
    return RuleDecision(**values)


# ------------------------------------------------------------------ solicitud: hash

def test_request_hash_is_derived_from_closed_canonical_projection():
    original = request()
    reordered = request(arguments={"body": "Invoice ready", "recipient": "client@example.test"})
    changed = request(arguments={"recipient": "other@example.test", "body": "Invoice ready"})

    assert original.request_hash == reordered.request_hash
    assert original.request_hash != changed.request_hash
    assert original.arguments["recipient"] == "client@example.test"
    with pytest.raises(TypeError):
        original.arguments["recipient"] = "tampered@example.test"
    with pytest.raises((AttributeError, TypeError)):
        original.request_hash = "sha256:" + "0" * 64
    with pytest.raises(AuthorityEventValidationError):
        request(request_hash="sha256:" + "0" * 64)
    assert request(arguments={"nested": {}}).request_hash != request(arguments={"nested": []}).request_hash


def test_request_hash_includes_the_catalog_identity():
    assert request().request_hash != request(catalogo=OTRO_CATALOG).request_hash
    # mismo blob del pin -> misma identidad -> mismo hash
    assert request().request_hash == request(
        catalogo=catalogo_del_pin(_bytes_catalogo("test fixture"))).request_hash


def test_request_arguments_are_normalized_to_nfc_before_hashing():
    composed = request(arguments={"texto": "café"})
    decomposed = request(arguments={"texto": "café"})
    assert composed.arguments["texto"] == "café"
    assert decomposed.arguments["texto"] == "café"
    assert composed.request_hash == decomposed.request_hash
    with pytest.raises(AuthorityEventValidationError, match="claves duplicadas tras NFC"):
        request(arguments={"é": 1, "é": 2})


def test_request_hash_includes_quantity_unit():
    assert request(quantity_unit="mensajes").request_hash != request(quantity_unit="correos").request_hash


@pytest.mark.parametrize("arguments", [{"nested": object()}, {"x": 1.5}])
def test_request_rejects_opaque_arguments(arguments):
    with pytest.raises(AuthorityEventValidationError):
        request(arguments=arguments)


@pytest.mark.parametrize("valor", [2**53 + 1, -(2**53) - 1, 2**80])
@pytest.mark.parametrize("forma", ["directo", "anidado", "lista"])
def test_request_arguments_integers_are_range_limited(valor, forma):
    arguments = {"directo": {"n": valor}, "anidado": {"a": {"b": valor}}, "lista": {"l": [1, valor]}}[forma]
    with pytest.raises(AuthorityEventValidationError, match="rango"):
        request(arguments=arguments)


def test_request_arguments_accept_integers_at_the_range_edge_and_booleans():
    request(arguments={"max": 2**53, "min": -(2**53), "ok": True, "l": [0, 2**53]})


# ------------------------------------------------ solicitud: catalogo por tipo exacto (M2)

@pytest.mark.parametrize("catalogo", [
    pytest.param(None, id="ninguno"),
    pytest.param(_a_mano(), id="MappingProxyType-a-mano"),
    pytest.param(dict(_CLASES), id="dict"),
    pytest.param(_subclase_forjada(), id="subclase-de-CatalogoTopes"),
    pytest.param("actos_externos mensajes monto_dinero usd", id="cadena"),
    pytest.param(_a_mano({**_CLASES, "actos_externos": ["workers", "mensajes"]}), id="a-mano-con-workers"),
])
def test_request_requires_exact_sealed_catalog_type(catalogo):
    with pytest.raises(AuthorityEventValidationError, match="catálogo"):
        request(catalogo=catalogo)


def test_hand_built_catalog_cannot_be_sealed_outside_the_snapshot():
    with pytest.raises(CatalogoTopesInvalido):
        CatalogoTopes(_CLASES, oid_pin="a" * 40)


# ------------------------------------------------------------ solicitud: unidades y rangos

@pytest.mark.parametrize(
    ("quantity", "quantity_unit"),
    [(1, None), (None, "mensajes")],
)
def test_request_requires_quantity_and_unit_together(quantity, quantity_unit):
    with pytest.raises(AuthorityEventValidationError):
        request(quantity=quantity, quantity_unit=quantity_unit)


@pytest.mark.parametrize("unidad", ["recipients", "Mensajes", "mensajes​", "mensajes ", "",
                                    *FORBIDDEN])
def test_request_quantity_unit_must_be_in_the_catalog(unidad):
    with pytest.raises(AuthorityEventValidationError):
        request(quantity_unit=unidad)


def test_request_units_and_currency_must_be_exact_str_not_a_subclass_with_eq_true():
    with pytest.raises(AuthorityEventValidationError):
        request(quantity_unit=AlwaysEqualStr("workers"))
    with pytest.raises(AuthorityEventValidationError):
        request(quantity=None, quantity_unit=None, amount=5, currency=AlwaysEqualStr("zzz"))


def test_request_currency_uses_the_catalog_form():
    assert request(quantity=None, quantity_unit=None, amount=5, currency="usd").amount == 5
    for moneda in ("USD", "zzz", "ZZZ", "eur", "tokens"):
        with pytest.raises(AuthorityEventValidationError, match="catálogo"):
            request(quantity=None, quantity_unit=None, amount=5, currency=moneda)


def test_request_rejects_currency_without_amount_and_amount_without_currency():
    with pytest.raises(AuthorityEventValidationError, match="amount y currency"):
        request(quantity=None, quantity_unit=None, currency="usd")
    with pytest.raises(AuthorityEventValidationError, match="amount y currency"):
        request(quantity=None, quantity_unit=None, amount=5)


@pytest.mark.parametrize("campo", ["quantity", "amount"])
def test_request_numbers_have_the_2_pow_53_ceiling_and_reject_bool(campo):
    extra = ({"quantity_unit": "mensajes"} if campo == "quantity"
             else {"currency": "usd", "quantity": None, "quantity_unit": None})
    assert getattr(request(**{campo: MAX_CANTIDAD}, **extra), campo) == MAX_CANTIDAD
    for malo in (MAX_CANTIDAD + 1, 2**80, 0, -1, True):
        with pytest.raises(AuthorityEventValidationError):
            request(**{campo: malo}, **extra)


def test_request_validates_uuid7_and_exactly_one_action_or_money_limit():
    with pytest.raises(AuthorityEventValidationError):
        request(request_id="not-a-uuid")
    with pytest.raises(AuthorityEventValidationError):
        request(quantity=0)
    with pytest.raises(AuthorityEventValidationError):
        request(quantity=1, amount=50, currency="usd")


# --------------------------------------------------------- limites: derivados de #370

def test_limits_are_derived_from_the_validated_rule_without_parallel_fields():
    regla = _regla_obligating()
    limites = limites_de(regla, CATALOG)
    assert isinstance(limites, RuleLimits)
    assert limites.limites is regla.obligation_limits
    assert limites.tope is regla.tope
    assert limites.catalogo is CATALOG
    nombres = {f.name for f in dataclasses.fields(RuleLimits)}
    assert not nombres & {"quantity", "quantity_unit", "amount", "currency", "tokens",
                          "duration_seconds", "cost_minor_units", "cost_currency",
                          "frequency_count", "frequency_unit", "frequency_window_seconds"}


def test_limits_with_money_and_usd_amount_are_derived():
    regla = _regla_obligating(quantity=None, amount={"currency": "USD", "max": 500})
    assert limites_de(regla, CATALOG).limites.amount == Monto("USD", 500)


def test_no_caller_can_build_limits_by_hand():
    base = limites_de(_regla_obligating(), CATALOG)
    with pytest.raises(AuthorityEventValidationError, match="limites_de"):
        RuleLimits(CATALOG, base.limites, base.tope, base.rule_id, base.rule_hash)
    with pytest.raises(AuthorityEventValidationError, match="limites_de"):
        RuleLimits(catalogo=CATALOG, limites=LimitesObligatorios(Cantidad("mensajes", 1), None, None),
                   tope=None, rule_id="rule-one", rule_hash="sha256:" + "0" * 64)
    with pytest.raises(AuthorityEventValidationError):
        limites_de(dict(), CATALOG)                    # no es una ReglaValidada


def test_limits_survive_no_dataclasses_replace():
    """`replace` vuelve a llamar al constructor: no puede reproducir la procedencia."""
    base = limites_de(_regla_obligating(), CATALOG)
    with pytest.raises(AuthorityEventValidationError, match="limites_de"):
        dataclasses.replace(base, limites=LimitesObligatorios(Cantidad("mensajes", 2**53), None, None))
    with pytest.raises(AuthorityEventValidationError, match="limites_de"):
        dataclasses.replace(base, rule_id="otra-regla")


def test_limits_carry_the_rule_identity():
    regla = _regla_obligating()
    limites = limites_de(regla, CATALOG)
    assert limites.rule_id == regla.rule_id
    assert limites.rule_hash == limites_de(_regla_obligating(), CATALOG).rule_hash
    otra = _regla_obligating(quantity={"unit": "mensajes", "max": 7})
    assert limites_de(otra, CATALOG).rule_hash != limites.rule_hash


@pytest.mark.parametrize("rule_id", ["Regla Mala!", "", AlwaysEqualStr("x y")])
def test_limits_require_a_well_formed_rule_id(rule_id):
    with pytest.raises(AuthorityEventValidationError, match="rule_id"):
        limites_de(dataclasses.replace(_regla_obligating(), rule_id=rule_id), CATALOG)


@pytest.mark.parametrize("periodo", ["workers", "Por_Hora", "", "hora", "por_hora ", 3600,
                                     AlwaysEqualStr("workers"), "por_hora\u200b", "por\u0301_hora"])
def test_tope_period_must_be_a_frequency_subid_of_the_catalog(periodo):
    tope = Tope("frecuencia", "frecuencia.por_hora", 5, periodo)
    with pytest.raises(AuthorityEventValidationError, match="period"):
        limites_de(_a_mano_regla(tope=tope), CATALOG)


@pytest.mark.parametrize("periodo", ["por_hora", "por_dia"])
def test_tope_period_in_the_catalog_is_accepted(periodo):
    tope = Tope("frecuencia", "frecuencia.por_hora", 5, periodo)
    assert limites_de(_a_mano_regla(tope=tope), CATALOG).tope.period == periodo


def test_tope_must_be_the_exact_tope_type_not_a_subclass():
    class TopeFalso(Tope):
        pass

    base = _regla_obligating().tope
    falso = TopeFalso(base.resource_class, base.resource, base.maximum, base.period)
    with pytest.raises(AuthorityEventValidationError, match="Tope"):
        limites_de(_a_mano_regla(tope=falso), CATALOG)


@pytest.mark.parametrize("catalogo", [
    pytest.param(None, id="ninguno"),
    pytest.param(_a_mano(), id="MappingProxyType-a-mano"),
    pytest.param(_subclase_forjada(), id="subclase"),
    pytest.param(_a_mano({**_CLASES, "actos_externos": ["conexiones", "mensajes"]}), id="con-conexiones"),
])
def test_limits_require_exact_sealed_catalog_type(catalogo):
    with pytest.raises(AuthorityEventValidationError, match="catálogo"):
        limites_de(_regla_obligating(), catalogo)


def test_empty_limits_are_rejected():
    vacia = validar_regla(_datos("regla-ejemplo.yaml"), catalogo=CATALOG)
    with pytest.raises(AuthorityEventValidationError, match="vacío"):
        limites_de(vacia, CATALOG)


@pytest.mark.parametrize("unidad", ["llamadas", "recipients", "Mensajes", "mensajes​", *FORBIDDEN])
def test_limit_quantity_unit_must_be_in_the_catalog(unidad):
    with pytest.raises(AuthorityEventValidationError, match="actos_externos"):
        limites_de(_a_mano_regla(quantity=Cantidad(unidad, 5)), CATALOG)


def test_limit_units_and_currency_must_be_exact_str_not_a_subclass_with_eq_true():
    with pytest.raises(AuthorityEventValidationError):
        limites_de(_a_mano_regla(quantity=Cantidad(AlwaysEqualStr("workers"), 5)), CATALOG)
    with pytest.raises(AuthorityEventValidationError):
        limites_de(_a_mano_regla(quantity=None, amount=Monto(AlwaysEqualStr("ZZZ"), 5)), CATALOG)
    con_tope = _regla_obligating().tope
    with pytest.raises(AuthorityEventValidationError):
        limites_de(_a_mano_regla(tope=dataclasses.replace(
            con_tope, resource_class=AlwaysEqualStr("conexiones"))), CATALOG)
    with pytest.raises(AuthorityEventValidationError):
        limites_de(_a_mano_regla(tope=dataclasses.replace(
            con_tope, resource=AlwaysEqualStr("actos_externos.workers"))), CATALOG)


@pytest.mark.parametrize("moneda", ["ZZZ", "zzz", "usd", "EUR", "US", "USDX", "TÖK"])
def test_limit_currency_must_be_catalog_money_in_iso_form(moneda):
    with pytest.raises(AuthorityEventValidationError, match="monto_dinero"):
        limites_de(_a_mano_regla(quantity=None, amount=Monto(moneda, 5)), CATALOG)


@pytest.mark.parametrize("valor", [MAX_CANTIDAD + 1, 2**80, 0, -1, True, 1.0])
def test_limit_numbers_are_exact_positive_ints_up_to_2_pow_53(valor):
    for limites in (dict(quantity=Cantidad("mensajes", valor)),
                    dict(quantity=None, amount=Monto("USD", valor)),
                    dict(frequency=Frecuencia(valor, 3600)),
                    dict(frequency=Frecuencia(5, valor))):
        with pytest.raises(AuthorityEventValidationError):
            limites_de(_a_mano_regla(**limites), CATALOG)
    with pytest.raises(AuthorityEventValidationError):
        limites_de(_a_mano_regla(tope=dataclasses.replace(_regla_obligating().tope, maximum=valor)),
                   CATALOG)


def test_limit_numbers_accept_the_edge():
    limites_de(_a_mano_regla(quantity=Cantidad("mensajes", MAX_CANTIDAD),
                             frequency=Frecuencia(MAX_CANTIDAD, MAX_CANTIDAD)), CATALOG)


# El tope es el unico lugar de #370 donde viven frecuencia (por_hora/por_dia), duracion y
# tokens/costo: sin catalogo en esas clases no hay limite (M1, M5, M6).
@pytest.mark.parametrize(("clase", "recurso"), [
    pytest.param("frecuencia", "frecuencia.por_semana", id="M1-unidad-de-frecuencia-fuera"),
    pytest.param("duracion", "duracion.horas", id="M5-duracion-fuera"),
    pytest.param("tokens_costo", "tokens_costo.palabras", id="M5-tokens-fuera"),
    pytest.param("tokens_costo", "tokens_costo.eur", id="M6-moneda-de-costo-fuera"),
    pytest.param("actos_externos", "actos_externos.workers", id="subid-prohibido"),
    pytest.param("conexiones", "conexiones.maximo", id="clase-prohibida"),
    pytest.param("frecuencia", "duracion.segundos", id="clase-distinta-del-recurso"),
    pytest.param("duracion", "segundos", id="sin-clase"),
    pytest.param("duracion", "duracion.segundos.x", id="subid-compuesto"),
])
def test_tope_resource_must_come_from_the_catalog(clase, recurso):
    tope = Tope(resource_class=clase, resource=recurso, maximum=5, period="por_hora")
    with pytest.raises(AuthorityEventValidationError, match="tope"):
        limites_de(_a_mano_regla(tope=tope), CATALOG)


@pytest.mark.parametrize("recurso", ["frecuencia.por_hora", "duracion.segundos",
                                     "tokens_costo.tokens", "tokens_costo.usd",
                                     "monto_dinero.hnl", "actos_externos.pagos"])
def test_tope_resource_in_the_catalog_is_accepted(recurso):
    clase = recurso.partition(".")[0]
    tope = Tope(resource_class=clase, resource=recurso, maximum=5, period="por_hora")
    assert limites_de(_a_mano_regla(tope=tope), CATALOG).tope == tope


def test_rule_limits_cannot_represent_infrastructure_limits():
    """Con el catalogo REAL: ni como unidad, ni como moneda, ni como recurso del tope, ni
    como atributo; y el pin ni siquiera carga un catalogo que los traiga."""
    campos = {f.name for f in dataclasses.fields(RuleLimits)}
    for palabra in FORBIDDEN:
        assert palabra not in campos
        with pytest.raises(TypeError):
            RuleLimits(**{palabra: 2})
        with pytest.raises(AuthorityEventValidationError):
            limites_de(_a_mano_regla(quantity=Cantidad(palabra, 3)), CATALOG)
        with pytest.raises(AuthorityEventValidationError):
            limites_de(_a_mano_regla(tope=Tope(palabra, f"{palabra}.maximo", 3, "por_hora")), CATALOG)
        with pytest.raises(AuthorityEventValidationError):
            limites_de(_a_mano_regla(tope=Tope("actos_externos", f"actos_externos.{palabra}", 3,
                                               "por_hora")), CATALOG)
        with pytest.raises(AuthorityEventValidationError):
            request(quantity_unit=palabra)
        # el catalogo del pin real tampoco tiene la palabra, ni como clase ni como subid
        assert palabra not in CATALOG
        assert all(palabra not in subids for subids in CATALOG.values())
    # y un catalogo ratificado que los traiga no se carga: no hay CatalogoTopes con ellos
    from policy.rule_authority.errors import RuleSnapshotError
    for palabra in ("workers", "conexiones_db"):
        sucio = {**_CLASES, "actos_externos": ["mensajes", palabra]}
        with pytest.raises(RuleSnapshotError, match="D-4"):
            catalogo_del_pin(json.dumps({"version": 1, "decision": "x", "clases": sucio}).encode())


# ------------------------------------------------------------------- RuleEvaluation

def test_evaluation_requires_the_same_catalog_in_request_and_limits():
    limites = limites_de(_regla_obligating(), CATALOG)
    evaluation = RuleEvaluation(request(), limites)
    assert evaluation.request.catalogo is evaluation.limits.catalogo
    # otro objeto con el MISMO OID del pin = misma identidad
    igual = catalogo_del_pin(_bytes_catalogo("test fixture"))
    assert igual is not CATALOG and igual.oid_pin == CATALOG.oid_pin
    RuleEvaluation(request(catalogo=igual), limites)


def test_evaluation_rejects_requests_and_limits_from_different_catalogs():
    limites = limites_de(_regla_obligating(), CATALOG)
    assert OTRO_CATALOG.oid_pin != CATALOG.oid_pin
    with pytest.raises(AuthorityEventValidationError, match="catálogo"):
        RuleEvaluation(request(catalogo=OTRO_CATALOG), limites)
    otros = limites_de(_regla_obligating(), OTRO_CATALOG)
    with pytest.raises(AuthorityEventValidationError, match="catálogo"):
        RuleEvaluation(request(), otros)


def test_evaluation_requires_limits_of_the_same_rule():
    otra = dataclasses.replace(_regla_obligating(), rule_id="regla-distinta")
    with pytest.raises(AuthorityEventValidationError, match="rule_id"):
        RuleEvaluation(request(rule_id="regla-distinta"), limites_de(_regla_obligating(), CATALOG))
    with pytest.raises(AuthorityEventValidationError, match="rule_id"):
        RuleEvaluation(request(), limites_de(otra, CATALOG))
    # rule_id de la solicitud = el de la regla: pasa
    RuleEvaluation(request(rule_id="ejemplo-tope-mensajes"), limites_de(_regla_obligating(), CATALOG))


def test_evaluation_checks_rule_hash_when_the_request_carries_it():
    regla = _regla_obligating()
    limites = limites_de(regla, CATALOG)
    ok = request(rule_id=regla.rule_id, rule_hash=limites.rule_hash)
    RuleEvaluation(ok, limites)
    assert ok.request_hash != request(rule_id=regla.rule_id).request_hash
    otra = limites_de(_regla_obligating(quantity={"unit": "mensajes", "max": 7}), CATALOG)
    with pytest.raises(AuthorityEventValidationError, match="rule_hash"):
        RuleEvaluation(ok, otra)
    with pytest.raises(AuthorityEventValidationError, match="rule_hash"):
        request(rule_hash="no-es-hash")


@pytest.mark.parametrize("malo", [None, "x", object()])
def test_evaluation_requires_exact_request_and_limits_types(malo):
    limites = limites_de(_regla_obligating(), CATALOG)
    with pytest.raises(AuthorityEventValidationError):
        RuleEvaluation(malo, limites)
    with pytest.raises(AuthorityEventValidationError):
        RuleEvaluation(request(), malo)


# ------------------------------------------------------------------------ decision

_DENY_REASONS = ("AUTHORITY_INVALID", "CAPABILITY_MISMATCH", "GRANT_INVALID", "OUT_OF_SCOPE",
                 "RULE_CHANGED", "RULE_EXPIRED", "RULE_NOT_FOUND", "STOP_ACTIVE")


@pytest.mark.parametrize("razon", [*_DENY_REASONS, "STORAGE_FAILURE", "OK", ""])
def test_permit_carries_no_reason_code_at_all(razon):
    """M3: un PERMIT con cualquier razon (las de DENY incluidas) se rechaza."""
    with pytest.raises(AuthorityEventValidationError, match="PERMIT no lleva"):
        decision(status=RuleDecisionStatus.PERMIT, reason_code=razon)
    assert decision(status=RuleDecisionStatus.PERMIT, reason_code=None).reason_code is None


def test_storage_failure_is_not_a_persistable_reason():
    for status in (RuleDecisionStatus.DENY, RuleDecisionStatus.MISSING_RULE):
        with pytest.raises(AuthorityEventValidationError):
            decision(status=status, reason_code="STORAGE_FAILURE")


@pytest.mark.parametrize(
    ("status", "reason_code"),
    [
        (RuleDecisionStatus.MISSING_RULE, "STOP_ACTIVE"),
        (RuleDecisionStatus.DENY, "RULE_NOT_FOUND"),
        (RuleDecisionStatus.DENY, None),
    ],
)
def test_decision_status_and_reason_code_cannot_contradict(status, reason_code):
    with pytest.raises(AuthorityEventValidationError):
        decision(status=status, reason_code=reason_code)


def test_decision_request_hash_must_be_exact_str():
    with pytest.raises(AuthorityEventValidationError, match="request_hash"):
        decision(request_hash=AlwaysEqualStr("sha256:" + "a" * 64))


def test_decision_has_no_orphan_catalog_field():
    assert [f.name for f in dataclasses.fields(RuleDecision)] == [
        "request_id", "request_hash", "status", "required_rule_id", "reason_code", "decided_at_utc"]


# --------------------------------------------------------------------------- store

def _decision_de(solicitud, **overrides):
    return decision(request_id=solicitud.request_id, request_hash=solicitud.request_hash,
                    required_rule_id=solicitud.rule_id, **overrides)


def test_memory_decision_store_is_idempotent_by_request_id_and_hash():
    store = InMemoryRuleDecisionStore()
    submitted = request()
    dec = _decision_de(submitted)

    assert store.record(submitted, dec) == dec
    assert store.record(request(), dec) == dec
    assert store.get(submitted) == dec
    assert store.get(request(arguments={"recipient": "different"})) is None
    changed = request(arguments={"recipient": "different"})
    with pytest.raises(AuthorityStateError):
        store.record(changed, _decision_de(changed))


def test_memory_store_binds_the_catalog_identity_through_the_hash():
    store = InMemoryRuleDecisionStore()
    submitted = request()
    store.record(submitted, _decision_de(submitted))
    assert store.get(request(catalogo=OTRO_CATALOG)) is None


def test_memory_store_cannot_inject_decisions_in_constructor():
    with pytest.raises(TypeError):
        InMemoryRuleDecisionStore(_decisions={})


def test_memory_store_rejects_decision_for_different_request():
    store = InMemoryRuleDecisionStore()
    submitted = request()
    other = request(resource_id="message:invoice-43")
    dec = _decision_de(other, status=RuleDecisionStatus.MISSING_RULE, reason_code="RULE_NOT_FOUND")

    with pytest.raises(AuthorityStateError):
        store.record(submitted, dec)
    assert store.get(submitted) is None


def test_memory_store_record_requires_a_real_rule_decision():
    store = InMemoryRuleDecisionStore()
    submitted = request()
    real = _decision_de(submitted)

    class Parecida:
        request_id = real.request_id
        request_hash = real.request_hash
        required_rule_id = real.required_rule_id
        status = RuleDecisionStatus.PERMIT
        reason_code = None
        decided_at_utc = NOW

    for falsa in (Parecida(), real.__dict__ if hasattr(real, "__dict__") else {}, None, "x"):
        with pytest.raises(TypeError):
            store.record(submitted, falsa)
    with pytest.raises(TypeError):
        store.record("no-es-solicitud", real)
    assert store.get(submitted) is None


def test_store_record_keeps_the_stop_lease_until_the_decision_is_persisted():
    """Rompe si el futuro evaluador llama ``record`` fuera de la frontera.

    ``record`` inicia un escritor real contra STOP. Debe quedar bloqueado hasta
    que la persistencia devuelve, porque el mismo contexto protegido contiene
    ambas operaciones.
    """
    class StoreDuranteLease(InMemoryRuleDecisionStore):
        entered: threading.Event
        release: threading.Event
        worker: threading.Thread

        def record(self, submitted, recorded):
            self.entered, self.release = threading.Event(), threading.Event()

            def writer() -> None:
                with stop.lease_exclusivo():
                    self.entered.set()
                    self.release.wait(timeout=5.0)

            self.worker = threading.Thread(target=writer, daemon=True)
            self.worker.start()
            limit = time.monotonic() + 5.0
            while not self.entered.is_set() and stop.esperando() == 0:
                assert time.monotonic() < limit, "el escritor no entro ni espero"
                time.sleep(0.0005)
            assert not self.entered.is_set(), "STOP se libero antes de persistir la decision"
            return super().record(submitted, recorded)

    checkpoint = CheckpointsEnMemoria()
    checkpoint.publicar("h1", anterior="")
    pin = TrustedPolicyPin("jax", "0" * 40, "1" * 40, "refs/heads/main")
    stop = StopFijo(activo=False, huella="sha256:stop")
    submitted = request()
    recorded = _decision_de(submitted)
    store = StoreDuranteLease()

    with leases_de_emision(
        pin=PinFijo(pin, "refs/heads/main"),
        checkpoint=checkpoint,
        stop=stop,
        reloj=RelojDeterminista(NOW),
        clasificacion=ClasificacionFija({}),
    ):
        assert store.record(submitted, recorded) == recorded
    try:
        assert store.entered.wait(timeout=5.0), "el escritor no progreso al cerrar la frontera"
    finally:
        store.release.set()
    store.worker.join(timeout=5.0)
    assert not store.worker.is_alive()
