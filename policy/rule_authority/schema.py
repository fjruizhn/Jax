"""Schema cerrado de regla Faro v1 (diseno F1.1 §7, mas R-4).

Una regla de ``policy/faro/<nombre>.yaml`` valida contra un shape COMPLETAMENTE
cerrado: campos extra se rechazan, los vocabularios son cerrados, los
identificadores van en NFC y los escalares pasan por el loader estricto (nada de
floats, coerciones ni booleans ambiguos). Esta funcion NO confiere autoridad:
solo decide que un documento tiene la FORMA de una regla. La autoridad nace del
snapshot sellado (``snapshot.py``) y de la ratificacion individual de Block 4.

R-4: la clase de recurso de un tope se DECLARA aqui (``tope.resource_class``,
vocabulario cerrado) y no se deduce del nombre. Las clases de agentes, enjambres
y conexiones no existen en el vocabulario: un tope sobre ellas no se puede ni
expresar. El guardia semantico de ``jax/faro/topes.py`` (D-4) sigue vigente en el
runtime; este schema duplica la lista para que la regla muera antes, al validar.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

from jax.faro.catalogo_topes import CatalogoTopes

from .errors import RuleSchemaError

SCHEMA_VERSION = "1.0"
KIND = "JAX_FARO_RULE"
EFFECT = "PERMIT"
ACTION_CLASSES = ("REVERSIBLE", "OBLIGATING")

# R-4, DECISION DE FERNANDO (2026-10-06): solo admiten tope las clases de ACTOS
# y DINERO del CATALOGO CERRADO compartido con el runtime
# (`jax/faro/catalogo_topes.py`, hoja de stdlib): monto de dinero, actos
# externos, frecuencia, duracion y tokens/costo. NUNCA conexiones,
# concurrencia, workers, hilos, procesos ni agentes (D-4). Una lista negra por
# palabras nunca cierra; el catalogo si: `resource` no es texto libre, es
# `<clase>.<subid>` con el subid en el catalogo DE ESA clase. Fuera de ahi no
# hay tope, en schema y en runtime, por igual.

# Techo por defecto de ttl_seconds: el kernel pasa el suyo de configuracion
# confiable (``ttl_max_seconds``); sin el, rige este. Es un MAXIMO, no un valor.
TTL_MAX_POR_DEFECTO = 3600

MAX_CANTIDAD = 2 ** 53        # el mismo entero representable sin perdida que topes.py

_RE_RULE_ID = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_RE_SUBJECT = re.compile(r"(?:human|actor):[a-z][a-z0-9-]{0,63}\Z")
_RE_CAPABILITY = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
_RE_OBJETIVO = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_RE_MONEDA = re.compile(r"[A-Z]{3}\Z")
_RE_UNIDAD = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")
_RE_RECURSO = re.compile(r"[a-z0-9_.-]{1,48}\Z")
_RE_PERIODO = re.compile(r"[A-Za-z0-9_.:-]{1,32}\Z")
_RE_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")

_CLAVES_TOPE_NIVEL = frozenset({"schema_version", "kind", "rule_id", "effect", "action_class",
                                "scope", "obligation_limits", "validity", "permit", "tope"})


def _nfc(valor: object, campo: str, patron: re.Pattern[str]) -> str:
    if not isinstance(valor, str) or not valor or unicodedata.normalize("NFC", valor) != valor \
            or not patron.fullmatch(valor):
        raise RuleSchemaError(f"{campo}: identificador no canonico")
    return valor


def _entero_positivo(valor: object, campo: str, *, tope: int = MAX_CANTIDAD) -> int:
    if isinstance(valor, bool) or not isinstance(valor, int) or valor <= 0 or valor > tope:
        raise RuleSchemaError(f"{campo}: debe ser entero positivo (<= {tope})")
    return valor


def _objeto_cerrado(valor: object, campo: str, claves: frozenset) -> dict:
    if not isinstance(valor, Mapping):
        raise RuleSchemaError(f"{campo}: debe ser objeto")
    sobrantes = set(valor) - claves
    if sobrantes:
        raise RuleSchemaError(f"{campo}: campos extra {sorted(sobrantes)}")
    return dict(valor)


def _lista_de_identificadores(valor: object, campo: str, patron: re.Pattern[str]) -> tuple[str, ...]:
    if not isinstance(valor, (list, tuple)):
        raise RuleSchemaError(f"{campo}: debe ser lista")
    if not valor:
        raise RuleSchemaError(f"{campo}: coleccion no vacia")
    items = tuple(_nfc(item, campo, patron) for item in valor)
    if len(items) != len(set(items)):
        raise RuleSchemaError(f"{campo}: duplicados")
    return tuple(sorted(items))


def _timestamp_utc(valor: object, campo: str) -> datetime:
    if not isinstance(valor, str) or not _RE_TIMESTAMP.fullmatch(valor):
        raise RuleSchemaError(f"{campo}: timestamp UTC exacto AAAA-MM-DDTHH:MM:SSZ")
    try:
        momento = datetime.strptime(valor, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise RuleSchemaError(f"{campo}: fecha inexistente") from exc
    return momento


def _validar_cantidad(valor: object) -> "Cantidad":
    datos = _objeto_cerrado(valor, "obligation_limits.quantity",
                            frozenset({"unit", "max"}))
    if "unit" not in datos or "max" not in datos:
        raise RuleSchemaError("obligation_limits.quantity: exige unit y max")
    unidad = _nfc(datos["unit"], "obligation_limits.quantity.unit", _RE_UNIDAD)
    maximo = _entero_positivo(datos["max"], "obligation_limits.quantity.max")
    return Cantidad(unit=unidad, max=maximo)


def _validar_monto(valor: object) -> "Monto":
    datos = _objeto_cerrado(valor, "obligation_limits.amount", frozenset({"currency", "max"}))
    if "currency" not in datos or "max" not in datos:
        raise RuleSchemaError("obligation_limits.amount: exige currency y max")
    moneda = _nfc(datos["currency"], "obligation_limits.amount.currency", _RE_MONEDA)
    # ISO 4217 completo se compara contra el contrato de la capability al evaluar;
    # aqui se fija la FORMA (3 mayusculas) y el maximo en unidades menores, entero.
    maximo = _entero_positivo(datos["max"], "obligation_limits.amount.max")
    return Monto(currency=moneda, max=maximo)


def _validar_frecuencia(valor: object) -> "Frecuencia":
    datos = _objeto_cerrado(valor, "obligation_limits.frequency",
                            frozenset({"max_occurrences", "window_seconds"}))
    if "max_occurrences" not in datos or "window_seconds" not in datos:
        raise RuleSchemaError("obligation_limits.frequency: exige max_occurrences y window_seconds")
    ocurrencias = _entero_positivo(datos["max_occurrences"], "frequency.max_occurrences")
    ventana = _entero_positivo(datos["window_seconds"], "frequency.window_seconds")
    return Frecuencia(max_occurrences=ocurrencias, window_seconds=ventana)


def _validar_tope(valor: object, catalogo: object) -> "Tope":
    datos = _objeto_cerrado(valor, "tope", frozenset({"resource_class", "resource",
                                                      "maximum", "period"}))
    for clave in ("resource_class", "resource", "maximum", "period"):
        if clave not in datos:
            raise RuleSchemaError(f"tope.{clave}: obligatorio")
    if catalogo is None:
        raise RuleSchemaError("tope: sin catalogo del pin no se topea nada (B-3)")
    if not isinstance(catalogo, CatalogoTopes):          # r7, MINOR-3: sellado o nada
        raise RuleSchemaError("tope: el catalogo se exige SELLADO (CatalogoTopes del pin)")
    clase = datos["resource_class"]
    if clase not in catalogo:
        raise RuleSchemaError("tope.resource_class: clase fuera del catalogo cerrado (R-4)")
    recurso = datos["resource"]
    if not isinstance(recurso, str) or not _RE_RECURSO.fullmatch(recurso):
        raise RuleSchemaError("tope.resource: no canonico")
    subid = recurso.partition(".")[2] if "." in recurso else ""
    if recurso.partition(".")[0] != clase or subid not in catalogo.get(clase, ()):
        raise RuleSchemaError(
            "tope.resource: fuera del catalogo cerrado — debe ser <clase>.<subid> de SU clase (R-4)")
    maximo = _entero_positivo(datos["maximum"], "tope.maximum")
    periodo = _nfc(datos["period"], "tope.period", _RE_PERIODO)
    return Tope(resource_class=clase, resource=recurso, maximum=maximo, period=periodo)


@dataclass(frozen=True)
class Cantidad:
    unit: str
    max: int


@dataclass(frozen=True)
class Monto:
    currency: str
    max: int          # unidades menores (centavos), entero exacto


@dataclass(frozen=True)
class Frecuencia:
    max_occurrences: int
    window_seconds: int


@dataclass(frozen=True)
class Tope:
    resource_class: str     # de CLASES_RECURSO_CON_TOPE (R-4)
    resource: str
    maximum: int
    period: str


@dataclass(frozen=True)
class Alcance:
    subjects: tuple[str, ...]
    capabilities: tuple[str, ...]
    objectives: tuple[str, ...]


@dataclass(frozen=True)
class LimitesObligatorios:
    quantity: Cantidad | None
    amount: Monto | None
    frequency: Frecuencia | None


@dataclass(frozen=True)
class Vigencia:
    not_before_utc: datetime
    not_after_utc: datetime | None      # [not_before, not_after): REVERSIBLE puede ser abierto


@dataclass(frozen=True)
class PermitConfig:
    ttl_seconds: int


@dataclass(frozen=True)
class ReglaValidada:
    """FORMA validada de una regla. No es autoridad: la autoridad es el snapshot
    sellado mas la ratificacion individual. Inmutable y sin sello a proposito."""
    schema_version: str
    kind: str
    rule_id: str
    effect: str
    action_class: str
    scope: Alcance
    obligation_limits: LimitesObligatorios
    validity: Vigencia
    permit: PermitConfig
    tope: Tope | None


def validar_regla(datos: object, *, ttl_max_seconds: int | None = None,
                  catalogo: object = None) -> ReglaValidada:
    """Valida el shape cerrado rule-v1. Falla cerrado con RuleSchemaError.

    ``ttl_max_seconds`` es el techo de configuracion confiable que aporta el
    kernel al evaluar; sin el, rige ``TTL_MAX_POR_DEFECTO``. ``catalogo`` es el
    catálogo cargado DESDE EL PIN (B-3): una regla con tope solo valida contra
    el catálogo del snapshot evaluado — sin catálogo, no se topea nada.
    """
    documento = _objeto_cerrado(datos, "regla", _CLAVES_TOPE_NIVEL)
    for clave in ("schema_version", "kind", "rule_id", "effect", "action_class",
                  "scope", "obligation_limits", "validity", "permit"):
        if clave not in documento:
            raise RuleSchemaError(f"falta {clave}")

    if documento["schema_version"] != SCHEMA_VERSION:
        raise RuleSchemaError("schema_version: unico valor 1.0")
    if documento["kind"] != KIND:
        raise RuleSchemaError("kind: unico valor JAX_FARO_RULE")
    if documento["effect"] != EFFECT:
        raise RuleSchemaError("effect: unico valor PERMIT")
    action_class = documento["action_class"]
    if action_class not in ACTION_CLASSES:
        raise RuleSchemaError("action_class: REVERSIBLE u OBLIGATING")
    rule_id = _nfc(documento["rule_id"], "rule_id", _RE_RULE_ID)

    alcance = _objeto_cerrado(documento["scope"], "scope",
                              frozenset({"subjects", "capabilities", "objectives"}))
    scope = Alcance(
        subjects=_lista_de_identificadores(alcance.get("subjects"), "scope.subjects", _RE_SUBJECT),
        capabilities=_lista_de_identificadores(alcance.get("capabilities"), "scope.capabilities",
                                               _RE_CAPABILITY),
        objectives=_lista_de_identificadores(alcance.get("objectives"), "scope.objectives",
                                             _RE_OBJETIVO),
    )

    limites = _objeto_cerrado(documento["obligation_limits"], "obligation_limits",
                              frozenset({"quantity", "amount", "frequency"}))
    for clave in ("quantity", "amount", "frequency"):
        if clave not in limites:
            raise RuleSchemaError(f"obligation_limits.{clave}: obligatorio (null si no aplica)")
    cantidad = _validar_cantidad(limites["quantity"]) if limites.get("quantity") is not None else None
    monto = _validar_monto(limites["amount"]) if limites.get("amount") is not None else None
    frecuencia = (_validar_frecuencia(limites["frequency"])
                  if limites.get("frequency") is not None else None)

    vigencia_datos = _objeto_cerrado(documento["validity"], "validity",
                                     frozenset({"not_before_utc", "not_after_utc"}))
    for clave in ("not_before_utc", "not_after_utc"):
        if clave not in vigencia_datos:
            raise RuleSchemaError(f"validity.{clave}: obligatorio (null si es abierto)")
    not_before = _timestamp_utc(vigencia_datos["not_before_utc"], "validity.not_before_utc")
    not_after = None
    if vigencia_datos.get("not_after_utc") is not None:
        not_after = _timestamp_utc(vigencia_datos["not_after_utc"], "validity.not_after_utc")
        if not_after <= not_before:
            raise RuleSchemaError("validity: not_after debe ser posterior a not_before")

    permit_datos = _objeto_cerrado(documento["permit"], "permit", frozenset({"ttl_seconds"}))
    if "ttl_seconds" not in permit_datos:
        raise RuleSchemaError("permit.ttl_seconds: obligatorio")
    techo_ttl = ttl_max_seconds if ttl_max_seconds is not None else TTL_MAX_POR_DEFECTO
    if techo_ttl <= 0:
        raise RuleSchemaError("ttl_max_seconds: techo de configuracion no positivo")
    ttl = _entero_positivo(permit_datos["ttl_seconds"], "permit.ttl_seconds", tope=techo_ttl)

    tope = (_validar_tope(documento["tope"], catalogo)
            if documento.get("tope") is not None else None)

    if action_class == "OBLIGATING":
        if not_after is None:
            raise RuleSchemaError("OBLIGATING exige vigencia finita (not_after_utc)")
        if frecuencia is None:
            raise RuleSchemaError("OBLIGATING exige frequency")
        if (cantidad is None) == (monto is None):
            raise RuleSchemaError("OBLIGATING exige exactamente uno de quantity o amount")

    return ReglaValidada(
        schema_version=SCHEMA_VERSION, kind=KIND, rule_id=rule_id, effect=EFFECT,
        action_class=action_class, scope=scope,
        obligation_limits=LimitesObligatorios(quantity=cantidad, amount=monto,
                                              frequency=frecuencia),
        validity=Vigencia(not_before_utc=not_before, not_after_utc=not_after),
        permit=PermitConfig(ttl_seconds=ttl), tope=tope,
    )
