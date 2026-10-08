"""Schema cerrado de regla Faro v1 (diseno F1.1 §7 + R-4).

Toda regla de ``policy/faro/*.yaml`` valida contra un shape cerrado: campos
extra se rechazan, los vocabularios son cerrados y los identificadores van en
NFC. Estas pruebas negativas son las que el diseno exige: cada defecto del
schema tiene la prueba que lo atrapa.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from policy.canonicalization.errors import StrictYAMLError
from policy.canonicalization.strict_yaml import load_strict_yaml
from tests.policy.catalogo_pin import catalogo_del_pin
from policy.rule_authority.errors import RuleSchemaError
from policy.rule_authority.schema import TTL_MAX_POR_DEFECTO, validar_regla as _validar

# El catalogo de la decision de Fernando (B-3): llega del pin, nunca de un global
# r7, MAJOR-1: el catalogo de las pruebas sale de un PIN de prueba con esta
# decision (mismas cinco clases que la de Fernando), nunca de bytes sueltos
CATALOGO = catalogo_del_pin(json.dumps(
    {"version": 1, "decision": "Fernando 2026-10-06: solo actos y dinero",
     "clases": {"monto_dinero": ["hnl", "usd"],
                "actos_externos": ["mensajes", "correos", "publicaciones", "compras", "pagos"],
                "frecuencia": ["por_hora", "por_dia"], "duracion": ["segundos"],
                "tokens_costo": ["tokens", "usd"]}}).encode())


def validar_regla(datos, **kwargs):
    kwargs.setdefault("catalogo", CATALOGO)
    return _validar(datos, **kwargs)

FIXTURES = Path(__file__).parent / "fixtures" / "faro_rules"


def _regla(nombre: str = "regla-ejemplo.yaml") -> dict:
    return load_strict_yaml((FIXTURES / nombre).read_text())


def test_el_ejemplo_del_diseno_es_valido() -> None:
    regla = validar_regla(_regla())
    assert regla.rule_id == "ejemplo-regla"
    assert regla.kind == "JAX_FARO_RULE"
    assert regla.effect == "PERMIT"
    assert regla.action_class == "REVERSIBLE"


def test_el_ejemplo_con_tope_es_valido_y_declara_su_clase() -> None:
    regla = validar_regla(_regla("regla-ejemplo-tope.yaml"))
    assert regla.tope is not None
    assert regla.tope.resource_class == "actos_externos"
    assert regla.tope.resource == "actos_externos.mensajes"
    assert regla.tope.maximum == 8


# ---------------------------------------------------------------- shape cerrado

@pytest.mark.parametrize("campo", [
    "schema_version", "kind", "rule_id", "effect", "action_class",
    "scope", "obligation_limits", "validity", "permit",
])
def test_falta_un_campo_obligatorio_rechaza(campo: str) -> None:
    datos = _regla()
    del datos[campo]
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_campo_extra_rechaza() -> None:
    datos = _regla()
    datos["nota"] = "campo extra"
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_campo_extra_en_objeto_interno_rechaza() -> None:
    for objeto, campo in [("scope", "extra"), ("obligation_limits", "extra"),
                          ("validity", "extra"), ("permit", "extra")]:
        datos = _regla()
        datos[objeto][campo] = 1
        with pytest.raises(RuleSchemaError):
            validar_regla(datos)


def test_objeto_interno_con_clave_faltante_rechaza() -> None:
    datos = _regla()
    del datos["scope"]["subjects"]
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


# ------------------------------------------------------- vocabularios cerrados

@pytest.mark.parametrize("campo,valor", [
    ("schema_version", "1.1"),
    ("schema_version", 1.0),
    ("kind", "JAX_FARO_OTRO"),
    ("effect", "DENY"),
    ("effect", "permit"),
    ("action_class", "IRREVERSIBLE"),
    ("action_class", "reversible"),
])
def test_valor_fuera_de_vocabulario_rechaza(campo: str, valor: object) -> None:
    datos = _regla()
    datos[campo] = valor
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


# ------------------------------------------------------------ identificadores

@pytest.mark.parametrize("rule_id", [
    "Ejemplo",             # mayuscula inicial
    "ejemplo_regla",       # guion bajo no va con el vocabulario de ids de regla
    "ejemplo regla",       # espacio
    "-ejemplo",            # guion inicial
    "cafe\u0301",          # NFD: no esta en NFC
    "",
])
def test_rule_id_no_canonico_rechaza(rule_id: str) -> None:
    datos = _regla()
    datos["rule_id"] = rule_id
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


@pytest.mark.parametrize("subject", ["ejemplo", "ACTOR:ejemplo", "actor:", "cafe\u0301:x"])
def test_subject_mal_formado_rechaza(subject: str) -> None:
    datos = _regla()
    datos["scope"]["subjects"] = [subject]
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


@pytest.mark.parametrize("capability", ["capability_id", "CAP-", "CAFE\u0301", ""])
def test_capability_mal_formada_rechaza(capability: str) -> None:
    datos = _regla()
    datos["scope"]["capabilities"] = [capability]
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_coleccion_vacia_rechaza() -> None:
    datos = _regla()
    datos["scope"]["objectives"] = []
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_coleccion_con_duplicados_rechaza() -> None:
    datos = _regla()
    datos["scope"]["subjects"] = ["actor:a", "actor:a"]
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_coleccion_que_no_es_lista_rechaza() -> None:
    datos = _regla()
    datos["scope"]["subjects"] = "actor:ejemplo"
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


# ------------------------------------------------------------------ vigencia

def test_timestamp_sin_z_rechaza() -> None:
    datos = _regla()
    datos["validity"]["not_before_utc"] = "2026-10-05T00:00:00"
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_timestamp_con_offset_rechaza() -> None:
    datos = _regla()
    datos["validity"]["not_before_utc"] = "2026-10-05T00:00:00-06:00"
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_timestamp_inexistente_rechaza() -> None:
    datos = _regla()
    datos["validity"]["not_before_utc"] = "2026-02-30T00:00:00Z"
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_not_after_antes_de_not_before_rechaza() -> None:
    datos = _regla()
    datos["validity"]["not_after_utc"] = "2026-10-04T00:00:00Z"
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_reversible_puede_tener_fin_abierto() -> None:
    regla = validar_regla(_regla())      # el ejemplo base es REVERSIBLE con fin abierto
    assert regla.validity.not_after_utc is None


# ------------------------------------------------------------------ permit.ttl

@pytest.mark.parametrize("ttl", [0, -1, "60", 1.5, True])
def test_ttl_no_entero_positivo_rechaza(ttl: object) -> None:
    datos = _regla()
    datos["permit"]["ttl_seconds"] = ttl
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_ttl_sobre_el_maximo_configurable_rechaza() -> None:
    datos = _regla()
    datos["permit"]["ttl_seconds"] = TTL_MAX_POR_DEFECTO + 1
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_el_techo_de_ttl_lo_aporta_la_configuracion_confiable() -> None:
    datos = _regla()
    datos["permit"]["ttl_seconds"] = 300
    validar_regla(datos, ttl_max_seconds=300)                 # el kernel pasa su config
    datos["permit"]["ttl_seconds"] = 301
    with pytest.raises(RuleSchemaError):
        validar_regla(datos, ttl_max_seconds=300)


# --------------------------------------------------------- obligation_limits

def test_obligating_sin_not_after_rechaza() -> None:
    datos = _regla("regla-ejemplo-tope.yaml")
    datos["validity"]["not_after_utc"] = None                # OBLIGATING exige fin finito
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_obligating_sin_frecuencia_rechaza() -> None:
    datos = _regla("regla-ejemplo-tope.yaml")
    datos["obligation_limits"]["frequency"] = None
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_obligating_sin_cantidad_ni_monto_rechaza() -> None:
    datos = _regla("regla-ejemplo-tope.yaml")
    datos["obligation_limits"]["quantity"] = None             # ya no queda ni quantity ni amount
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_obligating_con_cantidad_y_monto_rechaza() -> None:
    datos = _regla("regla-ejemplo-tope.yaml")
    datos["obligation_limits"]["amount"] = {"currency": "USD", "max": 100}
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_cantidad_sin_unidad_o_sin_maximo_rechaza() -> None:
    for mutacion in ({"max": 100}, {"unit": "llamadas"}):
        datos = _regla("regla-ejemplo-tope.yaml")
        datos["obligation_limits"]["quantity"] = mutacion
        with pytest.raises(RuleSchemaError):
            validar_regla(datos)


def test_cantidad_con_maximo_no_positivo_rechaza() -> None:
    datos = _regla("regla-ejemplo-tope.yaml")
    datos["obligation_limits"]["quantity"] = {"unit": "llamadas", "max": 0}
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_monto_con_moneda_no_iso_rechaza() -> None:
    datos = _regla("regla-ejemplo-tope.yaml")
    datos["obligation_limits"]["quantity"] = None
    datos["obligation_limits"]["amount"] = {"currency": "usd", "max": 100}
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_monto_con_flotante_rechaza() -> None:
    datos = dict(_regla("regla-ejemplo-tope.yaml"))
    datos["obligation_limits"]["quantity"] = None
    datos["obligation_limits"]["amount"] = {"currency": "USD", "max": 100.5}
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_frecuencia_sin_maximo_o_sin_ventana_rechaza() -> None:
    for mutacion in ({"max_occurrences": 10}, {"window_seconds": 60}):
        datos = _regla("regla-ejemplo-tope.yaml")
        datos["obligation_limits"]["frequency"] = mutacion
        with pytest.raises(RuleSchemaError):
            validar_regla(datos)


# ------------------------------------------------------------------ tope (R-4)

def test_tope_con_clase_fuera_de_vocabulario_rechaza() -> None:
    datos = _regla("regla-ejemplo-tope.yaml")
    datos["tope"]["resource_class"] = "agentes"              # D-4: ni existe la clase
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_el_vocabulario_de_clases_es_el_de_actos_y_dinero() -> None:
    # DECISION de Fernando (2026-10-06): solo actos y dinero; jamas infraestructura.
    assert tuple(sorted(CATALOGO)) == (
        "actos_externos", "duracion", "frecuencia", "monto_dinero", "tokens_costo",
    )


@pytest.mark.parametrize("clase", [
    "connections", "concurrencia", "workers", "procesos_hijos", "hilos",
    "tasks", "llamadas_paralelas", "agentes", "enjambre", "conexion",
])
def test_clase_de_infraestructura_o_agentes_no_existe_en_el_vocabulario(clase: str) -> None:
    datos = _regla("regla-ejemplo-tope.yaml")
    datos["tope"]["resource_class"] = clase
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


@pytest.mark.parametrize("recurso", [
    "agentes", "subagentes.globales", "enjambre.total", "swarm.workers",
    "conexiones.remotas", "connections.remotas", "puerto.connections",   # D-4, ambos idiomas
    "llm.paralelo", "workers.maximo", "hilos.pool", "procesos.hijos", "threads",
    "concurrencia.maxima", "concurrency.limit", "processes",
])
def test_tope_sobre_recurso_de_infraestructura_o_agentes_rechaza(recurso: str) -> None:
    # Aunque la clase declarada sea legitima (actos_externos), el nombre del
    # recurso delata infraestructura o agentes: NUNCA llevan tope.
    datos = _regla("regla-ejemplo-tope.yaml")
    datos["tope"]["resource"] = recurso
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_tope_sin_campos_obligatorios_rechaza() -> None:
    for campo in ("resource_class", "resource", "maximum", "period"):
        datos = _regla("regla-ejemplo-tope.yaml")
        del datos["tope"][campo]
        with pytest.raises(RuleSchemaError):
            validar_regla(datos)


def test_tope_con_campo_extra_rechaza() -> None:
    datos = _regla("regla-ejemplo-tope.yaml")
    datos["tope"]["extras"] = 1
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_tope_con_maximo_flotante_o_booleano_rechaza() -> None:
    for maximo in (8.5, True, "8"):
        datos = _regla("regla-ejemplo-tope.yaml")
        datos["tope"]["maximum"] = maximo
        with pytest.raises(RuleSchemaError):
            validar_regla(datos)


def test_tope_con_maximo_sobre_el_rango_representable_rechaza() -> None:
    datos = _regla("regla-ejemplo-tope.yaml")
    datos["tope"]["maximum"] = 2**53 + 1
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


class _StrSub(str):
    pass


@pytest.mark.parametrize("periodo", [
    "una hora!", "workers", "conexiones_db", "Por_Hora", "por_hora\u200b", "por\u200b_hora",
    "por_hora ", " por_hora", "hora", "", 3600, None, True, ["por_hora"],
    _StrSub("por_hora"), "frecuencia.por_hora", "duracion", "segundos",
    "por_h\u00f3ra", "por_hora\n",
])
def test_tope_con_periodo_fuera_del_catalogo_rechaza(periodo: object) -> None:
    datos = _regla("regla-ejemplo-tope.yaml")
    datos["tope"]["period"] = periodo
    with pytest.raises(RuleSchemaError) as excinfo:
        validar_regla(datos)
    assert "tope.period" in str(excinfo.value)


def _periodos_del_catalogo_real() -> list[str]:
    raiz = Path(__file__).resolve().parents[2]
    clases = json.loads((raiz / "policy" / "faro" / "catalogo-topes.json").read_text())["clases"]
    return list(clases["frecuencia"])


@pytest.mark.parametrize("periodo", _periodos_del_catalogo_real())
def test_tope_con_cada_periodo_del_catalogo_real_valida(periodo: str) -> None:
    datos = _regla("regla-ejemplo-tope.yaml")
    datos["tope"]["period"] = periodo
    assert validar_regla(datos).tope.period == periodo


def test_los_periodos_legitimos_no_estan_vacios() -> None:
    assert {"por_hora", "por_dia"} <= set(_periodos_del_catalogo_real())


def test_el_periodo_sale_del_catalogo_del_pin_no_de_una_lista_propia() -> None:
    """Un pin con otra frecuencia ('por_semana') la admite; 'por_hora' ausente, niega."""
    otro = catalogo_del_pin(json.dumps(
        {"version": 1, "decision": "x",
         "clases": {"monto_dinero": ["usd"], "actos_externos": ["mensajes"],
                    "frecuencia": ["por_semana"], "duracion": ["segundos"],
                    "tokens_costo": ["tokens"]}}).encode())
    datos = _regla("regla-ejemplo-tope.yaml")
    datos["tope"]["period"] = "por_semana"
    assert _validar(datos, catalogo=otro).tope.period == "por_semana"
    datos["tope"]["period"] = "por_hora"
    with pytest.raises(RuleSchemaError):
        _validar(datos, catalogo=otro)


# ------------------------------------------------ inmutabilidad y tipado duro

def test_la_regla_validada_es_profundamente_inmutable() -> None:
    regla = validar_regla(_regla())
    assert isinstance(regla.scope.subjects, tuple)
    with pytest.raises(Exception):
        regla.rule_id = "otro"        # type: ignore[misc]


def test_un_dict_que_no_es_dict_rechaza() -> None:
    with pytest.raises(RuleSchemaError):
        validar_regla(["no", "soy", "regla"])                 # type: ignore[arg-type]


def test_mutar_el_dict_de_entrada_despues_de_validar_no_cambia_la_regla() -> None:
    datos = _regla()
    regla = validar_regla(datos)
    copia = copy.deepcopy(datos)
    datos["scope"]["subjects"] = ["actor:otro"]
    datos["permit"]["ttl_seconds"] = 9999
    assert regla.scope.subjects == validar_regla(copia).scope.subjects
    assert regla.permit.ttl_seconds == 60


def test_digito_unicode_en_timestamp_rechaza() -> None:
    datos = _regla()
    datos["validity"]["not_before_utc"] = "\u0662\u0660\u0662\u0666-10-05T00:00:00Z"   # ٢٠٢٦
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_obligation_limits_sin_las_tres_claves_rechaza() -> None:
    datos = _regla()
    del datos["obligation_limits"]["amount"]
    with pytest.raises(RuleSchemaError) as excinfo:
        validar_regla(datos)
    assert "amount" in str(excinfo.value)          # muere por ESTA clave, no por otra


def test_validity_sin_not_after_rechaza() -> None:
    datos = _regla()
    del datos["validity"]["not_after_utc"]
    with pytest.raises(RuleSchemaError) as excinfo:
        validar_regla(datos)
    assert "not_after_utc" in str(excinfo.value)   # M20: prueba propia, razon propia


# ------------------------------------------- espejo JSON Schema (rule-v1)

def test_el_espejo_json_existe_y_sus_vocabularios_coinciden() -> None:
    import json
    espejo = json.loads((Path(__file__).resolve().parents[2] / "policy" / "faro"
                         / "schemas" / "rule-v1.schema.json").read_text())
    propiedades = espejo["properties"]
    assert propiedades["kind"]["enum"] == ["JAX_FARO_RULE"]
    assert propiedades["effect"]["enum"] == ["PERMIT"]
    assert propiedades["action_class"]["enum"] == ["REVERSIBLE", "OBLIGATING"]
    tope_objeto = propiedades["tope"]["oneOf"][1]
    clases = tope_objeto["properties"]["resource_class"]["enum"]
    assert tuple(clases) == tuple(sorted(CATALOGO))          # B-3: del catalogo, no de un global
    assert espejo.get("additionalProperties") is False


def _hacer_obligante_valida(datos: dict) -> None:
    datos["action_class"] = "OBLIGATING"
    datos["obligation_limits"] = {
        "quantity": {"unit": "mensajes", "max": 1},
        "amount": None,
        "frequency": {"max_occurrences": 1, "window_seconds": 60},
    }
    datos["validity"]["not_after_utc"] = "2026-10-05T00:01:00Z"


def _validador_espejo(espejo: dict):
    from datetime import datetime
    from jsonschema import Draft202012Validator, FormatChecker

    checker = FormatChecker()

    @checker.checks("date-time")
    def _fecha_utc_canonica(valor: object) -> bool:
        if not isinstance(valor, str):
            return True
        try:
            datetime.strptime(valor, "%Y-%m-%dT%H:%M:%SZ")
            return True
        except ValueError:
            return False

    return Draft202012Validator(espejo, format_checker=checker)


@pytest.mark.parametrize("cambio,esperado", [
    (lambda datos: datos.__setitem__("tope", None), True),
    (lambda datos: datos.__setitem__("tope", {
        "resource_class": "actos_externos", "resource": "actos_externos.mensajes",
        "maximum": 1, "period": "por_hora",
    }), True),
    (lambda datos: datos.__setitem__("tope", {
        "resource_class": "actos_externos", "resource": "actos_externos.mensajes",
        "maximum": 1, "period": "inventado",
    }), False),
    (lambda datos: datos.__setitem__("tope", {
        "resource_class": "actos_externos", "resource": "monto_dinero.hnl",
        "maximum": 1, "period": "por_hora",
    }), False),
    (_hacer_obligante_valida, True),
    (lambda datos: datos.__setitem__("action_class", "OBLIGATING"), False),
])
def test_el_espejo_json_y_python_aceptan_los_mismos_vectores_de_regla(cambio, esperado) -> None:
    """Una brecha permite que otra herramienta acepte una regla que Faro niega."""
    datos = _regla()
    cambio(datos)
    espejo = json.loads((Path(__file__).resolve().parents[2] / "policy" / "faro"
                         / "schemas" / "rule-v1.schema.json").read_text())
    json_acepta = not list(_validador_espejo(espejo).iter_errors(datos))
    try:
        validar_regla(datos)
        python_acepta = True
    except RuleSchemaError:
        python_acepta = False

    assert python_acepta is esperado
    assert json_acepta is esperado


def test_formato_json_schema_rechaza_fecha_calendaria_inexistente() -> None:
    datos = _regla()
    datos["validity"]["not_before_utc"] = "2026-02-30T00:00:00Z"
    espejo = json.loads((Path(__file__).resolve().parents[2] / "policy" / "faro"
                         / "schemas" / "rule-v1.schema.json").read_text())
    assert list(_validador_espejo(espejo).iter_errors(datos))
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_orden_temporal_es_semantica_python_que_schema_estandar_no_expresa() -> None:
    """El éxito estructural del espejo nunca sustituye el validador de autoridad."""
    datos = _regla()
    datos["validity"]["not_after_utc"] = datos["validity"]["not_before_utc"]
    espejo = json.loads((Path(__file__).resolve().parents[2] / "policy" / "faro"
                         / "schemas" / "rule-v1.schema.json").read_text())
    assert not list(_validador_espejo(espejo).iter_errors(datos))
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


# ------------------------------------- YAML estricto: la fuente ya es cerrada

def test_el_yaml_de_una_regla_con_clave_duplicada_no_carga() -> None:
    texto = (FIXTURES / "regla-ejemplo.yaml").read_text() + "\nrule_id: otra\n"
    with pytest.raises(StrictYAMLError):
        load_strict_yaml(texto)
