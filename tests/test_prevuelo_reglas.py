"""Reglas puras del pre-vuelo (spec 2026-09-17 §4.1, §4.3, §4.4, §4.6; desvíos
2, 5 y 13 del plan). Sin DB ni red.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import os
from decimal import Decimal

import pytest

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from jacobs import prevuelo_reglas as pr  # noqa: E402
from jacobs.models import Step  # noqa: E402
from motor_registry.output_validator import puede_pedir_reintento  # noqa: E402


def _despacho(**cambios):
    base = dict(
        clave_salud="jekyll", via_motor=False, transporte="http_openai_compat",
        provider_id="deepseek", base_url="https://api.example/v1", modelo="deepseek-v4-flash",
        max_tokens_param="max_tokens", max_output_tokens=8192, motor_max_tokens=0,
        precio_in=Decimal("0.27"), precio_out=Decimal("1.10"),
        tiene_herramientas=False, schema_con_reintento=False, persona=None,
    )
    base.update(cambios)
    return pr.Despacho(**base)


def _evaluar(d, **k):
    args = dict(min_output_tokens=0, credencial_activa=True, chars_entrada=2000,
                chars_por_token=2, max_iteraciones=5)
    args.update(k)
    return pr.evaluar_paso(3, "jekyll", d, **args)


def test_faceta_inexistente_bloquea():
    v, c = pr.evaluar_paso(0, "zz", None, min_output_tokens=0, credencial_activa=False,
                           chars_entrada=0, chars_por_token=2, max_iteraciones=5)
    assert [x.regla for x in v] == ["faceta_inexistente"]
    assert c.usd_max is None and c.motivo == "faceta_inexistente"


def test_contrato_completo_y_credencial_no_bloquea():
    v, _ = _evaluar(_despacho())
    assert v == []


def test_openai_compat_sin_max_tokens_param_bloquea():
    v, c = _evaluar(_despacho(max_tokens_param=None))
    assert [x.regla for x in v] == ["sin_contrato_de_salida"]
    assert "UPDATE model SET max_tokens_param" in v[0].detalle
    assert c.usd_max is None and c.motivo == "sin_contrato_de_salida"


def test_gemini_sin_max_output_tokens_bloquea():
    v, _ = _evaluar(_despacho(transporte="http_gemini", max_tokens_param=None, max_output_tokens=None))
    assert [x.regla for x in v] == ["sin_contrato_de_salida"]
    assert "max_output_tokens" in v[0].detalle


def test_tope_igual_al_minimo_pasa():
    v, _ = _evaluar(_despacho(max_output_tokens=8192), min_output_tokens=8192)
    assert v == []


def test_motor_por_debajo_del_minimo_dice_subir_motor_max_tokens():
    d = _despacho(clave_salud="kimi", via_motor=True, provider_id="moonshot", modelo="kimi-k3",
                  max_output_tokens=131072, motor_max_tokens=8000)
    v, _ = _evaluar(d, min_output_tokens=16384)
    assert [x.regla for x in v] == ["tope_insuficiente"]
    assert "8000" in v[0].detalle and "motor.max_tokens" in v[0].detalle and "kimi" in v[0].detalle


def test_catalogo_por_debajo_del_minimo_dice_model_max_output_tokens():
    v, _ = _evaluar(_despacho(max_output_tokens=4096), min_output_tokens=8192)
    assert [x.regla for x in v] == ["tope_insuficiente"]
    assert "model.max_output_tokens" in v[0].detalle and "deepseek-v4-flash" in v[0].detalle


def test_motor_max_tokens_cero_usa_el_tope_del_catalogo():
    d = _despacho(via_motor=True, max_output_tokens=131072, motor_max_tokens=0)
    _, c = _evaluar(d)
    assert c.tokens_out_max == 131072


def test_credencial_ausente_bloquea_en_transporte_que_cobra():
    v, _ = _evaluar(_despacho(), credencial_activa=False)
    assert [x.regla for x in v] == ["credencial_ausente"]
    assert "deepseek" in v[0].detalle


def test_ollama_sin_credencial_no_bloquea_y_cuesta_cero():
    d = _despacho(transporte="ollama", via_motor=True, max_tokens_param=None,
                  precio_in=None, precio_out=None)
    v, c = _evaluar(d, credencial_activa=False)
    assert v == []
    assert c.usd_max == Decimal(0) and c.motivo == "local"


def test_costo_acotado_es_la_formula_redondeada_hacia_arriba():
    _, c = _evaluar(_despacho(), chars_entrada=2001)
    # (1001 * 0.27 + 8192 * 1.10) / 1e6 = 0.00928147 -> 0.009282
    assert (c.tokens_in_max, c.tokens_out_max, c.llamadas_max) == (1001, 8192, 1)
    assert c.usd_max == Decimal("0.009282") and c.motivo == "acotado"


def test_gemini_directo_cuenta_el_reintento_de_grounding():
    _, c = _evaluar(_despacho(transporte="http_gemini", max_tokens_param=None))
    assert c.llamadas_max == 2


def test_motor_con_schema_validable_cuenta_el_reintento():
    _, c = _evaluar(_despacho(via_motor=True, schema_con_reintento=True))
    assert c.llamadas_max == 2


def test_motor_sin_schema_validable_una_llamada():
    _, c = _evaluar(_despacho(via_motor=True, schema_con_reintento=False))
    assert c.llamadas_max == 1


def test_motor_con_herramientas_cuenta_todas_las_iteraciones():
    _, c = _evaluar(_despacho(via_motor=True, tiene_herramientas=True, schema_con_reintento=True),
                    max_iteraciones=5)
    assert c.llamadas_max == 5


def test_precio_null_no_acotado_y_no_bloquea():
    v, c = _evaluar(_despacho(precio_in=None))
    assert v == []
    assert c.usd_max is None and c.motivo == "sin_precio"


def test_tokens_de_entrada_redondea_hacia_arriba():
    assert [pr.tokens_de_entrada(n, 2) for n in (0, 1, 4, 5)] == [0, 1, 2, 3]


def test_hyde_no_se_evalua_y_cuesta_cero_por_suscripcion():
    c = pr.costo_sin_evaluar(Step(step_index=1, facet="hyde", capability="execute"))
    assert c is not None and c.usd_max == Decimal(0) and c.motivo == "suscripcion"


def test_assemble_no_se_evalua_y_es_mecanico():
    c = pr.costo_sin_evaluar(Step(step_index=2, facet="ada", capability="assemble"))
    assert c is not None and c.usd_max == Decimal(0) and c.motivo == "mecanico"
    assert pr.costo_sin_evaluar(Step(step_index=0, facet="jekyll", capability="research")) is None


def _costo(paso, usd):
    return pr.CostoPaso(paso, "jekyll", "m", 1, 1, 1, None if usd is None else Decimal(usd), "acotado")


def test_veredicto_suma_solo_los_acotados():
    costos = [_costo(0, "0.100000"), _costo(1, None), _costo(2, "0.200000")]
    v = pr.armar_veredicto([], costos, [])
    assert v.ok and v.costo_max_usd == Decimal("0.300000") and v.hay_no_acotados
    malo = pr.armar_veredicto([pr.Violacion(0, "jekyll", "faceta_caida", "x")], costos, ["jekyll"])
    assert not malo.ok and malo.sondeadas == ("jekyll",)


def test_to_dict_serializa_decimales_como_texto():
    v = pr.armar_veredicto([], [_costo(0, "0.100000"), _costo(1, None), _costo(2, "0.200000")], [])
    d = json.loads(json.dumps(v.to_dict()))
    assert d["costo_max_usd"] == "0.300000"
    assert d["pasos_costo"][1]["usd_max"] is None
    assert d["hay_no_acotados"] is True


def test_puede_pedir_reintento_segun_el_validador():
    assert puede_pedir_reintento("code_patch.v1") is True
    assert puede_pedir_reintento("critique.v1") is False
    assert puede_pedir_reintento("") is False
    assert puede_pedir_reintento(None) is False
    assert puede_pedir_reintento("typo.v9") is True


# ---- Ronda de arreglo 1 (revisión, Rulings R9-R11) ----


def test_motor_con_herramientas_no_esta_acotado():
    """R9a: worker.py no recorta el historial creciente ni los resultados de
    tools (worker.py:970) -- un motor con herramientas no tiene tope de costo
    conocido, aunque llamadas_max siga siendo el cálculo normal (el clamp de
    max_iteraciones)."""
    d = _despacho(via_motor=True, tiene_herramientas=True, schema_con_reintento=True)
    v, c = _evaluar(d)
    assert v == []
    assert c.usd_max is None and c.motivo == "herramientas_sin_tope"
    assert c.llamadas_max == 5


def test_motor_con_reintento_de_schema_acumula_el_costo():
    """R9b: la llamada k de un reintento de schema (worker.py:832-838) manda
    la respuesta de las k-1 llamadas previas como parte del prompt -- la
    entrada de la llamada k es tokens_in + (k-1)*tokens_out, no tokens_in
    repetido. Con tokens_in=1001, tokens_out=8192, precio_in=0.27,
    precio_out=1.10: llamada 1 = 1001*0.27 + 8192*1.10 = 9281.47; llamada 2 =
    (1001+8192)*0.27 + 8192*1.10 = 11493.31; total 20774.78 / 1e6 = 0.02077478
    -> 0.020775 redondeado hacia arriba."""
    d = _despacho(via_motor=True, schema_con_reintento=True)
    _, c = _evaluar(d, chars_entrada=2001)
    assert c.llamadas_max == 2
    assert c.usd_max == Decimal("0.020775") and c.motivo == "acotado"


def test_gemini_grounding_reintento_no_acumula_cobra_el_doble_exacto():
    """R9c: _invoke_http_gemini reenvía el MISMO payload en el reintento sin
    grounding (executor.py:307-316) -- sin acumulación, el costo es
    exactamente 2x el de una sola llamada (sobre el bruto sin redondear:
    2*9281.47/1e6 = 0.01856294 -> 0.018563)."""
    d = _despacho(transporte="http_gemini", max_tokens_param=None)
    _, c = _evaluar(d, chars_entrada=2001)
    assert c.llamadas_max == 2
    assert c.usd_max == Decimal("0.018563") and c.motivo == "acotado"


def test_ollama_sin_max_output_tokens_bloquea():
    """Desvío 5 del plan: ollama falla cerrado igual que un transporte que
    cobra si no declara max_output_tokens -- sin este test, errores_de_contrato
    podría "arreglarse" para devolver [] en ollama sin que nada lo note."""
    v, c = _evaluar(_despacho(transporte="ollama", via_motor=True, max_tokens_param=None,
                              max_output_tokens=None))
    assert [x.regla for x in v] == ["sin_contrato_de_salida"]
    assert c.usd_max is None and c.motivo == "sin_contrato_de_salida"


def test_tope_efectivo_motor_por_encima_del_catalogo_no_lo_baja():
    """Mitad no cubierta de tope_efectivo: cuando motor.max_tokens es MAYOR
    que el tope del catálogo, gana el catálogo (el motor no puede ampliar el
    tope de la fila de `model`) y "qué subir" queda en model.max_output_tokens."""
    d = _despacho(via_motor=True, max_output_tokens=4096, motor_max_tokens=8000)
    assert pr.tope_efectivo(d) == (4096, "model.max_output_tokens")


def test_tope_efectivo_http_ignora_motor_max_tokens():
    """La otra mitad: un paso HTTP directo (via_motor=False) no tiene
    MotorPolicy -- motor_max_tokens es ruido y se ignora aunque venga con un
    valor bajo."""
    d = _despacho(via_motor=False, max_output_tokens=8192, motor_max_tokens=1)
    assert pr.tope_efectivo(d) == (8192, "model.max_output_tokens")


def test_llamadas_max_respeta_el_clamp_de_max_iteraciones():
    """Sin este test, quitar el min() de llamadas_max (motor sin herramientas
    con reintento de schema) no lo nota nada: con max_iteraciones=1 el
    reintento de schema NO cabe, tiene que quedar en 1."""
    d = _despacho(via_motor=True, schema_con_reintento=True)
    assert pr.llamadas_max(d, max_iteraciones=1) == 1


def test_subprocess_no_hyde_no_explota_y_es_suscripcion():
    """R11: subprocess sin contrato de salida (max_output_tokens=None) no
    puede pasar por tope_efectivo -- None < int revienta con TypeError. Toda
    faceta subprocess (no solo hyde) es suscripción: costo 0, sin chequeo de
    tope ni credencial."""
    d = _despacho(transporte="subprocess", max_tokens_param=None, max_output_tokens=None,
                  precio_in=None, precio_out=None)
    v, c = _evaluar(d)
    assert v == []
    assert c.usd_max == Decimal(0) and c.motivo == "suscripcion"


def test_transporte_desconocido_se_rechaza():
    """R10: un transporte fuera de {http_openai_compat, http_gemini, ollama,
    subprocess} no es un caso libre de contrato -- es uno que el pre-vuelo
    todavía no conoce, y falla cerrado con el nombre del transporte en el
    detalle."""
    v, c = _evaluar(_despacho(transporte="websocket_experimental"))
    assert [x.regla for x in v] == ["sin_contrato_de_salida"]
    assert "websocket_experimental" in v[0].detalle
    assert c.usd_max is None and c.motivo == "sin_contrato_de_salida"


def test_armar_veredicto_ordena_pasos_costo_y_sondeadas():
    costos = [_costo(2, "0.100000"), _costo(0, "0.200000"), _costo(1, None)]
    v = pr.armar_veredicto([], costos, ["zeta", "alfa"])
    assert [c.paso for c in v.pasos_costo] == [0, 1, 2]
    assert v.sondeadas == ("alfa", "zeta")


def test_violacion_regla_desconocida_rechaza():
    with pytest.raises(ValueError):
        pr.Violacion(0, "jekyll", "regla_inventada", "detalle")


def test_dos_errores_de_contrato_se_unen_con_pipe():
    v, _ = _evaluar(_despacho(max_tokens_param=None, max_output_tokens=None))
    assert [x.regla for x in v] == ["sin_contrato_de_salida"]
    assert " | " in v[0].detalle
    assert "max_tokens_param" in v[0].detalle
    assert "max_output_tokens" in v[0].detalle


# ---- Ronda de arreglo 2 (revisión, Ruling R9a acotada a TRANSPORTES_QUE_COBRAN) ----


def test_motor_ollama_con_herramientas_sigue_siendo_local():
    """R9a solo aplica a transportes que cobran por token
    (TRANSPORTES_QUE_COBRAN): ollama es gratis pase lo que pase en el bucle
    de herramientas -- "sin tope" no tiene sentido cuando el costo real es
    siempre $0. Tiene que seguir siendo motivo "local", NO
    "herramientas_sin_tope"."""
    d = _despacho(transporte="ollama", via_motor=True, max_tokens_param=None,
                  precio_in=None, precio_out=None, tiene_herramientas=True)
    v, c = _evaluar(d)
    assert v == []
    assert c.usd_max == Decimal(0) and c.motivo == "local"


def test_motor_con_herramientas_en_transporte_que_cobra_no_esta_acotado():
    """El otro lado de la misma regla: un transporte que SÍ cobra
    (http_openai_compat, en TRANSPORTES_QUE_COBRAN) con herramientas activas
    queda sin tope de costo conocido -- acá "sin tope" sí importa porque hay
    un precio real que podría dispararse."""
    d = _despacho(transporte="http_openai_compat", via_motor=True, tiene_herramientas=True)
    v, c = _evaluar(d)
    assert v == []
    assert c.usd_max is None and c.motivo == "herramientas_sin_tope"


# ---- Ronda de arreglo 3 (segunda re-revisión) ----


def test_costo_acumulado_redondea_hacia_arriba_no_al_mas_cercano():
    """La cuantización de costo_usd_acumulado_motor es ROUND_CEILING (hacia
    arriba), NO ROUND_HALF_EVEN (bancario): con tokens_out=0 la suma de las
    2 llamadas es 2*1*5117.2 = 10234.4, bruto = 0.0102344 -- el séptimo
    decimal es 4, POR DEBAJO de la mitad. ROUND_HALF_EVEN redondearía para
    abajo (0.010234, el mismo truncado); solo ROUND_CEILING sube a 0.010235.
    Sin este caso, cambiar el rounding a HALF_EVEN no lo nota nada (todos
    los demás casos del archivo tienen un séptimo decimal >= 5, donde los
    dos modos coinciden)."""
    c = pr.costo_usd_acumulado_motor(2, 1, 0, Decimal("5117.2"), Decimal("0"))
    assert c == Decimal("0.010235")


def test_http_directo_con_herramientas_no_es_motor_y_queda_acotado():
    """El "sin tope" de R9a es del Motor Registry (via_motor): un paso HTTP
    DIRECTO (via_motor=False) con tiene_herramientas=True no pasa por el
    bucle de worker.py -- tools ahí no significa historial sin límite, y el
    costo sigue siendo el cálculo normal de una sola llamada."""
    d = _despacho(via_motor=False, tiene_herramientas=True)
    _, c = _evaluar(d, chars_entrada=2001)
    assert c.usd_max == Decimal("0.009282") and c.motivo == "acotado"
