"""Herramientas cuya salida se captura (Ejecutor Fase 2).

Existen porque hay datos verdaderos que ninguna salida imprime literal: los
conteos (tarea 8 de U3, «14 servicios con certificado SSL») y las conversiones
(tarea 2, «89 GiB» escrito de cabeza como «~91 GB»). La herramienta hace la
operación y deja su resultado en una captura, donde sí es citable.
"""
from dataclasses import fields

import pytest

from jax.ejecutor.captura import CapturaCompleta, a_captura
from jax.ejecutor.cita import RESPALDADA, Afirmacion, verificar
from jax.ejecutor.herramientas import HerramientaRechazada, contar, convertir


def _origen(salida, truncada=False, codigo=0, maquina="atemai",
            comando="certbot certificates", motivos=()):
    return CapturaCompleta(
        maquina=maquina, comando=comando, codigo=codigo, salida=salida,
        stderr="", truncada=truncada, bytes_totales=len(salida.encode()),
        momento="2026-09-16T10:00:00+00:00", motivos_truncado=motivos)


def _citable(completa, literal):
    """La línea de la salida que contiene `literal`, citada contra la captura:
    tiene que quedar RESPALDADA por el verificador real."""
    lineas = [l for l in completa.salida.splitlines() if literal in l]
    assert lineas, f"{literal!r} no está literal en ninguna línea de {completa.salida!r}"
    campos = dict(maquina=completa.maquina, texto=literal,
                  comando=completa.comando, linea=lineas[0])
    # `cita.py` está sumando el campo `dato` en paralelo (liga la afirmación a
    # su cita). Cuando entre, quitar este condicional y pasar `dato` siempre.
    if "dato" in {f.name for f in fields(Afirmacion)}:
        campos["dato"] = literal
    afirmacion = Afirmacion(**campos)
    return verificar(afirmacion, [a_captura(completa)]).estado


# --- contar -----------------------------------------------------------------

_CATORCE_SSL = "\n".join(
    [f"  Certificate Name: sitio{i}.axioma-ia.io SSL" for i in range(14)]
    + ["  Expiry Date: 2026-12-01", "Found the following certs:"]) + "\n"


def test_contar_14_lineas_da_14_y_es_citable():
    c = contar(_origen(_CATORCE_SSL), "SSL")
    assert isinstance(c, CapturaCompleta)
    assert c.truncada is False
    assert c.codigo == 0
    assert c.maquina == "atemai"
    assert "14" in c.salida
    assert _citable(c, "14") == RESPALDADA


def test_la_linea_de_contar_nombra_sus_entradas():
    c = contar(_origen(_CATORCE_SSL), "SSL")
    linea = [l for l in c.salida.splitlines() if "14" in l][0]
    assert "'SSL'" in linea
    assert "certbot certificates" in linea
    assert "16" in linea  # total de líneas examinadas


def test_el_comando_de_contar_describe_la_operacion_y_su_origen():
    c = contar(_origen(_CATORCE_SSL), "SSL")
    assert "contar" in c.comando
    assert "'SSL'" in c.comando
    assert "certbot certificates" in c.comando
    assert "atemai" in c.comando
    assert "2026-09-16T10:00:00+00:00" in c.comando


def test_contar_es_subcadena_literal_y_distingue_mayusculas():
    origen = _origen("a.b\naxb\nA.B\n")
    assert "1 lineas" in contar(origen, "a.b").salida


def test_contar_sobre_captura_TRUNCADA_no_cuenta():
    """Tarea 9 de U3: 2 KB leídos de 85,9 KB. Un conteo sobre una salida
    cortada es un número que nadie imprimió y que además es falso."""
    origen = _origen(_CATORCE_SSL, truncada=True, motivos=("tope_bytes",))
    with pytest.raises(HerramientaRechazada, match="truncada"):
        contar(origen, "SSL")


def test_contar_sobre_comando_fallido_no_cuenta():
    """Un ssh caído deja stdout vacío: contar daría «0 servicios»."""
    with pytest.raises(HerramientaRechazada, match="código"):
        contar(_origen("", codigo=255), "SSL")


def test_contar_con_patron_vacio_no_cuenta():
    with pytest.raises(HerramientaRechazada, match="vacío"):
        contar(_origen(_CATORCE_SSL), "")


def test_contar_con_patron_multilinea_no_cuenta():
    with pytest.raises(HerramientaRechazada, match="salto de línea"):
        contar(_origen(_CATORCE_SSL), "SSL\nExpiry")


def test_el_patron_no_puede_partir_la_linea_citable():
    origen = _origen("x y\n", comando="echo 'raro\nde verdad'")
    c = contar(origen, "x")
    assert len(c.salida.splitlines()) == 1


# --- convertir --------------------------------------------------------------

def test_89_GiB_son_95_6_GB_y_no_91():
    c = convertir(89, "GiB", "GB", maquina="atemai")
    assert isinstance(c, CapturaCompleta)
    assert "95.6 GB" in c.salida
    assert "91 GB" not in c.salida
    assert _citable(c, "95.6 GB") == RESPALDADA


def test_convertir_escribe_factores_redondeo_y_valor_exacto():
    c = convertir(89, "GiB", "GB", maquina="atemai")
    linea = [l for l in c.salida.splitlines() if "95.6 GB" in l][0]
    assert "89 GiB" in linea
    assert "1024^3" in linea and "1000^3" in linea
    assert "1 decimal" in linea
    assert "95.563022336" in linea


def test_el_comando_de_convertir_es_reproducible():
    c = convertir(89, "GiB", "GB", maquina="atemai")
    assert c.comando == "convertir 89 GiB a GB (1 decimal)"
    assert c.maquina == "atemai"
    assert c.truncada is False and c.codigo == 0


def test_convertir_en_el_otro_sentido_y_decimales_explicitos():
    c = convertir("1863", "GB", "GiB", maquina="atemai", decimales=2)
    assert "1735.05 GiB" in c.salida
    assert "2 decimales" in c.salida


def test_convertir_redondea_mitad_hacia_arriba():
    # 1.25 KB -> 1.3 con mitad hacia arriba (no 1.2 del redondeo bancario)
    assert "1.3 KB" in convertir("1250", "B", "KB", maquina="m").salida


@pytest.mark.parametrize("unidad", ["G", "gb", "Gb", "GiB ", "", "XB"])
def test_unidad_desconocida_es_error_nunca_un_valor_adivinado(unidad):
    with pytest.raises(HerramientaRechazada, match="unidad"):
        convertir(89, unidad, "GB", maquina="atemai")
    with pytest.raises(HerramientaRechazada, match="unidad"):
        convertir(89, "GiB", unidad, maquina="atemai")


@pytest.mark.parametrize("valor", ["89,5", "abc", "nan", "inf", -1, True, None])
def test_valor_invalido_es_error(valor):
    with pytest.raises(HerramientaRechazada, match="valor"):
        convertir(valor, "GiB", "GB", maquina="atemai")


def test_convertir_sin_maquina_es_error():
    with pytest.raises(HerramientaRechazada, match="máquina"):
        convertir(89, "GiB", "GB", maquina=" ")


def test_decimales_invalidos_es_error():
    with pytest.raises(HerramientaRechazada, match="decimales"):
        convertir(89, "GiB", "GB", maquina="atemai", decimales=-1)


def test_convertir_no_pierde_precision_con_valores_grandes():
    """El contexto por defecto de Decimal tiene 28 dígitos: con más, el
    cálculo revienta o redondea en silencio. El exacto tiene que ser exacto."""
    c = convertir("123456789012345678901234567.891", "GiB", "B", maquina="m", decimales=0)
    milesimas = 123456789012345678901234567891 * 1024 ** 3
    redondeado = (milesimas + 500) // 1000
    exacto = f"{milesimas // 1000}.{milesimas % 1000:03d}".rstrip("0").rstrip(".")
    assert f"= {redondeado} B" in c.salida
    assert f"exacto: {exacto})" in c.salida


def test_convertir_que_no_cabe_en_la_precision_es_error_no_redondeo_silencioso():
    with pytest.raises(HerramientaRechazada, match="precisión"):
        convertir("1" * 250, "GiB", "B", maquina="m", decimales=0)
