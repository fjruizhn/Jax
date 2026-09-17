"""Herramientas cuya salida se captura (Ejecutor Fase 2).

Existen porque hay datos verdaderos que ninguna salida imprime literal: los
conteos (tarea 8 de U3, «14 servicios con certificado SSL») y las conversiones
(tarea 2, «89 GiB» escrito de cabeza como «~91 GB»). La herramienta hace la
operación y deja su resultado en una captura, donde sí es citable.

DECISIÓN (2026-09-16): la línea de salida es EVIDENCIA, no interfaz. Va en un
formato de máquina neutro (`clave=valor`, sin palabras naturales), como la
salida de `ss`: traducirla rompería la cita. Estos tests fijan la gramática
con un lector escrito aquí, independiente del módulo.
"""
import json
import re

import pytest

from jax.ejecutor import herramientas as T
from jax.ejecutor.captura import CapturaCompleta, a_captura
from jax.ejecutor.cita import RESPALDADA, Afirmacion, Motivo, verificar
from jax.ejecutor.herramientas import (
    SUBCADENA, TERMINA_EN, HerramientaRechazada, contar, convertir,
)


def _origen(salida, truncada=False, codigo=0, maquina="atemai",
            comando="certbot certificates", motivos=()):
    return CapturaCompleta(
        maquina=maquina, comando=comando, codigo=codigo, salida=salida,
        stderr="", truncada=truncada, bytes_totales=len(salida.encode()),
        momento="2026-09-16T10:00:00+00:00", motivos_truncado=motivos)


# Gramática del docstring de herramientas.py, reescrita a mano: campos
# `clave=valor` separados por UN espacio; clave [a-z_]+; valor = literal JSON
# (cadena entre comillas con escapes ASCII, número, true/false/null o arreglo).
_CAMPO = re.compile(r'([a-z_]+)=("(?:[^"\\]|\\.)*"|\[[^\]]*\]|[^ "\[]+)')


def _leer(linea):
    """La línea como lista de (clave, valor). Falla si sobra o falta algo."""
    pares, pos = [], 0
    while pos < len(linea):
        m = _CAMPO.match(linea, pos)
        assert m, f"no es clave=valor en la posición {pos}: {linea!r}"
        pares.append((m.group(1), json.loads(m.group(2))))
        pos = m.end()
        if pos < len(linea):
            assert linea[pos] == " ", f"separador que no es un espacio en {pos}: {linea!r}"
            pos += 1
    return pares


def _linea(completa):
    [linea] = completa.salida.splitlines()
    return linea


def _citable(completa, dato):
    """El dato citado desde LA línea de la salida, contra el verificador real."""
    afirmacion = Afirmacion(maquina=completa.maquina, comando=completa.comando,
                            linea=_linea(completa), dato=dato)
    return verificar(afirmacion, [a_captura(completa)]).estado


def _sin_palabras_naturales(linea):
    """Fuera de los valores entre comillas sólo hay claves, números, códigos
    JSON y separadores: ninguna palabra que traducir."""
    sin_cadenas = re.sub(r'"(?:[^"\\]|\\.)*"', '""', linea)
    palabras = set(re.findall(r"[^\W\d_]+", sin_cadenas))
    claves = {c for c, _ in _leer(linea)}
    partes_de_claves = {p for c in claves for p in c.split("_")}
    return palabras <= partes_de_claves | {"true", "false", "null"}


# --- contar -----------------------------------------------------------------

_CATORCE_SSL = "\n".join(
    [f"  Certificate Name: sitio{i}.axioma-ia.io SSL" for i in range(14)]
    + ["  Expiry Date: 2026-12-01", "Found the following certs:"]) + "\n"

MODOS = [SUBCADENA, TERMINA_EN]


def test_contar_14_lineas_da_14_y_es_citable():
    c = contar(_origen(_CATORCE_SSL), "SSL", modo=TERMINA_EN)
    assert isinstance(c, CapturaCompleta)
    assert c.truncada is False
    assert c.codigo == 0
    assert c.maquina == "atemai"
    assert dict(_leer(_linea(c)))["count"] == 14
    assert _citable(c, "14") == RESPALDADA
    assert _citable(c, "16") == RESPALDADA  # el total también es un token entero


def test_la_salida_de_contar_es_exactamente_esta_linea():
    c = contar(_origen(_CATORCE_SSL), "SSL", modo=TERMINA_EN)
    assert c.salida == (
        'count=14 total=16 mode="termina_en" pattern="SSL" literal=true '
        'case_sensitive=true source_machine="atemai" source_command="certbot certificates"\n')


@pytest.mark.parametrize("modo", MODOS)
def test_el_comando_de_contar_es_de_maquina_y_nombra_modo_origen_y_momento(modo):
    c = contar(_origen(_CATORCE_SSL), "SSL", modo=modo)
    assert c.comando.startswith("contar ")
    assert _leer(c.comando.removeprefix("contar ")) == [
        ("mode", modo), ("pattern", "SSL"), ("literal", True), ("case_sensitive", True),
        ("source_machine", "atemai"), ("source_command", "certbot certificates"),
        ("source_moment", "2026-09-16T10:00:00+00:00")]
    assert _sin_palabras_naturales(c.comando.removeprefix("contar "))


@pytest.mark.parametrize("modo", MODOS)
def test_la_linea_de_contar_no_tiene_palabras_naturales(modo):
    assert _sin_palabras_naturales(_linea(contar(_origen(_CATORCE_SSL), "SSL", modo=modo)))


def test_contar_es_subcadena_literal_y_distingue_mayusculas():
    origen = _origen("a.b\naxb\nA.B\n")
    assert dict(_leer(_linea(contar(origen, "a.b", modo=SUBCADENA))))["count"] == 1


# Valores hostiles: espacios, comillas, `=`, un campo falso, barra invertida,
# saltos de línea, separador Unicode, override bidi y no ASCII.
_HOSTILES = [
    "a b", 'di"jo', "k=v", ' count=99 total=1', "c:\\ruta", "x\ny", "x\u2028y",
    "x\u202ey", "año", "'", "",
]


@pytest.mark.parametrize("hostil", _HOSTILES)
def test_los_valores_hostiles_se_escapan_y_se_recuperan_exactos(hostil):
    """Escape inequívoco: cada valor se lee de vuelta idéntico, un `count=99`
    metido en el comando de origen no fabrica otro campo, y la línea es ASCII
    imprimible (nada que un terminal o un navegador dibuje distinto)."""
    patron = hostil if hostil and hostil.splitlines() == [hostil] else "x"
    origen = _origen("x y\n", comando=f"echo {hostil}", maquina=f"m{hostil}")
    c = contar(origen, patron, modo=SUBCADENA)
    linea = _linea(c)
    campos = _leer(linea)
    assert [k for k, _ in campos] == ["count", "total", "mode", "pattern", "literal",
                                      "case_sensitive", "source_machine", "source_command"]
    d = dict(campos)
    assert (d["pattern"], d["source_machine"], d["source_command"]) == (
        patron, f"m{hostil}", f"echo {hostil}")
    assert linea.isascii() and linea.isprintable()
    assert c.comando.isascii() and c.comando.isprintable()
    assert dict(_leer(c.comando.removeprefix("contar ")))["source_command"] == f"echo {hostil}"


# Forma REAL de la tarea 8 de U3 (nombres de relleno, no del corpus): el nombre
# del archivo sale en el listado, en la cabecera `=== … ===` y en el error de
# `cat`. El criterio del modelo fue «los archivos `.ssl.conf`».
_LISTADO_Y_CATS = "\n".join(
    [f"sitio{i}.conf" for i in range(3)] + [f"sitio{i}.ssl.conf" for i in range(2)]
    + ["---"]
    + [l for i in range(2) for l in (
        f"=== sitio{i}.ssl.conf ===",
        f"cat: /etc/nginx/conf.d/domains/sitio{i}.ssl.conf: Permission denied", "")]
) + "\n"


def _cuenta(completa):
    d = dict(_leer(_linea(completa)))
    return d["count"], d["total"]


def test_termina_en_no_cuenta_la_cabecera_ni_el_error_de_cat():
    """Medido contra U3, tarea 8: por subcadena `.ssl.conf` da 42 de 112 y no
    14, porque el nombre aparece tres veces por archivo. Terminar en el sufijo
    separa el listado de la cabecera (`… ===`) y del error (`…: Permission
    denied`)."""
    origen = _origen(_LISTADO_Y_CATS, comando="ls domains/")
    assert _cuenta(contar(origen, ".ssl.conf", modo=SUBCADENA)) == (6, 12)
    assert _cuenta(contar(origen, ".ssl.conf", modo=TERMINA_EN)) == (2, 12)


def test_termina_en_es_literal_y_no_recorta_espacios():
    """Sin normalizar: una línea con un espacio al final no termina en el patrón."""
    origen = _origen("a.ssl.conf \nA.SSL.CONF\nb.ssl.conf\n")
    assert _cuenta(contar(origen, ".ssl.conf", modo=TERMINA_EN)) == (1, 3)


def test_contar_sin_modo_no_elige_uno_en_silencio():
    with pytest.raises(TypeError):
        contar(_origen(_CATORCE_SSL), "SSL")


@pytest.mark.parametrize("modo", ["", "regex", "TERMINA_EN", None, "empieza_en"])
def test_contar_con_modo_desconocido_no_cuenta(modo):
    with pytest.raises(HerramientaRechazada) as rechazo:
        contar(_origen(_CATORCE_SSL), "SSL", modo=modo)
    assert rechazo.value.motivo == Motivo(
        T.MODO_DESCONOCIDO, (("modo", repr(modo)), ("aceptados", (SUBCADENA, TERMINA_EN))))


@pytest.mark.parametrize("modo", MODOS)
def test_contar_sobre_captura_TRUNCADA_no_cuenta(modo):
    """Tarea 9 de U3: 2 KB leídos de 85,9 KB. Un conteo sobre una salida
    cortada es un número que nadie imprimió y que además es falso."""
    origen = _origen(_CATORCE_SSL, truncada=True, motivos=("tope_bytes",))
    with pytest.raises(HerramientaRechazada) as rechazo:
        contar(origen, "SSL", modo=modo)
    assert rechazo.value.motivo == Motivo(T.ORIGEN_TRUNCADO, (
        ("motivos", ("tope_bytes",)), ("comando", "certbot certificates"), ("maquina", "atemai")))


@pytest.mark.parametrize("modo", MODOS)
@pytest.mark.parametrize("codigo", [255, None])
def test_contar_sobre_comando_fallido_o_sin_codigo_no_cuenta(modo, codigo):
    """Un ssh caído deja stdout vacío: contar daría «0 servicios»."""
    with pytest.raises(HerramientaRechazada) as rechazo:
        contar(_origen("", codigo=codigo), "SSL", modo=modo)
    assert rechazo.value.motivo == Motivo(T.ORIGEN_CON_CODIGO_NO_CERO, (
        ("codigo", codigo), ("comando", "certbot certificates"), ("maquina", "atemai")))


@pytest.mark.parametrize("modo", MODOS)
def test_contar_con_patron_vacio_no_cuenta(modo):
    with pytest.raises(HerramientaRechazada) as rechazo:
        contar(_origen(_CATORCE_SSL), "", modo=modo)
    assert rechazo.value.motivo == Motivo(T.PATRON_VACIO)


@pytest.mark.parametrize("modo", MODOS)
def test_contar_con_patron_multilinea_no_cuenta(modo):
    with pytest.raises(HerramientaRechazada) as rechazo:
        contar(_origen(_CATORCE_SSL), "SSL\nExpiry", modo=modo)
    assert rechazo.value.motivo == Motivo(T.PATRON_MULTILINEA, (("patron", "SSL\nExpiry"),))


@pytest.mark.parametrize("modo", MODOS)
def test_el_patron_no_puede_partir_la_linea_citable(modo):
    origen = _origen("x y\n", comando="echo 'raro\nde verdad'")
    c = contar(origen, "x", modo=modo)
    assert len(c.salida.splitlines()) == 1


# --- convertir --------------------------------------------------------------

def test_89_GiB_son_95_6_GB_y_no_91():
    c = convertir(89, "GiB", "GB", maquina="atemai")
    assert isinstance(c, CapturaCompleta)
    assert dict(_leer(_linea(c)))["result"] == 95.6
    assert "91" not in c.salida
    assert _citable(c, "95.6") == RESPALDADA
    assert _citable(c, "89") == RESPALDADA


def test_la_salida_de_convertir_es_exactamente_esta_linea():
    c = convertir(89, "GiB", "GB", maquina="atemai")
    assert c.salida == (
        'value=89 from="GiB" to="GB" result=95.6 exact=95.563022336 '
        'factor_from=1073741824 factor_to=1000000000 rounding="half_up" decimals=1\n')
    assert _sin_palabras_naturales(_linea(c))


def test_el_comando_de_convertir_es_reproducible():
    c = convertir(89, "GiB", "GB", maquina="atemai")
    assert c.comando == 'convertir value=89 from="GiB" to="GB" rounding="half_up" decimals=1'
    assert c.maquina == "atemai"
    assert c.truncada is False and c.codigo == 0


def test_convertir_en_el_otro_sentido_y_decimales_explicitos():
    c = convertir("1863", "GB", "GiB", maquina="atemai", decimales=2)
    d = dict(_leer(_linea(c)))
    assert (d["result"], d["decimals"]) == (1735.05, 2)
    assert _citable(c, "1735.05") == RESPALDADA


def test_convertir_conserva_los_decimales_pedidos_en_el_resultado():
    """`decimals=2` escribe dos decimales aunque el segundo sea cero: el
    resultado dice a qué precisión se redondeó."""
    c = convertir("1250", "B", "KB", maquina="m", decimales=2)
    assert " result=1.25 " in c.salida
    c = convertir("1200", "B", "KB", maquina="m", decimales=2)
    assert " result=1.20 " in c.salida
    assert _citable(c, "1.20") == RESPALDADA


def test_convertir_redondea_mitad_hacia_arriba():
    # 1.25 KB -> 1.3 con mitad hacia arriba (no 1.2 del redondeo bancario)
    c = convertir("1250", "B", "KB", maquina="m")
    assert " result=1.3 " in c.salida
    assert _citable(c, "1.3") == RESPALDADA


@pytest.mark.parametrize("unidad", ["G", "gb", "Gb", "GiB ", "", "XB"])
def test_unidad_desconocida_es_error_nunca_un_valor_adivinado(unidad):
    for desde, hacia in ((unidad, "GB"), ("GiB", unidad)):
        with pytest.raises(HerramientaRechazada) as rechazo:
            convertir(89, desde, hacia, maquina="atemai")
        assert rechazo.value.motivo.codigo == T.UNIDAD_DESCONOCIDA
        assert dict(rechazo.value.motivo.datos)["unidad"] == repr(unidad)


@pytest.mark.parametrize("valor,codigo", [
    ("89,5", "valor_no_numerico"), ("abc", "valor_no_numerico"), (True, "valor_no_numerico"),
    (None, "valor_no_numerico"), ("nan", "valor_fuera_de_rango"),
    ("inf", "valor_fuera_de_rango"), (-1, "valor_fuera_de_rango"),
])
def test_valor_invalido_es_error(valor, codigo):
    with pytest.raises(HerramientaRechazada) as rechazo:
        convertir(valor, "GiB", "GB", maquina="atemai")
    assert rechazo.value.motivo == Motivo(codigo, (("valor", repr(valor)),))


def test_convertir_sin_maquina_es_error():
    with pytest.raises(HerramientaRechazada) as rechazo:
        convertir(89, "GiB", "GB", maquina=" ")
    assert rechazo.value.motivo == Motivo(T.SIN_MAQUINA)


def test_decimales_invalidos_es_error():
    with pytest.raises(HerramientaRechazada) as rechazo:
        convertir(89, "GiB", "GB", maquina="atemai", decimales=-1)
    assert rechazo.value.motivo == Motivo(T.DECIMALES_INVALIDOS, (("decimales", "-1"),))


def test_convertir_no_pierde_precision_con_valores_grandes():
    """El contexto por defecto de Decimal tiene 28 dígitos: con más, el
    cálculo revienta o redondea en silencio. El exacto tiene que ser exacto."""
    c = convertir("123456789012345678901234567.891", "GiB", "B", maquina="m", decimales=0)
    milesimas = 123456789012345678901234567891 * 1024 ** 3
    redondeado = (milesimas + 500) // 1000
    exacto = f"{milesimas // 1000}.{milesimas % 1000:03d}".rstrip("0").rstrip(".")
    assert f" result={redondeado} " in c.salida
    assert f" exact={exacto} " in c.salida
    assert _citable(c, str(redondeado)) == RESPALDADA


def test_convertir_que_no_cabe_en_la_precision_es_error_no_redondeo_silencioso():
    with pytest.raises(HerramientaRechazada) as rechazo:
        convertir("1" * 250, "GiB", "B", maquina="m", decimales=0)
    assert rechazo.value.motivo.codigo == T.PRECISION_EXCEDIDA


# --- Los rechazos tampoco escriben prosa ---

def test_el_mensaje_de_un_rechazo_es_su_codigo_y_sus_datos_en_formato_de_maquina():
    """`str(rechazo)` puede terminar en un registro: es el código seguido de
    los datos con la misma gramática `clave=valor` de las salidas."""
    with pytest.raises(HerramientaRechazada) as rechazo:
        contar(_origen(_CATORCE_SSL, truncada=True, motivos=("tope_bytes",)), "SSL", modo=SUBCADENA)
    mensaje = str(rechazo.value)
    assert mensaje == ('origen_truncado motivos=["tope_bytes"] '
                       'comando="certbot certificates" maquina="atemai"')
    codigo, resto = mensaje.split(" ", 1)
    assert codigo == T.ORIGEN_TRUNCADO
    assert _leer(resto) == [("motivos", ["tope_bytes"]), ("comando", "certbot certificates"),
                            ("maquina", "atemai")]
