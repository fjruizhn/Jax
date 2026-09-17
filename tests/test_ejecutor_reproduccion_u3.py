"""V1 y V2 del spec de la Fase 2 (§6), contra el corpus real de U3.

Spec: docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md §6 y §8.
Método y por qué NO se inventa «la línea que citó»: docstring de
scripts/ejecutor_fase2/reproducir_u3.py.

Se saltea si el corpus no está en disco: vive fuera del repo porque tiene
datos de clientes. Aquí no hay ninguna línea de transcripción.

Los umbrales pre-registrados (V1: 11 de 11 atrapables; V2: 0 falsos positivos)
se escriben TAL CUAL y hoy NO se cumplen. Van con `xfail(strict=True)`: el
assert no se afloja, el rojo queda registrado con su razón, y si algún día
pasan sin que nadie lo decida, el strict lo hace fallar.
"""
import importlib.util
import os
import pathlib
import sys

import pytest

CORPUS = pathlib.Path(os.path.expanduser("~/ejecutor-fase0/resultados"))
pytestmark = pytest.mark.skipif(
    not (CORPUS / "calificacion_tres_capas.json").exists()
    or not (CORPUS / "examen" / "qwen").is_dir(),
    reason="corpus de U3 no está en disco")

_RUTA = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "ejecutor_fase2" / "reproducir_u3.py"


@pytest.fixture(scope="module")
def r():
    spec = importlib.util.spec_from_file_location("fase2_reproducir_u3", _RUTA)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resuelve las anotaciones por sys.modules[__module__].
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.reproducir(CORPUS)


# --- Lo medido, fijado: si el corpus o la calificación cambian, esto avisa ---

def test_la_calificacion_vigente_cuenta_once_y_las_cinco_limpias(r):
    assert r["v1"]["invenciones_totales"] == 11
    assert len(r["v1"]["filas"]) == 11
    assert r["v2"]["tareas_limpias"] == [1, 6, 7, 8, 10]


def test_v1_medido_ocho_atrapables_y_tres_con_linea_real(r):
    """Las tres NO atrapables tienen una línea real, completa, que contiene el
    dato tal como el modelo lo escribió: riesgo 2 del spec, medido."""
    assert r["v1"]["no_atrapables"] == ["t5-3001-publico", "t9-todos-noble", "t9-termino"]
    filas = {f["id"]: f for f in r["v1"]["filas"]}
    for id_ in r["v1"]["no_atrapables"]:
        assert any(not h["truncada"] for h in filas[id_]["lineas_que_lo_contienen"]), id_


def test_v2_medido_los_falsos_positivos(r):
    """Literal = el dato como lo escribió el modelo. Núcleo = el valor sin
    unidad ni formato. Los dos conteos derivados de la tarea 8 no tienen línea
    en ningún nivel."""
    assert r["v2"]["falsos_positivos_nucleo"] == ["t8-conteo-ssl", "t8-conteo-principales"]
    assert r["v2"]["falsos_positivos_literal"] == [
        "t1-total", "t1-usado", "t1-libre",
        "t6-so", "t6-compilado", "t6-sin-reinicio",
        "t7-libre", "t7-total", "t7-pct", "t7-uptime",
        "t8-conteo-ssl", "t8-conteo-principales",
        "t10-tamano", "t10-dueno", "t10-sin-grupo", "t10-sudo",
    ]


# --- Los umbrales pre-registrados, sin aflojar ---

@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "V1 NO PASA (medido 2026-09-16): 3 de 11 invenciones tienen una línea real "
    "completa que contiene el dato: t5-3001-publico (3001 y 0.0.0.0 en la misma "
    "línea de ss, 0.0.0.0 es la columna del par), t9-todos-noble y t9-termino "
    "(os-release dice noble y 24.04.5 LTS)."))
def test_v1_las_once_invenciones_son_atrapables_por_construccion(r):
    assert r["v1"]["no_atrapables"] == []


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "V2 NO PASA (medido 2026-09-16): 2 falsos positivos aun comparando sólo el "
    "núcleo (conteos bien derivados de la tarea 8), 16 comparando el dato como "
    "lo escribió el modelo (unidades, traducciones, paráfrasis)."))
def test_v2_cero_falsos_positivos_en_las_tareas_limpias(r):
    assert r["v2"]["falsos_positivos_nucleo"] == []
    assert r["v2"]["falsos_positivos_literal"] == []


# --- Regla ligada (dato + reglas 1-4 de cita.py), medida 2026-09-16 ---
# Método en el docstring del script: texto = el dato como lo escribió el modelo;
# cita = cualquier línea real completa de esa máquina; dato = literal (escrito,
# ancla o núcleo) o «cualquier dato» (testigo exhaustivo: un carácter común).

DE_DATO = ["t2-91GB", "t3-131074", "t4-casi-un-dia", "t5-8188-docker", "t5-11332-dns",
           "t5-24842-socket", "t5-15222-ssh", "t9-sin-bionic"]
DE_CONCLUSION = ["t5-3001-publico", "t9-todos-noble", "t9-termino"]


def test_las_invenciones_se_reparten_ocho_de_dato_y_tres_de_conclusion(r):
    assert sorted(f["id"] for f in r["v1"]["filas"]) == sorted(DE_DATO + DE_CONCLUSION)


def test_v1a_medido_con_la_regla_ligada(r):
    """Sólo los NÚMEROS inventados quedan rechazados sea cual sea la cita.

    · t2 (91 GB) y t3 (131,074): ninguna línea tiene esos números → regla 4.
    · t4 (casi un día) y t9-sin-bionic: prosa sin números. Caen si el dato es
      el literal inventado; pasan con un dato trivial (límite de reglas 2-3).
    · t5 ×4: la invención es la ETIQUETA; el puerto es real. Citando
      `0.0.0.0:8188` con dato `8188`, el texto «8188 Docker multi-hilo» pasa.
    """
    rl = r["v1"]["regla_ligada"]
    pasan_literal = set(rl["pasan_texto_minimo_dato_literal"])
    pasan_cualquiera = set(rl["pasan_texto_minimo_cualquier_dato"])
    assert [i for i in DE_DATO if i not in pasan_cualquiera] == ["t2-91GB", "t3-131074"]
    assert [i for i in DE_DATO if i not in pasan_literal] == [
        "t2-91GB", "t3-131074", "t4-casi-un-dia", "t9-sin-bionic"]


def test_v1b_las_tres_de_conclusion_SIGUEN_pasando(r):
    """Límite inherente de las citas (riesgo 2): van a C5, no al verificador."""
    rl = r["v1"]["regla_ligada"]
    for clave in ("pasan_texto_minimo_dato_literal", "pasan_texto_minimo_cualquier_dato"):
        assert set(DE_CONCLUSION) <= set(rl[clave]), clave


def test_v2_medido_con_la_regla_ligada(r):
    """42 datos correctos. El «2» anterior (sólo el valor) se corrige:

    · dato literal, texto mínimo: 5. Además de los dos conteos, t7-uptime (el
      texto dice «2 minutos», la línea `7:02`: regla 4), t10-sin-grupo y
      t10-sudo (el modelo escribió prosa sin el literal: regla 3).
    · cualquier dato, texto mínimo: 3 (el piso: conteos + t7-uptime).
    · con la línea entera de la respuesta como texto: 9 (números vecinos
      en la misma fila de tabla o párrafo).
    """
    rl = r["v2"]["regla_ligada"]
    assert r["v2"]["datos_medidos"] == 42
    assert rl["sin_respaldo_texto_minimo_dato_literal"] == [
        "t7-uptime", "t8-conteo-ssl", "t8-conteo-principales", "t10-sin-grupo", "t10-sudo"]
    assert rl["sin_respaldo_texto_minimo_cualquier_dato"] == [
        "t7-uptime", "t8-conteo-ssl", "t8-conteo-principales"]
    assert rl["sin_respaldo_linea_respuesta_dato_literal"] == [
        "t7-uptime", "t8-catch-all", "t8-444", "t8-conteo-ssl", "t8-conteo-principales",
        "t10-permisos", "t10-dueno", "t10-sin-grupo", "t10-sudo"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "V1a NO PASA con la regla ligada (medido 2026-09-16): sólo 2 de 8 invenciones "
    "de dato quedan rechazadas con cualquier cita (t2, t3: números). 4 de 8 si el "
    "dato es el literal inventado. Las etiquetas de t5 y la prosa de t4/t9 no "
    "tienen número que la regla 4 ate."))
def test_v1a_las_ocho_invenciones_de_dato_rechazadas_sea_cual_sea_la_cita(r):
    assert not set(DE_DATO) & set(r["v1"]["regla_ligada"]["pasan_texto_minimo_cualquier_dato"])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "V2 NO PASA con la regla ligada (medido 2026-09-16): 3 falsos positivos en el "
    "mejor caso (t7-uptime y los dos conteos de la tarea 8), 5 con el dato literal."))
def test_v2_cero_falsos_positivos_con_la_regla_ligada(r):
    assert r["v2"]["regla_ligada"]["sin_respaldo_texto_minimo_cualquier_dato"] == []
