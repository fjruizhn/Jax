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


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "HUECO de cita.verificar (hallado 2026-09-16): no mira `texto`. Las 11 "
    "invenciones salen `respaldada` citando una línea real de su propia captura. "
    "Arreglarlo es cambiar el verificador: decisión aparte."))
def test_ninguna_invencion_pasa_citando_una_linea_real(r):
    assert r["v1"]["pasan_con_cita_real_hoy"] == []
