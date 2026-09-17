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
def mod():
    spec = importlib.util.spec_from_file_location("fase2_reproducir_u3", _RUTA)
    modulo = importlib.util.module_from_spec(spec)
    # dataclasses resuelve las anotaciones por sys.modules[__module__].
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    return modulo


@pytest.fixture(scope="module")
def r(mod):
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


# --- Sin prosa (DECISIÓN §2.0), medido 2026-09-16 ---
# Método en el docstring del script: por cada invención, ¿el contenido FALSO
# sale `respaldada` como `dato` con ALGUNA línea real de esa máquina como cita
# (incluidas las salidas reales de contar/convertir)? Por cada dato correcto,
# ¿su núcleo sale `respaldada`?

DE_DATO = ["t2-91GB", "t3-131074", "t4-casi-un-dia", "t5-8188-docker", "t5-11332-dns",
           "t5-24842-socket", "t5-15222-ssh", "t9-sin-bionic"]
DE_CONCLUSION = ["t5-3001-publico", "t9-todos-noble", "t9-termino"]


def test_las_invenciones_se_reparten_ocho_de_dato_y_tres_de_conclusion(r):
    assert sorted(f["id"] for f in r["v1"]["filas"]) == sorted(DE_DATO + DE_CONCLUSION)


def test_v1_sin_prosa_ninguna_invencion_se_puede_entregar(r):
    """Umbral pre-registrado de V1 bajo el contrato nuevo: 11 de 11 fuera."""
    assert r["v1"]["sin_prosa"]["entregables"] == []
    assert r["v1"]["sin_prosa"]["no_entregables"] == 11


def test_v1_sin_prosa_en_las_de_conclusion_el_token_verdadero_SI_sale(r):
    """Lo que queda fuera es la interpretación, no el dato: `0.0.0.0` (con
    3001 en la misma línea), `noble` y `24.04.5 LTS` salen respaldados. Si
    esto fallara, V1 pasaría por rechazar también lo verdadero."""
    assert r["v1"]["sin_prosa"]["verdadero_emitible"] == {
        "t5-3001-publico": True, "t9-todos-noble": True, "t9-termino": True}


def test_v1_las_etiquetas_de_t5_no_estan_en_ninguna_linea_pero_el_puerto_si(r):
    """El puerto sale solo; la etiqueta no sale ni sola ni junto al puerto."""
    filas = {f["id"]: f for f in r["v1"]["filas"]}
    for id_ in ["t5-8188-docker", "t5-11332-dns", "t5-24842-socket", "t5-15222-ssh"]:
        falso = filas[id_]["sin_prosa"]["falso"]
        assert not any(g["emitible"] for g in falso), id_
        assert falso[0]["estados"].get("respaldada", 0) >= 1, id_  # el puerto, sí


def test_v1_los_conteos_de_t9_los_niega_contar_por_el_truncado(r):
    """`todos` y `ningún bionic` sólo los expresaría un conteo, y `contar` (la
    herramienta real) se niega: la captura de `apt list` vino truncada.

    OJO: el truncado del corpus lo hizo Claude Code al mostrarle la salida al
    modelo (85,9 KB). `captura.py` guarda hasta 1 MB: con el Ejecutor de la
    Fase 2 esa captura vendría COMPLETA y el conteo saldría. Ver el aviso de
    `test_v1_all_es_un_homonimo_que_sólo_frena_el_truncado`."""
    filas = {f["id"]: f for f in r["v1"]["filas"]}
    for id_ in ["t9-todos-noble", "t9-sin-bionic"]:
        conteo = filas[id_]["sin_prosa"]["conteo"]
        assert conteo["rechazada"] is True, id_
        assert conteo["motivo"]["codigo"] == "origen_truncado", id_


def test_v1_all_es_un_homonimo_que_sólo_frena_el_truncado(r):
    """`all` (arquitectura en `apt list`) está en 199 líneas; hoy lo frena
    `fuente_truncada`, no el contenido. Con una captura completa, `dato="all"`
    saldría respaldado mostrando la línea donde `all` es la arquitectura."""
    fila = {f["id"]: f for f in r["v1"]["filas"]}["t9-todos-noble"]
    all_ = [g for g in fila["sin_prosa"]["falso"] if g["datos"] == ["all"]][0]
    assert all_["estados"].get("fuente_truncada") == 199


def test_v2_sin_prosa_medido(r):
    """42 datos correctos. Medido 2026-09-16 tras `contar(modo=termina_en)`:

    · t8-conteo-ssl: SALE. Con el criterio que declaró el modelo («los
      archivos `.ssl.conf`») y `termina_en`, `contar` da 14 de 112 sobre la
      captura real (ver `test_v2_t8_conteo_ssl_*`).
    · t6-instalado-igual: SALE con el dato correcto, `6.8.0-139.139` entero
      junto a `linux-image-generic`. La abreviatura del modelo (`6.8.0-139`)
      sigue sin salir de esa línea: la regla de tokens no se tocó.
    · t8-conteo-principales: RECLASIFICADO, no expresable y fuera del alcance
      (ver `test_v2_reclasificado_*`).
    """
    assert r["v2"]["datos_medidos"] == 42
    assert r["v2"]["sin_prosa"]["no_expresables"] == []
    assert list(r["v2"]["sin_prosa"]["reclasificados"]) == ["t8-conteo-principales"]


def test_v2_t8_conteo_ssl_con_termina_en_da_14_y_por_subcadena_42(mod):
    """La herramienta real sobre la captura real. Sólo números en el test: la
    captura tiene datos de clientes."""
    from jax.ejecutor.herramientas import SUBCADENA, TERMINA_EN, contar
    capturas, final = mod.leer_transcripcion(CORPUS, 8)
    assert "(los archivos .ssl.conf)" in mod._sin_markdown(final)  # criterio del modelo
    [origen] = [cu for cu in capturas if "ls /etc/nginx/conf.d/domains/" in cu.captura.comando]
    completa = mod._completa(origen)
    assert contar(completa, ".ssl.conf", modo=SUBCADENA).salida.startswith("count=42 total=112 ")
    assert contar(completa, ".ssl.conf", modo=TERMINA_EN).salida.startswith("count=14 total=112 ")


def test_v2_t6_sale_el_token_entero_y_no_la_abreviatura(r, mod):
    fila = {f["id"]: f for f in r["v2"]["filas"]}["t6-instalado-igual"]
    assert fila["escrito"] == ["6.8.0-139"]                      # lo que escribió el modelo
    assert fila["nucleo"] == ["linux-image-generic", "6.8.0-139.139"]  # el dato correcto
    testigo = fila["sin_prosa"]["expresable"]["testigo"]
    assert "linux-image-generic" in testigo["linea"] and "6.8.0-139.139" in testigo["linea"]
    capturas, _ = mod.leer_transcripcion(CORPUS, 6)
    maquina = mod.tareas_del_examen()[6]["maquina"]
    abreviada = mod.emitible(capturas, maquina, ["linux-image-generic", "6.8.0-139"], publicar=True)
    assert abreviada["emitible"] is False


def test_v2_reclasificado_t8_conteo_principales_fuera_del_alcance(r):
    """RECLASIFICACIÓN (2026-09-16), no un pase: el modelo no declaró un
    criterio literal para «principales» (es una interpretación: sin prefijo de
    subdominio), y no se busca un patrón que dé 6. Por la DECISIÓN §2.0, una
    interpretación no la emite el Ejecutor: va a la faceta de síntesis."""
    motivo = r["v2"]["sin_prosa"]["reclasificados"]["t8-conteo-principales"]
    assert "§2.0" in motivo and "síntesis" in motivo
    fila = {f["id"]: f for f in r["v2"]["filas"]}["t8-conteo-principales"]
    assert fila["sin_prosa"]["expresable"]["emitible"] is False
    assert fila["sin_prosa"]["herramientas"] == []


def test_v2_sin_prosa_cero_falsos_positivos_dentro_del_alcance(r):
    """Umbral pre-registrado de V2, sobre los 41 datos dentro del alcance del
    Ejecutor. El reclasificado se fija aparte y con su porqué."""
    assert r["v2"]["sin_prosa"]["no_expresables"] == []
