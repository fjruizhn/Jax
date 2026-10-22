"""Detector de estructura (capa 2) y consecuencias (capa 4), sin DB ni red.

Qué se revisa lo decide el CANAL, no los verbos: una salida con schema declara
qué campos AFIRMAN y cuáles PROPONEN; un diff afirma con `--- a/X` y propone con
`--- /dev/null`; en texto libre solo cuentan backticks, forma de ruta,
`tabla.columna` con izquierda tabla real y `*.service|timer`. La prosa nunca es
candidata. Lo que el job escribió (propias) o marcó `nuevo:` es propuesta; un
candidato en una oración que termina en «?» es hipótesis.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from jax.anti_fabricacion import consecuencia, estructura
from jax.anti_fabricacion.consecuencia import Modo, aplicar, leer_modo
from jax.anti_fabricacion.estructura import (
    LIMPIO, MARCADO, NO_CORRIO, NO_VERIFICABLE, verificar,
)
from jax.anti_fabricacion.whitelist import Whitelist


def _wl(**kw) -> Whitelist:
    base = dict(
        tablas=frozenset({"jacobs_steps", "axioma_config"}),
        columnas=frozenset({"step_id", "status", "config_key"}),
        pares=frozenset({"jacobs_steps.step_id", "jacobs_steps.status", "axioma_config.config_key"}),
        archivos=frozenset({"jax/core/x.py", "las_manos/motor_registry/worker.py", "README.md"}),
        directorios=frozenset({"jax", "jax/core", "las_manos", "las_manos/motor_registry"}),
        simbolos=frozenset({"MAX_WRITE_BYTES", "authorize_and_execute_tool_call", "JobStore"}),
        sql=frozenset({"JSON_EXTRACT", "COALESCE", "SELECT", "InnoDB"}),
        rutas_app=frozenset({"/motor/dispatch", "/health"}),
        claves=frozenset({"jax_local", "kimi"}),
        modelos=frozenset({"qwen3.6"}),
        raices_fs=frozenset({"etc", "usr", "srv", "home", "var", "tmp"}),
    )
    base.update(kw)
    return Whitelist(**base)


def _v(salida, wl=None, corpus="", propias=(), **kw):
    return verificar(salida, wl or _wl(), corpus, propias, **kw)


def _patch(files_modified, diff="", description="d"):
    return json.dumps({"diff": diff, "files_modified": files_modified, "description": description})


# --- canal: salida con schema ------------------------------------------------------

def test_files_modified_con_archivo_inexistente_se_marca():
    v = _v(_patch(["services/bodegas.py"]), schema="code_patch.v1")
    assert v.estado == MARCADO
    assert [i.candidato for i in v.items] == ["services/bodegas.py"]
    assert v.items[0].clase == "sin_respaldo"


def test_files_modified_con_archivo_real_es_limpio():
    v = _v(_patch(["las_manos/motor_registry/worker.py"]), schema="code_patch.v1")
    assert v.estado == LIMPIO and not v.items


def test_campo_que_propone_no_se_revisa_pero_el_que_afirma_si():
    plan = json.dumps({"plan": "Crear `services/bodegas.py`", "steps": [], "patches": [],
                       "tests": [], "risk_notes": "tocar `services/otro.py`", "human_review_needed": False})
    assert _v(plan, schema="code_swarm.v1").estado == LIMPIO
    arq = json.dumps({"summary": "El módulo `services/bodegas.py` ya existe", "risks": [],
                      "recommendations": "agregar `services/nuevo.py`"})
    v = _v(arq, schema="architecture_review.v1")
    assert v.estado == MARCADO and [i.candidato for i in v.items] == ["services/bodegas.py"]


def test_bug_hunt_afirma_bugs_y_pasos_y_propone_el_arreglo():
    bug = json.dumps({"bugs_found": "falla en `jacobs_steps.columna_fantasma`", "severity": "high",
                      "reproduction_steps": "correr `las_manos/no_existe.py`", "suggested_fix": "editar `x/y/z.py`"})
    v = _v(bug, schema="bug_hunt.v1")
    assert {i.candidato for i in v.items} == {"jacobs_steps.columna_fantasma", "las_manos/no_existe.py"}


def test_cada_campo_del_mapa_de_canales_existe_en_su_schema():
    from jax.anti_fabricacion.estructura import CANALES_POR_SCHEMA  # noqa: F401  (misma fuente que usa el detector)
    import importlib
    ov = importlib.import_module("las_manos.motor_registry.output_validator")
    assert set(ov.CANALES_POR_SCHEMA) == {"code_swarm.v1", "code_patch.v1", "architecture_review.v1", "bug_hunt.v1"}
    for schema, canales in ov.CANALES_POR_SCHEMA.items():
        for canal, campos in canales.items():
            for campo in campos:
                assert campo in ov.SCHEMAS[schema], (schema, canal, campo)
    assert set(ov.CANALES_POR_SCHEMA) <= set(ov.SCHEMAS)


# --- canal: diff ---------------------------------------------------------------------

def test_diff_con_a_sobre_archivo_inexistente_se_marca():
    diff = "--- a/no/existe.py\n+++ b/no/existe.py\n@@ -1 +1 @@\n-x\n+y\n"
    v = _v(_patch([], diff=diff), schema="code_patch.v1")
    assert v.estado == MARCADO and v.items[0].candidato == "no/existe.py"


def test_diff_con_a_sobre_archivo_real_es_limpio():
    diff = "--- a/jax/core/x.py\n+++ b/jax/core/x.py\n@@ -1 +1 @@\n-x\n+y\n"
    assert _v(_patch([], diff=diff), schema="code_patch.v1").estado == LIMPIO


def test_diff_con_dev_null_propone_no_afirma():
    diff = "--- /dev/null\n+++ b/services/bodegas.py\n@@ -0,0 +1 @@\n+X = 1\n"
    v = _v(_patch([], diff=diff), schema="code_patch.v1")
    assert v.estado == LIMPIO and "services/bodegas.py" in v.propuestos


def test_diff_sin_schema_en_texto_libre_tambien_se_lee_como_diff():
    diff = "diff --git a/no/existe.py b/no/existe.py\n--- a/no/existe.py\n+++ b/no/existe.py\n@@ -1 +1 @@\n-x\n+y\n"
    assert _v(diff).estado == MARCADO


def test_lo_que_el_diff_define_es_propio():
    diff = ("--- /dev/null\n+++ b/n.py\n@@ -0,0 +1,3 @@\n+class NuevaClase:\n+    pass\n+def hacer_cosa():\n+    pass\n")
    v = _v(diff + "\nUsa `NuevaClase` y `hacer_cosa`.")
    assert v.estado == LIMPIO


# --- canal: texto libre ----------------------------------------------------------------

def test_prosa_nunca_es_candidata():
    assert _v("El servicio de bodegas llama a services y a authorize_and_execute_tool_call otra vez").estado == LIMPIO


def test_ruta_con_forma_de_ruta_fuera_de_backticks_si_cuenta():
    assert _v("Edité services/bodegas.py para arreglarlo").estado == MARCADO
    assert _v("Edité jax/core/x.py para arreglarlo").estado == LIMPIO


def test_fracciones_urls_y_pares_de_palabras_no_son_rutas():
    assert _v("lectura/escritura, TCP/IP, 3/4 y https://example.com/a/b.html").estado == LIMPIO


def test_tabla_columna_falsa_se_marca_y_la_real_no():
    assert _v("mira `jacobs_steps.columna_fantasma`").estado == MARCADO
    assert _v("mira `jacobs_steps.status`").estado == LIMPIO


def test_punto_con_izquierda_que_no_es_tabla_no_es_candidato():
    assert _v("usa `os.path.join` y `self.estado`").estado == LIMPIO


def test_constante_real_y_funcion_sql_real_son_limpias():
    assert _v("el tope es `MAX_WRITE_BYTES` y se lee con `JSON_EXTRACT`").estado == LIMPIO


def test_simbolo_inventado_en_backticks_se_marca():
    v = _v("implementa `AuditSink` Protocol")
    assert v.estado == MARCADO and v.items[0].candidato == "AuditSink"


def test_backticks_con_comandos_o_palabras_sueltas_no_se_revisan():
    assert _v("corre `git status --porcelain` y mira `true` y `foo`").estado == LIMPIO


def test_nombre_oculto_sin_directorio_no_es_candidato():
    """Hallazgo de calibración (U1, 2026-10-22): `.bak.conf` en una respuesta de
    infraestructura se marcaba como ruta inexistente. Un dotfile suelto es
    configuración de la máquina, no una afirmación sobre el repo."""
    wl = _wl(archivos=frozenset({"jax/core/x.py", "deploy/nginx.conf"}))
    assert _v("mira `.bak.conf` y `.env.conf`", wl=wl).estado == LIMPIO
    assert _v("mira `bodegas.conf`", wl=wl).estado == MARCADO


def test_ruta_absoluta_de_la_raiz_del_sistema_no_es_de_este_detector():
    assert _v("lee `/etc/jax/.env` y /srv/jax-data/x.json").estado == LIMPIO


def test_ruta_de_la_app_inventada_se_marca_y_la_real_no():
    assert _v("llama a `/services/bodegas`").estado == MARCADO
    assert _v("llama a `/motor/dispatch`").estado == LIMPIO


def test_fuente_opcional_caida_da_no_verificable_sin_acusar():
    v = _v("llama a `/services/bodegas`", wl=_wl(rutas_app=frozenset(), no_disponibles=frozenset({"rutas_app"})))
    assert v.estado == NO_VERIFICABLE and not v.items and "rutas_app" in v.no_verificables


def test_servicio_systemd_esta_fuera_de_v1_y_se_declara():
    v = _v("reinicia `jax-las-manos.service`")
    assert v.estado == LIMPIO and "systemd" in v.fuera_de_alcance


# --- propuesto, hipotesis, corpus confiable, heredado ---------------------------------

def test_marca_nuevo_en_backticks_es_propuesta():
    v = _v("crea `nuevo: services/bodegas.py` y `nuevo: BodegaRepo`")
    assert v.estado == LIMPIO and {"services/bodegas.py", "BodegaRepo"} <= set(v.propuestos)


def test_lo_que_el_job_escribio_es_propuesto():
    v = _v("modifiqué `services/bodegas.py`", propias=["services/bodegas.py"])
    assert v.estado == LIMPIO


def test_candidato_en_oracion_con_signo_de_pregunta_es_hipotesis():
    v = _v("Hay un error. ¿Existirá `services/bodegas.py` en el repo?")
    assert v.estado == LIMPIO and "services/bodegas.py" in v.hipotesis
    assert _v("Hay un error en `services/bodegas.py`. Otra cosa?").estado == MARCADO


def test_corpus_confiable_respalda_lo_que_la_persona_dijo():
    objetivo = "Revisa services/bodegas.py y la clase AuditSink"
    assert _v("miré `services/bodegas.py` y `AuditSink`", corpus=objetivo).estado == LIMPIO


def test_heredado_de_otro_modelo_no_es_limpio():
    previo = "El paso anterior propuso implementar `AuditSink` Protocol"
    v = _v("implementé `AuditSink`", previos=[previo])
    assert v.estado == MARCADO and v.items[0].clase == "heredado"
    solo_en_la_salida = _v("implementé `OtroInventado`", previos=[previo])
    assert solo_en_la_salida.items[0].clase == "sin_respaldo"


def test_la_salida_de_otro_modelo_nunca_respalda_aunque_el_corpus_la_cite():
    v = _v("usé `AuditSink`", corpus="objetivo sin nada", previos=["...AuditSink..."])
    assert v.estado == MARCADO


# --- forma del veredicto ----------------------------------------------------------------

def test_veredicto_se_serializa_a_json_con_estado_e_items():
    v = _v(_patch(["services/bodegas.py"]), schema="code_patch.v1")
    d = json.loads(json.dumps(v.como_dict()))
    assert d["estado"] == MARCADO and d["items"][0]["candidato"] == "services/bodegas.py"
    assert {"propuestos", "hipotesis", "no_verificables", "fuera_de_alcance"} <= set(d)


def test_no_corrio_es_su_propio_estado():
    v = estructura.no_corrio("detector_no_disponible")
    assert v.estado == NO_CORRIO and v.motivo == "detector_no_disponible"


# --- capa 4: aplicar --------------------------------------------------------------------

def _marcado():
    return _v(_patch(["services/bodegas.py"]), schema="code_patch.v1")


def test_aplicar_duro_con_marcado_falla_con_el_mensaje_de_la_spec():
    d = aplicar(_marcado(), Modo.DURO)
    assert d.fallar and d.error.startswith("estructura sin respaldo: ") and "services/bodegas.py" in d.error
    assert d.avisar


def test_aplicar_sombra_con_marcado_no_cambia_el_estado_pero_guarda_y_avisa():
    d = aplicar(_marcado(), Modo.SOMBRA)
    assert not d.fallar and d.error is None and d.avisar
    assert d.guardar["_estructura"]["estado"] == MARCADO and d.guardar["_unbacked_claims"]


@pytest.mark.parametrize("modo", [Modo.DURO, Modo.SOMBRA])
def test_aplicar_guarda_siempre_ambos_campos_y_no_acusa_si_no_hay_marca(modo):
    for v in (_v("nada que ver"), _v("llama a `/services/bodegas`", wl=_wl(rutas_app=frozenset(), no_disponibles=frozenset({"rutas_app"}))),
              estructura.no_corrio("x")):
        d = aplicar(v, modo)
        assert not d.fallar and not d.avisar
        assert "_estructura" in d.guardar and d.guardar["_unbacked_claims"] == []


# --- capa 4: modo desde axioma_config ---------------------------------------------------

class _Cur:
    def __init__(self, fila):
        self.fila = fila

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, sql, args=None):
        assert "axioma_config" in sql and "config_key" in sql
        self.args = args

    async def fetchone(self):
        return self.fila


class _Conn:
    def __init__(self, fila):
        self.fila = fila

    def cursor(self):
        return _Cur(self.fila)

    def close(self):
        pass


def _conectar(fila=None, falla=False):
    async def conectar():
        if falla:
            raise ConnectionError("db caida")
        return _Conn(fila)
    return conectar


@pytest.mark.parametrize("fila, esperado", [
    (("sombra",), Modo.SOMBRA), (("duro",), Modo.DURO), (("SOMBRA ",), Modo.SOMBRA),
    (("quizas",), Modo.DURO), ((" ",), Modo.DURO), (None, Modo.DURO),
])
def test_modo_lee_axioma_config_y_todo_lo_raro_es_duro(fila, esperado):
    assert asyncio.run(leer_modo(_conectar(fila))) == esperado


def test_modo_con_la_base_ilegible_es_duro():
    assert asyncio.run(leer_modo(_conectar(falla=True))) == Modo.DURO
