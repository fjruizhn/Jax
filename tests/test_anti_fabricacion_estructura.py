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
    LIMPIO, MARCADO, NO_CORRIO, NO_VERIFICABLE, Previo, verificar,
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
    assert v.estado == LIMPIO and {"NuevaClase", "hacer_cosa"} <= set(v.propuestos)


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
    assert _v("el tope es `MAX_WRITE_BYTE_INVENTADO`").estado == LIMPIO  # fuera de alcance (regla 2)


def test_identificador_suelto_en_backticks_ya_no_es_candidato():
    """Enmienda 1, regla 2: solo 4 formas son candidatas. `AuditSink` o
    `calcular_total` (sin «/» ni «.») NO se revisan: límite declarado, esos casos
    de U4 quedan como «fuera de alcance»."""
    assert _v("implementa `AuditSink` Protocol y `calcular_total_bodegas`").estado == LIMPIO
    assert _v("`bodegas` es una tabla").estado == LIMPIO


def test_backticks_con_comandos_o_palabras_sueltas_no_se_revisan():
    assert _v("corre `git status --porcelain` y mira `true` y `foo`").estado == LIMPIO


def test_archivo_suelto_sin_barra_tampoco_es_candidato():
    wl = _wl(archivos=frozenset({"jax/core/x.py", "deploy/nginx.conf"}))
    assert _v("mira `bodegas.conf` y `worker.py`", wl=wl).estado == LIMPIO
    assert _v("mira `deploy/bodegas.conf`", wl=wl).estado == MARCADO


def test_nombre_oculto_sin_directorio_no_es_candidato():
    """Hallazgo de calibración (U1, 2026-10-22): `.bak.conf` en una respuesta de
    infraestructura se marcaba como ruta inexistente. Un dotfile suelto es
    configuración de la máquina, no una afirmación sobre el repo."""
    wl = _wl(archivos=frozenset({"jax/core/x.py", "deploy/nginx.conf"}))
    assert _v("mira `.bak.conf` y `.env.conf`", wl=wl).estado == LIMPIO


def test_ruta_absoluta_de_la_raiz_del_sistema_no_es_de_este_detector():
    assert _v("lee `/etc/jax/.env` y /srv/jax-data/x.json").estado == LIMPIO


def test_ruta_de_la_app_inventada_se_marca_y_la_real_no():
    assert _v("llama a `/services/bodegas`").estado == MARCADO
    assert _v("llama a `/motor/dispatch`").estado == LIMPIO


def test_fuente_opcional_caida_da_no_verificable_sin_acusar():
    v = _v("llama a `/services/bodegas`", wl=_wl(rutas_app=frozenset(), no_disponibles=frozenset({"rutas_app"})))
    assert v.estado == NO_VERIFICABLE and not v.items and "rutas_app" in v.no_verificables


def test_unidad_de_servicio_sin_fuente_es_no_verificable_nunca_limpia():
    """Enmienda 2, regla 1: en v1 no hay fuente de unidades en vivo. Una unidad que el repo
    versionado no tiene queda visible como no_verificable: ni limpia ni acusada."""
    for texto in ("reinicié `jax-las-manos.service`", "el temporizador `respaldo-diario.timer` corre",
                  "reinicié jax-las-manos.service ayer"):
        v = _v(texto)
        assert v.estado == NO_VERIFICABLE and "systemd" in v.no_verificables and not v.items, texto
        assert not aplicar(v, Modo.DURO).fallar
    arq = json.dumps({"summary": "El unit `jax-las-manos.service` ya arranca", "risks": [], "recommendations": "n/a"})
    assert _v(arq, schema="architecture_review.v1").estado == NO_VERIFICABLE


def test_unidad_que_el_repo_versionado_tiene_si_es_limpia():
    wl = _wl(archivos=frozenset({"jax/core/x.py", "ops/systemd/jax-las-manos.service", "ops/systemd/respaldo-diario.timer"}))
    assert _v("reinicié `jax-las-manos.service` y `respaldo-diario.timer`.", wl=wl).estado == LIMPIO
    assert _v("reinicié `otra-cosa.service`", wl=wl).estado == NO_VERIFICABLE


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
    objetivo = "Revisa services/bodegas.py y la tabla jacobs_steps"
    assert _v("miré `services/bodegas.py`", corpus=objetivo).estado == LIMPIO


def test_heredado_sin_veredicto_del_origen_es_marcado():
    previo = "El paso anterior creó `services/bodegas.py`"
    v = _v("modifiqué `services/bodegas.py`", previos=[previo])
    assert v.estado == MARCADO and v.items[0].clase == "heredado"
    solo_en_la_salida = _v("modifiqué `services/otro.py`", previos=[previo])
    assert solo_en_la_salida.items[0].clase == "sin_respaldo"


def test_heredado_cuyo_origen_tiene_veredicto_es_no_verificable_y_no_acusa():
    """Enmienda 1, regla 3: visible, nunca limpio, no falla en duro."""
    previo = Previo("El paso anterior creó `services/bodegas.py`", veredicto="limpio")
    v = _v("modifiqué `services/bodegas.py`", previos=[previo])
    assert v.estado == NO_VERIFICABLE and not v.items and "heredado" in v.no_verificables
    assert not aplicar(v, Modo.DURO).fallar


def test_la_salida_de_otro_modelo_nunca_respalda_aunque_el_corpus_la_cite():
    v = _v("usé `services/bodegas.py`", corpus="objetivo sin nada", previos=["...services/bodegas.py..."])
    assert v.estado == MARCADO


def test_ruta_no_encontrada_con_archivos_o_checkout_de_jax_caidos_es_no_verificable():
    for cat in ("archivos", "jax_checkout"):
        v = _v("edité `services/bodegas.py`", wl=_wl(no_disponibles=frozenset({cat})))
        assert v.estado == NO_VERIFICABLE and cat in v.no_verificables and not v.items


def test_veredicto_dice_si_cuenta_para_la_sombra():
    assert estructura.no_corrio("x").como_dict()["cuenta_para_sombra"] is False
    assert _v("nada").como_dict()["cuenta_para_sombra"] is True


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


def test_la_constante_de_exclusiones_es_una_sola_y_dice_lo_que_la_enmienda():
    from jax.anti_fabricacion import exclusiones, whitelist
    assert exclusiones.EXCLUSIONES == frozenset({".git", ".claude-flow", "node_modules", ".venv", "__pycache__"})
    assert whitelist.EXCLUSIONES is exclusiones.EXCLUSIONES


# --- Enmienda 2: mapa completo de canales, puntuación, `nuevo:` de línea, tabla.columna -------

def _campos():
    from motor_registry.output_validator import CANALES_POR_SCHEMA, SCHEMAS
    return CANALES_POR_SCHEMA, SCHEMAS


def test_todo_campo_de_todo_schema_esta_declarado_en_exactamente_un_canal():
    canales, schemas = _campos()
    assert set(canales) == set(schemas)
    for schema, campos in schemas.items():
        declarados = [c for canal in ("afirma", "propone", "diff") for c in canales[schema][canal]]
        assert sorted(declarados) == sorted(campos), (schema, set(campos) ^ set(declarados))


def _payload(schema, campo, texto):
    _c, schemas = _campos()
    d = {f: ([] if f in ("steps", "tests", "risks", "patches") else ("" if f != "human_review_needed" else False))
         for f in schemas[schema]}
    d[campo] = texto
    return json.dumps(d)


def _por_canal(canal):
    import importlib
    ov = importlib.import_module("las_manos.motor_registry.output_validator")
    return [(s, c) for s, m in ov.CANALES_POR_SCHEMA.items() for c in m[canal]]


def _rutas_de(schema):
    import importlib
    return importlib.import_module("las_manos.motor_registry.output_validator").CANALES_POR_SCHEMA[schema]["rutas"]


def test_files_modified_con_puntuacion_o_vacio():
    assert _v(_patch(["jax/core/x.py"]), schema="code_patch.v1").estado == LIMPIO
    v = _v(_patch(["services/bodegas.py,"]), schema="code_patch.v1")
    assert [i.candidato for i in v.items] == ["services/bodegas.py"]


def test_en_todo_campo_que_afirma_una_tabla_columna_falsa_y_una_ruta_falsa_se_marcan():
    for schema, campo in _por_canal("afirma"):
        if campo == "human_review_needed" or campo in _rutas_de(schema):
            continue   # booleano sin texto / lista de rutas (probada aparte)
        for texto in ("Falla al leer `jacobs_steps.columna_x`.", "Leí (jacobs_steps.columna_x), no existe.",
                      "Abrí las_manos/no_existe_jamas.py: falla."):
            v = _v(_payload(schema, campo, texto), schema=schema)
            assert v.estado == MARCADO, (schema, campo, texto)
        real = _v(_payload(schema, campo, "Leí `jacobs_steps.status` en jax/core/x.py."), schema=schema)
        assert real.estado == LIMPIO, (schema, campo)


def test_en_todo_campo_que_propone_lo_inexistente_no_se_revisa():
    for schema, campo in _por_canal("propone"):
        v = _v(_payload(schema, campo, "Habría que crear `a/b/nuevo.py` y leer `jacobs_steps.columna_x`."), schema=schema)
        assert v.estado == LIMPIO, (schema, campo)


def test_las_listas_de_los_campos_que_afirman_tambien_se_revisan():
    d = json.dumps({"plan": "", "steps": ["Edité x/y/falso.py", "Corrí la prueba"], "patches": [],
                    "tests": ["probé `jacobs_steps.nada`"], "risk_notes": "", "human_review_needed": False})
    v = _v(d, schema="code_swarm.v1")
    assert {i.candidato for i in v.items} == {"x/y/falso.py", "jacobs_steps.nada"}


@pytest.mark.parametrize("texto", [
    "Edité services/bodegas.py.", "Edité (services/bodegas.py).", "Edité services/bodegas.py: falla.",
    "Edité `services/bodegas.py`, luego paré.", 'Edité "services/bodegas.py";', "¿Edité services/bodegas.py!",
    "Edité services/bodegas.py...",
])
def test_la_puntuacion_final_se_recorta_antes_de_comparar(texto):
    v = _v(texto)
    assert [i.candidato for i in v.items] == ["services/bodegas.py"], texto


@pytest.mark.parametrize("texto", ["Edité jax/core/x.py.", "Edité (jax/core/x.py).", "Edité `jax/core/x.py`, y paré.",
                                   "Edité jax/core/x.py: bien.", "Lo que hay es jax/core/x.py;"])
def test_la_puntuacion_final_no_vuelve_inexistente_una_ruta_real(texto):
    assert _v(texto).estado == LIMPIO, texto


@pytest.mark.parametrize("texto", [
    "- nuevo: services/bodegas.py — repositorio de bodegas",
    "* Nuevo: `services/bodegas.py` (a crear)",
    "  • nuevo: services/bodegas.py.",
    "nuevo: services/bodegas.py",
    "Lista:\n- nuevo: services/bodegas.py — x\n- nuevo: services/otro.py — y",
])
def test_nuevo_como_prefijo_de_linea_o_vineta_vale_como_propuesta(texto):
    v = _v(texto)
    assert v.estado == LIMPIO and "services/bodegas.py" in v.propuestos, texto


def test_nuevo_a_mitad_de_linea_o_en_otra_linea_no_exime():
    assert _v("no es nuevo: services/bodegas.py").estado == MARCADO
    assert _v("- nuevo: services/a.py\n- services/b.py").estado == MARCADO


# --- Auditoría única sobre 2da939db -----------------------------------------------------------

def test_un_archivo_con_extension_conocida_nunca_es_tabla_columna():
    """MAJOR 2: `errors.py`, `provider.py`, `motor.py` tienen un nombre base igual a una tabla real."""
    wl = _wl(tablas=frozenset({"errors", "provider", "motor", "jacobs_steps"}))
    for texto in ("Edité errors.py", "Edité `provider.py`", "Leí motor.py.", "Mira `motor.md` y `errors.json`",
                  "Abrí provider.sql, motor.sh y motor.toml"):
        v = _v(texto, wl=wl)
        assert v.estado == LIMPIO and not v.items, (texto, [i.candidato for i in v.items])
    # una tabla.columna de verdad sigue marcándose
    assert _v("Leí `motor.columna_falsa`", wl=wl).estado == MARCADO


def test_un_nombre_sin_directorio_no_es_candidato_de_ruta():
    wl = _wl(archivos=frozenset({"jax/core/x.py"}))
    assert _v("Edité `inexistente.py` y bodegas.py", wl=wl).estado == LIMPIO


def test_una_columna_que_se_llama_como_una_extension_sigue_siendo_valida():
    wl = _wl(tablas=frozenset({"t"}), pares=frozenset({"t.json"}))
    assert _v("Leí `t.json`", wl=wl).estado == LIMPIO


def test_una_entrada_enorme_no_se_revisa_y_es_no_verificable():
    """MINOR 6: tope de bytes de entrada."""
    from jax.anti_fabricacion.estructura import MAX_BYTES_ENTRADA
    assert MAX_BYTES_ENTRADA == 256 * 1024
    v = _v("Edité services/bodegas.py. " * 20000)
    assert v.estado == NO_VERIFICABLE and "tamano" in v.no_verificables and not v.items


@pytest.mark.parametrize("patologico", [
    "a." * 120000, "a/" * 120000, "```x\n" * 50000, "`" * 250000, "a" * 250000, ":" * 250000,
    "a.service " * 25000, "http://" * 35000, "- nuevo: " * 25000, "(" * 250000, "jacobs_steps." * 18000,
], ids=["puntos", "barras", "vallas", "backticks", "letras", "dos_puntos", "unidades", "urls", "nuevo", "parentesis", "tabla"])
def test_ninguna_entrada_dentro_del_tope_tarda_mas_de_unos_segundos(patologico):
    """MINOR 6: sin backtracking catastrófico, medido."""
    import time
    t = time.monotonic()
    _v(patologico[:256 * 1024])
    assert time.monotonic() - t < 3.0


def test_el_tope_de_unbacked_claims_cuenta_el_total():
    """MINOR 7."""
    texto = " ".join(f"`falta/n{i}.py`" for i in range(150))
    v = _v(texto)
    d = aplicar(v, Modo.SOMBRA)
    assert len(d.guardar["_unbacked_claims"]) == 100 and d.guardar["_unbacked_claims_total"] == 150
    assert d.guardar["_estructura"]["items_total"] == 150


def test_el_veredicto_cuenta_cuantas_veces_se_uso_nuevo_y_la_pregunta_como_escape():
    """13: para ver en la fase sombra si un modelo aprende a abusarlos."""
    texto = ("- nuevo: a/uno.py — x\n- nuevo: a/dos.py\nCrea `nuevo: a/tres.py`.\n"
             "¿Existirá `a/cuatro.py`? ¿O `a/cinco.py`?\nHay `a/seis.py`.")
    v = _v(texto)
    d = v.como_dict()
    assert d["usos_nuevo"] == 3 and d["usos_hipotesis"] == 2, d
    assert [i.candidato for i in v.items] == ["a/seis.py"]
    assert _v("nada").como_dict()["usos_nuevo"] == 0


def test_el_aviso_y_el_audit_llevan_los_contadores():
    from jax.anti_fabricacion.consecuencia import mensaje_aviso
    v = _v("- nuevo: a/uno.py\nHay `a/seis.py`.")
    assert "nuevo" in mensaje_aviso(job_id="j", motor="m", capability="c", veredicto=v, modo=Modo.SOMBRA, pipeline_id=None)


def test_leer_modo_reporta_por_que_cayo_a_duro():
    motivos: list[str] = []
    assert asyncio.run(leer_modo(_conectar(None), motivos=motivos)) == Modo.DURO
    assert motivos and "falta" in motivos[0]
    m2: list[str] = []
    assert asyncio.run(leer_modo(_conectar(falla=True), motivos=m2)) == Modo.DURO and m2
    m3: list[str] = []
    assert asyncio.run(leer_modo(_conectar(("quizas",)), motivos=m3)) == Modo.DURO and m3
    m4: list[str] = []
    assert asyncio.run(leer_modo(_conectar(("sombra",)), motivos=m4)) == Modo.SOMBRA and not m4
    m5: list[str] = []
    assert asyncio.run(leer_modo(_conectar(("duro",)), motivos=m5)) == Modo.DURO and not m5
