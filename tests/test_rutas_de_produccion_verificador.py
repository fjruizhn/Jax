"""ops/rutas_de_produccion_verificador.py -- los 6 casos reales donde la
versión anterior en bash fallaba ABIERTO (auditoría de escalón 3, PR
jax#277, MAJOR-3), más el caso de clave duplicada, probados contra la
función real -- no contra el filesystem: `resolver`/`ejecutar` se inyectan
de mentira, así que estos tests corren en cualquier lado, sin sudo, sin
jaxsvc, sin /srv ni /home reales.

Los 6 casos del auditor:
  1. clave en alcance ausente
  2. valor vacío
  3. valor entre comillas (caso feliz Y caso ambiguo)
  4. espacios junto al `=`
  5. symlink que resuelve a /home/fruiz/jax
  6. JAX_WORKSPACE_DIR bajo /home con mensaje falso (ahora: excepción
     declarada, no un accidente de substring)
  + clave duplicada (2+ apariciones de la misma clave en alcance)

Corre con:
  python -m pytest tests/test_rutas_de_produccion_verificador.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from rutas_de_produccion_verificador import (  # noqa: E402
    EXCEPCIONES_FASE_A,
    KEYS_EN_ALCANCE,
    PERMITIDOS_EN_ALCANCE,
    ValorAmbiguo,
    normalizar_valor,
    parsear_env,
    verificar_fase_a,
    verificar_fase_b,
)

ENTORNO_SANO = {
    "JAX_CONFIG_PATH": ["/srv/jax-prod/jax/config/config.toml"],
    "JAX_AUDIT_LOG_PATH": ["/var/log/jax/las_manos/audit.jsonl"],
    "JAX_REPO_BASE": ["/srv/jax-data/repo"],
}


def _resolver_identidad(mapa: dict[str, str] | None = None):
    """resolver de mentira: por defecto, la ruta real == la ruta cruda.
    `mapa` permite simular un symlink: ruta cruda -> ruta real distinta."""
    mapa = mapa or {}

    def resolver(ruta: str) -> str | None:
        return mapa.get(ruta, ruta)
    return resolver


def _ejecutar_siempre_ok(argv):
    from subprocess import CompletedProcess
    return CompletedProcess(argv, 0, stdout="", stderr="")


def _ejecutar_siempre_falla(argv):
    from subprocess import CompletedProcess
    return CompletedProcess(argv, 1, stdout="", stderr="")


# --- parsear_env / normalizar_valor -----------------------------------------

def test_parsear_env_ignora_comentarios_y_lineas_en_blanco():
    texto = "# comentario\n\nJAX_CONFIG_PATH=/x\n"
    assert parsear_env(texto) == {"JAX_CONFIG_PATH": ["/x"]}


def test_parsear_env_detecta_clave_duplicada_por_longitud_de_lista():
    texto = "JAX_CONFIG_PATH=/x\nJAX_CONFIG_PATH=/y\n"
    assert parsear_env(texto)["JAX_CONFIG_PATH"] == ["/x", "/y"]


def test_normalizar_valor_sin_comillas_sin_espacios():
    assert normalizar_valor("/srv/jax-prod/jax") == "/srv/jax-prod/jax"


def test_normalizar_valor_entre_comillas_dobles():
    assert normalizar_valor('"/srv/jax-prod/jax"') == "/srv/jax-prod/jax"


def test_normalizar_valor_entre_comillas_simples():
    assert normalizar_valor("'/srv/jax-prod/jax'") == "/srv/jax-prod/jax"


def test_normalizar_valor_vacio_es_cadena_vacia_no_error():
    assert normalizar_valor("") == ""
    assert normalizar_valor('""') == ""


def test_normalizar_valor_comillas_a_medias_es_ambiguo():
    with pytest.raises(ValorAmbiguo):
        normalizar_valor('"/srv/jax-prod/jax')


def test_normalizar_valor_comillas_mezcladas_es_ambiguo():
    with pytest.raises(ValorAmbiguo):
        normalizar_valor("\"/srv/jax-prod/jax'")


def test_normalizar_valor_con_espacio_suelto_es_ambiguo():
    with pytest.raises(ValorAmbiguo):
        normalizar_valor(" /srv/jax-prod/jax")
    with pytest.raises(ValorAmbiguo):
        normalizar_valor("/srv/jax-prod/jax ")


# --- Caso 1: clave en alcance ausente ---------------------------------------

def test_caso_1_clave_en_alcance_ausente():
    entorno = dict(ENTORNO_SANO)
    del entorno["JAX_REPO_BASE"]
    resultado = verificar_fase_a(entorno, _resolver_identidad())
    assert not resultado.ok
    claves_con_problema = {h.clave for h in resultado.problemas}
    assert "JAX_REPO_BASE" in claves_con_problema
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_REPO_BASE")
    assert "ausente" in motivo


# --- Caso 2: valor vacío -----------------------------------------------------

def test_caso_2_valor_vacio():
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_AUDIT_LOG_PATH"] = [""]
    resultado = verificar_fase_a(entorno, _resolver_identidad())
    assert not resultado.ok
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_AUDIT_LOG_PATH")
    assert "vacío" in motivo


# --- Caso 3: comillas --------------------------------------------------------

def test_caso_3_comillas_valor_sano_pasa():
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_CONFIG_PATH"] = ['"/srv/jax-prod/jax/config/config.toml"']
    resultado = verificar_fase_a(entorno, _resolver_identidad())
    assert resultado.ok, resultado.problemas


def test_caso_3_comillas_ambiguas_falla():
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_CONFIG_PATH"] = ['"/srv/jax-prod/jax/config/config.toml']  # sin cerrar
    resultado = verificar_fase_a(entorno, _resolver_identidad())
    assert not resultado.ok
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_CONFIG_PATH")
    assert "ambigu" in motivo.lower() or "interpretable" in motivo.lower()


# --- Caso 4: espacios junto al "=" ------------------------------------------

def test_caso_4_espacios_junto_al_igual_la_clave_queda_ausente():
    """`JAX_CONFIG_PATH = /x` (espacio antes del "=") no matchea _LINEA:
    la clave nunca se registra en parsear_env, así que Fase A la ve como
    "ausente" -- FALLA, no un valor mal leído en silencio."""
    texto = (
        "JAX_CONFIG_PATH = /srv/jax-prod/jax/config/config.toml\n"
        "JAX_AUDIT_LOG_PATH=/var/log/jax/las_manos/audit.jsonl\n"
        "JAX_REPO_BASE=/srv/jax-data/repo\n"
    )
    entorno = parsear_env(texto)
    assert "JAX_CONFIG_PATH" not in entorno
    resultado = verificar_fase_a(entorno, _resolver_identidad())
    assert not resultado.ok
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_CONFIG_PATH")
    assert "ausente" in motivo


# --- Caso 5: symlink que resuelve a /home/fruiz/jax -------------------------

def test_caso_5_symlink_a_home_fruiz_jax_se_atrapa_por_la_ruta_real():
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_REPO_BASE"] = ["/srv/jax-data/repo-symlink"]
    resolver = _resolver_identidad({"/srv/jax-data/repo-symlink": "/home/fruiz/jax/repo"})
    resultado = verificar_fase_a(entorno, resolver)
    assert not resultado.ok
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_REPO_BASE")
    assert "/home/fruiz/jax/repo" in motivo


def test_caso_5_ruta_que_no_resuelve_ni_existe_es_falla():
    entorno = dict(ENTORNO_SANO)

    def resolver_none(ruta):
        return None if ruta == ENTORNO_SANO["JAX_REPO_BASE"][0] else ruta
    resultado = verificar_fase_a(entorno, resolver_none)
    assert not resultado.ok
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_REPO_BASE")
    assert "no se pudo resolver" in motivo


# --- Caso 6: JAX_WORKSPACE_DIR bajo /home -----------------------------------

def test_caso_6_workspace_dir_bajo_home_es_excepcion_declarada_no_pasa_por_accidente():
    assert "JAX_WORKSPACE_DIR" in EXCEPCIONES_FASE_A
    assert EXCEPCIONES_FASE_A["JAX_WORKSPACE_DIR"].strip()

    entorno = dict(ENTORNO_SANO)
    entorno["JAX_WORKSPACE_DIR"] = ["/home/fruiz/jax-workspace"]
    resultado = verificar_fase_a(entorno, _resolver_identidad())
    assert resultado.ok, resultado.problemas
    # Y si NO estuviera en la excepción, con el chequeo generico /home/*
    # amplio, SI se atraparia -- lo probamos quitandola de una copia local.
    excepciones_de_prueba = dict(EXCEPCIONES_FASE_A)
    del excepciones_de_prueba["JAX_WORKSPACE_DIR"]
    import rutas_de_produccion_verificador as mod
    original = mod.EXCEPCIONES_FASE_A
    try:
        mod.EXCEPCIONES_FASE_A = excepciones_de_prueba
        resultado_sin_excepcion = verificar_fase_a(entorno, _resolver_identidad())
    finally:
        mod.EXCEPCIONES_FASE_A = original
    assert not resultado_sin_excepcion.ok, (
        "sin la excepcion declarada, JAX_WORKSPACE_DIR bajo /home/ tendria "
        "que fallar -- si esto pasa, el chequeo generico /home/* no esta "
        "funcionando")


def test_cualquier_otra_ruta_bajo_home_sin_excepcion_falla():
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_UNA_CLAVE_NUEVA_DIR"] = ["/home/otrousuario/algo"]
    resultado = verificar_fase_a(entorno, _resolver_identidad())
    assert not resultado.ok
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_UNA_CLAVE_NUEVA_DIR")
    assert "/home/" in motivo


# --- Clave duplicada ---------------------------------------------------------

def test_clave_en_alcance_duplicada_falla():
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_CONFIG_PATH"] = [
        "/srv/jax-prod/jax/config/config.toml",
        "/srv/jax-prod/jax/config/otro.toml",
    ]
    resultado = verificar_fase_a(entorno, _resolver_identidad())
    assert not resultado.ok
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_CONFIG_PATH")
    assert "duplicada" in motivo


# --- Caso feliz --------------------------------------------------------------

def test_entorno_sano_pasa_fase_a():
    resultado = verificar_fase_a(ENTORNO_SANO, _resolver_identidad())
    assert resultado.ok, resultado.problemas
    assert resultado.revisadas == len(KEYS_EN_ALCANCE)


def test_las_tres_claves_en_alcance_estan_en_permitidos_por_construccion():
    """Guarda de coherencia: los propios valores de ENTORNO_SANO (usados en
    todos los tests de arriba como base) tienen que empezar con alguno de
    PERMITIDOS_EN_ALCANCE -- si no, los tests de arriba estarian probando
    contra un caso que ya es un falso positivo."""
    for clave in KEYS_EN_ALCANCE:
        valor = ENTORNO_SANO[clave][0]
        assert any(valor.startswith(p) for p in PERMITIDOS_EN_ALCANCE), (clave, valor)


# --- Fase B: jaxsvc puede leer/escribir --------------------------------------

def test_fase_b_reporta_si_jaxsvc_no_puede_leer():
    resultado = verificar_fase_b(ENTORNO_SANO, _ejecutar_siempre_falla)
    claves = {h.clave for h in resultado.problemas}
    assert claves == set(KEYS_EN_ALCANCE)


def test_fase_b_pasa_si_jaxsvc_puede_todo():
    resultado = verificar_fase_b(ENTORNO_SANO, _ejecutar_siempre_ok)
    assert not resultado.problemas


# --- verificar() de punta a punta, contra texto crudo ------------------------

def test_verificar_de_punta_a_punta_ok():
    texto = (
        "JAX_CONFIG_PATH=/srv/jax-prod/jax/config/config.toml\n"
        "JAX_AUDIT_LOG_PATH=/var/log/jax/las_manos/audit.jsonl\n"
        "JAX_REPO_BASE=/srv/jax-data/repo\n"
    )
    import rutas_de_produccion_verificador as mod
    original_resolver = mod.resolver_como_jaxsvc
    original_ejecutar = mod._ejecutar_sudo_real
    try:
        mod.resolver_como_jaxsvc = lambda ruta, ejecutar=None: ruta
        mod._ejecutar_sudo_real = _ejecutar_siempre_ok
        ok, reporte = mod.verificar(texto)
    finally:
        mod.resolver_como_jaxsvc = original_resolver
        mod._ejecutar_sudo_real = original_ejecutar
    assert ok, reporte


def test_verificar_de_punta_a_punta_rojo_contra_home_fruiz():
    """El caso real de HOY (2026-09-25) contra producción, reproducido con
    valores de mentira: las tres claves bajo /home/fruiz/jax."""
    texto = (
        "JAX_CONFIG_PATH=/home/fruiz/jax/config/config.toml\n"
        "JAX_AUDIT_LOG_PATH=/home/fruiz/jax/las_manos/logs/audit.jsonl\n"
        "JAX_REPO_BASE=/home/fruiz/jax/repo\n"
    )
    import rutas_de_produccion_verificador as mod
    original_resolver = mod.resolver_como_jaxsvc
    try:
        mod.resolver_como_jaxsvc = lambda ruta, ejecutar=None: ruta
        ok, reporte = mod.verificar(texto)
    finally:
        mod.resolver_como_jaxsvc = original_resolver
    assert not ok
    assert "JAX_CONFIG_PATH" in reporte
    assert "JAX_AUDIT_LOG_PATH" in reporte
    assert "JAX_REPO_BASE" in reporte
