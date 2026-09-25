"""ops/rutas_de_produccion_verificador.py -- las dos rondas de auditoría de
escalón 3 sobre PR jax#277.

**Ronda 1 (MAJOR-3):** los 6 casos reales donde la versión anterior en bash
fallaba ABIERTO -- clave en alcance ausente, valor vacío, valor entre
comillas, espacios junto al `=`, un symlink que resuelve a
`/home/fruiz/jax`, y `JAX_WORKSPACE_DIR` bajo `/home` con mensaje falso --
más el caso de clave duplicada.

**Ronda 2 (MAJOR-A):** el parser de la ronda 1 seguía fallando abierto --
`_LINEA.match` sobre la línea SIN recortar ignoraba en silencio una línea
indentada o con espacio junto al `=` (que systemd SÍ interpreta, y puede
pisar la línea correcta con una mala). `parsear_env` ahora reporta CUALQUIER
línea que no calce el patrón estricto como error de parseo, con su número
de línea -- y CUALQUIER error de parseo tira abajo `verificar()` entero, no
importa qué clave esté cerca. Además: Fase C (verdad efectiva contra
`/proc/<MainPID>/environ`), lista de permitidos POR CLAVE (ronda 1 tenía
una lista compartida entre las tres), y el chequeo genérico (claves fuera
de alcance) ya no se salta en silencio ante duplicadas/ambiguas/realpath
fallido.

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
    EXISTENCIA_OPCIONAL,
    KEYS_EN_ALCANCE,
    PERMITIDOS_POR_CLAVE,
    ValorAmbiguo,
    normalizar_valor,
    parsear_env,
    verificar_fase_a,
    verificar_fase_b,
    verificar_fase_c,
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
    entorno, errores = parsear_env(texto)
    assert entorno == {"JAX_CONFIG_PATH": ["/x"]}
    assert errores == []


def test_parsear_env_detecta_clave_duplicada_por_longitud_de_lista():
    texto = "JAX_CONFIG_PATH=/x\nJAX_CONFIG_PATH=/y\n"
    entorno, errores = parsear_env(texto)
    assert entorno["JAX_CONFIG_PATH"] == ["/x", "/y"]
    assert errores == []


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


# --- Ronda 2, MAJOR-A: líneas que systemd aplica y este módulo antes ignoraba

def test_linea_indentada_es_error_de_parseo_con_numero_de_linea():
    """El caso real del auditor: una línea indentada con la clave correcta
    ANTES no calzaba el regex y se descartaba en silencio -- si coexistía
    con una línea limpia y correcta, este módulo decía "todo bien" mientras
    systemd (que SÍ interpreta la indentada) podía terminar cargando la
    ruta mala."""
    texto = (
        "JAX_CONFIG_PATH=/srv/jax-prod/jax/config/config.toml\n"
        "  JAX_CONFIG_PATH=/home/fruiz/jax/config/config.toml\n"
    )
    entorno, errores = parsear_env(texto)
    assert len(errores) == 1
    assert "línea 2" in errores[0].clave
    assert "indentada" in errores[0].motivo or "=" in errores[0].motivo
    # Y ese error tiene que tirar abajo la verificación completa, no sólo
    # quedar reportado y ser ignorado por el llamador.
    resultado = verificar_fase_a(entorno, _resolver_identidad(), errores_de_parseo=errores)
    assert not resultado.ok
    assert any("línea 2" in h.clave for h in resultado.problemas)


def test_espacio_junto_al_igual_es_error_de_parseo_con_numero_de_linea():
    texto = (
        "JAX_AUDIT_LOG_PATH=/var/log/jax/las_manos/audit.jsonl\n"
        "JAX_AUDIT_LOG_PATH = /home/fruiz/jax/las_manos/logs/audit.jsonl\n"
    )
    entorno, errores = parsear_env(texto)
    assert len(errores) == 1
    assert "línea 2" in errores[0].clave
    resultado = verificar_fase_a(entorno, _resolver_identidad(), errores_de_parseo=errores)
    assert not resultado.ok


def test_linea_que_termina_en_barra_invertida_es_error_de_parseo():
    """Continuación de línea de systemd -- este módulo no la reproduce, así
    que la rechaza en vez de adivinar cómo se uniría con la siguiente.
    La línea 1 (que termina en "\\") y la línea 2 (el resto de la
    continuación, que sola no calza NOMBRE=valor) quedan las DOS marcadas --
    ninguna se cuela como si la continuación se hubiera unido sola."""
    texto = "JAX_CONFIG_PATH=/srv/jax-prod/jax/config/con\\\nfig.toml\n"
    entorno, errores = parsear_env(texto)
    assert len(errores) == 2
    assert "JAX_CONFIG_PATH" not in entorno
    motivo_linea_1 = next(h.motivo for h in errores if h.clave == "línea 1")
    assert "barra invertida" in motivo_linea_1 or "\\" in motivo_linea_1


def test_un_error_de_parseo_lejos_de_las_claves_en_alcance_igual_tira_todo_abajo():
    """Ronda 2: "no importa si está cerca de una clave en alcance". Una
    línea rota en una clave IRRELEVANTE (ni en alcance, ni excepción, ni
    ruta) igual invalida el resultado completo."""
    texto = (
        "JAX_CONFIG_PATH=/srv/jax-prod/jax/config/config.toml\n"
        "JAX_AUDIT_LOG_PATH=/var/log/jax/las_manos/audit.jsonl\n"
        "JAX_REPO_BASE=/srv/jax-data/repo\n"
        "  ALGUNA_OTRA_COSA=valor\n"
    )
    entorno, errores = parsear_env(texto)
    assert len(errores) == 1
    resultado = verificar_fase_a(entorno, _resolver_identidad(), errores_de_parseo=errores)
    assert not resultado.ok, "un error de parseo en una clave ajena tiene que igual invalidar todo"


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


# --- Caso 4: espacios junto al "=" (vía parsear_env, ver ronda 2 arriba) ----

def test_caso_4_espacios_junto_al_igual_via_parsear_env_es_error_de_parseo():
    texto = (
        "JAX_CONFIG_PATH = /srv/jax-prod/jax/config/config.toml\n"
        "JAX_AUDIT_LOG_PATH=/var/log/jax/las_manos/audit.jsonl\n"
        "JAX_REPO_BASE=/srv/jax-data/repo\n"
    )
    entorno, errores = parsear_env(texto)
    assert "JAX_CONFIG_PATH" not in entorno
    assert len(errores) == 1
    resultado = verificar_fase_a(entorno, _resolver_identidad(), errores_de_parseo=errores)
    assert not resultado.ok


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


def test_jax_kill_switch_path_ausente_pasa_pero_bajo_home_falla():
    """Hallazgo real de la ronda 2: JAX_KILL_SWITCH_PATH sólo existe cuando
    JAX está pausado -- `realpath -e` fallando (ruta ausente) es el estado
    SANO por defecto. Ronda 3, MINOR-6d: la excepción es MÁS ANGOSTA de lo
    que parecía -- exime sólo la existencia, no la ubicación. Si esta clave
    apuntara bajo /home/, seguiría siendo un problema real (el freno de
    emergencia no debería vivir en el home de un usuario)."""
    assert "JAX_KILL_SWITCH_PATH" in EXISTENCIA_OPCIONAL
    assert "JAX_KILL_SWITCH_PATH" not in EXCEPCIONES_FASE_A

    entorno_ausente = dict(ENTORNO_SANO)
    entorno_ausente["JAX_KILL_SWITCH_PATH"] = ["/etc/jax/interruptor/PAUSE"]

    def resolver_ausente(ruta):
        return None if ruta == "/etc/jax/interruptor/PAUSE" else ruta
    resultado = verificar_fase_a(entorno_ausente, resolver_ausente)
    assert resultado.ok, resultado.problemas

    entorno_bajo_home = dict(ENTORNO_SANO)
    entorno_bajo_home["JAX_KILL_SWITCH_PATH"] = ["/home/fruiz/PAUSE"]

    def resolver_ausente_bajo_home(ruta):
        return None if ruta == "/home/fruiz/PAUSE" else ruta
    resultado_home = verificar_fase_a(entorno_bajo_home, resolver_ausente_bajo_home)
    assert not resultado_home.ok, "JAX_KILL_SWITCH_PATH bajo /home/ (aunque ausente) tendria que fallar"
    motivo = next(h.motivo for h in resultado_home.problemas if h.clave == "JAX_KILL_SWITCH_PATH")
    assert "/home/" in motivo


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


def test_clave_fuera_de_alcance_duplicada_tambien_falla():
    """Ronda 2, MINOR: antes el chequeo genérico se saltaba en silencio
    (`continue`) ante una clave duplicada que no fuera una de las tres en
    alcance. Ahora también es FALLA."""
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_OTRA_RUTA_DIR"] = ["/srv/algo", "/srv/otro"]
    resultado = verificar_fase_a(entorno, _resolver_identidad())
    assert not resultado.ok
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_OTRA_RUTA_DIR")
    assert "duplicada" in motivo


def test_clave_fuera_de_alcance_ambigua_tambien_falla():
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_OTRA_RUTA_DIR"] = ['"sin cerrar']
    resultado = verificar_fase_a(entorno, _resolver_identidad())
    assert not resultado.ok
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_OTRA_RUTA_DIR")
    assert "ambigu" in motivo.lower() or "interpretable" in motivo.lower()


def test_clave_fuera_de_alcance_con_realpath_fallido_tambien_falla():
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_OTRA_RUTA_DIR"] = ["/srv/no-existe"]

    def resolver_none(ruta):
        return None if ruta == "/srv/no-existe" else ruta
    resultado = verificar_fase_a(entorno, resolver_none)
    assert not resultado.ok
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_OTRA_RUTA_DIR")
    assert "no se pudo resolver" in motivo


# --- Ronda 2, MINOR: lista de permitidos POR CLAVE --------------------------

def test_permitidos_por_clave_es_especifico_no_compartido():
    """JAX_CONFIG_PATH apuntando al destino de JAX_REPO_BASE (o viceversa)
    tiene que FALLAR -- antes (ronda 1) una lista compartida lo habría
    dejado pasar."""
    assert PERMITIDOS_POR_CLAVE["JAX_CONFIG_PATH"] != PERMITIDOS_POR_CLAVE["JAX_REPO_BASE"]
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_CONFIG_PATH"] = ["/srv/jax-data/repo/config.toml"]  # destino de REPO_BASE, no el suyo
    resultado = verificar_fase_a(entorno, _resolver_identidad())
    assert not resultado.ok
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_CONFIG_PATH")
    assert "no está bajo ninguno" in motivo


def test_repo_base_no_confunde_con_directorio_hermano_de_nombre_parecido():
    """`_bajo_prefijo`: /srv/jax-data/repo-otro-nombre NO es
    /srv/jax-data/repo ni algo debajo -- un startswith pelado lo habria
    dejado pasar."""
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_REPO_BASE"] = ["/srv/jax-data/repo-otro-nombre"]
    resultado = verificar_fase_a(entorno, _resolver_identidad())
    assert not resultado.ok


# --- Caso feliz --------------------------------------------------------------

def test_entorno_sano_pasa_fase_a():
    resultado = verificar_fase_a(ENTORNO_SANO, _resolver_identidad())
    assert resultado.ok, resultado.problemas
    assert resultado.revisadas == len(KEYS_EN_ALCANCE)


def test_las_tres_claves_en_alcance_estan_en_permitidos_por_construccion():
    for clave in KEYS_EN_ALCANCE:
        valor = ENTORNO_SANO[clave][0]
        assert any(valor.startswith(p) for p in PERMITIDOS_POR_CLAVE[clave]), (clave, valor)


# --- Fase B: jaxsvc puede leer/escribir --------------------------------------

def test_fase_b_reporta_si_jaxsvc_no_puede_leer():
    resultado = verificar_fase_b(ENTORNO_SANO, _ejecutar_siempre_falla)
    claves = {h.clave for h in resultado.problemas}
    assert claves == set(KEYS_EN_ALCANCE)


def test_fase_b_pasa_si_jaxsvc_puede_todo():
    resultado = verificar_fase_b(ENTORNO_SANO, _ejecutar_siempre_ok)
    assert not resultado.problemas


# --- Fase C: verdad efectiva contra /proc/<MainPID>/environ -----------------

def test_fase_c_sin_ningun_servicio_revisado_es_problema():
    """Ronda 3, MINOR-1: antes, con los DOS servicios sin PID,
    `resultado.problemas` quedaba vacío -- "todo bien" por accidente, sin
    haber verificado NADA. Ahora es un problema explícito."""
    resultado = verificar_fase_c(
        _resolver_identidad(),
        obtener_pid=lambda servicio: None,
    )
    assert resultado.problemas, "sin ningun servicio revisado, Fase C tiene que fallar, no dar verde por vacio"
    assert set(resultado.servicios_sin_pid) == {"jax-las-manos", "jax-platform"}
    assert resultado.servicios_revisados == []


def test_fase_c_un_servicio_sin_pid_pero_el_otro_sano_no_es_problema_por_eso_solo():
    """Con AL MENOS UN servicio revisado de verdad, que el otro esté
    inactivo es un aviso (servicios_sin_pid), no una falla por sí sola --
    distinto del caso "los dos sin PID" de arriba."""
    environ_sano = {clave: valores[0] for clave, valores in ENTORNO_SANO.items()}
    resultado = verificar_fase_c(
        _resolver_identidad(),
        obtener_pid=lambda servicio: "123" if servicio == "jax-las-manos" else None,
        leer_environ=lambda pid: dict(environ_sano),
    )
    assert not resultado.problemas, resultado.problemas
    assert resultado.servicios_revisados == ["jax-las-manos"]
    assert resultado.servicios_sin_pid == ["jax-platform"]


def test_fase_c_entorno_vivo_sano_pasa():
    environ_sano = {clave: valores[0] for clave, valores in ENTORNO_SANO.items()}
    resultado = verificar_fase_c(
        _resolver_identidad(),
        obtener_pid=lambda servicio: "12345",
        leer_environ=lambda pid: dict(environ_sano),
    )
    assert not resultado.problemas
    assert set(resultado.servicios_revisados) == {"jax-las-manos", "jax-platform"}


def test_fase_c_entorno_vivo_malo_falla_por_servicio():
    environ_malo = {
        "JAX_CONFIG_PATH": "/home/fruiz/jax/config/config.toml",
        "JAX_AUDIT_LOG_PATH": "/home/fruiz/jax/las_manos/logs/audit.jsonl",
        "JAX_REPO_BASE": "/home/fruiz/jax/repo",
    }
    resultado = verificar_fase_c(
        _resolver_identidad(),
        obtener_pid=lambda servicio: "12345",
        leer_environ=lambda pid: dict(environ_malo),
    )
    assert len(resultado.problemas) == 2 * len(KEYS_EN_ALCANCE)  # 2 servicios x 3 claves
    for servicio in ("jax-las-manos", "jax-platform"):
        for clave in KEYS_EN_ALCANCE:
            assert any(h.clave == f"{servicio}:{clave}" for h in resultado.problemas)


def test_fase_c_clave_faltante_en_entorno_vivo_es_problema():
    environ_incompleto = {"JAX_CONFIG_PATH": ENTORNO_SANO["JAX_CONFIG_PATH"][0]}
    resultado = verificar_fase_c(
        _resolver_identidad(),
        obtener_pid=lambda servicio: "1",
        leer_environ=lambda pid: dict(environ_incompleto),
    )
    faltantes = [h for h in resultado.problemas if "no está en el entorno vivo" in h.motivo]
    assert len(faltantes) == 2 * 2  # 2 servicios x 2 claves faltantes (AUDIT_LOG_PATH, REPO_BASE)


def test_fase_c_no_puede_leer_proc_environ_es_problema():
    resultado = verificar_fase_c(
        _resolver_identidad(),
        obtener_pid=lambda servicio: "1",
        leer_environ=lambda pid: None,
    )
    assert len(resultado.problemas) == 2
    assert all("no se pudo leer" in h.motivo for h in resultado.problemas)


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
    original_fase_c = mod.verificar_fase_c
    try:
        mod.resolver_como_jaxsvc = lambda ruta, ejecutar=None: ruta
        mod._ejecutar_sudo_real = _ejecutar_siempre_ok
        mod.verificar_fase_c = lambda resolver: mod.ResultadoFaseC()
        ok, reporte = mod.verificar(texto)
    finally:
        mod.resolver_como_jaxsvc = original_resolver
        mod._ejecutar_sudo_real = original_ejecutar
        mod.verificar_fase_c = original_fase_c
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
    original_fase_c = mod.verificar_fase_c
    try:
        mod.resolver_como_jaxsvc = lambda ruta, ejecutar=None: ruta
        mod.verificar_fase_c = lambda resolver: mod.ResultadoFaseC()
        ok, reporte = mod.verificar(texto)
    finally:
        mod.resolver_como_jaxsvc = original_resolver
        mod.verificar_fase_c = original_fase_c
    assert not ok
    assert "JAX_CONFIG_PATH" in reporte
    assert "JAX_AUDIT_LOG_PATH" in reporte
    assert "JAX_REPO_BASE" in reporte


def test_verificar_de_punta_a_punta_un_error_de_parseo_tira_todo_abajo():
    """Ronda 2: aunque las tres claves en alcance estén perfectas, un error
    de parseo en cualquier otra línea tiene que dar rojo -- confianza en el
    archivo completo, no sólo en las líneas que a este módulo le interesan."""
    texto = (
        "JAX_CONFIG_PATH=/srv/jax-prod/jax/config/config.toml\n"
        "JAX_AUDIT_LOG_PATH=/var/log/jax/las_manos/audit.jsonl\n"
        "JAX_REPO_BASE=/srv/jax-data/repo\n"
        "  ALGO_INDENTADO=valor\n"
    )
    import rutas_de_produccion_verificador as mod
    original_resolver = mod.resolver_como_jaxsvc
    original_fase_c = mod.verificar_fase_c
    try:
        mod.resolver_como_jaxsvc = lambda ruta, ejecutar=None: ruta
        mod.verificar_fase_c = lambda resolver: mod.ResultadoFaseC()
        ok, reporte = mod.verificar(texto)
    finally:
        mod.resolver_como_jaxsvc = original_resolver
        mod.verificar_fase_c = original_fase_c
    assert not ok
    assert "línea 4" in reporte


# --- Ronda 3: DEFECTO real -- jaxsvc_puede_leer/escribir usaban `test -- `,
# que `test` no soporta (a diferencia de `realpath`, que sí). Los tests de
# arriba de Fase B sólo ejercitaban un `ejecutar` de mentira -- nunca el
# comando real -- así que nadie lo vio hasta correr `--verificar` de verdad
# contra producción (Paso 8.4 del runbook). Estos SÍ corren el comando real.
# ---------------------------------------------------------------------------

RAIZ_PRODUCCION = "/srv/jax-prod/jax"


def _motivo_de_skip_fuera_de_produccion() -> str | None:
    if not Path(RAIZ_PRODUCCION).is_dir():
        return f"esta máquina no tiene {RAIZ_PRODUCCION} -- no es el host de producción de jax"
    return None


def test_jaxsvc_puede_leer_contra_el_comando_real_no_el_mock():
    """El caso que habría atrapado el DEFECTO de ronda 3: SIN mock,
    contra una ruta que jaxsvc SÍ puede leer de verdad
    (config.toml del checkout de despliegue, jaxsvc:jaxsvc 664). Con el
    `--` viejo, `sudo -u jaxsvc test -r -- <ruta>` daba rc=2 (error de uso
    de `test`, no una respuesta real) y esta función devolvía `False`
    incondicionalmente -- este test habría fallado con el código viejo."""
    motivo = _motivo_de_skip_fuera_de_produccion()
    if motivo:
        pytest.skip(motivo)
    from rutas_de_produccion_verificador import jaxsvc_puede_leer
    assert jaxsvc_puede_leer("/srv/jax-prod/jax/config/config.toml") is True


def test_jaxsvc_puede_escribir_contra_el_comando_real_no_el_mock():
    motivo = _motivo_de_skip_fuera_de_produccion()
    if motivo:
        pytest.skip(motivo)
    from rutas_de_produccion_verificador import jaxsvc_puede_escribir
    # /srv/jax-data/repo/documents: jaxsvc escribe ahí de verdad
    # (jacobs/executor.py._persist_step_to_repo) -- si esto no existe,
    # el corte de producción de esta tarea no llegó a correr todavía.
    ruta = "/srv/jax-data/repo/documents"
    if not Path(ruta).is_dir():
        pytest.skip(f"{ruta} no existe -- el corte de producción no corrió")
    assert jaxsvc_puede_escribir(ruta) is True


def test_jaxsvc_puede_leer_rechaza_ruta_no_absoluta_sin_necesitar_produccion():
    """La defensa que reemplaza a `--`: sin necesitar sudo ni jaxsvc, una
    ruta relativa (que nunca debería llegar acá, pero por si acaso) se
    rechaza ANTES de construir el comando de `test`."""
    from rutas_de_produccion_verificador import jaxsvc_puede_leer
    with pytest.raises(ValueError):
        jaxsvc_puede_leer("ruta/relativa")


def test_jaxsvc_puede_leer_da_false_para_ruta_absoluta_inexistente():
    """Sin mock: una ruta absoluta que no existe da False de verdad (no un
    error de uso de `test` disfrazado de False)."""
    motivo = _motivo_de_skip_fuera_de_produccion()
    if motivo:
        pytest.skip(motivo)
    from rutas_de_produccion_verificador import jaxsvc_puede_leer
    assert jaxsvc_puede_leer("/no/existe/de/verdad/2026-09-25") is False


# --- Ronda 4, MINOR-6: mock puro (corre en CUALQUIER CI, sin sudo/jaxsvc) --
# que exige el argv EXACTO -- mutación de control: reintroducir "--" en el
# código tiene que hacer fallar este test. Probado a mano (ver el informe):
# con "--" reintroducido, `capturado["argv"]` trae un elemento de más y el
# `==` estricto falla; sin él, pasa.
# ---------------------------------------------------------------------------

def test_jaxsvc_puede_leer_arma_argv_exacto_sin_guion_guion():
    capturado = {}

    def ejecutar_capturando(argv):
        capturado["argv"] = argv
        from subprocess import CompletedProcess
        return CompletedProcess(argv, 0, stdout="", stderr="")

    from rutas_de_produccion_verificador import jaxsvc_puede_leer
    jaxsvc_puede_leer("/algun/archivo", ejecutar_capturando)
    assert capturado["argv"] == ["sudo", "-n", "-u", "jaxsvc", "test", "-r", "/algun/archivo"]


def test_jaxsvc_puede_escribir_arma_argv_exacto_sin_guion_guion():
    capturado = {}

    def ejecutar_capturando(argv):
        capturado["argv"] = argv
        from subprocess import CompletedProcess
        return CompletedProcess(argv, 0, stdout="", stderr="")

    from rutas_de_produccion_verificador import jaxsvc_puede_escribir
    jaxsvc_puede_escribir("/algun/archivo", ejecutar_capturando)
    assert capturado["argv"] == ["sudo", "-n", "-u", "jaxsvc", "test", "-w", "/algun/archivo"]


# --- Ronda 4, MINOR-1: JAX_KILL_SWITCH_PATH con realpath -m, no con el
# valor crudo -- travesía, symlink de directorio, y ruta relativa.
# ---------------------------------------------------------------------------

def test_kill_switch_con_travesia_hacia_home_falla():
    """`/etc/jax/../../home/fruiz/PAUSE` -- el string crudo NO empieza con
    "/home", pero resuelto (realpath -m) SÍ termina ahí."""
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_KILL_SWITCH_PATH"] = ["/etc/jax/../../home/fruiz/PAUSE"]

    def resolver_m(ruta):
        return "/home/fruiz/PAUSE" if ruta == "/etc/jax/../../home/fruiz/PAUSE" else ruta
    resultado = verificar_fase_a(
        entorno, _resolver_identidad(), resolver_permitiendo_ausente=resolver_m)
    assert not resultado.ok
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_KILL_SWITCH_PATH")
    assert "/home/" in motivo


def test_kill_switch_symlink_de_directorio_hacia_home_falla():
    """Un symlink de directorio (no del archivo final) que lleva a /home --
    realpath -m sigue symlinks de los componentes intermedios."""
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_KILL_SWITCH_PATH"] = ["/etc/jax/interruptor-symlink/PAUSE"]

    def resolver_m(ruta):
        return "/home/fruiz/interruptor-real/PAUSE" if ruta == "/etc/jax/interruptor-symlink/PAUSE" else ruta
    resultado = verificar_fase_a(
        entorno, _resolver_identidad(), resolver_permitiendo_ausente=resolver_m)
    assert not resultado.ok


def test_kill_switch_ruta_relativa_falla_con_hallazgo():
    """PAUSE relativa -- MINOR-2 la rechaza en _valor_unico_normalizado
    ANTES de llegar a ningún resolver."""
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_KILL_SWITCH_PATH"] = ["PAUSE"]
    resultado = verificar_fase_a(entorno, _resolver_identidad())
    assert not resultado.ok
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_KILL_SWITCH_PATH")
    assert "absoluta" in motivo


def test_kill_switch_ausente_pero_absoluto_sigue_pasando():
    """Caso sano intacto: ausente (realpath -m devuelve la misma ruta
    porque nada la desvía) sigue pasando."""
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_KILL_SWITCH_PATH"] = ["/etc/jax/interruptor/PAUSE"]

    def resolver_m(ruta):
        return ruta  # nada la desvía, ausente y sin travesía
    resultado = verificar_fase_a(
        entorno, _resolver_identidad(), resolver_permitiendo_ausente=resolver_m)
    assert resultado.ok, resultado.problemas


# --- Ronda 4, MINOR-2: ruta no absoluta -> Hallazgo, no traceback --------

def test_clave_en_alcance_no_absoluta_da_hallazgo_no_traceback():
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_CONFIG_PATH"] = ["config/relativo.toml"]
    # No debe levantar ninguna excepción -- si _valor_unico_normalizado no
    # atrapara esto, jaxsvc_puede_leer (llamado indirectamente si Fase A
    # diera ok) levantaría ValueError sin capturar.
    resultado = verificar_fase_a(entorno, _resolver_identidad())
    assert not resultado.ok
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_CONFIG_PATH")
    assert "absoluta" in motivo


def test_clave_fuera_de_alcance_no_absoluta_da_hallazgo_no_traceback():
    entorno = dict(ENTORNO_SANO)
    entorno["JAX_OTRA_RUTA_DIR"] = ["relativo/tambien"]
    resultado = verificar_fase_a(entorno, _resolver_identidad())
    assert not resultado.ok
    motivo = next(h.motivo for h in resultado.problemas if h.clave == "JAX_OTRA_RUTA_DIR")
    assert "absoluta" in motivo


def test_verificar_de_punta_a_punta_con_ruta_relativa_no_revienta():
    """Fin a fin: una clave en alcance con valor relativo no debe hacer
    que verificar() levante una excepción -- tiene que devolver (False,
    reporte), como cualquier otro problema."""
    texto = (
        "JAX_CONFIG_PATH=config/relativo.toml\n"
        "JAX_AUDIT_LOG_PATH=/var/log/jax/las_manos/audit.jsonl\n"
        "JAX_REPO_BASE=/srv/jax-data/repo\n"
    )
    import rutas_de_produccion_verificador as mod
    original_resolver = mod.resolver_como_jaxsvc
    original_fase_c = mod.verificar_fase_c
    try:
        mod.resolver_como_jaxsvc = lambda ruta, ejecutar=None: ruta
        mod.verificar_fase_c = lambda resolver: mod.ResultadoFaseC()
        ok, reporte = mod.verificar(texto)  # no debe lanzar
    finally:
        mod.resolver_como_jaxsvc = original_resolver
        mod.verificar_fase_c = original_fase_c
    assert not ok
    assert "absoluta" in reporte
