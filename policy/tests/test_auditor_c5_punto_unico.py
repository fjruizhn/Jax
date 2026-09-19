#!/usr/bin/env python3
# policy/tests/test_auditor_c5_punto_unico.py
"""C5 se audita SIEMPRE con la faceta que `eleccion_c5.elegir_y_resolver_auditor` elige
para la misión -- nunca con `cfg.auditor_faceta` (la de nube) resuelta por su cuenta.

**El hallazgo real, revisión 2026-09-18.** `scripts/ejecutor_contratos/mision_de_humo.py`
validaba el arranque con `hosts_mision={maquina}` (que SÍ pasa por la elección nueva,
`arranque.py::p_c5`) pero después auditaba la entrega real con
`resolve_facet(cfg.auditor_faceta)`, resuelto por su cuenta -- SIEMPRE el auditor de
nube, sin importar qué máquina fuera. Contra `bridge` (datos de clientes), el arranque
aprobaba (validó el auditor LOCAL) y el script mandaba capturas reales, comandos y líneas
citadas a Thot (OpenAI) igual: el gate que el auditor local vino a cerrar, abierto por el
costado -- "la puerta valida con uno y audita con otro". El docstring del script decía
"nunca contra un servidor de clientes", pero la máquina es un argumento de CLI: nada lo
hacía cumplir. Arreglado en el mismo commit que agrega este detector.

Enforcement mecánico, no una lectura manual: un archivo que llama `resolve_facet(...)`
con una expresión que menciona literalmente `cfg.auditor_faceta` (la clave DE NUBE) en el
argumento es sospechoso -- el único lugar legítimo para leer esa clave es
`eleccion_c5.elegir_auditor_faceta`/`elegir_y_resolver_auditor`, que decide CUÁL faceta
mirar según los hosts de la misión. `policy/tests/test_no_fail_open_except.py` (P10) usa
el mismo criterio -- grep mecánico sobre el árbol, con EXCEPCIONES declaradas y motivo
escrito, nunca una lista que alguien tiene que recordar actualizar.

Auto-verificado (Principio VII -- un freno sin prueba no es freno): un archivo sintético
con el patrón prohibido, sin excepción, tiene que aparecer; retirado (o con excepción),
tiene que desaparecer. Ver test_un_archivo_con_el_patron_se_detecta.
"""
from __future__ import annotations

import re
from pathlib import Path

_THIS_REPO_ROOT = Path(__file__).resolve().parents[2]

# Mismo criterio que test_archivos_de_test_wireados_en_ci.py.
EXCLUDE_DIR_NAMES = {
    ".venv", "venv", "node_modules", ".git", ".worktrees", "worktrees",
    "__pycache__", "dist", "build", ".superpowers", ".claude", ".claude-flow",
    ".pytest_cache",
}

# `resolve_facet(` seguido, en la misma expresión (hasta el paréntesis que cierra), de
# `cfg.auditor_faceta` SIN el sufijo `_local` -- eso es la clave de nube. No matchea
# `cfg.auditor_faceta_local` (positivo: el local SÍ se puede resolver directo, es el
# default correcto para una misión sin hosts) ni `elegir_auditor_faceta(cfg, ...)` (la
# función que SÍ decide bien).
_PATRON = re.compile(r"resolve_facet\([^)]*\bcfg\.auditor_faceta\b(?!_local)[^)]*\)")

# Excepciones EXPLÍCITAS, con motivo escrito -- nunca por olvido. Mismo formato que
# test_archivos_de_test_wireados_en_ci.py::EXCEPCIONES.
EXCEPCIONES: dict[str, str] = {
    "scripts/ejecutor_contratos/probar_c5.py": (
        "herramienta de MEDICIÓN manual (ver su propio docstring: '--auditor FACETA "
        "... para MEDIR y para VER FALLAR'), no una misión: nunca recibe una máquina "
        "por CLI, `hosts_mision` es SIEMPRE el literal {'hall9000'} y sólo manda "
        "canarios sintéticos y la trampa/limpia de trampa_c5.json -- nunca datos "
        "reales de un servidor. El override es la función explícita del script; "
        "`validar_eleccion` sigue corriendo con los datos REALES del inventario "
        "(hosts_de_la_mision), no supuestos -- ver el hallazgo menor de la misma "
        "revisión, ya corregido."
    ),
}


_ESTE_ARCHIVO = Path(__file__).resolve()


def _iter_py_files(repo_root: Path):
    for path in repo_root.rglob("*.py"):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIR_NAMES for part in path.relative_to(repo_root).parts):
            continue
        # Se excluye a sí mismo: el docstring y el motivo de EXCEPCIONES citan el patrón
        # prohibido en prosa (para explicarlo), no como código real -- sin esto el
        # detector se marcaría a sí mismo como una violación cada vez que se corre.
        if path.resolve() == _ESTE_ARCHIVO:
            continue
        yield path


def archivos_con_resolucion_por_su_cuenta(
        repo_root: Path = _THIS_REPO_ROOT, excepciones: dict[str, str] | None = None) -> list[str]:
    excepciones = EXCEPCIONES if excepciones is None else excepciones
    hallados = []
    for path in _iter_py_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        if rel in excepciones:
            continue
        try:
            texto = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # fail-soft: un archivo ilegible no es el patrón prohibido, es otro problema
            continue
        if _PATRON.search(texto):
            hallados.append(rel)
    return sorted(hallados)


# --- guardas del propio control (Principio I: no dar verde por vacío) ---------------------

def test_el_escaneo_ve_archivos_conocidos():
    vistos = list(_iter_py_files(_THIS_REPO_ROOT))
    assert vistos, "el escaneo no vio NINGUN archivo .py"
    assert any(p.as_posix().endswith("jax/ejecutor/contratos/eleccion_c5.py") for p in vistos), (
        "el escaneo no ve eleccion_c5.py: ¿cambió la estructura del árbol?")


def test_las_excepciones_siguen_vigentes():
    for rel, motivo in EXCEPCIONES.items():
        assert motivo and motivo.strip(), f"excepcion sin motivo escrito: {rel}"
        assert (_THIS_REPO_ROOT / rel).is_file(), f"la excepcion {rel!r} ya no existe: sobra, hay que quitarla"


def test_un_archivo_con_el_patron_se_detecta(tmp_path):
    (tmp_path / "sin_excepcion.py").write_text(
        "async def f(conn):\n"
        "    cfg = await leer_config(conn)\n"
        "    auditor_f = await resolve_facet(cfg.auditor_faceta)\n"
    )
    (tmp_path / "con_el_local_ok.py").write_text(
        "async def f(conn):\n"
        "    cfg = await leer_config(conn)\n"
        "    auditor_f = await resolve_facet(cfg.auditor_faceta_local)\n"
    )
    (tmp_path / "con_la_funcion_correcta_ok.py").write_text(
        "async def f(conn):\n"
        "    cfg = await leer_config(conn)\n"
        "    faceta = elegir_auditor_faceta(cfg, hay_datos_de_clientes=True)\n"
        "    auditor_f = await resolve_facet(faceta)\n"
    )
    hallados = archivos_con_resolucion_por_su_cuenta(repo_root=tmp_path, excepciones={})
    assert hallados == ["sin_excepcion.py"], hallados


# --- el control real ------------------------------------------------------------------

def test_ningun_archivo_resuelve_el_auditor_de_nube_por_su_cuenta():
    hallados = archivos_con_resolucion_por_su_cuenta()
    assert not hallados, (
        "archivos que resuelven cfg.auditor_faceta (la faceta DE NUBE) directo con "
        "resolve_facet(), sin pasar por eleccion_c5.elegir_auditor_faceta/"
        "elegir_y_resolver_auditor -- el mismo patrón del hallazgo crítico de la "
        "revisión 2026-09-18 (mision_de_humo.py auditaba con nube aunque el arranque "
        "hubiera validado el auditor local):\n  " + "\n  ".join(hallados) +
        "\n\nUsá eleccion_c5.elegir_y_resolver_auditor(conn, cfg=cfg, "
        "hosts_mision=<hosts reales de la misión>, resolve_facet=resolve_facet), o "
        "declará una excepción en EXCEPCIONES con motivo escrito."
    )


if __name__ == "__main__":
    import sys

    import pytest as _pytest

    sys.exit(_pytest.main([__file__, "-v"]))
