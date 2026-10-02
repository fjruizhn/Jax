"""T16 (2026-10-02, decisión de Fernando D-5 del spec El Faro): el REPL se retiró.

Hallazgo F-1: el Hyde del REPL (`jax/muscles/subprocess_muscle.py`, lanzado desde
`jax/core/main.py`) corría Claude con `--allowedTools "Write,Edit,Read,Bash"` y el
token de la suscripción en su entorno. Se cierra quitando el camino entero: el REPL
interactivo, `jax --task` (que usaba `/command` de jax-platform) y ese músculo.

Estos tests fallan contra el árbol anterior al retiro. Puros: leen el árbol, no
importan nada de lo retirado.
"""
from __future__ import annotations

import ast
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RETIRADOS = ("jax/core/main.py", "jax/muscles/subprocess_muscle.py")
FUERA = {".venv", "venv", ".git", "node_modules", "__pycache__"}


def test_los_archivos_del_repl_ya_no_estan_en_el_arbol():
    assert [rel for rel in RETIRADOS if (RAIZ / rel).exists()] == []


def _modulos_importados(ruta: Path) -> list[str]:
    modulos: list[str] = []
    for nodo in ast.walk(ast.parse(ruta.read_text(encoding="utf-8"))):
        if isinstance(nodo, ast.Import):
            modulos += [a.name for a in nodo.names]
        elif isinstance(nodo, ast.ImportFrom):
            base = nodo.module or ""
            modulos.append(base)
            modulos += [f"{base}.{a.name}" for a in nodo.names]
    return modulos


def test_nada_importa_el_repl_ni_el_musculo_de_subproceso():
    prohibidos = {"jax.core.main", "jax.muscles.subprocess_muscle"}
    hallazgos = []
    for ruta in RAIZ.rglob("*.py"):
        if FUERA & set(ruta.relative_to(RAIZ).parts):
            continue
        if ruta == Path(__file__).resolve():
            continue
        try:
            importados = _modulos_importados(ruta)
        except SyntaxError:
            continue
        hallazgos += [f"{ruta.relative_to(RAIZ)}: {m}" for m in importados if m in prohibidos]
    assert hallazgos == []
