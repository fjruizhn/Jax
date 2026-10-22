#!/usr/bin/env python3
"""Quien llama a `cli_sandbox.run_cli` tiene que cablear tambien `preparar_arranque`
(auditoria del SHA 174da8c, MINOR-35).

`preparar_arranque()` purga el estado que dejo un proceso anterior en el directorio de
credencial de cada perfil de suscripcion. Es BLOQUEANTE (se llama via `asyncio.to_thread`) y el
modulo `cli_sandbox` no la dispara solo: un import no puede borrar archivos. Hoy no la llama
nadie, porque tampoco llama nadie a `run_cli` (los pasos 6 y 9 todavia no existen). Sin este
control, el dia que esos pasos lleguen, el arranque que olviden cablear no lo nota nadie: el
estado de un proceso viejo se queda en la credencial.

Regla (AST, sin ejecutar nada): todo modulo `.py` de un repo escaneado que llame a `run_cli`
(`run_cli(...)`, `cli_sandbox.run_cli(...)` o un alias `from cli_sandbox import run_cli as x`)
tiene que REFERIR a `preparar_arranque` dentro de una funcion (`def`/`async def`): llamarla o
pasarla como argumento, p. ej. `await asyncio.to_thread(cli_sandbox.preparar_arranque)`. Una
referencia a nivel de modulo no cuenta (correria al importar), ni un comentario, ni una
cadena, ni la definicion de otra funcion que se llame igual.

Quedan fuera: `cli_sandbox.py` y `_cli_sandbox_test.py` en el ROOT del repo (el nucleo y su
test dedicado) y los archivos de test (`test_*.py`, `*_test.py`, o bajo `tests/`). Un modulo que
de verdad no deba arrancar (otro proceso lo hace por el) va en `_SIN_ARRANQUE_JUSTIFICADO`, con
la ruta relativa al root y el porque; hoy esta vacio.

LO QUE NO CUBRE (residuo conocido, igual que los demas escaneos por AST de esta carpeta): que
el arranque de verdad se ejecute antes de la primera llamada, o que el modulo que llama a
`run_cli` sea el mismo que arranca (basta con que el modulo cablee las dos cosas); codigo fuera
de los repos escaneados (en CI solo jax: jax-platform no existe en el runner); y nombres
calculados (`getattr(m, "run_" + "cli")`). Es una barrera de revision, no de ejecucion.

Corre con:
  JAX_PLATFORM_REPO_ROOT=/nonexistent python3 -m pytest policy/tests/test_run_cli_solo_con_preparar_arranque.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

_THIS_REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_roots() -> list[Path]:
    roots = [_THIS_REPO_ROOT]
    env_root = os.environ.get("JAX_PLATFORM_REPO_ROOT")
    roots.append(Path(env_root) if env_root else _THIS_REPO_ROOT.parent / "jax-platform")
    return roots


REPO_ROOTS = _repo_roots()

EXCLUDE_DIR_NAMES = {
    ".venv", "venv", "node_modules", ".git", ".worktrees", "worktrees",
    "__pycache__", "dist", "build",
}

#: El nucleo y su test dedicado, SOLO en el root del repo.
_NUCLEO = frozenset({"cli_sandbox.py", "_cli_sandbox_test.py"})

#: ruta relativa al root -> por que ese modulo llama a `run_cli` sin cablear el arranque.
#: Vacio hoy. Cada entrada se justifica por escrito: no es una allowlist ciega, el test
#: `test_cada_exencion_declarada_se_justifica_y_sigue_llamando_a_run_cli` la mantiene honesta.
_SIN_ARRANQUE_JUSTIFICADO: dict[str, str] = {}

_RUN_CLI = "run_cli"
_ARRANQUE = "preparar_arranque"


def _nombres_importados(tree: ast.AST, objetivo: str) -> set[str]:
    """Los nombres locales bajo los que `objetivo` entra por `from <m> import objetivo [as x]`."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name == objetivo:
                    out.add(a.asname or a.name)
    return out


def _nombres_de(nodo: ast.AST) -> set[str]:
    """Nombres sueltos y atributos referidos bajo `nodo` (no cadenas ni comentarios)."""
    out = set()
    for n in ast.walk(nodo):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def _llama_a_run_cli(tree: ast.AST) -> bool:
    alias = _nombres_importados(tree, _RUN_CLI) | {_RUN_CLI}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if (isinstance(f, ast.Name) and f.id in alias) or (isinstance(f, ast.Attribute) and f.attr == _RUN_CLI):
                return True
    return False


def _refiere_al_arranque_en_una_funcion(tree: ast.AST) -> bool:
    alias = _nombres_importados(tree, _ARRANQUE) | {_ARRANQUE}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in fn.body:
            for n in ast.walk(sub):
                if isinstance(n, ast.Name) and n.id in alias:
                    return True
                if isinstance(n, ast.Attribute) and n.attr == _ARRANQUE:
                    return True
    return False


def _llama_a_run_cli_sin_arranque(tree: ast.AST) -> bool:
    return _llama_a_run_cli(tree) and not _refiere_al_arranque_en_una_funcion(tree)


def _es_de_test(rel: Path) -> bool:
    return (
        rel.name.startswith("test_") or rel.name.endswith("_test.py")
        or any(p in ("tests", "test") for p in rel.parts[:-1])
    )


def _es_el_nucleo(root: Path, path: Path) -> bool:
    """El `cli_sandbox.py` del ROOT (el symlink las_manos/cli_sandbox.py resuelve ahi)."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return len(rel.parts) == 1 and rel.name in _NUCLEO


def _iter_python_files():
    for root in REPO_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(root)
            if any(part in EXCLUDE_DIR_NAMES for part in rel.parts):
                continue
            yield root, path


def encontrar_llamadores_sin_arranque() -> tuple[list[str], int]:
    """(rutas que llaman a run_cli sin cablear preparar_arranque, archivos escaneados)."""
    violaciones, vistos = [], 0
    for root, path in _iter_python_files():
        rel = path.relative_to(root)
        if _es_el_nucleo(root, path) or _es_de_test(rel):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (UnicodeDecodeError, SyntaxError):
            continue
        vistos += 1
        if _llama_a_run_cli_sin_arranque(tree) and rel.as_posix() not in _SIN_ARRANQUE_JUSTIFICADO:
            violaciones.append(str(path))
    return violaciones, vistos


# --------------------------------------------------------------------------
# El control sobre el arbol real
# --------------------------------------------------------------------------

def test_ningun_llamador_de_run_cli_deja_el_arranque_sin_cablear() -> None:
    violaciones, vistos = encontrar_llamadores_sin_arranque()
    assert vistos > 0, "el escaneo no vio ningun archivo: un control vacio no controla nada"
    assert not violaciones, (
        "llaman a cli_sandbox.run_cli sin referir a preparar_arranque en una funcion de arranque "
        f"(ver el docstring de este archivo): {violaciones}"
    )


# --------------------------------------------------------------------------
# Autopruebas sobre snippets (strings, no archivos plantados en el arbol)
# --------------------------------------------------------------------------

def _marca(fuente: str) -> bool:
    return _llama_a_run_cli_sin_arranque(ast.parse(fuente))


def test_run_cli_sin_preparar_arranque_se_marca() -> None:
    assert _marca("import cli_sandbox\nasync def f():\n    return await cli_sandbox.run_cli('codex')\n")
    assert _marca("from cli_sandbox import run_cli\nasync def f():\n    return await run_cli('codex')\n")
    assert _marca("from cli_sandbox import run_cli as rc\nasync def f():\n    return await rc('codex')\n")
    assert _marca("def f(m):\n    return m.run_cli('codex')\n")


def test_con_las_dos_llamadas_no_se_marca() -> None:
    assert not _marca(
        "import asyncio, cli_sandbox\n"
        "async def arrancar():\n    await asyncio.to_thread(cli_sandbox.preparar_arranque)\n"
        "async def f():\n    return await cli_sandbox.run_cli('codex')\n"
    )
    assert not _marca(
        "import cli_sandbox\n"
        "def arrancar():\n    return cli_sandbox.preparar_arranque()\n"
        "async def f():\n    return await cli_sandbox.run_cli('codex')\n"
    )
    assert not _marca(
        "from cli_sandbox import run_cli, preparar_arranque as pa\n"
        "async def arrancar():\n    await asyncio.to_thread(pa)\n"
        "async def f():\n    return await run_cli('codex')\n"
    )


def test_sin_run_cli_no_hay_nada_que_marcar() -> None:
    assert not _marca("import cli_sandbox\nasync def arrancar():\n    await asyncio.to_thread(cli_sandbox.preparar_arranque)\n")
    assert not _marca("x = 1\n")
    assert not _marca("def f(run_cli):\n    return run_cli\n")  # nombra, no llama


def test_lo_que_no_es_una_referencia_en_una_funcion_no_cuenta_como_cablear() -> None:
    llamada = "async def f():\n    return await cli_sandbox.run_cli('codex')\n"
    # a nivel de modulo: correria al importar
    assert _marca("import cli_sandbox\ncli_sandbox.preparar_arranque()\n" + llamada)
    # en un comentario o una cadena
    assert _marca("# preparar_arranque()\n" + llamada)
    assert _marca("'preparar_arranque'\n" + llamada)
    assert _marca("def doc():\n    '''hay que llamar a preparar_arranque'''\n" + llamada)
    # definir una funcion con ese nombre no es cablearla
    assert _marca("def preparar_arranque():\n    pass\n" + llamada)


def test_las_exenciones_son_el_nucleo_los_tests_y_la_lista_explicita(tmp_path, monkeypatch) -> None:
    malo = "import cli_sandbox\nasync def f():\n    return await cli_sandbox.run_cli('codex')\n"
    raiz = tmp_path / "repo"
    archivos = {
        "cli_sandbox.py": malo,                  # el nucleo, en el root: exento
        "_cli_sandbox_test.py": malo,            # su test dedicado: exento
        "tests/test_algo.py": malo,              # test: exento
        "algo_test.py": malo,                    # test: exento
        "tools/cli_sandbox.py": malo,            # otro archivo con el nombre del nucleo: NO exento
        "paso6/chat.py": malo,                   # llamador real sin arranque: marcado
        "paso9/jacobs.py": malo.replace("async def f", "async def arrancar():\n"
                                        "    await asyncio.to_thread(cli_sandbox.preparar_arranque)\nasync def f"),
        "paso9/solo_arranca.py": "import cli_sandbox\nasync def a():\n    cli_sandbox.preparar_arranque()\n",
    }
    for rel, contenido in archivos.items():
        (raiz / rel).parent.mkdir(parents=True, exist_ok=True)
        (raiz / rel).write_text(contenido)
    modulo = sys.modules[__name__]
    monkeypatch.setattr(modulo, "REPO_ROOTS", [raiz])
    violaciones, vistos = encontrar_llamadores_sin_arranque()
    assert {Path(v).relative_to(raiz).as_posix() for v in violaciones} == {"tools/cli_sandbox.py", "paso6/chat.py"}
    assert vistos == 4  # tools/cli_sandbox.py, paso6/chat.py, paso9/jacobs.py, paso9/solo_arranca.py
    # con la justificacion escrita, el llamador deja de marcarse
    monkeypatch.setattr(modulo, "_SIN_ARRANQUE_JUSTIFICADO", {"paso6/chat.py": "lo arranca el servicio X"})
    violaciones, _ = encontrar_llamadores_sin_arranque()
    assert {Path(v).relative_to(raiz).as_posix() for v in violaciones} == {"tools/cli_sandbox.py"}


def test_cada_exencion_declarada_se_justifica_y_sigue_llamando_a_run_cli() -> None:
    """Una exencion sin motivo, o de un archivo que ya no llama a run_cli, es letra muerta."""
    for rel, porque in _SIN_ARRANQUE_JUSTIFICADO.items():
        assert porque.strip(), f"{rel}: la exencion no dice por que"
        for root in REPO_ROOTS:
            ruta = root / rel
            if ruta.exists():
                assert _llama_a_run_cli(ast.parse(ruta.read_text(encoding="utf-8"))), (
                    f"{rel} ya no llama a run_cli: quitarlo de _SIN_ARRANQUE_JUSTIFICADO"
                )
                break
        else:
            pytest.fail(f"{rel} figura como exento y no existe en ningun repo escaneado")
