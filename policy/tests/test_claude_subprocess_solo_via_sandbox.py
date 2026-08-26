#!/usr/bin/env python3
"""Gobernanza de sub-agentes de Claude Code (DEUDA.md, "Sub-agentes de
Claude Code sin gobernanza real"): el UNICO punto de entrada aprobado
para lanzar `claude` como subproceso es
hyde_sandbox.py::run_sandboxed_claude() -- aplica el sandbox de bwrap y
el lock cross-proceso (flock(2)) que serializa las invocaciones entre el
proceso de las_manos y el del REPL (ver Task 1/2 del plan que agregó este
scanner).

Enforcement mecanico y acotado (mismo espiritu que
test_no_fail_open_except.py): un archivo que (a) lanza un subproceso por
CUALQUIERA de las formas conocidas (asyncio.create_subprocess_exec/_shell,
loop.subprocess_exec, subprocess.run/Popen/call/check_call/check_output,
os.system, os.exec*), Y (b) menciona "claude" en un STRING LITERAL del AST
(no en un comentario `#`), DEBE ser uno de los dos archivos aprobados del
root del repo -- cualquier otro archivo con esa combinacion esta lanzando
`claude` (o algo que lo referencia) por fuera del sandbox compartido.

Los dos archivos exentos, ambos por ruta resuelta en el ROOT del repo (no
por basename en cualquier lado del arbol, ver ALLOWED_FILENAMES):
  - hyde_sandbox.py -- el modulo aprobado, tambien alcanzable via el
    symlink las_manos/hyde_sandbox.py (es el mismo archivo).
  - _hyde_sandbox_test.py -- su archivo de tests dedicado; su
    subprocess.Popen lanza sys.executable para probar el flock(2) entre
    dos procesos de SO reales, nunca `claude`.

El criterio (b) mira string literals del AST y NO comentarios: los
comentarios de Python se descartan antes de parsear y nunca llegan al AST.
Un archivo que apenas COMENTA sobre claude/Hyde mientras lanza `git` por
subprocess no es una violacion -- caso real que este scanner marcaba mal
(las_manos/motor_registry/tool_authority.py), y que se iba a repetir
porque Hyde se discute por todo el codebase. Los docstrings SI cuentan.

La lista de formas se amplio 2026-08-25 (review final de rama): antes solo
cubria las dos async, asi que un `subprocess.run(["claude", ...])` pasaba
limpio. El item de DEUDA.md que este scanner cierra habla de "cualquier
OTRO músculo/automatización que dispare `claude`", y este repo ya usa la
API sincrona en 9+ archivos para otras cosas (git, etc.) -- es un idioma
vivo aca, no una hipotesis.

No detecta un lanzamiento de `claude` disfrazado (sin la palabra literal
"claude" en el archivo, ej. leida de una variable de entorno con otro
nombre) -- eso queda como residuo conocido, mismo criterio que P10 con
las formas mas sutiles del patron fail-open.

ALCANCE REAL EN CI (misma limitacion honesta que
test_no_fail_open_except.py, ver tambien el header de
.github/workflows/policy.yml): _repo_roots() incluye un fallback a
jax-platform, pero en el runner de GitHub Actions ese directorio NO existe
(repo privado separado, sin checkout cruzado configurado -- requeriria un
PAT/secret nuevo), asi que `if not root.exists(): continue` hace que en CI
solo se escanee el arbol de jax. La verificacion manual 2026-08-25 contra
los dos repos (cero falsos positivos: ningun otro call site de subprocess
-- las_manos/workers/ssh_worker.py, jax/voice/tts.py, jax/voice/ears.py,
jax-platform/backend/api/command.py -- menciona "claude") fue eso, manual;
no es una garantia que este job sostenga corrida a corrida para
jax-platform.

Corre con:
  python3 policy/tests/test_claude_subprocess_solo_via_sandbox.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

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

# Exencion por RUTA RELATIVA, no por nombre pelado: tienen que ser los
# archivos que viven en el root de uno de los repos escaneados. Con
# `path.name in ALLOWED_FILENAMES` un hipotetico tools/hyde_sandbox.py
# quedaba exento tambien, sin ser el modulo real y aprobado.
#
# Se compara sobre la ruta RESUELTA (readlink) a proposito: el modulo real
# vive en el root del repo y se alcanza tambien como
# las_manos/hyde_sandbox.py, que es un SYMLINK a ../hyde_sandbox.py (mismo
# patron que facet_resolver.py / credential_resolver.py). Es el mismo
# archivo, no una copia -- exentarlo no afloja nada: un tools/hyde_sandbox.py
# que fuera un archivo DISTINTO resuelve a tools/hyde_sandbox.py y sigue
# siendo violacion.
#
# _hyde_sandbox_test.py es el archivo de tests DEDICADO del modulo aprobado.
# Su subprocess.Popen (ClaudeSubprocessLockRealCrossProcessTest) lanza
# sys.executable + un script worker para probar que el flock(2) serializa
# entre procesos de SO REALES -- nunca lanza `claude`. Es infraestructura de
# test necesaria del modulo aprobado, misma categoria que el modulo mismo.
# La exencion es de ESE nombre exacto en el root, no de "cualquier test":
# un *_test.py generico que lance `claude` sigue siendo violacion.
ALLOWED_FILENAMES = frozenset({"hyde_sandbox.py", "_hyde_sandbox_test.py"})

# Todas las formas de lanzar un subproceso que este repo podria usar. Las
# dos async eran las unicas cubiertas hasta la review final de rama
# (2026-08-25) -- un subprocess.run(["claude", ...]) pasaba limpio.
# os.exec* se matchea por prefijo (execl/execle/execlp/execlpe/execv/
# execve/execvp/execvpe). loop.subprocess_exec se matchea solo por nombre
# de atributo, sin verificar que el objeto sea un event loop: imprecision
# aceptada a proposito, igual que el resto de este scanner de nivel AST.
_SUBPROCESS_CALL_NAMES = {
    "create_subprocess_exec", "create_subprocess_shell", "subprocess_exec",
    "run", "Popen", "call", "check_call", "check_output",
    "system",
}

_SUBPROCESS_CALL_PREFIXES = ("exec",)

# Nombres genericos que SOLO cuentan como lanzamiento de subproceso si se
# llaman como atributo de `subprocess`/`os` (subprocess.run(...),
# os.system(...)). Sin esto, cualquier `x.run(...)` o `self.call(...)` del
# codebase daria un falso positivo masivo.
_QUALIFIED_ONLY_NAMES = {"run", "call", "check_call", "check_output", "system"}
_SUBPROCESS_MODULES = {"subprocess", "os"}


def _iter_python_files():
    """Rinde pares (root, path) -- el root hace falta para poder exentar
    por ruta relativa, no por nombre de archivo pelado."""
    for root in REPO_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            rel_parts = path.relative_to(root).parts
            if any(part in EXCLUDE_DIR_NAMES for part in rel_parts):
                continue
            yield root, path


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_subprocess_launch(node: ast.Call) -> bool:
    name = _call_name(node.func)
    if name is None:
        return False
    if name not in _SUBPROCESS_CALL_NAMES and not name.startswith(_SUBPROCESS_CALL_PREFIXES):
        return False
    if name in _QUALIFIED_ONLY_NAMES or name.startswith(_SUBPROCESS_CALL_PREFIXES):
        # Solo cuenta como subprocess.run / os.system / os.execv, etc.
        func = node.func
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
            return False
        if func.value.id not in _SUBPROCESS_MODULES:
            return False
    return True


def _calls_create_subprocess(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_subprocess_launch(node):
            return True
    return False


def _references_claude_literal(tree: ast.Module) -> bool:
    """True si el archivo menciona "claude" en un STRING LITERAL del AST
    (incluidos docstrings), no en un comentario `#`.

    Los comentarios de Python se descartan antes de parsear y NUNCA forman
    parte del AST -- por eso este chequeo los excluye de raiz, que es
    exactamente el alcance correcto: un comentario no puede ejecutar nada.
    El chequeo de texto crudo (`"claude" in source.lower()`) daba falso
    positivo con cualquier archivo que apenas COMENTARA sobre claude/Hyde
    -- caso real: las_manos/motor_registry/tool_authority.py, cuyos
    subprocess.run son todos `git` y cuya unica mencion de "claude" es un
    comentario que dice que Hyde corre claude FUERA de ese modulo. Hyde se
    discute por todo este codebase, asi que sin este fix el falso positivo
    se iba a repetir. Los docstrings SI cuentan (son ast.Constant): una
    referencia real, aunque inusual, podria esconderse ahi."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "claude" in node.value.lower():
                return True
    return False


def _is_approved_sandbox_file(root: Path, path: Path) -> bool:
    """Los archivos aprobados son el hyde_sandbox.py del ROOT de un repo
    escaneado y su archivo de tests dedicado -- no cualquier archivo que se
    llame asi. Se resuelve el symlink primero
    (las_manos/hyde_sandbox.py -> ../hyde_sandbox.py es el mismo archivo)."""
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(root.resolve())
    except ValueError:
        return False
    return len(rel.parts) == 1 and rel.name in ALLOWED_FILENAMES


def find_naked_claude_subprocess_files() -> list[str]:
    violations = []
    for root, path in _iter_python_files():
        if _is_approved_sandbox_file(root, path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        # Pre-filtro barato sobre el texto crudo: descarta rapido los
        # archivos que NO pueden matchear, sin pagar el ast.parse. No
        # alcanza por si solo -- ve comentarios, que el AST no (ver
        # _references_claude_literal).
        if "claude" not in source.lower():
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        if _references_claude_literal(tree) and _calls_create_subprocess(tree):
            violations.append(str(path))
    return violations


# --------------------------------------------------------------------------
# Auto-tests del scanner. Viven ACA y no en un archivo aparte a proposito:
# este archivo ya corre en CI (job no-naked-claude-subprocess), y operan
# sobre SNIPPETS de codigo fuente (strings), no sobre archivos plantados en
# el arbol -- asi ningun fixture puede caer dentro de un root escaneado ni
# enredarse con la logica de exencion de _hyde_sandbox_test.py. Nota de
# consistencia: los `subprocess.run(...)` de los snippets de abajo son
# STRING LITERALS, no llamadas reales, asi que este archivo no se convierte
# en violacion de su propia regla (verificado: el scanner lo da OK).
# --------------------------------------------------------------------------

def _detects(source: str) -> bool:
    """Aplica los dos criterios del scanner a un snippet, igual que
    find_naked_claude_subprocess_files() a un archivo real."""
    tree = ast.parse(source)
    return _references_claude_literal(tree) and _calls_create_subprocess(tree)


def test_detecta_subprocess_run_sincrono() -> None:
    assert _detects(
        "import subprocess\n"
        "subprocess.run(['claude', '--print'], capture_output=True)\n"
    ), "subprocess.run(['claude', ...]) debe ser violacion (era invisible hasta 2026-08-25)"


def test_detecta_os_system() -> None:
    assert _detects("import os\nos.system('claude --print hola')\n")


def test_detecta_popen_y_exec_y_loop_subprocess_exec() -> None:
    assert _detects("from subprocess import Popen\nPopen(['claude'])\n")
    assert _detects("import os\nos.execvp('claude', ['claude'])\n")
    assert _detects("loop.subprocess_exec(proto, 'claude', '--print')\n")


def test_detecta_las_formas_async_originales() -> None:
    assert _detects("asyncio.create_subprocess_exec('claude', '--print')\n")
    assert _detects("asyncio.create_subprocess_shell('claude --print')\n")


def test_claude_solo_en_comentario_no_es_violacion() -> None:
    """Regresion del falso positivo real de
    las_manos/motor_registry/tool_authority.py: lanza `git` por subprocess y
    su unica mencion de claude es un comentario `#`. Los comentarios no
    llegan al AST -- no son señal de que el archivo referencie claude."""
    assert not _detects(
        "import subprocess\n"
        "# Este archivo NO lanza claude -- Hyde corre `claude -p` en otro modulo.\n"
        "subprocess.run(['git', 'status'])  # nada de claude aca\n"
    )


def test_claude_en_docstring_si_cuenta() -> None:
    """Los docstrings SI son ast.Constant -- una referencia real, aunque
    inusual, podria esconderse ahi. Solo se excluyen los comentarios `#`."""
    assert _detects(
        "import subprocess\n"
        "def go():\n"
        "    '''lanza claude'''\n"
        "    return subprocess.run(['x'])\n"
    )


def test_subprocess_sin_mencionar_claude_no_es_violacion() -> None:
    assert not _detects("import subprocess\nsubprocess.run(['git', 'status'])\n")


def test_menciona_claude_pero_no_lanza_subprocess_no_es_violacion() -> None:
    assert not _detects("CLI = 'claude'\nprint(CLI)\n")


def test_run_generico_no_calificado_no_cuenta() -> None:
    """`run`/`call`/`system` solo cuentan sobre subprocess/os -- si no,
    cualquier self.run(...) del codebase seria falso positivo."""
    assert not _detects("self.run(['claude'])\n")
    assert not _detects("runner.call('claude')\n")


def test_exencion_es_solo_para_el_root_del_repo() -> None:
    """La exencion es por ruta resuelta en el ROOT, no por basename: un
    tools/hyde_sandbox.py que fuera un archivo DISTINTO sigue siendo
    violacion."""
    root = _THIS_REPO_ROOT
    assert _is_approved_sandbox_file(root, root / "hyde_sandbox.py")
    assert _is_approved_sandbox_file(root, root / "_hyde_sandbox_test.py")
    assert not _is_approved_sandbox_file(root, root / "tools" / "hyde_sandbox.py")
    assert not _is_approved_sandbox_file(root, root / "tools" / "_hyde_sandbox_test.py")


def test_symlink_del_modulo_aprobado_sigue_exento() -> None:
    """las_manos/hyde_sandbox.py es un symlink a ../hyde_sandbox.py -- el
    mismo archivo, alcanzado por el path por el que lo importa las_manos."""
    symlink = _THIS_REPO_ROOT / "las_manos" / "hyde_sandbox.py"
    if symlink.exists():
        assert _is_approved_sandbox_file(_THIS_REPO_ROOT, symlink)


def test_no_naked_claude_subprocess() -> None:
    violations = find_naked_claude_subprocess_files()
    assert not violations, (
        f"{len(violations)} archivo(s) lanzan un subproceso y "
        "mencionan 'claude', fuera de hyde_sandbox.py::run_sandboxed_claude() "
        "-- cualquier invocacion de `claude` como subproceso debe pasar por "
        "ese wrapper (sandbox de bwrap + lock cross-proceso via flock):\n"
        + "\n".join(violations)
    )


def main() -> int:
    violations = find_naked_claude_subprocess_files()
    if violations:
        print(f"FAIL — {len(violations)} archivo(s):")
        for v in violations:
            print(f"  {v}")
        return 1
    print(
        "OK — ningun subprocess de 'claude' fuera de hyde_sandbox.py "
        "(+ su test dedicado _hyde_sandbox_test.py)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
