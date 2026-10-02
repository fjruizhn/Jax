#!/usr/bin/env python3
"""Un `Titular` solo nace en `cli_sandbox.exigir_titular` (auditoria 2026-10-02,
ronda 2, MINOR-16).

`cli_sandbox.Titular` es la prueba de que la compuerta de la suscripcion autorizo a
un usuario. En tiempo de ejecucion NO es infalsificable: quien importe el modulo puede
leer `_SELLO`, entrar a `_EMITIENDO`, escribir `emitido_mono` con `object.__setattr__`,
escribir en el registro `_EMITIDOS` o llamar `object.__new__(Titular)` (la docstring de
`Titular` lo dice). Lo que hace
visible un atajo asi es ESTE control, de revision de codigo:

  fuera de `cli_sandbox.py` y de `_cli_sandbox_test.py` (en el ROOT de un repo
  escaneado; el symlink las_manos/cli_sandbox.py es el mismo archivo), el AST no puede
  - nombrar `_SELLO`, `_EMITIENDO`, `_EMITIDOS` ni `emitido_mono` (como nombre, atributo o cadena:
    `object.__setattr__(t, "emitido_mono", x)` cuenta), ni con una concatenacion de
    literales (`"emitido" + "_mono"`);
  - declarar una subclase de `Titular` (`class X(Titular)`, `type("X", (Titular,), {})`);
  - fabricar una instancia saltandose el constructor (`object.__new__(Titular)`,
    `Titular.__new__(Titular)`).

LO QUE NO CUBRE (residuo conocido, mismo criterio que el scanner de subprocesos):
codigo que no este en los repos escaneados (en CI solo jax: jax-platform no existe en
el runner), y formas que el AST no ve: `exec`/`eval` de texto, nombres calculados con
algo mas que `+` de literales, o `vars(cli_sandbox)["_SELLO"]` con la clave armada en
tiempo de ejecucion. Es una barrera de revision, no de ejecucion.

Corre con:
  JAX_PLATFORM_REPO_ROOT=/nonexistent python3 -m pytest policy/tests/test_titular_solo_via_exigir_titular.py

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

#: Los dos unicos archivos que pueden tocar el sello: el modulo y su test, en el ROOT.
ALLOWED_FILENAMES = frozenset({"cli_sandbox.py", "_cli_sandbox_test.py"})

#: Internos de `Titular` que nadie fuera del modulo puede nombrar.
# Armados con "".join y no escritos enteros: el escaneo pliega las concatenaciones de
# literales y marcaria a este mismo archivo (que no es de los aprobados).
_PROHIBIDOS = frozenset({
    "".join(("_SEL", "LO")), "".join(("_EMITIEN", "DO")), "".join(("_EMITI", "DOS")),
    "".join(("emitido", "_mono")),
})
_CLASE = "Titular"


def _plegar(nodo: ast.AST, _prof: int = 0) -> str | None:
    """Una cadena hecha solo de literales unidos con `+`, o None."""
    if _prof > 6:
        return None
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
        return nodo.value
    if isinstance(nodo, ast.BinOp) and isinstance(nodo.op, ast.Add):
        izq = _plegar(nodo.left, _prof + 1)
        der = _plegar(nodo.right, _prof + 1) if izq is not None else None
        return None if izq is None or der is None else izq + der
    return None


def _nombra_titular(nodo: ast.AST) -> bool:
    return (isinstance(nodo, ast.Name) and nodo.id == _CLASE) or (
        isinstance(nodo, ast.Attribute) and nodo.attr == _CLASE
    )


def _usos_indebidos(tree: ast.AST) -> list[str]:
    """Las formas prohibidas que hay en `tree`, una descripcion por cada una."""
    hallados: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in _PROHIBIDOS:
            hallados.append(f"nombre {n.id}")
        elif isinstance(n, ast.Attribute) and n.attr in _PROHIBIDOS:
            hallados.append(f"atributo {n.attr}")
        elif isinstance(n, (ast.Constant, ast.BinOp)):
            texto = _plegar(n)
            if texto in _PROHIBIDOS:
                hallados.append(f"cadena {texto!r}")
        elif isinstance(n, ast.ClassDef) and any(_nombra_titular(b) for b in n.bases):
            hallados.append(f"subclase de {_CLASE} ({n.name})")
        elif isinstance(n, ast.Call):
            f = n.func
            # object.__new__(Titular) / Titular.__new__(Titular) / cualquier X.__new__(... Titular ...)
            if isinstance(f, ast.Attribute) and f.attr == "__new__" and (
                _nombra_titular(f.value) or any(_nombra_titular(a) for a in n.args)
            ):
                hallados.append(f"{_CLASE}.__new__")
            # type("X", (Titular,), {})
            elif isinstance(f, ast.Name) and f.id == "type" and len(n.args) == 3 and isinstance(
                n.args[1], (ast.Tuple, ast.List)
            ) and any(_nombra_titular(e) for e in n.args[1].elts):
                hallados.append(f"subclase de {_CLASE} via type()")
    return hallados


def _iter_python_files():
    for root in REPO_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in EXCLUDE_DIR_NAMES for part in path.relative_to(root).parts):
                continue
            yield root, path


def _es_archivo_aprobado(root: Path, path: Path) -> bool:
    """El `cli_sandbox.py` / `_cli_sandbox_test.py` del ROOT de un repo escaneado (se
    resuelve el symlink de las_manos/), no cualquier archivo con ese nombre."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return len(rel.parts) == 1 and rel.name in ALLOWED_FILENAMES


def find_violations() -> list[str]:
    violaciones = []
    for root, path in _iter_python_files():
        if _es_archivo_aprobado(root, path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (UnicodeDecodeError, SyntaxError):
            continue
        for uso in _usos_indebidos(tree):
            violaciones.append(f"{path}: {uso}")
    return violaciones


# --------------------------------------------------------------------------
# Autopruebas (Principio VII): operan sobre SNIPPETS, no sobre archivos del arbol.
# --------------------------------------------------------------------------

def _usos(source: str) -> list[str]:
    return _usos_indebidos(ast.parse(source))


def test_detecta_los_internos_del_sello() -> None:
    assert _usos("import cli_sandbox\nx = cli_sandbox._SELLO\n")
    assert _usos("from cli_sandbox import _SELLO\nt = _SELLO\n")
    assert _usos("from cli_sandbox import _EMITIENDO\n_EMITIENDO.set(True)\n")
    assert _usos("import cli_sandbox\ncli_sandbox._EMITIENDO.set(True)\n")
    assert _usos("t = algo()\nt.emitido_mono = 0\n")
    assert _usos("print(titular.emitido_mono)\n")
    # MINOR-24: el registro de titulares emitidos tampoco se toca desde fuera
    assert _usos("import cli_sandbox\ncli_sandbox._EMITIDOS[t] = (1, 1, 'chat')\n")
    assert _usos("from cli_sandbox import _EMITIDOS\n_EMITIDOS.clear()\n")
    assert _usos("getattr(cli_sandbox, '_EMITI' + 'DOS')\n")


def test_detecta_el_nombre_como_cadena_aunque_este_partido() -> None:
    assert _usos("object.__setattr__(t, 'emitido_mono', 0.0)\n")
    assert _usos("object.__setattr__(t, '_sello', None)\ngetattr(cli_sandbox, '_SELLO')\n")
    assert _usos("getattr(t, 'emitido' + '_mono')\n")
    assert _usos("getattr(cli_sandbox, '_SEL' + 'LO')\n")


def test_detecta_subclases_de_titular() -> None:
    assert _usos("from cli_sandbox import Titular\nclass Falso(Titular):\n    pass\n")
    assert _usos("import cli_sandbox\nclass Falso(cli_sandbox.Titular):\n    pass\n")
    assert _usos("from cli_sandbox import Titular\nT = type('T', (Titular,), {})\n")
    assert _usos("import cli_sandbox\nT = type('T', (cli_sandbox.Titular,), {})\n")


def test_detecta_la_fabricacion_sin_constructor() -> None:
    assert _usos("from cli_sandbox import Titular\nt = object.__new__(Titular)\n")
    assert _usos("import cli_sandbox\nt = object.__new__(cli_sandbox.Titular)\n")
    assert _usos("from cli_sandbox import Titular\nt = Titular.__new__(Titular)\n")


def test_no_inventa_violaciones() -> None:
    # usar Titular como tipo o pedirlo a exigir_titular es el uso correcto
    assert not _usos("from cli_sandbox import Titular, exigir_titular\n"
                     "async def f(u, t):\n    t2: Titular = await exigir_titular(u, t, 'chat')\n    return t2.user_id\n")
    assert not _usos("class Titular:\n    pass\n")          # otra clase con el mismo nombre
    assert not _usos("class Otra(Base):\n    pass\n")
    assert not _usos("t = object.__new__(Otra)\n")
    assert not _usos("_SELLO_REAL = '/srv/jax-data/facet-cache-seal'\n")   # nombre parecido, distinto
    assert not _usos("x = 'emitido'\ny = 'mono'\n")
    assert not _usos("T = type('T', (Base,), {})\n")


def test_los_archivos_aprobados_son_solo_los_del_root() -> None:
    root = _THIS_REPO_ROOT
    assert _es_archivo_aprobado(root, root / "cli_sandbox.py")
    assert _es_archivo_aprobado(root, root / "_cli_sandbox_test.py")
    assert _es_archivo_aprobado(root, root / "las_manos" / "cli_sandbox.py")  # symlink al mismo archivo
    assert not _es_archivo_aprobado(root, root / "tools" / "cli_sandbox.py")
    assert not _es_archivo_aprobado(root, root / "hyde_sandbox.py")


def test_un_archivo_plantado_fuera_de_los_aprobados_aparece(tmp_path) -> None:
    """Control del control sobre el recorrido real: el mismo contenido pasa en el
    `cli_sandbox.py` del root y se marca en cualquier otro lado."""
    raiz = tmp_path / "repo"
    (raiz / "tools").mkdir(parents=True)
    plantado = "import cli_sandbox\nt = object.__new__(cli_sandbox.Titular)\nt2 = cli_sandbox._SELLO\n"
    (raiz / "cli_sandbox.py").write_text(plantado)
    (raiz / "tools" / "cli_sandbox.py").write_text(plantado)
    (raiz / "tools" / "otro.py").write_text(plantado)
    global REPO_ROOTS
    anteriores = REPO_ROOTS
    REPO_ROOTS = [raiz]
    try:
        encontrados = {Path(v.split(":")[0]).relative_to(raiz).as_posix() for v in find_violations()}
    finally:
        REPO_ROOTS = anteriores
    assert encontrados == {"tools/cli_sandbox.py", "tools/otro.py"}


def test_el_escaneo_ve_el_modulo_aprobado_y_el_arbol_real() -> None:
    """Un control que no ve nada da verde sin haber mirado: tiene que haber archivos
    escaneados y el modulo aprobado tiene que contener los internos (si no, el
    control de arriba protege un nombre que ya no existe)."""
    archivos = list(_iter_python_files())
    assert len(archivos) > 100, "el escaneo no ve el arbol"
    fuente = (_THIS_REPO_ROOT / "cli_sandbox.py").read_text(encoding="utf-8")
    for nombre in _PROHIBIDOS:
        assert nombre in fuente, f"{nombre} ya no existe en cli_sandbox.py: actualizar este control"
    assert "class Titular" in fuente


def test_nadie_fabrica_un_titular_fuera_de_cli_sandbox() -> None:
    violaciones = find_violations()
    assert not violaciones, (
        f"{len(violaciones)} uso(s) de los internos de `Titular` fuera de cli_sandbox.py: "
        "un Titular solo lo construye cli_sandbox.exigir_titular:\n" + "\n".join(violaciones)
    )


def main() -> int:
    violaciones = find_violations()
    if violaciones:
        print(f"FAIL — {len(violaciones)} uso(s):")
        for v in violaciones:
            print(f"  {v}")
        return 1
    print("OK — nada fuera de cli_sandbox.py toca los internos de Titular")
    return 0


if __name__ == "__main__":
    sys.exit(main())
