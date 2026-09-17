#!/usr/bin/env python3
"""P10 — ningún validador o gate puede fallar abierto ante error o
ausencia de señal, incluyendo vía excepción sin capturar (REFORMAS-v3.1.md,
Apendice C-bis, six-impossible-things.html).

Copia hermana de jax-platform/backend/tests/test_no_fail_open_except.py. NO
es un import compartido: son repos privados separados y un checkout cruzado
en CI pediría una credencial nueva. Si se edita la lógica de detección en un
lado, hay que replicarlo en el otro.

Enforcement mecánico y acotado, no un analizador general: un bloque `except`
cuyo cuerpo es únicamente `pass` (o `pass` + comentarios) traga el error sin
propagarlo, sin loguearlo y sin dejar ningún rastro.

REGLA AMPLIA (portada de jax-platform el 2026-09-15, cerrando la deuda de
sincronización que ese archivo declaraba): todo `except` AMPLIO —`Exception`,
`BaseException`, desnudo, o una tupla que incluya alguno— cuyo cuerpo no
relanza (ningún `raise` en el cuerpo) también necesita la marca. Seguir de
largo con un log, o con `x = None`, sigue siendo un fail-soft que tiene que
decir por qué. Rige sobre todo el árbol, tests incluidos.

Marcado, no allowlist: un except legítimo (fail-soft real: nadie depende de
que esa operación haya funcionado) se marca con un comentario en la MISMA
línea del `except`, formato `# fail-soft: <razón específica de este sitio>`.
Sin esa marca, es una violación. Una marca genérica («# fail-soft: ok»)
pasaría el test pero no cumple el trato.

Corre con:
  python3 policy/tests/test_no_fail_open_except.py
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

_THIS_REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_roots() -> list[Path]:
    """Los árboles a escanear. El segundo es jax-platform, que vive fuera de
    este repo.

    ENDURECIDO 2026-09-15: antes se añadía la ruta adivinada
    `_THIS_REPO_ROOT.parent / "jax-platform"` sin comprobar que existiera, y
    `rglob` sobre un directorio inexistente no falla: devuelve vacío. Corriendo
    desde un worktree (`/home/fruiz/worktrees/jax-failopen`) la ruta adivinada
    era `/home/fruiz/worktrees/jax-platform`, que no existe, así que el control
    **decía cubrir dos repos y cubría uno, en silencio**. Es fail-open del
    propio control: exactamente lo que esta prueba existe para prohibir.

    Ahora la ausencia se declara (`ROOTS_AUSENTES`) y hay un test que la mira.
    """
    roots = [_THIS_REPO_ROOT]
    env_root = os.environ.get("JAX_PLATFORM_REPO_ROOT")
    candidato = Path(env_root) if env_root else _THIS_REPO_ROOT.parent / "jax-platform"
    if candidato.is_dir():
        roots.append(candidato)
    else:
        ROOTS_AUSENTES.append(candidato)
    return roots


ROOTS_AUSENTES: list[Path] = []
REPO_ROOTS = _repo_roots()


# Archivos que NO son módulos Python aunque terminen en .py. Se declaran con
# motivo, igual que en tests/test_aiomysql_connect_timeout_tripwire.py: un
# archivo ilegible que no esté aquí ES una violación, y hay un test que lo
# comprueba en los dos sentidos.
# Vacía desde el 2026-09-16 (E-01): el único no-módulo real del árbol,
# _director_patch/routes_block.py, se retiró. El mecanismo se conserva y se
# ejercita con un .py roto declarado a propósito dentro del test.
_NO_PARSEA: dict[str, str] = {}


def _declarado_no_parsea(path: Path) -> bool:
    for root in REPO_ROOTS:
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
        if rel in _NO_PARSEA:
            return True
    return False

EXCLUDE_DIR_NAMES = {
    ".venv", "venv", "node_modules", ".git", ".worktrees", "worktrees",
    "__pycache__", "dist", "build",
}

FAIL_SOFT_MARKER = "# fail-soft:"
# CALIBRACIÓN 2026-09-15. `# fail-closed:` ya se usaba en el árbol
# (check_memory_schema_drift.py) para un `except` que NO sigue de largo: corta
# con un veredicto negativo. El detector sólo miraba `# fail-soft:` y rechazaba
# una marca correcta y más estricta que la que pedía. Un control que obliga a
# escribir una etiqueta falsa para pasar no está midiendo lo que dice medir.
FAIL_CLOSED_MARKER = "# fail-closed:"
MARCAS = (FAIL_SOFT_MARKER, FAIL_CLOSED_MARKER)


def _iter_python_files():
    for root in REPO_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            # Partes RELATIVAS a la raiz (Task 3, 2026-09-15): con `path.parts`
            # absolutas, un checkout que vive bajo un directorio llamado
            # `worktrees` (p. ej. /home/fruiz/worktrees/...) se excluia ENTERO
            # y el test pasaba sin escanear nada.
            if any(part in EXCLUDE_DIR_NAMES for part in path.relative_to(root).parts):
                continue
            yield path


def _is_bare_pass_except(node: ast.ExceptHandler) -> bool:
    body = [
        stmt for stmt in node.body
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Constant)
        or not isinstance(stmt.value.value, str)  # descarta docstrings/comentarios-como-string
    ]
    return len(body) == 1 and isinstance(body[0], ast.Pass)


_BROAD_NAMES = {"Exception", "BaseException"}


def _is_broad(tipo: ast.expr | None) -> bool:
    """Desnudo, Exception, BaseException (tambien `x.Exception`), o una tupla
    que incluya alguno."""
    if tipo is None:
        return True
    if isinstance(tipo, ast.Tuple):
        return any(_is_broad(e) for e in tipo.elts)
    if isinstance(tipo, ast.Name):
        return tipo.id in _BROAD_NAMES
    if isinstance(tipo, ast.Attribute):
        return tipo.attr in _BROAD_NAMES
    return False


def _corta_el_flujo(node: ast.ExceptHandler) -> bool:
    """El bloque NO sigue de largo: relanza, o termina el proceso.

    CALIBRACIÓN 2026-09-15: antes esto era `_reraises()` y sólo buscaba
    `ast.Raise`, así que marcaba como fail-open cuatro bloques de
    `scripts/manual_motor_v02_integration.py` que imprimen FAIL y hacen
    `sys.exit(1)` —— es decir, que fallan CERRADO, que es justo lo que esta
    prueba quiere. Marcar al que hace lo correcto no es rigor: obliga a
    escribir `# fail-soft:` sobre un bloque que no es fail-soft, o sea a
    meter una afirmación falsa en el código para que el control calle.
    (Medido en el blueprint de agent-dashboard-v3: sin corregir sus falsos
    positivos, el detector marcaba al agente que más aportaba, 90 % de sus
    posts, frente al 55 % del que sí fabricaba.)

    `sys.exit()` levanta `SystemExit`, así que a efectos de flujo es
    equivalente a relanzar: nadie aguas abajo sigue creyendo que todo fue bien.
    """
    for n in ast.walk(node):
        if isinstance(n, ast.Raise):
            return True
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr in {"exit", "_exit"}:
                return True
            if isinstance(f, ast.Name) and f.id == "exit":
                return True
    return False


def violations_in_source(source: str, filename: str = "<sintetico>") -> list[int]:
    """Lineas de los `except` que violan la regla, sobre codigo fuente puro
    (lo usan el escaneo del repo y los casos sinteticos)."""
    lines = source.splitlines()
    tree = ast.parse(source, filename=filename)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not (_is_bare_pass_except(node)
                or (_is_broad(node.type) and not _corta_el_flujo(node))):
            continue
        if any(m in lines[node.lineno - 1] for m in MARCAS):
            continue
        out.append(node.lineno)
    return sorted(out)


def find_fail_open_excepts(files=None) -> list[str]:
    violations = []
    for path in (_iter_python_files() if files is None else files):
        try:
            source = path.read_text(encoding="utf-8")
            lineas = violations_in_source(source, str(path))
        except (SyntaxError, UnicodeDecodeError):
            if _declarado_no_parsea(path):
                continue
            # Fix ronda 1 (2026-09-15): antes `continue` -- un archivo que el
            # escaner no puede leer quedaba sin revisar, en silencio.
            violations.append(f"{path}: no se pudo parsear")
            continue
        violations.extend(f"{path}:{n}" for n in lineas)
    return violations


# --- casos sinteticos puros (Task 3, 2026-09-15) ---------------------------
# Cada uno fija una frontera de la regla sin depender del arbol del repo.

def _src(cuerpo_except: str, tipo: str = "Exception", marca: str = "") -> str:
    cabecera = f"except {tipo}:" if tipo else "except:"
    return (
        "def f():\n"
        "    try:\n"
        "        g()\n"
        f"    {cabecera}{marca}\n"
        f"        {cuerpo_except}\n"
    )


def test_sintetico_except_amplio_que_sigue_de_largo_sin_marca_falla():
    # El caso que encontro el review: no es `pass`, pero se traga el error igual.
    assert violations_in_source(_src("x = None")) == [4]


def test_sintetico_except_amplio_con_marca_pasa():
    assert violations_in_source(_src("x = None", marca="  # fail-soft: razon concreta")) == []


def test_sintetico_tupla_que_incluye_exception_sin_marca_falla():
    assert violations_in_source(_src("x = None", tipo="(ValueError, Exception)")) == [4]


def test_sintetico_tupla_con_baseexception_por_atributo_sin_marca_falla():
    assert violations_in_source(_src("x = None", tipo="(KeyError, builtins.BaseException)")) == [4]


def test_sintetico_except_amplio_que_relanza_pasa():
    assert violations_in_source(_src("log(); raise")) == []


def test_sintetico_relanzar_en_cualquier_punto_del_cuerpo_cuenta():
    src = (
        "def f():\n"
        "    try:\n"
        "        g()\n"
        "    except Exception as e:\n"
        "        if malo(e):\n"
        "            raise RuntimeError('x') from e\n"
        "        x = None\n"
    )
    assert violations_in_source(src) == []


def test_sintetico_except_desnudo_pass_sin_marca_sigue_fallando():
    assert violations_in_source(_src("pass", tipo="")) == [4]


def test_sintetico_except_estrecho_pass_sin_marca_sigue_fallando():
    # Regla vieja intacta: un except-pass de cualquier tipo necesita la marca.
    assert violations_in_source(_src("pass", tipo="ValueError")) == [4]


def test_sintetico_except_estrecho_que_sigue_de_largo_no_es_asunto_de_esta_regla():
    assert violations_in_source(_src("x = None", tipo="ValueError")) == []


def test_sintetico_funcion_anidada_tambien_se_escanea():
    src = (
        "def f():\n"
        "    def g():\n"
        "        try:\n"
        "            h()\n"
        "        except Exception:\n"
        "            return None\n"
        "    return g\n"
    )
    assert violations_in_source(src) == [5]


def test_sintetico_la_marca_en_el_cuerpo_no_alcanza():
    src = (
        "def f():\n"
        "    try:\n"
        "        g()\n"
        "    except Exception:\n"
        "        # fail-soft: en el cuerpo, no en la linea del except\n"
        "        x = None\n"
    )
    assert violations_in_source(src) == [4]


def test_sintetico_except_amplio_que_solo_loguea_sin_marca_falla():
    # Fix ronda 1 (2026-09-15): un _reraises que contara un logger.* como
    # relanzar dejaba verdes todos los demas casos. Loguear no es relanzar.
    assert violations_in_source(_src('logger.warning("x")')) == [4]


def test_sintetico_except_amplio_que_solo_loguea_con_marca_pasa():
    assert violations_in_source(_src('logger.warning("x")', marca="  # fail-soft: razon concreta")) == []


# Medido el 2026-09-15 (fix ronda 1): 155 archivos .py en el escaneo. El piso
# va por debajo a proposito: crecer no rompe; perder un arbol entero si.
PISO_ARCHIVOS_ESCANEADOS = 170


def test_el_escaneo_del_repo_ve_archivos_de_produccion():
    # Guardia del propio control: un escaneo que no ve nada pasa siempre.
    vistos = {p.relative_to(_THIS_REPO_ROOT).as_posix()
              for p in _iter_python_files()
              if _THIS_REPO_ROOT in p.parents}
    for prefijo in ("jacobs/", "las_manos/", "policy/", "tests/", "scripts/"):
        assert any(v.startswith(prefijo) for v in vistos), f"el escaneo no ve nada bajo {prefijo}"
    assert len(vistos) >= PISO_ARCHIVOS_ESCANEADOS, (
        f"el escaneo ve {len(vistos)} archivos (< {PISO_ARCHIVOS_ESCANEADOS}): se excluyo un arbol")


def test_los_roots_ausentes_se_declaran_en_vez_de_desaparecer():
    """Un root que no existe no puede irse en silencio: `rglob` sobre un
    directorio inexistente devuelve vacio y el escaneo pasaria creyendo que
    cubrio ese arbol. Si falta, tiene que estar dicho."""
    assert all(r.is_dir() for r in REPO_ROOTS), (
        f"hay un REPO_ROOT declarado que no existe: {[str(r) for r in REPO_ROOTS if not r.is_dir()]}")
    if ROOTS_AUSENTES:
        # No es un fallo: jax-platform puede no estar al lado (CI de jax solo).
        # Lo que seria un fallo es que nadie lo supiera.
        print(f"AVISO: no se escaneo {[str(r) for r in ROOTS_AUSENTES]} "
              f"(definir JAX_PLATFORM_REPO_ROOT para incluirlo)")


def _entradas_que_sobran(no_parsea: dict[str, str], raiz: Path) -> list[str]:
    """Entradas de _NO_PARSEA que ya no se justifican: el archivo no existe o
    ya parsea. Separado del test para ejercitarlo con archivos de mentira: con
    la tabla vacía (E-01, 2026-09-16) un bucle sobre la tabla real pasa sin
    comprobar nada."""
    sobran = []
    for rel in no_parsea:
        ruta = raiz / rel
        if not ruta.exists():
            sobran.append(f"{rel} ya no existe: retirar la entrada de _NO_PARSEA")
            continue
        try:
            ast.parse(ruta.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        sobran.append(f"{rel} ya parsea: retirar la entrada de _NO_PARSEA")
    return sobran


def test_ninguna_entrada_de_NO_PARSEA_sobra():
    assert _entradas_que_sobran(_NO_PARSEA, _THIS_REPO_ROOT) == []


def test_el_control_de_entradas_sobrantes_se_pone_rojo(tmp_path):
    (tmp_path / "roto.py").write_text("def f(:\n    pass\n")
    (tmp_path / "sano.py").write_text("x = 1\n")
    tabla = {"roto.py": "roto", "sano.py": "ya parsea", "fantasma.py": "no existe"}
    assert _entradas_que_sobran(tabla, tmp_path) == [
        "sano.py ya parsea: retirar la entrada de _NO_PARSEA",
        "fantasma.py ya no existe: retirar la entrada de _NO_PARSEA",
    ]


def test_un_archivo_roto_declarado_no_es_violacion(tmp_path, monkeypatch):
    roto = tmp_path / "roto.py"
    roto.write_text("def f(:\n    pass\n")
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOTS", [tmp_path])
    monkeypatch.setitem(_NO_PARSEA, "roto.py", "roto a propósito para este test")
    assert find_fail_open_excepts(files=[roto]) == []


def test_un_archivo_roto_no_declarado_si_es_violacion(tmp_path):
    roto = tmp_path / "no_declarado.py"
    roto.write_text("def f(:\n    pass\n")
    assert find_fail_open_excepts(files=[roto]) == [f"{roto}: no se pudo parsear"]


def test_un_archivo_que_no_se_puede_parsear_es_violacion(tmp_path):
    # Fix ronda 1 (2026-09-15): antes se salteaba en silencio, y un archivo
    # que el escaner no puede leer nunca se revisaba.
    roto = tmp_path / "roto.py"
    roto.write_text("def f(:\n    pass\n")
    no_utf8 = tmp_path / "latin.py"
    no_utf8.write_bytes(b"x = '\xff\xfe'\n")
    violaciones = find_fail_open_excepts(files=[roto, no_utf8])
    assert violaciones == [f"{roto}: no se pudo parsear", f"{no_utf8}: no se pudo parsear"]


def test_no_fail_open_except() -> None:
    violations = find_fail_open_excepts()
    assert not violations, (
        f"{len(violations)} except (except-pass, o amplio sin relanzar) sin marcar "
        "'# fail-soft: <razón>' en la linea del except:\n"
        + "\n".join(violations)
    )


def main() -> int:
    violations = find_fail_open_excepts()
    if violations:
        print(f"FAIL — {len(violations)} except fail-open sin marca encontrados:")
        for v in violations:
            print(f"  {v}")
        return 1
    print("OK — cero except fail-open sin marca en el codigo fuente escaneado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
