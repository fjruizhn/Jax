#!/usr/bin/env python3
"""TRIPWIRE -- ninguna llamada `aiomysql.connect(...)` NI
`aiomysql.create_pool(...)` del árbol se abre sin `connect_timeout`.

Hallazgo de revisión, Tarea 2b (tanda A, PR-B, hallazgo de la revisión de la
Tarea 6). Ronda de arreglo 1 (2026-09-14): `MotorCatalog.from_db()` abría
`aiomysql.connect(...)` sin `connect_timeout` -- el default de aiomysql
(0.3.2) es `None`, que `asyncio.wait_for()` interpreta como "sin límite": si
la DB se cuelga en vez de rechazar, la conexión espera para siempre. El
arreglo puntual de ese sitio no cerraba el hallazgo real: grepear
`aiomysql.connect(` sobre el árbol completo encontró otros 19 sitios con el
mismo patrón (jax/core/, las_manos/ -- raíz y motor_registry/ -- y jacobs/,
producción y tests/probes por igual).

Ronda de arreglo 2 (2026-09-14): `aiomysql.create_pool(...)` acepta el MISMO
`connect_timeout` (lo reenvía a cada `Connection` que abre) y tiene el MISMO
default `None` -- es el mismo hallazgo, no uno distinto, así que no se
difiere: 3 sitios más (`jax/memory/db.py`, `scripts/migrar_embeddings.py`,
`tests/test_migrar_embeddings_io.py`) y el propio tripwire pasa a vigilar las
dos formas. Arreglar los sitios sin un control que los vigile deja el
próximo (el que se escriba mañana) exactamente tan descubierto como estaba
este.

Es un test de CLASE, no de casos -- mismo criterio que
test_no_blocking_in_async.py: recorre TODOS los .py del árbol (excluyendo
venvs/.git/__pycache__) y falla ante CUALQUIER `aiomysql.connect(...)` o
`aiomysql.create_pool(...)` nuevo sin el kwarg `connect_timeout`, con
archivo:línea. A diferencia de un grep de texto, un AST no confunde un
comentario o un docstring que MENCIONA `aiomysql.connect(`/`create_pool(`
(como el de este mismo archivo, o el de jax/core/db_connect_config.py) con
una llamada real -- por eso el detector propio de este archivo NO aparece en
sus propios hallazgos.

`connect_timeout=<algo>` cuenta como cumplido aunque el valor no sea
`db_connect_timeout_seconds()` -- el helper es la forma RECOMENDADA (ver
jax/core/db_connect_config.py), pero este tripwire vigila el contrato real
(el kwarg wireado, que es lo que aiomysql necesita para no esperar sin
límite), no una implementación particular.

Ronda de arreglo 3 (re-revisión, 2026-09-14), dos huecos Minor cerrados:

1. El detector comparaba `ast.unparse(func)` contra un set fijo de dos
   strings (`"aiomysql.connect"`/`"aiomysql.create_pool"`), así que NO veía
   `from aiomysql import connect` (con o sin `as`), `import aiomysql as db`
   -> `db.connect(`, ni `aiomysql.pool.create_pool(`. Hoy ninguna de esas
   formas existe en el repo (medido), pero un detector que solo reconoce
   la forma literal de hoy no cierra el hallazgo -- lo hace depender de que
   nadie escriba mañana un import distinto. Ahora resuelve alias por
   archivo (`_alias_info`): recorre los `Import`/`ImportFrom` del propio
   árbol ANTES de buscar llamadas, arma qué nombres valen como el módulo
   `aiomysql`, cuáles como el submódulo `aiomysql.pool`, y cuáles como
   `connect`/`create_pool` importados directo -- y con eso reconoce las
   cinco formas.
2. Un archivo que no parsea (`SyntaxError`) se salteaba en silencio
   (`return []`) -- invisible para el tripwire, no vigilado. Ahora hace
   FALLAR el test con su ruta, salvo que esté en `_NO_PARSEA` con el motivo
   por el que no es Python válido y por qué no puede tener un
   `aiomysql.connect`/`create_pool` escondido.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Mismo criterio de exclusión que test_no_blocking_in_async.py: nada de
# venvs, control de versiones ni cachés de bytecode. Sin "docs"/"workspace"
# acá a propósito -- a diferencia del chequeo de bloqueos (que solo le
# importa el código de SERVICIO), un `aiomysql.connect(` sin timeout en un
# script suelto de cualquier lado del árbol se cuelga igual si alguien lo
# corre contra una DB colgada.
EXCLUIDOS = {".git", "__pycache__", ".pytest_cache", "node_modules"}

# Archivos que NO son Python válido -- ast.parse() no los puede leer, así
# que no hay AST que recorrer. Cada entrada es una excepción EXPLÍCITA con
# el motivo, no un salteo silencioso: un archivo roto que no está acá hace
# FALLAR el tripwire (ver test_un_archivo_roto_no_declarado_hace_fallar).
# Vacía desde el 2026-09-16 (E-01): el único no-módulo real del árbol,
# _director_patch/routes_block.py, se retiró. El mecanismo se conserva y se
# ejercita con un .py roto declarado a propósito dentro del test.
_NO_PARSEA: dict[str, str] = {}


def _excluido(path: Path) -> bool:
    for parte in path.parts:
        if parte in EXCLUIDOS:
            return True
        minuscula = parte.lower()
        if "venv" in minuscula or "scratch" in minuscula:
            return True
    return False


def _archivos_py(raiz: Path = RAIZ):
    for p in raiz.rglob("*.py"):
        if _excluido(p):
            continue
        yield p


# Las dos funciones que abren un socket nuevo con aiomysql y aceptan
# connect_timeout: connect() para una conexión suelta, create_pool() para un
# pool (lo reenvía a cada Connection que abre -- mismo default None, mismo
# hallazgo).
_FUNCIONES_VIGILADAS = {"connect", "create_pool"}


def _alias_info(arbol: ast.AST) -> tuple[set[str], set[str], dict[str, str]]:
    """Resuelve, para ESTE archivo, con qué nombres se puede llegar a
    aiomysql.connect/create_pool -- no todo el mundo escribe
    `aiomysql.connect(`.

    Devuelve (alias_del_modulo, alias_del_submodulo_pool, alias_de_funcion):
      - alias_del_modulo: nombres que valen como el módulo `aiomysql`
        entero (`aiomysql` siempre; +1 por cada `import aiomysql as X`).
      - alias_del_submodulo_pool: nombres que valen como `aiomysql.pool`
        (`import aiomysql.pool as X`, `from aiomysql import pool as X`).
      - alias_de_funcion: {nombre: "connect"|"create_pool"} para lo
        importado directo (`from aiomysql import connect`,
        `from aiomysql import create_pool as cp`,
        `from aiomysql.pool import create_pool`).

    Recorre TODO el árbol (no solo el nivel de módulo) a propósito: un
    import dentro de una función (como hacen varios tests de este repo)
    también tiene que resolver -- este es un tripwire heurístico, no un
    resolvedor de scopes."""
    alias_modulo = {"aiomysql"}
    alias_pool = set()
    alias_funcion: dict[str, str] = {}

    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            for alias in nodo.names:
                if alias.name == "aiomysql" and alias.asname:
                    alias_modulo.add(alias.asname)
                elif alias.name == "aiomysql.pool" and alias.asname:
                    alias_pool.add(alias.asname)
        elif isinstance(nodo, ast.ImportFrom):
            if nodo.module == "aiomysql":
                for alias in nodo.names:
                    if alias.name in _FUNCIONES_VIGILADAS:
                        alias_funcion[alias.asname or alias.name] = alias.name
                    elif alias.name == "pool":
                        alias_pool.add(alias.asname or "pool")
            elif nodo.module == "aiomysql.pool":
                for alias in nodo.names:
                    if alias.name == "create_pool":
                        alias_funcion[alias.asname or alias.name] = "create_pool"

    return alias_modulo, alias_pool, alias_funcion


def _funcion_vigilada(nodo_func: ast.expr, alias_modulo: set[str],
                      alias_pool: set[str], alias_funcion: dict[str, str]) -> str | None:
    """Si `nodo_func` (el `.func` de un ast.Call) es una llamada a
    connect/create_pool de aiomysql por CUALQUIERA de las formas de import
    que este archivo resuelve, devuelve cuál ("connect"/"create_pool");
    si no, None."""
    if isinstance(nodo_func, ast.Name):
        # connect(...) / create_pool(...) / c(...) -- bare, vía `from
        # aiomysql import connect [as c]`.
        return alias_funcion.get(nodo_func.id)

    if isinstance(nodo_func, ast.Attribute) and nodo_func.attr in _FUNCIONES_VIGILADAS:
        base = nodo_func.value
        if isinstance(base, ast.Name):
            # aiomysql.connect(...) / db.connect(...) (import aiomysql as db)
            if base.id in alias_modulo:
                return nodo_func.attr
            # ap.create_pool(...) (import aiomysql.pool as ap)
            if base.id in alias_pool and nodo_func.attr == "create_pool":
                return nodo_func.attr
        elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
            # aiomysql.pool.create_pool(...)
            if (base.value.id in alias_modulo and base.attr == "pool"
                    and nodo_func.attr == "create_pool"):
                return nodo_func.attr
    return None


def _hallazgos_en(path: Path, raiz: Path = RAIZ) -> list[str]:
    fuente = path.read_text(encoding="utf-8", errors="replace")
    try:
        arbol = ast.parse(fuente)
    except SyntaxError:
        rel = str(path.relative_to(raiz))
        if rel in _NO_PARSEA:
            return []
        return [
            f"{rel}: no parsea (SyntaxError) -- si no es Python válido a "
            "propósito, agregalo a _NO_PARSEA con el motivo; si es un bug, "
            "arreglalo. No se saltea en silencio."
        ]

    alias_modulo, alias_pool, alias_funcion = _alias_info(arbol)
    encontrados = []

    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        forma = _funcion_vigilada(nodo.func, alias_modulo, alias_pool, alias_funcion)
        if forma is None:
            continue
        try:
            nombre = ast.unparse(nodo.func)
        except Exception:  # fail-soft: solo afecta al NOMBRE que se imprime en el hallazgo; el hallazgo se reporta igual con `forma`
            nombre = forma
        tiene_timeout = any(kw.arg == "connect_timeout" for kw in nodo.keywords)
        if not tiene_timeout:
            encontrados.append(f"{path.relative_to(raiz)}:{nodo.lineno}: {nombre}() sin connect_timeout")
    return encontrados


@contextmanager
def _temporal_fuera_del_checkout():
    """Un .py sintético para los controles, dentro de un directorio temporal
    FUERA del checkout: una corrida matada o dos en paralelo no dejan un
    archivo roto a la vista del escaneo del árbol completo. Se escanea con
    `_hallazgos_en(ruta, ruta.parent)`."""
    with tempfile.TemporaryDirectory(prefix="jax-tripwire-") as directorio:
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=directorio, delete=True) as fh:
            yield fh


class AiomysqlConnectTimeoutTripwireTest(unittest.TestCase):
    def test_ningun_aiomysql_connect_sin_connect_timeout(self):
        hallazgos = []
        for p in _archivos_py():
            hallazgos.extend(_hallazgos_en(p))
        self.assertEqual(
            hallazgos, [],
            "aiomysql.connect()/create_pool() sin connect_timeout (sin "
            "límite si la DB se cuelga en vez de rechazar -- ver "
            "jax/core/db_connect_config.py):\n  " + "\n  ".join(hallazgos),
        )

    def test_el_detector_reconoce_un_sitio_sin_connect_timeout(self):
        """El freno se ejercita: si el detector no atrapa un caso obvio, el
        verde del test de arriba no significa nada (un control que no falla
        no valida nada)."""
        codigo = (
            "import aiomysql\n"
            "async def f():\n"
            "    return await aiomysql.connect(host='x', port=3306)\n"
        )
        with _temporal_fuera_del_checkout() as fh:
            fh.write(codigo)
            fh.flush()
            hallazgos = _hallazgos_en(Path(fh.name), Path(fh.name).parent)
        self.assertEqual(len(hallazgos), 1, f"el detector no vio el sitio sin connect_timeout: {hallazgos}")

    def test_el_detector_reconoce_un_create_pool_sin_connect_timeout(self):
        """Mismo control que arriba, para la otra forma vigilada
        (create_pool) -- connect() y create_pool() son detectores
        independientes en el código, así que uno solo no prueba el otro."""
        codigo = (
            "import aiomysql\n"
            "async def f():\n"
            "    return await aiomysql.create_pool(host='x', port=3306, minsize=1, maxsize=5)\n"
        )
        with _temporal_fuera_del_checkout() as fh:
            fh.write(codigo)
            fh.flush()
            hallazgos = _hallazgos_en(Path(fh.name), Path(fh.name).parent)
        self.assertEqual(len(hallazgos), 1, f"el detector no vio el create_pool sin connect_timeout: {hallazgos}")

    # -----------------------------------------------------------------
    # Ronda de arreglo 3: formas de import alternativas. Cada una es su
    # propio archivo sintético -- si el resolvedor de alias se rompe para
    # una forma, esta es la única que lo nota (las demás pasan igual).
    # -----------------------------------------------------------------

    def test_el_detector_ve_from_aiomysql_import_connect_sin_alias(self):
        codigo = (
            "from aiomysql import connect\n"
            "async def f():\n"
            "    return await connect(host='x', port=3306)\n"
        )
        with _temporal_fuera_del_checkout() as fh:
            fh.write(codigo)
            fh.flush()
            hallazgos = _hallazgos_en(Path(fh.name), Path(fh.name).parent)
        self.assertEqual(len(hallazgos), 1, f"no vio `from aiomysql import connect` + connect(): {hallazgos}")

    def test_el_detector_ve_from_aiomysql_import_connect_con_alias(self):
        codigo = (
            "from aiomysql import connect as c\n"
            "async def f():\n"
            "    return await c(host='x', port=3306)\n"
        )
        with _temporal_fuera_del_checkout() as fh:
            fh.write(codigo)
            fh.flush()
            hallazgos = _hallazgos_en(Path(fh.name), Path(fh.name).parent)
        self.assertEqual(len(hallazgos), 1, f"no vio `from aiomysql import connect as c` + c(): {hallazgos}")

    def test_el_detector_ve_from_aiomysql_import_create_pool(self):
        codigo = (
            "from aiomysql import create_pool\n"
            "async def f():\n"
            "    return await create_pool(host='x', port=3306, minsize=1)\n"
        )
        with _temporal_fuera_del_checkout() as fh:
            fh.write(codigo)
            fh.flush()
            hallazgos = _hallazgos_en(Path(fh.name), Path(fh.name).parent)
        self.assertEqual(len(hallazgos), 1, f"no vio `from aiomysql import create_pool`: {hallazgos}")

    def test_el_detector_ve_import_aiomysql_as_alias(self):
        codigo = (
            "import aiomysql as db\n"
            "async def f():\n"
            "    return await db.connect(host='x', port=3306)\n"
        )
        with _temporal_fuera_del_checkout() as fh:
            fh.write(codigo)
            fh.flush()
            hallazgos = _hallazgos_en(Path(fh.name), Path(fh.name).parent)
        self.assertEqual(len(hallazgos), 1, f"no vio `import aiomysql as db` + db.connect(): {hallazgos}")

    def test_el_detector_ve_aiomysql_pool_create_pool(self):
        codigo = (
            "import aiomysql\n"
            "async def f():\n"
            "    return await aiomysql.pool.create_pool(host='x', port=3306, minsize=1)\n"
        )
        with _temporal_fuera_del_checkout() as fh:
            fh.write(codigo)
            fh.flush()
            hallazgos = _hallazgos_en(Path(fh.name), Path(fh.name).parent)
        self.assertEqual(len(hallazgos), 1, f"no vio `aiomysql.pool.create_pool(`: {hallazgos}")

    def test_el_detector_ve_import_aiomysql_pool_as_alias(self):
        codigo = (
            "import aiomysql.pool as ap\n"
            "async def f():\n"
            "    return await ap.create_pool(host='x', port=3306, minsize=1)\n"
        )
        with _temporal_fuera_del_checkout() as fh:
            fh.write(codigo)
            fh.flush()
            hallazgos = _hallazgos_en(Path(fh.name), Path(fh.name).parent)
        self.assertEqual(len(hallazgos), 1, f"no vio `import aiomysql.pool as ap` + ap.create_pool(): {hallazgos}")

    def test_el_detector_acepta_las_formas_alternativas_con_connect_timeout(self):
        """Control positivo de las formas de arriba: con connect_timeout,
        ninguna reporta -- el resolvedor de alias no infla falsos positivos."""
        codigo = (
            "from aiomysql import connect as c, create_pool\n"
            "import aiomysql as db\n"
            "async def f():\n"
            "    await c(host='x', port=3306, connect_timeout=10)\n"
            "    await create_pool(host='x', port=3306, minsize=1, connect_timeout=10)\n"
            "    await db.connect(host='x', port=3306, connect_timeout=10)\n"
        )
        with _temporal_fuera_del_checkout() as fh:
            fh.write(codigo)
            fh.flush()
            hallazgos = _hallazgos_en(Path(fh.name), Path(fh.name).parent)
        self.assertEqual(hallazgos, [], f"formas alternativas CON connect_timeout no deberían reportarse: {hallazgos}")

    def test_el_detector_no_confunde_texto_con_una_llamada_real(self):
        """Un comentario o docstring que MENCIONA `aiomysql.connect(`/
        `create_pool(` (como el de este propio archivo) no es una llamada --
        el AST no lo ve como ast.Call, a diferencia de un grep de texto."""
        codigo = (
            '"""Este módulo habla de aiomysql.connect( y de '
            'aiomysql.create_pool( en un docstring, no los llama."""\n'
            "# aiomysql.connect( también en un comentario.\n"
            "x = 1\n"
        )
        with _temporal_fuera_del_checkout() as fh:
            fh.write(codigo)
            fh.flush()
            hallazgos = _hallazgos_en(Path(fh.name), Path(fh.name).parent)
        self.assertEqual(hallazgos, [], f"texto confundido con una llamada real: {hallazgos}")

    def test_el_detector_acepta_connect_timeout_explicito(self):
        """Control positivo: un sitio QUE SÍ pasa connect_timeout (las dos
        formas) no aparece entre los hallazgos."""
        codigo = (
            "import aiomysql\n"
            "async def f():\n"
            "    await aiomysql.connect(host='x', port=3306, connect_timeout=10)\n"
            "    await aiomysql.create_pool(host='x', port=3306, minsize=1, connect_timeout=10)\n"
        )
        with _temporal_fuera_del_checkout() as fh:
            fh.write(codigo)
            fh.flush()
            hallazgos = _hallazgos_en(Path(fh.name), Path(fh.name).parent)
        self.assertEqual(hallazgos, [], f"un sitio con connect_timeout no debería reportarse: {hallazgos}")

    # -----------------------------------------------------------------
    # Ronda de arreglo 3: un archivo que no parsea no puede quedar invisible.
    # -----------------------------------------------------------------

    def test_un_archivo_roto_no_declarado_hace_fallar_el_tripwire(self):
        """Un SyntaxError en un archivo que NO está en _NO_PARSEA tiene que
        aparecer como hallazgo (con su ruta), no saltearse en silencio --
        ese era el hueco Minor de la re-revisión."""
        codigo = "def f(:\n    pasa\n"  # SyntaxError a propósito
        with _temporal_fuera_del_checkout() as fh:
            fh.write(codigo)
            fh.flush()
            ruta = Path(fh.name)
            self.assertNotIn(str(ruta.relative_to(ruta.parent)), _NO_PARSEA)
            hallazgos = _hallazgos_en(ruta, ruta.parent)
        self.assertEqual(len(hallazgos), 1, f"un archivo roto no declarado debería reportarse: {hallazgos}")
        self.assertIn("no parsea", hallazgos[0])

    def test_un_archivo_roto_declarado_en_NO_PARSEA_no_rompe_la_corrida(self):
        """Control de la excepción explícita. Hasta el 2026-09-16 se probaba con
        _director_patch/routes_block.py; se retiró (E-01) y la tabla quedó
        vacía, así que el camino se ejercita con un .py roto declarado acá."""
        from unittest.mock import patch
        with _temporal_fuera_del_checkout() as fh:
            fh.write("def f(:\n    pasa\n")
            fh.flush()
            ruta = Path(fh.name)
            rel = str(ruta.relative_to(ruta.parent))
            with patch.dict(_NO_PARSEA, {rel: "roto a propósito para este test"}):
                self.assertEqual(_hallazgos_en(ruta, ruta.parent), [])
            self.assertEqual(len(_hallazgos_en(ruta, ruta.parent)), 1, "sin la declaración tiene que reportarse")


if __name__ == "__main__":
    unittest.main()


class ControlesFueraDelCheckoutTest(unittest.TestCase):
    """Revisión final del frente E: los controles escribían su .py sintético
    (a veces roto a propósito) en la raíz del checkout. Una corrida matada a
    mitad de camino, o dos corridas en paralelo, dejaban ese archivo a la vista
    del escaneo del árbol completo y todos los tripwires se ponían rojos. El
    escaneo recibe la raíz inyectada y los controles escriben en un directorio
    temporal fuera del checkout."""

    def test_los_controles_no_escriben_en_el_checkout(self):
        arbol = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        ofensores = [
            n.value.lineno for n in ast.walk(arbol)
            if isinstance(n, ast.keyword) and n.arg == "dir"
            and isinstance(n.value, ast.Name) and n.value.id == "RAIZ"
        ]
        self.assertEqual(ofensores, [], f"un control escribe dentro del checkout (dir=RAIZ) en las líneas {ofensores}")

    def test_la_raiz_del_escaneo_se_inyecta(self):
        with tempfile.TemporaryDirectory(prefix="jax-tripwire-") as d:
            ruta = Path(d) / "roto.py"
            ruta.write_text("def f(:\n    pass\n", encoding="utf-8")
            hallazgos = _hallazgos_en(ruta, Path(d))
        self.assertEqual(len(hallazgos), 1, hallazgos)
        self.assertTrue(hallazgos[0].startswith("roto.py"), hallazgos)
