"""Whitelist derivada por job, nunca escrita a mano (anti-fabricación, capa 2).

QUÉ ES. El conjunto de nombres que EXISTEN en este momento: tablas, columnas y
pares `tabla.columna` de `information_schema` (la base de la conexión), rutas de
`git ls-files -s` del repo de trabajo, funciones/palabras/motores/charsets de
MariaDB (`information_schema.SQL_FUNCTIONS`, `KEYWORDS`, `ENGINES`,
`CHARACTER_SETS`) y las constantes/funciones/clases del código, por `ast` sobre
los `.py` versionados. Opcionales: rutas de la app LAS MANOS (en proceso),
claves de las tablas capability/facet/motor + herramientas, y los `model_id` de
la tabla `model`. Fuera de v1: systemd y las rutas de jax-platform.

OBLIGATORIAS vs OPCIONALES. Si una fuente obligatoria (huella, esquema, git)
no responde, `derivar` levanta `DetectorNoDisponible`: JAMÁS devuelve una lista
vacía, porque una whitelist vacía marcaría TODO (o, peor, un detector que cae
en silencio dejaría pasar todo). Una fuente opcional caída se declara en
`Whitelist.no_disponibles` y el verificador responde `no_verificable` para esa
categoría: visible, no pasa limpio, no acusa.

CACHÉ (LAS CUATRO DEL RENDIMIENTO, regla 2: toda caché declara su invalidación
en el mismo commit que la crea).
  * Qué se cachea: la `Whitelist` ya armada, en un memo de 4 entradas por
    proceso (`_MEMO`, LRU). Lo caro es leer `information_schema.COLUMNS` entero
    y recorrer `git ls-files`; lo barato es la clave.
  * Clave: (huella de esquema, SHA de HEAD, hash de `git status --porcelain`).
    La huella es `MD5(GROUP_CONCAT(... ORDER BY ...))` sobre `COLUMNS`, con
    `group_concat_max_len` fijado en la sesión (el default de 1 MiB truncaría
    en silencio y daría la misma huella con esquemas distintos; si el resultado
    llega al tope, se falla en vez de aceptar una huella truncada).
  * Invalidación: SIN TTL. La clave se recalcula y se compara en CADA uso, así
    que un cambio de esquema, un commit o un archivo sucio invalida de inmediato
    y TAMBIÉN entre procesos (cada proceso compara contra el estado real, no
    contra otro proceso). Lo que cuesta cada uso es una consulta agregada
    (EXPLAIN: `type=ALL` con `key=TABLE_SCHEMA`, "Scanned 1 database", sin
    filesort ni temporary; ~750 columnas) más `git rev-parse` y `git status`.
  * Caché de `ast`: por SHA de blob (`_CACHE_AST`). Un blob con el mismo SHA
    tiene el mismo contenido: no hay nada que invalidar. Tope blando: si pasa de
    `_TOPE_CACHE_AST` entradas se vacía entera (se repuebla en el siguiente job).
  * Si el tenant o el usuario cambiaran lo cacheado, la clave los incluiría; no
    lo hacen: esquema y repo son del despliegue, no de quien pide.

Nada bloqueante en el camino async: `git` y `ast` van en `asyncio.to_thread`; la
base es aiomysql.
"""
from __future__ import annotations

import asyncio
import ast
import builtins
import hashlib
import keyword
import logging
import os
import re
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Iterable

try:
    # LAS MANOS / Jacobs (cwd las_manos/, jax.core no es importable ahí).
    from facet_resolver import _db_conn
except ImportError:  # pragma: no cover - según desde dónde se importe
    # Scripts y REPL (PYTHONPATH=. desde la raíz del repo).
    from jax.core.facet_resolver import _db_conn

logger = logging.getLogger(__name__)

#: Entradas del memo de whitelists por proceso (spec: 4).
MAX_MEMO = 4
#: `group_concat_max_len` de la sesión de la huella (el default, 1 MiB, trunca).
GROUP_CONCAT_MAX_LEN = 16 * 1024 * 1024
#: Tope blando de la caché de `ast`.
_TOPE_CACHE_AST = 20000
#: Un .py más grande que esto no se parsea (no es código escrito a mano).
_MAX_BYTES_PY = 2 * 1024 * 1024
_TIMEOUT_GIT_S = 30

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

SQL_HUELLA = (
    "SELECT MD5(GROUP_CONCAT(CONCAT_WS('.', TABLE_NAME, COLUMN_NAME, COLUMN_TYPE) "
    "ORDER BY TABLE_NAME, ORDINAL_POSITION SEPARATOR ',')), COUNT(*), "
    "LENGTH(GROUP_CONCAT(CONCAT_WS('.', TABLE_NAME, COLUMN_NAME, COLUMN_TYPE) "
    "ORDER BY TABLE_NAME, ORDINAL_POSITION SEPARATOR ',')) "
    "FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE()"
)
SQL_COLUMNAS = ("SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE()")
SQL_FUNCIONES = "SELECT FUNCTION FROM information_schema.SQL_FUNCTIONS"
SQL_PALABRAS = "SELECT WORD FROM information_schema.KEYWORDS"
SQL_MOTORES = "SELECT ENGINE FROM information_schema.ENGINES"
SQL_CHARSETS = "SELECT CHARACTER_SET_NAME FROM information_schema.CHARACTER_SETS"
SQL_CLAVES = ("SELECT `key` FROM capability UNION SELECT `key` FROM facet "
              "UNION SELECT `key` FROM motor")
SQL_MODELOS = "SELECT model_id FROM model"


class DetectorNoDisponible(RuntimeError):
    """Una fuente OBLIGATORIA no respondió: no hay whitelist, nunca una vacía."""


@dataclass(frozen=True)
class EsquemaDB:
    columnas: tuple[tuple[str, str], ...] = ()
    funciones: frozenset[str] = frozenset()
    palabras: frozenset[str] = frozenset()
    motores: frozenset[str] = frozenset()
    charsets: frozenset[str] = frozenset()


@dataclass(frozen=True)
class EstadoGit:
    head: str = ""
    status_hash: str = ""
    entradas: tuple[tuple[str, str], ...] = ()  # (sha del blob, ruta)


@dataclass(frozen=True)
class Fuentes:
    """Las fuentes de una derivación. Inyectables: los tests no tocan DB ni git."""
    huella: Callable[[], Awaitable[str]]
    esquema: Callable[[], Awaitable[EsquemaDB]]
    git: Callable[[], Awaitable[EstadoGit]]
    simbolos: Callable[[EstadoGit], Awaitable[frozenset[str]]] | None = None
    rutas_app: Callable[[], Awaitable[frozenset[str]]] | None = None
    claves: Callable[[], Awaitable[frozenset[str]]] | None = None
    modelos: Callable[[], Awaitable[frozenset[str]]] | None = None


@dataclass(frozen=True)
class Whitelist:
    tablas: frozenset[str] = frozenset()
    columnas: frozenset[str] = frozenset()
    pares: frozenset[str] = frozenset()
    archivos: frozenset[str] = frozenset()
    directorios: frozenset[str] = frozenset()
    simbolos: frozenset[str] = frozenset()
    sql: frozenset[str] = frozenset()
    rutas_app: frozenset[str] = frozenset()
    claves: frozenset[str] = frozenset()
    modelos: frozenset[str] = frozenset()
    raices_fs: frozenset[str] = frozenset()
    no_disponibles: frozenset[str] = field(default_factory=frozenset)
    clave_cache: tuple[str, str, str] = ("", "", "")


_MEMO: "OrderedDict[tuple[str, str, str], Whitelist]" = OrderedDict()
_CACHE_AST: dict[str, frozenset[str]] = {}


# --------------------------------------------------------------------------
# git (síncrono: se llama con to_thread)
# --------------------------------------------------------------------------

def _git(raiz: Path, *args: str) -> bytes:
    try:
        r = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(raiz), *args],
            capture_output=True, timeout=_TIMEOUT_GIT_S,
            # status sin lock de índice: no compite con el commit del job
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"},
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise DetectorNoDisponible(f"git {args[0]} no respondió en {raiz}: {exc}") from exc
    if r.returncode != 0:
        raise DetectorNoDisponible(
            f"git {args[0]} falló en {raiz} (rc={r.returncode}): {r.stderr.decode('utf-8', 'replace')[:200]}")
    return r.stdout


def leer_git(raiz) -> EstadoGit:
    """HEAD, hash de `git status --porcelain` y `git ls-files -s` de `raiz`."""
    raiz = Path(raiz)
    _git(raiz, "rev-parse", "--git-dir")  # no es un repo -> DetectorNoDisponible
    try:
        head = _git(raiz, "rev-parse", "HEAD").decode().strip()
    except DetectorNoDisponible:
        head = "(sin commits)"  # repo recién creado: válido, con lista de archivos vacía o solo del índice
    estado = hashlib.sha1(_git(raiz, "status", "--porcelain", "-z")).hexdigest()
    entradas = []
    for registro in _git(raiz, "ls-files", "-s", "-z").split(b"\0"):
        if not registro:
            continue
        meta, _, ruta = registro.partition(b"\t")
        partes = meta.split()
        if len(partes) != 3 or not ruta:
            raise DetectorNoDisponible(f"salida inesperada de git ls-files -s: {registro[:80]!r}")
        entradas.append((partes[1].decode(), ruta.decode("utf-8", "replace")))
    return EstadoGit(head=head, status_hash=estado, entradas=tuple(entradas))


# --------------------------------------------------------------------------
# ast (síncrono: se llama con to_thread)
# --------------------------------------------------------------------------

def _nombres_de(arbol: ast.AST) -> frozenset[str]:
    """Todo identificador que el código nombra: definiciones, usos, atributos,
    parámetros, imports y cadenas con forma de identificador (claves de dict,
    nombres de variables de entorno). Un detector de «existe en el código»."""
    nombres: set[str] = set()
    for n in ast.walk(arbol):
        if isinstance(n, ast.Name):
            nombres.add(n.id)
        elif isinstance(n, ast.Attribute):
            nombres.add(n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nombres.add(n.name)
        elif isinstance(n, ast.arg):
            nombres.add(n.arg)
        elif isinstance(n, ast.keyword) and n.arg:
            nombres.add(n.arg)
        elif isinstance(n, ast.alias):
            nombres.update(p for p in (n.name.split(".") + [n.asname or ""]) if p)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            nombres.add(n.name)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) <= 80 \
                and _IDENT.match(n.value):
            nombres.add(n.value)
    return frozenset(nombres)


def simbolos_de_archivos(raiz, entradas: Iterable[tuple[str, str]]) -> frozenset[str]:
    """Símbolos de los `.py` de `entradas` [(sha de blob, ruta)], con caché por
    SHA de blob. Un archivo con error de sintaxis no aporta símbolos (se cachea:
    el blob no cambia); uno ilegible por el sistema de archivos no se cachea."""
    raiz = Path(raiz)
    if len(_CACHE_AST) > _TOPE_CACHE_AST:
        _CACHE_AST.clear()
    total: set[str] = set()
    for sha, ruta in entradas:
        if not ruta.endswith(".py"):
            continue
        hit = _CACHE_AST.get(sha)
        if hit is None:
            try:
                p = raiz / ruta
                if p.stat().st_size > _MAX_BYTES_PY:
                    hit = frozenset()
                else:
                    hit = _nombres_de(ast.parse(p.read_text(encoding="utf-8")))
            except (SyntaxError, ValueError, UnicodeDecodeError):
                hit = frozenset()
            except OSError:  # fail-soft: archivo borrado entre ls-files y la lectura; no se cachea, el próximo job lo reintenta
                continue
            _CACHE_AST[sha] = hit
        total |= hit
    return frozenset(total)


def _simbolos_del_lenguaje() -> frozenset[str]:
    return frozenset(dir(builtins)) | frozenset(keyword.kwlist) | frozenset(sys.stdlib_module_names)


# --------------------------------------------------------------------------
# fuentes por defecto (DB aiomysql, git, ast)
# --------------------------------------------------------------------------

async def _consulta(sql: str, *, antes: tuple[str, ...] = ()) -> list[tuple]:
    conn = await _db_conn()
    try:
        async with conn.cursor() as cur:
            for previa in antes:
                await cur.execute(previa)
            await cur.execute(sql)
            return list(await cur.fetchall())
    finally:
        conn.close()


async def _huella_db() -> str:
    filas = await _consulta(SQL_HUELLA, antes=(f"SET SESSION group_concat_max_len = {int(GROUP_CONCAT_MAX_LEN)}",))
    huella, cuenta, largo = filas[0]
    if not cuenta or not huella:
        raise DetectorNoDisponible("information_schema.COLUMNS no devolvió columnas: esquema vacío o sin base")
    if largo >= GROUP_CONCAT_MAX_LEN:
        raise DetectorNoDisponible("la huella de esquema llegó al tope de group_concat_max_len: truncada")
    return str(huella)


async def _esquema_db() -> EsquemaDB:
    columnas = await _consulta(SQL_COLUMNAS)
    if not columnas:
        raise DetectorNoDisponible("information_schema.COLUMNS no devolvió columnas: esquema vacío o sin base")

    def unicos(filas):
        return frozenset(str(f[0]) for f in filas if f[0])

    return EsquemaDB(
        columnas=tuple((str(t), str(c)) for t, c in columnas),
        funciones=unicos(await _consulta(SQL_FUNCIONES)),
        palabras=unicos(await _consulta(SQL_PALABRAS)),
        motores=unicos(await _consulta(SQL_MOTORES)),
        charsets=unicos(await _consulta(SQL_CHARSETS)),
    )


async def _claves_db() -> frozenset[str]:
    return frozenset(str(f[0]) for f in await _consulta(SQL_CLAVES) if f[0])


async def _modelos_db() -> frozenset[str]:
    return frozenset(str(f[0]) for f in await _consulta(SQL_MODELOS) if f[0])


async def _rutas_de_la_app() -> frozenset[str]:
    """Rutas de la app LAS MANOS que ya está cargada en ESTE proceso
    (`server.app`). Sin la app cargada no hay fuente: opcional caída."""
    modulo = sys.modules.get("server")
    app = getattr(modulo, "app", None)
    rutas = getattr(app, "routes", None)
    if not rutas:
        raise LookupError("la app LAS MANOS no está cargada en este proceso")
    return frozenset(r.path for r in rutas if isinstance(getattr(r, "path", None), str))


def fuentes_por_defecto(raiz_repo, rutas_app=None, herramientas: Iterable[str] = ()) -> Fuentes:
    """Fuentes reales: base por aiomysql, git y ast en hilo. `herramientas` son
    los nombres de `TOOLS_CATALOG`, que el worker conoce y este paquete no puede
    importar (no es importable desde ambos lados)."""
    raiz = Path(raiz_repo)
    herramientas = frozenset(herramientas)

    async def git() -> EstadoGit:
        return await asyncio.to_thread(leer_git, raiz)

    async def simbolos(estado: EstadoGit) -> frozenset[str]:
        return await asyncio.to_thread(simbolos_de_archivos, raiz, estado.entradas)

    async def claves() -> frozenset[str]:
        return herramientas | await _claves_db()

    return Fuentes(
        huella=_huella_db, esquema=_esquema_db, git=git, simbolos=simbolos,
        rutas_app=rutas_app or _rutas_de_la_app, claves=claves, modelos=_modelos_db,
    )


# --------------------------------------------------------------------------
# derivar
# --------------------------------------------------------------------------

def _directorios(rutas: Iterable[str]) -> frozenset[str]:
    dirs: set[str] = set()
    for r in rutas:
        partes = r.split("/")[:-1]
        for i in range(1, len(partes) + 1):
            dirs.add("/".join(partes[:i]))
    return frozenset(dirs)


def _raices_fs() -> frozenset[str]:
    return frozenset(os.listdir("/"))


async def derivar(fuentes: Fuentes) -> Whitelist:
    """Whitelist vigente. Obligatorias caídas -> DetectorNoDisponible."""
    try:
        huella = await fuentes.huella()
        git = await fuentes.git()
    except DetectorNoDisponible:
        raise
    except Exception as exc:  # fail-closed: sin la clave no se sirve ni el memo; se re-levanta como DetectorNoDisponible
        raise DetectorNoDisponible(f"fuente obligatoria caída (huella/git): {type(exc).__name__}: {exc}") from exc

    clave = (huella, git.head, git.status_hash)
    memo = _MEMO.get(clave)
    if memo is not None:
        _MEMO.move_to_end(clave)
        return memo

    try:
        esquema = await fuentes.esquema()
        simbolos = await fuentes.simbolos(git) if fuentes.simbolos else frozenset()
    except DetectorNoDisponible:
        raise
    except Exception as exc:  # fail-closed: esquema/símbolos son obligatorios; nunca una whitelist vacía
        raise DetectorNoDisponible(f"fuente obligatoria caída (esquema): {type(exc).__name__}: {exc}") from exc

    no_disponibles: set[str] = set()
    opcionales: dict[str, frozenset[str]] = {}
    for nombre in ("rutas_app", "claves", "modelos"):
        fuente = getattr(fuentes, nombre)
        if fuente is None:
            no_disponibles.add(nombre)
            continue
        try:
            opcionales[nombre] = frozenset(await fuente())
        except Exception as exc:  # fail-soft DECLARADO: la categoría pasa a no_verificable, visible en el veredicto
            logger.warning("anti_fabricacion: fuente opcional %s caída: %s: %s", nombre, type(exc).__name__, exc)
            no_disponibles.add(nombre)
    try:
        raices = await asyncio.to_thread(_raices_fs)
    except OSError:  # fail-soft DECLARADO: sin raíces del sistema no se distingue una ruta de la app de una del sistema
        raices = frozenset()
        no_disponibles.add("raices_fs")

    archivos = frozenset(ruta for _sha, ruta in git.entradas)
    wl = Whitelist(
        tablas=frozenset(t for t, _c in esquema.columnas),
        columnas=frozenset(c for _t, c in esquema.columnas),
        pares=frozenset(f"{t}.{c}" for t, c in esquema.columnas),
        archivos=archivos,
        directorios=_directorios(archivos),
        simbolos=frozenset(simbolos) | _simbolos_del_lenguaje(),
        sql=esquema.funciones | esquema.palabras | esquema.motores | esquema.charsets,
        rutas_app=opcionales.get("rutas_app", frozenset()),
        claves=opcionales.get("claves", frozenset()),
        modelos=opcionales.get("modelos", frozenset()),
        raices_fs=raices,
        no_disponibles=frozenset(no_disponibles),
        clave_cache=clave,
    )
    _MEMO[clave] = wl
    while len(_MEMO) > MAX_MEMO:
        _MEMO.popitem(last=False)
    return wl
