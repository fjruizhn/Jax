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
  * Consultas opcionales (EXPLAIN 2026-10-22, MariaDB 12.3): claves de
    capability/facet/motor y `model_id` son lecturas completas de tablas de 4 a
    269 filas, sin WHERE ni ORDER BY; el modo se lee por clave primaria
    (`const table`). Corren solo cuando la clave del memo cambia.
  * Raíces (Enmienda 1): workspace = `git ls-files` + árbol en disco (mismo encierro
    de tool_authority: nada fuera de la raíz, sin seguir enlaces; EXCLUSIONES; tope de
    TOPE_ENTRADAS, y si se pasa la categoría `archivos` es no_verificable). Los no
    versionados aportan SOLO la ruta; los símbolos (ast) salen solo de versionados. El
    checkout desplegado de jax es fuente opcional: solo `git ls-files`. La huella de
    `git status` descarta lo que está bajo EXCLUSIONES (el daemon de Ruflo escribe en
    `.claude-flow/` sin parar y antes invalidaba el memo en cada job), y usa
    `--ignored` para que un archivo nuevo ignorado por git también invalide. El SHA del
    checkout de jax no está en la clave: cambia con un despliegue, que reinicia el proceso.
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

#: Enmienda 1 (2026-10-22), regla 1: UNA constante de exclusiones, compartida por el
#: recorrido del árbol en disco y por la huella de `git status` (regla 6). Son
#: directorios de estado/dependencias que cambian solos (el daemon de Ruflo escribe
#: en `.claude-flow/` sin parar) y no son afirmaciones sobre el trabajo.
EXCLUSIONES = frozenset({".git", ".claude-flow", "node_modules", ".venv", "__pycache__"})
#: Tope de entradas del recorrido en disco. Si se pasa, la categoría `archivos` es
#: `no_verificable`: nunca se da por limpio un árbol que no se pudo ver entero.
TOPE_ENTRADAS = 20000
#: Checkout desplegado de jax (fuente OPCIONAL de rutas). Se lee con `git ls-files`,
#: sin recorrer el disco. Lo fija el worker (con override por entorno).
JAX_CHECKOUT_POR_DEFECTO = "/srv/jax-prod/jax"

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
    #: árbol en disco del workspace: (rutas relativas, truncado por el tope). None = no se pidió.
    disco: Callable[[], Awaitable[tuple[frozenset[str], bool]]] | None = None
    #: `git ls-files` del checkout desplegado de jax. None = no se pidió.
    codigo_jax: Callable[[], Awaitable[frozenset[str]]] | None = None


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


def _excluido(ruta: str) -> bool:
    return any(parte in EXCLUSIONES for parte in ruta.split("/"))


def _estado_sin_exclusiones(salida: bytes) -> bytes:
    """`git status --porcelain -z` sin las rutas bajo EXCLUSIONES (regla 6): lo que
    cambia dentro de `.claude-flow/` no invalida el memo. Un renombre/copia trae un
    registro extra (la ruta de origen) que se descarta junto con el suyo."""
    registros = salida.split(b"\0")
    kept, i = [], 0
    while i < len(registros):
        r = registros[i]
        i += 1
        if not r:
            continue
        ruta = r[3:].decode("utf-8", "replace")
        origen = None
        if r[:1] in (b"R", b"C") or r[1:2] in (b"R", b"C"):
            origen = registros[i].decode("utf-8", "replace") if i < len(registros) else ""
            i += 1
        if _excluido(ruta) or (origen is not None and _excluido(origen)):
            continue
        kept.append(r + (b"\0" + origen.encode() if origen is not None else b""))
    return b"\0".join(kept)


def recorrer_disco(raiz) -> tuple[frozenset[str], bool]:
    """Rutas (relativas, con «/») de los archivos visibles bajo `raiz`, con el mismo
    encierro de tool_authority: nada fuera de `raiz` y sin seguir enlaces (un enlace,
    a archivo o a directorio, no aporta ninguna ruta). Salta EXCLUSIONES. Devuelve
    (rutas, truncado): si hay más de TOPE_ENTRADAS archivos, corta y lo dice."""
    raiz_real = os.path.realpath(raiz)
    rutas: set[str] = set()
    for actual, dirs, archivos in os.walk(raiz_real, followlinks=False):
        if os.path.realpath(actual) != actual and not os.path.realpath(actual).startswith(raiz_real + os.sep):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in EXCLUSIONES and not os.path.islink(os.path.join(actual, d))]
        for nombre in archivos:
            completo = os.path.join(actual, nombre)
            if os.path.islink(completo) or nombre in EXCLUSIONES:
                continue
            rutas.add(os.path.relpath(completo, raiz_real).replace(os.sep, "/"))
            if len(rutas) > TOPE_ENTRADAS:
                return frozenset(rutas), True
    return frozenset(rutas), False


async def rutas_de_checkout(raiz) -> frozenset[str]:
    """`git ls-files` de un checkout (el desplegado de jax). Solo rutas versionadas."""
    salida = await asyncio.to_thread(_git, Path(raiz), "ls-files", "-z")
    return frozenset(r.decode("utf-8", "replace") for r in salida.split(b"\0") if r)


def leer_git(raiz) -> EstadoGit:
    """HEAD, hash de `git status --porcelain` y `git ls-files -s` de `raiz`."""
    raiz = Path(raiz)
    _git(raiz, "rev-parse", "--git-dir")  # no es un repo -> DetectorNoDisponible
    try:
        head = _git(raiz, "rev-parse", "HEAD").decode().strip()
    except DetectorNoDisponible:
        head = "(sin commits)"  # repo recién creado: válido, con lista de archivos vacía o solo del índice
    estado = hashlib.sha1(_estado_sin_exclusiones(_git(raiz, "status", "--porcelain", "-z", "--ignored"))).hexdigest()
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


def fuentes_por_defecto(raiz_repo, rutas_app=None, herramientas: Iterable[str] = (), jax_checkout=None) -> Fuentes:
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

    async def disco() -> tuple[frozenset[str], bool]:
        return await asyncio.to_thread(recorrer_disco, raiz)

    codigo_jax = (lambda: rutas_de_checkout(jax_checkout)) if jax_checkout else None
    return Fuentes(
        huella=_huella_db, esquema=_esquema_db, git=git, simbolos=simbolos,
        rutas_app=rutas_app or _rutas_de_la_app, claves=claves, modelos=_modelos_db,
        disco=disco, codigo_jax=codigo_jax,
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
        except Exception as exc:  # fail-soft: declarado, la categoría pasa a no_verificable, visible en el veredicto
            logger.warning("anti_fabricacion: fuente opcional %s caída: %s: %s", nombre, type(exc).__name__, exc)
            no_disponibles.add(nombre)
    try:
        raices = await asyncio.to_thread(_raices_fs)
    except OSError:  # fail-soft: declarado, sin raíces del sistema no se distingue una ruta de la app de una del sistema
        raices = frozenset()
        no_disponibles.add("raices_fs")

    archivos = frozenset(ruta for _sha, ruta in git.entradas)   # versionados: de ellos salen los símbolos
    if fuentes.disco is not None:
        try:
            en_disco, truncado = await fuentes.disco()
            archivos |= frozenset(en_disco)     # no versionados: SOLO la ruta
            if truncado:
                no_disponibles.add("archivos")
        except Exception as exc:  # fail-soft: declarado, sin el árbol en disco la categoría archivos pasa a no_verificable
            logger.warning("anti_fabricacion: árbol en disco no disponible: %s: %s", type(exc).__name__, exc)
            no_disponibles.add("archivos")
    if fuentes.codigo_jax is not None:
        try:
            archivos |= frozenset(await fuentes.codigo_jax())
        except Exception as exc:  # fail-soft: declarado, sin el checkout de jax las rutas no halladas pasan a no_verificable
            logger.warning("anti_fabricacion: checkout de jax no disponible: %s: %s", type(exc).__name__, exc)
            no_disponibles.add("jax_checkout")
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
