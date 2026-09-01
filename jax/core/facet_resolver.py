"""
Resolver de facetas — Bloque C (facet/facet_binding como fuente unica
faceta->modelo). Archivo unico real dentro de jax (Bloque 2, 2026-08-21):
las_manos/facet_resolver.py es symlink a este -- jacobs lo importa via
PYTHONPATH apuntando a las_manos, el REPL via el paquete jax.core. jax-platform
sigue con copia propia aparte (repo distinto, con su propio
credential_resolver.py local -- un symlink cruzado de repos no sobrevive un
clone fresco); ver scripts/check_facet_resolver_sync.py para detectar drift
entre ambas. Consume resolve_credential_instrumented, no reimplementa Fase 1.
Ver jax-platform/docs/fase2-facetas-diseno.md.
"""
import logging
import os
import time
from dataclasses import dataclass

import aiomysql

try:
    # las_manos y jacobs (PYTHONPATH incluye las_manos/, sin la raiz del
    # paquete jax): credential_resolver.py vive directo en las_manos/.
    from credential_resolver import resolve_credential_instrumented, CredentialUnavailableError
except ImportError:
    # REPL (PYTHONPATH=. desde la raiz del repo, `python -m jax.core.main`):
    # las_manos/ no esta en sys.path, solo el paquete jax.core.
    from jax.core.credential_resolver import resolve_credential_instrumented, CredentialUnavailableError

logger = logging.getLogger("facet_resolver")

FACET_CACHE_TTL_SECONDS = int(os.getenv("FACET_CACHE_TTL_SECONDS", "30"))
FACET_STALE_MAX_SECONDS = int(os.getenv("FACET_STALE_MAX_SECONDS", "300"))


class FacetUnavailableError(Exception):
    """FAIL-CLOSED: sin facet activa con binding role='primary'. El llamador
    declara estado degradado — nunca cae a un default hardcodeado."""


@dataclass
class ResolvedFacet:
    key: str
    provider_id: str
    base_url: str | None
    model: str
    credential: str
    transport: str
    persona: str | None
    params: dict | None


class _CacheEntry:
    __slots__ = ("value", "fetched_at", "fetched_at_wall")

    def __init__(self, value: ResolvedFacet, fetched_at: float, fetched_at_wall: float):
        self.value = value
        # DOS relojes a proposito, y NO son intercambiables:
        #   fetched_at      = time.monotonic() -- inmune a saltos de NTP, es el
        #                     unico valido para medir el TTL.
        #   fetched_at_wall = time.time() (epoch) -- es el unico comparable con
        #                     el st_mtime del sello. Ver _entrada_sellada().
        # Guardar uno solo y comparar el mtime contra monotonic no da error:
        # da un veredicto constante, y el sello deja de invalidar nada.
        self.fetched_at = fetched_at
        self.fetched_at_wall = fetched_at_wall


_cache: dict[str, _CacheEntry] = {}


# ---------------------------------------------------------------------------
# Sello de invalidacion cross-proceso (Q3, 2026-09-01)
# ---------------------------------------------------------------------------
# `_cache` es por-proceso y este archivo esta espejado en TRES procesos:
# jax-platform (Mesa web), las_manos/Jacobs y el REPL. Hasta hoy solo Mesa web
# invalidaba al rebindear -- los otros dos despachaban contra el binding VIEJO
# hasta FACET_CACHE_TTL_SECONDS, y la sonda por rebinding no lo podia ver
# porque sondea justamente el unico camino invalidado.
#
# El sello es un archivo vacio: el escritor le toca el mtime, y cada
# resolve_facet compara ese mtime contra el instante de RELOJ DE PARED en que
# cacheo. Un os.stat cuesta ~1 us y no sale de la maquina -- los tres procesos
# corren en hall9000, mismo filesystem, User=fruiz en los dos units. No agrega
# red, ni dependencia, ni un modo de falla nuevo: ver _seal_mtime().
FACET_SEAL_PATH = os.getenv("JAX_FACET_SEAL_PATH", "/srv/jax-data/facet-cache-seal")


def _seal_mtime() -> float | None:
    """mtime del sello, o None si no se puede leer.

    MODO DE FALLA DECLARADO -- aca, que es donde se lee, y no solo en el PR.
    None significa "no hay senal de invalidacion", NUNCA "invalidar". Si el
    sello falta o es ilegible se vuelve al techo que YA existe hoy: el TTL de
    FACET_CACHE_TTL_SECONDS. No es un fail-open nuevo; el sello solo puede
    ADELANTAR una invalidacion que el TTL iba a hacer igual.

    Invalidar ante un sello ilegible seria peor, y por eso no se hace:
    convertiria un archivo faltante en "sin cache" -- un round-trip a la DB
    por request en los tres procesos -- o sea una regresion de rendimiento
    silenciosa, y encima un modo de falla NUEVO introducido por el arreglo.
    """
    try:
        return os.stat(FACET_SEAL_PATH).st_mtime
    except OSError:
        return None


def _tocar_sello() -> bool:
    """Toca el mtime del sello. Devuelve True si quedo escrito.

    Un sello por INSTALACION, no por facet: un rebind hace que los otros
    procesos re-consulten todos los facets que tengan cacheados, no solo el
    rebindeado. Es deliberado -- un rebind es raro y manual, el costo es un
    punado de queries una sola vez, y un sello por clave agregaria un ciclo
    de vida (creacion, limpieza) que nadie pidio.

    Falla SUAVE y RUIDOSA a proposito: si el sello no se puede escribir, los
    otros procesos se quedan con el TTL de FACET_CACHE_TTL_SECONDS -- el techo
    de hoy, no algo peor -- y el escritor no revienta la aprobacion de un
    binding por no haber podido tocar un archivo de cache.
    """
    try:
        directorio = os.path.dirname(FACET_SEAL_PATH)
        if directorio:
            os.makedirs(directorio, exist_ok=True)
        with open(FACET_SEAL_PATH, "a"):
            pass
        os.utime(FACET_SEAL_PATH, None)
        return True
    except OSError as e:
        logger.error(
            f"facet_resolver seal_write_failed path={FACET_SEAL_PATH} "
            f"reason={type(e).__name__} -- los otros procesos caen al TTL de "
            f"{FACET_CACHE_TTL_SECONDS}s"
        )
        return False


def _entrada_sellada(entry: _CacheEntry) -> bool:
    """True si el sello quedo MAS NUEVO que el momento en que se cacheo.

    Compara `st_mtime` (reloj de pared, epoch) contra `entry.fetched_at_wall`
    (`time.time()`, el MISMO reloj). NUNCA contra `entry.fetched_at`, que es
    `time.monotonic()`: monotonic cuenta desde un origen arbitrario del
    sistema, sin relacion con epoch, asi que compararlo con un mtime da un
    veredicto CONSTANTE -- "invalidar siempre" si el origen es chico (en
    Linux, el boot), "no invalidar nunca" si fuera mayor que epoch. En los dos
    casos el sello no invalida nada util y el bug del cache queda igual, pero
    con codigo nuevo que aparenta resolverlo. Ese es el modo de falla
    silencioso de este arreglo, y esta clavado por tests desde los dos lados
    (test_facet_resolver_seal.py, los dos tests con `monotonic` forzado).
    """
    mtime = _seal_mtime()
    if mtime is None:
        return False
    return mtime > entry.fetched_at_wall


def invalidate_facet_cache(facet_key: str) -> bool:
    """Invalida el facet en LOS TRES procesos. Devuelve True si habia algo en
    el cache de ESTE proceso.

    Existe porque `_cache` tiene TTL de 30 s y NINGUN escritor de
    facet_binding la invalidaba: durante esos 30 s post-aprobacion,
    resolve_facet() sigue devolviendo el modelo VIEJO. Para un turno de chat
    es un retardo tolerable; para la sonda por rebinding -- que dispara dentro
    de esa misma ventana -- significa validar el binding anterior y reportar
    `ok` sobre el nuevo sin haberlo tocado.

    Jacobs y el REPL no comparten este `_cache`: se enteran por el sello, que
    se toca PRIMERO. El valor de retorno describe solo el cache local, que es
    lo unico que esta funcion puede afirmar de primera mano.
    """
    _tocar_sello()
    return _cache.pop(facet_key, None) is not None


async def _db_conn() -> aiomysql.Connection:
    host = os.environ.get("JAX_DB_HOST")
    port = os.environ.get("JAX_DB_PORT")
    if not host or not port:
        raise RuntimeError(
            "JAX_DB_HOST/JAX_DB_PORT no están seteados -- sin default "
            "silencioso a localhost:3306 (esa instancia está muerta, ver "
            "memoria jax-dual-mariadb-instances). Sourceá /etc/jax/.env o "
            "exportalos a mano antes de conectar."
        )
    return await aiomysql.connect(
        host=host,
        port=int(port),
        user=os.getenv("JAX_DB_USER", ""),
        password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.getenv("JAX_DB_NAME", "jax_memory"),
        charset="utf8mb4",
        autocommit=True,
    )


async def _query_facet(facet_key: str) -> ResolvedFacet:
    """D1.1 paso 4 (fase2-facetas-diseno.md:233-237), completado 2026-08-19:
    lee el modelo via model_ref -> model.model_id, no b.model_id (texto
    libre, quedaba desincronizado de las aprobaciones de
    model_binding_proposal). facet_binding.model_id se conserva de
    solo-lectura un ciclo de cutover antes de dropearla, no se usa mas aca."""
    conn = await _db_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT f.transport, f.persona, p.base_url, b.provider_id, m.model_id, b.params "
                "FROM facet f "
                "JOIN facet_binding b ON b.facet_key = f.`key` AND b.role = 'primary' "
                "JOIN provider p ON p.id = b.provider_id "
                "JOIN model m ON m.id = b.model_ref "
                "WHERE f.`key` = %s AND f.status = 'active'",
                (facet_key,),
            )
            row = await cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise FacetUnavailableError(f"sin binding activo para facet '{facet_key}'")
    transport, persona, base_url, provider_id, model_id, params = row

    credential = ""
    if transport not in ("ollama", "subprocess"):  # ollama/subprocess no usan credencial de proveedor gestionada aqui
        try:
            credential = await resolve_credential_instrumented(provider_id)
        except CredentialUnavailableError as e:
            raise FacetUnavailableError(f"facet '{facet_key}': {e}") from e

    return ResolvedFacet(
        key=facet_key, provider_id=provider_id, base_url=base_url, model=model_id,
        credential=credential, transport=transport, persona=persona, params=params,
    )


async def load_facet_registry() -> dict:
    """Una sola query, para bootstrap SINCRONO-de-arranque (REPL): quien
    existe, con que modelo, label e icono. A diferencia de resolve_facet()
    (por-request, con cache/TTL), esto se llama UNA vez al arrancar un
    proceso de sesion unica — mismo patron que build_muscles() ya usaba
    para leer config.toml una vez, solo que ahora la fuente es la DB.
    Devuelve {key: {model, provider_id, display_name, icon, auto_selectable}}."""
    conn = await _db_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT f.`key`, f.display_name, f.icon, f.auto_selectable, "
                "b.provider_id, m.model_id "
                "FROM facet f JOIN facet_binding b ON b.facet_key = f.`key` AND b.role = 'primary' "
                "JOIN model m ON m.id = b.model_ref "
                "WHERE f.status = 'active'"
            )
            rows = await cur.fetchall()
    finally:
        conn.close()
    return {
        key: {
            "display_name": display_name, "icon": icon,
            "auto_selectable": bool(auto_sel),
            "provider_id": provider_id, "model": model_id,
        }
        for key, display_name, icon, auto_sel, provider_id, model_id in rows
    }


async def resolve_facet(facet_key: str) -> ResolvedFacet:
    now = time.monotonic()
    # Se toma ANTES de la query a proposito: si el sello se toca MIENTRAS esta
    # query corre, el valor que traiga ya nace sospechoso y la proxima
    # resolucion lo descarta. Conservador en la direccion correcta.
    now_wall = time.time()
    cached = _cache.get(facet_key)
    if cached is not None and _entrada_sellada(cached):
        # DESCARTAR, no solo saltear el TTL: un escritor la declaro obsoleta.
        # Sacarla del dict tambien la saca del camino de `serving_stale` de
        # abajo -- servir un valor que sabemos superado seria peor que
        # declarar estado degradado, y es el binding viejo que esta ronda
        # vino a matar.
        _cache.pop(facet_key, None)
        cached = None
    if cached and (now - cached.fetched_at) < FACET_CACHE_TTL_SECONDS:
        return cached.value
    try:
        value = await _query_facet(facet_key)
        _cache[facet_key] = _CacheEntry(value, now, now_wall)
        return value
    except Exception as e:
        if cached and (now - cached.fetched_at) < FACET_STALE_MAX_SECONDS:
            logger.warning(f"facet_resolver key={facet_key} db_unreachable=1 serving_stale reason={type(e).__name__}")
            return cached.value
        logger.error(f"facet_resolver key={facet_key} FAIL_CLOSED reason={type(e).__name__}")
        raise FacetUnavailableError(facet_key) from e
