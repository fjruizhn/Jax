"""Idempotencia de `POST /procesamiento/trabajos` (jax-platform E3, auditoria 2026-10-06).

El problema. El despachador de documentos de jax-platform manda un trozo de archivos a LAS
MANOS y recibe `202 {job_id}`. Si su proceso se corta entre ese 202 y el momento en que ata
las filas al `job_id`, al reiniciar las filas siguen `en_cola` y el trozo se manda otra vez:
el OCR se duplica. No corrompe datos (el duplicado procesa los mismos archivos de forma
atomica y su resultado se ignora), pero cuesta OCR.

La solucion. El llamador manda un `Idempotency-Key` ESTABLE (derivado de las filas, igual en
un reenvio despues de un reinicio). Aqui se guarda la clave de forma durable, con UNIQUE
`(identidad_servicio, clave)`, junto al `job_id` que se le dio. Una clave ya vista devuelve el
MISMO trabajo, sin crear otro.

Protocolo (RECLAMAR -> CREAR), pensado para que dos pedidos simultaneos con la misma clave
creen UN solo trabajo:

  1. `reclamar`: INSERT de (identidad, clave, hash del pedido, job_id candidato). Si el UNIQUE
     rechaza el INSERT, la perdedora lee la fila de la ganadora (`gano=False`).
  2. Quien gano crea el trabajo con ESE job_id (`JobStore.create(job_id=...)`). Si crearlo
     falla, `liberar` borra el reclamo (solo si sigue siendo suyo) para que el reintento no
     herede un job_id que no existe.
  3. Si el proceso muere entre 1 y 2 queda un reclamo HUERFANO (un job_id sin trabajo). Un
     reintento no lo trata como trabajo existente: si es mas joven que la gracia (el trabajo
     puede estarse creando ahora mismo) pide esperar (503); pasada la gracia lo retoma con un
     CAS sobre el job_id (`tomar_huerfana`): de varios reintentos a la vez retoma UNO. Si la ganadora
     original no estaba muerta sino lenta (tardo mas que la gracia entre el INSERT y crear el trabajo),
     al terminar de crearlo relee el reclamo, ve que ya no es suyo, CANCELA su trabajo antes de programar el
     OCR y devuelve el que figura en la tabla: queda un solo trabajo vivo, no dos.

  4. Un reenvio cuyo trabajo ya existe pero FALLO o se cancelo (p. ej. LAS MANOS se reinicio y marco `failed`
     lo que corria) sin que el llamador llegara a saber su job_id, se trata como el huerfano: se retoma con
     el mismo CAS y se crea un trabajo nuevo. Si el llamador lo hubiera conocido, ya habria atado sus filas y no
     reenviaria.

El hash del pedido (dueno + project_uuid + rutas) hace que la misma clave con OTRO pedido sea un
conflicto (409) y no un trabajo equivocado.

La tabla se crea al arrancar LAS MANOS con `CREATE TABLE IF NOT EXISTS` (mismo camino que las
tablas de Jacobs: `jacobs.store.init_tables`); repetirlo no cambia nada. Se purga por edad
(`JAX_PROCESAMIENTO_IDEMPOTENCIA_TTL_SEGUNDOS`): hoy el almacen de trabajos no se poda, asi que
la clave vive un plazo acotado y no para siempre; el reenvio que protege ocurre en minutos.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass

from jacobs import store as jacobs_store
from processing_ownership import ProcessingOwnershipContext

logger = logging.getLogger(__name__)

NOMBRE_TABLA = "procesamiento_idempotencia"
ENCABEZADO = "Idempotency-Key"

#: 16 a 128 caracteres ASCII imprimibles sin espacios. El llamador deriva la clave con un hash; una
#: clave corta o con texto libre seria adivinable o ambigua.
_CLAVE_VALIDA = re.compile(r"[A-Za-z0-9._:\-]{16,128}")

_VARIABLE_TTL = "JAX_PROCESAMIENTO_IDEMPOTENCIA_TTL_SEGUNDOS"
_VARIABLE_GRACIA = "JAX_PROCESAMIENTO_IDEMPOTENCIA_GRACIA_SEGUNDOS"
_TTL_POR_DEFECTO = 7 * 24 * 3600
_GRACIA_POR_DEFECTO = 60
_VARIABLE_ESPERA = "JAX_PROCESAMIENTO_IDEMPOTENCIA_ESPERA_MS"
_ESPERA_POR_DEFECTO_MS = 2000
_PURGA_CADA_SEGUNDOS = 3600
_LOTE_DE_PURGA = 1000

_DDL = f"""
CREATE TABLE IF NOT EXISTS {NOMBRE_TABLA} (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    identidad_servicio VARCHAR(32) CHARACTER SET ascii NOT NULL,
    clave VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    solicitud_hash CHAR(64) CHARACTER SET ascii NOT NULL,
    job_id VARCHAR(36) CHARACTER SET ascii NOT NULL,
    creado_en DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uq_procesamiento_idem_clave (identidad_servicio, clave),
    KEY idx_procesamiento_idem_creado (creado_en)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

SQL_INSERTAR = (f"INSERT INTO {NOMBRE_TABLA} (identidad_servicio, clave, solicitud_hash, job_id) "
                "VALUES (%s, %s, %s, %s)")
SQL_BUSCAR = (f"SELECT id, solicitud_hash, job_id, TIMESTAMPDIFF(MICROSECOND, creado_en, NOW(6)) / 1000000 "
              f"FROM {NOMBRE_TABLA} WHERE identidad_servicio = %s AND clave = %s")
SQL_TOMAR = f"UPDATE {NOMBRE_TABLA} SET job_id = %s, creado_en = NOW(6) WHERE id = %s AND job_id = %s"
SQL_LIBERAR = f"DELETE FROM {NOMBRE_TABLA} WHERE identidad_servicio = %s AND clave = %s AND job_id = %s"
SQL_PURGAR = f"DELETE FROM {NOMBRE_TABLA} WHERE creado_en < NOW(6) - INTERVAL %s SECOND LIMIT %s"

_ERROR_DE_CLAVE_DUPLICADA = 1062


def _entero_de_entorno(nombre: str, por_defecto: int, minimo: int) -> int:
    crudo = os.environ.get(nombre)
    if crudo is None:
        return por_defecto
    try:
        valor = int(crudo)
    except ValueError:
        raise RuntimeError(f"{nombre}={crudo!r} no es un entero") from None
    if valor < minimo:
        raise RuntimeError(f"{nombre}={valor} es menor que {minimo}")
    return valor


def ttl_segundos() -> int:
    """Cuanto vive una clave. Presente pero invalida -> RuntimeError (se detecta al arrancar)."""
    return _entero_de_entorno(_VARIABLE_TTL, _TTL_POR_DEFECTO, 60)


def gracia_segundos() -> int:
    """Cuanto espera un reintento antes de tratar un reclamo sin trabajo como huerfano."""
    return _entero_de_entorno(_VARIABLE_GRACIA, _GRACIA_POR_DEFECTO, 1)


def espera_ms() -> int:
    """Cuanto espera (como mucho) una perdedora de la carrera a que la ganadora termine de crear el
    trabajo, antes de contestar 503 reintentable. 0 = no espera."""
    return _entero_de_entorno(_VARIABLE_ESPERA, _ESPERA_POR_DEFECTO_MS, 0)


def abreviar(clave: str) -> str:
    """Hash corto de la clave para el log: identifica el reenvio sin dejar la clave entera."""
    return hashlib.sha256(clave.encode("utf-8", "backslashreplace")).hexdigest()[:12]


def clave_valida(clave: object) -> bool:
    return isinstance(clave, str) and _CLAVE_VALIDA.fullmatch(clave) is not None


def hash_de_solicitud(ownership: ProcessingOwnershipContext, project_uuid: str, rutas: list[str]) -> str:
    """Huella del pedido: el dueno autenticado, el proyecto y las rutas (en orden canonico). La misma
    clave con otra huella es un conflicto. `ensure_ascii` + `backslashreplace` aceptan hasta un
    surrogate solitario sin lanzar (la ruta ya paso por la validacion de la ruta)."""
    cuerpo = json.dumps([ownership.tenant_id, ownership.user_id, ownership.project_id, project_uuid, sorted(rutas)],
                        ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(cuerpo.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Reclamo:
    id: int
    job_id: str
    solicitud_hash: str
    antiguedad_s: float
    gano: bool


async def init_tabla() -> None:
    """Crea la tabla si no existe. Se llama al arrancar LAS MANOS; repetirla no cambia nada."""
    async with jacobs_store.conexion(desechable=True) as conn:
        async with conn.cursor() as cur:
            await cur.execute(_DDL)


def _valores(fila) -> tuple:
    return tuple(fila.values()) if isinstance(fila, dict) else tuple(fila)


async def buscar(identidad: str, clave: str) -> Reclamo | None:
    async with jacobs_store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(SQL_BUSCAR, (identidad, clave))
            fila = await cur.fetchone()
    if fila is None:
        return None
    id_, huella, job_id, edad = _valores(fila)
    return Reclamo(id=int(id_), job_id=str(job_id), solicitud_hash=str(huella), antiguedad_s=float(edad), gano=False)


def _es_clave_duplicada(exc: BaseException) -> bool:
    return bool(exc.args) and exc.args[0] == _ERROR_DE_CLAVE_DUPLICADA


async def reclamar(identidad: str, clave: str, solicitud_hash: str, job_id: str) -> Reclamo:
    """INSERT atomico del reclamo. `gano=True` si ESTA llamada lo inserto (y entonces le toca crear el
    trabajo `job_id`); si el UNIQUE lo rechazo, devuelve la fila de la ganadora con `gano=False`."""
    try:
        async with jacobs_store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(SQL_INSERTAR, (identidad, clave, solicitud_hash, job_id))
                return Reclamo(id=int(cur.lastrowid), job_id=job_id, solicitud_hash=solicitud_hash,
                               antiguedad_s=0.0, gano=True)
    except Exception as exc:  # fail-closed salvo la clave duplicada, que es el caso normal de un reenvio: lo demas se relanza
        if not _es_clave_duplicada(exc):
            raise
    existente = await buscar(identidad, clave)
    if existente is None:
        # La ganadora libero su reclamo entre el INSERT rechazado y esta lectura: un reintento lo resuelve.
        return await reclamar(identidad, clave, solicitud_hash, job_id)
    return existente


async def tomar_huerfana(reclamo: Reclamo, job_id_nuevo: str) -> bool:
    """CAS: pasa un reclamo sin trabajo a `job_id_nuevo` solo si sigue apuntando al job_id viejo. De
    varios reintentos a la vez, True para UNO (la ganadora lenta se entera al releer; ver el protocolo arriba)."""
    async with jacobs_store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(SQL_TOMAR, (job_id_nuevo, reclamo.id, reclamo.job_id))
            return cur.rowcount == 1


async def liberar(identidad: str, clave: str, job_id: str) -> None:
    """Borra el reclamo SOLO si sigue siendo de `job_id` (no el de quien lo retomo despues)."""
    async with jacobs_store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(SQL_LIBERAR, (identidad, clave, job_id))


async def purgar_vencidas() -> int:
    """Borra por lotes las claves mas viejas que el TTL (usa `idx_procesamiento_idem_creado`)."""
    total = 0
    while True:
        async with jacobs_store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(SQL_PURGAR, (ttl_segundos(), _LOTE_DE_PURGA))
                borradas = cur.rowcount
        total += borradas
        if borradas < _LOTE_DE_PURGA:
            return total


_ultima_purga = 0.0


async def purgar_si_toca() -> None:
    """Purga a lo sumo una vez por hora y por proceso; nunca propaga un fallo (es limpieza)."""
    global _ultima_purga
    ahora = time.monotonic()
    if _ultima_purga and ahora - _ultima_purga < _PURGA_CADA_SEGUNDOS:
        return
    _ultima_purga = ahora
    try:
        borradas = await purgar_vencidas()
        if borradas:
            logger.info("procesamiento_idempotencia: %s clave(s) vencida(s) purgadas", borradas)
    except Exception:  # fail-soft: la purga es limpieza; sin ella la tabla crece un poco mas y la proxima hora lo reintenta
        logger.warning("procesamiento_idempotencia: no se pudo purgar", exc_info=True)
