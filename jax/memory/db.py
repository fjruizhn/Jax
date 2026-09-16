"""
JAX 2.0 - Conector de Memoria Persistente
===========================================
Conecta JAX a su memoria en MariaDB (base jax_memory en hall9000).

Filosofia de diseno (contrato firmado por Claude + DeepSeek + Hipatia):
  - Driver: aiomysql (Python puro, sin compilar Cython).
  - Tolerante a fallos: si la base cae, JAX SIGUE conversando. La memoria
    es un "plus", nunca un punto de falla. Igual que el kill switch.
  - Pool pequeno (1-5 conexiones), autocommit.
  - Fire-and-forget al guardar: no agrega latencia a la respuesta de JAX.
  - Alcance: start/save/end/health_check. La extraccion de hechos
    (worker batch) es una pieza aparte, no esta.

En memoria de Jairo Urbina.
"""

import asyncio
import os
import uuid
import logging
import functools
import json
import math
from datetime import datetime
from typing import Awaitable, Callable, Optional

import aiomysql
import httpx

from .embedding_config import CONFIG as EMBED, zero_vector_text
from jax.core.db_connect_config import db_connect_timeout_seconds
from .migrations import ensure_schema

logger = logging.getLogger("jax.memory")

# Modelo, dimension y columna de los embeddings: configuracion, no codigo
# (2026-09-12, migracion a bge-m3). Ver jax/memory/embedding_config.py y
# scripts/migrar_embeddings.py. Todo lo de abajo lee EMBED en cada uso, no al
# importar. EMBEDDING_DIM queda como alias de lectura para quien lo importaba.
EMBEDDING_DIM = EMBED.dim


# ------------------------------------------------------------
# Mapeo de faceta -> valor exacto del ENUM `role` en la tabla
# messages. El esquema define:
#   ENUM('user','jax_local','jekyll','hyde','hipatia')
# Si guardamos un valor fuera de ese set, MariaDB RECHAZA el
# insert en silencio (y como es fire-and-forget, no se nota).
# Por eso normalizamos SIEMPRE antes de insertar.
# ------------------------------------------------------------
_ROLE_MAP = {
    "user": "user",
    "jax": "jax_local",
    "jax_local": "jax_local",
    "local": "jax_local",
    "jekyll": "jekyll",
    "hyde": "hyde",
    "hipatia": "hipatia",
    "thot": "thot",
    "kimi": "kimi",
    "ada": "ada",
}


def _normalize_role(role: Optional[str]) -> str:
    """Traduce cualquier nombre de faceta al valor valido del ENUM.
    Si no reconoce el valor, cae a 'jax_local' (nunca rompe el insert)."""
    if not role:
        return "jax_local"
    return _ROLE_MAP.get(role.strip().lower(), "jax_local")


# ------------------------------------------------------------
# Deteccion de corriones/duplicados por distancia vectorial.
# Un solo par de umbrales, dos usos: dedup (item #3) y correccion
# (item #1) comparten la misma banda "correction_candidate" porque
# ambas son, en el fondo, la misma pregunta ("hay algo casi igual a
# esto ya guardado?"), solo que se resuelven distinto segun si el
# extractor marco is_correction o no.
# ------------------------------------------------------------
DUP_DISTANCE_THRESHOLD = 0.05
CORRECTION_DISTANCE_THRESHOLD = 0.25


def _validate_importance(importance: Optional[int]) -> Optional[int]:
    """1-5 valido -> se guarda tal cual. Cualquier otra cosa (fuera de
    rango, no-entero, None) -> None (neutral, sin score)."""
    if isinstance(importance, int) and 1 <= importance <= 5:
        return importance
    return None


def _should_skip_as_duplicate(band: str, is_correction: bool) -> bool:
    """True si save_fact debe NO insertar (duplicado casi textual, y no es
    una correccion — una correccion SI se inserta aunque el texto sea casi
    identico al viejo, ver _should_supersede)."""
    return band == "duplicate" and not is_correction


def _should_supersede(candidate_exists: bool, band: str, is_correction: bool,
                      verify_confirmed: bool) -> bool:
    """True si save_fact debe marcar el candidato como superseded_by el
    fact nuevo. Requiere: hay candidato, el extractor marco is_correction,
    la banda no es 'unrelated' (esta lo bastante cerca como para ser el
    mismo tema), y el chequeo de verificacion (si se paso uno) confirmo."""
    return candidate_exists and is_correction and band != "unrelated" and verify_confirmed


def classify_fact_distance(distancia: float) -> str:
    """Clasifica una distancia coseno contra el fact activo mas cercano.
    Devuelve 'duplicate' | 'correction_candidate' | 'unrelated'."""
    if distancia < DUP_DISTANCE_THRESHOLD:
        return "duplicate"
    if distancia < CORRECTION_DISTANCE_THRESHOLD:
        return "correction_candidate"
    return "unrelated"


def _blend_query(user_text: str, recent_history: Optional[list]) -> str:
    """Combina el mensaje actual con los ultimos 4 turnos (500 chars c/u) del
    historial de la conversacion en curso, para que el embedding de busqueda
    capture el hilo, no solo la ultima frase (que puede ser ambigua sola:
    'y eso por que?'). Sin historial, devuelve user_text sin cambios
    (comportamiento identico al de antes de este blend)."""
    if not recent_history:
        return user_text
    turnos = recent_history[-4:]
    piezas = [t["content"][:500] for t in turnos if t.get("content")]
    piezas.append(user_text)
    return "\n".join(piezas)


# ------------------------------------------------------------
# Bypass de categoria para preguntas de completeness (item #4). Preguntas
# tipo "que proyectos tenes activos" no se responden bien con similitud
# vectorial contra UN fact — necesitan TODOS los facts de una categoria.
# Deteccion por keyword, mismo estilo que el router hibrido de JAX (no
# hace falta un LLM para esto).
# ------------------------------------------------------------
# Orden importa: se evalua de arriba a abajo y gana el primer match. Las mas
# especificas van primero — "de mis finanzas"/"de mis socios" tambien
# matchean el patron generico "de mi" de la categoria 'user', asi que 'user'
# (el catch-all) va al final.
_COMPLETENESS_PATTERNS = {
    "project": ("que proyectos", "cuales proyectos", "en que proyectos"),
    "preference": ("mis preferencias", "que preferis", "como te gusta que"),
    "technical": ("que decisiones tecnicas", "que elegimos", "que decisiones tomamos"),
    "social": ("mis relaciones", "que sabes de mis contactos", "quienes son mis socios"),
    "financial": ("mis finanzas", "que sabes de mis finanzas", "mi situacion financiera"),
    "user": ("que sabes de mi", "que sabes sobre mi", "todo lo que sabes de mi"),
}


def detect_completeness_intent(text: str) -> Optional[str]:
    """Detecta si el texto es una pregunta de 'dame todo lo que sepas de X'
    en vez de una pregunta puntual. Devuelve el fact_type a barrer completo
    via get_facts(), o None si es una pregunta normal (solo retrieval
    semantico, como siempre)."""
    normalizado = text.lower()
    for fact_type, patrones in _COMPLETENESS_PATTERNS.items():
        if any(p in normalizado for p in patrones):
            return fact_type
    return None


# ------------------------------------------------------------
# Reranking opcional con cross-encoder (item #7). Import perezoso: si
# sentence-transformers no esta instalado, _get_reranker() devuelve None y
# el caller sigue sin reranking (no rompe nada). El modelo es chico
# (~90MB, ms-marco-MiniLM-L-6-v2) y corre en CPU — reordenar un puñado de
# candidatos no necesita GPU.
# ------------------------------------------------------------
_reranker = None


def _get_reranker():
    """Carga el cross-encoder la primera vez que se usa. None si el paquete
    no esta instalado o si la carga del modelo falla por cualquier motivo."""
    global _reranker
    if _reranker is None:
        try:
            # Cache de pesos en /opt/jax (140G libres), no en ~/.cache (raiz,
            # espacio ajustado) — mismo motivo por el que el venv vive ahi.
            os.environ.setdefault("HF_HOME", "/opt/jax/hf-cache")
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:  # fail-soft: el reranker es opcional (item #7); sin el paquete o sin el modelo se devuelve None y el caller conserva el orden por distancia cosine, que ya es un orden valido
            logger.warning(f"reranker no disponible ({e}), se sigue sin reranking")
            _reranker = False
    return _reranker or None


def _merge_search_results(rows_a: list, rows_b: list, limit: int) -> list:
    """Une dos listas de resultados de busqueda (dicts con 'content',
    'created_at', 'distancia'), dedupe por (content, created_at) quedandose
    con la distancia mas chica, ordena por distancia y corta a `limit`."""
    merged: dict = {}
    for r in rows_a + rows_b:
        key = (r["content"], r["created_at"])
        if key not in merged or r["distancia"] < merged[key]["distancia"]:
            merged[key] = r
    return sorted(merged.values(), key=lambda r: r["distancia"])[:limit]


# --- Embeddings "vector cero" -----------------------------------------------
# messages.embedding y facts.embedding son VECTOR(EMBEDDING_DIM) NOT NULL con
# DEFAULT vector cero en produccion (el VECTOR KEY exige NOT NULL). Una fila
# queda en ceros mientras save_message() espera el embedding -- o para
# siempre, si Ollama falla: 24 filas asi en jax_memory, todas del 2026-06-09.
#
# VEC_DISTANCE_COSINE contra un vector de norma cero da NaN, y el NaN rompe
# todo lo que viene despues. Medido 2026-09-11:
#   - `IS NULL` no lo atrapa: para el servidor no es NULL;
#   - el ORDER BY ... ASC lo ubica en cualquier lado -- en produccion, primero:
#     search_similar_messages("hola", user_id=1) devolvia 5 filas, las 5 asi,
#     y api/chat.py tumbaba el turno con `None < 0.8`;
#   - aiomysql lo entrega como None en un camino y como 0.0 en otro.
# El 0.0 es el caso grave: una distancia 0.0 falsa es indistinguible de un
# duplicado exacto (add_fact la leeria como tal). Por eso la exclusion va EN
# SQL, sobre el embedding guardado, y no filtrando distancias en Python.
def _col(prefijo: str = "") -> str:
    """Columna de embeddings configurada (validada como identificador en
    embedding_config), con prefijo de alias opcional ("m.")."""
    return f"{prefijo}{EMBED.column}"


def _nonzero_embedding_sql(column: str) -> str:
    """Predicado SQL: `column` tiene norma distinta de cero. El literal sale de
    la dimension configurada, no de entrada de usuario."""
    return f"VEC_DISTANCE_EUCLIDEAN({column}, VEC_FromText('{zero_vector_text(EMBED.dim)}')) > 0"


def _zero_embedding_sql(column: str) -> str:
    """Predicado SQL inverso: `column` sigue en vector cero. Es el guard del
    backfill -- nunca pisa un embedding real, ni el que un save_message()
    tardio acabe de escribir."""
    return f"VEC_DISTANCE_EUCLIDEAN({column}, VEC_FromText('{zero_vector_text(EMBED.dim)}')) = 0"


# Tablas con embedding y la columna de texto que se vectoriza. Lista cerrada:
# el nombre de tabla se interpola en el SQL de backfill_zero_embeddings().
_BACKFILL_TABLES = {"messages": "content", "facts": "fact_text"}


def _is_degenerate_embedding(embedding: Optional[list]) -> bool:
    """Un embedding de CONSULTA de norma cero da NaN contra todas las filas,
    incluidas las sanas: no hay busqueda posible con el."""
    return not embedding or not any(embedding)


def _finite_distance_rows(rows: list, origen: str) -> list:
    """Segunda capa: descarta filas cuya distancia no es un numero finito, y lo
    registra. Con _nonzero_embedding_sql() en el WHERE no deberia descartar
    nada; si lo hace, aparecio otra causa de NaN y hay que investigarla -- por
    eso WARNING y no silencio. NO detecta un NaN entregado como 0.0: esa es la
    razon de que la primera capa sea obligatoria."""
    buenas = [
        r for r in rows
        if isinstance(r.get("distancia"), (int, float)) and math.isfinite(r["distancia"])
    ]
    if len(buenas) != len(rows):
        logger.warning(
            f"{origen}: {len(rows) - len(buenas)} fila(s) con distancia no finita "
            f"descartada(s) pese al filtro de vector cero"
        )
    return buenas


def db_error_handler(func):
    """Decorador: cualquier error de DB se loguea y se traga.
    La conversacion NUNCA se interrumpe por un fallo de memoria.

    CONTRATO, explicitado el 2026-09-16 tras la auditoria P10. `None` significa
    SIEMPRE "no se pudo completar". En una LECTURA es ambiguo a proposito
    (None = fallo; una lectura vacia devuelve [] o False segun el metodo), pero
    en una ESCRITURA no hay ambiguedad posible: no existe el caso "no habia
    nada que escribir", asi que **None de un metodo de escritura es un fallo y
    quien llama TIENE que mirarlo**.

    Eso era exactamente el defecto: `save_fact` llamaba a `supersede_fact` sin
    comprobar el retorno y despues logueaba "fact N corrige a fact M", dejando
    dos hechos contradictorios activos mientras el registro afirmaba lo
    contrario. El decorador no estaba de mas; faltaba que el llamador leyera lo
    que devuelve. Misma familia que `connect()` descartando el booleano de
    `ensure_schema()`, arreglado en la misma ronda.

    Metodos de ESCRITURA decorados, cuyo None hay que comprobar siempre:
    delete_fact, end_conversation, mark_action_item_done, mark_processed,
    save_action_item, save_decision, save_fact, _save_message_impl,
    save_person, start_conversation, supersede_fact, touch_person_mentions,
    verify_fact.
    """
    @functools.wraps(func)  # conserva nombre, docstring y __wrapped__ (2026-09-16)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:  # fail-soft: contrato declarado arriba -- None SIEMPRE significa 'no se pudo'; en escrituras no hay ambiguedad y el llamador TIENE que mirarlo (save_fact/supersede_fact y connect/ensure_schema, arreglados 2026-09-16)
            logger.error(f"DB error en {func.__name__}: {e}", exc_info=True)
            return None
    return wrapper


class BusquedaDeFactFallida(RuntimeError):
    """La busqueda del fact mas cercano no se pudo completar.

    ARREGLADO 2026-09-16. `_find_nearest_fact` devolvia None tanto cuando NO
    HAY candidato como cuando la consulta FALLABA, y `save_fact` traduce ese
    None a band="unrelated". Con is_correction=True eso significa que NUNCA se
    entra al bloque de correccion: el fact ERRONEO que se queria corregir sigue
    activo y el nuevo entra como independiente. La memoria queda con DOS hechos
    contradictorios vivos y ni una linea que lo diga.

    El docstring vendia ese None como "fail-safe: el caller inserta", y eso solo
    es cierto para el caso "no hay candidato". Cuando la consulta se cae, nadie
    dijo que no hubiera nada que corregir: es que no se pudo mirar.
    """


class MemoryDB:
    """Memoria persistente de JAX sobre MariaDB."""

    def __init__(self):
        # None = todavia no se intento conectar. True/False = resultado de la
        # ultima migracion. Lo lee health_check(): un esquema a medias NO es
        # una base sana, aunque responda al SELECT.
        self.schema_ok: Optional[bool] = None
        self.pool: Optional[aiomysql.Pool] = None
        self.config: dict = {}
        # CORRECCION (vs diseno de Deep): guardamos referencia fuerte a las
        # tareas fire-and-forget. Sin esto, Python puede recolectar la tarea
        # con el garbage collector ANTES de que termine, perdiendo el mensaje
        # en silencio. Es un gotcha conocido de asyncio.create_task().
        self._pending_tasks: set[asyncio.Task] = set()

    # --------------------------------------------------------
    # Ciclo de vida del pool
    # --------------------------------------------------------
    async def connect(self, host: str, user: str, password: str,
                      database: str, port: int | None = None) -> bool:
        """Inicializa el pool. Devuelve True si conecto, False si fallo
        (sin lanzar excepcion: JAX debe arrancar aunque la memoria falle)."""
        if port is None:
            env_port = os.environ.get("JAX_DB_PORT")
            if not env_port:
                raise RuntimeError(
                    "JAX_DB_PORT no está seteado -- sin default silencioso a "
                    "3306 (esa instancia está muerta, ver memoria "
                    "jax-dual-mariadb-instances). Sourceá /etc/jax/.env o "
                    "pasá port= explícito."
                )
            port = int(env_port)
        # --- Calidad de la busqueda vectorial (HNSW) -----------------------
        # `idx_embedding` es un indice APROXIMADO: desde que la busqueda dejo
        # de unir con `conversations` el optimizador si lo usa, y eso cambia el
        # resultado -- ya no es el vecino exacto sino uno muy cercano.
        # `mhnsw_ef_search` es cuantos candidatos explora el grafo: mas alto,
        # mas exacto y mas lento. Medido el 2026-09-11 sobre jax_memory (1.607
        # filas, 768 dims, indice reconstruido con M=16), recall@5 contra la
        # busqueda exacta:
        #     ef=20 (DEFAULT DE MARIADB) ... 50,7 %  <- inaceptable
        #     ef=100 ...................... 86,7 %
        #     ef=400 ...................... 93,3 %   0,92 ms
        #     ef=1000 ..................... 90,7 %   (satura: no compensa)
        # La exacta cuesta 49 ms. O sea 400 da 54x mas rapido perdiendo ~1 de
        # cada 14 vecinos del top-5, y el que entra en su lugar esta a una
        # distancia casi identica (+0,0005 medido).
        # Re-medido el 2026-09-12 sobre una copia de produccion (1.607 filas,
        # 30 consultas), por DISTANCIA: un resultado cuenta si esta a <= la
        # distancia del 5o vecino exacto. messages tiene duplicados exactos
        # (16 de 30 consultas con empates en el top-5), y comparar CONJUNTOS
        # de ids castiga empates que el indice no pierde -- el 93,3 % de arriba
        # probablemente arrastra ese sesgo. Por distancia, ef=400:
        #     768 dims (nomic) ....... 100 %   0,89 ms  (exacta 79 ms)
        #     1024 dims (bge-m3) ..... 100 %   1,02 ms  (exacta 107 ms)
        # ef=400 alcanza para las dos dimensiones.
        #
        # Se fija en la CONEXION y no por consulta: una sentencia mas por turno
        # de chat seria un round-trip regalado. Configurable porque el punto
        # optimo depende del volumen y de las dimensiones -- hay que volver a
        # medirlo cuando la tabla crezca un orden de magnitud.
        ef_search = int(os.getenv("JAX_MEMORY_HNSW_EF_SEARCH", "400"))
        self.config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "db": database,
            "init_command": f"SET SESSION mhnsw_ef_search = {ef_search}",
        }
        try:
            self.pool = await aiomysql.create_pool(
                minsize=1,
                maxsize=5,
                autocommit=True,
                charset="utf8mb4",
                # Hallazgo de revisión, Tarea 2b (tanda A, ronda de arreglo 2,
                # 2026-09-14): mismo bug que aiomysql.connect() sin
                # connect_timeout -- create_pool() también espera sin límite
                # si la DB se cuelga al abrir cada conexión del pool (ver
                # jax/core/db_connect_config.py).
                connect_timeout=db_connect_timeout_seconds(),
                **self.config,
            )
            logger.info(f"MemoryDB conectada a {database}@{host} (pool 1-5)")
            # Esquema al dia antes de servir: barato (cuatro SELECT sobre
            # catalogo). Ver jax/memory/migrations.py.
            #
            # ARREGLADO 2026-09-16: el valor de retorno se DESCARTABA. Como
            # ensure_schema() aplica DDL + backfill sin transaccion (autocommit,
            # y en MariaDB el DDL no se revierte), un fallo a mitad deja la base
            # en un estado parcial -- por ejemplo la columna agregada y el
            # backfill sin correr, que es justo el caso que el propio archivo
            # describe como "una migracion a medias que se ve como JAX se olvido
            # de todo". Y todo JAX seguia arrancando como si el esquema
            # estuviera al dia, con una unica linea de log que nadie mira.
            self.schema_ok = await ensure_schema(self.pool)
            if not self.schema_ok:
                logger.error(
                    "MemoryDB: el esquema NO esta al dia y no se pudo completar la "
                    "migracion. La base responde, pero puede faltar una columna o un "
                    "backfill: la busqueda por scope (user_id/project_id) puede devolver "
                    "MENOS de lo que hay, sin error. health_check() devuelve False "
                    "mientras dure. Revisar el error de la migracion, arriba."
                )
            return True
        except Exception as e:  # fail-soft: no se traga el fallo, se reporta como return False y self.pool=None; el caller decide si arranca sin memoria (JAX debe arrancar aunque la DB no responda)
            logger.error(f"MemoryDB no pudo conectar: {e}")
            self.pool = None
            return False

    async def close(self):
        """Cierra el pool tras esperar las tareas pendientes de guardado."""
        # Esperamos a que terminen los guardados en vuelo (con timeout corto)
        if self._pending_tasks:
            try:
                await asyncio.wait(self._pending_tasks, timeout=3.0)
            except Exception as e:  # fail-soft: es el shutdown; las tareas en vuelo se perderian igual al cerrar el pool tres lineas mas abajo, y no cerrar el pool por esto dejaria conexiones colgadas
                logger.error(f"Error esperando tareas pendientes: {e}")
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logger.info("MemoryDB pool cerrado")
            self.pool = None

    @property
    def is_connected(self) -> bool:
        return self.pool is not None

    # --------------------------------------------------------
    # Health check (incluye verificacion de VECTOR)
    # --------------------------------------------------------
    @db_error_handler
    async def health_check(self) -> Optional[bool]:
        """Verifica que la base responde, que VECTOR funciona y que el esquema
        esta al dia.

        El esquema entra aca a proposito (2026-09-16): una base que responde
        pero a la que le falta una columna o un backfill NO esta sana -- sirve
        menos datos de los que tiene y no da error. Un flag que nadie consulta
        es el mismo defecto que se acaba de arreglar en jacobs/reaper.py
        (`error: True` escrito y jamas leido), asi que este se lee aqui.
        """
        if not self.pool:
            return None
        if self.schema_ok is False:
            return False
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT VEC_ToText(VEC_FromText('[1,2,3]'))")
                row = await cur.fetchone()
                return row is not None

    # --------------------------------------------------------
    # Embeddings (vectorizacion via Ollama local)
    # --------------------------------------------------------
    async def get_embedding(self, text: str) -> Optional[list]:
        """Vectoriza texto con el modelo configurado (EMBED.model) via Ollama local.
        Devuelve lista de EMBED.dim floats, o None si falla (JAX sigue sin embeddings).
        Una dimension distinta de la configurada se descarta: escribirla en la
        columna fallaria, o peor, compararia vectores de modelos distintos."""
        try:
            # Tope de contexto de los modelos; truncamos para evitar 500.
            texto = text[:4000] if len(text) > 4000 else text
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    "http://localhost:11434/api/embed",
                    json={"model": EMBED.model, "input": texto},
                )
                resp.raise_for_status()
                embeddings = resp.json().get("embeddings") or []
                embedding = embeddings[0] if embeddings else None
                if isinstance(embedding, list) and len(embedding) == EMBED.dim:
                    return embedding
                logger.warning(
                    f"Embedding con dimension incorrecta: "
                    f"{len(embedding) if embedding else None}"
                )
                return None
        except Exception as e:  # fail-soft: el fallo se reporta como return None, los callers lo chequean antes de usarlo, y la fila que queda en vector cero la reintenta backfill_zero_embeddings() en la pasada siguiente del worker
            logger.error(f"get_embedding fallo: {e}")
            return None

    async def backfill_zero_embeddings(self, table: str, limit: int = 50) -> dict:
        """Reintenta el embedding de hasta `limit` filas de `table` que siguen
        en vector cero. Es el reintento que save_message() y add_fact() no
        hacen: insertan primero y vectorizan despues, y si Ollama falla la
        fila queda en ceros -- excluida de toda busqueda (ver
        _nonzero_embedding_sql), o sea perdida en silencio. Lo corre el worker
        de memoria en cada pasada.

        Una fila cuyo embedding vuelve a fallar se deja como esta (no se
        inventa un vector) y se reintenta en la pasada siguiente. El UPDATE
        esta guardado por "sigue en ceros": correrlo dos veces, o en paralelo
        con un save_message() tardio, no pisa un embedding real.

        Devuelve {"pendientes", "reparadas", "fallidas"} de ESTA pasada."""
        texto_col = _BACKFILL_TABLES.get(table)
        if texto_col is None:
            raise ValueError(f"backfill_zero_embeddings: tabla no permitida {table!r}")
        resultado = {"pendientes": 0, "reparadas": 0, "fallidas": 0}
        if not self.pool:
            return resultado

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT id, {texto_col} FROM {table} "
                    f"WHERE {_zero_embedding_sql(_col())} ORDER BY id LIMIT %s",
                    (limit,),
                )
                filas = await cur.fetchall()
        resultado["pendientes"] = len(filas)

        for fila_id, texto in filas:
            embedding = await self.get_embedding(texto)
            if _is_degenerate_embedding(embedding):
                resultado["fallidas"] += 1
                continue
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"UPDATE {table} SET {_col()} = VEC_FromText(%s) "
                        f"WHERE id = %s AND {_zero_embedding_sql(_col())}",
                        (json.dumps(embedding), fila_id),
                    )
                    resultado["reparadas"] += cur.rowcount

        if resultado["fallidas"]:
            logger.warning(
                f"backfill_zero_embeddings({table}): {resultado['fallidas']} fila(s) "
                f"siguen en ceros (get_embedding sin resultado); se reintentan en la "
                f"proxima pasada"
            )
        return resultado

    # --------------------------------------------------------
    # Conversaciones
    # --------------------------------------------------------
    @db_error_handler
    async def start_conversation(self, source: str = "terminal",
                                 user_id: Optional[int] = None,
                                 tenant_id: Optional[int] = None,
                                 project_id: Optional[int] = None) -> Optional[str]:
        """Crea una conversacion nueva. Devuelve su UUID (o None si fallo).

        Scope de dos niveles (opcional, retrocompatible):
          - project_id NOT NULL -> memoria de PROYECTO (compartida por el equipo).
          - project_id NULL     -> memoria INDIVIDUAL de user_id (privada).
        """
        if not self.pool:
            return None
        conv_uuid = str(uuid.uuid4())
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO conversations "
                    "(conversation_uuid, source, started_at, user_id, tenant_id, project_id) "
                    "VALUES (%s, %s, NOW(), %s, %s, %s)",
                    (conv_uuid, source, user_id, tenant_id, project_id),
                )
        logger.info(f"Conversacion iniciada: {conv_uuid[:8]} ({source}) "
                    f"user={user_id} project={project_id}")
        return conv_uuid

    @db_error_handler
    async def end_conversation(self, conversation_uuid: Optional[str]) -> Optional[bool]:
        """Marca la conversacion como terminada y lista para el worker de memoria."""
        if not self.pool or not conversation_uuid:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE conversations "
                    "SET ended_at = NOW(), memory_processed = FALSE "
                    "WHERE conversation_uuid = %s",
                    (conversation_uuid,),
                )
        logger.info(f"Conversacion cerrada: {conversation_uuid[:8]}")
        return True

    # --------------------------------------------------------
    # Mensajes (fire-and-forget)
    # --------------------------------------------------------
    def save_message(self, conversation_uuid: Optional[str], role: str, content: str,
                     facet: Optional[str] = None, model: Optional[str] = None,
                     latency_ms: Optional[int] = None) -> "Optional[asyncio.Task]":
        """Lanza el guardado en background SIN esperar (latencia 0 para JAX).

        Nota: NO es async a proposito — se llama sin await desde el REPL, que
        sigue ignorando el valor de retorno igual que siempre (cero cambio de
        comportamiento ahi). Devuelve el Task en vez de None para que un
        caller que SI necesite confirmacion (ej. Mesa web, shadow validation)
        pueda opcionalmente `await` lo que ya se le devuelve hoy y descarta:
        `task = db.save_message(...); result = await task` da
        `{"conversation_id": int, "turn_number": int}` si guardo, `None` si
        fallo (error ya logueado por @db_error_handler, este solo expone el
        veredicto al caller que lo pida). Ver
        save-message-fire-and-forget-sin-garantia (memoria, 2026-08-18).

        La tarea se registra en _pending_tasks para que el GC no la mate."""
        if not self.pool or not conversation_uuid:
            return None
        task = asyncio.create_task(
            self._save_message_impl(conversation_uuid, role, content,
                                    facet, model, latency_ms)
        )
        # Referencia fuerte + auto-limpieza al terminar (fix tasks huerfanas)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    @db_error_handler
    async def _save_message_impl(self, conversation_uuid, role, content,
                                 facet, model, latency_ms):
        """Guardado real, corre en background. Errores se tragan y se
        loguean (decorador) -- devuelve None en ese caso. Si guarda bien,
        devuelve {"conversation_id": int, "turn_number": int} para que un
        caller que awaitee el Task de save_message() tenga con que
        identificar la fila real, sin FK nuevo ni cambiar el schema."""
        role_enum = _normalize_role(role)
        facet_enum = _normalize_role(facet) if facet else None

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                # 1. conversation_id desde el uuid
                # El scope viaja con el id: se copia al mensaje para que la
                # busqueda semantica pueda filtrar SIN unir con conversations
                # (ese JOIN saca al optimizador del indice vectorial HNSW --
                # 58,5 ms contra 0,4 ms medidos el 2026-09-11). Es seguro
                # copiarlo porque el scope de una conversacion es inmutable:
                # ningun UPDATE del arbol lo toca, y hay un test centinela que
                # se pone rojo si alguien lo hace mutable.
                await cur.execute(
                    "SELECT id, user_id, project_id FROM conversations "
                    "WHERE conversation_uuid = %s",
                    (conversation_uuid,),
                )
                row = await cur.fetchone()
                if not row:
                    logger.error(f"Conversacion no encontrada: {conversation_uuid[:8]}")
                    return
                conv_id, conv_user_id, conv_project_id = row[0], row[1], row[2]

                # 2. turn_number = ultimo + 1
                await cur.execute(
                    "SELECT COALESCE(MAX(turn_number), 0) + 1 FROM messages "
                    "WHERE conversation_id = %s",
                    (conv_id,),
                )
                turn = (await cur.fetchone())[0]

                # 3. insertar el mensaje (role ya normalizado al ENUM)
                await cur.execute(
                    "INSERT INTO messages "
                    "(conversation_id, turn_number, role, content, facet_used, model, "
                    "latency_ms, user_id, project_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (conv_id, turn, role_enum, content, facet_enum, model, latency_ms,
                     conv_user_id, conv_project_id),
                )

                # 4. actualizar contador de turnos
                await cur.execute(
                    "UPDATE conversations SET total_turns = total_turns + 1 WHERE id = %s",
                    (conv_id,),
                )

        # 5. vectorizar fuera del bloque — no retiene conexion mientras Ollama trabaja
        embedding = await self.get_embedding(content)
        if embedding and self.pool:
            vec_str = json.dumps(embedding)
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"UPDATE messages SET {_col()} = VEC_FromText(%s) "
                        "WHERE conversation_id = %s AND turn_number = %s",
                        (vec_str, conv_id, turn),
                    )

        logger.debug(f"Mensaje guardado: conv={conversation_uuid[:8]} turn={turn} role={role_enum}")
        return {"conversation_id": conv_id, "turn_number": turn}

    # --------------------------------------------------------
    # Metodos para el WORKER de extraccion (batch, post-conversacion)
    # --------------------------------------------------------
    @db_error_handler
    async def get_unprocessed_conversations(self, limit: int = 10) -> Optional[list]:
        """Devuelve conversaciones cerradas que el worker aun no proceso.
        Retorna lista de dicts {id, uuid, user_id, project_id} o None si fallo.
        user_id/project_id viajan para que los facts hereden el scope de origen."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, conversation_uuid, user_id, project_id FROM conversations "
                    "WHERE ended_at IS NOT NULL AND memory_processed = FALSE "
                    "ORDER BY ended_at ASC LIMIT %s",
                    (limit,),
                )
                rows = await cur.fetchall()
                return [{"id": r[0], "uuid": r[1], "user_id": r[2],
                         "project_id": r[3]} for r in rows]

    @db_error_handler
    async def get_conversation_messages(self, conv_id: int) -> Optional[list]:
        """Trae los mensajes de una conversacion, en orden.
        Retorna lista de dicts {role, content} o None si fallo."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT role, content FROM messages "
                    "WHERE conversation_id = %s ORDER BY turn_number ASC",
                    (conv_id,),
                )
                rows = await cur.fetchall()
                return [{"role": r[0], "content": r[1]} for r in rows]

    @db_error_handler
    async def get_last_session_messages(self, limit: int = 20) -> Optional[list]:
        """Trae los ultimos N mensajes de la conversacion mas reciente terminada.
        Devuelve lista de dicts {role, content} en orden cronologico, o None."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id FROM conversations WHERE ended_at IS NOT NULL "
                    "ORDER BY ended_at DESC LIMIT 1"
                )
                row = await cur.fetchone()
                if not row:
                    return []
                last_conv_id = row[0]
                await cur.execute(
                    "SELECT role, content FROM messages "
                    "WHERE conversation_id = %s ORDER BY turn_number DESC LIMIT %s",
                    (last_conv_id, limit)
                )
                rows = await cur.fetchall()
                return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    async def _find_nearest_fact(self, embedding: list, user_id: Optional[int],
                                 project_id: Optional[int]) -> Optional[dict]:
        """Busca el fact ACTIVO (superseded_by IS NULL) mas cercano al
        embedding dado, scoped por user_id/project_id igual que
        search_similar_messages. None si no hay pool, no hay embedding, o
        no hay ningun fact con embedding real en ese scope (fail-safe: el
        caller inserta). Los facts con vector cero no son candidatos: su
        distancia es NaN y puede llegar como 0.0, que add_fact leeria como un
        duplicado exacto (ver _nonzero_embedding_sql)."""
        if not self.pool or _is_degenerate_embedding(embedding):
            return None
        vec_str = json.dumps(embedding)
        clauses = ["superseded_by IS NULL", _nonzero_embedding_sql(_col())]
        params: list = []
        scope = []
        if project_id is not None:
            scope.append("project_id = %s")
            params.append(project_id)
        if user_id is not None:
            scope.append("(project_id IS NULL AND user_id = %s)")
            params.append(user_id)
        if scope:
            clauses.append("(" + " OR ".join(scope) + ")")
        where = " AND ".join(clauses)
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute(
                        "SELECT id, fact_text, "
                        f"VEC_DISTANCE_COSINE({_col()}, VEC_FromText(%s)) AS distancia "
                        f"FROM facts WHERE {where} "
                        f"ORDER BY VEC_DISTANCE_COSINE({_col()}, VEC_FromText(%s)) ASC "
                        "LIMIT 1",
                        ([vec_str] + params + [vec_str]),
                    )
                    row = await cur.fetchone()
                    if not row:
                        return None
                    filas = _finite_distance_rows([dict(row)], "_find_nearest_fact")
                    return filas[0] if filas else None
        except Exception as e:
            logger.error(f"_find_nearest_fact fallo: {e}")
            raise BusquedaDeFactFallida(str(e)) from e

    @db_error_handler
    async def supersede_fact(self, old_fact_id: int, new_fact_id: int) -> Optional[bool]:
        """Marca old_fact_id como reemplazado por new_fact_id. No borra nada:
        la historia de una correccion queda reconstruible."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE facts SET superseded_by = %s, superseded_at = NOW() "
                    "WHERE id = %s",
                    (new_fact_id, old_fact_id),
                )
        return True

    @db_error_handler
    async def save_fact(self, fact_text: str, fact_type: str,
                        source_message_id: Optional[int] = None,
                        source_facet: Optional[str] = None,
                        confidence: float = 0.7,
                        user_id: Optional[int] = None,
                        project_id: Optional[int] = None,
                        is_correction: bool = False,
                        verify_correction_fn: Optional[
                            Callable[[str, str], Awaitable[bool]]] = None,
                        source_fact_ids: Optional[list] = None,
                        importance: Optional[int] = None,
                        ) -> Optional[bool]:
        """Guarda un hecho extraido. confidence 0.7 + is_verified=FALSE por
        defecto: nada entra como verdad absoluta sin que Fernando lo revise.

        Scope de dos niveles: project_id NOT NULL -> fact de proyecto (compartido);
        NULL -> fact individual de user_id.

        Antes de insertar, busca el fact activo mas cercano en el mismo scope:
          - 'duplicate' y NO is_correction -> no inserta (devuelve False).
          - is_correction=True y HAY candidato ('duplicate' o
            'correction_candidate' — una correccion bien parecida en texto
            puede caer en cualquiera de las dos bandas) -> inserta y marca
            el viejo como superseded_by el nuevo, PERO solo si
            verify_correction_fn (cuando se pasa) confirma que el nuevo texto
            realmente contradice/actualiza al viejo. La distancia vectorial
            sola puede acertar el candidato equivocado por coincidencia de
            embedding (bug real documentado por Beelink) — verify_correction_fn
            es un chequeo extra opcional (ej: una llamada LLM dedicada desde
            worker.py), no obligatorio. Sin el, se confia en la distancia sola
            (comportamiento de antes).
          - cualquier otro caso (incluido 'unrelated', o sin candidato, o sin
            embedding) -> inserta normal. Fail-safe: ante duda, INSERT gana.

        source_fact_ids (opcional): lista de ids de facts de los que este
        fact fue derivado — usado por el worker de sintesis de segundo orden
        (item #8) para que un insight sea trazable hasta los hechos
        verificados que lo originaron. None para facts normales (extraccion
        directa de conversacion).

        importance (opcional, 1-5): que tan central es este hecho a la
        identidad o trabajo de Fernando (NO es lo mismo que confidence, que
        es la certeza del extractor). Se usa para priorizar que facts
        sobreviven el limite de get_facts() cuando hay mas de los que
        entran. Fuera de 1-5 o None -> se guarda NULL (neutral)."""
        if not self.pool:
            return None
        # Validar fact_type contra el ENUM del esquema
        valid_types = ("user", "technical", "social", "preference", "project", "financial")
        ftype = fact_type if fact_type in valid_types else "user"
        imp = _validate_importance(importance)

        # Embedding ANTES del insert: lo necesitamos para decidir dedup/correccion.
        embedding = await self.get_embedding(fact_text)
        try:
            candidate = await self._find_nearest_fact(embedding, user_id, project_id) \
                if embedding else None
        except BusquedaDeFactFallida as e:
            # FAIL-CLOSED hacia la insercion (2026-09-16). Si no se pudo mirar
            # que habia, no se puede afirmar que no habia nada.
            #   - En una CORRECCION, insertar es lo peor que se puede hacer: el
            #     fact erroneo se queda activo porque nadie lo supersede, y el
            #     nuevo entra al lado. Dos hechos contradictorios vivos. Mejor
            #     no guardar y que el extractor reintente.
            #   - En un fact normal, el unico riesgo de no mirar es duplicar, asi
            #     que se sigue, pero DICIENDOLO en el log: un duplicado callado
            #     es como empieza una memoria sucia.
            if is_correction:
                logger.error(
                    "save_fact: no se pudo buscar el fact a corregir (%s) -- NO se "
                    "inserta. Insertar dejaria el hecho erroneo activo y el nuevo al "
                    "lado, contradiciendose. Texto: %r",
                    e, fact_text[:80],
                )
                return None
            logger.warning(
                "save_fact: no se pudo buscar duplicados (%s) -- se inserta igual; "
                "puede quedar un duplicado. Texto: %r", e, fact_text[:80],
            )
            candidate = None
        band = classify_fact_distance(candidate["distancia"]) if candidate else "unrelated"

        if _should_skip_as_duplicate(band, is_correction):
            logger.info(f"save_fact: duplicado de fact {candidate['id']}, no se inserta "
                        f"({fact_text[:60]!r})")
            return False

        fact_id: Optional[int] = None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO facts "
                    "(fact_uuid, fact_text, fact_type, confidence, source_message_id, "
                    "source_facet, is_verified, user_id, project_id, source_fact_ids, "
                    "importance) "
                    "VALUES (UUID(), %s, %s, %s, %s, %s, FALSE, %s, %s, %s, %s)",
                    (fact_text, ftype, confidence, source_message_id, source_facet,
                     user_id, project_id,
                     json.dumps(source_fact_ids) if source_fact_ids else None,
                     imp),
                )
                await cur.execute("SELECT LAST_INSERT_ID()")
                fact_id = (await cur.fetchone())[0]

        if fact_id and embedding:
            vec_str = json.dumps(embedding)
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"UPDATE facts SET {_col()} = VEC_FromText(%s) WHERE id = %s",
                        (vec_str, fact_id),
                    )

        if fact_id and candidate is not None and is_correction and band != "unrelated":
            confirmado = True
            if verify_correction_fn is not None:
                try:
                    confirmado = await verify_correction_fn(candidate["fact_text"], fact_text)
                except Exception as e:  # fail-soft: falla CERRADO hacia la accion destructiva -- si el verificador no confirma no se hace supersede, o sea el fact viejo sigue activo en vez de desaparecer por un error del verificador
                    # Fail-safe: si el chequeo extra falla, NO se aplica la
                    # correccion (el candidato podria ser el equivocado) pero
                    # el fact nuevo ya quedo insertado igual.
                    logger.error(f"verify_correction_fn fallo: {e}, no se aplica la correccion")
                    confirmado = False
            if _should_supersede(True, band, is_correction, confirmado):
                # ARREGLADO 2026-09-16: antes se llamaba a supersede_fact y se
                # escribia "fact N corrige a fact M" SIN MIRAR el resultado.
                # supersede_fact esta decorado con db_error_handler, o sea que
                # ante cualquier error devuelve None en silencio -- y el log
                # afirmaba una correccion que no habia ocurrido, con el hecho
                # viejo todavia activo al lado del nuevo. Dos hechos
                # contradictorios vivos, y el registro diciendo lo contrario.
                #
                # Insertar y supersedar son UNA operacion en intencion: si la
                # segunda mitad no se puede completar, la primera se deshace.
                # Mejor no guardar la correccion y que el extractor reintente,
                # que dejar la memoria contradiciendose.
                if await self.supersede_fact(candidate["id"], fact_id):
                    logger.info(f"save_fact: fact {fact_id} corrige a fact {candidate['id']} "
                                f"(banda={band})")
                else:
                    revertido = await self.delete_fact(fact_id)
                    if revertido:
                        logger.error(
                            "save_fact: no se pudo marcar el fact %d como reemplazado por "
                            "el %d; se revirtio la insercion para no dejar dos hechos "
                            "contradictorios activos. La correccion NO quedo guardada: %r",
                            candidate["id"], fact_id, fact_text[:80],
                        )
                    else:
                        logger.critical(
                            "save_fact: no se pudo marcar el fact %d como reemplazado por "
                            "el %d, y TAMPOCO se pudo revertir la insercion del %d. La "
                            "memoria tiene AHORA MISMO dos hechos contradictorios activos "
                            "y hay que resolverlo a mano: %r",
                            candidate["id"], fact_id, fact_id, fact_text[:80],
                        )
                    return None
            else:
                logger.info(f"save_fact: fact {fact_id} NO confirmo correccion sobre "
                            f"candidato {candidate['id']} (banda={band}), queda como fact nuevo")

        return True

    @db_error_handler
    async def save_decision(self, title: str, chosen: str, reasoning: str,
                            context: Optional[str] = None,
                            user_id: Optional[int] = None,
                            project_id: Optional[int] = None) -> Optional[bool]:
        """Guarda una decision extraida (con scope de dos niveles)."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO decisions "
                    "(decision_uuid, title, context, chosen_option, reasoning, "
                    "user_id, project_id) "
                    "VALUES (UUID(), %s, %s, %s, %s, %s, %s)",
                    (title, context, chosen, reasoning, user_id, project_id),
                )
        return True

    @db_error_handler
    async def save_action_item(self, description: str,
                               due_date: Optional[str] = None,
                               source_conversation_id: Optional[int] = None,
                               user_id: Optional[int] = None,
                               project_id: Optional[int] = None) -> Optional[bool]:
        """Guarda un pendiente extraido (con scope de dos niveles)."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO action_items "
                    "(action_uuid, description, due_date, source_conversation_id, status, "
                    "user_id, project_id) "
                    "VALUES (UUID(), %s, %s, %s, 'pending', %s, %s)",
                    (description, due_date, source_conversation_id, user_id, project_id),
                )
        return True

    @db_error_handler
    async def get_decisions(self, limit: int = 20,
                            user_id: Optional[int] = None,
                            project_id: Optional[int] = None) -> Optional[list]:
        """Lista decisiones registradas, mas recientes primero (comando
        /decisions). Scope de dos niveles opcional, igual que get_facts
        (sin scope si ambos None -- uso simple del REPL)."""
        if not self.pool:
            return None
        query = ("SELECT id, title, context, chosen_option, reasoning, outcome, "
                 "made_by_facet, made_at, project_id FROM decisions")
        conditions = []
        params: list = []
        scope_clauses = []
        if project_id is not None:
            scope_clauses.append("project_id = %s")
            params.append(project_id)
        if user_id is not None:
            scope_clauses.append("(project_id IS NULL AND user_id = %s)")
            params.append(user_id)
        if scope_clauses:
            conditions.append("(" + " OR ".join(scope_clauses) + ")")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY made_at DESC LIMIT %s"
        params.append(limit)
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, tuple(params))
                return await cur.fetchall()

    @db_error_handler
    async def get_action_items(self, limit: int = 20,
                               status: Optional[str] = "pending",
                               user_id: Optional[int] = None,
                               project_id: Optional[int] = None) -> Optional[list]:
        """Lista pendientes registrados, mas recientes primero (comando
        /pendientes). status=None trae todos los estados; por defecto solo
        'pending' (lo accionable, que es lo que Fernando revisa)."""
        if not self.pool:
            return None
        query = ("SELECT id, description, status, due_date, reminder_date, "
                 "context_facet, completed_at, created_at, project_id FROM action_items")
        conditions = []
        params: list = []
        if status:
            conditions.append("status = %s")
            params.append(status)
        scope_clauses = []
        if project_id is not None:
            scope_clauses.append("project_id = %s")
            params.append(project_id)
        if user_id is not None:
            scope_clauses.append("(project_id IS NULL AND user_id = %s)")
            params.append(user_id)
        if scope_clauses:
            conditions.append("(" + " OR ".join(scope_clauses) + ")")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, tuple(params))
                return await cur.fetchall()

    @db_error_handler
    async def mark_action_item_done(self, item_id: int) -> Optional[bool]:
        """Marca un pendiente como completado (comando /pendientes done)."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE action_items SET status='done', completed_at=NOW() "
                    "WHERE id=%s",
                    (item_id,),
                )
                return cur.rowcount > 0

    @db_error_handler
    async def save_person(self, name: str, nickname: Optional[str] = None) -> Optional[bool]:
        """Registra una persona (comando /person new). Escritor EXPLICITO
        (mismo criterio que projects, ronda 4): reconocer una persona es una
        decision humana consciente, no algo que JAX deba inferir de una
        conversacion. NO puebla honor_memory (pendiente deliberado,
        semantica de Fernando -- ver CONTEXT.md, no proponer significado
        aca). `name` es NOT NULL en el schema real -- hueco del diseño de
        ronda 7 (solo listaba nickname/relationship_start/last_mentioned),
        detectado al implementar."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO people (person_uuid, name, nickname, relationship_start) "
                    "VALUES (UUID(), %s, %s, CURDATE())",
                    (name, nickname),
                )
        return True

    @db_error_handler
    async def get_people(self, limit: int = 20) -> Optional[list]:
        """Lista personas registradas, mas recientes primero (comando
        /person list)."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT id, person_uuid, name, nickname, relationship_start, "
                    "last_mentioned, honor_memory FROM people "
                    "ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
                return await cur.fetchall()

    @db_error_handler
    async def touch_person_mentions(self, names_or_nicknames: list) -> Optional[int]:
        """Actualiza last_mentioned=CURDATE() para las personas cuyo name o
        nickname aparece en `names_or_nicknames` (llamado desde el worker de
        destilacion, jax/memory/worker.py, sobre las conversaciones que ya
        procesa cada 20 min -- reusa esa deteccion en vez de construir una
        nueva, tal como diseñado en ronda 7). Devuelve cuantas filas se
        actualizaron, o None si fallo."""
        if not self.pool or not names_or_nicknames:
            return 0
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                placeholders = ",".join(["%s"] * len(names_or_nicknames))
                await cur.execute(
                    f"UPDATE people SET last_mentioned=CURDATE() "
                    f"WHERE name IN ({placeholders}) OR nickname IN ({placeholders})",
                    (*names_or_nicknames, *names_or_nicknames),
                )
                return cur.rowcount

    @db_error_handler
    async def mark_processed(self, conv_id: int) -> Optional[bool]:
        """Marca la conversacion como ya procesada por el worker."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE conversations SET memory_processed = TRUE, "
                    "memory_processed_at = NOW() WHERE id = %s",
                    (conv_id,),
                )
        return True

    # --------------------------------------------------------
    # Busqueda semantica
    # --------------------------------------------------------
    @db_error_handler
    async def contar_filas(self, tabla: str) -> Optional[int]:
        """Cuenta las filas de una tabla del esquema de memoria.

        Lista blanca y no interpolacion libre: el nombre de tabla no se puede
        parametrizar en SQL, asi que la unica forma segura es que solo existan
        los nombres que este modulo conoce.
        """
        if tabla not in ("messages", "facts", "conversations"):
            raise ValueError(f"tabla no permitida: {tabla!r}")
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"SELECT COUNT(*) FROM {tabla}")
                return (await cur.fetchone())[0]

    async def search_similar_messages(self, query: str, limit: int = 5,
                                      user_id: Optional[int] = None,
                                      project_id: Optional[int] = None,
                                      recent_history: Optional[list] = None) -> list:
        """Busca mensajes similares a query usando distancia vectorial.

        Scope de dos niveles (opcional):
          - project_id NOT NULL -> incluye memoria del PROYECTO (compartida).
          - user_id    NOT NULL -> incluye memoria INDIVIDUAL (project_id IS NULL).
          - ambos None          -> sin filtro de scope (global; retrocompat REPL viejo).
        recent_history (opcional): ultimos turnos [{"role":.., "content":..}] de
        la conversacion en curso. Si se pasa, el embedding se calcula sobre
        query + esos turnos (no solo la ultima frase), asi una respuesta corta
        y ambigua ("y eso por que?") sigue trayendo contexto relevante.
        Devuelve lista de dicts {content, role, created_at, started_at, distancia}.
        Si Ollama falla, la base falla, o no hay embeddings: devuelve [] (nunca None)."""
        if not self.pool:
            return []

        blended_query = _blend_query(query, recent_history)
        embedding = await self.get_embedding(blended_query)
        if embedding is None:
            return []
        if _is_degenerate_embedding(embedding):
            logger.warning("search_similar_messages: embedding de consulta de norma cero, sin busqueda")
            return []

        vec_str = json.dumps(embedding)

        # --- Busqueda dual (item nuevo, sugerido por el tutorial de Beelink
        # v2): ademas del embedding mezclado con el historial, tambien se
        # busca con el mensaje CRUDO cuando el blend cambio algo. Una
        # pregunta nueva sin relacion con los ultimos turnos puede diluirse
        # si solo se busca con la version mezclada. Solo se paga el costo de
        # un segundo embedding/query cuando realmente hay blend (recent_history
        # no vacio Y distinto de query) — sin historial, es exactamente el
        # comportamiento de antes (una sola busqueda).
        raw_vec_str: Optional[str] = None
        if recent_history and blended_query != query:
            raw_embedding = await self.get_embedding(query)
            if raw_embedding is not None and not _is_degenerate_embedding(raw_embedding):
                raw_vec_str = json.dumps(raw_embedding)

        # --- WHERE de scope (dos dimensiones) -----------------------------
        # (project_id = P)  OR  (project_id IS NULL AND user_id = U)
        # El scope se filtra sobre `messages` (columnas desnormalizadas desde
        # conversations), NO uniendo con conversations: el JOIN dejaba fuera al
        # indice vectorial HNSW y convertia cada busqueda en un scan.
        scope_sql = ""
        scope_params: list = []
        clauses = []
        if project_id is not None:
            clauses.append("m.project_id = %s")
            scope_params.append(project_id)
        if user_id is not None:
            clauses.append("(m.project_id IS NULL AND m.user_id = %s)")
            scope_params.append(user_id)
        # Las filas con vector cero nunca son candidatas, haya scope o no
        # (ver _nonzero_embedding_sql).
        where = [_nonzero_embedding_sql(_col("m."))]
        if clauses:
            where.append("(" + " OR ".join(clauses) + ")")
        scope_sql = "WHERE " + " AND ".join(where) + " "
        # ------------------------------------------------------------------

        # --- Decay temporal (item #6, OPCIONAL) ----------------------------
        # DECAY_LAMBDA=0.0 por defecto: se pide exactamente `limit` filas y
        # el orden queda IDENTICO al de antes de este bloque (mismo query
        # SQL de siempre, sin tocar).
        #
        # Se rechazo prenderlo por defecto porque a la escala actual de JAX
        # (~100 facts) no hace falta y en Beelink causo una falla silenciosa
        # de retrieval por horas (ver memoria
        # beelink-ai-dashboard-memory-system.md). Revisar solo si facts
        # escala a miles de filas.
        #
        # El decay se aplica EN PYTHON sobre el pool de candidatos, no en
        # SQL: un intento inicial de sumar el decay dentro de
        # VEC_DISTANCE_COSINE() en la propia query rompio contra filas con
        # embedding "vector cero" (mensajes sin embedding real, default del
        # esquema) con "DOUBLE value is out of range" — la MISMA clase de
        # fragilidad de datos que ya afecto a Beelink. Haciendolo en Python
        # sobre un pool ya trardo, un embedding cero simplemente da un mal
        # ranking en ese candidato puntual, nunca rompe la query entera.
        decay_lambda = float(os.getenv("JAX_MEMORY_DECAY_LAMBDA", "0.0"))
        rerank_enabled = os.getenv("JAX_MEMORY_RERANK") == "1"
        fetch_limit = limit * 3 if (decay_lambda or rerank_enabled) else limit

        async def _run(v: str) -> list:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute(
                        "SELECT m.content, m.role, m.created_at, "
                        "m.created_at AS started_at, "
                        f"VEC_DISTANCE_COSINE({_col('m.')}, VEC_FromText(%s)) AS distancia "
                        "FROM messages m "
                        + scope_sql +
                        f"ORDER BY VEC_DISTANCE_COSINE({_col('m.')}, VEC_FromText(%s)) ASC "
                        "LIMIT %s",
                        ([v] + scope_params + [v, fetch_limit]),
                    )
                    return _finite_distance_rows(
                        [dict(r) for r in await cur.fetchall()], "search_similar_messages")

        try:
            rows = await _run(vec_str)
            if raw_vec_str:
                rows_raw = await _run(raw_vec_str)
                rows = _merge_search_results(rows, rows_raw, fetch_limit)
        except Exception as e:
            logger.error(f"search_similar_messages fallo: {e}")
            return []

        if decay_lambda:
            def _decayed(r: dict) -> float:
                # distancia reportada NO se toca (el filtro < 0.8 rio arriba
                # sigue leyendo cosine puro) — el decay solo reordena candidatos.
                try:
                    edad_dias = (datetime.now() - r["created_at"]).days
                    return r["distancia"] + decay_lambda * edad_dias
                except Exception:  # fail-soft: el decay solo reordena candidatos y esta apagado por defecto (JAX_MEMORY_DECAY_LAMBDA=0.0); sin fecha usable ese candidato conserva su distancia cosine real
                    return r["distancia"]  # fail-safe: sin fecha usable, no decae
            rows.sort(key=_decayed)

        # --- Reranking con cross-encoder (item #7, OPCIONAL) ----------------
        # JAX_MEMORY_RERANK=1 para activarlo. Sin la env var, o sin el paquete
        # sentence-transformers instalado, esto es un no-op total (fail-safe
        # de disponibilidad): el orden queda igual al de siempre.
        if rerank_enabled and rows:
            reranker = _get_reranker()
            if reranker is not None:
                try:
                    pares = [(blended_query, r["content"]) for r in rows]
                    scores = reranker.predict(pares)
                    for r, s in zip(rows, scores):
                        r["_rerank_score"] = float(s)
                    rows.sort(key=lambda r: r["_rerank_score"], reverse=True)
                    for r in rows:
                        del r["_rerank_score"]
                except Exception as e:  # fail-soft: el reranking es opcional (item #7) y solo reordena; si falla, rows queda en el orden por distancia cosine que ya traia de SQL
                    logger.error(f"reranking fallo, se usa el orden previo: {e}")

        return rows[:limit]

    # --------------------------------------------------------
    # Gestion de facts (comando /fact: control de calidad)
    # --------------------------------------------------------
    @db_error_handler
    async def get_facts(self, only_unverified: bool = True,
                        only_verified: bool = False,
                        fact_type: Optional[str] = None,
                        limit: int = 20,
                        user_id: Optional[int] = None,
                        project_id: Optional[int] = None) -> Optional[list]:
        """Lista facts ACTIVOS (superseded_by IS NULL — un fact corregido
        nunca vuelve a aparecer aca). Por defecto solo los no verificados
        (a revisar). only_verified=True hace lo opuesto: solo facts que
        Fernando ya reviso (usado por el sintetizador de segundo orden,
        item #8 — nunca sintetiza sobre ruido no confirmado).
        Scope de dos niveles opcional (igual que search_similar_messages):
          - project_id NOT NULL -> facts del proyecto; user_id -> facts individuales.
          - ambos None -> sin filtro de scope (retrocompat).
        Devuelve lista de dicts o None si fallo."""
        if not self.pool:
            return None
        query = ("SELECT id, fact_text, fact_type, confidence, is_verified, "
                 "source_facet, created_at, source_fact_ids, importance FROM facts")
        conditions = ["superseded_by IS NULL"]
        params: list = []
        if only_unverified:
            conditions.append("is_verified = FALSE")
        elif only_verified:
            conditions.append("is_verified = TRUE")
        if fact_type:
            conditions.append("fact_type = %s")
            params.append(fact_type)
        # Scope de dos dimensiones (project compartido / individual de user)
        scope_clauses = []
        if project_id is not None:
            scope_clauses.append("project_id = %s")
            params.append(project_id)
        if user_id is not None:
            scope_clauses.append("(project_id IS NULL AND user_id = %s)")
            params.append(user_id)
        if scope_clauses:
            conditions.append("(" + " OR ".join(scope_clauses) + ")")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        # Mas importante primero (NULL = neutral, va al final de los que si
        # tienen score) — asi si hay que cortar por `limit`, sobreviven los
        # facts mas centrales, no simplemente los mas recientes.
        query += " ORDER BY COALESCE(importance, 0) DESC, created_at DESC LIMIT %s"
        params.append(limit)

        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, tuple(params))
                return await cur.fetchall()

    @db_error_handler
    async def get_scopes_with_verified_facts(self, min_facts: int = 5) -> Optional[list]:
        """Devuelve los scopes (user_id, project_id) que tienen al menos
        min_facts facts verificados y activos. Usado por el sintetizador de
        segundo orden (item #8) para saber sobre que scopes vale la pena
        correr — nunca sintetiza sobre un scope con pocos facts."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT user_id, project_id, COUNT(*) AS n_facts FROM facts "
                    "WHERE is_verified = TRUE AND superseded_by IS NULL "
                    "GROUP BY user_id, project_id "
                    "HAVING COUNT(*) >= %s",
                    (min_facts,),
                )
                return await cur.fetchall()

    @db_error_handler
    async def verify_fact(self, fact_id: int) -> Optional[bool]:
        """Marca un fact como verificado. confidence NO se toca (es ortogonal:
        confidence = certeza del extractor, is_verified = validacion de Fernando)."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                affected = await cur.execute(
                    "UPDATE facts SET is_verified = TRUE, verified_at = NOW() "
                    "WHERE id = %s",
                    (fact_id,),
                )
                return affected > 0

    @db_error_handler
    async def delete_fact(self, fact_id: int) -> Optional[bool]:
        """Borra un fact. Irreversible — el caller debe confirmar antes."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                affected = await cur.execute(
                    "DELETE FROM facts WHERE id = %s", (fact_id,)
                )
                return affected > 0

    @db_error_handler
    async def get_fact_text(self, fact_id: int) -> Optional[str]:
        """Devuelve el texto de un fact (para mostrarlo al confirmar borrado)."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT fact_text FROM facts WHERE id = %s", (fact_id,)
                )
                row = await cur.fetchone()
                return row[0] if row else None
