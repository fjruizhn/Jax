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
import json
from datetime import datetime
from typing import Awaitable, Callable, Optional

import aiomysql
import httpx

logger = logging.getLogger("jax.memory")


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
        except Exception as e:
            logger.warning(f"reranker no disponible ({e}), se sigue sin reranking")
            _reranker = False
    return _reranker or None


def db_error_handler(func):
    """Decorador: cualquier error de DB se loguea y se traga.
    La conversacion NUNCA se interrumpe por un fallo de memoria."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"DB error en {func.__name__}: {e}", exc_info=True)
            return None
    return wrapper


class MemoryDB:
    """Memoria persistente de JAX sobre MariaDB."""

    def __init__(self):
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
            port = int(os.getenv("JAX_DB_PORT", "3306"))
        self.config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "db": database,
        }
        try:
            self.pool = await aiomysql.create_pool(
                minsize=1,
                maxsize=5,
                autocommit=True,
                charset="utf8mb4",
                **self.config,
            )
            logger.info(f"MemoryDB conectada a {database}@{host} (pool 1-5)")
            return True
        except Exception as e:
            logger.error(f"MemoryDB no pudo conectar: {e}")
            self.pool = None
            return False

    async def close(self):
        """Cierra el pool tras esperar las tareas pendientes de guardado."""
        # Esperamos a que terminen los guardados en vuelo (con timeout corto)
        if self._pending_tasks:
            try:
                await asyncio.wait(self._pending_tasks, timeout=3.0)
            except Exception as e:
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
        """Verifica que la base responde y que VECTOR funciona."""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT VEC_ToText(VEC_FromText('[1,2,3]'))")
                row = await cur.fetchone()
                return row is not None

    # --------------------------------------------------------
    # Embeddings (vectorizacion via Ollama local)
    # --------------------------------------------------------
    async def get_embedding(self, text: str) -> Optional[list]:
        """Vectoriza texto con nomic-embed-text via Ollama local.
        Devuelve lista de 768 floats, o None si falla (JAX sigue sin embeddings)."""
        try:
            # nomic-embed-text tiene limite de contexto; truncamos para evitar 500.
            texto = text[:4000] if len(text) > 4000 else text
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": texto},
                )
                resp.raise_for_status()
                embedding = resp.json().get("embedding")
                if isinstance(embedding, list) and len(embedding) == 768:
                    return embedding
                logger.warning(
                    f"Embedding con dimension incorrecta: "
                    f"{len(embedding) if embedding else None}"
                )
                return None
        except Exception as e:
            logger.error(f"get_embedding fallo: {e}")
            return None

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
                     latency_ms: Optional[int] = None) -> None:
        """Lanza el guardado en background SIN esperar (latencia 0 para JAX).

        Nota: NO es async a proposito — se llama sin await desde el REPL.
        La tarea se registra en _pending_tasks para que el GC no la mate."""
        if not self.pool or not conversation_uuid:
            return
        task = asyncio.create_task(
            self._save_message_impl(conversation_uuid, role, content,
                                    facet, model, latency_ms)
        )
        # Referencia fuerte + auto-limpieza al terminar (fix tasks huerfanas)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    @db_error_handler
    async def _save_message_impl(self, conversation_uuid, role, content,
                                 facet, model, latency_ms):
        """Guardado real, corre en background. Errores se tragan (decorador)."""
        role_enum = _normalize_role(role)
        facet_enum = _normalize_role(facet) if facet else None

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                # 1. conversation_id desde el uuid
                await cur.execute(
                    "SELECT id FROM conversations WHERE conversation_uuid = %s",
                    (conversation_uuid,),
                )
                row = await cur.fetchone()
                if not row:
                    logger.error(f"Conversacion no encontrada: {conversation_uuid[:8]}")
                    return
                conv_id = row[0]

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
                    "(conversation_id, turn_number, role, content, facet_used, model, latency_ms) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (conv_id, turn, role_enum, content, facet_enum, model, latency_ms),
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
                        "UPDATE messages SET embedding = VEC_FromText(%s) "
                        "WHERE conversation_id = %s AND turn_number = %s",
                        (vec_str, conv_id, turn),
                    )

        logger.debug(f"Mensaje guardado: conv={conversation_uuid[:8]} turn={turn} role={role_enum}")

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
        no hay ningun fact en ese scope (fail-safe: el caller inserta)."""
        if not self.pool or not embedding:
            return None
        vec_str = json.dumps(embedding)
        clauses = ["superseded_by IS NULL"]
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
                        "VEC_DISTANCE_COSINE(embedding, VEC_FromText(%s)) AS distancia "
                        f"FROM facts WHERE {where} "
                        "ORDER BY VEC_DISTANCE_COSINE(embedding, VEC_FromText(%s)) ASC "
                        "LIMIT 1",
                        ([vec_str] + params + [vec_str]),
                    )
                    row = await cur.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.error(f"_find_nearest_fact fallo: {e}")
            return None

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
        imp = importance if isinstance(importance, int) and 1 <= importance <= 5 else None

        # Embedding ANTES del insert: lo necesitamos para decidir dedup/correccion.
        embedding = await self.get_embedding(fact_text)
        candidate = await self._find_nearest_fact(embedding, user_id, project_id) \
            if embedding else None
        band = classify_fact_distance(candidate["distancia"]) if candidate else "unrelated"

        if band == "duplicate" and not is_correction:
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
                        "UPDATE facts SET embedding = VEC_FromText(%s) WHERE id = %s",
                        (vec_str, fact_id),
                    )

        if fact_id and is_correction and candidate is not None and band != "unrelated":
            confirmado = True
            if verify_correction_fn is not None:
                try:
                    confirmado = await verify_correction_fn(candidate["fact_text"], fact_text)
                except Exception as e:
                    # Fail-safe: si el chequeo extra falla, NO se aplica la
                    # correccion (el candidato podria ser el equivocado) pero
                    # el fact nuevo ya quedo insertado igual.
                    logger.error(f"verify_correction_fn fallo: {e}, no se aplica la correccion")
                    confirmado = False
            if confirmado:
                await self.supersede_fact(candidate["id"], fact_id)
                logger.info(f"save_fact: fact {fact_id} corrige a fact {candidate['id']} "
                            f"(banda={band})")
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
            if raw_embedding is not None:
                raw_vec_str = json.dumps(raw_embedding)

        # --- WHERE de scope (dos dimensiones) -----------------------------
        # (project_id = P)  OR  (project_id IS NULL AND user_id = U)
        scope_sql = ""
        scope_params: list = []
        clauses = []
        if project_id is not None:
            clauses.append("c.project_id = %s")
            scope_params.append(project_id)
        if user_id is not None:
            clauses.append("(c.project_id IS NULL AND c.user_id = %s)")
            scope_params.append(user_id)
        if clauses:
            scope_sql = "WHERE (" + " OR ".join(clauses) + ") "
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
                        "SELECT m.content, m.role, m.created_at, c.started_at, "
                        "VEC_DISTANCE_COSINE(m.embedding, VEC_FromText(%s)) AS distancia "
                        "FROM messages m "
                        "JOIN conversations c ON m.conversation_id = c.id "
                        + scope_sql +
                        "ORDER BY VEC_DISTANCE_COSINE(m.embedding, VEC_FromText(%s)) ASC "
                        "LIMIT %s",
                        ([v] + scope_params + [v, fetch_limit]),
                    )
                    return [dict(r) for r in await cur.fetchall()]

        try:
            rows = await _run(vec_str)
            if raw_vec_str:
                rows_raw = await _run(raw_vec_str)
                # Merge por (content, created_at): si el mismo mensaje salio
                # en las dos busquedas, se queda con la distancia mas chica.
                merged: dict[tuple, dict] = {}
                for r in rows + rows_raw:
                    key = (r["content"], r["created_at"])
                    if key not in merged or r["distancia"] < merged[key]["distancia"]:
                        merged[key] = r
                rows = sorted(merged.values(), key=lambda r: r["distancia"])[:fetch_limit]
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
                except Exception:
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
                except Exception as e:
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
