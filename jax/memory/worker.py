"""
JAX 2.0 — Worker de extraccion de memoria.

Proceso BATCH (no interactivo). Corre cada 15-30 min via systemd timer.
Lee las conversaciones cerradas que aun no fueron procesadas, le pide a un
extractor (LLM) que destile HECHOS, DECISIONES y PENDIENTES, y los guarda
en la memoria de JAX.

Filosofia (decidido por Claude + DeepSeek + Hipatia):
  - La memoria limpia es innegociable. Por eso el extractor es DeepSeek
    (confiable, JSON estable), no un modelo chico que pueda alucinar hechos
    y envenenar el pozo del que JAX bebe.
  - Es batch: la velocidad no importa, solo la calidad. Por eso vale usar
    el mejor extractor disponible aunque tarde unos segundos.
  - Nada entra como verdad: los hechos se guardan con confidence=0.7 y
    is_verified=FALSE. Fernando los revisa antes de que sean "verdad".
  - MIGRACION FUTURA: el dia que haya un modelo local potente (qwen 32B en
    GPU), se cambia el extractor pasando otro muscle a EXTRACTOR_FACTORY.
    El worker no cambia; solo cambia quien extrae.

Uso:
    set -a; source /etc/jax/.env; set +a
    PYTHONPATH=. .venv/bin/python -m jax.memory.worker

En memoria de Jairo Urbina.
"""

from __future__ import annotations

import asyncio
import json
import os
import logging

from jax.memory.db import MemoryDB
from jax.muscles.base import HttpMuscle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s: %(message)s",
)
logger = logging.getLogger("jax.memory.worker")


# Bloque de categorias prohibidas — compartido entre el extractor (worker.py)
# y el sintetizador de segundo orden (synthesis_worker.py). Una sola fuente
# de verdad: si esto cambia, cambia para los dos procesos que escriben facts.
FORBIDDEN_CATEGORIES_BLOCK = """NUNCA extraigas, sin excepcion, aunque parezca memorable o duradero (categorias prohibidas):
- Salud o condiciones medicas (fisicas o mentales, propias o de terceros).
- Raza, etnia, u origen nacional.
- Orientacion sexual o identidad de genero.
- Estado migratorio o de ciudadania.
- Afiliacion religiosa o creencias religiosas.
- Afiliacion o inclinacion politica.
- Datos financieros sensibles de terceros (salarios, deudas, patrimonio de personas que no son Fernando).
Esta regla tiene prioridad sobre "que si es memorable": si algo entra en estas categorias, NO
lo extraigas como hecho aunque sea duradero o relevante. Ante la duda de si algo cae en una
categoria prohibida, NO lo extraigas."""


# Prompt de extraccion. Acotado, pide JSON puro sin adornos.
EXTRACTION_PROMPT = """Analiza esta conversacion entre Fernando y JAX (su asistente personal).
Tu trabajo es extraer SOLO lo que valga la pena recordar DENTRO DE SEIS MESES.
Se ESTRICTO: ante la duda, NO lo extraigas. Memoria con ruido es peor que memoria vacia.

QUE SI es memorable (extraelo):
- Preferencias estables de Fernando ("prefiere disenar la arquitectura antes de codificar").
- Datos personales duraderos (rol, empresa, familia, ubicacion, herramientas que usa).
- Proyectos y su estado ("JAX corre en hall9000", "el producto financiero es SaaS para bancos").
- Decisiones tecnicas o de negocio con su razon ("eligio MariaDB sobre Postgres porque...").
- Relaciones importantes (personas, socios, clientes).
- Pendientes concretos y accionables que quedaron sin resolver.

QUE NO es memorable (IGNORALO, no lo extraigas):
- Preguntas puntuales o curiosidades del momento ("pregunto que significa tal palabra").
- El hecho de que Fernando este probando, saludando, o charlando casual.
- Pedidos de informacion que ya fueron respondidos en la misma conversacion.
- Cualquier cosa que no aporte conocimiento duradero sobre Fernando o sus proyectos.

""" + FORBIDDEN_CATEGORIES_BLOCK + """

EJEMPLOS:
- MAL (NO extraer): "Fernando pregunto por el significado de una palabra rara."
- MAL (NO extraer): "Fernando esta haciendo una prueba de memoria."
- BIEN (extraer): "Fernando es banquero de tercera generacion y disena productos financieros para LATAM."
- BIEN (extraer): "Fernando prefiere infraestructura propia y desconfia de depender de servicios de terceros."
- BIEN (extraer, is_correction=true): Fernando dijo antes que el modelo activo era
  qwen3-coder:30b y en esta conversacion dice que ahora es gpt-oss:120b.

CORRECCIONES: si un hecho de esta conversacion CONTRADICE o ACTUALIZA algo que
Fernando dijo antes (ej: "el modelo activo es X" cuando antes habia dicho que
era Y; "ya no trabajo en tal empresa"), marca ese hecho con "is_correction":
true. Si es informacion nueva que no reemplaza nada anterior, "is_correction":
false. Ante la duda, false (fail-safe: mejor un hecho nuevo de mas que una
correccion aplicada de menos).

IMPORTANCIA: para cada hecho, calificalo de 1 a 5 en "importance" segun que tan
CENTRAL es a la identidad o al trabajo de Fernando (NO es que tan seguro estas
de que sea cierto — eso ya lo cubre confidence, no lo repitas aca):
1 = detalle menor o de contexto pasajero.
3 = relevante pero no definitorio.
5 = definitorio: quien es, a que se dedica, decisiones que dan forma a su trabajo.
Ante la duda entre dos valores, elegi el mas bajo.

Categorias:
1. HECHOS: conocimiento duradero sobre Fernando, sus preferencias, proyectos o relaciones.
2. DECISIONES: elecciones tecnicas o de negocio que se tomaron, con su razon.
3. PENDIENTES: cosas concretas y accionables que quedaron por hacer.

Conversacion:
{conversation}

Responde UNICAMENTE con JSON valido, sin texto antes ni despues, sin markdown:
{{"facts": [{{"text": "...", "type": "user|technical|social|preference|project|financial", "is_correction": false, "importance": 3}}],
"decisions": [{{"title": "...", "chosen": "...", "reasoning": "..."}}],
"action_items": [{{"description": "...", "due_date": null}}]}}

Si una categoria no tiene nada memorable, deja su lista vacia. Es perfectamente valido
devolver listas vacias si la conversacion no tiene nada digno de recordar a largo plazo.
JAMAS inventes informacion que no este explicita en la conversacion."""


# Verificacion extra antes de aplicar una correccion (ver db.py:save_fact,
# parametro verify_correction_fn). La distancia vectorial sola puede acertar
# el candidato equivocado por coincidencia de embedding — esto es un segundo
# chequeo, mas caro (una llamada LLM mas), pero solo se dispara cuando ya
# hay un candidato de por medio, no en cada fact.
VERIFICATION_PROMPT = """Un sistema de memoria detecto que estos dos hechos podrian estar
relacionados (el nuevo podria ser una correccion del viejo, por similitud de embedding).

HECHO VIEJO (guardado antes): {old}
HECHO NUEVO (recien extraido, marcado como posible correccion): {new}

Es una correccion real SOLO si el hecho nuevo contradice o actualiza directamente al viejo
(son sobre la misma cosa, y el nuevo cambia el dato). NO es una correccion si:
- Son sobre temas distintos que coincidieron por casualidad en el embedding.
- El nuevo agrega detalle sin contradecir al viejo (eso son dos hechos validos, no uno
  reemplazando al otro).

Responde UNICAMENTE con JSON, sin texto antes ni despues, sin markdown:
{{"es_correccion": true}} o {{"es_correccion": false}}"""


# Extractor: por defecto DeepSeek (faceta confiable). El system_prompt es
# minimo porque el prompt de extraccion ya trae toda la instruccion.
def build_extractor() -> HttpMuscle:
    """Crea el muscle extractor. Hoy DeepSeek; manana, un local."""
    return HttpMuscle(
        name="extractor",
        provider="deepseek",
        model_default="deepseek-v4-flash",
        models_allowed=["deepseek-v4-flash", "deepseek-v4-pro"],
        system_prompt="Sos un extractor de informacion. Respondes solo con JSON valido.",
        timeout=120.0,
    )


# Tamano maximo de conversacion (en chars) que se manda de una sola vez al
# extractor. ~3000 tokens, conservador para dejar espacio al resto del
# prompt fijo dentro del contexto de DeepSeek. Conversaciones mas largas se
# parten en varios pedidos (ver _chunk_conversation).
MAX_CHARS_PER_EXTRACTION = 12000


def _chunk_conversation(conv_text: str, max_chars: int = MAX_CHARS_PER_EXTRACTION) -> list[str]:
    """Parte conv_text en trozos de hasta max_chars, cortando en limites de
    linea (nunca a mitad de un mensaje). Conversaciones cortas: un solo
    trozo (comportamiento identico al de antes de esta funcion)."""
    if len(conv_text) <= max_chars:
        return [conv_text]
    lineas = conv_text.split("\n")
    chunks: list[str] = []
    actual: list[str] = []
    largo = 0
    for linea in lineas:
        if largo + len(linea) + 1 > max_chars and actual:
            chunks.append("\n".join(actual))
            actual, largo = [], 0
        actual.append(linea)
        largo += len(linea) + 1
    if actual:
        chunks.append("\n".join(actual))
    return chunks


def _parse_json(raw: str) -> dict | None:
    """Parsea el JSON del extractor, tolerando ```json ... ``` y ruido."""
    if not raw:
        return None
    text = raw.strip()
    # Quitar fences de markdown si los hubiera
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text[3:] else text
        text = text.replace("json", "", 1).strip("` \n")
    # Intentar aislar el objeto JSON
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _build_verify_correction_fn(extractor: HttpMuscle):
    """Devuelve la funcion que db.save_fact() usa como verify_correction_fn:
    una llamada LLM extra que confirma si un candidato de correccion es real,
    antes de que la distancia vectorial sola decida reemplazar un fact.
    Fail-safe: cualquier error (timeout, JSON invalido) se propaga como
    excepcion y save_fact ya lo trata como 'no confirmado'."""
    async def verify(old_text: str, new_text: str) -> bool:
        raw = await extractor.invoke(
            VERIFICATION_PROMPT.format(old=old_text, new=new_text), decorate=False
        )
        data = _parse_json(raw)
        if data is None:
            raise ValueError("verificacion de correccion: JSON no parseable")
        return bool(data.get("es_correccion", False))
    return verify


async def process_one(db: MemoryDB, extractor: HttpMuscle, conv: dict) -> bool:
    """Procesa UNA conversacion. Devuelve True si la marco procesada."""
    conv_id = conv["id"]
    messages = await db.get_conversation_messages(conv_id)
    if not messages:
        # Sin mensajes: marcar procesada igual (no hay nada que extraer)
        await db.mark_processed(conv_id)
        return True

    # Armar el texto de la conversacion. Si es muy larga, se parte en varios
    # pedidos al extractor (item #5): una conversacion larga puede exceder
    # el contexto util del extractor o inflar costo/latencia sin necesidad.
    conv_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    chunks = _chunk_conversation(conv_text)
    if len(chunks) > 1:
        logger.info(f"conv {conv_id}: conversacion larga ({len(conv_text)} chars), "
                    f"partida en {len(chunks)} pedidos al extractor")

    data: dict = {"facts": [], "decisions": [], "action_items": []}
    for i, chunk in enumerate(chunks):
        # Pedir extraccion al LLM
        try:
            # decorate=False: extraccion interna a JSON, sin etiqueta de autoridad
            # (un sello al final rompería el parseo del JSON).
            raw = await extractor.invoke(
                EXTRACTION_PROMPT.format(conversation=chunk), decorate=False
            )
        except Exception as e:
            logger.error(f"conv {conv_id}: extractor fallo (chunk {i+1}/{len(chunks)}): {e}")
            return False  # NO marcar procesada: se reintenta ENTERA en la proxima corrida

        chunk_data = _parse_json(raw)
        if chunk_data is None:
            logger.error(f"conv {conv_id}: JSON no parseable (chunk {i+1}/{len(chunks)}), "
                         f"se reintentara")
            return False

        data["facts"].extend(chunk_data.get("facts", []))
        data["decisions"].extend(chunk_data.get("decisions", []))
        data["action_items"].extend(chunk_data.get("action_items", []))

    # Scope de origen: los facts heredan el scope de su conversacion.
    # project_id NOT NULL -> memoria de proyecto; NULL -> individual de user_id.
    conv_user = conv.get("user_id")
    conv_proj = conv.get("project_id")

    # Guardar lo extraido (confidence 0.7, is_verified FALSE en facts)
    verify_correction_fn = _build_verify_correction_fn(extractor)
    n_facts = n_dec = n_act = 0
    for f in data.get("facts", []):
        if f.get("text"):
            saved = await db.save_fact(f["text"], f.get("type", "user"),
                                       source_facet="extractor",
                                       user_id=conv_user, project_id=conv_proj,
                                       is_correction=bool(f.get("is_correction", False)),
                                       verify_correction_fn=verify_correction_fn,
                                       importance=f.get("importance"))
            if saved:
                n_facts += 1
    for d in data.get("decisions", []):
        if d.get("title") and d.get("chosen"):
            await db.save_decision(d["title"], d["chosen"],
                                   d.get("reasoning", ""),
                                   user_id=conv_user, project_id=conv_proj)
            n_dec += 1
    for a in data.get("action_items", []):
        if a.get("description"):
            await db.save_action_item(a["description"], a.get("due_date"),
                                      source_conversation_id=conv_id,
                                      user_id=conv_user, project_id=conv_proj)
            n_act += 1

    await db.mark_processed(conv_id)
    logger.info(f"conv {conv_id}: {n_facts} hechos, {n_dec} decisiones, {n_act} pendientes "
                f"(scope user={conv_user} project={conv_proj})")
    return True


async def run_once(limit: int = 10) -> None:
    """Una corrida del worker: procesa hasta `limit` conversaciones."""
    jax_db_host = os.environ.get("JAX_DB_HOST")
    if not jax_db_host:
        raise RuntimeError(
            "JAX_DB_HOST no está seteado -- sin default silencioso a "
            "localhost (esa instancia está muerta, ver memoria "
            "jax-dual-mariadb-instances). Sourceá /etc/jax/.env."
        )
    db = MemoryDB()
    ok = await db.connect(
        host=jax_db_host,
        user=os.getenv("JAX_DB_USER", ""),
        password=os.getenv("JAX_DB_PASSWORD", ""),
        database=os.getenv("JAX_DB_NAME", "jax_memory"),
    )
    if not ok:
        logger.error("No se pudo conectar a la memoria. Abortando corrida.")
        return

    try:
        pendientes = await db.get_unprocessed_conversations(limit=limit)
        if not pendientes:
            logger.info("No hay conversaciones pendientes de procesar.")
            return

        logger.info(f"Procesando {len(pendientes)} conversacion(es)...")
        extractor = build_extractor()
        for conv in pendientes:
            await process_one(db, extractor, conv)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(run_once())
