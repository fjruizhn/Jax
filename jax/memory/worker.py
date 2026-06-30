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

EJEMPLOS:
- MAL (NO extraer): "Fernando pregunto por el significado de una palabra rara."
- MAL (NO extraer): "Fernando esta haciendo una prueba de memoria."
- BIEN (extraer): "Fernando es banquero de tercera generacion y disena productos financieros para LATAM."
- BIEN (extraer): "Fernando prefiere infraestructura propia y desconfia de depender de servicios de terceros."

Categorias:
1. HECHOS: conocimiento duradero sobre Fernando, sus preferencias, proyectos o relaciones.
2. DECISIONES: elecciones tecnicas o de negocio que se tomaron, con su razon.
3. PENDIENTES: cosas concretas y accionables que quedaron por hacer.

Conversacion:
{conversation}

Responde UNICAMENTE con JSON valido, sin texto antes ni despues, sin markdown:
{{"facts": [{{"text": "...", "type": "user|technical|social|preference|project|financial"}}],
"decisions": [{{"title": "...", "chosen": "...", "reasoning": "..."}}],
"action_items": [{{"description": "...", "due_date": null}}]}}

Si una categoria no tiene nada memorable, deja su lista vacia. Es perfectamente valido
devolver listas vacias si la conversacion no tiene nada digno de recordar a largo plazo.
JAMAS inventes informacion que no este explicita en la conversacion."""


# Extractor: por defecto DeepSeek (faceta confiable). El system_prompt es
# minimo porque el prompt de extraccion ya trae toda la instruccion.
def build_extractor() -> HttpMuscle:
    """Crea el muscle extractor. Hoy DeepSeek; manana, un local."""
    return HttpMuscle(
        name="extractor",
        provider="deepseek",
        model_default="deepseek-chat",
        models_allowed=["deepseek-chat", "deepseek-reasoner"],
        system_prompt="Sos un extractor de informacion. Respondes solo con JSON valido.",
        timeout=120.0,
    )


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


async def process_one(db: MemoryDB, extractor: HttpMuscle, conv: dict) -> bool:
    """Procesa UNA conversacion. Devuelve True si la marco procesada."""
    conv_id = conv["id"]
    messages = await db.get_conversation_messages(conv_id)
    if not messages:
        # Sin mensajes: marcar procesada igual (no hay nada que extraer)
        await db.mark_processed(conv_id)
        return True

    # Armar el texto de la conversacion
    conv_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    # Pedir extraccion al LLM
    try:
        # decorate=False: extraccion interna a JSON, sin etiqueta de autoridad
        # (un sello al final rompería el parseo del JSON).
        raw = await extractor.invoke(
            EXTRACTION_PROMPT.format(conversation=conv_text), decorate=False
        )
    except Exception as e:
        logger.error(f"conv {conv_id}: extractor fallo: {e}")
        return False  # NO marcar procesada: se reintenta en la proxima corrida

    data = _parse_json(raw)
    if data is None:
        logger.error(f"conv {conv_id}: JSON no parseable, se reintentara")
        return False

    # Scope de origen: los facts heredan el scope de su conversacion.
    # project_id NOT NULL -> memoria de proyecto; NULL -> individual de user_id.
    conv_user = conv.get("user_id")
    conv_proj = conv.get("project_id")

    # Guardar lo extraido (confidence 0.7, is_verified FALSE en facts)
    n_facts = n_dec = n_act = 0
    for f in data.get("facts", []):
        if f.get("text"):
            await db.save_fact(f["text"], f.get("type", "user"),
                               source_facet="extractor",
                               user_id=conv_user, project_id=conv_proj)
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
    db = MemoryDB()
    ok = await db.connect(
        host=os.getenv("JAX_DB_HOST", "localhost"),
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
