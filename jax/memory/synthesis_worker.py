"""
JAX 2.0 — Worker de sintesis de segundo orden (memoria, item #8 del roadmap).

Proceso BATCH separado del worker de extraccion (worker.py). Mientras
worker.py destila hechos SUELTOS de conversaciones, este busca patrones que
conectan hechos YA VERIFICADOS por Fernando dentro de un mismo scope, y
propone un insight de mas alto nivel — si es que hay uno real.

Mitigaciones obligatorias (este es el item de mayor riesgo del roadmap —
literalmente donde ocurrio el incidente de privacidad de Beelink que
origino esta comparacion):
  - Solo lee facts con is_verified=TRUE: nunca sintetiza sobre ruido no
    confirmado por Fernando.
  - Reusa el MISMO bloque de categorias prohibidas que worker.py
    (FORBIDDEN_CATEGORIES_BLOCK, importado — no reescrito).
  - Los insights se guardan via db.save_fact(), que por diseño siempre
    entra con is_verified=FALSE: la sintesis NUNCA se auto-verifica, pasa
    por el mismo gate de revision humana que cualquier otro fact.
  - Timer systemd separado (jax-memory-synthesis.timer), instalado pero
    NO habilitado por defecto — activarlo es una decision explicita.

Uso:
    set -a; source /etc/jax/.env; set +a
    PYTHONPATH=. .venv/bin/python -m jax.memory.synthesis_worker

En memoria de Jairo Urbina.
"""

from __future__ import annotations

import asyncio
import os
import logging

from jax.memory.db import MemoryDB
from jax.memory.worker import (
    FORBIDDEN_CATEGORIES_BLOCK,
    _parse_json,
)
from jax.muscles.base import HttpMuscle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [synthesis] %(levelname)s: %(message)s",
)
logger = logging.getLogger("jax.memory.synthesis_worker")


# Scope minimo para que valga la pena buscar patrones. Menos que esto y no
# hay suficiente material para un insight real (ver roadmap: "baja
# prioridad... a la escala actual de JAX").
MIN_VERIFIED_FACTS = 5


SYNTHESIS_PROMPT = """Estos son hechos sobre Fernando que YA fueron revisados y confirmados
por el mismo Fernando (no son ruido de extraccion automatica, son verdad confirmada).

Tu trabajo NO es repetir ninguno de ellos. Es buscar si, mirandolos TODOS juntos, hay un
patron o conexion de mas alto nivel que ninguno de ellos dice por separado — algo que
solo se ve al conectar dos o mas.

Se MUY estricto: si no hay ningun patron genuino, no inventes uno. La mayoria de las
veces la respuesta correcta es una lista vacia. Un insight forzado o obvio (que ya sea
practicamente lo mismo que uno de los hechos de entrada) es peor que no decir nada.

""" + FORBIDDEN_CATEGORIES_BLOCK + """

EJEMPLOS:
- MAL (no es un insight, es un hecho repetido): si un hecho dice "Fernando usa MariaDB"
  y otro dice "Fernando prefiere infraestructura propia", NO generes "Fernando usa
  MariaDB porque prefiere infraestructura propia" — eso es solo juntar dos hechos, no
  un patron nuevo.
- BIEN (es un insight real): si tres hechos separados mencionan que distintos proyectos
  de Fernando (JAX, un producto financiero, y otro sistema) todos evitan depender de
  servicios de terceros, el patron real es "Fernando tiene una preferencia consistente,
  ya demostrada en multiples proyectos, por infraestructura autohospedada" — eso conecta
  algo que ningun hecho individual dice.

Cada hecho tiene un id entre corchetes al principio — usalo para citar EXACTAMENTE de que
hechos sale cada insight, en "source_ids".

Hechos verificados de este scope:
{facts}

Responde UNICAMENTE con JSON valido, sin texto antes ni despues, sin markdown:
{{"insights": [{{"text": "...", "type": "user|technical|social|preference|project|financial",
"source_ids": [1, 2]}}]}}

source_ids es OBLIGATORIO para cada insight: los ids (numeros, sin corchetes) de los hechos
de entrada que conectaste para llegar a ese insight — minimo 2 (un insight de un solo hecho
no es un patron, es una repeticion). Si no hay ningun patron genuino que conecte estos
hechos, devolve la lista vacia. Es perfectamente valido devolver una lista vacia — de
hecho, es lo mas comun."""


def build_synthesizer() -> HttpMuscle:
    """Crea el muscle sintetizador. Mismo extractor confiable (DeepSeek) que
    worker.py — sintetizar mal es tan costoso como extraer mal."""
    return HttpMuscle(
        name="synthesizer",
        provider="deepseek",
        model_default="deepseek-v4-flash",
        models_allowed=["deepseek-v4-flash", "deepseek-v4-pro"],
        system_prompt="Sos un analista que busca patrones. Respondes solo con JSON valido.",
        timeout=120.0,
    )


async def process_scope(db: MemoryDB, synthesizer: HttpMuscle,
                        user_id: int | None, project_id: int | None) -> int:
    """Procesa UN scope (user_id, project_id). Devuelve cuantos insights guardo."""
    facts = await db.get_facts(only_unverified=False, only_verified=True,
                               limit=50, user_id=user_id, project_id=project_id)
    if not facts or len(facts) < MIN_VERIFIED_FACTS:
        return 0

    valid_ids = {f["id"] for f in facts}
    facts_text = "\n".join(f"[{f['id']}] {f['fact_text']}" for f in facts)

    try:
        # decorate=False: igual que worker.py, esto es extraccion interna a
        # JSON, no una respuesta que Fernando vea directamente.
        raw = await synthesizer.invoke(
            SYNTHESIS_PROMPT.format(facts=facts_text), decorate=False
        )
    except Exception as e:
        logger.error(f"scope user={user_id} project={project_id}: synthesizer fallo: {e}")
        return 0

    data = _parse_json(raw)
    if data is None:
        logger.error(f"scope user={user_id} project={project_id}: JSON no parseable")
        return 0

    n_saved = 0
    for insight in data.get("insights", []):
        if not insight.get("text"):
            continue
        # Solo se guardan los ids que realmente vinieron en la entrada — un
        # id inventado por el LLM (alucinado) se descarta, no rompe el save.
        source_ids = [i for i in insight.get("source_ids", []) if i in valid_ids]
        saved = await db.save_fact(
            insight["text"], insight.get("type", "user"),
            source_facet="synthesis",
            user_id=user_id, project_id=project_id,
            source_fact_ids=source_ids or None,
        )
        if saved:
            n_saved += 1

    if n_saved:
        logger.info(f"scope user={user_id} project={project_id}: {n_saved} insight(s) "
                    f"nuevo(s), de {len(facts)} facts verificados")
    return n_saved


async def run_once() -> None:
    """Una corrida del sintetizador: recorre todos los scopes con suficientes
    facts verificados y busca insights en cada uno."""
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
        scopes = await db.get_scopes_with_verified_facts(min_facts=MIN_VERIFIED_FACTS)
        if not scopes:
            logger.info("Ningun scope con suficientes facts verificados todavia.")
            return

        logger.info(f"Analizando {len(scopes)} scope(s) con >= {MIN_VERIFIED_FACTS} "
                    f"facts verificados...")
        synthesizer = build_synthesizer()
        total = 0
        for scope in scopes:
            total += await process_scope(db, synthesizer, scope["user_id"], scope["project_id"])
        logger.info(f"Corrida terminada: {total} insight(s) nuevo(s) en total.")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(run_once())
