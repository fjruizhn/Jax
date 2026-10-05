"""Tripwire del recall del indice vectorial HNSW de `messages`.

Estaba en worker.py y nadie lo llamaba (codigo muerto, auditoria 2026-10-05). Ahora lo ejecuta
`jax-memory-vector-health` (embedding_worker.run_b9_vector_health): si hay que volver a medir el
recall, la unidad sale con error y el aviso llega por OnFailure.

Filas de `messages` con las que se midio el recall (jax#128, 2026-09-11): ef_search=400 ->
93,3 % de recall@5. Ese numero es una propiedad de ESE tamano, no del sistema: el recall de un
grafo HNSW se degrada al crecer la tabla, y lo hace EN SILENCIO. DEUDA.md decia "volver a medirlo
cuando crezca un orden de magnitud", una condicion que no vigila nadie sin fecha ni dueno.

Para silenciarlo: volver a medir (ver el mensaje) y SUBIR FILAS_AL_MEDIR_RECALL a las filas con
las que se midio, en el mismo cambio que registra la medicion.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

from jax.memory.embedding_config import CONFIG as EMBED

logger = logging.getLogger("jax.memory.recall_tripwire")

FILAS_AL_MEDIR_RECALL = 1607
FACTOR_DE_REMEDICION = 10


async def recall_requiere_remedicion(contar_filas: Callable[[], Awaitable[Optional[int]]]) -> bool:
    """True si `messages` crecio un orden de magnitud desde la ultima medicion del recall.

    Un fallo al contar NO se traga (a diferencia de la version del worker): esto es el chequeo de
    salud, y un chequeo que no puede mirar no puede dar verde.
    """
    filas = await contar_filas()
    if filas is None:
        raise RuntimeError("tripwire de recall: no se pudo contar messages")
    if filas < FILAS_AL_MEDIR_RECALL * FACTOR_DE_REMEDICION:
        return False
    logger.error(
        f"messages tiene {filas} filas y el recall del indice HNSW se midio con "
        f"{FILAS_AL_MEDIR_RECALL} (93,3 % con ef_search=400). Un grafo HNSW pierde "
        f"recall al crecer, y lo hace sin error: hay que VOLVER A MEDIRLO contra la "
        f"busqueda exacta (IGNORE INDEX idx_{EMBED.column}) y, si bajo, subir "
        f"JAX_MEMORY_HNSW_EF_SEARCH. Medir por DISTANCIA, no por ids: messages "
        f"tiene duplicados exactos y los empates hacen fallar una comparacion "
        f"de conjuntos sin que el indice pierda nada. Ver DEUDA.md."
    )
    return True
