"""Candado de proceso para crear y continuar pipelines (spec 2026-09-17 §5.2
regla 4). Vivía en routes.py como _pipeline_create_lock; se muda acá para que
continuar.py (endpoint y CLI) tome el MISMO objeto sin importar routes.

Su justificación sigue siendo la de T2 (2026-08-19, ver routes.py): Jacobs
corre en un solo proceso uvicorn y un asyncio.Lock no reserva conexión DB
mientras build() llama a un LLM. No cruza procesos: entre LAS MANOS y el CLI,
el cupo de MAX_PARALLEL_PIPELINES lo serializa el candado con nombre de
MariaDB (store.candado_de_activos, ola final F3), tomado por dentro de este
alrededor de recontar + escribir; un continue doble del mismo pipeline lo
serializa además la transacción con SELECT ... FOR UPDATE y la época.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio

candado_de_creacion = asyncio.Lock()
