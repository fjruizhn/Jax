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

QUÉ SERIALIZA HOY, medido contra el código (m6 de la re-revisión final,
2026-09-17) -- el texto de T2 describía una sección crítica más chica que la
real:
  - crear (routes.py::create_pipeline): recuento de activos, build() (20-40 s
    si planifica con un LLM), el PRE-VUELO COMPLETO -- incluidas las sondas,
    hasta sonda_timeout_s + 1 = 21 s por ronda -- y la transacción bajo el
    candado de MariaDB;
  - continuar (continuar.py::continuar, endpoint y CLI): el análisis, el
    pre-vuelo con sus sondas y su transacción, con el MISMO objeto.
O sea: dentro de un proceso, un crear y un continuar no corren a la vez, y
ninguno de los dos empieza mientras el otro sondea. Lo de adentro está
acotado (build, sonda, candado y transacción tienen plazo), y con
MAX_PARALLEL_PIPELINES=3 el tráfico real es bajo, así que se deja como está.
Sacar el pre-vuelo de la sección crítica sería un cambio de la regla 4 del
spec §5.2 ("dentro del candado"), no una limpieza: si alguna vez molesta, es
una decisión escrita, no un ajuste al pasar.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio

candado_de_creacion = asyncio.Lock()
