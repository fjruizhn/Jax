"""Configuración del pre-vuelo de Jacobs (spec 2026-09-17 §4.5 y §4.6).

Mismo patrón que jax/core/db_connect_config.py: se lee en CADA llamada (un
cambio de entorno vale en el próximo pre-vuelo, sin reiniciar) y un valor
inválido NO cae a un default silencioso -- lanza RuntimeError con el nombre de
la variable. Un typo en el entorno que desactivara la sonda o cambiara el
divisor de costo sería un fail-open con apariencia de configuración.

Defaults declarados, no medidos:
- JAX_PREVUELO_SONDA_TIMEOUT_S=20: la sonda pide 16 tokens; 20 s cubre la
  latencia de arranque de un proveedor sano sin dejar al usuario esperando el
  timeout de un paso entero (300 s). Se revisa con los números de la Task 15.
- JAX_PREVUELO_CHARS_POR_TOKEN=2: cota INFERIOR de caracteres por token.
  Menos caracteres por token = más tokens = más costo: sobreestima, nunca
  subestima (spec §4.6).
- JAX_PREVUELO_SONDA_MAX_TOKENS=16: la sonda mide disponibilidad, no calidad;
  se manda el menor entre esto y model.max_output_tokens.
- JAX_PREVUELO_DB_POOL_MAX=5 (Task 15b, 2026-09-17): conexiones del pool de
  lectura del pre-vuelo (jacobs/store.py::conexion_de_lectura). Medido en
  proceso contra jax_memory_test (task-15b-report.md), p95 a c=25 / c=50:
  maxsize 5 = 30,3 / 54,8 ms; 10 = 31,1 / 60,1 ms; 25 = 38,4 / 56,2 ms. Un
  pool más grande no mejora: el límite es la CPU del único event loop, no la
  base. 5 es el menor medido y deja margen en una MariaDB compartida con
  max_connections=151. Se lee al CREAR el pool: un cambio vale después de
  reiniciar el proceso (o de store.cerrar_pool()).

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import os

SONDA_TIMEOUT_S = "JAX_PREVUELO_SONDA_TIMEOUT_S"
CHARS_POR_TOKEN = "JAX_PREVUELO_CHARS_POR_TOKEN"
SONDA_MAX_TOKENS = "JAX_PREVUELO_SONDA_MAX_TOKENS"
DB_POOL_MAX = "JAX_PREVUELO_DB_POOL_MAX"


def _entero_positivo(nombre: str, crudo: str) -> int:
    try:
        valor = int(crudo)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{nombre}={crudo!r} no es un entero: corregilo en el entorno "
            f"(/etc/jax/.env) -- no se asume ningún valor"
        ) from exc
    if valor <= 0:
        raise RuntimeError(f"{nombre}={valor} tiene que ser mayor que cero")
    return valor


def sonda_timeout_s() -> int:
    """Segundos que la sonda espera al proveedor."""
    return _entero_positivo(SONDA_TIMEOUT_S, os.getenv(SONDA_TIMEOUT_S, "20"))


def chars_por_token() -> int:
    """Divisor de caracteres a tokens para el costo máximo de entrada."""
    return _entero_positivo(CHARS_POR_TOKEN, os.getenv(CHARS_POR_TOKEN, "2"))


def sonda_max_tokens() -> int:
    """Techo de tokens de salida que pide la sonda."""
    return _entero_positivo(SONDA_MAX_TOKENS, os.getenv(SONDA_MAX_TOKENS, "16"))


def db_pool_max() -> int:
    """Conexiones máximas del pool de lectura del pre-vuelo."""
    return _entero_positivo(DB_POOL_MAX, os.getenv(DB_POOL_MAX, "5"))
