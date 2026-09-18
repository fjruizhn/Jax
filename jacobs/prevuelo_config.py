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
- El tamaño del pool de conexiones YA NO es de este módulo (Ruling R38,
  2026-09-17): el pool pasó a ser de todo el store de Jacobs y su variable es
  JAX_DB_POOL_MAX, leída en jacobs/store.py::db_pool_max (ahí está la
  derivación del default). JAX_PREVUELO_DB_POOL_MAX no la lee nadie (ni
  /etc/jax/.env la define: verificado al renombrar).
  (JAX_PREVUELO_CANDADO_TIMEOUT_S se retiró el 2026-09-17 con el candado con
  nombre de MariaDB: el cupo ya no se espera, se decide dentro de la misma
  sentencia que escribe. Una variable que no lee nadie es una trampa: el día
  que alguien la ponga en /etc/jax/.env va a creer que cambió algo.)

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import os

SONDA_TIMEOUT_S = "JAX_PREVUELO_SONDA_TIMEOUT_S"
CHARS_POR_TOKEN = "JAX_PREVUELO_CHARS_POR_TOKEN"
SONDA_MAX_TOKENS = "JAX_PREVUELO_SONDA_MAX_TOKENS"


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


