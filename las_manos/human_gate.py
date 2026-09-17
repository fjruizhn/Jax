"""
LAS MANOS — Human gate: tokens de aprobación de un solo uso.

Hasta el 2026-09-17 los tokens vivían en un dict del proceso y los emitía
`POST /human_gate/token`, SIN autenticación: cualquier proceso local (Hyde,
cualquier proceso de `fruiz`, `axioma`) pedía un token y se autoaprobaba la
operación. Y `/motor/dispatch` aceptaba como token cualquier string no vacío.

Contrato (el mismo de jacobs/subpipelines.py, frente F):

- Emisión SOLO del lado del servidor y SIN ruta HTTP: `emitir_token_gate()` la
  llama la herramienta `las_manos/emitir_token_gate.py`, que Fernando corre con
  las credenciales de la base (/etc/jax/.env). Sin esa credencial no se emite.
- En la base queda SOLO el sha256 del token.
- Un solo uso, consumo atómico: `UPDATE … WHERE usado_at IS NULL AND vence_at > now`.
- El diagnóstico posterior SOLO rotula el rechazo; nunca lo decide.

Sin caché: el consumo es por PK y cada token sirve una vez.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from enum import Enum

from jacobs import store

TOKEN_BYTES = 32
LARGO_TOKEN_REF = 12

# Límites del contrato, no preferencias (mismo rango que los tokens de
# sub-pipelines): un TTL de horas deja vivo demasiado tiempo un token filtrado.
TTL_MINIMO, TTL_MAXIMO = 10, 3600


class Motivo(str, Enum):
    FALTA_TOKEN       = "falta_token"
    TOKEN_DESCONOCIDO = "token_desconocido"
    TOKEN_USADO       = "token_usado"
    TOKEN_VENCIDO     = "token_vencido"
    # El UPDATE dijo que no pero el diagnóstico ya no ve la causa (el estado
    # cambió entre las dos lecturas). Se rechaza igual.
    ESTADO_CAMBIO     = "estado_cambio"


@dataclass(frozen=True)
class TokenEmitido:
    token: str
    vence_at: float


@dataclass(frozen=True)
class ResultadoGate:
    aceptado: bool
    motivo: Motivo | None = None


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_ref(token_hash: str) -> str:
    """Lo único del token que puede ir a un log."""
    return token_hash[:LARGO_TOKEN_REF]


def ttl_segundos(gate_cfg: dict) -> int:
    """`[human_gate].token_ttl_seconds` de config.toml, validado. Fuera de rango
    o mal tipado -> ValueError (fail-closed: server.py lo lee al importar)."""
    valor = gate_cfg.get("token_ttl_seconds")
    if not isinstance(valor, int) or isinstance(valor, bool):
        raise ValueError(f"[human_gate].token_ttl_seconds={valor!r} no es un entero")
    if not TTL_MINIMO <= valor <= TTL_MAXIMO:
        raise ValueError(
            f"[human_gate].token_ttl_seconds={valor} fuera de rango [{TTL_MINIMO}, {TTL_MAXIMO}]"
        )
    return valor


async def emitir_token_gate(emitido_por: str, ttl_segundos: int) -> TokenEmitido:
    """Emite un token de un solo uso. Devuelve el token en claro, y es la única
    vez que existe: en la base queda su sha256. SIN ruta HTTP."""
    token = secrets.token_urlsafe(TOKEN_BYTES)
    emitido_at = time.time()
    vence_at = emitido_at + ttl_segundos
    await store.human_gate_token_emitir(hash_token(token), emitido_por, emitido_at, vence_at)
    return TokenEmitido(token=token, vence_at=vence_at)


def _motivo(diagnostico: dict | None, ahora: float) -> Motivo:
    if diagnostico is None:
        return Motivo.TOKEN_DESCONOCIDO
    if diagnostico["usado_at"] is not None:
        return Motivo.TOKEN_USADO
    if diagnostico["vence_at"] <= ahora:
        return Motivo.TOKEN_VENCIDO
    return Motivo.ESTADO_CAMBIO


async def consumir_token_gate(token: str | None, uso: str) -> ResultadoGate:
    """Consume el token para `uso` (p. ej. `execute:<request_id>`). Un error de
    la base se PROPAGA: sin veredicto no hay aprobación (fail-closed)."""
    if not token or not token.strip():
        return ResultadoGate(False, Motivo.FALTA_TOKEN)
    token_hash = hash_token(token)
    if await store.human_gate_token_consumir(token_hash, uso, time.time()):
        return ResultadoGate(True)
    diagnostico = await store.human_gate_token_diagnostico(token_hash)
    return ResultadoGate(False, _motivo(diagnostico, time.time()))
