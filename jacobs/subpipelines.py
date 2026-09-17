"""
Jacobs — Contrato de sub-pipelines (Ada multiagente).

Frente F del spec 2026-09-16 (docs/superpowers/specs/2026-09-16-hallazgos-auditoria-jax-design.md).
Hasta este frente, `validate_create` aceptaba para `invoked_by="ada"` CUALQUIER
string no vacío como `subpipeline_token`, y la profundidad nunca llegaba a la
política (routes.py no la pasaba): el candado existía en el nombre y no en el
efecto. Acá vive el contrato real:

- Emisión (solo servidor, SIN ruta HTTP: las rutas de Jacobs no autentican):
  `emitir_token_subpipeline()`.
- Consumo atómico: `consumir_token_subpipeline()` -- un UPDATE condicionado,
  una fila afectada o rechazo.
- La profundidad sale de la fila del token, nunca del cuerpo del pedido.

"Padre activo" = `jacobs_pipelines.status='running'` del padre, SOLAMENTE
(enmienda de Fernando 2026-09-16). El token queda atado a `parent_step` (el
step de Ada que produjo la delegación) en CUALQUIER estado: en el modo "plan
de delegación" Jacobs emite los tokens DESPUÉS de que el step `delegate`
terminó. Se conserva la verificación de que `parent_step` existe y pertenece
a `parent_pipeline_id` (emisión: además de la faceta `ada`); el estado del
paso no cuenta para nada.

Sin caché: la config son dos os.environ.get por llamada, nada caro que valga la
pena guardar ni invalidar. Se valida también al arrancar LAS MANOS
(server.py::_jacobs_init): un valor inválido tumba el arranque.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn

from jacobs import policy, store

ENV_TTL = "JAX_SUBPIPELINE_TOKEN_TTL_SECONDS"
ENV_MAX_PROFUNDIDAD = "JAX_MAX_SUBPIPELINE_DEPTH"

# 300 s cubre el peor caso medido de espera por el candado de creación
# (routes._pipeline_create_lock serializa build() de 20-40 s, hasta 3 en cola).
TTL_POR_DEFECTO = 300
MAX_PROFUNDIDAD_POR_DEFECTO = 3

# Límites del contrato, no preferencias: un TTL de horas deja vivo demasiado
# tiempo un token filtrado, y más de 5 niveles es un árbol de agentes que nadie
# audita a mano. Moverlos es una decisión de Fernando, no un ajuste de .env.
TTL_MINIMO, TTL_MAXIMO = 10, 3600
PROFUNDIDAD_MINIMA, PROFUNDIDAD_MAXIMA = 1, 5

TOKEN_BYTES = 32
LARGO_TOKEN_REF = 12


@dataclass(frozen=True)
class ConfigSubpipelines:
    ttl_segundos: int
    max_profundidad: int


def _entero_acotado(nombre: str, por_defecto: int, minimo: int, maximo: int) -> int:
    crudo = os.environ.get(nombre)
    if crudo is None:
        return por_defecto
    try:
        valor = int(crudo.strip())
    except ValueError as exc:
        raise ValueError(f"{nombre}={crudo!r} no es un entero") from exc
    if not minimo <= valor <= maximo:
        raise ValueError(f"{nombre}={valor} fuera de rango [{minimo}, {maximo}]")
    return valor


def config_subpipelines() -> ConfigSubpipelines:
    """Lee y valida la config. Variable ausente -> default acotado; presente
    pero vacía, no entera o fuera de rango -> ValueError (fail-closed)."""
    return ConfigSubpipelines(
        ttl_segundos=_entero_acotado(ENV_TTL, TTL_POR_DEFECTO, TTL_MINIMO, TTL_MAXIMO),
        max_profundidad=_entero_acotado(
            ENV_MAX_PROFUNDIDAD, MAX_PROFUNDIDAD_POR_DEFECTO,
            PROFUNDIDAD_MINIMA, PROFUNDIDAD_MAXIMA,
        ),
    )


class Motivo(str, Enum):
    TOKEN_DESCONOCIDO    = "token_desconocido"
    TOKEN_USADO          = "token_usado"
    TOKEN_VENCIDO        = "token_vencido"
    PADRE_NO_COINCIDE    = "padre_no_coincide"
    PADRE_DESCONOCIDO    = "padre_desconocido"
    PADRE_INACTIVO       = "padre_inactivo"
    PASO_DESCONOCIDO     = "paso_desconocido"
    PASO_NO_ES_ADA       = "paso_no_es_ada"
    PROFUNDIDAD_EXCEDIDA = "profundidad_excedida"
    KILL_SWITCH_ACTIVO   = "kill_switch_activo"
    # La sentencia atómica dijo que no, pero el diagnóstico posterior ya no ve
    # la causa (el estado cambió entre las dos lecturas). Se rechaza igual.
    ESTADO_CAMBIO        = "estado_cambio"
    PLAN_RECHAZADO       = "plan_rechazado"


@dataclass(frozen=True)
class TokenConsumido:
    parent_pipeline_id: str
    parent_step: str
    depth: int


@dataclass(frozen=True)
class ConsumoRechazado:
    motivo: Motivo


class EmisionRechazada(Exception):
    def __init__(self, motivo: Motivo) -> None:
        super().__init__(motivo.value)
        self.motivo = motivo


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_ref(token_hash: str) -> str:
    """Lo único del token que puede ir a un evento o a un log."""
    return token_hash[:LARGO_TOKEN_REF]


def motivo_emision(diagnostico: dict, parent_pipeline_id: str, max_profundidad: int) -> Motivo:
    """Por qué el INSERT … SELECT de la emisión no insertó. `diagnostico` sale de
    store.subpipeline_emision_diagnostico(). El estado del paso de Ada que
    delegó no cuenta (enmienda 2026-09-16): `paso_status` se usa SOLO como
    prueba de existencia del paso."""
    if diagnostico["padre_status"] is None:
        return Motivo.PADRE_DESCONOCIDO
    if diagnostico["paso_status"] is None or diagnostico["paso_pipeline_id"] != parent_pipeline_id:
        return Motivo.PASO_DESCONOCIDO
    if diagnostico["padre_status"] != "running":
        return Motivo.PADRE_INACTIVO
    if diagnostico["paso_facet"] != "ada":
        return Motivo.PASO_NO_ES_ADA
    if diagnostico["padre_depth"] + 1 > max_profundidad:
        return Motivo.PROFUNDIDAD_EXCEDIDA
    return Motivo.ESTADO_CAMBIO


def motivo_consumo(
    diagnostico: dict | None, parent_pipeline_id: str, ahora: float, max_profundidad: int,
) -> Motivo:
    """Por qué el UPDATE del consumo afectó 0 filas. `diagnostico` sale de
    store.subpipeline_token_diagnostico() (None = el hash no existe). El
    estado del paso que delegó no cuenta: solo que exista y sea del padre
    (enmienda 2026-09-16)."""
    if diagnostico is None:
        return Motivo.TOKEN_DESCONOCIDO
    if diagnostico["usado_at"] is not None:
        return Motivo.TOKEN_USADO
    if diagnostico["vence_at"] <= ahora:
        return Motivo.TOKEN_VENCIDO
    if diagnostico["parent_pipeline_id"] != parent_pipeline_id:
        return Motivo.PADRE_NO_COINCIDE
    if diagnostico["depth_hijo"] > max_profundidad:
        return Motivo.PROFUNDIDAD_EXCEDIDA
    if diagnostico["padre_status"] != "running":
        return Motivo.PADRE_INACTIVO
    if diagnostico["paso_step_id"] is None:
        return Motivo.PASO_DESCONOCIDO
    return Motivo.ESTADO_CAMBIO


async def _rechazar_emision(parent_pipeline_id: str, parent_step: str, motivo: Motivo) -> NoReturn:
    await store.event_append(
        parent_pipeline_id, "SUBPIPELINE_RECHAZADO",
        {"fase": "emision", "motivo": motivo.value}, parent_step,
    )
    raise EmisionRechazada(motivo)


async def emitir_token_subpipeline(parent_pipeline_id: str, parent_step: str) -> str:
    """Emite un token de un solo uso para que el paso de Ada `parent_step` del
    pipeline `parent_pipeline_id` cree UN hijo. El paso puede haber terminado
    (modo plan de delegación): lo que se exige es que el pipeline padre siga
    `running`.

    Solo servidor: no hay ruta HTTP (las rutas de Jacobs no autentican). La
    llama el emisor dentro del proceso de Jacobs. Devuelve el token en claro, y
    es la única vez que existe: en la base queda su sha256 y en el evento, los
    primeros 12 caracteres del hash."""
    cfg = config_subpipelines()
    if policy.check_kill_switch():
        await _rechazar_emision(parent_pipeline_id, parent_step, Motivo.KILL_SWITCH_ACTIVO)
    token = secrets.token_urlsafe(TOKEN_BYTES)
    token_hash = hash_token(token)
    emitido_at = time.time()
    vence_at = emitido_at + cfg.ttl_segundos
    depth_hijo = await store.subpipeline_token_emitir(
        token_hash, parent_pipeline_id, parent_step, emitido_at, vence_at, cfg.max_profundidad,
    )
    if depth_hijo is None:
        diagnostico = await store.subpipeline_emision_diagnostico(parent_pipeline_id, parent_step)
        await _rechazar_emision(
            parent_pipeline_id, parent_step,
            motivo_emision(diagnostico, parent_pipeline_id, cfg.max_profundidad),
        )
    await store.event_append(
        parent_pipeline_id, "SUBPIPELINE_TOKEN_EMITIDO",
        {"token_ref": token_ref(token_hash), "depth_hijo": depth_hijo, "vence_at": vence_at},
        parent_step,
    )
    return token
