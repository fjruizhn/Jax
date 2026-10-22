"""Adaptador de lectura de memoria B9 para El Faro 0.4.

El reader es una dependencia inyectada; este módulo no abre conexiones por su
cuenta. La identidad procede del socket y nunca de argumentos MCP. B9 sigue
siendo memoria contextual, no autoridad ni evidencia de estado actual.
"""
from __future__ import annotations

import asyncio
import math
from typing import Any

from jax.memory.b9 import ScopeContext, ScopeDenied

from .no_confiable import envolver_payload_no_confiable


MAX_BYTES_PAYLOAD_MEMORIA = 8192
MARCA_TRUNCADO = "\n[contenido truncado por límite de memoria]"


class MemoriaNoDisponible(RuntimeError):
    """El adaptador no está habilitado o su fuente no puede responder."""


class AdaptadorMemoria:
    def __init__(self, lector: Any | None = None, *, timeout_s: float = 5.0):
        if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s debe ser positivo y finito")
        self._lector = lector
        self._timeout_s = float(timeout_s)

    async def buscar(self, identidad, consulta: str, limite: int) -> list[dict[str, str]]:
        if identidad is None:
            return []
        if self._lector is None:
            raise MemoriaNoDisponible("memoria no disponible")
        if not isinstance(consulta, str) or not consulta.strip() or len(consulta) > 200:
            raise ValueError("consulta debe tener entre 1 y 200 caracteres")
        if not isinstance(limite, int) or isinstance(limite, bool) or not 1 <= limite <= 100:
            raise ValueError("limite fuera de rango (1..100)")
        try:
            ejecucion = identidad.ejecucion
            usuario, tenant = ejecucion.usuario, ejecucion.tenant
            if not isinstance(usuario, str) or not usuario.strip() or not isinstance(tenant, str) or not tenant.strip():
                raise ScopeDenied("identidad incompleta")
            scope = ScopeContext(actor_principal=f"user:{usuario}", actor_type="USER",
                                 subject_user_id=usuario, tenant_id=tenant)
            scope.validate()
            # El B9 reader es limitado y ordena por revisión reciente. El filtro
            # lexical provisional opera solo sobre esos datos ya scopeados.
            envelopes = await asyncio.wait_for(self._lector.retrieve(scope, limit=100), self._timeout_s)
        except Exception as exc:
            raise MemoriaNoDisponible("memoria no disponible") from exc

        needle = consulta.casefold().strip()
        encontrados = []
        for envelope in envelopes:
            contenido = envelope.revision.payload
            if not isinstance(contenido, str) or needle not in contenido.casefold():
                continue
            if len(contenido) > MAX_BYTES_PAYLOAD_MEMORIA:
                prefijo = contenido[:MAX_BYTES_PAYLOAD_MEMORIA].encode("utf-8")
            else:
                prefijo = contenido.encode("utf-8")
            truncado = len(prefijo) > MAX_BYTES_PAYLOAD_MEMORIA or len(contenido) > MAX_BYTES_PAYLOAD_MEMORIA
            contenido_visible = prefijo[:MAX_BYTES_PAYLOAD_MEMORIA].decode("utf-8", "replace")
            if truncado:
                contenido_visible += MARCA_TRUNCADO
            encontrados.append({
                "memory_id": str(envelope.identity.memory_id),
                "revision_id": str(envelope.revision.revision_id),
                "content": envolver_payload_no_confiable(contenido_visible),
            })
            if len(encontrados) == limite:
                break
        return encontrados


async def crear_pool_memoria_prueba(cfg):
    """Abre exclusivamente el perfil de prueba permitido por `ConfigMemoria`."""
    import aiomysql
    from jax.core.db_connect_config import db_connect_timeout_seconds
    from jax.memory.b9_mariadb import MariaDBB9Reader

    # Revalidar en el mismo límite que abre la conexión: ni los llamadores
    # internos ni una instancia mutada pueden redirigir el helper a producción.
    from jax.faro.config import ConfigFaroInvalida, ConfigMemoria
    if not isinstance(cfg, ConfigMemoria):
        raise ConfigFaroInvalida("memoria no disponible: configuración de prueba inválida")
    if (cfg.habilitada is not True
            or (cfg.host, cfg.port, cfg.usuario, cfg.base) != ("127.0.0.1", 3308, "jax_test", "jax_memory_test")
            or not isinstance(cfg.clave, str) or not cfg.clave
            or not isinstance(cfg.timeout_s, (int, float)) or isinstance(cfg.timeout_s, bool)
            or not math.isfinite(cfg.timeout_s) or not 0 < cfg.timeout_s <= 30):
        raise ConfigFaroInvalida("memoria no disponible: la conexión solo permite el perfil local de prueba")

    pool = await aiomysql.create_pool(
        host=cfg.host, port=cfg.port, user=cfg.usuario, password=cfg.clave, db=cfg.base,
        minsize=1, maxsize=4, cursorclass=aiomysql.DictCursor, autocommit=True,
        connect_timeout=db_connect_timeout_seconds(),
    )
    return pool, MariaDBB9Reader(pool, max_payload_bytes=MAX_BYTES_PAYLOAD_MEMORIA,
                                rollback_timeout_s=cfg.timeout_s)
