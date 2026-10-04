"""Adaptador de lectura de memoria B9 para El Faro 0.4.

El reader es una dependencia inyectada; este módulo no abre conexiones por su
cuenta. La identidad procede del socket y nunca de argumentos MCP. B9 sigue
siendo memoria contextual, no autoridad ni evidencia de estado actual.
"""
from __future__ import annotations

from typing import Any

from jax.memory.b9 import ScopeContext, ScopeDenied


class MemoriaNoDisponible(RuntimeError):
    """El adaptador no está habilitado o su fuente no puede responder."""


class AdaptadorMemoria:
    def __init__(self, lector: Any | None = None):
        self._lector = lector

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
            envelopes = await self._lector.retrieve(scope, limit=100)
        except Exception as exc:
            raise MemoriaNoDisponible("memoria no disponible") from exc

        needle = consulta.casefold().strip()
        encontrados = []
        for envelope in envelopes:
            contenido = envelope.revision.payload
            if not isinstance(contenido, str) or needle not in contenido.casefold():
                continue
            encontrados.append({
                "memory_id": str(envelope.identity.memory_id),
                "revision_id": str(envelope.revision.revision_id),
                "trust": "untrusted_source",
                "content": contenido,
            })
            if len(encontrados) == limite:
                break
        return encontrados


async def crear_pool_memoria_prueba(cfg):
    """Abre exclusivamente el perfil de prueba permitido por `ConfigMemoria`."""
    import aiomysql
    from jax.core.db_connect_config import db_connect_timeout_seconds
    from jax.memory.b9_mariadb import MariaDBB9Reader

    pool = await aiomysql.create_pool(
        host=cfg.host, port=cfg.port, user=cfg.usuario, password=cfg.clave, db=cfg.base,
        minsize=1, maxsize=4, cursorclass=aiomysql.DictCursor, autocommit=True,
        connect_timeout=db_connect_timeout_seconds(),
    )
    return pool, MariaDBB9Reader(pool)
