# jax/ejecutor/contratos/auditor_cliente.py
"""Llamada a la faceta auditora de C5 (resuelta en vivo; nunca un modelo en código).

Sólo transporte `http_openai_compat` (Thot, Kimi, Ada según la semilla): otro
transporte → AuditorNoSoportado, y el arranque se niega. Error del proveedor, red caída,
tope vencido o forma inesperada → AuditorIlegible("proveedor_fallo"): quien llama frena.
La excepción de origen NO se encadena: un error HTTP puede traer la llave o el cuerpo.
Cliente HTTP compartido (E-24).
"""
from __future__ import annotations

from pathlib import Path

import httpx

from jax.core.cliente_http_compartido import obtener_cliente_http
from jax.ejecutor.contratos import auditor as A

_INSTRUCCIONES = Path(__file__).with_name("auditor_instrucciones.md")
TRANSPORTE = "http_openai_compat"


class AuditorNoSoportado(RuntimeError):
    """`args[0]` es el transporte."""


def instrucciones() -> str:
    return _INSTRUCCIONES.read_text(encoding="utf-8")


async def auditar(lote: A.Lote, *, faceta, max_tokens: int, cliente=None, tope_s: float = 120.0) -> A.Revision:
    if faceta.transport != TRANSPORTE:
        raise AuditorNoSoportado(faceta.transport)
    cliente = cliente or obtener_cliente_http()
    cuerpo = {"model": faceta.model, "max_completion_tokens": max_tokens,
              "messages": A.mensajes(lote, instrucciones())}
    try:
        r = await cliente.post(faceta.base_url.rstrip("/") + "/chat/completions", json=cuerpo,
                               headers={"authorization": f"Bearer {faceta.credential}"}, timeout=tope_s)
        r.raise_for_status()
        texto = r.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        raise A.AuditorIlegible("proveedor_fallo") from None
    return A.interpretar(lote, texto)
