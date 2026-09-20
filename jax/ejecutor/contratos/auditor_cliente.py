# jax/ejecutor/contratos/auditor_cliente.py
"""Llamada a la faceta auditora de C5 (resuelta en vivo; nunca un modelo en código).

Transportes soportados: `http_openai_compat` (Thot, Kimi, Ada según la semilla) y
`ollama` (el auditor local, spec 2026-09-18-auditor-local-opcion.md). Los dos hablan el
MISMO body de OpenAI-compat por HTTP -- 'ollama' es sólo la etiqueta que usa el proveedor
local sin credencial gestionada (mismo motivo que la faceta 'jax_local': facet_resolver
exime de `credential` a los transportes 'ollama'/'subprocess'; pedirle una llave a un
Ollama que no la usa sería inventar un secreto de mentira). Otro transporte →
AuditorNoSoportado, y el arranque se niega. Error del proveedor, red caída, tope vencido
o forma inesperada → AuditorIlegible("proveedor_fallo"): quien llama frena. La excepción
de origen NO se encadena: un error HTTP puede traer la llave o el cuerpo. Cliente HTTP
compartido (E-24).
"""
from __future__ import annotations

from pathlib import Path

import httpx

from jax.core.cliente_http_compartido import obtener_cliente_http
from jax.ejecutor.contratos import auditor as A

_INSTRUCCIONES = Path(__file__).with_name("auditor_instrucciones.md")
TRANSPORTE = "http_openai_compat"
TRANSPORTES_SOPORTADOS = (TRANSPORTE, "ollama")


class AuditorNoSoportado(RuntimeError):
    """`args[0]` es el transporte."""


def instrucciones() -> str:
    return _INSTRUCCIONES.read_text(encoding="utf-8")


async def auditar(lote: A.Lote, *, faceta, max_tokens: int, cliente=None, tope_s: float = 120.0) -> A.Revision:
    if faceta.transport not in TRANSPORTES_SOPORTADOS:
        raise AuditorNoSoportado(faceta.transport)
    cliente = cliente or obtener_cliente_http()
    cuerpo = {"model": faceta.model, "max_completion_tokens": max_tokens,
              "messages": A.mensajes(lote, instrucciones())}
    try:
        # Sin credencial NO se manda la cabecera: la ausencia es un hecho del transporte
        # ('ollama' esta exento por facet_resolver), no un valor vacio que se serializa.
        # La f-string incondicional producia "Bearer " -- con espacio final -- y h11
        # rechaza una cabecera con espacio al final: LocalProtocolError, subclase de
        # httpx.HTTPError, que el except de abajo disfrazaba de "el modelo escribio mal".
        cabeceras = {"authorization": f"Bearer {faceta.credential}"} if faceta.credential else {}
        r = await cliente.post(faceta.base_url.rstrip("/") + "/chat/completions", json=cuerpo,
                               headers=cabeceras, timeout=tope_s)
        r.raise_for_status()
        texto = r.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        raise A.AuditorIlegible("proveedor_fallo") from None
    return A.interpretar(lote, texto)
