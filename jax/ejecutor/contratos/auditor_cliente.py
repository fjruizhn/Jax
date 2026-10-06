# jax/ejecutor/contratos/auditor_cliente.py
"""Llamada a la faceta auditora de C5 (resuelta en vivo; nunca un modelo en código).

Transportes soportados: `http_openai_compat` (Thot, Kimi, Ada según la semilla) y
`ollama` (el auditor local, spec 2026-09-18-auditor-local-opcion.md). Los dos hablan el
MISMO body de OpenAI-compat por HTTP -- 'ollama' es sólo la etiqueta que usa el proveedor
local sin credencial gestionada (mismo motivo que la faceta 'jax_local': facet_resolver
exime de `credential` a los transportes 'ollama'/'subprocess'; pedirle una llave a un
Ollama que no la usa sería inventar un secreto de mentira). Otro transporte →
AuditorNoSoportado, y el arranque se niega. Error del proveedor, red caída
o forma inesperada → AuditorIlegible("proveedor_fallo"), y el plazo vencido →
AuditorIlegible("proveedor_plazo"): quien llama frena en los dos casos. La excepción
de origen NO se encadena: un error HTTP puede traer la llave o el cuerpo. Cliente HTTP
compartido (E-24).

`tope_s` es OBLIGATORIO y sale de `axioma_config` (`ejecutor.c5_tope_s`, ver
`eleccion_c5.ConfigC5.tope_s`): sin valor por omision, un consumidor que lo olvide no corre
en vez de heredar un plazo escrito en codigo.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import httpx

from jax.core.cliente_http_compartido import obtener_cliente_http
from jax.ejecutor.contratos import auditor as A

_INSTRUCCIONES = Path(__file__).with_name("auditor_instrucciones.md")
TRANSPORTE = "http_openai_compat"
TRANSPORTES_SOPORTADOS = (TRANSPORTE, "ollama")
_CODIGOS_PROVEEDOR_PUBLICABLES = frozenset({
    "insufficient_quota", "rate_limit_exceeded", "invalid_api_key", "model_not_found",
    "context_length_exceeded", "server_error", "bad_request", "authentication_error",
})


def _codigo_proveedor_http(respuesta) -> str | None:
    """Extrae solo códigos conocidos; nunca propaga mensaje, cuerpo ni cabeceras."""
    try:
        error = respuesta.json().get("error", {})
        if not isinstance(error, dict):
            return None
        for campo in ("code", "type"):
            codigo = error.get(campo)
            if isinstance(codigo, str) and codigo in _CODIGOS_PROVEEDOR_PUBLICABLES:
                return codigo
    except (ValueError, AttributeError):  # fail-soft: código proveedor es solo detalle; el status HTTP conserva el rechazo
        pass
    return None


class AuditorNoSoportado(RuntimeError):
    """`args[0]` es el transporte."""


def instrucciones(modo: str = "COMPLETO") -> str:
    texto = _INSTRUCCIONES.read_text(encoding="utf-8")
    if modo not in ("COMPLETO", "SOLO_ORDENES"):
        raise ValueError("modo_auditoria_desconocido")
    orden_inicio, orden_fin = "<!-- ORDENES_COMPARTIDAS: inicio -->", "<!-- ORDENES_COMPARTIDAS: fin -->"
    if texto.count(orden_inicio) != 1 or texto.count(orden_fin) != 1:
        raise ValueError("instrucciones_ordenes_compartidas_ausentes")
    _, cola_ordenes = texto.split(orden_inicio, 1)
    ordenes, _ = cola_ordenes.split(orden_fin, 1)
    inicio, fin = "<!-- SOLO_ORDENES: inicio -->", "<!-- SOLO_ORDENES: fin -->"
    if texto.count(inicio) != 1 or texto.count(fin) != 1:
        raise ValueError("instrucciones_solo_ordenes_ausentes")
    anterior, cola = texto.split(inicio, 1)
    solo, posterior = cola.split(fin, 1)
    if posterior.strip():
        raise ValueError("instrucciones_solo_ordenes_fuera_de_bloque")
    if modo == "SOLO_ORDENES":
        return ordenes.strip() + "\n\n" + solo.strip()
    return anterior.replace(orden_inicio, "").replace(orden_fin, "").rstrip()


async def auditar(lote: A.Lote, *, faceta, max_tokens: int, tope_s: float, cliente=None,
                  modo: str = "COMPLETO") -> A.Revision:
    if faceta.transport not in TRANSPORTES_SOPORTADOS:
        raise AuditorNoSoportado(faceta.transport)
    cliente = cliente or obtener_cliente_http()
    try:
        cuerpo = {"model": faceta.model, "max_completion_tokens": max_tokens,
                  "messages": A.mensajes(lote, instrucciones(modo), modo=modo)}
    except A.AuditorIlegible as exc:
        raise A.AuditorIlegible(exc.codigo, modo=modo, faceta=getattr(faceta, "key", None)) from None
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
    except httpx.TimeoutException:
        # Plazo vencido: distinto de "el proveedor fallo" (la cola detras del cerebro en la unica
        # ranura de la GPU es la causa conocida). Para quien llama es lo mismo: falla cerrado.
        raise A.AuditorIlegible("proveedor_plazo", modo=modo, faceta=getattr(faceta, "key", None)) from None
    except httpx.HTTPStatusError as exc:
        raise A.AuditorIlegible("proveedor_fallo", modo=modo, faceta=getattr(faceta, "key", None),
                                proveedor_codigo=_codigo_proveedor_http(exc.response)) from None
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        raise A.AuditorIlegible("proveedor_fallo", modo=modo, faceta=getattr(faceta, "key", None)) from None
    try:
        revision = A.interpretar(lote, texto, modo=modo)
    except A.AuditorIlegible as exc:
        raise A.AuditorIlegible(exc.codigo, modo=modo, faceta=getattr(faceta, "key", None)) from None
    return replace(revision, faceta=getattr(faceta, "key", None))
