"""
Fuentes de grounding de Gemini, verificables por otro modelo (2026-09-12).

Gemini devuelve cada fuente como una redirección opaca de Google
(`vertexaisearch.cloud.google.com/grounding-api-redirect/...`) con solo una
etiqueta de dominio. En la tercera E2E de la cadena (pipeline b2d87971) la
auditoría no pudo contrastar ninguna de las 6 fuentes de hipatia.

Acá cada fuente gana dos cosas:
- la URL FINAL: se sigue la redirección leyendo solo el header `Location`
  (HEAD; si no viene, GET en streaming sin leer el cuerpo). Si no se
  resuelve, se conserva la de Google y se marca -- nunca se inventa una URL;
- las citas: los fragmentos de la respuesta que esa fuente respalda, según
  `groundingSupports` (segment.text + groundingChunkIndices).

Sin caché: los tokens de redirección son únicos por respuesta; cachearlos no
ahorraría nada medible.

Dónde vive (2026-09-12): en jax/core, la capa base, porque lo usan Jacobs
(jacobs/executor.py) y el HttpMuscle del REPL (jax/muscles/base.py). El REPL no
puede depender de jacobs/ (capas invertidas), y el proceso de LAS MANOS no
importa `jax.*`: le llega por el symlink las_manos/grounding_sources.py, igual
que facet_resolver. Un solo archivo, dos rutas de import.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_REDIRECT_HOST = "vertexaisearch.cloud.google.com"
_REDIRECT_PATH = "/grounding-api-redirect/"
# Medido 2026-09-12 contra redirecciones reales: 0,2-0,45 s por fuente. 4 s
# es el tope por fuente; se resuelven todas a la vez.
RESOLVE_TIMEOUT_SECONDS = 4.0


def build_sources(chunks: list[dict] | None, supports: list[dict] | None) -> list[dict]:
    """Una fuente por URI (en el orden de los chunks), con las citas que
    respalda. Dos chunks con la misma URI son la misma fuente."""
    sources: list[dict] = []
    index_by_uri: dict[str, int] = {}
    chunk_to_source: dict[int, int] = {}
    for i, chunk in enumerate(chunks or []):
        web = chunk.get("web") or {}
        uri = web.get("uri")
        if not uri:
            continue
        if uri not in index_by_uri:
            index_by_uri[uri] = len(sources)
            sources.append({"title": web.get("title") or uri, "url": uri, "quotes": []})
        chunk_to_source[i] = index_by_uri[uri]

    for support in supports or []:
        text = ((support.get("segment") or {}).get("text") or "").strip()
        if not text:
            continue
        for chunk_index in support.get("groundingChunkIndices") or []:
            source_index = chunk_to_source.get(chunk_index)
            if source_index is not None and text not in sources[source_index]["quotes"]:
                sources[source_index]["quotes"].append(text)
    return sources


def _is_grounding_redirect(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == _REDIRECT_HOST and parsed.path.startswith(_REDIRECT_PATH)


async def _resolve_one(client: httpx.AsyncClient, source: dict) -> None:
    url = source["url"]
    if not _is_grounding_redirect(url):
        source["final_url"], source["resolved"] = url, True
        return
    location = None
    try:
        head = await client.head(url, follow_redirects=False)
        location = head.headers.get("location") if head.is_redirect else None
        if not location:
            async with client.stream("GET", url, follow_redirects=False) as resp:
                location = resp.headers.get("location") if resp.is_redirect else None
    except httpx.HTTPError as exc:
        logger.warning("Fuente '%s': redirección no resuelta (%s)", source.get("title"), exc)
    source["final_url"] = location or None
    source["resolved"] = bool(location)


def _merge_by_final_url(sources: list[dict]) -> None:
    """Fusiona las fuentes que resolvieron a la MISMA URL final (2026-09-14).

    Gemini da un token de redirección distinto por chunk, así que dos chunks
    del mismo documento llegan como URIs distintas y `build_sources` no las
    puede ver iguales (pipeline 04e02b09: [1] y [2] con la misma URL). Recién
    acá se sabe. Queda la primera, con las citas de las dos, sin repetir y en
    orden. Una fuente NO resuelta no se fusiona con nada: sin URL final no se
    sabe qué documento es. Muta la lista recibida porque los llamadores
    (jacobs/executor.py, jax/muscles/base.py) ignoran el retorno."""
    merged: list[dict] = []
    by_final_url: dict[str, dict] = {}
    for source in sources:
        final_url = source.get("final_url") if source.get("resolved") else None
        first = by_final_url.get(final_url) if final_url else None
        if first is None:
            if final_url:
                by_final_url[final_url] = source
            merged.append(source)
            continue
        quotes = first.setdefault("quotes", [])
        for quote in source.get("quotes") or []:
            if quote not in quotes:
                quotes.append(quote)
    sources[:] = merged


async def resolve_redirects(sources: list[dict], client: httpx.AsyncClient | None = None) -> list[dict]:
    """Agrega `final_url` y `resolved` a cada fuente, todas en paralelo, y
    fusiona las que resultan ser el mismo documento (ver _merge_by_final_url).
    `client` es para tests (MockTransport); en producción se crea uno."""
    if not sources:
        return sources
    if client is None:
        async with httpx.AsyncClient(timeout=RESOLVE_TIMEOUT_SECONDS) as own:
            await asyncio.gather(*(_resolve_one(own, s) for s in sources))
    else:
        await asyncio.gather(*(_resolve_one(client, s) for s in sources))
    _merge_by_final_url(sources)
    return sources


def render_sources_block(sources: list[dict]) -> str:
    """Texto de fuentes para el paso siguiente y para el .md del repo."""
    lines: list[str] = []
    for i, source in enumerate(sources, 1):
        title = source.get("title") or ""
        if source.get("resolved") and source.get("final_url"):
            lines.append(f"[{i}] {title} — {source['final_url']}")
        else:
            lines.append(
                f"[{i}] {title} — redirección NO resuelta (enlace opaco de Google): {source.get('url', '')}"
            )
        quotes = source.get("quotes") or []
        for quote in quotes:
            lines.append(f'    > "{quote}"')
        if not quotes:
            lines.append("    (sin fragmento de la respuesta asociado a esta fuente)")
    return "\n".join(lines)
