"""
Fuentes de grounding duplicadas por la redirección de Google (DEUDA.md,
anotados de la ronda b8f80733, 2026-09-12).

Visto en vivo en el pipeline 04e02b09: `[1]` y `[2]` con la MISMA URL final.
Gemini entrega un token de redirección distinto por chunk, así que
`build_sources` (que deduplica por URI) no puede verlas iguales: recién
después de `resolve_redirects` se sabe que dos fuentes son el mismo
documento. Ahí se fusionan, conservando la primera y sumando sus citas.

Una fuente cuya redirección NO se resolvió no se fusiona con nada: sin URL
final no hay forma de saber qué documento es, y fusionar a ciegas escondería
una fuente real.

Los llamadores (jacobs/executor.py y jax/muscles/base.py) ignoran el valor
de retorno y siguen usando la lista que pasaron, así que la fusión tiene que
ocurrir sobre ESA lista.

Ningún test sale a la red: httpx.MockTransport.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import unittest

import httpx

from jax.core import grounding_sources as gs

_BASE = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
R1, R2, R3 = _BASE + "AAA", _BASE + "BBB", _BASE + "CCC"
F1 = "https://www.postgresql.org/about/news/postgresql-186-released/"
F2 = "https://es.wikipedia.org/wiki/PostgreSQL"


def _client(redirecciones: dict[str, str]) -> httpx.AsyncClient:
    """HEAD a una redirección conocida -> 302 con su Location; cualquier otra
    cosa -> 404 (redirección no resuelta)."""

    def handler(request: httpx.Request) -> httpx.Response:
        destino = redirecciones.get(str(request.url))
        if destino:
            return httpx.Response(302, headers={"location": destino})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _resolver(sources: list[dict], redirecciones: dict[str, str]) -> list[dict]:
    async def correr():
        async with _client(redirecciones) as client:
            return await gs.resolve_redirects(sources, client=client)

    return asyncio.run(correr())


class FuentesDuplicadasTest(unittest.TestCase):
    def test_dos_redirecciones_al_mismo_documento_quedan_como_una_fuente(self):
        sources = [
            {"title": "postgresql.org", "url": R1, "quotes": ["q1"]},
            {"title": "postgresql.org", "url": R2, "quotes": ["q2", "q1"]},
        ]
        _resolver(sources, {R1: F1, R2: F1})
        # Sobre la lista que pasó el llamador: los dos call sites ignoran el retorno.
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["final_url"], F1)
        self.assertEqual(sources[0]["quotes"], ["q1", "q2"])

    def test_se_conserva_el_orden_de_primera_aparicion(self):
        sources = [
            {"title": "a", "url": R1, "quotes": ["q1"]},
            {"title": "b", "url": R2, "quotes": ["q2"]},
            {"title": "c", "url": R3, "quotes": ["q3"]},
        ]
        _resolver(sources, {R1: F1, R2: F2, R3: F1})
        self.assertEqual([s["final_url"] for s in sources], [F1, F2])
        self.assertEqual(sources[0]["quotes"], ["q1", "q3"])
        self.assertEqual(sources[1]["quotes"], ["q2"])

    def test_las_no_resueltas_no_se_fusionan(self):
        sources = [
            {"title": "a", "url": R1, "quotes": ["q1"]},
            {"title": "b", "url": R2, "quotes": ["q2"]},
        ]
        _resolver(sources, {})
        self.assertEqual(len(sources), 2)
        self.assertEqual([s["resolved"] for s in sources], [False, False])

    def test_el_bloque_renderizado_numera_sin_repetir_el_documento(self):
        sources = [
            {"title": "a", "url": R1, "quotes": ["q1"]},
            {"title": "b", "url": R2, "quotes": ["q2"]},
        ]
        _resolver(sources, {R1: F1, R2: F1})
        bloque = gs.render_sources_block(sources)
        self.assertEqual(bloque.count(F1), 1)
        self.assertNotIn("[2]", bloque)


if __name__ == "__main__":
    unittest.main()
