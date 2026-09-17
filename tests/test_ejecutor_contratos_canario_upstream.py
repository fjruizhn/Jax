# tests/test_ejecutor_contratos_canario_upstream.py
"""Upstream Anthropic falso para el canario de C1: sigue un guion de tool_use y
anota los tool_result que le devuelve el arnés. Sin modelo."""
import asyncio
import json

import httpx

from jax.ejecutor.contratos.canario_upstream import Resultado, UpstreamCanario, guion_bash

_HERRAMIENTAS = [{"name": "Bash", "input_schema": {"type": "object"}}]


def _eventos(texto):
    return [json.loads(l[len("data: "):]) for l in texto.splitlines() if l.startswith("data: ")]


def test_sigue_el_guion_y_anota_los_resultados():
    async def escenario():
        guion = [guion_bash("toolu_a", "echo a"), guion_bash("toolu_b", "echo b")]
        async with UpstreamCanario(guion, "127.0.0.1", 0) as up, httpx.AsyncClient() as cli:
            url = f"http://127.0.0.1:{up.puerto}/v1/messages"
            base = {"model": "canario", "stream": True, "tools": _HERRAMIENTAS, "max_tokens": 10}
            r1 = await cli.post(url, json={**base, "messages": [{"role": "user", "content": "x"}]})
            # Forma medida del arnés real (2026-09-17): el último mensaje es `system`.
            recordatorio = {"role": "system", "content": [{"type": "text", "text": "recordatorio"}]}
            r2 = await cli.post(url, json={**base, "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_a", "is_error": True,
                 "content": [{"type": "text", "text": "bloqueado"}]}]}, recordatorio]})
            r3 = await cli.post(url, json={**base, "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_b", "content": "b"}]}, recordatorio]})
            otra = await cli.post(url, json={"model": "chico", "messages": [], "max_tokens": 5})
            hola = await cli.head(f"http://127.0.0.1:{up.puerto}/api/hello")
            return r1.text, r2.text, r3.text, otra.json(), hola.status_code, dict(up.resultados), list(up.peticiones)

    t1, t2, t3, otra, hola, resultados, peticiones = asyncio.run(asyncio.wait_for(escenario(), 20))
    assert hola == 200
    inicio = [e for e in _eventos(t1) if e["type"] == "content_block_start"][0]
    assert inicio["content_block"] == {"type": "tool_use", "id": "toolu_a", "name": "Bash", "input": {}}
    deltas = "".join(e["delta"]["partial_json"] for e in _eventos(t1) if e["type"] == "content_block_delta")
    assert json.loads(deltas) == {"command": "echo a"}
    assert [e for e in _eventos(t2) if e["type"] == "content_block_start"][0]["content_block"]["id"] == "toolu_b"
    assert [e for e in _eventos(t3) if e["type"] == "message_delta"][0]["delta"]["stop_reason"] == "end_turn"
    assert otra["stop_reason"] == "end_turn"
    assert resultados == {"toolu_a": Resultado("toolu_a", True, "bloqueado"), "toolu_b": Resultado("toolu_b", False, "b")}
    assert [(p[0], p[3]) for p in peticiones] == [("POST", True), ("POST", True), ("POST", True), ("POST", False), ("HEAD", False)]
