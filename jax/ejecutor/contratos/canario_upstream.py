# jax/ejecutor/contratos/canario_upstream.py
"""Upstream con API de mensajes de Anthropic, FALSO, para el canario de C1.

Sigue un guion: cada petición «principal» (la que trae herramientas) recibe el
siguiente `tool_use` del guion; cuando el guion se acaba, `end_turn`. Anota los
`tool_result` que devuelve el arnés. Cualquier otra petición (sin herramientas,
p. ej. un modelo chico para títulos) recibe `end_turn` sin consumir el guion; toda
petición queda en `peticiones` (método, ruta, modelo, con_herramientas) para medir
qué manda el arnés de verdad (Task 9).

asyncio + h11, como proxy_carril. Una petición por conexión.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import h11

_LEER = 65536


@dataclass(frozen=True)
class Resultado:
    tool_use_id: str
    es_error: bool
    contenido: str


def guion_bash(tool_use_id: str, comando: str) -> dict:
    return {"id": tool_use_id, "name": "Bash", "input": {"command": comando}}


def _sse(eventos) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in eventos)


def _mensaje(contenido_bloques, stop_reason):
    return {"id": "msg_canario", "type": "message", "role": "assistant", "model": "canario",
            "content": contenido_bloques, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}}


def _stream_tool_use(paso) -> bytes:
    return _sse([
        {"type": "message_start", "message": _mensaje([], None)},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": paso["id"], "name": paso["name"], "input": {}}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": json.dumps(paso["input"])}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
         "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ])


def _stream_fin() -> bytes:
    return _sse([
        {"type": "message_start", "message": _mensaje([], None)},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ])


def _texto_de(contenido) -> str:
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "".join(b.get("text", "") for b in contenido if isinstance(b, dict))
    return ""


class UpstreamCanario:
    def __init__(self, guion: list, host: str, puerto: int):
        self._guion = list(guion)
        self._host, self._puerto_pedido = host, puerto
        self.resultados: dict = {}
        self.peticiones: list = []
        self.puerto: int | None = None
        self._servidor = None

    async def __aenter__(self):
        self._servidor = await asyncio.start_server(self._atender, self._host, self._puerto_pedido)
        self.puerto = self._servidor.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self._servidor.close()
        await self._servidor.wait_closed()
        return False

    def _anotar_resultados(self, cuerpo: dict) -> None:
        # TODOS los mensajes, no el último: medido 2026-09-17 con el arnés 2.1.273, el
        # último mensaje es `role: system` (un recordatorio) y el tool_result va antes.
        for mensaje in cuerpo.get("messages") or []:
            if not isinstance(mensaje, dict) or not isinstance(mensaje.get("content"), list):
                continue
            for bloque in mensaje["content"]:
                if isinstance(bloque, dict) and bloque.get("type") == "tool_result":
                    self.resultados[bloque["tool_use_id"]] = Resultado(
                        bloque["tool_use_id"], bool(bloque.get("is_error", False)), _texto_de(bloque.get("content")))

    def _responder(self, ruta: str, cuerpo: dict):
        if ruta.endswith("/count_tokens"):
            return "application/json", json.dumps({"input_tokens": 1}).encode()
        con_herramientas = bool(cuerpo.get("tools"))
        if con_herramientas:
            self._anotar_resultados(cuerpo)
        paso = self._guion.pop(0) if con_herramientas and self._guion else None
        if cuerpo.get("stream"):
            return "text/event-stream", (_stream_tool_use(paso) if paso else _stream_fin())
        bloques = ([{"type": "tool_use", **paso}] if paso else [{"type": "text", "text": "ok"}])
        return "application/json", json.dumps(_mensaje(bloques, "tool_use" if paso else "end_turn")).encode()

    async def _atender(self, reader, writer):
        conn = h11.Connection(h11.SERVER)
        peticion, trozos = None, []
        try:
            while True:
                evento = conn.next_event()
                if evento is h11.NEED_DATA:
                    conn.receive_data(await reader.read(_LEER))
                    continue
                if isinstance(evento, h11.Request):
                    peticion = evento
                elif isinstance(evento, h11.Data):
                    trozos.append(evento.data)
                elif isinstance(evento, (h11.EndOfMessage, h11.ConnectionClosed)):
                    break
            if peticion is None:
                return
            ruta = peticion.target.split(b"?", 1)[0].decode("latin-1")
            try:
                cuerpo = json.loads(b"".join(trozos) or b"{}")
            except ValueError:
                cuerpo = {}
            self.peticiones.append((peticion.method.decode(), ruta, cuerpo.get("model"), bool(cuerpo.get("tools"))))
            if peticion.method != b"POST":
                # Medido: el arnés arranca con `HEAD /api/hello`. Una respuesta a HEAD no lleva cuerpo.
                writer.write(conn.send(h11.Response(status_code=200, headers=[
                    ("content-length", "0"), ("connection", "close")])))
                writer.write(conn.send(h11.EndOfMessage()))
                await writer.drain()
                return
            tipo, datos = self._responder(ruta, cuerpo)
            writer.write(conn.send(h11.Response(status_code=200, headers=[
                ("content-type", tipo), ("content-length", str(len(datos))), ("connection", "close")])))
            writer.write(conn.send(h11.Data(data=datos)))
            writer.write(conn.send(h11.EndOfMessage()))
            await writer.drain()
        except (ConnectionError, h11.RemoteProtocolError):  # fail-soft: upstream de canario; si el arnés corta, el canario no ve su resultado y verificar_c1 lo reporta como fallo
            return
        finally:
            writer.close()
