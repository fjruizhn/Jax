"""Authenticated model ingress for the dedicated jaxqwen identity."""
from __future__ import annotations

import asyncio
import json
import os
import socket
import struct

import httpx
import pytest

from jax.ejecutor.proxy_carril import Config, ConfigInvalida, _Proxy, arrancar, config_desde_entorno
from tests.test_ejecutor_proxy_carril import MODELO_PERMITIDO, Upstream, _CUERPO, _correr
from jax.ejecutor.contratos.pausa import latir


def test_peercred_requires_exact_dedicated_uid():
    class Peer:
        def __init__(self, uid): self.uid = uid
        def getsockopt(self, *_): return struct.pack("3i", 123, self.uid, 45)

    assert _Proxy._peer_uid_permitido(Peer(1001), 1001)
    assert not _Proxy._peer_uid_permitido(Peer(1000), 1001)
    class Broken:
        def getsockopt(self, *_): raise OSError("credential unavailable")
    assert not _Proxy._peer_uid_permitido(Broken(), 1001)
    assert not _Proxy._peer_uid_permitido(object(), 1001)


def test_qwen_socket_config_is_all_or_nothing_and_absolute():
    base = {
        "JAX_PROXY_CARRIL_UPSTREAM": "http://127.0.0.1:9",
        "JAX_PROXY_CARRIL_RAIZ": "/tmp/locks", "JAX_PROXY_CARRIL_TOPE_S": "5",
        "JAX_PROXY_CARRIL_PUERTO": "0", "JAX_EJECUTOR_REGISTRO": "/tmp/log",
        "JAX_EJECUTOR_PAUSA": "/tmp/pause", "JAX_EJECUTOR_VIGIA_LATIDO": "/tmp/beat",
        "JAX_EJECUTOR_VIGIA_LATIDO_MAX_S": "30", "JAX_PROXY_CARRIL_MODELO": "qwen-fixed", "JAX_PROXY_CARRIL_PENSAMIENTO": "libre",
        "JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS": "1024",
    }
    assert config_desde_entorno(base).jaxqwen_socket is None
    for update in ({"JAX_PROXY_CARRIL_JAXQWEN_SOCKET": "/tmp/qwen.sock"},
                   {"JAX_PROXY_CARRIL_JAXQWEN_UID": "1001"},
                   {"JAX_PROXY_CARRIL_JAXQWEN_GID": "1002"}):
        with pytest.raises(ConfigInvalida): config_desde_entorno({**base, **update})
    complete = {**base, "JAX_PROXY_CARRIL_JAXQWEN_SOCKET": "/tmp/qwen.sock",
                "JAX_PROXY_CARRIL_JAXQWEN_UID": "1001", "JAX_PROXY_CARRIL_JAXQWEN_GID": "1002"}
    assert config_desde_entorno(complete).jaxqwen_uid == 1001
    with pytest.raises(ConfigInvalida):
        config_desde_entorno({**complete, "JAX_PROXY_CARRIL_JAXQWEN_SOCKET": "relative.sock"})


def test_unix_transport_authenticates_uid_and_preserves_fixed_proxy_gates(tmp_path):
    async def scenario():
        async with Upstream(modo="error") as upstream:
            parent = tmp_path / "run"
            parent.mkdir(mode=0o700)
            path = parent / "jaxqwen.sock"
            cfg = Config(upstream=upstream.url, raiz=tmp_path / "locks", tope_s=2,
                         host="127.0.0.1", puerto=0, registro=tmp_path / "registro.jsonl",
                         pausa=tmp_path / "PAUSA", latido=tmp_path / "latido", latido_max_s=60,
                         modelo=MODELO_PERMITIDO, max_salida_tokens=1024, pensamiento="libre",
                         jaxqwen_socket=path, jaxqwen_uid=os.getuid(), jaxqwen_gid=os.getgid())
            cfg.raiz.mkdir()
            latir(cfg.latido)
            server = await arrancar(cfg)
            try:
                assert path.is_socket()
                assert path.stat().st_mode & 0o777 == 0o660
                transport = httpx.AsyncHTTPTransport(uds=str(path))
                async with httpx.AsyncClient(transport=transport, base_url="http://jaxqwen") as client:
                    ok = await client.post("/v1/messages", content=_CUERPO)
                    rejected = await client.post("/v1/messages", content=b'{"model":"other","max_tokens":1}')
                assert ok.status_code == 500  # fake upstream reached only through authenticated peer
                assert rejected.status_code == 403  # existing exact model gate remains active
                assert len(upstream.recibidas) == 1
            finally:
                server.close()
                await server.wait_closed()
                assert not path.exists()
    _correr(scenario())


def test_unix_listener_refuses_preexisting_socket_path(tmp_path):
    async def scenario():
        parent = tmp_path / "run"
        parent.mkdir(mode=0o700)
        path = parent / "jaxqwen.sock"
        path.write_text("attacker replacement", encoding="utf-8")
        cfg = Config(upstream="http://127.0.0.1:9", raiz=tmp_path / "locks", tope_s=1,
                     host="127.0.0.1", puerto=0, registro=tmp_path / "registro.jsonl",
                     pausa=tmp_path / "PAUSA", latido=tmp_path / "latido", latido_max_s=60,
                     modelo=MODELO_PERMITIDO, max_salida_tokens=1024, pensamiento="libre",
                     jaxqwen_socket=path, jaxqwen_uid=os.getuid(), jaxqwen_gid=os.getgid())
        cfg.raiz.mkdir()
        with pytest.raises(ConfigInvalida): await arrancar(cfg)
        assert path.read_text(encoding="utf-8") == "attacker replacement"
    _correr(scenario())


def test_por_jaxqwen_con_apagado_el_thinking_llega_intacto_y_por_la_entrada_normal_no(tmp_path):
    """La reescritura de `apagado` es SOLO de la entrada del Ejecutor (decisión de jax-14,
    2026-10-03); el socket de jaxqwen sigue byte a byte. La validación de claves ambiguas
    (MAJOR-5) sí aplica a las dos entradas."""
    cuerpo = (b'{"model":"modelo-permitido","max_tokens":1024,"thinking":{"type":"enabled","budget_tokens":2048},'
              b'"messages":[{"role":"user","content":"x"}]}')
    ambiguo = b'{"model":"modelo-permitido","Max_Tokens":999999,"max_tokens":1,"messages":[]}'

    async def scenario():
        async with Upstream(modo="error") as upstream:
            parent = tmp_path / "run"
            parent.mkdir(mode=0o700)
            path = parent / "jaxqwen.sock"
            cfg = Config(upstream=upstream.url, raiz=tmp_path / "locks", tope_s=2,
                         host="127.0.0.1", puerto=0, registro=tmp_path / "registro.jsonl",
                         pausa=tmp_path / "PAUSA", latido=tmp_path / "latido", latido_max_s=60,
                         modelo=MODELO_PERMITIDO, max_salida_tokens=1024, pensamiento="apagado",
                         jaxqwen_socket=path, jaxqwen_uid=os.getuid(), jaxqwen_gid=os.getgid())
            cfg.raiz.mkdir()
            latir(cfg.latido)
            server = await arrancar(cfg)
            try:
                transport = httpx.AsyncHTTPTransport(uds=str(path))
                async with httpx.AsyncClient(transport=transport, base_url="http://jaxqwen") as client:
                    await client.post("/v1/messages", content=cuerpo)
                    rechazado = await client.post("/v1/messages", content=ambiguo)
                por_tcp = server.sockets[0].getsockname()[1]
                async with httpx.AsyncClient() as client:
                    await client.post(f"http://127.0.0.1:{por_tcp}/v1/messages", content=cuerpo)
                return rechazado.status_code, [r[3] for r in upstream.recibidas]
            finally:
                server.close()
                await server.wait_closed()

    rechazado, recibidas = _correr(scenario())
    assert rechazado == 403
    assert len(recibidas) == 2
    assert recibidas[0] == cuerpo, "por jaxqwen el cuerpo llega byte a byte"
    assert json.loads(recibidas[1])["thinking"] == {"type": "disabled"}, "por la entrada del Ejecutor se impone"


def test_por_jaxqwen_tambien_se_rechaza_la_herramienta_de_servidor_y_lo_anidado_ambiguo(tmp_path):
    """MAJOR-A y la herramienta de servidor aplican a las DOS entradas (no solo a la del Ejecutor)."""
    servidor = (b'{"model":"modelo-permitido","max_tokens":1,"messages":[],'
                b'"tools":[{"type":"web_search_20250305","name":"web_search"}]}')
    bloque = (b'{"model":"modelo-permitido","max_tokens":1,"messages":[{"role":"assistant","content":['
              b'{"type":"web_search_tool_result","tool_use_id":"s","content":[]}]}]}')
    anidado = (b'{"model":"modelo-permitido","max_tokens":1,"messages":['
               b'{"role":"user","content":[{"Type":"tool_result","tool_use_id":"t","content":"x"}]}]}')

    async def scenario():
        async with Upstream(modo="error") as upstream:
            parent = tmp_path / "run"
            parent.mkdir(mode=0o700)
            path = parent / "jaxqwen.sock"
            cfg = Config(upstream=upstream.url, raiz=tmp_path / "locks", tope_s=2,
                         host="127.0.0.1", puerto=0, registro=tmp_path / "registro.jsonl",
                         pausa=tmp_path / "PAUSA", latido=tmp_path / "latido", latido_max_s=60,
                         modelo=MODELO_PERMITIDO, max_salida_tokens=1024, pensamiento="libre",
                         jaxqwen_socket=path, jaxqwen_uid=os.getuid(), jaxqwen_gid=os.getgid())
            cfg.raiz.mkdir()
            latir(cfg.latido)
            server = await arrancar(cfg)
            try:
                transport = httpx.AsyncHTTPTransport(uds=str(path))
                async with httpx.AsyncClient(transport=transport, base_url="http://jaxqwen") as client:
                    a = await client.post("/v1/messages", content=servidor)
                    b = await client.post("/v1/messages", content=anidado)
                    c = await client.post("/v1/messages", content=bloque)
                return (a.status_code, a.json()["error"]["type"], b.status_code, b.json()["error"]["type"],
                        c.status_code, c.json()["error"]["type"], len(upstream.recibidas))
            finally:
                server.close()
                await server.wait_closed()

    assert _correr(scenario()) == (403, "herramienta_de_servidor", 403, "pedido_ambiguo",
                                   403, "herramienta_de_servidor", 0)
