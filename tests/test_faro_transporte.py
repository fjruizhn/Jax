"""El Faro, paso 0.2 (auditoria): el transporte del Puerto. Quien puede hablar con un socket, que
un proceso no pueda agotar la memoria del servicio y que el servicio no corra con privilegios.

- MAJOR-2/3: usuario de servicio sin privilegios (nunca root), directorio de sockets 0700 del usuario del
  servicio, a la jaula se le entrega SOLO su socket (y su token) por bind de archivo; y, para que otra ejecucion
  del MISMO uid no pueda suplantar a la dueña, un token aleatorio de la ejecucion en el handshake.
- MAJOR-4: presupuesto global de bytes en vuelo (sin tope de conexiones, D-4), `max_mensaje` de ~1 MiB.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
from pathlib import Path

import pytest

from jax.faro import paquete
from jax.faro.bitacora import Bitacora
from jax.faro.config import ConfigFaro, ConfigFaroInvalida, ConfigPuerto
from jax.faro.paquete import cargar_paquete
from jax.faro.transporte import PresupuestoBytes, ServidorPuerto
from tests._faro_utils import _git, cliente_por_rele, corre, ejecucion, puerto, repo_de_juguete


@pytest.fixture
def cargado(tmp_path):
    repo = repo_de_juguete(tmp_path)
    cfg = ConfigFaro(repo=repo, sha=_git(repo, "rev-parse", "HEAD"), destino=tmp_path / "eco", uid_duenio=os.getuid())
    paquete.construir_paquete(cfg)
    return cargar_paquete(cfg)


@pytest.fixture
def cfgp(tmp_path):
    d = tmp_path / "run"
    d.mkdir(mode=0o700)
    return ConfigPuerto(socket_dir=d)


def _handshake(token: str) -> bytes:
    return b"FARO-TOKEN " + token.encode() + b"\n"


async def _abrir(srv, enviar: bytes | None):
    lector, escritor = await asyncio.open_unix_connection(str(srv.ruta_socket))
    if enviar is not None:
        escritor.write(enviar)
        await escritor.drain()
    return lector, escritor


PING = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode() + b"\n"


# --------------------------------------------------------------------------- #
# MAJOR-2/3: usuario sin privilegios, directorio 0700, token de la ejecucion  #
# --------------------------------------------------------------------------- #

def test_el_servicio_no_corre_como_root(cfgp, cargado, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0)

    async def caso():
        with pytest.raises(ConfigFaroInvalida, match="root"):
            async with puerto(cfgp, cargado):
                pass
    corre(caso())


def test_el_directorio_de_sockets_tiene_que_ser_del_usuario_del_servicio(cfgp, cargado, monkeypatch):
    real = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: real + 1)

    async def caso():
        with pytest.raises(ConfigFaroInvalida, match="usuario del servicio"):
            async with puerto(cfgp, cargado):
                pass
    corre(caso())


@pytest.mark.parametrize("modo", [0o750, 0o755, 0o701, 0o770, 0o777, 0o710])
def test_el_directorio_de_sockets_no_admite_ningun_permiso_de_grupo_ni_otros(cfgp, cargado, modo):
    cfgp.socket_dir.chmod(modo)

    async def caso():
        with pytest.raises(ConfigFaroInvalida, match="0700|privado|grupo"):
            async with puerto(cfgp, cargado):
                pass
    corre(caso())


def test_el_directorio_es_privado_y_la_jaula_recibe_solo_su_socket_y_su_token(cfgp, cargado):
    """Con el directorio en 0700 otro usuario no puede listarlo ni llegar a los sockets de otros runs;
    a la jaula se le monta por bind SOLO su `<run_id>.sock` y su `<run_id>.token` (ver el argv de bwrap en el plan)."""
    async def caso():
        async with puerto(cfgp, cargado, ej=ejecucion(run_id="run-a")) as a, puerto(cfgp, cargado, ej=ejecucion(run_id="run-b")) as b:
            assert stat.S_IMODE(os.lstat(cfgp.socket_dir).st_mode) == 0o700
            assert sorted(p.name for p in cfgp.socket_dir.iterdir()) == ["run-a.sock", "run-a.token", "run-b.sock", "run-b.token"]
            assert a.ruta_token != b.ruta_token and a.ruta_token.read_text() != b.ruta_token.read_text()
    corre(caso())


def test_el_token_es_aleatorio_largo_y_su_archivo_es_de_solo_lectura(cfgp, cargado):
    async def caso():
        async with puerto(cfgp, cargado) as srv:
            token = srv.ruta_token.read_text().strip()
            assert len(token) >= 32 and token.isascii()
            assert stat.S_IMODE(os.lstat(srv.ruta_token).st_mode) == 0o444
        assert not srv.ruta_token.exists()
    corre(caso())


def test_sin_el_token_correcto_no_se_habla_mcp(cfgp, cargado, monkeypatch):
    llamadas = []
    monkeypatch.setattr(paquete.PaqueteCargado, "buscar", lambda self, *a, **k: llamadas.append(1) or [])

    async def caso():
        async with puerto(cfgp, cargado) as srv:
            casos = {
                "mal": _handshake("x" * 43), "sin_handshake": PING, "vacio": b"\n",
                "token_de_otro": _handshake("A" * len(srv.ruta_token.read_text().strip())),
            }
            for nombre, datos in casos.items():
                lector, escritor = await _abrir(srv, datos + PING)
                assert await asyncio.wait_for(lector.read(), 5) == b"", nombre   # cerrada, sin responder MCP
                escritor.close()
            await asyncio.sleep(0.1)
        return srv.registros

    registros = corre(caso())
    rechazos = [r for r in registros if r.get("evento") == "conexion_rechazada"]
    assert len(rechazos) == 4 and {r["motivo"] for r in rechazos} == {"token_invalido"}
    assert not [r for r in registros if r.get("evento") == "llamada"] and llamadas == []


def test_el_token_de_otra_ejecucion_del_mismo_uid_no_suplanta(cfgp, cargado):
    async def caso():
        async with puerto(cfgp, cargado, ej=ejecucion(run_id="run-a")) as a, puerto(cfgp, cargado, ej=ejecucion(run_id="run-b")) as b:
            lector, escritor = await _abrir(a, _handshake(b.ruta_token.read_text().strip()) + PING)
            assert await asyncio.wait_for(lector.read(), 5) == b""
            escritor.close()
            # el correcto, en cambio, entra
            lector, escritor = await _abrir(a, _handshake(a.ruta_token.read_text().strip()) + PING)
            linea = await asyncio.wait_for(lector.readline(), 10)
            assert json.loads(linea)["id"] == 1
            escritor.close()
        return a.registros
    registros = corre(caso())
    assert [r["motivo"] for r in registros if r.get("evento") == "conexion_rechazada"] == ["token_invalido"]


def test_el_token_nunca_aparece_en_la_bitacora_ni_en_el_log(cfgp, cargado, caplog):
    import logging
    token = {}

    async def caso():
        async with ServidorPuerto(cfgp, ejecucion(), cargado, Bitacora()) as srv:
            token["t"] = srv.ruta_token.read_text().strip()
            async with cliente_por_rele(srv) as c:
                await c.call_tool("skills.leer", {"nombre": "alfa"})
            lector, escritor = await _abrir(srv, _handshake("mal"))
            await asyncio.wait_for(lector.read(), 5)
            escritor.close()
    with caplog.at_level(logging.DEBUG):
        corre(caso())
    assert token["t"] and not any(token["t"] in r.getMessage() for r in caplog.records)


def test_un_handshake_que_no_llega_se_corta_por_tiempo(tmp_path, cargado):
    d = tmp_path / "r"
    d.mkdir(mode=0o700)

    async def caso():
        async with puerto(ConfigPuerto(socket_dir=d, handshake_s=0.3), cargado) as srv:
            lector, escritor = await _abrir(srv, b"FARO-TOKEN ")   # incompleto y callado
            assert await asyncio.wait_for(lector.read(), 5) == b""
            escritor.close()
        return srv.registros
    assert [r["motivo"] for r in corre(caso()) if r.get("evento") == "conexion_rechazada"] == ["token_invalido"]


def test_con_el_token_correcto_por_el_rele_todo_funciona(cfgp, cargado):
    async def caso():
        async with puerto(cfgp, cargado) as srv, cliente_por_rele(srv) as c:
            r = await c.call_tool("skills.leer", {"nombre": "alfa"})
            assert not r.is_error
    corre(caso())


def test_un_token_equivocado_en_el_archivo_del_rele_no_conecta(cfgp, cargado, tmp_path):
    falso = tmp_path / "falso.token"
    falso.write_text("t" * 43 + "\n")

    async def caso():
        async with puerto(cfgp, cargado) as srv:
            with pytest.raises(Exception):
                async with cliente_por_rele(srv, token_file=falso) as c:
                    await c.list_tools()
    corre(caso())


# --------------------------------------------------------------------------- #
# MAJOR-4: memoria                                                            #
# --------------------------------------------------------------------------- #

def test_la_configuracion_por_defecto_acota_el_mensaje_a_un_mebibyte(tmp_path):
    c = ConfigPuerto(socket_dir=tmp_path)
    assert c.max_mensaje == 1024 * 1024 and c.presupuesto_bytes >= 4 * c.max_mensaje
    assert ConfigPuerto.desde_entorno({"JAX_FARO_SOCKET_DIR": str(tmp_path), "JAX_FARO_PRESUPUESTO_BYTES": "33554432"}).presupuesto_bytes == 33554432
    with pytest.raises(ConfigFaroInvalida):
        ConfigPuerto(socket_dir=tmp_path, max_mensaje=1024 * 1024, presupuesto_bytes=1024)   # no cabe ni un mensaje
    with pytest.raises(ConfigFaroInvalida):
        ConfigPuerto.desde_entorno({"JAX_FARO_SOCKET_DIR": str(tmp_path), "JAX_FARO_PRESUPUESTO_BYTES": "mucho"})


def test_el_presupuesto_bloquea_cuando_no_alcanza_y_libera_en_orden():
    async def caso():
        p = PresupuestoBytes(100)
        await p.adquirir(60)
        esperando = asyncio.create_task(p.adquirir(60))
        await asyncio.sleep(0.05)
        assert not esperando.done() and p.libre == 40
        await p.liberar(60)
        await asyncio.wait_for(esperando, 1)
        assert p.libre == 40
        await p.liberar(60)
        assert p.libre == 100
    asyncio.run(caso())


def test_pedir_mas_que_el_total_se_acota_al_total_y_no_se_cuelga():
    async def caso():
        p = PresupuestoBytes(100)
        await asyncio.wait_for(p.adquirir(10_000), 1)
        assert p.libre == 0
        await p.liberar(10_000)
        assert p.libre == 100
    asyncio.run(caso())


def test_una_espera_cancelada_no_pierde_bytes():
    async def caso():
        p = PresupuestoBytes(100)
        await p.adquirir(100)
        t = asyncio.create_task(p.adquirir(50))
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        await p.liberar(100)
        assert p.libre == 100
    asyncio.run(caso())


def _rss_kib() -> int:
    for linea in Path("/proc/self/status").read_text().splitlines():
        if linea.startswith("VmRSS:"):
            return int(linea.split()[1])
    raise AssertionError("sin VmRSS")


def test_muchas_conexiones_con_mensajes_grandes_no_agotan_la_memoria(tmp_path, cargado):
    """120 conexiones autenticadas, cada una a medio mandar casi un mensaje maximo (sin salto de linea): lo
    que el servicio retiene queda acotado por el presupuesto, no por el numero de conexiones."""
    d = tmp_path / "r"
    d.mkdir(mode=0o700)
    cfg = ConfigPuerto(socket_dir=d, max_mensaje=1024 * 1024, presupuesto_bytes=16 * 1024 * 1024, mensaje_timeout_s=60)
    carga = b"x" * 900_000
    n = 120

    async def caso():
        async with puerto(cfg, cargado) as srv:
            bucle = asyncio.get_running_loop()
            token = _handshake(srv.ruta_token.read_text().strip())
            # calentamiento: una conexion normal para que la linea base ya incluya lo comun
            lector, escritor = await _abrir(srv, token + PING)
            await asyncio.wait_for(lector.readline(), 10)
            escritor.close()
            await asyncio.sleep(0.3)
            base = _rss_kib()
            socks = []
            for _ in range(n):
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.setblocking(False)
                await bucle.sock_connect(s, str(srv.ruta_socket))
                await bucle.sock_sendall(s, token)
                socks.append(s)
            envios = [asyncio.ensure_future(bucle.sock_sendall(s, carga)) for s in socks]
            pico = base
            for _ in range(25):
                await asyncio.sleep(0.1)
                pico = max(pico, _rss_kib())
            for e in envios:
                e.cancel()
            await asyncio.gather(*envios, return_exceptions=True)
            for s in socks:
                s.close()
            return (pico - base) / 1024   # MiB
    crecimiento = corre(caso())
    # sin presupuesto: ~120 * 0.9 MB * (copias) > 100 MiB; con presupuesto de 16 MiB queda muy por debajo
    assert crecimiento < 45, f"el RSS crecio {crecimiento:.1f} MiB"


def test_no_hay_tope_de_conexiones_cientos_de_conexiones_ociosas_se_sirven(tmp_path, cargado):
    d = tmp_path / "r"
    d.mkdir(mode=0o700)
    cfg = ConfigPuerto(socket_dir=d, presupuesto_bytes=8 * 1024 * 1024)   # presupuesto chico a proposito

    async def caso():
        async with puerto(cfg, cargado) as srv:
            token = _handshake(srv.ruta_token.read_text().strip())
            conexiones = [await _abrir(srv, token + PING) for _ in range(200)]
            for lector, _ in conexiones:
                assert json.loads(await asyncio.wait_for(lector.readline(), 20))["id"] == 1
            for _, escritor in conexiones:
                escritor.close()
    corre(caso())


def test_un_mensaje_a_medias_que_se_estanca_se_corta_y_libera_el_presupuesto(tmp_path, cargado):
    d = tmp_path / "r"
    d.mkdir(mode=0o700)
    cfg = ConfigPuerto(socket_dir=d, presupuesto_bytes=8 * 1024 * 1024, mensaje_timeout_s=0.3)

    async def caso():
        async with puerto(cfg, cargado) as srv:
            token = _handshake(srv.ruta_token.read_text().strip())
            lector, escritor = await _abrir(srv, token + b'{"jsonrpc": "2.0", "id": 1, "meth')   # y se queda callado
            assert await asyncio.wait_for(lector.read(), 5) == b""      # el Puerto lo corto
            escritor.close()
            assert srv.presupuesto.libre == cfg.presupuesto_bytes       # y no quedo nada retenido
    corre(caso())


def test_las_respuestas_grandes_tambien_pasan_por_el_presupuesto(tmp_path):
    (tmp_path / "g").mkdir()
    repo = repo_de_juguete(tmp_path / "g", {"common/skills/alfa/grande.md": "x" * 400_000})
    cfg_p = ConfigFaro(repo=repo, sha=_git(repo, "rev-parse", "HEAD"), destino=tmp_path / "eco", uid_duenio=os.getuid())
    paquete.construir_paquete(cfg_p)
    d = tmp_path / "r"
    d.mkdir(mode=0o700)

    async def caso():
        async with puerto(ConfigPuerto(socket_dir=d, presupuesto_bytes=8 * 1024 * 1024), cargar_paquete(cfg_p)) as srv:
            async with cliente_por_rele(srv) as c:
                r = await c.call_tool("skills.leer", {"nombre": "alfa", "archivo": "grande.md"})
                assert not r.is_error
            assert srv.presupuesto.libre == 8 * 1024 * 1024
    corre(caso())
