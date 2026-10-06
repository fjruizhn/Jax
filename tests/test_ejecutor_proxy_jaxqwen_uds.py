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
from jax.ejecutor.contratos.c3_control import registrar_auditor_c5
from jax.ejecutor.contratos.registro import verificar_cadena


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


def test_c5_selection_uses_the_proxy_as_the_only_c3_writer(tmp_path):
    async def scenario():
        async with Upstream(modo="error") as upstream:
            parent = tmp_path / "run"
            parent.mkdir(mode=0o700)
            path = parent / "c5.sock"
            cfg = Config(upstream=upstream.url, raiz=tmp_path / "locks", tope_s=2,
                         host="127.0.0.1", puerto=0, registro=tmp_path / "registro.jsonl",
                         pausa=tmp_path / "PAUSA", latido=tmp_path / "latido", latido_max_s=60,
                         modelo=MODELO_PERMITIDO, max_salida_tokens=1024, pensamiento="libre",
                         c5_control_socket=path, c5_control_uid=os.getuid(), c5_control_gid=os.getgid())
            cfg.raiz.mkdir()
            latir(cfg.latido)
            server = await arrancar(cfg)
            try:
                inválido = {"evento": "c5_auditor_elegido", "mision_id": "mision-invalida", "faceta": "thot",
                            "proveedor_id": "openai", "local": True, "modo": "SOLO_ORDENES"}
                reader, writer = await asyncio.open_unix_connection(str(path))
                cuerpo = json.dumps(inválido).encode()
                writer.write(struct.pack("!I", len(cuerpo)) + cuerpo)
                await writer.drain()
                assert await reader.readline() == b"ERROR\n"
                writer.close()
                await writer.wait_closed()
                await registrar_auditor_c5(mision_id="mision-1", faceta="thot", proveedor_id="openai",
                                           local=False, modo="SOLO_ORDENES",
                                           env={"JAX_PROXY_CARRIL_C5_SOCKET": str(path)})
                # Mismo evento: OK idempotente, sin entrada C3 duplicada.
                await registrar_auditor_c5(mision_id="mision-1", faceta="thot", proveedor_id="openai",
                                           local=False, modo="SOLO_ORDENES",
                                           env={"JAX_PROXY_CARRIL_C5_SOCKET": str(path)})
                # Reutilizar el ID con otra selección debe fallar cerrado.
                with pytest.raises(ValueError, match="registro_c3_c5_rechazado"):
                    await registrar_auditor_c5(mision_id="mision-1", faceta="ada", proveedor_id="openai",
                                               local=False, modo="SOLO_ORDENES",
                                               env={"JAX_PROXY_CARRIL_C5_SOCKET": str(path)})
                return verificar_cadena(cfg.registro), [json.loads(line) for line in cfg.registro.read_text().splitlines()]
            finally:
                await server.apagar()

    verificacion, eventos = asyncio.run(scenario())
    assert verificacion.ok
    evento = eventos[-1]
    assert {k: evento[k] for k in ("evento", "mision_id", "faceta", "proveedor_id", "local", "modo")} == {
        "evento": "c5_auditor_elegido", "mision_id": "mision-1", "faceta": "thot",
        "proveedor_id": "openai", "local": False, "modo": "SOLO_ORDENES"}
    assert evento["n"] == 2


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


def test_c5_control_socket_config_is_all_or_nothing_and_absolute():
    base = {
        "JAX_PROXY_CARRIL_UPSTREAM": "http://127.0.0.1:9", "JAX_PROXY_CARRIL_RAIZ": "/tmp/locks",
        "JAX_PROXY_CARRIL_TOPE_S": "5", "JAX_PROXY_CARRIL_PUERTO": "0",
        "JAX_EJECUTOR_REGISTRO": "/tmp/log", "JAX_EJECUTOR_PAUSA": "/tmp/pause",
        "JAX_EJECUTOR_VIGIA_LATIDO": "/tmp/beat", "JAX_EJECUTOR_VIGIA_LATIDO_MAX_S": "30",
        "JAX_PROXY_CARRIL_MODELO": "qwen-fixed", "JAX_PROXY_CARRIL_PENSAMIENTO": "libre",
        "JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS": "1024",
    }
    assert config_desde_entorno(base).c5_control_socket is None
    with pytest.raises(ConfigInvalida):
        config_desde_entorno({**base, "JAX_PROXY_CARRIL_C5_SOCKET": "/tmp/c5.sock"})
    complete = {**base, "JAX_PROXY_CARRIL_C5_SOCKET": "/tmp/c5.sock",
                "JAX_PROXY_CARRIL_C5_UID": "1001", "JAX_PROXY_CARRIL_C5_GID": "1002"}
    assert config_desde_entorno(complete).c5_control_uid == 1001
    with pytest.raises(ConfigInvalida):
        config_desde_entorno({**complete, "JAX_PROXY_CARRIL_C5_SOCKET": "relative.sock"})


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


# --------------------------------------------------------------------------
# Socket huérfano (2026-10-04): un stop de systemd no debe dejar el socket, y si
# queda uno propio sin oyente (SIGKILL, apagón) el arranque lo limpia. Todo en el
# tmp_path de pytest; ningún socket real de /run.
# --------------------------------------------------------------------------
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

from jax.ejecutor import proxy_carril

_RAIZ_REPO = Path(__file__).resolve().parents[1]


def _cfg_socket(tmp_path, path):
    cfg = Config(upstream="http://127.0.0.1:9", raiz=tmp_path / "locks", tope_s=1,
                 host="127.0.0.1", puerto=0, registro=tmp_path / "registro.jsonl",
                 pausa=tmp_path / "PAUSA", latido=tmp_path / "latido", latido_max_s=60,
                 modelo=MODELO_PERMITIDO, max_salida_tokens=1024, pensamiento="libre",
                 jaxqwen_socket=path, jaxqwen_uid=os.getuid(), jaxqwen_gid=os.getgid())
    cfg.raiz.mkdir(exist_ok=True)
    latir(cfg.latido)
    return cfg


def _dir_socket(tmp_path):
    parent = tmp_path / "run"
    parent.mkdir(mode=0o700)
    return parent / "jaxqwen.sock"


def _socket_huerfano(path):
    """Un socket Unix en disco SIN oyente: bind + listen + close deja el archivo."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(path))
    s.listen(1)
    s.close()
    assert path.is_socket()


def _socket_con_oyente(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(path))
    s.listen(1)
    return s


def _puerto_libre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _lanzar_proxy(tmp_path, path, puerto=0):
    """Proceso REAL (`python -m jax.ejecutor.proxy_carril`): lo que lanza systemd. Espera al socket."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(_RAIZ_REPO),
        "JAX_KILL_SWITCH_PATH": str(tmp_path / "KILL"),
        "JAX_PROXY_CARRIL_UPSTREAM": "http://127.0.0.1:9", "JAX_PROXY_CARRIL_RAIZ": str(tmp_path / "locks"),
        "JAX_PROXY_CARRIL_TOPE_S": "1", "JAX_PROXY_CARRIL_PUERTO": str(puerto),
        "JAX_EJECUTOR_REGISTRO": str(tmp_path / "registro.jsonl"),
        "JAX_EJECUTOR_PAUSA": str(tmp_path / "PAUSA"), "JAX_EJECUTOR_VIGIA_LATIDO": str(tmp_path / "latido"),
        "JAX_EJECUTOR_VIGIA_LATIDO_MAX_S": "60", "JAX_PROXY_CARRIL_MODELO": MODELO_PERMITIDO,
        "JAX_PROXY_CARRIL_PENSAMIENTO": "libre", "JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS": "1024",
        "JAX_PROXY_CARRIL_JAXQWEN_SOCKET": str(path),
        "JAX_PROXY_CARRIL_JAXQWEN_UID": str(os.getuid()), "JAX_PROXY_CARRIL_JAXQWEN_GID": str(os.getgid()),
    }
    (tmp_path / "locks").mkdir(exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "jax.ejecutor.proxy_carril"], cwd=_RAIZ_REPO, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    limite = time.monotonic() + 30
    while not path.exists():
        assert proc.poll() is None, proc.stderr.read().decode()
        assert time.monotonic() < limite, "el proxy no llegó a crear el socket"
        time.sleep(0.05)
    time.sleep(0.3)  # que los manejadores de señal estén instalados
    return proc


def _cabeceras_a_medias(puerto):
    c = socket.create_connection(("127.0.0.1", puerto), timeout=5)
    c.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Le")  # nunca termina
    time.sleep(0.2)
    return c


def _matar(proc):
    if proc.poll() is None:
        proc.kill()
        proc.wait()


@pytest.mark.parametrize("senal", [signal.SIGTERM, signal.SIGINT])
def test_parada_ordenada_por_senal_no_deja_socket_y_sale_con_cero(tmp_path, senal):
    path = _dir_socket(tmp_path)
    proc = _lanzar_proxy(tmp_path, path)
    try:
        proc.send_signal(senal)
        codigo = proc.wait(timeout=20)
    finally:
        _matar(proc)
    assert codigo == 0, proc.stderr.read().decode()
    assert not path.exists(), "un stop ordenado no debe dejar el socket"


def test_con_una_conexion_tcp_colgada_el_socket_unix_desaparece_al_instante(tmp_path):
    """MAJOR-1: antes, tras el SIGTERM el socket Unix seguía aceptando mientras se vaciaba el TCP."""
    path = _dir_socket(tmp_path)
    puerto = _puerto_libre()
    proc = _lanzar_proxy(tmp_path, path, puerto)
    colgada = _cabeceras_a_medias(puerto)
    try:
        t0 = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        while path.exists() and time.monotonic() - t0 < 5:
            time.sleep(0.02)
        assert time.monotonic() - t0 <= 0.5, "el socket tardó más de 0,5 s en desaparecer"
        assert proc.poll() is None, "el proceso debía seguir vivo, vaciando la conexión colgada"
        cliente = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            with pytest.raises((FileNotFoundError, ConnectionRefusedError)): cliente.connect(str(path))
        finally:
            cliente.close()
    finally:
        colgada.close()
        _matar(proc)


def test_con_una_conexion_colgada_sale_con_cero_tras_el_tope_y_corta_al_cliente(tmp_path):
    path = _dir_socket(tmp_path)
    puerto = _puerto_libre()
    proc = _lanzar_proxy(tmp_path, path, puerto)
    colgada = _cabeceras_a_medias(puerto)
    colgada.settimeout(0.1)
    try:
        t0 = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        codigo = proc.wait(timeout=proxy_carril.TOPE_VACIADO_S + 5)
        duracion = time.monotonic() - t0
        assert codigo == 0, proc.stderr.read().decode()
        assert duracion < proxy_carril.TOPE_VACIADO_S + 4, f"tardó {duracion:.1f} s: la colgada lo retuvo"
        colgada.settimeout(2)
        try:
            assert colgada.recv(1) == b"", "el cliente colgado debía ver el cierre"
        except ConnectionResetError:  # fail-soft: abort() cierra con RST; un reset ES el corte que la prueba espera
            pass
    finally:
        colgada.close()
        _matar(proc)


def test_socket_huerfano_propio_sin_oyente_se_limpia_y_el_proxy_arranca(tmp_path, caplog):
    async def scenario():
        path = _dir_socket(tmp_path)
        _socket_huerfano(path)
        viejo = os.lstat(path)
        server = await arrancar(_cfg_socket(tmp_path, path))
        try:
            nuevo = os.lstat(path)
            assert stat.S_ISSOCK(nuevo.st_mode)
            assert (nuevo.st_ino, nuevo.st_ctime_ns) != (viejo.st_ino, viejo.st_ctime_ns)
            transport = httpx.AsyncHTTPTransport(uds=str(path))
            async with httpx.AsyncClient(transport=transport, base_url="http://jaxqwen") as client:
                assert (await client.post("/v1/messages", content=b'{"model":"otro","max_tokens":1}')).status_code == 403
        finally:
            server.close()
            await server.wait_closed()
        assert not path.exists()
    with caplog.at_level("WARNING", logger=proxy_carril.log.name):
        _correr(scenario())
    assert any("socket_huerfano_limpiado" in r.getMessage() and "jaxqwen.sock" in r.getMessage()
               for r in caplog.records)


def test_socket_con_oyente_no_se_borra_y_el_arranque_falla_cerrado(tmp_path):
    async def scenario():
        path = _dir_socket(tmp_path)
        vivo = _socket_con_oyente(path)
        try:
            ino = os.lstat(path).st_ino
            with pytest.raises(ConfigInvalida): await arrancar(_cfg_socket(tmp_path, path))
            assert path.is_socket() and os.lstat(path).st_ino == ino
            cliente = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            cliente.settimeout(2)
            cliente.connect(str(path))  # el oyente sigue ahí
            cliente.close()
        finally:
            vivo.close()
    _correr(scenario())


def test_enlace_simbolico_no_se_borra_ni_siquiera_apuntando_a_un_socket_huerfano(tmp_path):
    async def scenario():
        path = _dir_socket(tmp_path)
        real = path.parent / "real.sock"
        _socket_huerfano(real)
        path.symlink_to(real)
        with pytest.raises(ConfigInvalida): await arrancar(_cfg_socket(tmp_path, path))
        assert path.is_symlink() and real.is_socket()
    _correr(scenario())


def test_enlace_roto_ni_archivo_regular_ni_directorio_se_borran(tmp_path):
    async def scenario():
        path = _dir_socket(tmp_path)
        path.symlink_to(path.parent / "no-existe")
        with pytest.raises(ConfigInvalida): await arrancar(_cfg_socket(tmp_path, path))
        assert path.is_symlink()
        path.unlink()
        path.write_text("archivo ajeno", encoding="utf-8")
        with pytest.raises(ConfigInvalida): await arrancar(_cfg_socket(tmp_path, path))
        assert path.read_text(encoding="utf-8") == "archivo ajeno"
        path.unlink()
        path.mkdir()
        with pytest.raises(ConfigInvalida): await arrancar(_cfg_socket(tmp_path, path))
        assert path.is_dir()
    _correr(scenario())


def test_socket_de_otro_uid_no_se_borra(tmp_path, monkeypatch):
    """No se puede crear un socket de otro uid sin root: se simula que NOSOTROS somos otro uid.
    Se llama a la limpieza directamente para aislar este chequeo del del dueño del padre."""
    path = _dir_socket(tmp_path)
    _socket_huerfano(path)
    monkeypatch.setattr(proxy_carril.os, "geteuid", lambda: os.getuid() + 1)
    with pytest.raises(ConfigInvalida): proxy_carril._despejar_socket_huerfano(path)
    assert path.is_socket()


def test_padre_de_otro_dueno_se_rechaza_aunque_el_modo_sea_correcto(tmp_path, monkeypatch):
    """MINOR-3: el padre debe ser del euid o de root, no solo tener buen modo."""
    parent = tmp_path / "run"
    parent.mkdir(mode=0o700)
    assert proxy_carril._padre_valido(parent)
    monkeypatch.setattr(proxy_carril.os, "geteuid", lambda: os.getuid() + 1)  # el padre pasa a ser "de otro"
    assert not proxy_carril._padre_valido(parent)
    (tmp_path / "enlace").symlink_to(parent)
    assert not proxy_carril._padre_valido(tmp_path / "enlace")


def test_padre_con_escritura_de_grupo_sigue_rechazando_aunque_el_socket_sea_huerfano(tmp_path):
    async def scenario():
        path = _dir_socket(tmp_path)
        _socket_huerfano(path)
        path.parent.chmod(0o770)
        with pytest.raises(ConfigInvalida): await arrancar(_cfg_socket(tmp_path, path))
        assert path.is_socket()
    _correr(scenario())


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignora los permisos del archivo")
def test_connect_con_error_distinto_de_econnrefused_no_borra(tmp_path):
    """EACCES (socket propio sin permiso de escritura) NO es 'huérfano': solo ECONNREFUSED lo es."""
    async def scenario():
        path = _dir_socket(tmp_path)
        _socket_huerfano(path)
        path.chmod(0o000)
        with pytest.raises(PermissionError): proxy_carril._sondear_oyente(path)
        with pytest.raises(ConfigInvalida): await arrancar(_cfg_socket(tmp_path, path))
        assert path.is_socket()
    _correr(scenario())


def test_sondeo_distingue_huerfano_oyente_y_ausente(tmp_path):
    path = _dir_socket(tmp_path)
    with pytest.raises(FileNotFoundError): proxy_carril._sondear_oyente(path)
    _socket_huerfano(path)
    assert proxy_carril._sondear_oyente(path) is None
    path.unlink()
    vivo = _socket_con_oyente(path)
    try:
        with pytest.raises(OSError): proxy_carril._sondear_oyente(path)
    finally:
        vivo.close()


def test_carrera_socket_reemplazado_entre_el_connect_y_el_unlink_no_borra_el_nuevo(tmp_path, monkeypatch):
    """Otro proceso empieza a escuchar justo después de que el connect() de prueba vio ECONNREFUSED:
    el socket nuevo (otro inodo) NO se borra."""
    oyentes = []

    async def scenario():
        path = _dir_socket(tmp_path)
        _socket_huerfano(path)
        real = proxy_carril._sondear_oyente

        def sondeo_con_reemplazo(ruta):
            real(ruta)  # ve ECONNREFUSED: el viejo está huérfano...
            os.unlink(ruta)  # ...y antes del unlink del proxy alguien pone uno nuevo y escucha
            oyentes.append(_socket_con_oyente(ruta))
        monkeypatch.setattr(proxy_carril, "_sondear_oyente", sondeo_con_reemplazo)
        with pytest.raises(ConfigInvalida): await arrancar(_cfg_socket(tmp_path, path))
        assert path.is_socket(), "el socket nuevo, con oyente, fue borrado"
        cliente = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        cliente.settimeout(2)
        cliente.connect(str(path))
        cliente.close()
    try:
        _correr(scenario())
    finally:
        for s in oyentes: s.close()


def test_apagar_deja_de_escuchar_de_inmediato_y_aborta_lo_colgado_al_agotar_el_tope(tmp_path):
    async def scenario():
        path = _dir_socket(tmp_path)
        server = await arrancar(_cfg_socket(tmp_path, path))
        puerto = server.sockets[0].getsockname()[1]
        lector, escritor = await asyncio.open_connection("127.0.0.1", puerto)
        escritor.write(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Le")
        await escritor.drain()
        await asyncio.sleep(0.1)
        tarea = asyncio.ensure_future(server.apagar(tope_s=0.5))
        await asyncio.sleep(0.1)
        assert not tarea.done(), "debía seguir vaciando la conexión colgada"
        assert not path.exists(), "el socket Unix debe irse al dejar de escuchar, no al terminar el vaciado"
        with pytest.raises(OSError): await asyncio.open_connection("127.0.0.1", puerto)
        await tarea
        try:
            assert await asyncio.wait_for(lector.read(), 2) == b""
        except ConnectionResetError:  # fail-soft: abort() cierra con RST; un reset ES el corte que la prueba espera
            pass
        escritor.close()
        with pytest.raises(OSError, match="registro_cerrado"):
            server._proxy.registro.anotar({"evento": "tarde"})
    _correr(scenario())


def test_close_no_borra_un_socket_que_fue_reemplazado(tmp_path):
    """MINOR-2: se borra el inodo que creamos, no lo que haya en la ruta."""
    oyentes = []

    async def scenario():
        path = _dir_socket(tmp_path)
        server = await arrancar(_cfg_socket(tmp_path, path))
        try:
            os.unlink(path)  # alguien lo reemplaza por otro, con oyente
            oyentes.append(_socket_con_oyente(path))
            server.close()
            assert path.is_socket(), "close() borró un socket que no era el suyo"
        finally:
            await server.wait_closed()
    try:
        _correr(scenario())
    finally:
        for s in oyentes: s.close()


def test_si_arrancar_falla_tras_el_bind_no_borra_un_socket_reemplazado(tmp_path, monkeypatch):
    oyentes = []

    async def scenario():
        path = _dir_socket(tmp_path)
        def chown_que_reemplaza_y_falla(*_a, **_k):
            os.unlink(path)
            oyentes.append(_socket_con_oyente(path))
            raise OSError("chown fallido")
        monkeypatch.setattr(proxy_carril.os, "chown", chown_que_reemplaza_y_falla)
        with pytest.raises(OSError, match="chown fallido"): await arrancar(_cfg_socket(tmp_path, path))
        assert path.is_socket(), "la rama de error borró un socket que no era el suyo"
    try:
        _correr(scenario())
    finally:
        for s in oyentes: s.close()
