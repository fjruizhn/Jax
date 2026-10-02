"""El Faro, paso 0.2: extremo a extremo. Un cliente MCP REAL (SDK oficial, `mcp.Client`) lanza
el rele `python -m jax.faro.relay` como subproceso stdio; el rele habla con el socket Unix del
Puerto; el Puerto sirve el paquete verificado. Es el camino que recorrera un motor dentro de
su jaula: sin atajos en proceso.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

import pytest

from jax.faro import paquete
from jax.faro.bitacora import Bitacora
from jax.faro.config import ConfigFaro, ConfigPuerto
from jax.faro.identidad import Ejecucion
from jax.faro.paquete import cargar_paquete
from jax.faro.transporte import ServidorPuerto
from tests._faro_utils import RAIZ, _git, cliente_por_rele, repo_de_juguete, servidor


@pytest.fixture
def montaje(tmp_path):
    repo = repo_de_juguete(tmp_path)
    cfg = ConfigFaro(repo=repo, sha=_git(repo, "rev-parse", "HEAD"), destino=tmp_path / "eco", uid_duenio=os.getuid())
    paquete.construir_paquete(cfg)
    d = tmp_path / "run"
    d.mkdir(mode=0o700)
    ej = Ejecucion(run_id="e2e-1", usuario="u", tenant="t", faceta="hyde", motor="qwen", pipeline="p",
                   entry_point="repl", id_correlacion="c", uid_esperado=os.getuid())
    return cfg, ConfigPuerto(socket_dir=d), ej


@pytest.mark.parametrize("modo", ["auto", "legacy"])
def test_un_cliente_mcp_real_lista_y_lee_una_skill_a_traves_del_rele(montaje, modo):
    cfg, cfgp, ej = montaje

    async def caso():
        async with servidor(cfgp, ej, cargar_paquete(cfg), Bitacora(emisores=[])) as srv:
            async with cliente_por_rele(srv, mode=modo) as c:
                recursos = await c.list_resources()
                uris = {str(r.uri) for r in recursos.resources}
                assert "ecosistema://constitucion" in uris
                plantillas = await c.list_resource_templates()
                assert any("skill://" in t.uri_template for t in plantillas.resource_templates)
                tools = {t.name for t in (await c.list_tools()).tools}
                assert {"skills.buscar", "skills.leer", "agentes.listar"} <= tools
                prompts = {p.name for p in (await c.list_prompts()).prompts}
                assert "skills.leer" in prompts
                # listar: las skills del paquete
                r = await c.call_tool("skills.buscar", {"consulta": ""})
                assert {"alfa", "beta"} <= {s["nombre"] for s in r.structured_content["result"]}
                # leer: por tool y por resource, mismos bytes
                por_tool = await c.call_tool("skills.leer", {"nombre": "alfa"})
                por_resource = await c.read_resource("skill://alfa")
                texto = por_resource.contents[0].text
                assert texto == (cfg.raiz_paquete / "skills/alfa/SKILL.md").read_text()
                assert texto in (por_tool.structured_content["result"], por_tool.content[0].text)
    asyncio.run(caso())


def test_el_rele_no_importa_el_sdk_de_mcp_ni_nada_pesado():
    """El rele viaja DENTRO de la jaula: solo biblioteca estandar."""
    codigo = ("import sys, jax.faro.relay; "
              "malos=[m for m in sys.modules if m.split('.')[0] in ('mcp','mcp_types','pydantic','starlette','anyio','httpx2')]; "
              "sys.exit(1 if malos else 0)")
    r = subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ, env={**os.environ, "PYTHONPATH": str(RAIZ)}, capture_output=True)
    assert r.returncode == 0, r.stderr.decode()


def test_el_rele_sin_socket_falla_rapido_y_no_ensucia_stdout(tmp_path):
    (tmp_path / "t").write_text("token\n")
    r = subprocess.run([sys.executable, "-m", "jax.faro.relay", "--socket", str(tmp_path / "no-hay.sock"), "--token-file", str(tmp_path / "t")],
                       cwd=RAIZ, env={**os.environ, "PYTHONPATH": str(RAIZ)}, capture_output=True, timeout=20, input=b"")
    assert r.returncode != 0 and r.stdout == b"" and b"no-hay.sock" in r.stderr


def test_el_rele_exige_la_ruta_del_socket(tmp_path):
    r = subprocess.run([sys.executable, "-m", "jax.faro.relay"], cwd=RAIZ, env={**os.environ, "PYTHONPATH": str(RAIZ)},
                       capture_output=True, timeout=20, input=b"")
    assert r.returncode != 0 and r.stdout == b""


def test_el_rele_termina_cuando_el_puerto_cierra(montaje):
    cfg, cfgp, ej = montaje

    async def caso():
        async with servidor(cfgp, ej, cargar_paquete(cfg), Bitacora(emisores=[])) as srv:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "jax.faro.relay", "--socket", str(srv.ruta_socket), "--token-file", str(srv.ruta_token), cwd=str(RAIZ),
                env={**os.environ, "PYTHONPATH": str(RAIZ)}, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode() + b"\n")
            await proc.stdin.drain()
            linea = await asyncio.wait_for(proc.stdout.readline(), 15)
            assert json.loads(linea)["id"] == 1
        # el Puerto cerro: el rele ve el cierre del socket y sale solo
        assert await asyncio.wait_for(proc.wait(), 15) == 0
    asyncio.run(caso())


def test_el_rele_exige_el_archivo_del_token_y_no_acepta_el_token_por_argv(montaje):
    cfg, cfgp, ej = montaje
    r = subprocess.run([sys.executable, "-m", "jax.faro.relay", "--socket", "/tmp/x.sock"], cwd=RAIZ,
                       env={**os.environ, "PYTHONPATH": str(RAIZ)}, capture_output=True, timeout=20, input=b"")
    assert r.returncode != 0 and r.stdout == b""
    r = subprocess.run([sys.executable, "-m", "jax.faro.relay", "--socket", "/tmp/x.sock", "--token", "abc"], cwd=RAIZ,
                       env={**os.environ, "PYTHONPATH": str(RAIZ)}, capture_output=True, timeout=20, input=b"")
    assert r.returncode != 0 and b"--token" in r.stderr  # no existe la opcion: el token solo viaja por archivo


def test_el_rele_con_un_archivo_de_token_ilegible_falla_sin_ensuciar_stdout(tmp_path):
    r = subprocess.run([sys.executable, "-m", "jax.faro.relay", "--socket", str(tmp_path / "s.sock"),
                        "--token-file", str(tmp_path / "no-existe")], cwd=RAIZ,
                       env={**os.environ, "PYTHONPATH": str(RAIZ)}, capture_output=True, timeout=20, input=b"")
    assert r.returncode == 2 and r.stdout == b"" and b"token" in r.stderr


def test_el_rele_con_half_close_deja_llegar_la_respuesta_antes_de_salir(montaje):
    """El cliente cierra su stdin tras enviar el pedido: el rele cierra solo el lado de escritura del socket
    (write_eof) y sigue copiando hasta que el Puerto responde y cierra."""
    cfg, cfgp, ej = montaje

    async def caso():
        async with servidor(cfgp, ej, cargar_paquete(cfg), Bitacora(emisores=[])) as srv:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "jax.faro.relay", "--socket", str(srv.ruta_socket), "--token-file", str(srv.ruta_token),
                cwd=str(RAIZ), env={**os.environ, "PYTHONPATH": str(RAIZ)},
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"}).encode() + b"\n")
            await proc.stdin.drain()
            proc.stdin.close()
            salida = await asyncio.wait_for(proc.stdout.read(), 20)
            assert json.loads(salida.splitlines()[0])["id"] == 7
            assert await asyncio.wait_for(proc.wait(), 20) == 0
    asyncio.run(caso())
