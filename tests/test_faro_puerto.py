"""El Faro, paso 0.2: el Puerto en modo lectura (spec §3 y §7 fase 0; plan
`2026-10-02-faro-fase-0.md`).

Todo se ejercita con un cliente MCP REAL (SDK oficial) hablando por el rele stdio con un
servidor que escucha en un socket Unix de verdad, dentro de un directorio temporal. Lo que
se defiende: el paquete que se sirve es el verificado y no otro (sin path traversal), la
identidad la fija el socket y no el cuerpo del pedido, con el freno puesto no se ejecuta
nada, y cada llamada deja su linea en la bitacora.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import stat
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp.shared.exceptions import MCPError

from jax.faro import paquete
from jax.faro.bitacora import Bitacora, _campo_log
from jax.faro.config import ConfigFaro, ConfigFaroInvalida, ConfigPuerto
from jax.faro.identidad import Ejecucion, Identidad
from jax.faro.puerto import Guardia, construir_servidor
from jax.faro.paquete import PaqueteCargado, cargar_paquete
from jax.faro.transporte import ServidorPuerto
from tests import _faro_utils as _u
from tests._faro_utils import _commit, _escribir, _git, cliente_por_rele, repo_de_juguete, servidor

SECRETO = "SECRETO-FUERA-DEL-PAQUETE-0451"


corre, _ejecucion, puerto = _u.corre, _u.ejecucion, _u.puerto


@pytest.fixture
def freno_propio(tmp_path, monkeypatch):
    """Un interruptor propio de la prueba (nunca el de produccion): se pone creando el archivo."""
    ruta = tmp_path / "freno" / "PAUSE"
    ruta.parent.mkdir()
    monkeypatch.setenv("JAX_KILL_SWITCH_PATH", str(ruta))
    return ruta


@pytest.fixture
def cfg(tmp_path):
    repo = repo_de_juguete(tmp_path, {"common/skills/alfa/referencia.md": "referencia larga de alfa\n"})
    return ConfigFaro(repo=repo, sha=_git(repo, "rev-parse", "HEAD"), destino=tmp_path / "ecosistema", uid_duenio=os.getuid())


@pytest.fixture
def cargado(cfg) -> PaqueteCargado:
    paquete.construir_paquete(cfg)
    return cargar_paquete(cfg)


@pytest.fixture
def cfg_puerto(tmp_path):
    d = tmp_path / "run"
    d.mkdir(mode=0o700)
    return ConfigPuerto(socket_dir=d)


@asynccontextmanager
async def puerto(cfg_puerto, cargado, ejecucion=None, registros=None, **kw):
    registros = registros if registros is not None else []
    bit = Bitacora(emisores=[registros.append])
    async with servidor(cfg_puerto, ejecucion or _ejecucion(), cargado, bit, **kw) as srv:
        srv.registros = registros
        yield srv


def _llamadas(registros):
    return [r for r in registros if r.get("evento") == "llamada"]


# --------------------------------------------------------------------------- #
# catalogo                                                                    #
# --------------------------------------------------------------------------- #

def test_expone_la_constitucion_sellada_como_resource(cfg_puerto, cargado, cfg):
    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            r = await c.read_resource("ecosistema://constitucion")
            texto = r.contents[0].text
            assert texto.startswith(f"<!-- claude-skills: SHA {cfg.sha} -->")
            assert "regla uno" in texto
    corre(caso())


def test_el_catalogo_es_solo_lectura_nada_lanza_agentes(cfg_puerto, cargado):
    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            nombres = {t.name for t in (await c.list_tools()).tools}
            assert nombres == {"skills.buscar", "skills.leer", "agentes.listar"}
            assert not any("lanzar" in n for n in nombres)
    corre(caso())


def test_agentes_listar_devuelve_solo_el_catalogo_sin_el_cuerpo(cfg_puerto, cargado):
    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            r = await c.call_tool("agentes.listar", {})
            assert not r.is_error
            texto = json.dumps(r.structured_content) + "".join(getattr(x, "text", "") for x in r.content)
            assert "explorador" in texto and "haiku" in texto and "Descubre cosas" in texto
            assert "CUERPO SECRETO DEL AGENTE" not in texto
    corre(caso())


def test_skills_buscar_encuentra_por_nombre_y_descripcion(cfg_puerto, cargado):
    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            r = await c.call_tool("skills.buscar", {"consulta": "inyeccion"})
            assert not r.is_error
            assert "beta" in json.dumps(r.structured_content) and "alfa" not in json.dumps(r.structured_content)
            r = await c.call_tool("skills.buscar", {"consulta": "no-existe-zzz"})
            assert not r.is_error and "beta" not in json.dumps(r.structured_content)
    corre(caso())


def test_skills_leer_devuelve_los_bytes_verificados_y_tambien_por_resource_y_prompt(cfg_puerto, cargado):
    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            r = await c.call_tool("skills.leer", {"nombre": "alfa"})
            assert not r.is_error and "cuerpo alfa" in json.dumps(r.structured_content) + str(r.content)
            r = await c.call_tool("skills.leer", {"nombre": "alfa", "archivo": "referencia.md"})
            assert "referencia larga de alfa" in json.dumps(r.structured_content) + str(r.content)
            rr = await c.read_resource("skill://alfa")
            assert "cuerpo alfa" in rr.contents[0].text
            p = await c.get_prompt("skills.leer", {"nombre": "beta"})
            assert "cuerpo beta" in str(p.messages)
    corre(caso())


# --------------------------------------------------------------------------- #
# lo que no esta en el paquete no existe (sin path traversal)                 #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("nombre", [
    "no-existe", "../constitucion/CLAUDE.md", "/etc/passwd", "alfa/../beta", "alfa/..", "..", ".", "",
    "alfa\0", "%2e%2e", "alfa/SKILL.md", "ALFA", " alfa", "alfa ", "skills/alfa", "\\..\\..\\etc\\passwd",
])
def test_una_skill_que_no_esta_en_el_paquete_no_existe(cfg_puerto, cargado, nombre):
    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            try:
                r = await c.call_tool("skills.leer", {"nombre": nombre})
            except MCPError:
                return  # un error de protocolo tambien es "no existe"
            assert r.is_error, f"{nombre!r} devolvio contenido: {r}"
            assert "cuerpo" not in str(r.content) and "root:" not in str(r.content)
    corre(caso())


@pytest.mark.parametrize("archivo", [
    "../beta/SKILL.md", "/etc/passwd", "scripts/../../beta/SKILL.md", "SKILL.md\0", "..", "", ".",
    "scripts/..", "./SKILL.md", "scripts//correr.sh", "no-existe.md",
])
def test_un_archivo_fuera_de_la_skill_no_existe(cfg_puerto, cargado, archivo):
    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            try:
                r = await c.call_tool("skills.leer", {"nombre": "alfa", "archivo": archivo})
            except MCPError:
                return
            assert r.is_error, f"{archivo!r} devolvio contenido: {r}"
    corre(caso())


@pytest.mark.parametrize("uri", [
    "skill://../../etc/passwd", "skill://%2e%2e%2fetc%2fpasswd", "skill:///etc/passwd", "skill://no-existe",
    "file:///etc/passwd", "ecosistema://constitucion/../../etc/passwd",
])
def test_un_resource_fuera_del_catalogo_no_existe(cfg_puerto, cargado, uri):
    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            with pytest.raises(MCPError):
                await c.read_resource(uri)
    corre(caso())


def test_un_symlink_plantado_despues_de_cargar_no_se_sirve(cfg_puerto, cargado, cfg, tmp_path):
    """El Puerto sirve los bytes que verifico al cargar (en memoria), no lo que haya en disco despues."""
    secreto = tmp_path / "secreto.txt"
    secreto.write_text(SECRETO)
    ruta = cfg.raiz_paquete / "skills/alfa/SKILL.md"
    ruta.unlink()
    ruta.symlink_to(secreto)

    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            r = await c.call_tool("skills.leer", {"nombre": "alfa"})
            vista = json.dumps(r.structured_content) + str(r.content)
            assert SECRETO not in vista and "cuerpo alfa" in vista
    corre(caso())


def test_un_symlink_presente_al_cargar_impide_arrancar(cfg):
    paquete.construir_paquete(cfg)
    (cfg.raiz_paquete / "skills/enlace").symlink_to("/etc")
    with pytest.raises(paquete.PaqueteNoVerifica):
        cargar_paquete(cfg)


def test_cargar_tambien_exige_integridad_byte_a_byte(cfg):
    paquete.construir_paquete(cfg)
    (cfg.raiz_paquete / "skills/beta/SKILL.md").write_text("alterada")
    with pytest.raises(paquete.PaqueteNoVerifica):
        cargar_paquete(cfg)


def test_el_puerto_no_arranca_con_un_paquete_que_no_verifica(cfg, cfg_puerto):
    paquete.construir_paquete(cfg)
    (cfg.raiz_paquete / "agentes/explorador.md").write_text("alterado")

    async def caso():
        with pytest.raises(paquete.PaqueteNoVerifica):
            cargado = cargar_paquete(cfg)
            async with puerto(cfg_puerto, cargado):
                pass
    corre(caso())
    assert list(cfg_puerto.socket_dir.iterdir()) == []


# --------------------------------------------------------------------------- #
# identidad: la fija el socket                                                #
# --------------------------------------------------------------------------- #

def test_una_identidad_en_el_cuerpo_del_pedido_se_ignora(cfg_puerto, cargado):
    falsa = {"usuario": "mallory", "user_id": "999", "tenant": "otro-tenant", "entry_point": "evil",
             "run_id": "run-ajeno", "faceta": "jacobs", "motor": "kimi", "id_correlacion": "corr-falsa"}

    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            for args in ({"consulta": "alfa", **falsa}, {"consulta": "alfa"}):
                with contextlib.suppress(MCPError):  # da igual si la valida o la rechaza: lo que se mira es la bitacora
                    await c.call_tool("skills.buscar", args, meta=falsa)
            with contextlib.suppress(MCPError):
                await c.read_resource("skill://alfa", meta={"identidad": falsa})
        return srv.registros

    registros = corre(caso())
    llamadas = _llamadas(registros)
    assert len(llamadas) >= 3
    for r in llamadas:
        assert (r["usuario"], r["tenant"], r["entry_point"], r["run_id"], r["faceta"], r["motor"], r["id_correlacion"]) == \
               ("u-real", "t-real", "repl", "run-1", "hyde", "codex", "corr-real")
    assert "mallory" not in json.dumps([{k: v for k, v in r.items() if k not in ("hash_args", "argumentos")} for r in registros])


def test_un_par_de_otro_uid_se_rechaza_sin_hablar_mcp(cfg_puerto, cargado, monkeypatch):
    llamadas = []
    original = PaqueteCargado.buscar
    monkeypatch.setattr(PaqueteCargado, "buscar", lambda self, *a, **k: llamadas.append(1) or original(self, *a, **k))

    async def caso():
        async with puerto(cfg_puerto, cargado, ejecucion=_ejecucion(uid_esperado=os.getuid() + 1)) as srv:
            with pytest.raises(Exception):
                async with cliente_por_rele(srv) as c:
                    await c.call_tool("skills.buscar", {"consulta": "alfa"})
            await asyncio.sleep(0.1)
        return srv.registros

    registros = corre(caso())
    assert llamadas == []
    rechazos = [r for r in registros if r.get("evento") == "conexion_rechazada"]
    assert len(rechazos) >= 1 and rechazos[0]["peer_uid"] == os.getuid() and rechazos[0]["run_id"] == "run-1"
    assert _llamadas(registros) == []


def test_cada_conexion_lleva_su_id_y_las_credenciales_del_par(cfg_puerto, cargado):
    async def caso():
        async with puerto(cfg_puerto, cargado) as srv:
            async with cliente_por_rele(srv) as a, cliente_por_rele(srv) as b:
                await asyncio.gather(a.read_resource("skill://alfa"), b.read_resource("skill://beta"))
        return srv.registros
    llamadas = [r for r in _llamadas(corre(caso())) if r["metodo"] == "resources/read"]
    assert len(llamadas) == 2 and len({r["id_conexion"] for r in llamadas}) == 2
    assert all(r["peer_uid"] == os.getuid() and isinstance(r["peer_pid"], int) for r in llamadas)


# --------------------------------------------------------------------------- #
# el freno                                                                    #
# --------------------------------------------------------------------------- #

def test_con_el_freno_puesto_nada_se_ejecuta(cfg_puerto, cargado, freno_propio, monkeypatch):
    ejecutadas = []
    original = PaqueteCargado.buscar
    monkeypatch.setattr(PaqueteCargado, "buscar", lambda self, *a, **k: ejecutadas.append(1) or original(self, *a, **k))

    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            assert not (await c.call_tool("skills.buscar", {"consulta": "alfa"})).is_error  # suelto: funciona
            assert len(ejecutadas) == 1
            freno_propio.write_text("pausa")
            for llamada in (
                lambda: c.call_tool("skills.buscar", {"consulta": "alfa"}),
                lambda: c.call_tool("skills.leer", {"nombre": "alfa"}),
                lambda: c.call_tool("agentes.listar", {}),
                lambda: c.read_resource("skill://alfa"),
                lambda: c.read_resource("ecosistema://constitucion"),
                lambda: c.get_prompt("skills.leer", {"nombre": "alfa"}),
                lambda: c.list_tools(),
            ):
                with pytest.raises(MCPError) as exc:
                    await llamada()
                assert exc.value.code == 423
            assert len(ejecutadas) == 1  # ni una sola mas
            freno_propio.unlink()
            assert not (await c.call_tool("skills.buscar", {"consulta": "alfa"})).is_error  # soltado: vuelve
        return srv.registros

    registros = corre(caso())
    denegadas = [r for r in _llamadas(registros) if r["decision"] == "denegado"]
    assert len(denegadas) == 7 and all(r["motivo"] == "freno" and r["hash_resultado"] == "" for r in denegadas)


def test_con_el_freno_puesto_antes_de_conectar_ni_el_handshake_se_atiende(cfg_puerto, cargado, freno_propio):
    freno_propio.write_text("pausa")

    async def caso():
        async with puerto(cfg_puerto, cargado) as srv:
            with pytest.raises(Exception):
                async with cliente_por_rele(srv) as c:
                    await c.list_tools()
        return srv.registros

    llamadas = _llamadas(corre(caso()))
    assert llamadas and all(r["decision"] == "denegado" for r in llamadas)


def test_sin_saber_donde_esta_el_freno_se_niega(cfg_puerto, cargado, monkeypatch):
    monkeypatch.delenv("JAX_KILL_SWITCH_PATH", raising=False)

    async def caso():
        async with puerto(cfg_puerto, cargado) as srv:
            with pytest.raises(Exception):
                async with cliente_por_rele(srv) as c:
                    await c.list_tools()
        return srv.registros

    llamadas = _llamadas(corre(caso()))
    assert llamadas and all(r["decision"] == "denegado" and r["motivo"] == "freno" for r in llamadas)


def test_un_freno_inyectado_que_falla_tambien_niega(cfg_puerto, cargado):
    def roto():
        raise OSError("no se puede mirar el freno")

    async def caso():
        async with puerto(cfg_puerto, cargado, freno=roto) as srv:
            with pytest.raises(Exception):
                async with cliente_por_rele(srv) as c:
                    await c.list_tools()
        return srv.registros

    assert all(r["decision"] == "denegado" for r in _llamadas(corre(caso())))


# --------------------------------------------------------------------------- #
# bitacora                                                                    #
# --------------------------------------------------------------------------- #

def test_cada_llamada_deja_su_registro_completo(cfg_puerto, cargado, cfg):
    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            await c.call_tool("skills.leer", {"nombre": "alfa"})
        return srv.registros

    registros = corre(caso())
    r = next(x for x in _llamadas(registros) if x["metodo"] == "tools/call")
    for campo in ("id_llamada", "id_correlacion", "id_conexion", "entry_point", "sha_paquete", "metodo", "objetivo",
                  "hash_args", "hash_resultado", "decision", "usuario", "tenant", "faceta", "motor", "pipeline", "run_id"):
        assert r[campo] not in (None, ""), campo
    assert r["objetivo"] == "skills.leer" and r["decision"] == "permitido" and r["sha_paquete"] == cfg.sha
    assert len(r["hash_args"]) == 64 and len(r["hash_resultado"]) == 64
    assert len({x["id_llamada"] for x in _llamadas(registros)}) == len(_llamadas(registros))  # uno por pedido


def test_el_mismo_pedido_da_el_mismo_hash_de_argumentos_y_otro_pedido_otro(cfg_puerto, cargado):
    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            await c.call_tool("skills.leer", {"nombre": "alfa"})
            await c.call_tool("skills.leer", {"nombre": "alfa"})
            await c.call_tool("skills.leer", {"nombre": "beta"})
        return srv.registros
    h = [r["hash_args"] for r in _llamadas(corre(caso())) if r["metodo"] == "tools/call"]
    assert h[0] == h[1] != h[2]


def test_un_error_de_la_herramienta_tambien_queda_registrado(cfg_puerto, cargado):
    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            with pytest.raises(MCPError):
                await c.read_resource("skill://no-existe")
        return srv.registros
    r = [x for x in _llamadas(corre(caso())) if x["metodo"] == "resources/read"][0]
    assert r["decision"] == "permitido" and r["resultado"] == "error"


async def _pedir_prompt_inexistente(cfg_puerto, cargado, nombre):
    """El nombre del prompt es lo que el cliente controla y queda como `objetivo` en la bitacora."""
    async with servidor(cfg_puerto, _ejecucion(), cargado, Bitacora()) as srv, cliente_por_rele(srv) as c:
        with pytest.raises(MCPError):
            await c.get_prompt(nombre, {})


def test_un_objetivo_con_saltos_de_linea_no_fabrica_una_segunda_linea_de_log(cfg_puerto, cargado, caplog):
    nombre = "x\nfaro_llamada decision=permitido usuario=root\rresultado=ok"
    with caplog.at_level(logging.INFO, logger="jax.faro.bitacora"):
        corre(_pedir_prompt_inexistente(cfg_puerto, cargado, nombre))
    lineas = [r.getMessage() for r in caplog.records if r.name == "jax.faro.bitacora"]
    assert any("prompts/get" in l for l in lineas)
    assert all("\n" not in l and "\r" not in l for l in lineas)
    assert not any(l.startswith("faro_llamada decision=permitido usuario=root") for l in lineas)


def test_un_valor_con_espacios_no_puede_fingir_un_campo_en_su_propia_linea(cfg_puerto, cargado, caplog):
    with caplog.at_level(logging.INFO, logger="jax.faro.bitacora"):
        corre(_pedir_prompt_inexistente(cfg_puerto, cargado, "x usuario=root decision=denegado"))
    linea = next(r.getMessage() for r in caplog.records if r.name == "jax.faro.bitacora" and "prompts/get" in r.getMessage())
    pares = shlex.split(linea)
    assert [p for p in pares if p.startswith("usuario=")] == ["usuario=u-real"]
    assert [p for p in pares if p.startswith("decision=")] == ["decision=permitido"]


def test_un_pedido_grande_llega_al_servidor_y_se_rechaza_por_su_contenido(cfg_puerto, cargado):
    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            r = await c.call_tool("skills.leer", {"nombre": "n" * 100_000})
            assert r.is_error and "excede" in str(r.content)
    corre(caso())


def test_un_pedido_mas_largo_que_el_limite_configurado_cierra_la_conexion(tmp_path, cargado):
    d = tmp_path / "run2"
    d.mkdir(mode=0o700)

    async def caso():
        async with puerto(ConfigPuerto(socket_dir=d, max_mensaje=4096), cargado) as srv:
            lector, escritor = await asyncio.open_unix_connection(str(srv.ruta_socket))
            escritor.write(b"FARO-TOKEN " + srv.ruta_token.read_bytes().strip() + b"\n")
            escritor.write(b"x" * 10_000 + b"\n")
            await escritor.drain()
            assert await asyncio.wait_for(lector.read(), 5) == b""  # el servidor cerro: EOF
            escritor.close()
        return srv.registros
    assert _llamadas(corre(caso())) == []


@pytest.mark.parametrize("valor,esperado", [
    ("a\nb", "a\\x20b"), ("a\r\nb\tc", "a\\x20b\\x20c"), ("x" * 500, "x" * 200), (None, "None"), (12, "12"),
    ("a\u2028b\u00a0c", "a\\x20b\\x20c"), ("a=b", "a\\x3db"),
])
def test_campo_log_es_de_una_linea_y_acotado(valor, esperado):
    assert _campo_log(valor) == esperado


# --------------------------------------------------------------------------- #
# el socket                                                                   #
# --------------------------------------------------------------------------- #

def test_el_socket_es_del_run_id_se_borra_al_cerrar_y_su_directorio_es_0700(cfg_puerto, cargado):
    async def caso():
        async with puerto(cfg_puerto, cargado, ejecucion=_ejecucion(run_id="abc-123")) as srv:
            assert srv.ruta_socket == cfg_puerto.socket_dir / "abc-123.sock"
            assert stat.S_IMODE(os.lstat(cfg_puerto.socket_dir).st_mode) == 0o700
            st = os.lstat(srv.ruta_socket)
            assert stat.S_ISSOCK(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o600  # 0600 de faro; a la jaula la deja pasar una ACL con nombre (test_faro_acl.py)
        assert not srv.ruta_socket.exists()
    corre(caso())


def test_dos_ejecuciones_con_el_mismo_run_id_no_comparten_socket(cfg_puerto, cargado):
    async def caso():
        async with puerto(cfg_puerto, cargado) as _:
            with pytest.raises(FileExistsError):
                async with puerto(cfg_puerto, cargado):
                    pass
    corre(caso())


@pytest.mark.parametrize("run_id", ["../x", "a/b", "", ".oculto", "x" * 100, "con espacio", "a\0b"])
def test_un_run_id_no_puede_escapar_del_directorio_de_sockets(run_id):
    with pytest.raises(ConfigFaroInvalida):
        _ejecucion(run_id=run_id)


def test_el_directorio_de_sockets_tiene_que_ser_privado(tmp_path, cargado):
    d = tmp_path / "abierto"
    d.mkdir()
    d.chmod(0o777)

    async def caso():
        with pytest.raises(ConfigFaroInvalida):
            async with servidor(ConfigPuerto(socket_dir=d), _ejecucion(), cargado, Bitacora()):
                pass
    corre(caso())


def test_la_configuracion_del_puerto_falla_cerrado(tmp_path):
    assert ConfigPuerto.desde_entorno({"JAX_FARO_SOCKET_DIR": str(tmp_path)}).socket_dir == tmp_path
    for malo in ({}, {"JAX_FARO_SOCKET_DIR": ""}, {"JAX_FARO_SOCKET_DIR": "relativo/x"}):
        with pytest.raises(ConfigFaroInvalida):
            ConfigPuerto.desde_entorno(malo)


def test_un_mensaje_grande_cruza_el_rele(tmp_path):
    grande = "linea de relleno\n" * 20000  # ~340 KB en una sola linea JSON
    (tmp_path / "g").mkdir()
    repo = repo_de_juguete(tmp_path / "g", {"common/skills/alfa/grande.md": grande})
    cfg = ConfigFaro(repo=repo, sha=_git(repo, "rev-parse", "HEAD"), destino=tmp_path / "eco", uid_duenio=os.getuid())
    paquete.construir_paquete(cfg)
    d = tmp_path / "run"
    d.mkdir(mode=0o700)

    async def caso():
        async with puerto(ConfigPuerto(socket_dir=d), cargar_paquete(cfg)) as srv, cliente_por_rele(srv) as c:
            r = await c.call_tool("skills.leer", {"nombre": "alfa", "archivo": "grande.md"})
            assert not r.is_error and len(json.dumps(r.structured_content) + str(r.content)) > 300_000
    corre(caso())


# --------------------------------------------------------------------------- #
# auditoria: guardia, _meta, argumentos, bitacora que puede fallar, logging    #
# --------------------------------------------------------------------------- #

def test_la_guardia_esta_instalada_en_el_middleware_del_servidor(cargado):
    ident = Identidad(_ejecucion(), "conn-1", peer_pid=1, peer_uid=os.getuid(), peer_gid=0)
    mcp = construir_servidor(cargado, identidad=ident, bitacora=Bitacora(emisores=[]), freno=lambda: False)
    assert any(isinstance(m, Guardia) for m in mcp.middleware)


def test__meta_no_cambia_el_hash_de_los_argumentos(cfg_puerto, cargado):
    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            await c.call_tool("skills.leer", {"nombre": "alfa"}, meta={"progressToken": "uno", "x": 1})
            await c.call_tool("skills.leer", {"nombre": "alfa"}, meta={"progressToken": "dos", "y": [2]})
        return srv.registros
    h = [r["hash_args"] for r in _llamadas(corre(caso())) if r["metodo"] == "tools/call"]
    assert len(h) == 2 and h[0] == h[1]


def test_la_bitacora_registra_los_argumentos_saneados_y_acotados(cfg_puerto, cargado):
    largo = "n\nombre=raro " + "z" * 500

    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            await c.call_tool("skills.leer", {"nombre": largo})
        return srv.registros
    r = next(x for x in _llamadas(corre(caso())) if x["metodo"] == "tools/call")
    assert "\n" not in r["argumentos"] and "=" not in r["argumentos"].replace("\\x3d", "") and len(r["argumentos"]) <= 200
    assert "nombre" in r["argumentos"]


def test_un_emisor_de_bitacora_que_falla_hace_fallar_la_llamada_y_no_entrega_el_resultado(cfg_puerto, cargado):
    def emisor(registro):
        if registro.get("metodo") == "tools/call":
            raise OSError("la tabla de la bitacora no esta disponible")

    async def caso():
        async with servidor(cfg_puerto, _ejecucion(), cargado, Bitacora(emisores=[emisor])) as srv, cliente_por_rele(srv) as c:
            try:
                r = await c.call_tool("skills.leer", {"nombre": "alfa"})
            except MCPError:
                return None
            return r
    r = corre(caso())
    assert r is None or (r.is_error and "cuerpo alfa" not in str(r.content) + json.dumps(r.structured_content))


def test_si_la_bitacora_falla_la_denegacion_por_freno_sigue_denegando(cfg_puerto, cargado, freno_propio):
    def siempre_falla(registro):
        if registro.get("metodo") == "tools/call":
            raise OSError("sin bitacora")

    async def caso():
        async with servidor(cfg_puerto, _ejecucion(), cargado, Bitacora(emisores=[siempre_falla])) as srv, cliente_por_rele(srv) as c:
            freno_propio.write_text("pausa")
            with pytest.raises(MCPError) as exc:
                await c.call_tool("skills.leer", {"nombre": "alfa"})
            assert exc.value.code == 423
    corre(caso())


def test_un_rechazo_de_conexion_se_mantiene_aunque_la_bitacora_falle(cfg_puerto, cargado):
    def siempre_falla(registro):
        raise OSError("sin bitacora")

    async def caso():
        async with servidor(cfg_puerto, _ejecucion(uid_esperado=os.getuid() + 1), cargado, Bitacora(emisores=[siempre_falla])) as srv:
            lector, escritor = await asyncio.open_unix_connection(str(srv.ruta_socket))
            assert await asyncio.wait_for(lector.read(), 5) == b""
            escritor.close()
    corre(caso())


def test_el_arranque_configura_el_logging_antes_del_primer_servidor_mcp(cfg_puerto, cargado, monkeypatch):
    raiz = logging.getLogger()
    monkeypatch.setattr(raiz, "handlers", [])      # un proceso de servicio recien arrancado: sin configurar

    async def caso():
        async with puerto(cfg_puerto, cargado) as srv, cliente_por_rele(srv) as c:
            await c.list_tools()
    corre(caso())
    assert raiz.handlers, "el servicio no configuro el logging"
    assert not any(type(h).__name__ == "RichHandler" for h in raiz.handlers), "lo configuro el SDK, no el servicio"
    from jax.faro.logs import FORMATO
    assert [h.formatter._fmt for h in raiz.handlers] == [FORMATO], "el formato es el del servicio, no el `%(message)s` del SDK"


@pytest.mark.parametrize("tenant", ["con espacio", "a|b", "x" * 65, "t\nx", "ñ"])
def test_una_ejecucion_no_puede_tener_un_tenant_que_los_topes_no_aceptarian(tenant):
    with pytest.raises(ConfigFaroInvalida, match="tenant"):
        _ejecucion(tenant=tenant)
