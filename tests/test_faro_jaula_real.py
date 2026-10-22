"""El Faro, paso 0.3c (R3): la jaula y el canal de control con bwrap REAL y uids REALES del sistema.

Necesita `bwrap` (bubblewrap) y `sudo -n` sin contrasena (para correr procesos de prueba como otro uid con
`setpriv`: `nobody` = 65534, `daemon` = 1, `www-data` = 33). Si falta cualquiera de los dos las pruebas se SALTAN y el job de CI
(`faro-jaula`) exige cero saltadas: un skip se ve igual que verde.

Lo que ejercita de verdad:
- el directorio de sockets (0700 de `faro`) no es listable ni alcanzable desde OTRO uid del sistema, aunque el
  camino hasta el sea transitable;
- el canal de control con `SO_PEERCRED` real: el orquestador (otro uid) crea una ejecucion; un tercer uid que
  SI llega al archivo (mismo grupo) se rechaza por su uid; un uid sin acceso al directorio ni conecta;
- la elevacion de uid del lanzador: el proceso que arranca `lanzar` corre con el kernel-uid de la jaula;
- con bwrap real (sin cambiar de uid): la jaula ve exactamente su socket, su token y el rele, no el directorio
  de sockets; un cliente MCP real habla con el Puerto a traves de la jaula; el token de otra ejecucion no entra.

LO QUE ANTES ESTABA ABIERTO y ahora se ejercita: bwrap CORRIENDO COMO el uid de la jaula con las fuentes en el
directorio 0700 de `faro`. Lo resuelve una ACL por ejecucion que pone `faro` sin root (directorio `--x`, socket
`rw-`, token `r--`, solo para el uid de esa jaula): ver `test_faro_acl.py` y las pruebas `test_acl_*` de aqui.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from jax.faro.bitacora import Bitacora
from jax.faro.config import ConfigPuerto
from jax.faro.control import ConfigControl, ServidorControl
from jax.faro.jaula import ConfigJaula, LanzadorJaula, argv_montajes
from jax.faro.transporte import ServidorPuerto
from tests._faro_utils import RAIZ, corre, ejecucion, paquete_listo, puerto

NOBODY, DAEMON, JAULA = 65534, 1, 33        # uids que existen en cualquier Ubuntu: nobody, daemon y www-data (sudo exige una cuenta)
GID = os.getgid()
PEDIDO = {"op": "crear", "usuario": "u-real", "tenant": "t-real", "faceta": "hyde", "motor": "codex",
          "pipeline": "p-real", "entry_point": "repl", "uid_jaula": JAULA}


def _sudo_ok() -> bool:
    return shutil.which("sudo") is not None and subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0


def _bwrap_ok() -> bool:
    ruta = shutil.which("bwrap")
    if ruta is None:
        return False
    r = subprocess.run([ruta, "--unshare-pid", "--die-with-parent", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
                        "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--", "/usr/bin/true"],
                       capture_output=True)
    return r.returncode == 0


@pytest.fixture
def requiere_sudo():
    if not _sudo_ok():
        pytest.skip("sin `sudo -n`: no se puede correr un proceso de prueba como otro uid")


@pytest.fixture
def requiere_bwrap():
    if not _bwrap_ok():
        pytest.skip("sin bwrap utilizable (bubblewrap no instalado o sin namespaces de usuario)")


@pytest.fixture
def base():
    """Un directorio al que cualquier uid puede entrar (0711) para que lo que falle dentro falle por el 0700
    de `faro` y no porque el camino hasta ahi no se pueda recorrer."""
    d = Path(tempfile.mkdtemp(prefix="faro-real-", dir="/tmp"))
    d.chmod(0o711)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mundo(base):
    (base / "pkg").mkdir()
    (base / "pkg").chmod(0o755)                    # el paquete exige ancestros sin escritura de grupo (la umask del host puede dar 0775)
    _cfg, cargado = paquete_listo(base / "pkg")
    d = base / "run"
    d.mkdir(mode=0o700)
    return type("Mundo", (), {"base": base, "cargado": cargado, "cfg_puerto": ConfigPuerto(socket_dir=d)})()


async def como_uid(uid: int, codigo: str, *args: str, grupo: int | None = None, plazo: float = 20.0):
    """Corre `python3 -c codigo args...` con el uid (y el gid) dados y SIN grupos suplementarios, y devuelve
    (returncode, stdout, stderr). Usa `sudo setpriv` en vez de `sudo -u/-g`: este ultimo exige reglas de grupo en
    sudoers (`(ALL:ALL)`) que un runner no tiene siempre, y `-u` exige una cuenta en el sistema."""
    gid = grupo if grupo is not None else uid
    p = await asyncio.create_subprocess_exec(
        "sudo", "-n", "/usr/bin/setpriv", f"--reuid={uid}", f"--regid={gid}", "--clear-groups", "--no-new-privs", "--",
        "/usr/bin/python3", "-c", codigo, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    salida, error = await asyncio.wait_for(p.communicate(), plazo)
    return p.returncode, salida.decode(), error.decode()


async def como_uid_argv(uid: int, argv: list[str], *, entrada: str = "", plazo: float = 30.0):
    p = await asyncio.create_subprocess_exec(
        "sudo", "-n", "/usr/bin/setpriv", f"--reuid={uid}", f"--regid={uid}", "--clear-groups", "--no-new-privs", "--", *argv,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    salida, error = await asyncio.wait_for(p.communicate(entrada.encode()), plazo)
    return p.returncode, salida.decode(), error.decode()


INTENTOS = r"""
import json, os, socket, sys
d, sock, tok = sys.argv[1:4]
r = {}
for nombre, f in (("listdir", lambda: os.listdir(d)), ("leer_token", lambda: open(tok).read()),
                  ("stat_socket", lambda: os.stat(sock)), ("stat_dir", lambda: os.stat(d))):
    try:
        f(); r[nombre] = "ok"
    except Exception as e:
        r[nombre] = type(e).__name__
s = socket.socket(socket.AF_UNIX)
try:
    s.connect(sock); r["conectar"] = "ok"
except Exception as e:
    r["conectar"] = type(e).__name__
print(json.dumps(r))
"""


# --------------------------------------------------------------------------- #
# el directorio de sockets, desde otro uid REAL                               #
# --------------------------------------------------------------------------- #

def _intentos(mundo, srv, otro=None):
    otro = otro or srv
    return (str(mundo.cfg_puerto.socket_dir), str(srv.ruta_socket), str(srv.ruta_token))


def test_el_directorio_de_sockets_no_es_listable_ni_alcanzable_desde_un_uid_que_no_es_la_jaula(mundo, requiere_sudo):
    async def caso():
        bit = Bitacora(emisores=[])
        async with ServidorPuerto(mundo.cfg_puerto, ejecucion(uid_esperado=NOBODY), mundo.cargado, bit) as srv:
            return await como_uid(DAEMON, INTENTOS, *_intentos(mundo, srv))
    rc, salida, error = corre(caso())
    assert rc == 0, error
    r = json.loads(salida)
    assert r["stat_dir"] == "ok"                                      # el camino hasta el directorio SI se recorre (existe, se ve)...
    assert r["listdir"] == r["leer_token"] == r["stat_socket"] == r["conectar"] == "PermissionError", r    # ...y la ACL no es suya


def test_acl_la_jaula_abre_lo_suyo_pero_no_lista_el_directorio_ni_toca_lo_de_otra_ejecucion(mundo, requiere_sudo):
    async def caso():
        bit = Bitacora(emisores=[])
        async with ServidorPuerto(mundo.cfg_puerto, ejecucion(run_id="run-a", uid_esperado=NOBODY), mundo.cargado, bit) as a, \
                ServidorPuerto(mundo.cfg_puerto, ejecucion(run_id="run-b", uid_esperado=JAULA), mundo.cargado, bit) as b:
            d = str(mundo.cfg_puerto.socket_dir)
            nobody_lo_suyo = await como_uid(NOBODY, INTENTOS, d, str(a.ruta_socket), str(a.ruta_token))
            nobody_lo_de_b = await como_uid(NOBODY, INTENTOS, d, str(b.ruta_socket), str(b.ruta_token))
            www_lo_suyo = await como_uid(JAULA, INTENTOS, d, str(b.ruta_socket), str(b.ruta_token))
            www_lo_de_a = await como_uid(JAULA, INTENTOS, d, str(a.ruta_socket), str(a.ruta_token))
            return [json.loads(x[1]) for x in (nobody_lo_suyo, nobody_lo_de_b, www_lo_suyo, www_lo_de_a)]
    suyo, ajeno, www_suyo, www_ajeno = corre(caso())
    # nobody: su socket (conecta) y su token (lee) si; el directorio, NO se lista (EACCES)
    assert suyo["listdir"] == "PermissionError" and suyo["stat_socket"] == "ok" and suyo["conectar"] == "ok" and suyo["leer_token"] == "ok"
    # ... y lo de run-b (otro uid) no: ni leer su token ni conectar a su socket
    assert ajeno["leer_token"] == "PermissionError" and ajeno["conectar"] == "PermissionError" and ajeno["listdir"] == "PermissionError"
    # www-data es la jaula de run-b: lo suyo si, lo de run-a no
    assert www_suyo["leer_token"] == "ok" and www_suyo["conectar"] == "ok" and www_suyo["listdir"] == "PermissionError"
    assert www_ajeno["leer_token"] == "PermissionError" and www_ajeno["conectar"] == "PermissionError"


def test_acl_al_cerrar_la_ejecucion_el_uid_ya_no_pasa_por_el_directorio(mundo, requiere_sudo):
    async def caso():
        bit = Bitacora(emisores=[])
        d = str(mundo.cfg_puerto.socket_dir)
        async with ServidorPuerto(mundo.cfg_puerto, ejecucion(uid_esperado=NOBODY), mundo.cargado, bit) as srv:
            dentro = json.loads((await como_uid(NOBODY, INTENTOS, d, str(srv.ruta_socket), str(srv.ruta_token)))[1])
            ruta_s, ruta_t = str(srv.ruta_socket), str(srv.ruta_token)
        fuera = json.loads((await como_uid(NOBODY, INTENTOS, d, ruta_s, ruta_t))[1])
        return dentro, fuera
    dentro, fuera = corre(caso())
    assert dentro["stat_socket"] == "ok" and fuera["stat_socket"] == "PermissionError" and fuera["listdir"] == "PermissionError"


def test_acl_bwrap_real_corriendo_como_el_uid_de_la_jaula_abre_sus_fuentes_y_habla_con_el_puerto(mundo, requiere_sudo, requiere_bwrap):
    """Lo que antes estaba ABIERTO: bwrap con el kernel-uid de la jaula, fuentes en el directorio 0700 de `faro`."""
    relay = mundo.base / "relay.py"
    shutil.copy(RAIZ / "jax" / "faro" / "relay.py", relay)
    relay.chmod(0o644)
    inicializar = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})

    async def caso():
        bit = Bitacora(emisores=[])
        regs = []
        bit = Bitacora(emisores=[regs.append])
        async with ServidorPuerto(mundo.cfg_puerto, ejecucion(uid_esperado=NOBODY), mundo.cargado, bit) as srv:
            args = [shutil.which("bwrap"), *argv_montajes(str(srv.ruta_socket), str(srv.ruta_token), relay=str(relay)), "--",
                    "/usr/bin/python3", "-m", "jax.faro.relay", "--socket", "/faro/puerto.sock", "--token-file", "/faro/token"]
            r = await como_uid_argv(NOBODY, args, entrada=inicializar + "\n")
            return r, regs
    (rc, salida, error), regs = corre(caso())
    assert '"serverInfo"' in salida and '"faro"' in salida, (rc, salida, error)
    assert not [x for x in regs if x.get("evento") == "conexion_rechazada"]
    assert [x["peer_uid"] for x in regs if x.get("evento") == "llamada"][0] == NOBODY            # el kernel vio el uid de la jaula


# --------------------------------------------------------------------------- #
# el canal de control con SO_PEERCRED real                                    #
# --------------------------------------------------------------------------- #

CLIENTE_CONTROL = r"""
import json, socket, sys
ruta, pedido = sys.argv[1], sys.argv[2]
s = socket.socket(socket.AF_UNIX)
try:
    s.connect(ruta)
except Exception as e:
    print(json.dumps({"conexion": type(e).__name__})); sys.exit(0)
try:
    s.sendall(pedido.encode() + b"\n")
    datos = s.makefile("rb").readline()
    print(json.dumps({"respuesta": json.loads(datos) if datos else None}))
except (ConnectionResetError, BrokenPipeError):
    print(json.dumps({"respuesta": None}))
"""


def _control(mundo, registros):
    d = mundo.base / "control"
    d.mkdir(mode=0o750)
    cfg = ConfigControl(control_dir=d, orquestador_uid=NOBODY, jaula_uid_min=20, jaula_uid_max=40, plazo_s=5.0)
    bit = Bitacora(emisores=[registros.append])
    return ServidorControl(cfg, lambda ej: ServidorPuerto(mundo.cfg_puerto, ej, mundo.cargado, bit), bit)


def test_el_canal_de_control_autentica_por_el_uid_real_del_kernel(mundo, requiere_sudo):
    registros = []

    async def caso():
        async with _control(mundo, registros) as srv:
            ruta, pedido = str(srv.ruta_socket), json.dumps(PEDIDO)
            sin_grupo = await como_uid(NOBODY, CLIENTE_CONTROL, ruta, pedido)                     # el orquestador, sin el grupo
            orquestador = await como_uid(NOBODY, CLIENTE_CONTROL, ruta, pedido, grupo=GID)        # el orquestador, con el grupo
            tercero = await como_uid(DAEMON, CLIENTE_CONTROL, ruta, pedido, grupo=GID)            # otro uid que SI llega al archivo
            return sin_grupo, orquestador, tercero, srv.ejecuciones
    sin_grupo, orquestador, tercero, ejecuciones = corre(caso())
    assert json.loads(sin_grupo[1]) == {"conexion": "PermissionError"}            # sin el grupo ni siquiera abre el socket
    r = json.loads(orquestador[1])["respuesta"]
    assert r["ok"] and r["uid_jaula"] == JAULA and list(ejecuciones) == [r["run_id"]]
    assert json.loads(tercero[1]) == {"respuesta": None}                           # llega al archivo y se le cierra por su uid
    rechazos = [x for x in registros if x["evento"] == "control_rechazado"]
    assert [(x["motivo"], x["peer_uid"]) for x in rechazos] == [("uid_no_autorizado", DAEMON)]
    creados = [x for x in registros if x["evento"] == "control_creado"]
    assert [x["peer_uid"] for x in creados] == [NOBODY]


def test_un_uid_de_jaula_real_no_abre_el_socket_de_control(mundo, requiere_sudo):
    """El uid de una jaula (aqui www-data, 33) no pertenece al grupo del directorio de control."""
    async def caso():
        async with _control(mundo, []) as srv:
            return await como_uid(JAULA, CLIENTE_CONTROL, str(srv.ruta_socket), json.dumps(PEDIDO)), srv.ejecuciones
    (rc, salida, _error), ejecuciones = corre(caso())
    assert json.loads(salida) == {"conexion": "PermissionError"} and ejecuciones == {}


# --------------------------------------------------------------------------- #
# la elevacion de uid del lanzador                                            #
# --------------------------------------------------------------------------- #

def test_lanzar_arranca_el_proceso_con_el_kernel_uid_de_la_jaula(mundo, requiere_sudo):
    falso = mundo.base / "bwrap-falso"
    falso.write_text("#!/usr/bin/python3\nimport os\nprint(os.getuid(), os.getgid())\n")
    falso.chmod(0o755)
    cfg = ConfigJaula(bwrap=falso, elevar=("/usr/bin/sudo", "-n", "-u", "#{uid}", "--"))

    async def caso():
        bit = Bitacora(emisores=[])
        async with ServidorPuerto(mundo.cfg_puerto, ejecucion(uid_esperado=JAULA), mundo.cargado, bit) as srv:
            p = await LanzadorJaula(cfg, bit).lanzar(srv, ["/usr/bin/true"])
            salida, _ = await asyncio.wait_for(p.communicate(), 20)
            return salida.decode().split()
    uid, _gid = corre(caso())
    assert int(uid) == JAULA != os.getuid()


# --------------------------------------------------------------------------- #
# bwrap real                                                                  #
# --------------------------------------------------------------------------- #

VISTA = r"""
import json, os, sys
r = {"faro": sorted(os.listdir("/faro")), "raiz": sorted(os.listdir("/")), "socket_dir_existe": os.path.exists(sys.argv[1]),
     "relay": sorted(os.path.join(d, f) for d, _, fs in os.walk("/faro/relay") for f in fs),
     "token": open("/faro/token").read().strip(), "env": sorted(os.environ)}
for nombre, ruta in (("escribir_token", "/faro/token"), ("escribir_usr", "/usr/escrito"), ("escribir_relay", "/faro/relay/jax/faro/relay.py")):
    try:
        open(ruta, "w").write("x"); r[nombre] = "ok"
    except Exception as e:
        r[nombre] = type(e).__name__
print(json.dumps(r))
"""


def _correr_bwrap(argv_montajes_, comando, **kw):
    return subprocess.run([shutil.which("bwrap"), *argv_montajes_, "--", *comando], capture_output=True, text=True, timeout=60, **kw)


def test_la_jaula_ve_solo_su_socket_su_token_y_el_rele(mundo, requiere_bwrap):
    async def caso():
        async with puerto(mundo.cfg_puerto, mundo.cargado) as srv:
            token = srv.ruta_token.read_text().strip()
            r = await asyncio.to_thread(_correr_bwrap, argv_montajes(str(srv.ruta_socket), str(srv.ruta_token)),
                                        ["/usr/bin/python3", "-c", VISTA, str(mundo.cfg_puerto.socket_dir)])
            return r, token
    r, token = corre(caso())
    assert r.returncode == 0, r.stderr
    vista = json.loads(r.stdout)
    assert vista["faro"] == ["puerto.sock", "relay", "token"]
    assert vista["relay"] == ["/faro/relay/jax/faro/relay.py"]
    assert vista["token"] == token
    assert vista["socket_dir_existe"] is False and "run" not in vista["raiz"]       # el directorio de sockets NO esta en la jaula
    assert vista["escribir_token"] != "ok" and vista["escribir_usr"] != "ok" and vista["escribir_relay"] != "ok"
    assert set(vista["env"]) - {"LC_CTYPE", "PWD"} == {"HOME", "PATH", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH"}    # entorno minimo (LC_CTYPE lo pone Python, PWD bwrap)


def _params_jaula(sock, tok):
    args = [*argv_montajes(sock, tok), "--", "/usr/bin/python3", "-m", "jax.faro.relay", "--socket", "/faro/puerto.sock",
            "--token-file", "/faro/token"]
    return StdioServerParameters(command=shutil.which("bwrap"), args=args)


def test_un_cliente_mcp_real_habla_con_el_puerto_a_traves_de_la_jaula(mundo, requiere_bwrap):
    async def caso():
        async with puerto(mundo.cfg_puerto, mundo.cargado) as srv:
            async with Client(_params_jaula(str(srv.ruta_socket), str(srv.ruta_token))) as c:
                r = await asyncio.wait_for(c.call_tool("skills.leer", {"nombre": "alfa"}), 30)
            return r, srv.registros
    r, registros = corre(caso())
    assert not r.is_error and "cuerpo alfa" in str(r.content)
    assert any(x.get("objetivo") == "skills.leer" and x.get("decision") == "permitido" for x in registros)


def test_el_token_de_otra_ejecucion_no_entra_ni_dentro_de_la_jaula(mundo, requiere_bwrap):
    async def caso():
        async with puerto(mundo.cfg_puerto, mundo.cargado, ej=ejecucion(run_id="run-a")) as a, \
                puerto(mundo.cfg_puerto, mundo.cargado, ej=ejecucion(run_id="run-b")) as b:
            # la jaula recibe el socket de A pero el token de B
            with pytest.raises(Exception):
                async with Client(_params_jaula(str(a.ruta_socket), str(b.ruta_token))) as c:
                    await asyncio.wait_for(c.call_tool("skills.leer", {"nombre": "alfa"}), 20)
            await asyncio.sleep(0.2)
            return a.registros
    registros = corre(caso())
    assert [x["motivo"] for x in registros if x.get("evento") == "conexion_rechazada"] == ["token_invalido"]
    assert not [x for x in registros if x.get("metodo") == "tools/call"]



def test_minor8_visudo_acepta_el_sudoers_de_ejemplo(requiere_sudo):
    r = subprocess.run(["sudo", "-n", "visudo", "-cf", str(RAIZ / "ops" / "faro" / "sudoers-jaula-ejemplo")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
