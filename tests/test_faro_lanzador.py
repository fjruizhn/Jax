"""El Faro, paso 0.3c (R3): el lanzador de jaula. Arma el argv de bwrap del plan: la jaula recibe por BIND DE
ARCHIVO solo SU socket y SU token (y el rele, de solo lectura); nunca el directorio de sockets. Corre con un uid
propio, distinto del de `faro` y del de root.

MODELO (medido en hall9000, bubblewrap 0.11.1): el uid de la jaula lo pone la ELEVACION (`JAX_FARO_JAULA_ELEVAR`,
con `{uid}`), no el argv. Un bwrap corriendo como root NO puede cambiar de uid dentro: el perfil de AppArmor
`bwrap//&unpriv_bwrap` confina a sus hijos y `setresuid` falla con EPERM (incluso con `--cap-add`); y bwrap
sin privilegios ya corre con el uid de quien lo lanza (su namespace de usuario no cambia el uid que ve el
kernel). Por eso el argv NO lleva `setpriv` ni `--cap-add`, y la configuracion exige `{uid}` en la elevacion.

Aqui, sin bwrap ni uids reales: la forma exacta del argv, la validacion del uid, la configuracion que falla
cerrado, el orden bitacora -> arranque y que el ejemplo versionado (`ops/faro/bwrap-ejemplo.txt`) no se desvie.
Con bwrap real y uids reales (sudo): `test_faro_jaula_real.py`.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from jax.faro import relay as modulo_relay
from jax.faro.bitacora import Bitacora
from jax.faro.config import ConfigFaroInvalida
from jax.faro.jaula import (ConfigJaula, DESTINO_RELAY, DESTINO_SOCKET, DESTINO_TOKEN, LanzadorJaula, argv_montajes,
                            ejemplo, validar_uid_jaula)
from tests._faro_utils import corre, ejecucion

RAIZ = Path(__file__).resolve().parents[1]
UID_JAULA = 50001
SOCKET_DIR = "/run/faro"


def _rutas(run_id="run-1"):
    return f"{SOCKET_DIR}/{run_id}.sock", f"{SOCKET_DIR}/{run_id}.token"


@pytest.fixture
def bwrap(tmp_path):
    ruta = tmp_path / "bwrap"
    ruta.write_text("#!/bin/sh\nexit 0\n")
    ruta.chmod(0o755)
    return ruta


ELEVAR = ("/usr/bin/sudo", "-n", "-u", "#{uid}", "--")


def _cfg(bwrap, elevar=ELEVAR):
    return ConfigJaula(bwrap=bwrap, elevar=tuple(elevar))


def _lanzador(bwrap, registros=None, **kw):
    bit = Bitacora(emisores=[registros.append] if registros is not None else [])
    return LanzadorJaula(_cfg(bwrap), bit, **kw)


def _pares(argv, opcion):
    return [(argv[i + 1], argv[i + 2]) for i, a in enumerate(argv) if a == opcion]


# --------------------------------------------------------------------------- #
# configuracion                                                               #
# --------------------------------------------------------------------------- #

def test_la_configuracion_sale_del_entorno(bwrap):
    cfg = ConfigJaula.desde_entorno({"JAX_FARO_BWRAP": str(bwrap), "JAX_FARO_JAULA_ELEVAR": "/usr/bin/sudo -n -u #{uid} --"})
    assert cfg.bwrap == bwrap and cfg.elevar == ("/usr/bin/sudo", "-n", "-u", "#{uid}", "--")


@pytest.mark.parametrize("falta", ["JAX_FARO_BWRAP", "JAX_FARO_JAULA_ELEVAR"])
def test_sin_bwrap_o_sin_elevacion_no_hay_lanzador(bwrap, falta):
    env = {"JAX_FARO_BWRAP": str(bwrap), "JAX_FARO_JAULA_ELEVAR": "/usr/bin/sudo -n -u #{uid} --"}
    env.pop(falta)
    with pytest.raises(ConfigFaroInvalida, match=falta):
        ConfigJaula.desde_entorno(env)


@pytest.mark.parametrize("bwrap_ruta,elevar", [
    ("relativa/bwrap", "/usr/bin/sudo -n -u #{uid} --"),
    ("/no/existe/bwrap", "/usr/bin/sudo -n -u #{uid} --"),
    ("__directorio__", "/usr/bin/sudo -n -u #{uid} --"),
    ("__no_ejecutable__", "/usr/bin/sudo -n -u #{uid} --"),
    ("__ok__", "sudo -n -u #{uid} --"),            # el primer elemento de la elevacion tiene que ser una ruta absoluta (PATH)
    ("__ok__", "   "),
    ("__ok__", "/usr/bin/sudo 'sin cerrar {uid}"),
    ("__ok__", "/usr/bin/sudo -n --"),             # sin {uid} la jaula correria con el uid de quien lanza: el de `faro`
])
def test_una_configuracion_de_jaula_insegura_o_rota_no_arranca(bwrap, tmp_path, bwrap_ruta, elevar):
    if bwrap_ruta == "__ok__":
        bwrap_ruta = str(bwrap)
    elif bwrap_ruta == "__directorio__":
        bwrap_ruta = str(tmp_path)
    elif bwrap_ruta == "__no_ejecutable__":
        f = tmp_path / "plano"
        f.write_text("x")
        f.chmod(0o644)
        bwrap_ruta = str(f)
    with pytest.raises(ConfigFaroInvalida):
        ConfigJaula.desde_entorno({"JAX_FARO_BWRAP": bwrap_ruta, "JAX_FARO_JAULA_ELEVAR": elevar})


# --------------------------------------------------------------------------- #
# el uid de la jaula                                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("uid", [0, os.geteuid(), -1, 2 ** 32 - 1, 2 ** 40, True, None, "1000", 1.5])
def test_el_uid_de_la_jaula_no_puede_ser_root_ni_el_del_servicio_ni_uno_raro(uid):
    with pytest.raises(ConfigFaroInvalida):
        validar_uid_jaula(uid)


def test_ni_el_uid_del_orquestador_ni_los_prohibidos_valen_como_jaula():
    with pytest.raises(ConfigFaroInvalida):
        validar_uid_jaula(990, prohibidos={990})
    assert validar_uid_jaula(UID_JAULA, prohibidos={990}) == UID_JAULA


def test_el_lanzador_se_niega_con_el_uid_de_faro_o_de_root(bwrap):
    l = _lanzador(bwrap)
    for uid in (0, os.geteuid()):
        with pytest.raises(ConfigFaroInvalida):
            l.argv(ejecucion(uid_esperado=uid), *_rutas(), ["/usr/bin/true"])


def test_el_lanzador_se_niega_con_el_uid_del_orquestador(bwrap):
    l = _lanzador(bwrap, uids_prohibidos={990})
    with pytest.raises(ConfigFaroInvalida):
        l.argv(ejecucion(uid_esperado=990), *_rutas(), ["/usr/bin/true"])


# --------------------------------------------------------------------------- #
# el argv                                                                     #
# --------------------------------------------------------------------------- #

def test_la_jaula_recibe_por_bind_de_archivo_solo_su_socket_su_token_y_el_rele(bwrap):
    sock, tok = _rutas()
    argv = _lanzador(bwrap).argv(ejecucion(uid_esperado=UID_JAULA), sock, tok, ["/usr/bin/python3", "-m", "jax.faro.relay"])
    rw, ro = _pares(argv, "--bind"), _pares(argv, "--ro-bind")
    assert rw == [(sock, DESTINO_SOCKET)]                                           # el socket: lectura y escritura
    assert (tok, DESTINO_TOKEN) in ro                                               # el token: solo lectura
    assert (str(Path(modulo_relay.__file__)), DESTINO_RELAY) in ro                  # el rele: solo lectura
    assert {origen for origen, _ in ro} == {tok, str(Path(modulo_relay.__file__)), "/usr"}
    assert not _pares(argv, "--dev-bind") and "--bind-try" not in argv and "--ro-bind-try" not in argv


def test_el_directorio_de_sockets_no_aparece_en_ningun_montaje(bwrap):
    sock, tok = _rutas()
    argv = _lanzador(bwrap).argv(ejecucion(uid_esperado=UID_JAULA), sock, tok, ["/usr/bin/true"])
    assert SOCKET_DIR not in argv and "/run" not in argv and "/" not in argv and "/run/faro/" not in argv
    montajes = [a for a in argv if a.startswith("/run/")]
    assert sorted(montajes) == sorted([sock, tok])                                  # lo unico que menciona /run son SUS dos archivos


def test_otra_ejecucion_no_aparece_en_el_argv_de_esta(bwrap):
    a, b = _rutas("run-a"), _rutas("run-b")
    argv_a = _lanzador(bwrap).argv(ejecucion(run_id="run-a", uid_esperado=UID_JAULA), *a, ["/usr/bin/true"])
    texto = " ".join(argv_a)
    assert "run-a" in texto and "run-b" not in texto


def test_el_aislamiento_de_la_jaula_esta_en_el_argv(bwrap):
    argv = _lanzador(bwrap).argv(ejecucion(uid_esperado=UID_JAULA), *_rutas(), ["/usr/bin/true"])
    for opcion in ("--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net", "--unshare-cgroup-try",
                   "--die-with-parent", "--new-session", "--clearenv"):
        assert opcion in argv, opcion
    # sin red, y SIN namespace de usuario: con uno, el kernel veria al par con el uid del servicio y no con el de la jaula
    for prohibida in ("--share-net", "--unshare-all", "--unshare-user", "--unshare-user-try", "--uid", "--gid", "--cap-add"):
        assert prohibida not in argv, prohibida


def test_el_argv_no_baja_de_uid_por_dentro_el_uid_lo_pone_la_elevacion(bwrap):
    comando = ["/usr/bin/python3", "-m", "jax.faro.relay", "--socket", DESTINO_SOCKET]
    argv = _lanzador(bwrap).argv(ejecucion(uid_esperado=UID_JAULA), *_rutas(), comando)
    assert argv[argv.index("--") + 1:] == comando                                    # tras el `--` va el comando tal cual, sin envoltorio
    assert not any("setpriv" in a or a.startswith("--reuid") or a.startswith("--cap") for a in argv)


def test_el_entorno_de_la_jaula_es_minimo_y_no_lleva_nada_secreto(bwrap, tmp_path):
    token = tmp_path / "t.token"
    token.write_text("TOKEN-SECRETO-DE-LA-EJECUCION\n")
    argv = _lanzador(bwrap).argv(ejecucion(uid_esperado=UID_JAULA), "/run/faro/run-1.sock", str(token), ["/usr/bin/true"])
    assert "TOKEN-SECRETO" not in " ".join(argv)                                    # el contenido del token nunca viaja por argv
    asignadas = {argv[i + 1]: argv[i + 2] for i, a in enumerate(argv) if a == "--setenv"}
    assert set(asignadas) == {"PATH", "HOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE"}
    assert asignadas["PYTHONPATH"] == "/faro/relay"


@pytest.mark.parametrize("comando", [[], "python", ["python3"], ["relativo/bin"], ["/usr/bin/x", 5], ["/usr/bin/x\x00y"], [""]])
def test_el_comando_tiene_que_ser_una_lista_con_ejecutable_absoluto_y_argumentos_de_texto(bwrap, comando):
    with pytest.raises(ConfigFaroInvalida):
        _lanzador(bwrap).argv(ejecucion(uid_esperado=UID_JAULA), *_rutas(), comando)


@pytest.mark.parametrize("sock,tok", [("relativo.sock", "/run/faro/x.token"), ("/run/faro/x.sock", "relativo.token"),
                                      ("/run/faro/x.sock\x00", "/run/faro/x.token"), ("", "/run/faro/x.token")])
def test_las_rutas_que_se_montan_son_absolutas_y_sin_nul(bwrap, sock, tok):
    with pytest.raises(ConfigFaroInvalida):
        _lanzador(bwrap).argv(ejecucion(uid_esperado=UID_JAULA), sock, tok, ["/usr/bin/true"])


def test_el_argv_completo_antepone_la_elevacion_y_el_bwrap_configurados(bwrap):
    l = LanzadorJaula(ConfigJaula(bwrap=bwrap, elevar=ELEVAR), Bitacora(emisores=[]))
    completo = l.argv_completo(ejecucion(uid_esperado=UID_JAULA), *_rutas(), ["/usr/bin/true"])
    assert completo[:6] == ["/usr/bin/sudo", "-n", "-u", f"#{UID_JAULA}", "--", str(bwrap)]
    assert completo[6:] == l.argv(ejecucion(uid_esperado=UID_JAULA), *_rutas(), ["/usr/bin/true"])


def test_la_elevacion_lleva_el_uid_de_la_jaula_en_cada_aparicion_de_uid(bwrap):
    l = LanzadorJaula(ConfigJaula(bwrap=bwrap, elevar=("/usr/bin/ayudante", "--uid={uid}", "--gid", "{uid}")), Bitacora(emisores=[]))
    completo = l.argv_completo(ejecucion(uid_esperado=UID_JAULA), *_rutas(), ["/usr/bin/true"])
    assert completo[:4] == ["/usr/bin/ayudante", f"--uid={UID_JAULA}", "--gid", str(UID_JAULA)]


def test_los_montajes_son_una_funcion_pura_de_las_dos_rutas():
    assert argv_montajes(*_rutas()) == argv_montajes(*_rutas())
    assert argv_montajes(*_rutas("run-a")) != argv_montajes(*_rutas("run-b"))


def test_el_ejemplo_versionado_es_el_que_genera_el_codigo():
    archivo = RAIZ / "ops" / "faro" / "bwrap-ejemplo.txt"
    assert archivo.read_text() == ejemplo()
    texto = ejemplo()
    assert "<run_id>" in texto and "<uid_jaula>" in texto


# --------------------------------------------------------------------------- #
# lanzar: bitacora primero, arranque despues                                  #
# --------------------------------------------------------------------------- #

class _SrvFalso:
    def __init__(self, ej, sock, tok):
        self.ejecucion, self.ruta_socket, self.ruta_token = ej, Path(sock), Path(tok)


def _srv(uid=UID_JAULA, **kw):
    ej = ejecucion(uid_esperado=uid, **kw)
    return _SrvFalso(ej, *_rutas(ej.run_id))


def test_lanzar_anota_antes_de_arrancar_y_arranca_con_el_argv_completo_y_un_entorno_limpio(bwrap):
    orden = []
    llamadas = []

    async def crear_proceso(*argv, **kw):
        orden.append("arranque")
        llamadas.append((argv, kw))
        return "proceso"

    async def caso():
        bit = Bitacora(emisores=[lambda r: orden.append(r["evento"])])
        l = LanzadorJaula(_cfg(bwrap), bit, crear_proceso=crear_proceso)
        return await l.lanzar(_srv(), ["/usr/bin/python3", "-c", "pass"])
    assert corre(caso()) == "proceso"
    assert orden == ["jaula_lanzada", "arranque"]
    argv, kw = llamadas[0]
    assert list(argv[:6]) == ["/usr/bin/sudo", "-n", "-u", f"#{UID_JAULA}", "--", str(bwrap)] and "/usr/bin/setpriv" not in argv
    assert set(kw["env"]) == {"PATH"} and kw["start_new_session"] is True
    assert kw["stdin"] is not None and "shell" not in kw


def test_lo_anotado_dice_quien_y_con_que_uid_y_no_lleva_el_token(bwrap, tmp_path):
    registros = []
    token = tmp_path / "x.token"
    token.write_text("TOKEN-SECRETO\n")

    async def crear_proceso(*a, **k):
        return None

    async def caso():
        l = _lanzador(bwrap, registros, crear_proceso=crear_proceso)
        srv = _srv()
        srv.ruta_token = token
        await l.lanzar(srv, ["/usr/bin/true"])
    corre(caso())
    r = registros[0]
    assert r["evento"] == "jaula_lanzada" and r["run_id"] == "run-1" and r["uid_jaula"] == UID_JAULA
    assert r["motor"] == "codex" and r["entry_point"] == "repl" and r["id_correlacion"] == "corr-real"
    assert len(r["hash_argv"]) == 64 and "TOKEN-SECRETO" not in str(r)


def test_si_la_bitacora_no_puede_anotar_no_se_lanza_nada(bwrap):
    lanzado = []

    async def crear_proceso(*a, **k):
        lanzado.append(1)

    def rota(registro):
        raise OSError("sin bitacora")

    async def caso():
        l = LanzadorJaula(_cfg(bwrap), Bitacora(emisores=[rota]), crear_proceso=crear_proceso)
        await l.lanzar(_srv(), ["/usr/bin/true"])
    with pytest.raises(OSError):
        corre(caso())
    assert lanzado == []


def test_si_el_arranque_falla_se_anota_y_se_propaga(bwrap):
    registros = []

    async def crear_proceso(*a, **k):
        raise FileNotFoundError("no existe sudo")

    async def caso():
        await _lanzador(bwrap, registros, crear_proceso=crear_proceso).lanzar(_srv(), ["/usr/bin/true"])
    with pytest.raises(FileNotFoundError):
        corre(caso())
    assert [r["evento"] for r in registros] == ["jaula_lanzada", "jaula_fallo"]
    assert registros[1]["run_id"] == "run-1" and registros[1]["motivo"] == "FileNotFoundError"


def test_con_un_uid_invalido_no_se_anota_ni_se_arranca_nada(bwrap):
    registros, lanzado = [], []

    async def crear_proceso(*a, **k):
        lanzado.append(1)

    async def caso():
        await _lanzador(bwrap, registros, crear_proceso=crear_proceso).lanzar(_srv(uid=0), ["/usr/bin/true"])
    with pytest.raises(ConfigFaroInvalida):
        corre(caso())
    assert registros == [] and lanzado == []


def test_lanzar_de_verdad_un_proceso_hijo_con_un_bwrap_falso(bwrap, tmp_path):
    """Sin bwrap real: un `bwrap` de mentira que vuelca sus argumentos. Ejercita el arranque REAL (subproceso, sin
    shell, entorno limpio, sesion propia) y que la elevacion del config se antepone."""
    salida = tmp_path / "args.txt"
    falso = tmp_path / "bwrap-falso"
    falso.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {salida}\nenv > {tmp_path}/env.txt\n")
    falso.chmod(0o755)
    elevador = tmp_path / "elevador"
    elevador.write_text(f"#!/bin/sh\necho \"$1\" > {tmp_path}/uid-pedido.txt\nshift\nexec \"$@\"\n")
    elevador.chmod(0o755)

    async def caso():
        l = LanzadorJaula(ConfigJaula(bwrap=falso, elevar=(str(elevador), "{uid}")), Bitacora(emisores=[]))
        p = await l.lanzar(_srv(), ["/usr/bin/true"])
        return await p.wait()
    assert corre(caso()) == 0
    args = salida.read_text().splitlines()
    assert "--bind" in args and DESTINO_SOCKET in args and "/usr/bin/setpriv" not in args
    assert (tmp_path / "uid-pedido.txt").read_text().strip() == str(UID_JAULA)           # la elevacion recibio el uid de la jaula
    entorno = dict(l.split("=", 1) for l in (tmp_path / "env.txt").read_text().splitlines() if "=" in l)
    assert set(entorno) <= {"PATH", "PWD", "SHLVL", "_", "OLDPWD"}                        # nada heredado del servicio
    assert os.environ.get("HOME") not in entorno.values()
