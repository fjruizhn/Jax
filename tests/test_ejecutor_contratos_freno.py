# tests/test_ejecutor_contratos_freno.py
"""Freno root del Ejecutor (C4) sin root: cgroup y /proc falsos, barrido remoto
falso. Nada de este archivo manda una señal a un proceso real."""
import json
import signal
import subprocess
from pathlib import Path

import pytest

from jax.core import interruptor
from jax.ejecutor.contratos import freno as F
from jax.ejecutor.contratos import pausa as P


def _cfg(tmp_path, remotos=(), uid=1001):
    return F.ConfigFreno(uid=uid, cuenta="axioma", cgroup=tmp_path / "cgroup", proc=tmp_path / "proc",
                         llave=Path("/etc/jax-ejecutor/freno/id_freno"),
                         known_hosts=Path("/etc/jax-ejecutor/freno/known_hosts"), remotos=tuple(remotos),
                         estado=tmp_path / "estado.json")


def _proc(tmp_path, procesos):
    for pid, (real, efectivo) in procesos.items():
        d = tmp_path / "proc" / str(pid)
        d.mkdir(parents=True)
        (d / "status").write_text(f"Name:\tx\nUid:\t{real}\t{efectivo}\t{efectivo}\t{efectivo}\n")
    (tmp_path / "proc" / "self").mkdir(parents=True, exist_ok=True)


def test_cgroup_kill_escribe_1_si_hay_slice(tmp_path):
    ruta = F.ruta_cgroup_kill(tmp_path / "cgroup", 1001)
    assert ruta == tmp_path / "cgroup" / "user.slice" / "user-1001.slice" / "cgroup.kill"
    assert F.matar_cgroup(tmp_path / "cgroup", 1001) is False
    ruta.parent.mkdir(parents=True)
    ruta.write_text("")
    assert F.matar_cgroup(tmp_path / "cgroup", 1001) is True
    assert ruta.read_text() == "1"


def test_pids_por_uid_real_o_efectivo(tmp_path):
    _proc(tmp_path, {10: (1001, 1001), 11: (0, 1001), 12: (1001, 0), 13: (1000, 1000)})
    assert sorted(F.pids_de(tmp_path / "proc", 1001)) == [10, 11, 12]


def test_pids_tolera_un_proceso_que_desaparece(tmp_path):
    _proc(tmp_path, {10: (1001, 1001)})
    (tmp_path / "proc" / "11").mkdir()  # sin status: murió entre listar y leer
    assert F.pids_de(tmp_path / "proc", 1001) == [10]


def test_matar_pids_tolera_los_que_ya_murieron():
    vistos = []

    def matar(pid, sig):
        vistos.append((pid, sig))
        if pid == 2:
            raise ProcessLookupError

    assert F.matar_pids([1, 2, 3], matar) == 2
    assert vistos == [(1, signal.SIGKILL), (2, signal.SIGKILL), (3, signal.SIGKILL)]


def test_argv_del_barrido_sin_comando_y_con_llave_propia(tmp_path):
    argv = F.argv_barrido(F.Remoto("bridge", "192.0.2.20", 58291), _cfg(tmp_path))
    assert argv[0] == "ssh" and argv[-1] == "axioma@192.0.2.20"
    assert ["-i", "/etc/jax-ejecutor/freno/id_freno"] == argv[1:3]
    assert "IdentitiesOnly=yes" in argv and "StrictHostKeyChecking=yes" in argv and "BatchMode=yes" in argv


@pytest.mark.parametrize("rc, salida, esperado", [
    (0, b"freno_remoto=ok quedan=0\n", "ok"),
    (0, b"freno_remoto=ok quedan=2\n", "fallo"),
    (0, b"freno_remoto=ok quedan=00\n", "fallo"),
    (255, b"", "fallo"),
])
def test_barrer_exige_cero_procesos(tmp_path, rc, salida, esperado):
    def correr(argv, **k):
        return subprocess.CompletedProcess(argv, rc, salida, b"")
    assert F.barrer(F.Remoto("bridge", "192.0.2.20", 58291), _cfg(tmp_path), correr) == esperado


def test_barrer_con_tiempo_agotado_o_sin_ssh(tmp_path):
    def vence(argv, **k):
        raise subprocess.TimeoutExpired(argv, 15)

    def sin_ssh(argv, **k):
        raise FileNotFoundError("ssh")
    r = F.Remoto("bridge", "192.0.2.20", 58291)
    assert F.barrer(r, _cfg(tmp_path), vence) == "tiempo_agotado"
    assert F.barrer(r, _cfg(tmp_path), sin_ssh) == "fallo"


class Reloj:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_paso_suelto_no_mata_nada(tmp_path):
    _proc(tmp_path, {10: (1001, 1001)})
    muertos = []
    fr = F.Freno(_cfg(tmp_path), frenos=lambda: (), matar=lambda p, s: muertos.append(p),
                 lanzar_barrido=lambda r: pytest.fail("barrió suelto"))
    assert fr.paso() == {"activo": False}
    assert muertos == []
    estado = json.loads((tmp_path / "estado.json").read_text())
    assert estado["activo"] is False and estado["uid_resuelto"] is True
    assert (tmp_path / "estado.json").stat().st_mode & 0o777 == 0o644, "el arranque lo lee sin ser root"


def test_paso_puesto_mata_local_cada_vuelta_y_barre_en_sus_repasos(tmp_path):
    _proc(tmp_path, {10: (1001, 1001), 13: (1000, 1000)})
    kill = F.ruta_cgroup_kill(tmp_path / "cgroup", 1001)
    kill.parent.mkdir(parents=True)
    kill.write_text("")
    remotos = (F.Remoto("bridge", "192.0.2.20", 58291), F.Remoto("prod", "192.0.2.10", 58291))
    reloj, muertos, barridos = Reloj(), [], []
    fr = F.Freno(_cfg(tmp_path, remotos), frenos=lambda: (F.PAUSA_DEL_EJECUTOR,), reloj=reloj,
                 matar=lambda p, s: muertos.append(p),
                 lanzar_barrido=lambda rs: barridos.append((reloj.t, tuple(r.nombre for r in rs))))
    for delta in (0.0, 0.25, 1.9, 2.1, 5.0, 10.5, 11.0):
        reloj.t = 100.0 + delta
        salida = fr.paso()
    assert muertos == [10] * 7, "el barrido local corre en CADA vuelta"
    assert barridos == [(100.0, ("bridge", "prod")), (102.1, ("bridge", "prod")), (110.5, ("bridge", "prod"))]
    assert salida["frenos"] == [F.PAUSA_DEL_EJECUTOR]
    assert salida["cgroup"] is True and kill.read_text() == "1", "cgroup.kill en cada vuelta"


def test_al_soltar_y_volver_a_poner_repasa_desde_cero(tmp_path):
    _proc(tmp_path, {})
    reloj, estado, barridos = Reloj(), {"puesto": True}, []
    fr = F.Freno(_cfg(tmp_path, (F.Remoto("bridge", "192.0.2.20", 58291),)),
                 frenos=lambda: (F.INTERRUPTOR_DE_JAX,) if estado["puesto"] else (),
                 reloj=reloj, matar=lambda p, s: None, lanzar_barrido=lambda rs: barridos.append(reloj.t))
    fr.paso()
    estado["puesto"] = False
    reloj.t = 101.0
    fr.paso()
    estado["puesto"] = True
    reloj.t = 102.0
    fr.paso()
    assert barridos == [100.0, 102.0]


def test_sin_uid_no_mata_pero_frena_remoto_y_lo_dice(tmp_path):
    _proc(tmp_path, {10: (1001, 1001)})
    muertos, barridos = [], []
    fr = F.Freno(_cfg(tmp_path, (F.Remoto("bridge", "192.0.2.20", 58291),), uid=None),
                 frenos=lambda: (F.INTERRUPTOR_DE_JAX,), matar=lambda p, s: muertos.append(p),
                 lanzar_barrido=lambda rs: barridos.append(rs))
    fr.paso()
    assert muertos == [] and len(barridos) == 1
    assert json.loads((tmp_path / "estado.json").read_text())["uid_resuelto"] is False


# --- los dos frenos: interruptor de JAX y pausa del Ejecutor ------------------

@pytest.fixture
def rutas(tmp_path, monkeypatch):
    d = tmp_path / "interruptor"
    d.mkdir()
    monkeypatch.setenv(interruptor.VARIABLE_RUTA, str(d / "PAUSE"))
    monkeypatch.setenv(P.VARIABLE_RUTA, str(d / "EJECUTOR_PAUSA"))
    return d


def test_frenos_sueltos(rutas):
    assert F.frenos_puestos() == ()


def test_frenos_con_el_interruptor(rutas):
    (rutas / "PAUSE").write_text("{}")
    assert F.frenos_puestos() == (F.INTERRUPTOR_DE_JAX,)


def test_frenos_con_la_pausa_del_ejecutor(rutas):
    P.poner_pausa(rutas / "EJECUTOR_PAUSA", {"origen": "c5"})
    assert F.frenos_puestos() == (F.PAUSA_DEL_EJECUTOR,)


def test_frenos_con_las_dos(rutas):
    (rutas / "PAUSE").write_text("{}")
    P.poner_pausa(rutas / "EJECUTOR_PAUSA", {"origen": "c5"})
    assert F.frenos_puestos() == (F.INTERRUPTOR_DE_JAX, F.PAUSA_DEL_EJECUTOR)


@pytest.mark.parametrize("variable", [interruptor.VARIABLE_RUTA, P.VARIABLE_RUTA])
def test_sin_configurar_cuenta_como_puesto(rutas, monkeypatch, variable):
    monkeypatch.delenv(variable)
    assert F.frenos_puestos() != ()


# --- configuración ------------------------------------------------------------

def _env(tmp_path, **extra):
    env = {"JAX_EJECUTOR_CUENTA": "no-existe-esta-cuenta", "JAX_EJECUTOR_FRENO_LLAVE": "/k",
           "JAX_EJECUTOR_FRENO_KNOWN_HOSTS": "/kh", "JAX_EJECUTOR_POLITICA": str(tmp_path / "no-hay.json"),
           "JAX_EJECUTOR_FRENO_ESTADO": str(tmp_path / "estado.json"), "JAX_EJECUTOR_ADMIN_USUARIO": "root"}
    env.update(extra)
    return env


def test_config_sin_politica_legible_sigue_matando_local(tmp_path):
    cfg, remotos_ok = F.config_desde_entorno(_env(tmp_path))
    assert cfg.remotos == () and remotos_ok is False and cfg.uid is None


def _uid_de_la_cuenta(monkeypatch, uid):
    class _Pw:
        pw_uid = uid
    monkeypatch.setattr(F.pwd, "getpwnam", lambda nombre: _Pw)


@pytest.mark.parametrize("uid", [0, 999, 4242])
def test_config_se_niega_a_matar_root_sistema_o_administrador(tmp_path, monkeypatch, uid):
    _uid_de_la_cuenta(monkeypatch, uid)
    monkeypatch.setattr(F, "_uid_del_administrador", lambda nombre: 4242)
    cfg, _ = F.config_desde_entorno(_env(tmp_path))
    assert cfg.uid is None


def test_config_con_uid_valido(tmp_path, monkeypatch):
    _uid_de_la_cuenta(monkeypatch, 1001)
    monkeypatch.setattr(F, "_uid_del_administrador", lambda nombre: 1000)
    cfg, _ = F.config_desde_entorno(_env(tmp_path))
    assert cfg.uid == 1001


class _H:
    def __init__(self, nombre, es_local):
        self.nombre, self.ip, self.puerto, self.es_local = nombre, f"192.0.2.{len(nombre)}", 58291, es_local


class _Pol:
    hosts = (_H("hall9000", True), _H("atemai", False), _H("bridge", False))


@pytest.mark.parametrize("habilitados, nombres, cargados", [
    (None, (), False),                       # variable ausente: ningún remoto, y se dice
    ("", (), False),                         # la llave del freno no está en ninguna remota todavía
    ("atemai", ("atemai",), False),          # falta bridge
    ("atemai,bridge", ("atemai", "bridge"), True),
    ("atemai,bridge,fantasma", ("atemai", "bridge"), False),  # un nombre que no está en la política
])
def test_remotos_habilitados_a_proposito(tmp_path, monkeypatch, habilitados, nombres, cargados):
    """Sin la llave del freno instalada en la remota, barrer sería un intento de ssh fallido
    cada vez que se frena (y un fail2ban podría banear al controlador). Se habilitan por nombre;
    `remotos_cargados` sólo es true con TODAS las remotas de la política habilitadas."""
    monkeypatch.setattr(F.politica, "cargar", lambda ruta, uid_de_la_cuenta: _Pol)
    _uid_de_la_cuenta(monkeypatch, 1001)
    monkeypatch.setattr(F, "_uid_del_administrador", lambda nombre: 1000)
    extra = {} if habilitados is None else {"JAX_EJECUTOR_FRENO_REMOTOS": habilitados}
    cfg, ok = F.config_desde_entorno(_env(tmp_path, **extra))
    assert tuple(r.nombre for r in cfg.remotos) == nombres and ok is cargados
