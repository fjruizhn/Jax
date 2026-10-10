"""El Faro, paso 0.6 / R-2: la unidad de systemd (cgroup) por ejecucion.

Cada jaula corre dentro de un SERVICIO TRANSITORIO del user manager, `faro-jaula-u<uid>.service`, con
`MemoryMax`, `CPUQuota` y `TasksMax` FINITOS y exigidos por configuracion (sin los tres, no hay
lanzamiento), `Type=exec`, `KillMode=control-group`, `OOMPolicy=kill`, `--pipe` y `--collect`. La unidad
es determinista por uid: systemd arbitra atomicamente dos arranques concurrentes y un reinicio del Faro
puede recuperar el estado consultando por su nombre.

El estado consultado por uid es TRIVALENTE (contrato del plan 2026-10-09-faro-scope-r2):
- LIVE: unidad activa y cgroup poblado (o cgroup poblado aunque la unidad se este apagando);
- EMPTY: systemd confirma que no hay unidad activa y el cgroup ya esta vacio o eliminado;
- UNKNOWN: manager, unidad, ControlGroup, cgroupfs o proceso no permiten concluir.
SOLO EMPTY libera un uid; LIVE y UNKNOWN lo conservan/rechazan (fallo cerrado).

Aqui el gestor se prueba con un doble de systemd-run/systemctl que implementa la mecanica observada en
systemd 259 (medida en hall9000, 2026-10-09): `--value` imprime LoadState/ActiveState/ControlGroup (una
linea por propiedad, vacia si no hay), los limites efectivos se informan normalizados (MemoryMax en bytes,
CPUQuotaPerSecUSec como timespan «500ms», TasksMax entero), `kill` sobre una unidad no cargada sale 1, y
una unidad `--collect` termina en `not-found` con rc 0. La prueba con el user manager REAL (si el runner
lo tiene) esta al final.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from jax.faro.config import ConfigFaroInvalida
from jax.faro.scope import (ConfigScope, EstadoUnidad, ScopeError, ScopeManager, nombre_unidad,
                            tiempospan_a_us)

UID = 50001
UNIDAD = nombre_unidad(UID)


# --------------------------------------------------------------------------- #
# configuracion: los tres limites, finitos y obligatorios                      #
# --------------------------------------------------------------------------- #

ENV = {"JAX_FARO_SCOPE_MEMORY_MAX": "256M", "JAX_FARO_SCOPE_CPU_QUOTA": "200%", "JAX_FARO_SCOPE_TASKS_MAX": "128"}


def test_la_configuracion_sale_del_entorno():
    cfg = ConfigScope.desde_entorno(ENV)
    assert (cfg.memory_max, cfg.cpu_quota, cfg.tasks_max, cfg.slice) == ("256M", "200%", "128", "faro.slice")
    assert cfg.bytes_memoria() == 256 * 1024 ** 2 and cfg.usec_cpu() == 2_000_000 and cfg.tareas() == 128
    assert ConfigScope.desde_entorno({**ENV, "JAX_FARO_SCOPE_SLICE": "faro-carga.slice"}).slice == "faro-carga.slice"


@pytest.mark.parametrize("falta", sorted(ENV))
def test_sin_uno_de_los_tres_limites_no_hay_scope(falta):
    with pytest.raises(ConfigFaroInvalida, match=falta):
        ConfigScope.desde_entorno({k: v for k, v in ENV.items() if k != falta})


@pytest.mark.parametrize("memoria", ["infinity", "inf", "0", "-5", "50%", "1.5G", "", " 8M ", "8X", "8 MB"])
def test_un_memory_max_no_finito_o_no_metrico_no_arranca(memoria):
    with pytest.raises(ConfigFaroInvalida):
        ConfigScope(memory_max=memoria, cpu_quota="200%", tasks_max="128")


@pytest.mark.parametrize("cpu", ["infinity", "0%", "0.0%", "-1%", "2", "200", "dos%", "", "200 %"])
def test_un_cpu_quota_sin_porcentaje_finito_positivo_no_arranca(cpu):
    with pytest.raises(ConfigFaroInvalida):
        ConfigScope(memory_max="256M", cpu_quota=cpu, tasks_max="128")


@pytest.mark.parametrize("tareas", ["infinity", "0", "-1", "32.5", "50%", "4K", "", "x"])
def test_un_tasks_max_no_entero_positivo_no_arranca(tareas):
    with pytest.raises(ConfigFaroInvalida):
        ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max=tareas)


@pytest.mark.parametrize("slice_", ["faro", "faro.service", "faro slice", "../evil.slice", "", "a/b.slice"])
def test_un_slice_que_no_es_un_nombre_de_slice_no_arranca(slice_):
    with pytest.raises(ConfigFaroInvalida):
        ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128", slice=slice_)


def test_los_sufijos_binarios_y_porcentajes_se_resuelven_y_tasksmax_es_entero():
    cfg = ConfigScope(memory_max="1G", cpu_quota="7%", tasks_max="4096")
    assert cfg.bytes_memoria() == 1024 ** 3 and cfg.usec_cpu() == 70_000 and cfg.tareas() == 4096
    assert ConfigScope(memory_max="512", cpu_quota="0.5%", tasks_max="7").bytes_memoria() == 512


@pytest.mark.parametrize("texto,us", [("500ms", 500_000), ("2s", 2_000_000), ("70ms", 70_000),
                                      ("1min 30s", 90_000_000), ("5ms", 5_000), ("100us", 100), ("1.5s", 1_500_000)])
def test_el_timespan_de_systemd_se_resuelve_a_microsegundos(texto, us):
    assert tiempospan_a_us(texto) == us


@pytest.mark.parametrize("texto", ["infinity", "500 ms", "abc", "", "5zz", "ms"])
def test_un_timespan_ilegible_no_pasa_por_verificacion(texto):
    with pytest.raises(ValueError):
        tiempospan_a_us(texto)


def test_el_nombre_de_unidad_es_determinista_por_uid_y_no_admite_inyeccion():
    assert nombre_unidad(50001) == "faro-jaula-u50001.service"
    for uid in (0, -1, 50001.0, True, "50001;evil", None):
        with pytest.raises(ConfigFaroInvalida):
            nombre_unidad(uid)


# --------------------------------------------------------------------------- #
# el doble de systemd: user manager de mentira sobre un tmp_path              #
# --------------------------------------------------------------------------- #

class ProcFalso:
    """El proceso de `systemd-run` (con `--pipe` vive mientras la unidad) o el de una consulta `systemctl`.
    `rc` es el codigo final del proceso; para un systemd-run ARRANCADO, mientras la unidad viva el
    returncode es None y al morir pasa a ser el de la unidad (255 si fue SIGKILL, 1 si nunca existio)."""

    def __init__(self, sistema, argv, *, rc=0, stdout=b"", stderr=b""):
        self._sistema, self._argv, self._stdout, self._stderr = sistema, argv, stdout, stderr
        self._rc = rc

    @property
    def returncode(self) -> int | None:
        if self._argv[0].endswith("systemd-run") and self._rc == 0:
            unidad = self._sistema.unidad_de_argv(self._argv)
            if unidad is not None and self._sistema.viva(unidad):
                return None                            # --pipe: no termina antes que el servicio
            return self._sistema.rc_final(unidad)
        return self._rc

    async def communicate(self):
        if self._argv[0].endswith("systemd-run"):
            if self.returncode is None:
                espera = asyncio.get_running_loop().create_future()
                await espera                           # como --pipe: termina cuando la unidad muera
            return b"", self._stderr
        return self._stdout, self._stderr


class SistemaDoble:
    """El user manager de mentira: unidades con LoadState/ActiveState, ControlGroup, limites efectivos y
    `cgroup.events` reales en `raiz` (un directorio normal, no cgroupfs). Normaliza los limites igual que
    systemd: MemoryMax en bytes, CPUQuotaPerSecUSec como timespan, TasksMax entero."""

    def __init__(self, raiz: Path, *, cfg: ConfigScope | None = None):
        self.cfg = cfg or ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128")
        self.raiz = raiz
        self.raiz.mkdir(parents=True, exist_ok=True)
        self.unidades: dict[str, dict] = {}
        self.arrancados: list[tuple[str, tuple[str, ...]]] = []       # unidades que NACIERON
        self.intentos: list[str] = []                                  # incluidos los rechazados por colision
        self.muertes: list[str] = []
        self.fallar_arranque: object | None = None       # Exception | "rc"
        self.fallar_show: str | None = None              # "exec" | "rc" | "basura" | "corta"
        self.kw: dict | None = None

    # -- helpers del doble -----------------------------------------------------------------
    def unidad_de_argv(self, argv) -> str | None:
        for a in argv:
            if a.startswith("--unit="):
                return a.split("=", 1)[1]
        return None

    def viva(self, unidad: str | None) -> bool:
        u = self.unidades.get(unidad or "")
        return bool(u) and not (u["activa"] is False) and u["poblada"] == 1

    def rc_final(self, unidad: str | None) -> int | None:
        u = self.unidades.get(unidad or "")
        return u.get("rc") if u else 1

    def _cgroup(self, unidad: str) -> Path:
        return self.raiz / unidad.removesuffix(".service")

    def _parsear_run(self, argv):
        propiedades: dict[str, str] = {}
        unidad, comando = None, []
        for i, a in enumerate(argv):
            if i == 0:
                continue
            if a.startswith("--unit="):
                unidad = a.split("=", 1)[1]
            elif a.startswith("--property="):
                k, v = a.split("=", 1)[1].split("=", 1)
                propiedades[k] = v
            elif not a.startswith("-") and unidad is not None:
                comando = list(argv[i:])
                break
        return unidad, propiedades, comando

    def _crear(self, unidad: str, comando, propiedades: dict[str, str]) -> int:
        self.intentos.append(unidad)
        if unidad in self.unidades:
            return 1                                        # medido: la unidad ya existe, el arranque falla
        efectivas = dict(propiedades)
        if "MemoryMax" in propiedades:
            efectivas["MemoryMax"] = str(self.cfg.bytes_memoria() if propiedades["MemoryMax"] == self.cfg.memory_max
                                         else int(propiedades["MemoryMax"]))
        if "CPUQuota" in propiedades:
            efectivas["CPUQuotaPerSecUSec"] = _timespan(self.cfg.usec_cpu())
            efectivas.pop("CPUQuota")
        if "TasksMax" in propiedades:
            efectivas["TasksMax"] = str(self.cfg.tareas())
        cg = self._cgroup(unidad)
        cg.mkdir(parents=True, exist_ok=True)
        (cg / "cgroup.events").write_text("populated 1\nfrozen 0\n")
        self.unidades[unidad] = {"load": "loaded", "activa": True, "poblada": 1,
                                 "cg": str(cg.relative_to(self.raiz)), "propiedades": efectivas, "rc": None}
        self.arrancados.append((unidad, tuple(comando)))
        return 0

    def matar_unidad(self, unidad: str) -> int:
        self.muertes.append(unidad)
        u = self.unidades.get(unidad)
        if u is None:
            return 1                                        # medido: kill de una unidad no cargada sale 1
        (self._cgroup(unidad) / "cgroup.events").write_text("populated 0\nfrozen 0\n")
        u.update(activa=False, rc=255)                      # SIGKILL --kill-whom=all a todo el cgroup
        return 0

    def recolectar(self, unidad: str) -> None:              # --collect: descarga la unidad muerta
        if self.unidades.pop(unidad, None) is not None:
            shutil.rmtree(self._cgroup(unidad), ignore_errors=True)

    # -- proceso de mentira ----------------------------------------------------------------
    async def crear_proceso(self, *argv, **kw):
        self.kw = kw
        if argv[0].endswith("systemd-run"):
            if self.fallar_arranque is not None:
                if isinstance(self.fallar_arranque, BaseException):
                    raise self.fallar_arranque
                return ProcFalso(self, argv, rc=1, stderr=b"Failed to start transient service unit\n")
            unidad, propiedades, comando = self._parsear_run(argv)
            rc = self._crear(unidad, comando, propiedades)
            if rc != 0:
                return ProcFalso(self, argv, rc=rc, stderr=b"Unit ... already exists\n")
            return ProcFalso(self, argv)
        if list(argv[1:3]) == ["--user", "show"]:
            rc, salida, err = self._show(argv)
            return ProcFalso(self, argv, rc=rc, stdout=salida, stderr=err)
        if list(argv[1:3]) == ["--user", "kill"]:
            for a in argv:
                if not a.startswith("-") and a.endswith(".service"):
                    return ProcFalso(self, argv, rc=self.matar_unidad(a))
        raise AssertionError(f"comando inesperado: {argv}")

    def _show(self, argv) -> tuple[int, bytes, bytes]:
        unidad = next(a for a in argv if not a.startswith("-") and a.endswith(".service"))
        nombres = [a.split("=", 1)[1] for a in argv if a.startswith("--property=")]
        if self.fallar_show == "exec":
            raise OSError("systemctl no arranca")
        if self.fallar_show == "rc":
            return 1, b"", b"Failed to connect to bus\n"
        if self.fallar_show == "basura":
            return 0, b"esto no son\npropiedades\n", b""
        u = self.unidades.get(unidad)
        if u is None:
            valores = {"LoadState": "not-found", "ActiveState": "inactive", "ControlGroup": ""}
        else:
            activo = u["activa"]
            valores = {"LoadState": u["load"], "ActiveState": activo if isinstance(activo, str)
                       else ("active" if activo else "inactive"), "ControlGroup": u["cg"]}
        if self.fallar_show == "corta":                     # a ControlGroup se lo trajo un gato
            valores.pop("ControlGroup")
        for nombre in nombres:
            if nombre in ("MemoryMax", "CPUQuotaPerSecUSec", "TasksMax"):
                valores[nombre] = u["propiedades"].get(nombre) if u else \
                    {"MemoryMax": "infinity", "CPUQuotaPerSecUSec": "infinity", "TasksMax": "96548"}[nombre]
            if nombre == "Description":
                valores[nombre] = u["propiedades"].get("Description", "") if u else ""
        # como systemd: clave=valor (el orden interno del manager no es el pedido)
        return 0, ("\n".join(f"{k}={v}" for k, v in valores.items() if k in nombres) + "\n").encode(), b""


def _timespan(us: int) -> str:
    return f"{us // 1_000_000}s" if us % 1_000_000 == 0 else f"{us // 1000}ms"


@pytest.fixture
def sistema(tmp_path):
    return SistemaDoble(tmp_path / "cgroup")


def gestor(sistema, *, cfg=None, **kw) -> ScopeManager:
    return ScopeManager(cfg or sistema.cfg, crear_proceso=sistema.crear_proceso, raiz_cgroup=sistema.raiz,
                        intervalo_s=0.01, **kw)


def _poblar(sistema, uid=UID, **estado):
    unidad = nombre_unidad(uid)
    cg = sistema._cgroup(unidad)
    cg.mkdir(parents=True, exist_ok=True)
    (cg / "cgroup.events").write_text(f"populated {estado.get('poblada', 1)}\nfrozen 0\n")
    sistema.unidades[unidad] = {"load": estado.get("load", "loaded"), "activa": estado.get("activa", True),
                                "poblada": estado.get("poblada", 1), "cg": str(cg.relative_to(sistema.raiz)),
                                "propiedades": {}, "rc": estado.get("rc", 0)}


def correr(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# los argumentos exactos de systemd-run y systemctl                           #
# --------------------------------------------------------------------------- #

def test_el_argv_de_arranque_es_el_del_diseno_sin_defaults_ni_sorpresas():
    g = ScopeManager(ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128", slice="faro.slice"))
    argv = g.argv_arrancar(UID, ["/usr/bin/sudo", "-n", "-u", "#50001", "--", "/usr/bin/bwrap", "--dev", "/dev"],
                           intento="abc123")
    assert argv[0] == "/usr/bin/systemd-run"
    assert argv[1:6] == ["--user", "--quiet", "--pipe", "--collect", "--unit=faro-jaula-u50001.service"]
    assert "--slice=faro.slice" in argv
    for propiedad in ("Type=exec", "KillMode=control-group", "OOMPolicy=kill",
                      "Description=FARO-R2-abc123",
                      "MemoryMax=256M", "CPUQuota=200%", "TasksMax=128"):
        assert f"--property={propiedad}" in argv, propiedad
    # ALLOWLIST CERRADA (MAJOR 3): el comando corre bajo `env -i` — nada del entorno del manager
    assert argv[-10:] == ["/usr/bin/env", "-i", "/usr/bin/sudo", "-n", "-u", "#50001",
                          "--", "/usr/bin/bwrap", "--dev", "/dev"]
    texto = " ".join(argv)
    assert "MemoryMax=infinity" not in texto and "--scope" not in texto and "--system" not in texto
    assert "--setenv" not in texto and "XDG_RUNTIME_DIR" not in texto


@pytest.mark.parametrize("comando", [[], "bwrap", ["/usr/bin/bwrap", "x\x00y"], ["relativo/bwrap"], [5]])
def test_el_argv_de_arranque_no_admite_un_comando_vacio_o_de_tipo_raro(comando):
    g = ScopeManager(ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128"))
    with pytest.raises(ConfigFaroInvalida):
        g.argv_arrancar(UID, comando, intento="abc123")


@pytest.mark.parametrize("intento", [None, "", "con espacio", "con\x00nul"])
def test_el_argv_de_arranque_exige_un_id_de_intento_usable(intento):
    g = ScopeManager(ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128"))
    with pytest.raises(ConfigFaroInvalida):
        g.argv_arrancar(UID, ["/usr/bin/true"], intento=intento)


def test_el_argv_de_kill_es_sigkill_a_toda_la_unidad():
    argv = ScopeManager(ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128")).argv_matar(UID)
    assert argv == ["/usr/bin/systemctl", "--user", "kill", "faro-jaula-u50001.service",
                    "--kill-whom=all", "--signal=SIGKILL"]


# --------------------------------------------------------------------------- #
# el trivalor: solo EMPTY libera                                              #
# --------------------------------------------------------------------------- #

def test_activa_y_poblada_es_live(sistema):
    _poblar(sistema, poblada=1)
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.LIVE


def test_not_found_es_empty_porque_systemd_confirmo_que_no_existe(sistema):
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.EMPTY


def test_muerta_y_recolectada_es_empty(sistema):
    _poblar(sistema)
    sistema.matar_unidad(UNIDAD)
    sistema.recolectar(UNIDAD)
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.EMPTY


def test_inactiva_con_cgroup_vacio_es_empty(sistema):
    _poblar(sistema, activa=False, poblada=0)
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.EMPTY


def test_inactiva_con_el_directorio_del_cgroup_ya_eliminado_es_empty(sistema):
    _poblar(sistema, activa=False, poblada=0)
    shutil.rmtree(sistema._cgroup(UNIDAD))
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.EMPTY


def test_inactiva_pero_con_procesos_en_el_cgroup_es_live_el_uid_no_se_libera(sistema):
    """El caso R-2: murio el proceso principal (o el cliente del Faro), quedan hijos/nietos en el cgroup."""
    _poblar(sistema, activa=False, poblada=1)
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.LIVE


def test_desactivandose_con_procesos_en_el_cgroup_es_live(sistema):
    _poblar(sistema, activa="deactivating", poblada=1)
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.LIVE


def test_desactivandose_y_vacia_es_empty(sistema):
    _poblar(sistema, activa="deactivating", poblada=0)
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.EMPTY


def test_activa_con_cgroup_vacio_es_unknown(sistema):
    _poblar(sistema, activa=True, poblada=0)
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.UNKNOWN


def test_activando_con_cgroup_vacio_es_unknown_nunca_empty(sistema):
    _poblar(sistema, activa="activating", poblada=0)
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.UNKNOWN


def test_activa_sin_control_group_es_unknown(sistema):
    _poblar(sistema, activa=True)
    sistema.unidades[UNIDAD]["cg"] = ""
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.UNKNOWN


def test_un_load_state_raro_es_unknown(sistema):
    _poblar(sistema)
    sistema.unidades[UNIDAD]["load"] = "masked"
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.UNKNOWN


def test_un_active_state_raro_es_unknown(sistema):
    _poblar(sistema, activa="mantenimiento")
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.UNKNOWN


@pytest.mark.parametrize("fallo", ["exec", "rc", "basura", "corta"])
def test_una_consulta_de_systemctl_rota_o_ilegible_es_unknown(sistema, fallo):
    _poblar(sistema)
    sistema.fallar_show = fallo
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.UNKNOWN


def test_un_cgroup_events_ilegible_es_unknown(sistema):
    _poblar(sistema)
    (sistema._cgroup(UNIDAD) / "cgroup.events").write_text("sin formato\n")
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.UNKNOWN


def test_un_cgroup_events_inaccesible_es_unknown(sistema):
    if os.geteuid() == 0:
        pytest.skip("root lee hasta un 0000")
    _poblar(sistema)
    eventos = sistema._cgroup(UNIDAD) / "cgroup.events"
    eventos.chmod(0o000)
    try:
        assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.UNKNOWN
    finally:
        eventos.chmod(0o644)


def test_el_control_group_se_le_de_systemd_nunca_se_concatena(sistema, monkeypatch):
    """Sin consulta a systemd no hay estado: el gestor no deduce la ruta del cgroup por su cuenta."""
    _poblar(sistema)
    g = gestor(sistema)
    consultas = []
    original = g._consultar

    async def espiar(unidad, propiedades):
        consultas.append((unidad, propiedades))
        return await original(unidad, propiedades)
    monkeypatch.setattr(g, "_consultar", espiar)
    assert correr(g.estado(UID)) is EstadoUnidad.LIVE
    assert consultas[0][0] == UNIDAD and "ControlGroup" in consultas[0][1]


def test_control_group_devuelve_la_ruta_que_reporta_systemd(sistema):
    _poblar(sistema)

    async def caso():
        return await gestor(sistema).control_group(UID)
    assert correr(caso()) == sistema.unidades[UNIDAD]["cg"]


# --------------------------------------------------------------------------- #
# arrancar: unidad verificada (activa, con los tres limites) antes de lista   #
# --------------------------------------------------------------------------- #

def test_arrancar_arranca_la_unidad_del_uid_con_el_comando_y_la_da_por_lista(sistema):
    async def caso():
        g = gestor(sistema)
        proceso = await g.arrancar(UID, ["/usr/bin/sudo", "-n", "-u", "#50001", "--", "/usr/bin/bwrap"])
        try:
            # el comando de la unidad corre bajo env -i (allowlist cerrada) con la elevacion dentro
            assert sistema.arrancados == [(UNIDAD, ("/usr/bin/env", "-i", "/usr/bin/sudo", "-n", "-u",
                                                    "#50001", "--", "/usr/bin/bwrap"))]
            assert await g.estado(UID) is EstadoUnidad.LIVE
            env = sistema.kw.get("env")
            # entorno minimo: PATH y, como mucho, las dos variables que ubican el bus del user manager
            assert env.get("PATH") == "/usr/bin:/bin" and set(env) <= {"PATH", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"}
            return proceso.returncode
        finally:
            await g.matar(UID)
    assert correr(caso()) is None                                          # --pipe: vive mientras la unidad viva


def test_arrancar_deja_los_tres_limites_efectivos_en_la_unidad(sistema):
    async def caso():
        await gestor(sistema).arrancar(UID, ["/usr/bin/true"])
        p = sistema.unidades[UNIDAD]["propiedades"]
        return p["MemoryMax"], p["CPUQuotaPerSecUSec"], p["TasksMax"]
    assert correr(caso()) == (str(256 * 1024 ** 2), "2s", "128")       # normalizado como systemd (bytes/timespan)


def test_arrancar_verifica_con_show_antes_de_dar_por_lista_la_unidad(sistema):
    """La verificacion lee ActiveState + ControlGroup + los tres limites de systemd, no lo que supone."""
    async def caso():
        g = gestor(sistema)
        shows = []
        original = sistema.crear_proceso

        async def proceso(*argv, **kw):
            if list(argv[1:3]) == ["--user", "show"]:
                shows.append(argv)
            return await original(*argv, **kw)
        g._crear_proceso = proceso
        await g.arrancar(UID, ["/usr/bin/true"])
        props = " ".join(a for a in shows[0] if a.startswith("--property="))
        for esperada in ("ActiveState", "ControlGroup", "Description", "MemoryMax", "CPUQuotaPerSecUSec", "TasksMax"):
            assert esperada in props, esperada
        assert "--value" not in shows[0]                    # clave=valor: el orden interno de systemd no es el pedido
    correr(caso())


def test_arrancar_con_la_unidad_ocupada_por_otro_falla_y_no_la_toca(sistema):
    async def caso():
        _poblar(sistema)                                       # ya hay una unidad viva con ese uid
        g = gestor(sistema)
        try:
            await g.arrancar(UID, ["/usr/bin/true"])
        except ScopeError as e:
            assert UNIDAD in str(e)
            return await g.estado(UID), sistema.muertes
        raise AssertionError("tenia que fallar: la unidad ya existe")
    estado, muertes = correr(caso())
    assert estado is EstadoUnidad.LIVE                 # la original sigue viva: NO se mata la ajena
    assert muertes == []


def test_block1_un_perdedor_tardio_no_adopta_la_unidad_del_ganador_ni_la_mata(sistema):
    """El caso exacto del BLOCK 1: nuestro systemd-run aun figura vivo (returncode None para siempre,
    como un rechazo que no llego al child watcher), la unidad activa es la del GANADOR (otra
    Description). La identidad por intento — no la temporizacion — decide: fallo, y el kill con guard
    deja al ganador intacto."""
    async def caso():
        _poblar(sistema, activa=True, poblada=1)
        sistema.unidades[UNIDAD]["propiedades"]["Description"] = "FARO-R2-ganador"
        sistema.unidades[UNIDAD]["propiedades"].update({"MemoryMax": str(sistema.cfg.bytes_memoria()),
                                                        "CPUQuotaPerSecUSec": _timespan(sistema.cfg.usec_cpu()),
                                                        "TasksMax": str(sistema.cfg.tareas())})
        g = gestor(sistema)

        async def proceso_tardio(*argv, **kw):
            p = await sistema.crear_proceso(*argv, **kw)
            if argv[0].endswith("systemd-run") and isinstance(p, ProcFalso):
                p._rc = 0                                      # vive para siempre: el rechazo nunca llega
            return p
        g._crear_proceso = proceso_tardio
        try:
            await g.arrancar(UID, ["/usr/bin/true"])
        except ScopeError as e:
            assert "otro intento" in str(e)
            return await g.estado(UID), sistema.muertes
        raise AssertionError("tenia que fallar por identidad ajena")
    estado, muertes = correr(caso())
    assert estado is EstadoUnidad.LIVE                 # la unidad del ganador sigue viva
    assert muertes == []                               # y jamas recibio nuestro kill


def test_major2_cancelada_la_creacion_del_subproceso_la_unidad_nacida_se_limpia(sistema):
    """MAJOR 2: la cancelacion llega MIENTRAS _crear_proceso esta en vuelo, despues de que el doble ya
    creo la unidad. La limpieza acotada la encuentra (Description nuestra) y la mata; nunca queda viva
    sin seguimiento."""
    async def caso():
        g = gestor(sistema)
        adentro = asyncio.Event()

        async def crear_lento(*argv, **kw):
            p = await sistema.crear_proceso(*argv, **kw)       # la unidad YA nacio con NUESTRA Description
            if argv[0].endswith("systemd-run"):
                adentro.set()
                await asyncio.Event().wait()                   # bloqueado antes de devolver el handle
            return p
        g._crear_proceso = crear_lento
        tarea = asyncio.create_task(g.arrancar(UID, ["/usr/bin/true"]))
        await asyncio.wait_for(adentro.wait(), 2)
        tarea.cancel()
        try:
            await tarea
        except asyncio.CancelledError:
            pass
        return sistema.muertes, await g.estado(UID)
    muertes, estado = corre_con_plazo(caso())
    assert muertes == [UNIDAD]                         # la unidad nacida de nuestro intento, muerta
    assert estado is EstadoUnidad.EMPTY


def corre_con_plazo(coro):
    async def envolver():
        return await asyncio.wait_for(coro, 10)
    return asyncio.run(envolver())


@pytest.mark.parametrize("romper", ["MemoryMax", "CPUQuotaPerSecUSec", "TasksMax"])
def test_arrancar_con_un_limite_efectivo_distinto_o_infinity_mata_y_falla_cerrado(sistema, romper):
    async def caso():
        g = gestor(sistema)
        original = sistema.crear_proceso

        async def proceso(*argv, **kw):
            p = await original(*argv, **kw)
            if argv[0].endswith("systemd-run"):
                u = sistema.unidades.get(UNIDAD)
                if u is not None:
                    u["propiedades"][romper] = "infinity"      # systemd informo un limite que no es el pedido
            return p
        g._crear_proceso = proceso
        try:
            await g.arrancar(UID, ["/usr/bin/true"])
        except ScopeError:
            return sistema.muertes, await g.estado(UID)
        raise AssertionError("tenia que fallar la verificacion")
    muertes, estado = correr(caso())
    assert muertes == [UNIDAD]                                 # no se deja una unidad sin verificar
    assert estado is EstadoUnidad.EMPTY                        # matada y vacia: liberable


def test_arrancar_que_no_logra_unidad_activa_en_el_plazo_mata_lo_que_haya_y_falla(sistema):
    cfg = ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128", plazo_arranque_s=0.15)

    async def caso():
        g = gestor(sistema, cfg=cfg)
        original = sistema.crear_proceso

        async def proceso(*argv, **kw):
            p = await original(*argv, **kw)
            if argv[0].endswith("systemd-run"):
                sistema.unidades[UNIDAD]["activa"] = False     # la unidad nunca llega a estar activa
            return p
        g._crear_proceso = proceso
        try:
            await g.arrancar(UID, ["/usr/bin/true"])
        except ScopeError:
            return sistema.intentos, sistema.muertes, await g.estado(UID)
        raise AssertionError("tenia que fallar por plazo")
    intentos, muertes, estado = correr(caso())
    assert intentos == [UNIDAD] and muertes == [UNIDAD]        # lo arrancado a medias se mata igual
    assert estado is EstadoUnidad.EMPTY


def test_arrancar_con_systemd_run_roto_no_arranca_nada_y_falla(sistema):
    sistema.fallar_arranque = "rc"

    async def caso():
        try:
            await gestor(sistema).arrancar(UID, ["/usr/bin/true"])
        except ScopeError:
            return sistema.arrancados
        raise AssertionError("tenia que fallar")
    assert correr(caso()) == []


def test_arrancar_cancelado_durante_la_verificacion_mata_la_unidad(sistema):
    async def caso():
        g = gestor(sistema, cfg=ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128",
                                            plazo_arranque_s=5.0))
        original = sistema.crear_proceso

        async def proceso(*argv, **kw):
            p = await original(*argv, **kw)
            if argv[0].endswith("systemd-run"):
                sistema.unidades[UNIDAD]["activa"] = False     # la verificacion no concluye: queda esperando
            return p
        g._crear_proceso = proceso
        tarea = asyncio.create_task(g.arrancar(UID, ["/usr/bin/true"]))
        await asyncio.sleep(0.05)                              # ya arranco, esta verificando
        tarea.cancel()
        try:
            await tarea
        except asyncio.CancelledError:
            await asyncio.sleep(0.02)
            return sistema.muertes, sistema.arrancados
        raise AssertionError("tenia que cancelarse")
    muertes, arrancados = correr(caso())
    assert arrancados and muertes == [UNIDAD]                  # lo cancelado no queda vivo ni verificado


def test_dos_arranques_concurrentes_con_el_mismo_uid_dejan_una_sola_unidad(sistema):
    async def caso():
        g = gestor(sistema)
        resultados = await asyncio.gather(*(g.arrancar(UID, ["/usr/bin/true"]) for _ in range(2)),
                                          return_exceptions=True)
        return ([r for r in resultados if not isinstance(r, BaseException)],
                [r for r in resultados if isinstance(r, ScopeError)],
                sistema.intentos, sistema.arrancados, sistema.muertes, await g.estado(UID))
    vivas, errores, intentos, arrancados, muertes, estado = correr(caso())
    assert len(vivas) == 1 and len(errores) == 1               # systemd arbitro: exactamente uno gana
    assert len(intentos) == 2 and len(arrancados) == 1
    assert muertes == []                                       # al perdedor no se le mata la unidad al ganador
    assert estado is EstadoUnidad.LIVE


# --------------------------------------------------------------------------- #
# matar y esperar_vacio                                                        #
# --------------------------------------------------------------------------- #

def test_matar_manda_sigkill_a_todos_los_procesos_de_la_unidad(sistema):
    _poblar(sistema)

    async def caso():
        await gestor(sistema).matar(UID)
        return sistema.muertes
    assert correr(caso()) == [UNIDAD]


def test_matar_una_unidad_que_no_existe_es_un_error_visible(sistema):
    async def caso():
        try:
            await gestor(sistema).matar(UID)
        except ScopeError:
            return True
        return False
    assert correr(caso())


def test_matar_a_una_unidad_no_mata_a_su_hermana(sistema):
    _poblar(sistema, UID)
    _poblar(sistema, UID + 1)

    async def caso():
        g = gestor(sistema)
        await g.matar(UID)
        return await g.estado(UID), await g.estado(UID + 1)
    assert correr(caso()) == (EstadoUnidad.EMPTY, EstadoUnidad.LIVE)


def test_esperar_vacio_espera_la_confirmacion_empty(sistema):
    _poblar(sistema, activa=False, poblada=1)                  # muriendo: hijos vivos aun en el cgroup

    async def caso():
        g = gestor(sistema)

        async def vaciar():
            await asyncio.sleep(0.03)
            sistema.unidades[UNIDAD]["poblada"] = 0
            (sistema._cgroup(UNIDAD) / "cgroup.events").write_text("populated 0\nfrozen 0\n")
        asyncio.create_task(vaciar())
        return await g.esperar_vacio(UID, plazo_s=2.0)
    assert correr(caso()) is EstadoUnidad.EMPTY


def test_esperar_vacio_por_plazo_devuelve_lo_que_haya_y_nunca_empty(sistema):
    _poblar(sistema, activa=False, poblada=1)

    async def caso():
        return await gestor(sistema).esperar_vacio(UID, plazo_s=0.05)
    assert correr(caso()) is EstadoUnidad.LIVE


def test_esperar_vacio_con_consulta_rota_devuelve_unknown(sistema):
    sistema.fallar_show = "rc"

    async def caso():
        return await gestor(sistema).esperar_vacio(UID, plazo_s=0.05)
    assert correr(caso()) is EstadoUnidad.UNKNOWN


def test_esperar_vacio_con_una_consulta_rapida_no_espera_de_mas(sistema):
    async def caso():
        t0 = time.monotonic()
        assert await gestor(sistema).esperar_vacio(UID, plazo_s=5.0) is EstadoUnidad.EMPTY
        return time.monotonic() - t0
    assert correr(caso()) < 1.0


# --------------------------------------------------------------------------- #
# el user manager REAL (si el runner lo tiene; se registra la limitacion)      #
# --------------------------------------------------------------------------- #

def _user_manager_ok() -> bool:
    if shutil.which("systemd-run") is None or shutil.which("systemctl") is None:
        return False
    try:
        r = subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True, timeout=5)
        return r.returncode in (0, 1)                          # running o degraded: el bus responde
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.mark.systemd_integration
@pytest.mark.skipif(not _user_manager_ok(), reason="sin user manager de systemd: limitacion registrada (plan 0.6, paso 3)")
class TestUserManagerReal:
    """Arranque, verificacion y kill REALES del user manager (medido en hall9000, systemd 259).
    Cada unidad usa un uid de prueba unico y se mata al terminar; nada queda instalado."""

    UID_BASE = 50100 + (os.getpid() % 500)
    _siguiente = 0

    def _uid(self) -> int:
        TestUserManagerReal._siguiente += 1
        return self.UID_BASE + TestUserManagerReal._siguiente * 7

    def _gestor(self, **kw) -> ScopeManager:
        return ScopeManager(ConfigScope(memory_max="64M", cpu_quota="10%", tasks_max="32", plazo_arranque_s=15.0, **kw))

    @staticmethod
    async def _limpiar(g, uid):
        with contextlib.suppress(Exception):
            await g.matar(uid)
        await g.esperar_vacio(uid, plazo_s=5.0)

    def test_arranca_verifica_limites_y_mata_incluidos_hijos_y_nietos(self):
        uid, g = self._uid(), self._gestor()

        async def caso():
            with contextlib.suppress(Exception):
                await g.matar(uid)                             # una unidad colgante de otra corrida no florea esto
            proceso = await g.arrancar(uid, ["/bin/sh", "-c", "sleep 300 & sleep 300 & sleep 300 & wait"])
            assert await g.estado(uid) is EstadoUnidad.LIVE
            cg = await g.control_group(uid)
            assert cg and Path("/sys/fs/cgroup", cg.removeprefix("/")).is_dir()
            await g.matar(uid)
            return await g.esperar_vacio(uid, plazo_s=10.0), proceso.returncode
        try:
            estado, rc = correr(caso())
        finally:
            correr(self._limpiar(g, uid))
        assert estado is EstadoUnidad.EMPTY
        assert rc != 0                                         # SIGKILL: ni limpio ni 0

    def test_matar_una_unidad_no_mata_a_la_hermana(self):
        a, b, g = self._uid(), self._uid() + 1000, self._gestor()

        async def caso():
            await g.arrancar(a, ["/bin/sleep", "300"])
            await g.arrancar(b, ["/bin/sleep", "300"])
            await g.matar(a)
            return await g.esperar_vacio(a, plazo_s=10.0), await g.estado(b)
        try:
            vacia, hermana = correr(caso())
        finally:
            correr(self._limpiar(g, b))
            correr(self._limpiar(g, a))
        assert vacia is EstadoUnidad.EMPTY and hermana is EstadoUnidad.LIVE

    def test_un_nieto_con_setsid_sigue_en_el_cgroup_y_muere_con_la_unidad(self):
        uid, g = self._uid(), self._gestor()

        async def caso():
            await g.arrancar(uid, ["/bin/sh", "-c", "setsid /bin/sleep 300 & sleep 300"])
            await asyncio.sleep(0.3)                           # que el nieto nazca
            assert await g.estado(uid) is EstadoUnidad.LIVE
            await g.matar(uid)
            return await g.esperar_vacio(uid, plazo_s=10.0)
        try:
            estado = correr(caso())
        finally:
            correr(self._limpiar(g, uid))
        assert estado is EstadoUnidad.EMPTY

    def test_major3_el_entorno_persistente_del_manager_no_llega_a_la_cadena_de_la_unidad(self, tmp_path):
        """MAJOR 3 con systemd REAL: el servicio transitorio hereda el entorno del user manager; con la
        allowlist cerrada (`env -i`), nada de ese entorno llega a la cadena de la unidad. Las variables
        del bus existen en esta sesion (sin ellas `--user` no conecta): la prueba las usa de centinela."""
        uid, g = self._uid(), self._gestor()
        salida = tmp_path / "env-de-la-unidad.txt"

        async def caso():
            await g.arrancar(uid, ["/bin/sh", "-c", f"env > {salida}; sleep 5"])
            viva = await g.estado(uid)
            await g.matar(uid)
            return viva, await g.esperar_vacio(uid, plazo_s=10.0), salida.read_text()
        try:
            viva, estado, volcado = correr(caso())
        finally:
            correr(self._limpiar(g, uid))
        assert viva is EstadoUnidad.LIVE and estado is EstadoUnidad.EMPTY
        for prohibida in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "HOME", "LANG", "PATH"):
            assert f"{prohibida}=" not in volcado, prohibida

    def test_el_lanzador_arranca_la_unidad_y_la_jaula_recibe_su_uid_por_la_elevacion(self, tmp_path):
        """El camino completo de `lanzar` (0.6/R-2) con systemd-run REAL y un bwrap de mentira: la unidad
        del uid se verifica con sus limites, `jaula_viva` mira el cgroup (no un Popen), la elevacion recibe
        el uid de la jaula y el kill de la unidad mata el proceso --pipe."""
        from types import SimpleNamespace

        from jax.faro.bitacora import Bitacora
        from jax.faro.jaula import ConfigJaula, LanzadorJaula
        from tests._faro_utils import ejecucion as _ejecucion

        uid = self._uid()
        g = self._gestor()
        salida = tmp_path / "args.txt"
        falso = tmp_path / "bwrap-falso"
        falso.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {salida}\nsleep 30\n")
        falso.chmod(0o755)
        elevador = tmp_path / "elevador"
        elevador.write_text(f"#!/bin/sh\necho \"$1\" > {tmp_path}/uid-pedido.txt\nshift\nexec \"$@\"\n")
        elevador.chmod(0o755)
        srv = SimpleNamespace(ejecucion=_ejecucion(run_id=f"run-{uid}", uid_esperado=uid),
                              ruta_socket=Path(f"/run/faro/run-{uid}.sock"),
                              ruta_token=Path(f"/run/faro/run-{uid}.token"))

        async def caso():
            l = LanzadorJaula(ConfigJaula(bwrap=falso, elevar=(str(elevador), "{uid}")), Bitacora(emisores=[]), g)
            proceso = await l.lanzar(srv, ["/usr/bin/true"])
            viva = await l.jaula_viva(uid)                     # por el cgroup, no por el Popen
            await g.matar(uid)
            return viva, await g.esperar_vacio(uid, plazo_s=10.0), proceso
        try:
            viva, estado, proceso = correr(caso())
        finally:
            correr(self._limpiar(g, uid))
        assert viva is True and estado is EstadoUnidad.EMPTY
        assert proceso.returncode != 0                         # SIGKILL de su unidad
        args = salida.read_text().splitlines()
        assert "--bind" in args and "/faro/puerto.sock" in args and "/usr/bin/setpriv" not in args
        assert (tmp_path / "uid-pedido.txt").read_text().strip() == str(uid)   # la elevacion recibio el uid de la jaula
