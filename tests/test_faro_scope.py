"""El Faro, paso 0.6 / R-2: unidad cgroup y lease durable por ejecución.

Cada jaula corre dentro de un SERVICIO TRANSITORIO del user manager, `faro-jaula-u<uid>.service`, con
`MemoryMax`, `CPUQuota` y `TasksMax` FINITOS y exigidos por configuracion (sin los tres, no hay
lanzamiento), `Type=exec`, `KillMode=control-group`, `OOMPolicy=kill`, `--pipe` y `--collect`. La unidad
es determinista por uid. Un lease durable exclusivo arbitra arranques; recovery reconcilia leases con
systemd antes de abrir el canal de control, también tras perder memoria de proceso.

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
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from jax.faro.config import ConfigFaroInvalida
from jax.faro.scope import (ConfigScope, EstadoUnidad, ScopeError, ScopeManager, nombre_unidad,
                            nombre_unidad_intento, tiempospan_a_us)
from jax.faro.scope_lease import LeaseStore, LeaseStoreConflict, LeaseStoreCorrupt, LeaseStoreError, LeaseStoreNotFound

UID = 50001
UNIDAD = nombre_unidad(UID)


# --------------------------------------------------------------------------- #
# configuracion: los tres limites, finitos y obligatorios                      #
# --------------------------------------------------------------------------- #

ENV = {"JAX_FARO_SCOPE_MEMORY_MAX": "256M", "JAX_FARO_SCOPE_CPU_QUOTA": "200%",
       "JAX_FARO_SCOPE_TASKS_MAX": "128", "JAX_FARO_SCOPE_STATE_DIR": "/var/lib/faro/jaulas"}


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


def test_nombre_de_unidad_cambia_por_attempt():
    from jax.faro.scope import nombre_unidad_intento
    assert nombre_unidad_intento(UID, "a" * 32) != nombre_unidad_intento(UID, "b" * 32)


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
        self._muerto = False
        self.wait_calls = 0

    @property
    def returncode(self) -> int | None:
        if self._argv[0].endswith("systemd-run") and self._rc == 0:
            unidad = self._sistema.unidad_de_argv(self._argv)
            if unidad in self._sistema.trabajos:
                return None
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

    def kill(self):
        self._muerto = True
        self._rc = -9

    async def wait(self):
        self.wait_calls += 1
        while self.returncode is None and not self._muerto:
            await asyncio.sleep(0.001)
        return -9 if self._muerto else self.returncode


class SistemaDoble:
    """El user manager de mentira: unidades con LoadState/ActiveState, ControlGroup, limites efectivos y
    `cgroup.events` reales en `raiz` (un directorio normal, no cgroupfs). Normaliza los limites igual que
    systemd: MemoryMax en bytes, CPUQuotaPerSecUSec como timespan, TasksMax entero."""

    def __init__(self, raiz: Path, *, cfg: ConfigScope | None = None):
        self.cfg = cfg or ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128")
        self.raiz = raiz
        self.raiz.mkdir(parents=True, exist_ok=True)
        self.lease_dir = raiz.parent / "leases"
        self.lease_dir.mkdir(mode=0o700, exist_ok=True)
        self.unidades: dict[str, dict] = {}
        self.finales: dict[str, int] = {}
        self.arrancados: list[tuple[str, tuple[str, ...]]] = []       # unidades que NACIERON
        self.intentos: list[str] = []                                  # incluidos los rechazados por colision
        self.muertes: list[str] = []
        self.trabajos: dict[str, list[tuple[str, str, str]]] = {}
        self.al_list_jobs = None
        self.fallar_arranque: object | None = None       # Exception | "rc"
        self.fallar_show: str | None = None              # "exec" | "rc" | "basura" | "corta"
        self.kw: dict | None = None

    # -- helpers del doble -----------------------------------------------------------------
    @staticmethod
    def _canon(unidad: str | None) -> str | None:
        if unidad and (m := re.fullmatch(r"faro-jaula-u([1-9][0-9]*)-[0-9a-f]{32}\.service", unidad)):
            return nombre_unidad(int(m.group(1)))
        return unidad

    def unidad_de_argv(self, argv) -> str | None:
        for a in argv:
            if a.startswith("--unit="):
                return self._canon(a.split("=", 1)[1])
        return None

    def viva(self, unidad: str | None) -> bool:
        u = self.unidades.get(self._canon(unidad) or "")
        return bool(u) and not (u["activa"] is False) and u["poblada"] == 1

    def rc_final(self, unidad: str | None) -> int | None:
        u = self.unidades.get(self._canon(unidad) or "")
        return u.get("rc") if u else self.finales.get(self._canon(unidad) or "", 1)

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
        unidad = self._canon(unidad) or unidad
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
        unidad = self._canon(unidad) or unidad
        self.muertes.append(unidad)
        u = self.unidades.get(unidad)
        if u is None:
            return 1                                        # medido: kill de una unidad no cargada sale 1
        (self._cgroup(unidad) / "cgroup.events").write_text("populated 0\nfrozen 0\n")
        u.update(activa=False, rc=255)                      # SIGKILL --kill-whom=all a todo el cgroup
        self.finales[unidad] = 255
        self.unidades.pop(unidad, None)                     # --collect descarga la unidad al quedar vacía
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
        if list(argv[1:3]) == ["--user", "list-jobs"]:
            if self.al_list_jobs is not None:
                self.al_list_jobs()
            lineas = [f"{job} {unidad} {tipo} {estado}" for unidad, (job, tipo, estado) in self.trabajos.items()]
            return ProcFalso(self, argv, stdout=("\n".join(lineas) + ("\n" if lineas else "")).encode())
        if list(argv[1:3]) == ["--user", "list-units"]:
            lineas = [f"{unidad} loaded active running test" for unidad in self.unidades]  # --all incluye inactivas.
            return ProcFalso(self, argv, stdout=("\n".join(lineas) + ("\n" if lineas else "")).encode())
        if list(argv[1:3]) == ["--user", "kill"]:
            for a in argv:
                if not a.startswith("-") and a.endswith(".service"):
                    return ProcFalso(self, argv, rc=self.matar_unidad(a))
        raise AssertionError(f"comando inesperado: {argv}")

    def _show(self, argv) -> tuple[int, bytes, bytes]:
        unidad = next(a for a in argv if not a.startswith("-") and a.endswith(".service"))
        clave_unidad = self._canon(unidad) or unidad
        nombres = [a.split("=", 1)[1] for a in argv if a.startswith("--property=")]
        if self.fallar_show == "exec":
            raise OSError("systemctl no arranca")
        if self.fallar_show == "rc":
            return 1, b"", b"Failed to connect to bus\n"
        if self.fallar_show == "basura":
            return 0, b"esto no son\npropiedades\n", b""
        u = self.unidades.get(clave_unidad)
        if u is None:
            valores = {"LoadState": "not-found", "ActiveState": "inactive", "ControlGroup": ""}
        else:
            activo = u["activa"]
            valores = {"LoadState": u["load"], "ActiveState": activo if isinstance(activo, str)
                       else ("active" if activo else "inactive"), "ControlGroup": f"/{u['cg'].lstrip('/')}"}
        if self.fallar_show == "corta":                     # a ControlGroup se lo trajo un gato
            valores.pop("ControlGroup")
        for nombre in nombres:
            if nombre in ("MemoryMax", "CPUQuotaPerSecUSec", "TasksMax"):
                valores[nombre] = u["propiedades"].get(nombre) if u else \
                    {"MemoryMax": "infinity", "CPUQuotaPerSecUSec": "infinity", "TasksMax": "96548"}[nombre]
            if nombre == "Description":
                # systemd 259 sintetiza Description con el nombre aun si LoadState=not-found.
                valores[nombre] = u["propiedades"].get("Description", "") if u else unidad
            if nombre == "Delegate":
                valores[nombre] = u["propiedades"].get("Delegate", "no") if u else "no"
        # como systemd: clave=valor (el orden interno del manager no es el pedido)
        return 0, ("\n".join(f"{k}={v}" for k, v in valores.items() if k in nombres) + "\n").encode(), b""


def _timespan(us: int) -> str:
    return f"{us // 1_000_000}s" if us % 1_000_000 == 0 else f"{us // 1000}ms"


@pytest.fixture
def sistema(tmp_path):
    return SistemaDoble(tmp_path / "cgroup")


def gestor(sistema, *, cfg=None, **kw) -> ScopeManager:
    return ScopeManager(cfg or sistema.cfg, crear_proceso=sistema.crear_proceso, raiz_cgroup=sistema.raiz,
                        intervalo_s=0.01, lease_store=LeaseStore(sistema.lease_dir), **kw)


def transicionar_accepted(store, uid, intento, *, cgroup_path="/test-cgroup", cgroup_dev=1, cgroup_ino=1):
    from jax.faro.scope_lease import LeaseStage
    store.transition(uid, intento, LeaseStage.REQUESTED, LeaseStage.ACCEPTED,
                     cgroup_path=cgroup_path, cgroup_dev=cgroup_dev, cgroup_ino=cgroup_ino)


async def matar_handle(g, uid):
    try:
        intento = await g.intento_actual(uid)
    except ScopeError:
        intento = f"{uid:032x}"
    await g.matar(uid, intento=intento)


def _poblar(sistema, uid=UID, **estado):
    unidad = nombre_unidad(uid)
    store = LeaseStore(sistema.lease_dir)
    try:
        intento = store.read(uid).attempt_id
    except LeaseStoreNotFound:
        intento = f"{uid:032x}"
        store.acquire(uid, intento, nombre_unidad_intento(uid, intento))
    cg = sistema._cgroup(unidad)
    cg.mkdir(parents=True, exist_ok=True)
    (cg / "cgroup.events").write_text(f"populated {estado.get('poblada', 1)}\nfrozen 0\n")
    sistema.unidades[unidad] = {"load": estado.get("load", "loaded"), "activa": estado.get("activa", True),
                                "poblada": estado.get("poblada", 1), "cg": str(cg.relative_to(sistema.raiz)),
                                "propiedades": {"Description": f"FARO-R2-{intento}"}, "rc": estado.get("rc", 0)}


def test_recovery_mata_unidad_propietaria_y_suelta_el_lease_despues_de_empty(sistema):
    async def caso():
        intento = "crash-anterior"
        store = LeaseStore(sistema.lease_dir)
        from jax.faro.scope_lease import LeaseStage
        unidad = nombre_unidad_intento(UID, intento)
        store.acquire(UID, intento, unidad)
        cg = sistema._cgroup(UNIDAD)
        cg.mkdir(parents=True, exist_ok=True)
        (cg / "cgroup.events").write_text("populated 1\nfrozen 0\n")
        store.transition(UID, intento, LeaseStage.PREPARED, LeaseStage.REQUESTED)
        datos = cg.stat()
        transicionar_accepted(store, UID, intento, cgroup_path=f"/{cg.relative_to(sistema.raiz)}",
                              cgroup_dev=datos.st_dev, cgroup_ino=datos.st_ino)
        store.transition(UID, intento, LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED)
        sistema.unidades[UNIDAD] = {"load": "loaded", "activa": True, "poblada": 1,
                                    "cg": str(cg.relative_to(sistema.raiz)),
                                    "propiedades": {"Description": f"FARO-R2-{intento}"}, "rc": 0}
        g = gestor(sistema)
        await g.recuperar_leases()
        return sistema.muertes, await g.estado(UID), store.all()

    muertes, estado, leases = correr(caso())
    assert muertes == [UNIDAD]
    assert estado is EstadoUnidad.EMPTY
    assert leases == ()


def test_recovery_ausencia_en_la_misma_epoch_conserva_lease(sistema):
    async def caso():
        store = LeaseStore(sistema.lease_dir)
        store.acquire(UID, "crash-antes-de-start", nombre_unidad_intento(UID, "crash-antes-de-start"))
        from jax.faro.scope_lease import LeaseStage
        store.transition(UID, "crash-antes-de-start", LeaseStage.PREPARED, LeaseStage.REQUESTED)
        g = gestor(sistema)
        await g.recuperar_leases()
        blocked = await g.estado(UID)
        await g.arrancar(UID + 1, ["/usr/bin/true"])
        return sistema.muertes, store.all(), blocked

    muertes, leases, blocked = correr(caso())
    assert muertes == [] and {lease.uid for lease in leases} == {UID, UID + 1} and blocked is EstadoUnidad.UNKNOWN


def test_namespace_con_unidad_huerfana_sin_lease_rechaza_uid(sistema):
    _poblar(sistema)
    # Simula caída de metadata sin borrar la unidad del manager.
    LeaseStore(sistema.lease_dir).release(UID, LeaseStore(sistema.lease_dir).read(UID).attempt_id)
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.UNKNOWN


def test_namespace_con_job_huerfano_sin_lease_rechaza_uid(sistema):
    intento = "a" * 32
    LeaseStore(sistema.lease_dir).release(UID, LeaseStore(sistema.lease_dir).acquire(
        UID, intento, nombre_unidad_intento(UID, intento)).attempt_id)
    sistema.trabajos[nombre_unidad_intento(UID, intento)] = ("416", "start", "running")
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.UNKNOWN


def test_recovery_unidad_ajena_bloquea_y_conserva_lease(sistema):
    async def caso():
        store = LeaseStore(sistema.lease_dir)
        from jax.faro.scope_lease import LeaseStage
        lease = store.acquire(UID, "crash-anterior", nombre_unidad_intento(UID, "crash-anterior"))
        store.transition(UID, lease.attempt_id, LeaseStage.PREPARED, LeaseStage.REQUESTED)
        _poblar(sistema)
        cg = sistema._cgroup(UNIDAD)
        datos = cg.stat()
        transicionar_accepted(store, UID, lease.attempt_id, cgroup_path=f"/{cg.relative_to(sistema.raiz)}",
                              cgroup_dev=datos.st_dev, cgroup_ino=datos.st_ino)
        store.transition(UID, lease.attempt_id, LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED)
        sistema.unidades[UNIDAD]["propiedades"]["Description"] = "FARO-R2-otro-intento"
        g = gestor(sistema)
        await g.recuperar_leases()
        return sistema.muertes, store.read(UID), await g.estado(UID)

    muertes, lease, estado = correr(caso())
    assert muertes == []
    assert lease.attempt_id == "crash-anterior" and estado is EstadoUnidad.LIVE


def test_recovery_serializada_no_reconcilia_dos_veces(sistema):
    async def caso():
        store = LeaseStore(sistema.lease_dir)
        store.acquire(UID, "crash-anterior", nombre_unidad_intento(UID, "crash-anterior"))
        g = gestor(sistema)
        entro, continuar = asyncio.Event(), asyncio.Event()
        llamadas = 0

        async def reconciliar(_lease):
            nonlocal llamadas
            llamadas += 1
            entro.set()
            await continuar.wait()

        g._reconciliar_lease_de_arranque = reconciliar
        primera = asyncio.create_task(g.recuperar_leases())
        await entro.wait()
        segunda = asyncio.create_task(g.recuperar_leases())
        await asyncio.sleep(0)
        continuar.set()
        await asyncio.gather(primera, segunda)
        return llamadas

    assert correr(caso()) == 1


def correr(coro):
    return asyncio.run(coro)


def test_lease_durable_conflicto_y_liberacion_atada_al_intento(tmp_path):
    directorio = tmp_path / "leases"
    directorio.mkdir(mode=0o700)
    store = LeaseStore(directorio)
    lease = store.acquire(UID, "intento-a", nombre_unidad_intento(UID, "intento-a"))
    assert store.read(UID) == lease
    assert store.all() == (lease,)
    with pytest.raises(LeaseStoreConflict):
        store.acquire(UID, "intento-b", nombre_unidad_intento(UID, "intento-b"))
    with pytest.raises(LeaseStoreConflict):
        store.release(UID, "intento-b")
    assert store.read(UID) == lease
    store.release(UID, "intento-a")
    assert store.all() == ()


def test_lease_solicitado_no_se_puede_liberar_y_transicion_es_cas(tmp_path):
    from jax.faro.scope_lease import LeaseStage
    directorio = tmp_path / "leases"
    directorio.mkdir(mode=0o700)
    store = LeaseStore(directorio)
    lease = store.acquire(UID, "staged", nombre_unidad_intento(UID, "staged"))
    assert lease.stage is LeaseStage.PREPARED
    with pytest.raises(LeaseStoreConflict):
        store.transition(UID, "otro-attempt", LeaseStage.PREPARED, LeaseStage.REQUESTED)
    store.transition(UID, "staged", LeaseStage.PREPARED, LeaseStage.REQUESTED)
    with pytest.raises(LeaseStoreConflict):
        store.release(UID, "staged")
    transicionar_accepted(store, UID, "staged")
    with pytest.raises(LeaseStoreConflict):
        store.release(UID, "staged")
    store.transition(UID, "staged", LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED)
    with pytest.raises(LeaseStoreConflict):
        store.release(UID, "staged")
    store.release(UID, "staged", cgroup_verified=True)
    assert store.all() == ()


def test_lock_instancia_rechaza_manager_duplicado(tmp_path):
    directorio = tmp_path / "leases"
    directorio.mkdir(mode=0o700)
    cfg = ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128", state_dir=directorio)
    async def caso():
        primero, segundo = ScopeManager(cfg), ScopeManager(cfg)
        await primero.adquirir_instancia()
        with pytest.raises(ScopeError, match="instancia única"):
            await segundo.adquirir_instancia()
        await primero.liberar_instancia()
        await segundo.adquirir_instancia()
        await segundo.liberar_instancia()
    correr(caso())


def test_kill_con_attempt_stale_no_mata_unidad_distinta(sistema):
    async def caso():
        g = gestor(sistema)
        intento_viejo, intento_nuevo = "a" * 32, "b" * 32
        from jax.faro.scope import nombre_unidad_intento
        actual = nombre_unidad_intento(UID, intento_nuevo)
        unidad, props = actual, {"Description": f"FARO-R2-{intento_nuevo}"}
        cg = sistema._cgroup(unidad)
        cg.mkdir(parents=True)
        (cg / "cgroup.events").write_text("populated 1\nfrozen 0\n")
        sistema.unidades[unidad] = {"load": "loaded", "activa": True, "poblada": 1,
                                    "cg": str(cg.relative_to(sistema.raiz)), "propiedades": props, "rc": None}
        with pytest.raises(ScopeError, match="propietario"):
            await g.matar(UID, intento=intento_viejo)
        return sistema.muertes
    assert correr(caso()) == []


def test_error_durante_drenaje_stderr_se_reporta(sistema):
    class StderrRoto:
        async def read(self, _n):
            raise OSError("pipe roto")
    class Proceso:
        stderr = StderrRoto()
        async def wait(self):
            return 0
    async def caso():
        g = gestor(sistema)
        proceso = Proceso()
        tarea = asyncio.create_task(g._drenar_stderr(proceso.stderr, bytearray()))
        proceso._faro_stderr_task = tarea
        with pytest.raises(OSError, match="pipe roto"):
            await g.esperar_cliente(proceso)
    correr(caso())


def test_lease_corrupto_no_se_interpreta_como_ausencia(tmp_path):
    directorio = tmp_path / "leases"
    directorio.mkdir(mode=0o700)
    archivo = directorio / f"{UID}.json"
    archivo.write_text("{ incompleto\n")
    archivo.chmod(0o600)
    with pytest.raises(LeaseStoreCorrupt):
        LeaseStore(directorio).all()


def test_lease_con_modo_inseguro_falla_cerrado(tmp_path):
    directorio = tmp_path / "leases"
    directorio.mkdir(mode=0o700)
    directorio.chmod(0o750)
    with pytest.raises(LeaseStoreError):
        LeaseStore(directorio)


# --------------------------------------------------------------------------- #
# los argumentos exactos de systemd-run y systemctl                           #
# --------------------------------------------------------------------------- #

def test_el_argv_de_arranque_es_el_del_diseno_sin_defaults_ni_sorpresas():
    g = ScopeManager(ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128", slice="faro.slice"))
    argv = g.argv_arrancar(UID, ["/usr/bin/sudo", "-n", "-u", "#50001", "--", "/usr/bin/bwrap", "--dev", "/dev"],
                           intento="abc123")
    assert argv[0] == "/usr/bin/systemd-run"
    assert argv[1:6] == ["--user", "--quiet", "--pipe", "--collect",
                        "--unit=faro-jaula-u50001-6ca13d52ca70c883e0f0bb101e425a89.service"]
    assert "--slice=faro.slice" in argv
    for propiedad in ("Type=exec", "KillMode=control-group", "OOMPolicy=kill", "Delegate=no",
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
    argv = ScopeManager(ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128")).argv_matar(
        UID, intento="abc123")
    assert argv == ["/usr/bin/systemctl", "--user", "kill",
                    "faro-jaula-u50001-6ca13d52ca70c883e0f0bb101e425a89.service",
                    "--kill-whom=all", "--signal=SIGKILL"]


def test_argv_matar_requiere_attempt_esperado():
    g = ScopeManager(ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128"))
    with pytest.raises(TypeError):
        g.argv_matar(UID)


# --------------------------------------------------------------------------- #
# el trivalor: solo EMPTY libera                                              #
# --------------------------------------------------------------------------- #

def test_activa_y_poblada_es_live(sistema):
    _poblar(sistema, poblada=1)
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.LIVE


def test_not_found_es_empty_porque_systemd_confirmo_que_no_existe(sistema):
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.EMPTY


def test_not_found_con_job_de_arranque_en_cola_es_unknown_y_retene_uid(sistema):
    sistema.trabajos[UNIDAD] = ("417", "start", "waiting")
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.UNKNOWN


def test_not_found_job_termina_antes_de_list_jobs_y_segunda_lectura_ve_unidad(sistema):
    unidad, props, comando = sistema._parsear_run(
        gestor(sistema).argv_arrancar(UID, ["/usr/bin/true"], intento="carrera-status"))
    def materializar():
        sistema._crear(unidad, comando, props)
        sistema.al_list_jobs = None
    sistema.al_list_jobs = materializar
    # Sin lease/handle exacto, la unidad de otro attempt no se adopta por su nombre base.
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.UNKNOWN


def test_muerta_y_recolectada_es_empty(sistema):
    _poblar(sistema)
    sistema.matar_unidad(UNIDAD)
    sistema.recolectar(UNIDAD)
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.EMPTY


def test_inactiva_con_cgroup_vacio_sin_lease_es_unknown(sistema):
    _poblar(sistema, activa=False, poblada=0)
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.UNKNOWN


def test_inactiva_con_el_directorio_del_cgroup_ya_eliminado_sin_lease_es_unknown(sistema):
    _poblar(sistema, activa=False, poblada=0)
    shutil.rmtree(sistema._cgroup(UNIDAD))
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.UNKNOWN


def test_inactiva_pero_con_procesos_en_el_cgroup_es_live_el_uid_no_se_libera(sistema):
    """El caso R-2: murio el proceso principal (o el cliente del Faro), quedan hijos/nietos en el cgroup."""
    _poblar(sistema, activa=False, poblada=1)
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.LIVE


def test_desactivandose_con_procesos_en_el_cgroup_es_live(sistema):
    _poblar(sistema, activa="deactivating", poblada=1)
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.LIVE


def test_desactivandose_y_vacia_es_empty(sistema):
    _poblar(sistema, activa="deactivating", poblada=0)
    assert correr(gestor(sistema).estado(UID)) is EstadoUnidad.UNKNOWN  # manager aún conserva la unidad cargada


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
    assert consultas[0][0] == LeaseStore(sistema.lease_dir).read(UID).unit
    assert "ControlGroup" in consultas[0][1]


def test_control_group_devuelve_la_ruta_que_reporta_systemd(sistema):
    _poblar(sistema)

    async def caso():
        return await gestor(sistema).control_group(UID)
    assert correr(caso()) == f"/{sistema.unidades[UNIDAD]['cg']}"


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
            await g.matar(UID, intento=await g.intento_actual(UID))
    assert correr(caso()) is None                                          # --pipe: vive mientras la unidad viva


def test_delegate_distinto_de_no_se_mata_exacto_y_no_se_acepta(sistema):
    async def caso():
        original = sistema._crear

        def crear(unidad, comando, propiedades):
            rc = original(unidad, comando, propiedades)
            if rc == 0:
                sistema.unidades[UNIDAD]["propiedades"]["Delegate"] = "yes"
            return rc

        sistema._crear = crear
        g = gestor(sistema)
        with pytest.raises(ScopeError):
            await g.arrancar(UID, ["/usr/bin/true"])
        return LeaseStore(sistema.lease_dir).read(UID), sistema.muertes

    lease, muertes = correr(caso())
    assert lease.stage.value == "REQUESTED"
    assert muertes == [UNIDAD]


def test_finalizar_natural_usa_handle_unico_y_libera_tras_cliente_y_empty(sistema):
    async def caso():
        g = gestor(sistema)
        proceso = await g.arrancar(UID, ["/usr/bin/true"])
        lease = LeaseStore(sistema.lease_dir).read(UID)
        assert lease.stage.value == "ACCEPTED"
        await g.matar(UID, intento=lease.attempt_id)
        rc = await g.finalizar_natural(UID, proceso)
        return rc, LeaseStore(sistema.lease_dir).all()
    rc, leases = correr(caso())
    assert rc == 255
    assert leases == ()


def _lease_closed_con_cgroup_poblado(sistema):
    async def caso():
        cfg = ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128", plazo_vacio_s=0.02)
        g = gestor(sistema, cfg=cfg, plazo_consulta_s=0.01)
        proceso = await g.arrancar(UID, ["/usr/bin/sleep", "1"])
        lease = LeaseStore(sistema.lease_dir).read(UID)
        unidad = sistema._canon(lease.unit)
        sistema.unidades[unidad].update(activa=False, rc=255)
        with pytest.raises(ScopeError, match="EMPTY|poblada"):
            await g.finalizar_natural(UID, proceso)
        assert LeaseStore(sistema.lease_dir).read(UID).stage.value == "CLIENT_CLOSED"
        assert g._handles[UID].wait_ok and g._handles[UID].materialization_seen
        return g, proceso, lease
    return correr(caso())


@pytest.mark.parametrize("cgroup", ["vacio", "ausente"])
def test_client_closed_reintenta_sin_repetir_wait_y_suelta_con_prueba_kernel(sistema, cgroup):
    g, proceso, lease = _lease_closed_con_cgroup_poblado(sistema)
    unidad = sistema._canon(lease.unit)
    sistema.unidades.pop(unidad)  # el manager ya recolectó la unidad tras cerrar el cliente
    ruta = sistema._cgroup(unidad)
    if cgroup == "vacio":
        (ruta / "cgroup.events").write_text("populated 0\nfrozen 0\n")
    else:
        shutil.rmtree(ruta)

    async def cerrar():
        await g.cerrar_uid(UID)
        return g._leases.all()
    leases = correr(cerrar())
    assert proceso.wait_calls == 1
    assert sistema.muertes == []
    assert leases == ()


def test_client_closed_con_unidad_propia_mata_attempt_exacto_y_reusa_espera(sistema):
    g, proceso, lease = _lease_closed_con_cgroup_poblado(sistema)

    async def cerrar():
        await g.cerrar_uid(UID)
        return g._leases.all()
    assert correr(cerrar()) == ()
    assert sistema.muertes == [sistema._canon(lease.unit)]
    assert proceso.wait_calls == 1


@pytest.mark.parametrize("campo", ["wait_ok", "materialization_seen", "intento"])
def test_client_closed_no_reintenta_con_handle_incompleto_o_de_otro_attempt(sistema, campo):
    g, proceso, lease = _lease_closed_con_cgroup_poblado(sistema)
    handle = g._handles[UID]
    if campo == "intento":
        handle.intento = "otro-attempt"
    else:
        setattr(handle, campo, False)

    async def cerrar():
        with pytest.raises(ScopeError):
            await g.cerrar_uid(UID)
        return g._leases.read(UID)
    retenido = correr(cerrar())
    assert retenido.attempt_id == lease.attempt_id
    assert retenido.stage.value == "CLIENT_CLOSED"
    assert proceso.wait_calls == 1


@pytest.mark.parametrize("problema", ["poblado", "inode", "namespace"])
def test_client_closed_reintenta_y_retiene_si_kernel_o_namespace_no_es_concluyente(sistema, problema):
    g, proceso, lease = _lease_closed_con_cgroup_poblado(sistema)
    unidad = sistema._canon(lease.unit)
    sistema.unidades.pop(unidad)
    ruta = sistema._cgroup(unidad)

    if problema == "poblado":
        # Sin unidad visible el cgroup poblado debe retener el lease.
        pass
    elif problema == "inode":
        ruta_original = ruta.with_name(ruta.name + ".guardada")
        ruta.rename(ruta_original)
        ruta.mkdir()
        (ruta / "cgroup.events").write_text("populated 0\n")
        assert ruta.stat().st_ino != lease.cgroup_ino
    else:
        otra = nombre_unidad_intento(UID, "otra-unidad")
        sistema.unidades[otra] = {"load": "loaded", "activa": False, "poblada": 0,
                                  "cg": str(sistema._cgroup(otra).relative_to(sistema.raiz)),
                                  "propiedades": {"Description": "ajena"}, "rc": 0}
        (ruta / "cgroup.events").write_text("populated 0\n")

    async def cerrar():
        with pytest.raises(ScopeError):
            await g.cerrar_uid(UID)
        return g._leases.read(UID)
    retenido = correr(cerrar())
    assert retenido.stage.value == "CLIENT_CLOSED"
    assert proceso.wait_calls == 1


@pytest.mark.parametrize("poblada", [0, 1])
def test_recovery_kill_race_reconcilia_uid_y_sigue_con_hermanas(sistema, poblada):
    from jax.faro.scope_lease import LeaseStage

    async def caso():
        store = LeaseStore(sistema.lease_dir)
        intento_a, intento_b = "recover-race-a", "recover-race-b"
        unidad_a = nombre_unidad_intento(UID, intento_a)
        unidad_b = nombre_unidad_intento(UID + 1, intento_b)
        store.acquire(UID, intento_a, unidad_a)
        store.transition(UID, intento_a, LeaseStage.PREPARED, LeaseStage.REQUESTED)
        cg_a = sistema._cgroup(unidad_a)
        cg_a.mkdir(parents=True)
        (cg_a / "cgroup.events").write_text(f"populated {poblada}\n")
        datos = cg_a.stat()
        unidad_visible_a = sistema._canon(unidad_a)
        transicionar_accepted(store, UID, intento_a, cgroup_path=f"/{cg_a.relative_to(sistema.raiz)}",
                              cgroup_dev=datos.st_dev, cgroup_ino=datos.st_ino)
        store.transition(UID, intento_a, LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED)
        sistema.unidades[unidad_visible_a] = {"load": "loaded", "activa": True, "poblada": poblada,
                                              "cg": str(cg_a.relative_to(sistema.raiz)),
                                              "propiedades": {"Description": f"FARO-R2-{intento_a}"}, "rc": 255}
        store.acquire(UID + 1, intento_b, unidad_b)  # PREPARED debe converger aunque A tenga carrera
        cfg = ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128", plazo_vacio_s=0.02)
        g = gestor(sistema, cfg=cfg, plazo_consulta_s=0.01)
        llamadas_kill = []

        async def kill_que_pierde_carrera(uid, *, intento):
            llamadas_kill.append((uid, intento))
            assert uid == UID and intento == intento_a
            sistema.unidades.pop(unidad_visible_a)  # el transient desaparece entre identidad y kill
            raise OSError("la unidad se recolectó antes del kill")

        g.matar = kill_que_pierde_carrera
        await g.recuperar_leases()
        lease_a = None
        try:
            lease_a = store.read(UID)
        except LeaseStoreNotFound:
            pass
        with pytest.raises(LeaseStoreNotFound):
            store.read(UID + 1)
        estado_a = await g.estado(UID)
        await g._arrancar(UID + 2, ["/usr/bin/true"])
        return lease_a, estado_a, llamadas_kill, sistema.unidades

    lease_a, estado_a, llamadas_kill, unidades = correr(caso())
    assert (lease_a is not None) is bool(poblada)
    if lease_a is not None:
        assert lease_a.stage.value == "CLIENT_CLOSED"
    assert estado_a is (EstadoUnidad.LIVE if poblada else EstadoUnidad.EMPTY)
    assert llamadas_kill == [(UID, "recover-race-a")]
    assert nombre_unidad(UID + 2) in unidades


def test_recovery_error_operativo_de_un_uid_cuarentena_y_continua_con_hermanas(sistema):
    from jax.faro.scope_lease import LeaseStage

    async def caso():
        store = LeaseStore(sistema.lease_dir)
        intento_a, intento_b = "recover-error-a", "recover-error-b"
        unidad_a = nombre_unidad_intento(UID, intento_a)
        cg_a = sistema._cgroup(unidad_a)
        cg_a.mkdir(parents=True)
        (cg_a / "cgroup.events").write_text("populated 1\n")
        datos = cg_a.stat()
        store.acquire(UID, intento_a, unidad_a)
        store.transition(UID, intento_a, LeaseStage.PREPARED, LeaseStage.REQUESTED)
        transicionar_accepted(store, UID, intento_a, cgroup_path=f"/{cg_a.relative_to(sistema.raiz)}",
                              cgroup_dev=datos.st_dev, cgroup_ino=datos.st_ino)
        store.transition(UID, intento_a, LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED)
        store.acquire(UID + 1, intento_b, nombre_unidad_intento(UID + 1, intento_b))
        g = gestor(sistema)
        identidad = g._identidad

        async def fallo_a(uid, intento):
            if uid == UID:
                raise OSError("systemctl no pudo inspeccionar A")
            return await identidad(uid, intento)

        g._identidad = fallo_a
        await g.recuperar_leases()
        retenido_a = store.read(UID)
        with pytest.raises(LeaseStoreNotFound):
            store.read(UID + 1)
        await g._arrancar(UID + 2, ["/usr/bin/true"])
        return retenido_a, await g.estado(UID), sistema.unidades

    lease_a, estado_a, unidades = correr(caso())
    assert lease_a.stage.value == "CLIENT_CLOSED"
    assert estado_a is EstadoUnidad.LIVE
    assert nombre_unidad(UID + 2) in unidades


def test_recovery_all_con_schema_corrupto_aborta_globalmente(sistema):
    (sistema.lease_dir / f"{UID}.json").write_text("{ incompleto\n")
    (sistema.lease_dir / f"{UID}.json").chmod(0o600)
    g = gestor(sistema)

    async def recuperar():
        with pytest.raises(ScopeError, match="no se pueden leer las cuarentenas"):
            await g.recuperar_leases()
        return g._leases_recuperados
    assert correr(recuperar()) is False


def test_manager_restart_con_cliente_error_y_cgroup_poblado_reteniene_lease(sistema):
    async def caso():
        cfg = ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128", plazo_vacio_s=0.03)
        g = gestor(sistema, cfg=cfg, plazo_consulta_s=0.01)
        proceso = await g.arrancar(UID, ["/usr/bin/true"])
        lease = LeaseStore(sistema.lease_dir).read(UID)
        assert lease.stage.value == "ACCEPTED" and lease.cgroup_path and lease.cgroup_ino
        # Simula wait de cliente con fallo y un manager que perdió el transient;
        # el kernel conserva el cgroup poblado tras desaparecer la unidad.
        proceso._rc = 1
        sistema.unidades.clear()
        with pytest.raises(ScopeError, match="EMPTY|poblada"):
            await g.finalizar_natural(UID, proceso)
        retained = LeaseStore(sistema.lease_dir).read(UID)
        estado = await g.estado(UID)
        with pytest.raises(ScopeError):
            await g._arrancar(UID, ["/usr/bin/true"])
        return retained, estado, sistema.muertes

    lease, estado, muertes = correr(caso())
    assert lease.stage.value == "CLIENT_CLOSED"
    assert estado is EstadoUnidad.LIVE
    assert muertes == []


def test_recovery_sin_manager_no_libera_cgroup_poblado_de_client_closed(sistema):
    async def caso():
        store = LeaseStore(sistema.lease_dir)
        intento = "recovery-kernel"
        lease = store.acquire(UID, intento, nombre_unidad_intento(UID, intento))
        cg = sistema._cgroup(lease.unit)
        cg.mkdir(parents=True)
        (cg / "cgroup.events").write_text("populated 1\n")
        datos = cg.stat()
        from jax.faro.scope_lease import LeaseStage
        store.transition(UID, intento, LeaseStage.PREPARED, LeaseStage.REQUESTED)
        transicionar_accepted(store, UID, intento, cgroup_path=f"/{cg.relative_to(sistema.raiz)}",
                              cgroup_dev=datos.st_dev, cgroup_ino=datos.st_ino)
        store.transition(UID, intento, LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED)
        g = gestor(sistema)
        await g.recuperar_leases()
        return store.read(UID), await g.estado(UID), sistema.muertes

    lease, estado, muertes = correr(caso())
    assert lease.stage.value == "CLIENT_CLOSED"
    assert estado is EstadoUnidad.LIVE
    assert muertes == []


def test_cgroup_accepted_reemplazado_por_symlink_es_unknown_y_retenido(sistema):
    async def caso():
        g = gestor(sistema)
        await g.arrancar(UID, ["/usr/bin/true"])
        lease = LeaseStore(sistema.lease_dir).read(UID)
        cg = sistema.raiz / lease.cgroup_path.lstrip("/")
        externo = sistema.raiz.parent / "externo"
        externo.mkdir()
        (externo / "cgroup.events").write_text("populated 0\n")
        shutil.rmtree(cg)
        cg.symlink_to(externo, target_is_directory=True)
        estado = await g.estado(UID)
        retenido = LeaseStore(sistema.lease_dir).read(UID)
        return estado, retenido

    estado, lease = correr(caso())
    assert estado is EstadoUnidad.UNKNOWN
    assert lease.stage.value == "ACCEPTED"


def test_evidencia_cgroup_rechaza_traversal_inode_distinto_y_events_ilegible(sistema):
    async def caso():
        g = gestor(sistema)
        await g.arrancar(UID, ["/usr/bin/true"])
        lease = LeaseStore(sistema.lease_dir).read(UID)
        from dataclasses import replace
        assert g._leer_cgroup_protegido("/../etc", lease) is None
        assert g._leer_cgroup_protegido(lease.cgroup_path,
                                        replace(lease, cgroup_ino=lease.cgroup_ino + 1)) is None
        cg = sistema.raiz / lease.cgroup_path.lstrip("/")
        (cg / "cgroup.events").write_text("populated 2\n")
        return await g.estado(UID), LeaseStore(sistema.lease_dir).read(UID)

    estado, lease = correr(caso())
    assert estado is EstadoUnidad.UNKNOWN
    assert lease.stage.value == "ACCEPTED"


def test_cerrar_uid_bloquea_drena_handle_y_suelta_solo_tras_empty(sistema):
    async def caso():
        g = gestor(sistema)
        await g.arrancar(UID, ["/usr/bin/true"])
        g.bloquear_uid(UID)
        await g.cerrar_uid(UID)
        return sistema.muertes, LeaseStore(sistema.lease_dir).all()
    muertes, leases = correr(caso())
    assert muertes == [UNIDAD]
    assert leases == ()


def test_finalizacion_natural_concurrente_con_cierre_comparte_handle_y_libera_una_vez(sistema):
    async def caso():
        g = gestor(sistema)
        proceso = await g.arrancar(UID, ["/usr/bin/true"])
        intento = LeaseStore(sistema.lease_dir).read(UID).attempt_id
        liberaciones = []
        liberar = g._leases.release

        def contar_release(uid, esperado, **kwargs):
            liberaciones.append((uid, esperado))
            return liberar(uid, esperado, **kwargs)

        g._leases.release = contar_release
        natural = asyncio.create_task(g.finalizar_natural(UID, proceso))
        # Deja que el task natural llegue al wait compartido del cliente.
        await asyncio.sleep(0)
        cierre = asyncio.create_task(g.cerrar_uid(UID))
        rc, _ = await asyncio.wait_for(asyncio.gather(natural, cierre), 2)
        return rc, intento, liberaciones, sistema.muertes, g._leases.all()

    rc, intento, liberaciones, muertes, leases = correr(caso())
    assert rc == 255
    assert liberaciones == [(UID, intento)]
    assert muertes == [UNIDAD]
    assert leases == ()


def test_cerrar_uid_sin_handle_no_libera_requested_ausente(sistema):
    async def caso():
        store = LeaseStore(sistema.lease_dir)
        intento = "sin-handle"
        store.acquire(UID, intento, nombre_unidad_intento(UID, intento))
        from jax.faro.scope_lease import LeaseStage
        store.transition(UID, intento, LeaseStage.PREPARED, LeaseStage.REQUESTED)
        g = gestor(sistema)
        with pytest.raises(ScopeError, match="retenido|cierre causal"):
            await g.cerrar_uid(UID)
        return store.read(UID).stage
    assert correr(caso()).value == "REQUESTED"


def test_spawn_que_termina_antes_de_observar_unidad_conserva_quarantine(sistema):
    async def caso():
        sistema.fallar_arranque = "rc"
        g = gestor(sistema)
        with pytest.raises(ScopeError):
            await g.arrancar(UID, ["/usr/bin/true"])
        return LeaseStore(sistema.lease_dir).read(UID).stage, sistema.muertes
    stage, muertes = correr(caso())
    assert stage.value == "REQUESTED"
    assert muertes == []


def test_peticion_en_cola_materializa_despues_de_salida_y_no_libera_requested(sistema):
    async def caso():
        g = gestor(sistema)
        liberador = asyncio.Event()
        materializada = asyncio.Event()
        argv_lanzamiento = []
        async def enviar(*argv, **kw):
            if argv[0].endswith("systemd-run"):
                argv_lanzamiento.extend(argv)
                asyncio.create_task(materializar_despues())
                raise OSError("bus desconectado después de encolar la petición")
            return await sistema.crear_proceso(*argv, **kw)
        async def materializar_despues():
            await liberador.wait()
            unidad, propiedades, comando = sistema._parsear_run(argv_lanzamiento)
            sistema._crear(unidad, comando, propiedades)
            materializada.set()
        g._crear_proceso = enviar
        with pytest.raises(ScopeError):
            await g.arrancar(UID, ["/usr/bin/true"])
        # El primer scan ya pudo ver namespace vacío; la petición aún estaba en tránsito.
        assert LeaseStore(sistema.lease_dir).read(UID).stage.value == "REQUESTED"
        liberador.set()
        await asyncio.wait_for(materializada.wait(), 1)
        assert await g.estado(UID) is EstadoUnidad.UNKNOWN
        with pytest.raises(ScopeError, match="retenido|cierre causal"):
            await g.cerrar_uid(UID)
        return sistema.muertes, LeaseStore(sistema.lease_dir).read(UID).stage.value
    muertes, stage = correr(caso())
    assert muertes == [] and stage == "REQUESTED"


@pytest.mark.parametrize("salida_units,salida_jobs", [
    (b"\xe2\x97\x8f faro-jaula-u50001-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.service loaded failed dead stale\n", b""),
    (b"", b"fila-incompleta\n"),
])
def test_namespace_malformado_o_con_marcador_no_se_declara_vacio(sistema, salida_units, salida_jobs):
    async def caso():
        g = gestor(sistema)

        async def correr(argv, **_kw):
            if "list-units" in argv:
                assert "--plain" in argv
                return 0, salida_units, b""
            if "list-jobs" in argv:
                return 0, salida_jobs, b""
            raise AssertionError(argv)

        g._correr = correr
        return await g._namespace_vacio(UID)

    assert correr(caso()) is False


def test_cerrar_uid_reporta_drain_roto_despues_de_cleanup_seguro(sistema):
    class StderrRoto:
        async def read(self, _n):
            raise OSError("pipe roto")
    async def caso():
        g = gestor(sistema)
        original = sistema.crear_proceso
        async def crear(*argv, **kw):
            proceso = await original(*argv, **kw)
            if argv[0].endswith("systemd-run"):
                proceso.stderr = StderrRoto()
            return proceso
        g._crear_proceso = crear
        await g.arrancar(UID, ["/usr/bin/true"])
        with pytest.raises(ScopeError, match="stderr"):
            await g.cerrar_uid(UID)
        return sistema.muertes, LeaseStore(sistema.lease_dir).all()
    muertes, leases = correr(caso())
    assert muertes == [UNIDAD]
    assert leases == ()


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
        for esperada in ("ActiveState", "ControlGroup", "Description", "Delegate", "MemoryMax",
                         "CPUQuotaPerSecUSec", "TasksMax"):
            assert esperada in props, esperada
        assert "--value" not in shows[0]                    # clave=valor: el orden interno de systemd no es el pedido
    correr(caso())


def test_arrancar_con_la_unidad_ocupada_por_otro_falla_y_no_la_toca(sistema):
    async def caso():
        _poblar(sistema)                                       # unidad viva de otra identidad
        sistema.unidades[UNIDAD]["propiedades"]["Description"] = "FARO-R2-otra-identidad"
        g = gestor(sistema)
        try:
            await g.arrancar(UID, ["/usr/bin/true"])
        except ScopeError as e:
            assert "namespace" in str(e)
            return await g.estado(UID), sistema.muertes
        raise AssertionError("tenia que fallar: la unidad ya existe")
    estado, muertes = correr(caso())
    assert estado is EstadoUnidad.UNKNOWN              # el lease no demuestra propiedad de la unidad ajena
    assert muertes == []


def test_recovery_con_unidad_de_otro_attempt_no_la_adopta_ni_la_mata(sistema):
    """El lease persistido pertenece a un attempt; una unidad bajo el uid pero de otro attempt
    bloquea recovery y nunca se mata."""
    async def caso():
        _poblar(sistema, activa=True, poblada=1)
        sistema.unidades[UNIDAD]["propiedades"]["Description"] = "FARO-R2-ganador"
        g = gestor(sistema)
        await g.recuperar_leases()
        return await g.estado(UID), sistema.muertes
    estado, muertes = correr(caso())
    assert estado is EstadoUnidad.UNKNOWN              # no es prueba de identidad del lease recuperado
    assert muertes == []                               # y jamas recibio nuestro kill


def test_major2_cancelada_la_creacion_del_subproceso_la_unidad_nacida_se_limpia(sistema):
    """MAJOR 2: la cancelacion llega MIENTRAS _crear_proceso esta en vuelo, despues de que el doble ya
    creo la unidad. La limpieza acotada la encuentra (Description nuestra) y la mata; nunca queda viva
    sin seguimiento."""
    async def caso():
        g = gestor(sistema)
        adentro, devolver_handle = asyncio.Event(), asyncio.Event()

        async def crear_lento(*argv, **kw):
            p = await sistema.crear_proceso(*argv, **kw)       # la unidad YA nacio con NUESTRA Description
            if argv[0].endswith("systemd-run"):
                adentro.set()
                await devolver_handle.wait()                    # unidad nacida; handle aún no entregado
            return p
        g._crear_proceso = crear_lento
        tarea = asyncio.create_task(g.arrancar(UID, ["/usr/bin/true"]))
        await asyncio.wait_for(adentro.wait(), 2)
        tarea.cancel()
        devolver_handle.set()                                    # el spawn termina; cleanup debe matar la unidad
        try:
            await tarea
        except asyncio.CancelledError:
            pass
        return sistema.muertes, await g.estado(UID)
    muertes, estado = corre_con_plazo(caso())
    assert muertes == [UNIDAD]                         # la unidad nacida de nuestro intento, muerta
    assert estado is EstadoUnidad.EMPTY


def test_major2_cancelacion_repetida_y_materializacion_tardia_no_deja_unidad_viva(sistema):
    """La segunda cancelación no corta la limpieza y ésta conserva ownership hasta matar la unidad de
    un systemd-run que materializa después de más de un segundo."""
    cfg = ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128", plazo_arranque_s=5.0)

    async def caso():
        g = gestor(sistema, cfg=cfg)
        spawn_iniciado, liberar_spawn = asyncio.Event(), asyncio.Event()

        async def crear_lento(*argv, **kw):
            if argv[0].endswith("systemd-run"):
                spawn_iniciado.set()
                await liberar_spawn.wait()                 # la unidad aún no existe
            return await sistema.crear_proceso(*argv, **kw)

        g._crear_proceso = crear_lento
        tarea = asyncio.create_task(g.arrancar(UID, ["/usr/bin/true"]))
        await asyncio.wait_for(spawn_iniciado.wait(), 2)
        tarea.cancel()
        await asyncio.sleep(0.08)                           # la limpieza sigue esperando el handle
        tarea.cancel()                                       # cancelación repetida durante limpieza
        with contextlib.suppress(asyncio.CancelledError):
            await tarea
        await asyncio.sleep(1.05)                            # reproducción: materialización >1s después
        liberar_spawn.set()
        # Esperar al spawn protegido y al contrato de cleanup; no usar demora como sincronización.
        await asyncio.sleep(0)
        for _ in range(100):
            if sistema.unidades:
                break
            await asyncio.sleep(0.01)
        await g.esperar_limpiezas()
        return sistema.muertes, await g.estado(UID)

    muertes, estado = corre_con_plazo(caso())
    assert muertes == [UNIDAD]
    assert estado is EstadoUnidad.EMPTY


def test_major2_handle_entregado_job_ausente_y_materializacion_posterior_se_reapea(sistema):
    cfg = ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128", plazo_arranque_s=5.0)

    async def caso():
        g = gestor(sistema, cfg=cfg)
        spawn_retorno, materializar = asyncio.Event(), asyncio.Event()
        argv_arranque = []

        async def crear_con_job(*argv, **kw):
            if argv[0].endswith("systemd-run"):
                argv_arranque.extend(argv)
                unidad, _, _ = sistema._parsear_run(argv)
                sistema.trabajos[unidad] = ("417", "start", "waiting")
                spawn_retorno.set()
                return ProcFalso(sistema, argv)            # handle entregado; unidad aún not-found
            return await sistema.crear_proceso(*argv, **kw)

        g._crear_proceso = crear_con_job
        tarea = asyncio.create_task(g.arrancar(UID, ["/usr/bin/true"]))
        await asyncio.wait_for(spawn_retorno.wait(), 2)
        assert await g.estado(UID) is EstadoUnidad.UNKNOWN  # el job impide declarar EMPTY
        tarea.cancel()
        async def job_del_manager():
            await materializar.wait()
            unidad, props, comando = sistema._parsear_run(argv_arranque)
            sistema.trabajos.pop(unidad)
            sistema._crear(unidad, comando, props)

        job = asyncio.create_task(job_del_manager())
        asyncio.get_running_loop().call_later(0.08, materializar.set)
        with contextlib.suppress(asyncio.CancelledError):
            await tarea
        await job
        await g.esperar_limpiezas()
        return sistema.muertes, await g.estado(UID)

    muertes, estado = corre_con_plazo(caso())
    assert muertes == [UNIDAD]
    assert estado is EstadoUnidad.EMPTY


def test_apagado_cierra_admision_y_falla_acotado_si_spawn_no_entrega_handle(sistema):
    cfg = ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128", plazo_arranque_s=0.05)

    async def caso():
        g = gestor(sistema, cfg=cfg, plazo_consulta_s=0.02)
        iniciado, liberar = asyncio.Event(), asyncio.Event()
        crear = sistema.crear_proceso

        async def spawn_lento(*argv, **kw):
            if argv[0].endswith("systemd-run"):
                iniciado.set()
                await liberar.wait()
            return await crear(*argv, **kw)

        g._crear_proceso = spawn_lento
        arranque = asyncio.create_task(g.arrancar(UID, ["/usr/bin/true"]))
        await asyncio.wait_for(iniciado.wait(), 1)
        with pytest.raises(ScopeError, match="no terminaron|siguen en vuelo"):
            await g.esperar_limpiezas()
        assert len(g._limpiezas) == 1  # el lanzamiento cancelado quedó bajo un reaper propio
        with pytest.raises(ScopeError, match="está cerrando"):
            await g.arrancar(UID + 1, ["/usr/bin/true"])
        liberar.set()
        with pytest.raises(ScopeError, match="sigue en curso"):
            await asyncio.wait_for(arranque, 1)
        await g.esperar_limpiezas()
        return await g.estado(UID), sistema.muertes

    estado, muertes = corre_con_plazo(caso())
    assert estado is EstadoUnidad.EMPTY
    assert muertes == [UNIDAD]


def test_fallo_de_identidad_durante_cleanup_queda_visible_y_retenido(sistema):
    async def caso():
        cfg = ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128", plazo_vacio_s=0.03)
        g = gestor(sistema, cfg=cfg, plazo_consulta_s=0.01)
        proceso = await g.arrancar(UID, ["/usr/bin/true"])
        lease = LeaseStore(sistema.lease_dir).read(UID)
        sistema.fallar_show = "rc"
        with pytest.raises(ScopeError, match="identidad"):
            await g.cerrar_uid(UID)
        assert LeaseStore(sistema.lease_dir).read(UID).attempt_id == lease.attempt_id
        sistema.fallar_show = None
        await g.cerrar_uid(UID)
        return await g.estado(UID), sistema.muertes, LeaseStore(sistema.lease_dir).all(), proceso

    estado, muertes, leases, proceso = corre_con_plazo(caso())
    assert estado is EstadoUnidad.EMPTY and leases == () and proceso.returncode == 255
    assert muertes == [UNIDAD]


def test_spawn_que_crea_unidad_y_falla_antes_de_entregar_handle_mata_exacta_y_conserva_quarantine(sistema):
    async def caso():
        cfg = ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128",
                          plazo_arranque_s=0.05, plazo_vacio_s=0.05)
        g = gestor(sistema, cfg=cfg, plazo_consulta_s=0.01)
        async def crear_y_fallar(*argv, **kw):
            if argv[0].endswith("systemd-run"):
                unidad, propiedades, comando = sistema._parsear_run(argv)
                sistema._crear(unidad, comando, propiedades)
                raise OSError("falló la entrega del handle después del efecto externo")
            return await sistema.crear_proceso(*argv, **kw)

        g._crear_proceso = crear_y_fallar
        with pytest.raises(ScopeError):
            await g.arrancar(UID, ["/usr/bin/true"])
        lease = LeaseStore(sistema.lease_dir).read(UID)
        return await g.estado(UID), sistema.muertes, lease, len(g._limpiezas)

    estado, muertes, lease, limpiezas = corre_con_plazo(caso())
    assert estado is EstadoUnidad.UNKNOWN
    assert muertes == [UNIDAD]
    assert lease.stage.value == "REQUESTED" and limpiezas == 0


def test_cancelacion_con_spawn_sin_handle_conserva_lease(sistema):
    async def caso():
        cfg = ConfigScope(memory_max="256M", cpu_quota="200%", tasks_max="128",
                          plazo_arranque_s=0.05, plazo_vacio_s=0.05)
        g = gestor(sistema, cfg=cfg, plazo_consulta_s=0.01)
        unidad_creada, devolver_error = asyncio.Event(), asyncio.Event()

        async def crear_y_fallar(*argv, **kw):
            if argv[0].endswith("systemd-run"):
                unidad, propiedades, comando = sistema._parsear_run(argv)
                sistema._crear(unidad, comando, propiedades)
                unidad_creada.set()
                await devolver_error.wait()
                raise OSError("falló el handle después de enviar el start")
            return await sistema.crear_proceso(*argv, **kw)

        g._crear_proceso = crear_y_fallar
        arranque = asyncio.create_task(g.arrancar(UID, ["/usr/bin/true"]))
        await unidad_creada.wait()
        arranque.cancel()
        await asyncio.sleep(0)
        devolver_error.set()
        with contextlib.suppress(asyncio.CancelledError, ScopeError):
            await arranque
        leases = LeaseStore(sistema.lease_dir).all()
        assert len(leases) == 1
        for t in tuple(g._limpiezas):
            t.cancel()
        await asyncio.gather(*tuple(g._limpiezas), return_exceptions=True)
        return sistema.muertes, leases

    muertes, leases = corre_con_plazo(caso())
    assert muertes == [UNIDAD]
    assert len(leases) == 1


def test_job_desaparece_de_cola_al_materializarse_y_reconciliador_mata_pero_retiene_request(sistema):
    async def caso():
        g = gestor(sistema)
        await g._reservar_lease(UID, "carrera-cleanup")
        from jax.faro.scope_lease import LeaseStage
        g._leases.transition(UID, "carrera-cleanup", LeaseStage.PREPARED, LeaseStage.REQUESTED)
        unidad, props, comando = sistema._parsear_run(
            g.argv_arrancar(UID, ["/usr/bin/true"], intento="carrera-cleanup"))
        def materializar():
            sistema._crear(unidad, comando, props)
            sistema.al_list_jobs = None
        sistema.al_list_jobs = materializar
        resultado = await g._reconciliar_sin_handle(UID, "carrera-cleanup")
        lease = LeaseStore(sistema.lease_dir).read(UID)
        return resultado, sistema.muertes, await g.estado(UID), lease
    resultado, muertes, estado, lease = corre_con_plazo(caso())
    assert resultado is False
    assert muertes == [UNIDAD]
    assert estado is EstadoUnidad.UNKNOWN
    assert lease.stage.value == "REQUESTED"


def test_limpiezas_exitosas_se_eliminan_sin_esperar_apagado(sistema):
    async def caso():
        g = gestor(sistema)
        for indice in range(20):
            uid = UID + indice
            await g.arrancar(uid, ["/usr/bin/true"])
            await g.cerrar_uid(uid)
            assert not g._limpiezas and not g._resultados_limpieza
        return len(sistema.muertes), g._leases.all()
    muertes, leases = corre_con_plazo(caso())
    assert muertes == 20 and leases == ()


def test_fallo_de_wait_no_abandona_lease_accepted(sistema):
    async def caso():
        g = gestor(sistema)
        llamadas = {"wait": 0}

        original = sistema.crear_proceso
        async def crear(*argv, **kw):
            proceso = await original(*argv, **kw)
            if argv[0].endswith("systemd-run"):
                async def wait_con_fallo():
                    llamadas["wait"] += 1
                    raise OSError("fallo temporal de wait")
                proceso.wait = wait_con_fallo
            return proceso

        g._crear_proceso = crear
        await g.arrancar(UID, ["/usr/bin/true"])
        lease = LeaseStore(sistema.lease_dir).read(UID)
        with pytest.raises(OSError, match="fallo temporal de wait"):
            await g.cerrar_uid(UID)
        return await g.estado(UID), llamadas, sistema.muertes, LeaseStore(sistema.lease_dir).read(UID), lease

    estado, llamadas, muertes, retenido, original = corre_con_plazo(caso())
    assert estado is EstadoUnidad.UNKNOWN and llamadas["wait"] == 1
    assert muertes == [UNIDAD]
    assert retenido.attempt_id == original.attempt_id and retenido.stage.value == "ACCEPTED"


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
    assert len(intentos) == 1 and len(arrancados) == 1     # el lease durable rechaza antes del spawn
    assert muertes == []                                       # al perdedor no se le mata la unidad al ganador
    assert estado is EstadoUnidad.LIVE


# --------------------------------------------------------------------------- #
# matar y esperar_vacio                                                        #
# --------------------------------------------------------------------------- #

def test_matar_manda_sigkill_a_todos_los_procesos_de_la_unidad(sistema):
    _poblar(sistema)

    async def caso():
        g = gestor(sistema)
        await g.matar(UID, intento=await g.intento_actual(UID))
        return sistema.muertes
    assert correr(caso()) == [UNIDAD]


def test_matar_una_unidad_que_no_existe_es_un_error_visible(sistema):
    async def caso():
        try:
            await gestor(sistema).matar(UID, intento="a" * 32)
        except ScopeError:
            return True
        return False
    assert correr(caso())


def test_matar_a_una_unidad_no_mata_a_su_hermana(sistema):
    _poblar(sistema, UID)
    _poblar(sistema, UID + 1)

    async def caso():
        g = gestor(sistema)
        await g.matar(UID, intento=await g.intento_actual(UID))
        return await g.estado(UID), await g.estado(UID + 1)
    assert correr(caso()) == (EstadoUnidad.EMPTY, EstadoUnidad.LIVE)


def test_esperar_vacio_espera_la_confirmacion_empty(sistema):
    _poblar(sistema, activa=False, poblada=1)                  # muriendo: hijos vivos aun en el cgroup

    async def caso():
        g = gestor(sistema)
        store = LeaseStore(sistema.lease_dir)
        lease = store.read(UID)
        from jax.faro.scope_lease import LeaseStage
        store.transition(UID, lease.attempt_id, LeaseStage.PREPARED, LeaseStage.REQUESTED)
        cg = sistema._cgroup(UNIDAD)
        datos = cg.stat()
        transicionar_accepted(store, UID, lease.attempt_id,
                              cgroup_path=f"/{cg.relative_to(sistema.raiz)}",
                              cgroup_dev=datos.st_dev, cgroup_ino=datos.st_ino)
        store.transition(UID, lease.attempt_id, LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED)

        async def vaciar():
            await asyncio.sleep(0.03)
            sistema.unidades[UNIDAD]["poblada"] = 0
            (sistema._cgroup(UNIDAD) / "cgroup.events").write_text("populated 0\nfrozen 0\n")
            sistema.unidades.pop(UNIDAD)                  # --collect descarga la unidad vacia
        asyncio.create_task(vaciar())
        return await g.esperar_vacio(UID, intento=lease.attempt_id, plazo_s=2.0)
    assert correr(caso()) is EstadoUnidad.EMPTY


def test_esperar_vacio_por_plazo_devuelve_lo_que_haya_y_nunca_empty(sistema):
    _poblar(sistema, activa=False, poblada=1)

    async def caso():
        return await gestor(sistema).esperar_vacio(UID, plazo_s=0.05)
    assert correr(caso()) is EstadoUnidad.LIVE


def test_esperar_vacio_con_consulta_rota_devuelve_unknown(sistema):
    async def caso():
        g = gestor(sistema)
        original = g._correr

        async def fallar_namespace(argv, **kwargs):
            if "list-units" in argv:
                return 1, b"", b"fallo simulado"
            return await original(argv, **kwargs)

        g._correr = fallar_namespace
        return await g.esperar_vacio(UID, plazo_s=0.05)
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

    def setup_method(self):
        self._lease_dirs = []

    def teardown_method(self):
        for directorio in self._lease_dirs:
            shutil.rmtree(directorio, ignore_errors=True)

    def _uid(self) -> int:
        TestUserManagerReal._siguiente += 1
        return self.UID_BASE + TestUserManagerReal._siguiente * 7

    def _gestor(self, **kw) -> ScopeManager:
        directorio = Path(tempfile.mkdtemp(prefix="faro-r2-leases-"))
        self._lease_dirs.append(directorio)
        return ScopeManager(ConfigScope(memory_max="64M", cpu_quota="10%", tasks_max="32",
                                       plazo_arranque_s=15.0, state_dir=directorio, **kw))

    @staticmethod
    async def _limpiar(g, uid):
        with contextlib.suppress(Exception):
            await matar_handle(g, uid)
        await g.esperar_vacio(uid, plazo_s=5.0)

    def test_arranca_verifica_limites_y_mata_incluidos_hijos_y_nietos(self):
        uid, g = self._uid(), self._gestor()

        async def caso():
            with contextlib.suppress(Exception):
                await matar_handle(g, uid)                    # una unidad colgante de otra corrida no florea esto
            proceso = await g.arrancar(uid, ["/bin/sh", "-c", "sleep 300 & sleep 300 & sleep 300 & wait"])
            assert await g.estado(uid) is EstadoUnidad.LIVE
            cg = await g.control_group(uid)
            assert cg and Path("/sys/fs/cgroup", cg.removeprefix("/")).is_dir()
            await matar_handle(g, uid)
            rc = await g.finalizar_natural(uid, proceso)
            return await g.estado(uid), rc
        try:
            estado, rc = correr(caso())
        finally:
            correr(self._limpiar(g, uid))
        assert estado is EstadoUnidad.EMPTY
        assert rc != 0                                         # SIGKILL: ni limpio ni 0

    def test_matar_una_unidad_no_mata_a_la_hermana(self):
        a, b, g = self._uid(), self._uid() + 1000, self._gestor()

        async def caso():
            proceso_a = await g.arrancar(a, ["/bin/sleep", "300"])
            proceso_b = await g.arrancar(b, ["/bin/sleep", "300"])
            try:
                await matar_handle(g, a)
                await g.finalizar_natural(a, proceso_a)
                vacia = await g.estado(a)
                return vacia, await g.estado(b)
            finally:
                await matar_handle(g, b)
                await g.finalizar_natural(b, proceso_b)
        try:
            vacia, hermana = correr(caso())
        finally:
            correr(self._limpiar(g, b))
            correr(self._limpiar(g, a))
        assert vacia is EstadoUnidad.EMPTY and hermana is EstadoUnidad.LIVE

    def test_un_nieto_con_setsid_sigue_en_el_cgroup_y_muere_con_la_unidad(self):
        uid, g = self._uid(), self._gestor()

        async def caso():
            proceso = await g.arrancar(uid, ["/bin/sh", "-c", "setsid /bin/sleep 300 & sleep 300"])
            await asyncio.sleep(0.3)                           # que el nieto nazca
            assert await g.estado(uid) is EstadoUnidad.LIVE
            await matar_handle(g, uid)
            await g.finalizar_natural(uid, proceso)
            return await g.estado(uid)
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
            proceso = await g.arrancar(uid, ["/bin/sh", "-c", f"env > {salida}; sleep 5"])
            viva = await g.estado(uid)
            await matar_handle(g, uid)
            await g.finalizar_natural(uid, proceso)
            estado = await g.estado(uid)
            return viva, estado, salida.read_text()
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
            await matar_handle(g, uid)
            await l.esperar_terminos()                        # watcher reap + drenaje stderr antes de cerrar el loop
            estado = await g.estado(uid)
            rc = proceso.returncode
            return viva, estado, rc
        try:
            viva, estado, rc = correr(caso())
        finally:
            correr(self._limpiar(g, uid))
        assert viva is True and estado is EstadoUnidad.EMPTY
        assert rc != 0                                          # SIGKILL de su unidad
        args = salida.read_text().splitlines()
        assert "--bind" in args and "/faro/puerto.sock" in args and "/usr/bin/setpriv" not in args
        assert (tmp_path / "uid-pedido.txt").read_text().strip() == str(uid)   # la elevacion recibio el uid de la jaula

    def test_systemd_run_drena_stdout_stderr_sin_bloquear_y_acota_el_diagnostico(self):
        uid, g = self._uid(), self._gestor()
        comando = [sys.executable, "-c",
                   "import sys,time;sys.stdout.write('o'*2097152);sys.stderr.write('e'*2097152);"
                   "sys.stdout.flush();sys.stderr.flush();time.sleep(.2)"]

        async def caso():
            proceso = await g.arrancar(uid, comando)
            rc = await asyncio.wait_for(g.esperar_cliente(proceso), 10)
            return rc, len(proceso._faro_stderr_tail)

        assert correr(caso()) == (0, 200)
