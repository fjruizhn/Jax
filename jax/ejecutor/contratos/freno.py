# jax/ejecutor/contratos/freno.py
"""Freno en vuelo de la cuenta del Ejecutor (C4). Corre como ROOT: ejecutor-freno.service.

Mientras esté puesto CUALQUIERA de los dos frenos —el interruptor de JAX
(jax/core/interruptor.py, frente B, `JAX_KILL_SWITCH_PATH`, también la ruta heredada) o
la pausa propia del Ejecutor (jax/ejecutor/contratos/pausa.py, `JAX_EJECUTOR_PAUSA`, la
que pone C5)— en cada vuelta de INTERVALO_DE_SONDEO:
1. escribe "1" en cgroup.kill de user-<uid>.slice: todo lo que entró por ssh, incluido
   lo desprendido con setsid/nohup y lo que nazca mientras se mata (cgroup v2);
2. barre /proc y mata todo proceso con uid real o efectivo de la cuenta, esté donde
   esté (p. ej. fuera de su slice);
3. en los repasos REPASOS_S desde que se puso (0, 2 y 10 s) lanza —sin esperarlo— el
   barrido de cada máquina remota HABILITADA con la llave de comando forzado
   (ejecutor-freno-remoto). Un repaso tardío atrapa lo que arrancó justo al frenar.

Remotas habilitadas a propósito (`JAX_EJECUTOR_FRENO_REMOTOS`, nombres de la política
separados por coma): sin la llave del freno instalada en la remota, cada freno sería un
ssh fallido contra un servidor de clientes. `remotos_cargados` en el latido es true sólo
con TODAS las remotas de la política habilitadas; el arranque del Ejecutor (plan 6) lo exige.

FAIL-CLOSED: cualquiera de los dos frenos sin configurar = puesto. Política ilegible:
se sigue matando local y se declara en el latido. La cuenta nunca puede ser root, una
cuenta de sistema (uid < 1000) ni el administrador: en ese caso no se mata local y el
latido dice uid_resuelto=false (un error de config no mata la sesión de Fernando).

Sólo biblioteca estándar (se instala en /opt/ejecutor/lib y corre con python3 -I).
"""
from __future__ import annotations

import json
import logging
import os
import pwd
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from jax.core import interruptor
from jax.ejecutor.contratos import pausa
from jax.ejecutor.contratos import politica

REPASOS_S = (0.0, 2.0, 10.0)
UID_MINIMO = 1000
INTERRUPTOR_DE_JAX = "interruptor_de_jax"
PAUSA_DEL_EJECUTOR = "pausa_del_ejecutor"
_TOPE_BARRIDO_S = 15
_SALIDA_REMOTA_OK = b"freno_remoto=ok quedan=0"
log = logging.getLogger("ejecutor.freno")


@dataclass(frozen=True)
class Remoto:
    nombre: str
    ip: str
    puerto: int


@dataclass(frozen=True)
class ConfigFreno:
    uid: int | None
    cuenta: str
    cgroup: Path
    proc: Path
    llave: Path
    known_hosts: Path
    remotos: tuple
    estado: Path


def frenos_puestos() -> tuple:
    """Cuáles de los dos frenos están puestos. Sin configurar = puesto."""
    puestos = []
    try:
        if interruptor.interruptor_activo():
            puestos.append(INTERRUPTOR_DE_JAX)
    except interruptor.InterruptorSinConfigurar:
        puestos.append(INTERRUPTOR_DE_JAX)
    try:
        if pausa.pausa_puesta(pausa.ruta_de_la_pausa()):
            puestos.append(PAUSA_DEL_EJECUTOR)
    except pausa.PausaSinConfigurar:
        puestos.append(PAUSA_DEL_EJECUTOR)
    return tuple(puestos)


def ruta_cgroup_kill(cgroup: Path, uid: int) -> Path:
    return cgroup / "user.slice" / f"user-{uid}.slice" / "cgroup.kill"


def matar_cgroup(cgroup: Path, uid: int) -> bool:
    try:
        ruta_cgroup_kill(cgroup, uid).write_text("1")
    except FileNotFoundError:
        return False  # sin sesiones de la cuenta no existe su slice: no hay nada que matar ahí
    except OSError as exc:  # fail-soft: el barrido de /proc de esta misma vuelta mata igual; queda en el journal
        log.error("freno cgroup_fallo tipo=%s", type(exc).__name__)
        return False
    return True


def pids_de(proc: Path, uid: int) -> list[int]:
    pids = []
    for entrada in proc.iterdir():
        if not entrada.name.isdigit():
            continue
        try:
            for linea in (entrada / "status").read_text().splitlines():
                if linea.startswith("Uid:"):
                    real, efectivo = (int(x) for x in linea.split()[1:3])
                    if uid in (real, efectivo):
                        pids.append(int(entrada.name))
                    break
        except (FileNotFoundError, ProcessLookupError):
            continue  # el proceso terminó entre listar y leer
    return pids


def matar_pids(pids, matar=os.kill) -> int:
    muertos = 0
    for pid in pids:
        try:
            matar(pid, signal.SIGKILL)
            muertos += 1
        except ProcessLookupError:
            continue
    return muertos


def argv_barrido(r: Remoto, cfg: ConfigFreno) -> list[str]:
    return ["ssh", "-i", str(cfg.llave), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={cfg.known_hosts}",
            "-o", "ConnectTimeout=5", "-p", str(r.puerto), f"{cfg.cuenta}@{r.ip}"]


def barrer(r: Remoto, cfg: ConfigFreno, correr=subprocess.run) -> str:
    try:
        hecho = correr(argv_barrido(r, cfg), capture_output=True, timeout=_TOPE_BARRIDO_S, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return "tiempo_agotado"
    except OSError as exc:  # fail-soft: sin ssh no hay barrido; se devuelve "fallo" y el journal lo dice
        log.error("freno barrido=%s tipo=%s", r.nombre, type(exc).__name__)
        return "fallo"
    return "ok" if hecho.returncode == 0 and hecho.stdout.strip() == _SALIDA_REMOTA_OK else "fallo"


def _escribir_estado(ruta: Path, estado: dict) -> None:
    fd, temporal = tempfile.mkstemp(dir=ruta.parent, prefix=".estado-")
    with os.fdopen(fd, "w") as f:
        json.dump(estado, f)
    os.chmod(temporal, 0o644)  # el arranque (fruiz) lo lee; nadie más que root lo escribe
    os.replace(temporal, ruta)


class Freno:
    def __init__(self, cfg: ConfigFreno, *, frenos=frenos_puestos, reloj=time.monotonic, lanzar_barrido=None,
                 matar=os.kill, remotos_cargados: bool = True):
        self.cfg = cfg
        self._frenos = frenos
        self._reloj = reloj
        self._matar = matar
        self._remotos_cargados = remotos_cargados
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(cfg.remotos)))
        self._lanzar = lanzar_barrido or self._barrer_en_fondo
        self._desde = None
        self._repasos = set()

    def _barrer_en_fondo(self, remotos) -> None:
        for r in remotos:
            futuro = self._pool.submit(barrer, r, self.cfg)
            futuro.add_done_callback(lambda f, n=r.nombre: log.warning("freno barrido=%s resultado=%s", n, f.result()))

    def paso(self) -> dict:
        puestos = self._frenos()
        activo = bool(puestos)
        salida = {"activo": activo}
        if not activo:
            self._desde, self._repasos = None, set()
        else:
            salida["frenos"] = list(puestos)
            ahora = self._reloj()
            if self._desde is None:
                self._desde = ahora
            if self.cfg.uid is not None:
                salida["cgroup"] = matar_cgroup(self.cfg.cgroup, self.cfg.uid)
                salida["muertos"] = matar_pids(pids_de(self.cfg.proc, self.cfg.uid), self._matar)
            for i, repaso in enumerate(REPASOS_S):
                if i not in self._repasos and ahora - self._desde >= repaso:
                    self._repasos.add(i)
                    if self.cfg.remotos:
                        self._lanzar(self.cfg.remotos)
                    break
        _escribir_estado(self.cfg.estado, {"momento": time.time(), "activo": activo,
                                           "frenos": list(puestos),
                                           "remotos_cargados": self._remotos_cargados,
                                           "remotos": [r.nombre for r in self.cfg.remotos],
                                           "uid_resuelto": self.cfg.uid is not None})
        return salida


def _uid_del_administrador(nombre: str) -> int | None:
    try:
        return pwd.getpwnam(nombre).pw_uid
    except KeyError:
        return None


def _uid_de_la_cuenta(cuenta: str, admin: str) -> int | None:
    try:
        uid = pwd.getpwnam(cuenta).pw_uid
    except KeyError:
        return None
    if uid < UID_MINIMO or uid == _uid_del_administrador(admin):
        log.error("freno cuenta_rechazada uid=%d", uid)
        return None
    return uid


def config_desde_entorno(env) -> tuple[ConfigFreno, bool]:
    cuenta = env["JAX_EJECUTOR_CUENTA"]
    uid = _uid_de_la_cuenta(cuenta, env["JAX_EJECUTOR_ADMIN_USUARIO"])
    habilitados = [n.strip() for n in env.get("JAX_EJECUTOR_FRENO_REMOTOS", "").split(",") if n.strip()]
    remotos, cargados = (), False
    try:
        p = politica.cargar(env["JAX_EJECUTOR_POLITICA"], uid_de_la_cuenta=-1 if uid is None else uid)
    except (OSError, ValueError) as exc:  # fail-soft: sin política se sigue matando local; el latido dice remotos_cargados=false y el arranque se niega
        log.error("freno politica_ilegible tipo=%s", type(exc).__name__)
    else:
        de_la_politica = {h.nombre: h for h in p.hosts if not h.es_local}
        remotos = tuple(Remoto(h.nombre, h.ip, h.puerto) for n, h in de_la_politica.items() if n in habilitados)
        cargados = bool(de_la_politica) and set(habilitados) == set(de_la_politica)
    cfg = ConfigFreno(uid=uid, cuenta=cuenta, cgroup=Path("/sys/fs/cgroup"), proc=Path("/proc"),
                      llave=Path(env["JAX_EJECUTOR_FRENO_LLAVE"]), known_hosts=Path(env["JAX_EJECUTOR_FRENO_KNOWN_HOSTS"]),
                      remotos=remotos, estado=Path(env["JAX_EJECUTOR_FRENO_ESTADO"]))
    return cfg, cargados


def principal() -> int:
    logging.basicConfig(level=logging.INFO)
    cfg, cargados = config_desde_entorno(os.environ)
    freno = Freno(cfg, remotos_cargados=cargados)
    log.info("freno arranca uid_resuelto=%s remotos=%s remotos_cargados=%s",
             cfg.uid is not None, ",".join(r.nombre for r in cfg.remotos), cargados)
    activo_antes = False
    while True:
        estado = freno.paso()
        # Una línea al poner el freno y una por vuelta que mató algo: no cuatro por segundo.
        if estado["activo"] and (not activo_antes or estado.get("muertos")):
            log.warning("freno activo frenos=%s cgroup=%s muertos=%s", ",".join(estado["frenos"]),
                        estado.get("cgroup"), estado.get("muertos"))
        elif activo_antes and not estado["activo"]:
            log.warning("freno suelto")
        activo_antes = estado["activo"]
        time.sleep(interruptor.INTERVALO_DE_SONDEO)


if __name__ == "__main__":
    sys.exit(principal())
