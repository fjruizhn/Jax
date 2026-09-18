#!/usr/bin/env python3
# scripts/ejecutor_contratos/probar_c4.py
"""Simulacro real de C4 en hall9000, re-ejecutable por un tercero.

Qué se prueba: dos procesos de la cuenta que, si viven 5 s, dejan una marca en su home:
uno DESPRENDIDO (`setsid nohup`, sobrevive al cierre de su ssh) y uno al otro lado de un
`ssh -tt` (con pty, en primer plano). Primero una ronda de CONTROL sin freno: las dos marcas
TIENEN que aparecer; si no, el escenario no corría y una ronda con freno «pasaría» sola.
Después, con el freno puesto: a LECTURAS_S ningún proceso de la cuenta (`ps -u`) y, al
soltarlo, ninguna marca. Dos rondas con ENTRE_RONDAS_S de por medio.

El freno por omisión es la PAUSA DEL EJECUTOR (`JAX_EJECUTOR_PAUSA`, la que pone C5): sólo
frena al Ejecutor. `--freno global` usa el interruptor de JAX: la Mesa responde 423 mientras
dura; sólo con la Mesa sin uso, y se suelta en el acto.

Con `--remoto <máquina>` (plan 5 Parte B, Task 7) el escenario es el del Ejecutor de verdad:
desde la cuenta en hall9000, un `ssh -tt` a la cuenta en la remota (primer plano, con pty) y
un `setsid nohup` lanzado en la remota. Matar la cuenta local corta el ssh (SIGHUP a lo que
tiene pty); lo desprendido en la remota sólo lo mata el barrido con la llave del freno
(`JAX_EJECUTOR_FRENO_REMOTOS` tiene que incluir la máquina). En cada lectura también se cuenta
`ps -u <cuenta>` en la remota por ssh de administrador.

Uso: set -a; . <(sudo -n cat /etc/jax/.env); set +a
     PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_c4.py [--freno ejecutor|global]
         [--lecturas 1,6] [--rondas 2] [--remoto <máquina de la política>]
Sale 0 sólo con `c4_vivo=true`. No toca la DB. Siempre suelta el freno que puso (finally).
"""
import argparse
import asyncio
import json
import os
import pwd
import secrets
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jax.core import interruptor  # noqa: E402
from jax.ejecutor.contratos import cuenta_axioma, formato, pausa, politica  # noqa: E402

ESPERA_CONTROL_S = 11
VIDA_S = 8
ARRANQUE_MAX_S = 5  # el escenario remoto pasa por dos ssh: el freno se pone recién cuando se lo ve corriendo
ENTRE_RONDAS_S = 10
LIMPIEZA_S = 45  # lo que se le da a la remota para quedar SIN ningún proceso de la cuenta


# Vivos = no zombis: un proceso muerto que su padre todavía no cosechó ya no corre nada.
_VIVOS = "ps -u {cuenta} -o stat=,pid= | awk '$1 !~ /^Z/ {{print $2}}'"
# Con cmd: el gestor de sesión de systemd no ejecuta el trabajo del Ejecutor y logind lo repone
# apenas se lo mata; se distingue del trabajo, pero se exige que también se haya ido al cerrar.
_VIVOS_CMD = "ps -u {cuenta} -o stat=,pid=,cmd= | awk '$1 !~ /^Z/ {{$1=\"\"; print substr($0,2)}}'"
GESTOR_DE_SESION = ("/usr/lib/systemd/systemd --user", "/lib/systemd/systemd --user", "(sd-pam)")


def clasificar_procesos(salida: str) -> tuple[list[str], int]:
    """(pids que ejecutan trabajo, total de procesos vivos de la cuenta)."""
    trabajo, total = [], 0
    for linea in salida.splitlines():
        partes = linea.strip().split(None, 1)
        if not partes:
            continue
        total += 1
        pid, cmd = partes[0], (partes[1] if len(partes) > 1 else "")
        if cmd.strip() not in GESTOR_DE_SESION:
            trabajo.append(pid)
    return trabajo, total


def _procesos(cuenta: str) -> list[str]:
    r = subprocess.run(["sh", "-c", _VIVOS.format(cuenta=shlex.quote(cuenta))], capture_output=True, text=True)
    return r.stdout.split()


def _con_tty(argv: list[str]) -> list[str]:
    return [argv[0], "-tt", *argv[1:]]


def _en_la_remota(remoto, orden: str, *, tty: bool = False) -> str:
    """Lo que la cuenta, desde hall9000, corre en la remota (con su propia llave, como el Ejecutor)."""
    return (f"ssh {'-tt ' if tty else ''}-o BatchMode=yes -o StrictHostKeyChecking=yes -p {int(remoto.puerto)} "
            f"{shlex.quote(os.environ['JAX_EJECUTOR_CUENTA'])}@{shlex.quote(remoto.ip)} {shlex.quote(orden)}")


def argv_conteo_remoto(remoto, cuenta: str, admin: str, *, contar: bool = True) -> list[str]:
    """El conteo va como ROOT: con `/proc` montado `hidepid` (bridge, 2026-09-17) el administrador
    ve CERO procesos de la cuenta aunque estén vivos, y la prueba mediría su propia ceguera.
    Con `contar=False` devuelve la lista (pid y comando), que es lo que se clasifica."""
    ps = _VIVOS_CMD.format(cuenta=shlex.quote(cuenta))
    orden = "sudo -n sh -c " + shlex.quote(ps + " | wc -l" if contar else ps)
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=yes",
            "-p", str(remoto.puerto), f"{admin}@{remoto.ip}", orden]


def _remotos_detalle(remoto, cuenta: str) -> tuple[list[str], int]:
    """(pids de trabajo, total) en la remota. Ante cualquier error: (["?"], -1), que falla cerrado."""
    argv = argv_conteo_remoto(remoto, cuenta, os.environ["JAX_EJECUTOR_ADMIN_USUARIO"], contar=False)
    r = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    if r.returncode != 0:
        return ["?"], -1
    return clasificar_procesos(r.stdout)


def _procesos_remotos(remoto, cuenta: str) -> int:
    return _remotos_detalle(remoto, cuenta)[1]


async def _lanzar(c, nonce: str, remoto=None) -> list[asyncio.subprocess.Process]:
    desprendido = (f"setsid nohup sh -c 'sleep {VIDA_S} && touch \"$HOME/.c4-desprendido-{nonce}\"' "
                   f">/dev/null 2>&1 </dev/null & echo lanzado")
    anidado = f"sleep {VIDA_S} && touch \"$HOME/.c4-anidado-{nonce}\""
    if remoto is not None:
        desprendido = _en_la_remota(remoto, desprendido)
        anidado = _en_la_remota(remoto, anidado, tty=True)
    procs = []
    for argv in (cuenta_axioma.ssh_a_la_cuenta(c, desprendido), _con_tty(cuenta_axioma.ssh_a_la_cuenta(c, anidado))):
        procs.append(await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL, start_new_session=True))
    return procs


async def _cerrar(procs, tope_s: float) -> None:
    for p in procs:
        try:
            await asyncio.wait_for(p.wait(), tope_s)
        except asyncio.TimeoutError:
            p.kill()
            await p.wait()


async def _marcas(c, nonce: str, remoto=None) -> list[str] | None:
    orden = f'ls -1 "$HOME"/.c4-*-{nonce} 2>/dev/null; rm -f "$HOME"/.c4-*-{nonce}; true'
    rc, salida, _ = await cuenta_axioma.correr_en_la_cuenta(
        c, orden if remoto is None else _en_la_remota(remoto, orden), tope_s=30)
    if rc != 0:
        return None
    return sorted(l.rsplit("/", 1)[1] for l in salida.decode().split())


class Freno:
    def __init__(self, tipo: str):
        self.tipo = tipo
        self.ruta = pausa.ruta_de_la_pausa() if tipo == "ejecutor" else interruptor.ruta_del_interruptor()

    def puesto(self) -> bool:
        return pausa.pausa_puesta(self.ruta) if self.tipo == "ejecutor" else interruptor.interruptor_activo()

    def poner(self, nonce: str) -> bool:
        datos = {"origen": "probar_c4", "nonce": nonce}
        if self.tipo == "ejecutor":
            return pausa.poner_pausa(self.ruta, datos)
        return interruptor.escribir_pausa(self.ruta, json.dumps(datos))

    def soltar(self) -> None:
        if self.tipo == "ejecutor":
            self.ruta.unlink(missing_ok=True)
        else:
            interruptor.borrar_pausa(self.ruta)


async def ronda(c, freno: Freno, fallos: list, lecturas, *, con_freno: bool, n: int, remoto=None) -> None:
    nonce = secrets.token_hex(6)
    inicio = time.monotonic()
    procs = await _lanzar(c, nonce, remoto)
    etiqueta = ("ronda", n)
    # El escenario tiene que estar CORRIENDO (local y, con --remoto, en la remota) antes de frenar:
    # si no, una ronda con freno pasaría sin haber tenido nada que matar.
    vivos, remotos = [], None
    while time.monotonic() - inicio < ARRANQUE_MAX_S:
        await asyncio.sleep(0.5)
        vivos = await asyncio.to_thread(_procesos, c.nombre)
        remotos = await asyncio.to_thread(_procesos_remotos, remoto, c.nombre) if remoto is not None else None
        if vivos and (remoto is None or (remotos or 0) >= 2):
            break
    else:
        fallos.append(("escenario_no_arranco", (etiqueta, ("procesos", len(vivos)), ("procesos_remotos", remotos))))
        await _cerrar(procs, VIDA_S + 5)
        await _marcas(c, nonce, remoto)
        return
    if not con_freno:
        await asyncio.sleep(max(0.0, ESPERA_CONTROL_S - (time.monotonic() - inicio)))
        await _cerrar(procs, 10)
        marcas = await _marcas(c, nonce, remoto)
        print(formato.campos((("contrato", "c4"), etiqueta, ("control", True), ("procesos", len(vivos)),
                              ("procesos_remotos", remotos), ("marcas", ",".join(marcas or ())))))
        if marcas is None or len(marcas) != 2:
            fallos.append(("control_fallido", (etiqueta, ("marcas", ",".join(marcas or ())))))
        return
    if not freno.poner(nonce):
        fallos.append(("freno_ya_puesto", (etiqueta,)))
        await _cerrar(procs, 10)
        return
    puesto_en = time.monotonic()
    try:
        transcurrido = 0.0
        for lectura in lecturas:
            await asyncio.sleep(lectura - transcurrido)
            transcurrido = lectura
            vivos = await asyncio.to_thread(_procesos, c.nombre)
            trabajo, remotos = ([], None) if remoto is None else await asyncio.to_thread(
                _remotos_detalle, remoto, c.nombre)
            print(formato.campos((("contrato", "c4"), etiqueta, ("freno", freno.tipo), ("a_los_s", lectura),
                                  ("medido_s", round(time.monotonic() - puesto_en, 3)), ("procesos", len(vivos)),
                                  ("procesos_remotos", remotos), ("trabajo_remoto", len(trabajo)))))
            if vivos:
                fallos.append(("procesos_vivos", (etiqueta, ("a_los_s", lectura), ("pids", ",".join(vivos)))))
            if trabajo:
                fallos.append(("procesos_remotos_vivos", (etiqueta, ("a_los_s", lectura),
                                                          ("pids", ",".join(trabajo)))))
        await asyncio.sleep(max(0.0, VIDA_S + 1 - (time.monotonic() - inicio)))
    finally:
        freno.soltar()
    await _cerrar(procs, 10)
    await asyncio.sleep(1)
    if remoto is not None:
        # El gestor de sesión (systemd --user) no ejecuta el trabajo, pero puede lanzar unidades:
        # se le da LIMPIEZA_S para irse; si sigue ahí, la remota NO quedó vacía y es un fallo.
        limite = time.monotonic() + LIMPIEZA_S
        while True:
            _, total = await asyncio.to_thread(_remotos_detalle, remoto, c.nombre)
            if total == 0 or time.monotonic() > limite:
                break
            await asyncio.sleep(2)
        print(formato.campos((("contrato", "c4"), etiqueta, ("remota_vacia_en_s", round(LIMPIEZA_S if total else
                                                                                       time.monotonic() - puesto_en, 1)),
                              ("procesos_remotos", total))))
        if total != 0:
            fallos.append(("remota_no_quedo_vacia", (etiqueta, ("cuantos", total))))
    marcas = await _marcas(c, nonce, remoto)
    if marcas is None:
        fallos.append(("sin_acceso_despues", (etiqueta,)))
    elif marcas:
        fallos.append(("marca_creada", (etiqueta, ("marcas", ",".join(marcas)))))


async def principal(args) -> int:
    c = cuenta_axioma.cuenta_desde_entorno()
    freno = Freno(args.freno)
    lecturas = tuple(float(x) for x in args.lecturas.split(","))
    fallos: list = []
    remoto = None
    if args.remoto:
        p = politica.cargar(os.environ["JAX_EJECUTOR_POLITICA"], uid_de_la_cuenta=pwd.getpwnam(c.nombre).pw_uid)
        (remoto,) = [h for h in p.hosts if h.nombre == args.remoto and not h.es_local]
        habilitados = [n.strip() for n in os.environ.get("JAX_EJECUTOR_FRENO_REMOTOS", "").split(",")]
        if args.remoto not in habilitados:
            fallos.append(("remoto_sin_freno_habilitado", (("remoto", args.remoto),)))
    activo = await asyncio.to_thread(subprocess.run, ["systemctl", "is-active", "ejecutor-freno.service"],
                                     capture_output=True, text=True)
    if activo.stdout.strip() != "active":
        fallos.append(("freno_inactivo", ()))
    if (interruptor.interruptor_activo() or pausa.pausa_puesta(pausa.ruta_de_la_pausa())
            or await asyncio.to_thread(_procesos, c.nombre)
            or (remoto is not None and await asyncio.to_thread(_procesos_remotos, remoto, c.nombre) != 0)):
        print(formato.campos((("c4_vivo", False), ("codigo", "estado_inicial_no_limpio"))))
        return 2
    try:
        await ronda(c, freno, fallos, lecturas, con_freno=False, n=0, remoto=remoto)
        if not any(f[0] in ("control_fallido", "escenario_no_arranco") for f in fallos):
            for i in range(1, args.rondas + 1):
                if i > 1:
                    await asyncio.sleep(ENTRE_RONDAS_S)
                await ronda(c, freno, fallos, lecturas, con_freno=True, n=i, remoto=remoto)
    finally:
        if freno.puesto():
            freno.soltar()
    for codigo, datos in fallos:
        print(formato.campos((("contrato", "c4"), ("codigo", codigo)) + tuple(datos)))
    print(formato.campos((("c4_vivo", not fallos), ("freno", args.freno), ("rondas", args.rondas),
                          ("remoto", args.remoto or ""))))
    return 0 if not fallos else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freno", choices=("ejecutor", "global"), default="ejecutor")
    p.add_argument("--lecturas", default="1,6")
    p.add_argument("--rondas", type=int, default=2)
    p.add_argument("--remoto", default="")
    sys.exit(asyncio.run(principal(p.parse_args())))
