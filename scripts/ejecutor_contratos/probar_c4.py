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

Uso: set -a; . /etc/jax/.env; set +a
     PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_c4.py [--freno ejecutor|global]
         [--lecturas 1,6] [--rondas 2]
Sale 0 sólo con `c4_vivo=true`. No toca la DB. Siempre suelta el freno que puso (finally).
"""
import argparse
import asyncio
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jax.core import interruptor  # noqa: E402
from jax.ejecutor.contratos import cuenta_axioma, formato, pausa  # noqa: E402

ESPERA_CONTROL_S = 8
VIDA_S = 5
ENTRE_RONDAS_S = 10


def _procesos(cuenta: str) -> list[str]:
    r = subprocess.run(["ps", "-u", cuenta, "-o", "pid="], capture_output=True, text=True)
    return r.stdout.split()


def _con_tty(argv: list[str]) -> list[str]:
    return [argv[0], "-tt", *argv[1:]]


async def _lanzar(c, nonce: str) -> list[asyncio.subprocess.Process]:
    desprendido = (f"setsid nohup sh -c 'sleep {VIDA_S} && touch \"$HOME/.c4-desprendido-{nonce}\"' "
                   f">/dev/null 2>&1 </dev/null & echo lanzado")
    anidado = f"sleep {VIDA_S} && touch \"$HOME/.c4-anidado-{nonce}\""
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


async def _marcas(c, nonce: str) -> list[str] | None:
    rc, salida, _ = await cuenta_axioma.correr_en_la_cuenta(
        c, f'ls -1 "$HOME"/.c4-*-{nonce} 2>/dev/null; rm -f "$HOME"/.c4-*-{nonce}; true', tope_s=30)
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


async def ronda(c, freno: Freno, fallos: list, lecturas, *, con_freno: bool, n: int) -> None:
    nonce = secrets.token_hex(6)
    inicio = time.monotonic()
    procs = await _lanzar(c, nonce)
    await asyncio.sleep(1)
    etiqueta = ("ronda", n)
    if not con_freno:
        vivos = _procesos(c.nombre)
        await asyncio.sleep(ESPERA_CONTROL_S - 1)
        await _cerrar(procs, 10)
        marcas = await _marcas(c, nonce)
        print(formato.campos((("contrato", "c4"), etiqueta, ("control", True), ("procesos_a_1s", len(vivos)),
                              ("marcas", ",".join(marcas or ())))))
        if marcas is None or len(marcas) != 2 or not vivos:
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
            vivos = _procesos(c.nombre)
            print(formato.campos((("contrato", "c4"), etiqueta, ("freno", freno.tipo), ("a_los_s", lectura),
                                  ("medido_s", round(time.monotonic() - puesto_en, 3)), ("procesos", len(vivos)))))
            if vivos:
                fallos.append(("procesos_vivos", (etiqueta, ("a_los_s", lectura), ("pids", ",".join(vivos)))))
        await asyncio.sleep(max(0.0, VIDA_S + 1 - (time.monotonic() - inicio)))
    finally:
        freno.soltar()
    await _cerrar(procs, 10)
    await asyncio.sleep(1)
    marcas = await _marcas(c, nonce)
    if marcas is None:
        fallos.append(("sin_acceso_despues", (etiqueta,)))
    elif marcas:
        fallos.append(("marca_creada", (etiqueta, ("marcas", ",".join(marcas)))))


async def principal(args) -> int:
    c = cuenta_axioma.cuenta_desde_entorno()
    freno = Freno(args.freno)
    lecturas = tuple(float(x) for x in args.lecturas.split(","))
    fallos: list = []
    activo = subprocess.run(["systemctl", "is-active", "ejecutor-freno.service"], capture_output=True, text=True)
    if activo.stdout.strip() != "active":
        fallos.append(("freno_inactivo", ()))
    if interruptor.interruptor_activo() or pausa.pausa_puesta(pausa.ruta_de_la_pausa()) or _procesos(c.nombre):
        print(formato.campos((("c4_vivo", False), ("codigo", "estado_inicial_no_limpio"))))
        return 2
    try:
        await ronda(c, freno, fallos, lecturas, con_freno=False, n=0)
        if not any(f[0] == "control_fallido" for f in fallos):
            for i in range(1, args.rondas + 1):
                if i > 1:
                    await asyncio.sleep(ENTRE_RONDAS_S)
                await ronda(c, freno, fallos, lecturas, con_freno=True, n=i)
    finally:
        if freno.puesto():
            freno.soltar()
    for codigo, datos in fallos:
        print(formato.campos((("contrato", "c4"), ("codigo", codigo)) + tuple(datos)))
    print(formato.campos((("c4_vivo", not fallos), ("freno", args.freno), ("rondas", args.rondas))))
    return 0 if not fallos else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freno", choices=("ejecutor", "global"), default="ejecutor")
    p.add_argument("--lecturas", default="1,6")
    p.add_argument("--rondas", type=int, default=2)
    sys.exit(asyncio.run(principal(p.parse_args())))
