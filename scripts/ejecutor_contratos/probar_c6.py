#!/usr/bin/env python3
# scripts/ejecutor_contratos/probar_c6.py
"""Prueba REAL y NO DESTRUCTIVA de C6 en UNA máquina.

1. Precondición: ningún proceso de la cuenta vivo en esa máquina (revocar los mata).
2. Respaldar el archivo root de llaves.
3. Abrir una sesión CENTINELA de la cuenta (un `sleep` por ssh) y verla viva.
4. Revocar (`ejecutor-revocar`) y comprobar que no entra: sólo `Permission denied (publickey)`.
5. Comprobar que la centinela MURIÓ y que no queda ningún proceso de la cuenta.
6. REPONER (finally: también si algo falló), `cmp` contra el respaldo, y comprobar que vuelve a entrar.

Uso: set -a; . /etc/jax/.env; set +a
     PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_c6.py <nombre-de-máquina>
Sale 0 sólo con `c6_probado=true`.
"""
import asyncio
import os
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jax.ejecutor.contratos import cuenta_axioma, formato, revocacion  # noqa: E402
from revocar import argv_como_admin, comandos, correr, inventario  # noqa: E402

CENTINELA_S = 300  # la sesión centinela vive como mucho esto; revocar tiene que matarla mucho antes
ESPERA_S = 15      # cuánto se espera a que la centinela aparezca o muera


async def _root(h, env, orden):
    rc, salida, errores = await correr(argv_como_admin(h, env, orden), 60)
    return rc, salida.decode(errors="replace").strip(), errores.decode(errors="replace").strip()


async def _vivos(h, env, cuenta):
    rc, salida, _ = await _root(h, env, f"ps -u {shlex.quote(cuenta)} -o pid= | wc -l")
    return int(salida) if rc == 0 and salida.isdigit() else -1


async def _centinela(h, env, c):
    comando = f"sleep {CENTINELA_S}"
    if h.es_local:
        argv = cuenta_axioma.ssh_a_la_cuenta(c, comando)
    else:
        argv = cuenta_axioma.ssh_a_la_cuenta(c, revocacion.remoto_probar_entrada(h, c.nombre, comando))
    return await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                                                start_new_session=True)


async def _esperar(condicion):
    limite = time.monotonic() + ESPERA_S
    while time.monotonic() < limite:
        if await condicion():
            return True
        await asyncio.sleep(0.25)
    return False


async def principal(nombre: str, env) -> int:
    hosts = [x for x in inventario(env) if x.nombre == nombre]
    if len(hosts) != 1:
        print(formato.campos((("codigo", "maquina_fuera_del_inventario"), ("host", nombre))))
        return 2
    (h,) = hosts
    c = cuenta_axioma.cuenta_desde_entorno(env)
    archivo = env["JAX_EJECUTOR_LLAVES_ROOT"]
    respaldo = f"{archivo}.prueba-c6-{time.time_ns()}"
    revocar_en, probar_entrada = comandos(env)
    pasos = []

    vivos = await _vivos(h, env, c.nombre)
    if vivos != 0:
        print(formato.campos((("codigo", "cuenta_ocupada_o_admin_inaccesible"), ("host", nombre), ("vivos", vivos))))
        return 2
    rc, _, err = await _root(h, env, f"cp -a {shlex.quote(archivo)} {shlex.quote(respaldo)}")
    if rc != 0:
        print(formato.campos((("codigo", "respaldo_imposible"), ("host", nombre), ("stderr", err))))
        return 2
    centinela = None
    try:
        centinela = await _centinela(h, env, c)

        async def centinela_viva():
            return centinela.returncode is None and await _vivos(h, env, c.nombre) > 0

        pasos.append(("centinela_viva", await _esperar(centinela_viva)))
        (resultado,) = await revocacion.revocar_todas([h], revocar_en=revocar_en, probar_entrada=probar_entrada)
        pasos.append(("revocacion", resultado.estado))

        async def centinela_muerta():
            try:
                await asyncio.wait_for(centinela.wait(), 0.25)
            except asyncio.TimeoutError:
                return False
            return await _vivos(h, env, c.nombre) == 0

        pasos.append(("sesion_abierta_muere", await _esperar(centinela_muerta)))
    finally:
        if centinela is not None and centinela.returncode is None:
            centinela.kill()
            await centinela.wait()
        # La precondición fue «cero procesos de la cuenta»: lo que quede vivo lo dejó esta
        # prueba (una centinela que la revocación no mató). Sin pty, cortar el cliente ssh
        # no la mata en la máquina: se mata como root y se dice.
        if await _vivos(h, env, c.nombre) != 0:
            rc, _, _ = await _root(h, env, f"pkill -KILL -u {shlex.quote(c.nombre)}; true")
            pasos.append(("limpieza_de_restos", rc == 0))
        orden = (f"install -m 0644 -o root -g root {shlex.quote(respaldo)} {shlex.quote(archivo)} && "
                 f"cmp {shlex.quote(respaldo)} {shlex.quote(archivo)} && rm {shlex.quote(respaldo)}")
        rc, _, _ = await _root(h, env, orden)
        pasos.append(("repuesta_identica", rc == 0))
    rc, _, errores = await probar_entrada(h)
    pasos.append(("vuelve_a_entrar", revocacion.clasificar_intento(rc, errores) == revocacion.ENTRA))
    for clave, valor in pasos:
        print(formato.campos((("host", nombre), (clave, valor))))
    ok = pasos == [("centinela_viva", True), ("revocacion", revocacion.REVOCADA), ("sesion_abierta_muere", True),
                   ("repuesta_identica", True), ("vuelve_a_entrar", True)]
    print(formato.campos((("c6_probado", ok), ("host", nombre))))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(formato.campos((("codigo", "argumentos"),)))
        sys.exit(2)
    sys.exit(asyncio.run(principal(sys.argv[1], os.environ)))
