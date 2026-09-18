#!/usr/bin/env python3
# scripts/ejecutor_contratos/revocar.py
"""C6 — REVOCA el acceso del Ejecutor en TODAS las máquinas del inventario y corta sus sesiones.

Uso de emergencia:  set -a; . <(sudo -n cat /etc/jax/.env); set +a
                    PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/revocar.py --confirmo-revocar-todo
Sin la bandera no hace nada. Reponer el acceso es a mano, con los archivos .revocadas-* de cada máquina.
Lee /etc/jax/.env (producción) para JAX_EJECUTOR_* y la política exportada; no toca la DB.
El inventario se carga con `politica.cargar` (dueño y permisos comprobados): una política que
la cuenta pudo reescribir no decide a qué IP se manda la revocación.
"""
import argparse
import asyncio
import os
import pwd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jax.ejecutor.contratos import cuenta_axioma, formato, politica, revocacion  # noqa: E402


async def correr(argv, tope_s, env=None):
    p = await asyncio.create_subprocess_exec(*argv, env=env, stdout=asyncio.subprocess.PIPE,
                                             stderr=asyncio.subprocess.PIPE, start_new_session=True)
    try:
        salida, errores = await asyncio.wait_for(p.communicate(), tope_s)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        p.kill()
        await p.wait()
        raise
    return p.returncode, salida, errores


def argv_como_admin(h, env, remoto_root: str) -> list:
    """`remoto_root` corre como root en la máquina: sudo local, o ssh de administrador + sudo."""
    import shlex
    if h.es_local:
        return ["sudo", "-n", "sh", "-c", remoto_root]
    return revocacion.argv_admin(h, env["JAX_EJECUTOR_ADMIN_USUARIO"], f"sudo -n sh -c {shlex.quote(remoto_root)}")


def comandos(env):
    import shlex
    c = cuenta_axioma.cuenta_desde_entorno(env)
    archivo = env["JAX_EJECUTOR_LLAVES_ROOT"]

    async def revocar_en(h):
        orden = f"/usr/local/sbin/ejecutor-revocar {shlex.quote(archivo)} {shlex.quote(c.nombre)}"
        return await correr(argv_como_admin(h, env, orden), 60)

    async def probar_entrada(h, comando="true"):
        if h.es_local:
            return await correr(cuenta_axioma.ssh_a_la_cuenta(c, comando), 30, env={**os.environ, "LC_ALL": "C"})
        return await cuenta_axioma.correr_en_la_cuenta(c, revocacion.remoto_probar_entrada(h, c.nombre, comando),
                                                       tope_s=30)

    return revocar_en, probar_entrada


def inventario(env):
    c = cuenta_axioma.cuenta_desde_entorno(env)
    return politica.cargar(c.politica, uid_de_la_cuenta=pwd.getpwnam(c.nombre).pw_uid).hosts


async def principal(args, env) -> int:
    hosts = [h for h in inventario(env) if not args.solo or h.nombre == args.solo]
    if not hosts:
        print(formato.campos((("codigo", "maquina_fuera_del_inventario"),)))
        return 2
    revocar_en, probar_entrada = comandos(env)
    resultados = await revocacion.revocar_todas(hosts, revocar_en=revocar_en, probar_entrada=probar_entrada)
    for r in resultados:
        print(formato.campos((("host", r.host), ("estado", r.estado)) + tuple(r.detalle)))
    ok = all(r.estado == revocacion.REVOCADA for r in resultados)
    print(formato.campos((("c6_revocado_en_todas", ok),)))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirmo-revocar-todo", action="store_true")
    ap.add_argument("--solo", help="una sola máquina")
    a = ap.parse_args()
    if not a.confirmo_revocar_todo:
        print(formato.campos((("codigo", "falta_confirmacion"),)))
        sys.exit(2)
    sys.exit(asyncio.run(principal(a, os.environ)))
