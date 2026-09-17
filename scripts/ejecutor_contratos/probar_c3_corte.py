#!/usr/bin/env python3
"""C3 VISTO FALLAR con el arnés real: con el registro no escribible, la herramienta NO corre.

Re-ejecutable por un tercero en hall9000 (necesita sudo para `chattr`):
  set -a; . /etc/jax/.env; set +a
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:las_manos .venv/bin/python scripts/ejecutor_contratos/probar_c3_corte.py

Dos corridas del arnés real, como la cuenta del Ejecutor y dentro de su jaula, contra
una instancia DE PRUEBA del proxy (registro propio en un directorio temporal de /var/tmp,
ext4; nunca el de producción) que escucha en JAX_EJECUTOR_CANARIO_PUERTO (el cerco lo
deja pasar) y habla con el upstream falso del canario (sin GPU, sin modelo):

1. control: registro escribible → la herramienta corre (su marca aparece), queda
   `herramienta_pedida` y `resultado_devuelto`, la cadena cuadra;
2. rotura: `chattr +i` sobre el registro con el proxy ya abierto → el proxy no puede
   anotar y corta el stream: la marca NO aparece, el upstream no recibe `tool_result`,
   el registro no crece.

Salida: una línea `clave=valor` por hecho y `c3_corta=true|false` al final.
No toca la DB ni el registro de producción.
"""
import asyncio
import os
import secrets
import shlex
import sys
import tempfile
from pathlib import Path

from jax.ejecutor import proxy_carril
from jax.ejecutor.contratos import cuenta_axioma, formato
from jax.ejecutor.contratos.canario_upstream import UpstreamCanario, guion_bash
from jax.ejecutor.contratos.registro import verificar_cadena

_TOPE_S = 180


async def _sudo(*args) -> int:
    proc = await asyncio.create_subprocess_exec("sudo", "-n", *args)
    return await proc.wait()


async def _marca_existe(c, marca: str) -> bool:
    rc, salida, _ = await cuenta_axioma.correr_en_la_cuenta(
        c, f"test -e ~/{shlex.quote(marca)} && echo si || echo no", tope_s=30)
    return salida.strip() == b"si"


async def _corrida(c, dir_prueba: Path, puerto: int, marca: str, romper: bool) -> dict:
    registro = dir_prueba / "registro.jsonl"
    tool_use_id = "toolu_c3_" + secrets.token_hex(4)
    async with UpstreamCanario([guion_bash(tool_use_id, f"touch ~/{marca}")], "127.0.0.1", 0) as up:
        cfg = proxy_carril.Config(upstream=f"http://127.0.0.1:{up.puerto}", raiz=dir_prueba / "locks", tope_s=60,
                                  host="127.0.0.1", puerto=puerto, registro=registro)
        servidor = await proxy_carril.arrancar(cfg)
        try:
            if romper and await _sudo("chattr", "+i", str(registro)) != 0:
                raise SystemExit(formato.campos((("error", "chattr_fallo"),)))
            antes = registro.stat().st_size
            try:
                rc, _, _ = await cuenta_axioma.correr_en_la_cuenta(
                    c, cuenta_axioma.remoto_claude(c, base_url=f"http://127.0.0.1:{puerto}", modelo="canario",
                                                   prompt="Ejecuta el comando.", herramientas="Bash"),
                    entrada=b"clave-de-prueba\n", tope_s=_TOPE_S)
            finally:
                if romper:
                    await _sudo("chattr", "-i", str(registro))
            eventos = registro.read_text().splitlines()
            return {"rc": rc, "marca": await _marca_existe(c, marca), "resultado_subio": tool_use_id in up.resultados,
                    "crecio": registro.stat().st_size != antes, "lineas": len(eventos),
                    "pedida_anotada": any(tool_use_id in e and "herramienta_pedida" in e for e in eventos),
                    "cadena_ok": verificar_cadena(registro).ok}
        finally:
            servidor.close()
            await servidor.wait_closed()


async def principal() -> int:
    c = cuenta_axioma.cuenta_desde_entorno()
    puerto = int(os.environ["JAX_EJECUTOR_CANARIO_PUERTO"])
    nonce = secrets.token_hex(6)
    marcas = (f".c3-control-{nonce}", f".c3-corte-{nonce}")
    with tempfile.TemporaryDirectory(prefix="c3-corte-", dir="/var/tmp") as tmp:
        try:
            for sub in ("control/locks", "corte/locks"):
                (Path(tmp) / sub).mkdir(parents=True)
            control = await _corrida(c, Path(tmp) / "control", puerto, marcas[0], romper=False)
            corte = await _corrida(c, Path(tmp) / "corte", puerto, marcas[1], romper=True)
        finally:
            await cuenta_axioma.correr_en_la_cuenta(
                c, "rm -f " + " ".join(f"~/{shlex.quote(m)}" for m in marcas), tope_s=30)
    for nombre, d in (("control", control), ("corte", corte)):
        print(formato.campos(tuple((f"{nombre}_{k}", v) for k, v in d.items())))
    control_ok = control["marca"] and control["resultado_subio"] and control["pedida_anotada"] and control["cadena_ok"]
    corta = not corte["marca"] and not corte["resultado_subio"] and not corte["crecio"] and corte["cadena_ok"]
    print(formato.campos((("control_ok", bool(control_ok)), ("c3_corta", bool(control_ok and corta)))))
    return 0 if control_ok and corta else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(principal()))
