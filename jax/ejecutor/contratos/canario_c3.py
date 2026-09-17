# jax/ejecutor/contratos/canario_c3.py
"""Prueba viva de C3 (spec 2026-09-15 §4: «desde la jaula, escribir el registro falla»).

Como la cuenta del Ejecutor: intenta anexar al registro y abrir conexiones a los
servicios que el cerco tiene que tapar (`sondas`), y como control abre la del proxy,
que tiene que funcionar. Desde fuera: el tamaño del registro no cambió, tiene la
bandera append-only del sistema de archivos, y su cadena cuadra.

Si C3 ya estaba roto, el intento de escritura deja un byte suelto al final del
registro: el proxy no vuelve a abrirlo (RegistroCorrupto) y el Ejecutor queda
bloqueado hasta que alguien lo mire. Es a propósito: fallar ruidoso y cerrado.
"""
from __future__ import annotations

import fcntl
import os
import shlex
import struct
from pathlib import Path

from jax.ejecutor.contratos import cuenta_axioma
from jax.ejecutor.contratos.fallo import Fallo
from jax.ejecutor.contratos.registro import verificar_cadena

FS_IOC_GETFLAGS = 0x80086601
FS_APPEND_FL = 0x00000020
_TOPE_S = 60


def banderas(ruta) -> int:
    fd = os.open(ruta, os.O_RDONLY | os.O_CLOEXEC)
    try:
        return struct.unpack("<i", fcntl.ioctl(fd, FS_IOC_GETFLAGS, b"\0" * 8)[:4])[0]
    finally:
        os.close(fd)


def _guion(registro: Path, puerto_proxy: int, sondas) -> str:
    q = shlex.quote
    partes = []
    for p in sondas:
        partes.append(f'if timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/{int(p)}" 2>/dev/null; '
                      f'then echo "sonda={int(p)} abierta"; else echo "sonda={int(p)} cerrada"; fi')
    partes.append(f'if timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/{int(puerto_proxy)}" 2>/dev/null; '
                  'then echo proxy=abierto; else echo proxy=cerrado; fi')
    partes.append(f"if printf x 2>/dev/null >> {q(str(registro))}; "
                  "then echo registro_escrito=si; else echo registro_escrito=no; fi")
    return "; ".join(partes)


async def verificar_c3(c, *, registro: Path, puerto_proxy: int, sondas, correr=cuenta_axioma.correr_en_la_cuenta,
                       leer_banderas=banderas) -> tuple:
    antes = os.stat(registro).st_size
    rc, salida, _ = await correr(c, _guion(registro, puerto_proxy, sondas), tope_s=_TOPE_S)
    if rc == 255:
        return (Fallo("c3", "cuenta_inalcanzable", (("rc", rc),)),)
    vistas = dict(linea.split("=", 1) for linea in salida.decode(errors="replace").split("\n") if "=" in linea)
    fallos = []
    if vistas.get("registro_escrito") != "no":
        fallos.append(Fallo("c3", "registro_escribible_desde_la_jaula"))
    if os.stat(registro).st_size != antes:
        fallos.append(Fallo("c3", "registro_cambio_desde_la_jaula"))
    for p in sondas:
        if not salida.decode(errors="replace").count(f"sonda={int(p)} cerrada"):
            fallos.append(Fallo("c3", "cerco_abierto", (("puerto", int(p)),)))
    if vistas.get("proxy") != "abierto":
        fallos.append(Fallo("c3", "proxy_inalcanzable"))
    if not leer_banderas(registro) & FS_APPEND_FL:
        fallos.append(Fallo("c3", "registro_sin_append_only"))
    cadena = verificar_cadena(registro)
    if not cadena.ok:
        fallos.append(Fallo("c3", "cadena_rota", (("linea", cadena.primer_error), ("codigo", cadena.codigo))))
    return tuple(fallos)
