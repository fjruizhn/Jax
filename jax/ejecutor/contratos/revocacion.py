# jax/ejecutor/contratos/revocacion.py
"""Revocación de un golpe (C6). Spec 2026-09-15 §4: «un script quita la llave de
axioma en todas las máquinas del inventario; revocar y comprobar que no entra a ninguna».

- Primero las remotas, en paralelo; hall9000 al final (desde la cuenta se comprueban
  las remotas; revocada hall9000, ya no se entra a la cuenta).
- «No entra» es SÓLO `Permission denied (publickey)`. Una máquina que no contesta es
  NO_VERIFICABLE: no se sabe si se revocó, y se dice.
- Un fallo al revocar (salida, código o excepción) es ERROR_AL_REVOCAR, nunca silencio.
"""
from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass

MARCAS_DE_ACCESO = ("ejecutor-axioma", "ejecutor-controlador")
MARCA_FRENO = "ejecutor-freno"

ENTRA, DENEGADO, INALCANZABLE = "entra", "denegado", "inalcanzable"
REVOCADA, SIGUE_ENTRANDO, NO_VERIFICABLE, ERROR_AL_REVOCAR = (
    "revocada", "sigue_entrando", "no_verificable", "error_al_revocar")


@dataclass(frozen=True)
class Resultado:
    host: str
    estado: str
    detalle: tuple = ()


def clasificar_intento(rc: int, stderr: bytes) -> str:
    if rc == 0:
        return ENTRA
    if rc == 255 and b"Permission denied (publickey" in stderr:
        return DENEGADO
    return INALCANZABLE


def remoto_probar_entrada(h, cuenta: str) -> str:
    return (f"LC_ALL=C ssh -o BatchMode=yes -o PreferredAuthentications=publickey -o ConnectTimeout=5 "
            f"-o StrictHostKeyChecking=yes -p {int(h.puerto)} {shlex.quote(cuenta)}@{shlex.quote(h.ip)} true")


def argv_admin(h, usuario_admin: str, remoto: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
            "-p", str(h.puerto), f"{usuario_admin}@{h.ip}", remoto]


async def _uno(h, revocar_en, probar_entrada) -> Resultado:
    try:
        rc, salida, _ = await revocar_en(h)
    except Exception as exc:  # fail-soft: se reporta ERROR_AL_REVOCAR para esta máquina y el resto sigue; nunca cuenta como revocada
        return Resultado(h.nombre, ERROR_AL_REVOCAR, (("tipo", type(exc).__name__),))
    texto = salida.decode(errors="replace").strip()
    if rc != 0 or not texto.startswith("revocar=ok") or not texto.endswith("quedan=0"):
        return Resultado(h.nombre, ERROR_AL_REVOCAR, (("salida", texto),))
    rc, _, errores = await probar_entrada(h)
    intento = clasificar_intento(rc, errores)
    estado = {DENEGADO: REVOCADA, ENTRA: SIGUE_ENTRANDO}.get(intento, NO_VERIFICABLE)
    return Resultado(h.nombre, estado, (("stderr", errores.decode(errors="replace").strip()[:300]),) if estado != REVOCADA else ())


async def revocar_todas(hosts, *, revocar_en, probar_entrada) -> tuple:
    remotas = [h for h in hosts if not h.es_local]
    locales = [h for h in hosts if h.es_local]
    primero = await asyncio.gather(*(_uno(h, revocar_en, probar_entrada) for h in remotas))
    despues = [await _uno(h, revocar_en, probar_entrada) for h in locales]
    return tuple(primero) + tuple(despues)
