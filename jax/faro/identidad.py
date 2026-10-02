"""Quien pide, fijado por QUIEN CREO LA EJECUCION y por el socket, nunca por el pedido.

Spec §3 («La identidad sale del socket: quien pide la fija el servicio que creo la
ejecucion. Nunca viaja en el cuerpo del pedido, asi que no se repite el problema de
`motor_registry/models.py:49-56`») y §4 («La fija quien autentica, nunca el modelo»).

- `Ejecucion` la construye el SERVICIO que lanza el motor y abre el socket
  (`/run/faro/<run_id>.sock`): usuario, tenant, faceta, motor, pipeline, `run_id`,
  `entry_point`, id de correlacion y el uid que va a correr la jaula.
- `Identidad` es la `Ejecucion` mas lo que el KERNEL dice del par de cada conexion
  (`SO_PEERCRED`: pid, uid, gid) y un id de conexion. Es lo unico que el Puerto
  registra y consulta. Ningun campo sale de un argumento ni de `_meta`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import ConfigFaroInvalida

# `run_id` entra en una ruta de socket: un patron estricto es lo que impide `../` y similares.
_RE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


# El tenant entra en claves de topes y en rutas de bitacora: un patron estricto, UNO solo para el control y los topes.
RE_TENANT = re.compile(r"^[A-Za-z0-9_.:@-]{1,64}$")


@dataclass(frozen=True)
class Ejecucion:
    run_id: str
    usuario: str
    tenant: str
    faceta: str
    motor: str
    pipeline: str
    entry_point: str
    id_correlacion: str
    uid_esperado: int

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not _RE_RUN_ID.fullmatch(self.run_id):
            raise ConfigFaroInvalida(f"run_id invalido: {self.run_id!r}")
        for campo in ("usuario", "tenant", "faceta", "motor", "pipeline", "entry_point", "id_correlacion"):
            valor = getattr(self, campo)
            if not isinstance(valor, str) or not valor.strip():
                raise ConfigFaroInvalida(f"la ejecucion necesita {campo}: lo fija el servicio que la crea")
        if not RE_TENANT.fullmatch(self.tenant):
            raise ConfigFaroInvalida("el tenant solo admite letras, digitos y _ . : @ - (hasta 64): es la clave de sus topes")
        if not isinstance(self.uid_esperado, int) or isinstance(self.uid_esperado, bool) or self.uid_esperado < 0:
            raise ConfigFaroInvalida("uid_esperado tiene que ser un uid (entero >= 0)")


@dataclass(frozen=True)
class Identidad:
    ejecucion: Ejecucion
    id_conexion: str
    peer_pid: int
    peer_uid: int
    peer_gid: int

    def campos(self) -> dict:
        e = self.ejecucion
        return {
            "run_id": e.run_id, "usuario": e.usuario, "tenant": e.tenant, "faceta": e.faceta,
            "motor": e.motor, "pipeline": e.pipeline, "entry_point": e.entry_point,
            "id_correlacion": e.id_correlacion, "id_conexion": self.id_conexion,
            "peer_pid": self.peer_pid, "peer_uid": self.peer_uid, "peer_gid": self.peer_gid,
        }
