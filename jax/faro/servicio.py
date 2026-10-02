"""El punto de entrada del servicio del Faro: FALLA CERRADO si falta la bitacora durable (reauditoria R3).

`arrancar` valida TODO antes de aceptar una sola ejecucion, en este orden, y cualquier fallo aborta:
1. configuracion completa (`ConfigBitacoraDB` primero: sin la de la bitacora no hay servicio, no se deja a la
   operacion; luego `ConfigFaro` y `ConfigPuerto`);
2. no corre como root;
3. el paquete verifica y se carga (`cargar_paquete`: dueño, sin escritura ajena, sha256, oid), sin git;
4. el pool de la base se crea y una SONDA (`servicio_iniciado`) se INSERTA en la tabla encadenada: si la base
   esta caida, la tabla no existe o el usuario no puede insertar, no arranca.

Devuelve un `Servicio`: su `bitacora` ya lleva el emisor de la tabla (primero) y el del log, y `crear_puerto`
construye el `ServidorPuerto` de UNA ejecucion con esa bitacora. QUIEN pide crear una `Ejecucion` (canal de
control autenticado) y el lanzador de jaula con uid distinto de `faro` son el paso 0.3c del plan: este modulo
no abre ningun canal por el que un motor pueda crear o alterar una ejecucion.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .bitacora import Bitacora, emisor_logger
from .bitacora_db import ConfigBitacoraDB, EmisorTabla, crear_pool
from .config import ConfigFaro, ConfigFaroInvalida, ConfigPuerto
from .identidad import Ejecucion
from .logs import asegurar_logging
from .paquete import PaqueteCargado, cargar_paquete
from .transporte import ServidorPuerto

logger = logging.getLogger(__name__)


@dataclass
class Servicio:
    cfg_faro: ConfigFaro
    cfg_puerto: ConfigPuerto
    cfg_db: ConfigBitacoraDB
    paquete: PaqueteCargado
    pool: object
    emisor: EmisorTabla
    bitacora: Bitacora
    solo_pruebas_mismo_uid: bool = False

    def crear_puerto(self, ejecucion: Ejecucion) -> ServidorPuerto:
        return ServidorPuerto(self.cfg_puerto, ejecucion, self.paquete, self.bitacora,
                              solo_pruebas_mismo_uid=self.solo_pruebas_mismo_uid)

    async def cerrar(self) -> None:
        self.pool.close()
        await self.pool.wait_closed()


async def arrancar(env: Mapping[str, str], *, crear_pool: Callable = crear_pool,
                   solo_pruebas_mismo_uid: bool = False) -> Servicio:
    """`solo_pruebas_mismo_uid` es SOLO PARA PRUEBAS (argumento, nunca entorno): ver `ServidorPuerto`."""
    asegurar_logging()
    cfg_db = ConfigBitacoraDB.desde_entorno(env)             # lo primero: sin bitacora durable no hay servicio
    cfg_faro = ConfigFaro.desde_entorno(env, solo_pruebas_duenio_igual_servicio=solo_pruebas_mismo_uid)
    cfg_puerto = ConfigPuerto.desde_entorno(env)
    if os.geteuid() == 0:
        raise ConfigFaroInvalida("el servicio del Faro no corre como root: usa el usuario sin privilegios `faro`")
    paquete = await asyncio.to_thread(cargar_paquete, cfg_faro)
    pool = await crear_pool(cfg_db)
    try:
        emisor = EmisorTabla(pool)
        bitacora = Bitacora(emisores=[emisor, emisor_logger])
        await bitacora.registrar("servicio_iniciado", sha_paquete=paquete.sha, pid=os.getpid())   # la sonda
    except BaseException:
        pool.close()
        await pool.wait_closed()
        raise
    return Servicio(cfg_faro, cfg_puerto, cfg_db, paquete, pool, emisor, bitacora, solo_pruebas_mismo_uid)


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    """Valida el arranque. 0 = el servicio arranco (y, sin canal de control todavia, se cierra); 2 = no arranco."""
    try:
        servicio = asyncio.run(_verificar(env if env is not None else os.environ))
    except Exception as exc:  # fail-closed: el servicio no arranca; el motivo va a stderr y el codigo de salida es 2
        print(f"faro-servicio: NO ARRANCA: {exc}", file=sys.stderr)
        return 2
    del servicio
    return 0


async def _verificar(env: Mapping[str, str]) -> None:
    s = await arrancar(env)
    await s.cerrar()


if __name__ == "__main__":
    sys.exit(main())
