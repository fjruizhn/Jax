"""El punto de entrada del servicio del Faro: FALLA CERRADO si falta la bitacora durable (reauditoria R3).

`arrancar` valida TODO antes de aceptar una sola ejecucion, en este orden, y cualquier fallo aborta:
1. configuracion completa (`ConfigBitacoraDB` primero: sin la de la bitacora no hay servicio, no se deja a la
   operacion; luego `ConfigFaro` y `ConfigPuerto`);
2. no corre como root;
3. el paquete verifica y se carga (`cargar_paquete`: dueño, sin escritura ajena, sha256, oid), sin git;
4. el pool de la base se crea y una SONDA (`servicio_iniciado`) se INSERTA en la tabla encadenada: si la base
   esta caida, la tabla no existe o el usuario no puede insertar, no arranca.

Devuelve un `Servicio`: su `bitacora` ya lleva el emisor de la tabla (primero) y el del log, y `crear_puerto`
construye el `ServidorPuerto` de UNA ejecucion con esa bitacora y con el UNICO presupuesto de bytes del
servicio (`Servicio.presupuesto`, compartido por todas las ejecuciones).

ANCLA DE LA CADENA: cuando haya destino (lo fija 0.10), `arrancar` debe lanzar
`publicar_anclas_periodicamente(emisor, publicar, intervalo_s)` y cancelarla en `cerrar`. Hasta entonces no
se publica nada y la cola de la cadena puede truncarse sin que se note (ver el plan, 0.3a). QUIEN pide crear una `Ejecucion` es el canal de control autenticado (`control.py`, 0.3c): `Servicio.control()`
es la unica puerta, y solo el uid del orquestador configurado pasa. El aviso inmediato de cada denegacion
(`aviso.py`, 0.3b) cuelga de la bitacora como observador. El `main` de este modulo solo VERIFICA el arranque
(no sirve): el bucle del servicio y su unidad de systemd son el paso 0.10.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .aviso import Avisador, ConfigAviso, leer_credenciales
from .bitacora import Bitacora, emisor_logger
from .bitacora_db import ConfigBitacoraDB, EmisorTabla, crear_pool
from .config import ConfigFaro, ConfigFaroInvalida, ConfigMemoria, ConfigPuerto
from .control import ConfigControl, ServidorControl, validar_directorio_control
from .identidad import Ejecucion
from .logs import asegurar_logging
from .paquete import PaqueteCargado, cargar_paquete
from .transporte import PresupuestoBytes, ServidorPuerto
from .herramientas.memoria import AdaptadorMemoria, crear_pool_memoria_prueba

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
    presupuesto: PresupuestoBytes
    solo_pruebas_mismo_uid: bool = False
    cfg_aviso: ConfigAviso | None = None
    cfg_control: ConfigControl | None = None
    avisador: Avisador | None = None
    cfg_memoria: ConfigMemoria | None = None
    pool_memoria: object | None = None
    adaptador_memoria: AdaptadorMemoria | None = None

    def control(self, jaula_viva=None) -> ServidorControl:
        """El canal de control autenticado (0.3c): el unico lugar donde nace una `Ejecucion`. Se usa con
        `async with servicio.control() as c:`; sus ejecuciones comparten el UNICO presupuesto del servicio."""
        return ServidorControl(self.cfg_control, self.crear_puerto, self.bitacora, jaula_viva=jaula_viva)

    def crear_puerto(self, ejecucion: Ejecucion) -> ServidorPuerto:
        return ServidorPuerto(self.cfg_puerto, ejecucion, self.paquete, self.bitacora, presupuesto=self.presupuesto,
                              solo_pruebas_mismo_uid=self.solo_pruebas_mismo_uid,
                              adaptador_memoria=self.adaptador_memoria)

    async def cerrar(self) -> None:
        if self.avisador is not None:
            await self.avisador.cerrar()        # entrega lo pendiente (con plazo) y nunca lanza
        if self.pool_memoria is not None:
            self.pool_memoria.close()
            await self.pool_memoria.wait_closed()
        self.pool.close()
        await self.pool.wait_closed()


async def arrancar(env: Mapping[str, str], *, crear_pool: Callable = crear_pool,
                   solo_pruebas_mismo_uid: bool = False) -> Servicio:
    """`solo_pruebas_mismo_uid` es SOLO PARA PRUEBAS (argumento, nunca entorno): ver `ServidorPuerto`."""
    asegurar_logging()
    cfg_db = ConfigBitacoraDB.desde_entorno(env)             # lo primero: sin bitacora durable no hay servicio
    cfg_faro = ConfigFaro.desde_entorno(env, solo_pruebas_duenio_igual_servicio=solo_pruebas_mismo_uid)
    cfg_puerto = ConfigPuerto.desde_entorno(env)
    cfg_aviso = ConfigAviso.desde_entorno(env)              # 0.3b: sin aviso de las denegaciones no hay servicio
    cfg_control = ConfigControl.desde_entorno(env, solo_pruebas_mismo_uid=solo_pruebas_mismo_uid)    # 0.3c
    cfg_memoria = ConfigMemoria.desde_entorno(env)           # B9 desactivada por defecto; allowlist solo de test
    if os.geteuid() == 0:
        raise ConfigFaroInvalida("el servicio del Faro no corre como root: usa el usuario sin privilegios `faro`")
    validar_directorio_control(cfg_control)                 # un 0777 no deja arrancar (antes de tocar la base)
    credenciales = leer_credenciales(cfg_aviso.creds)       # falla cerrado: ausentes, incompletas o con escritura ajena
    paquete = await asyncio.to_thread(cargar_paquete, cfg_faro)
    pool = await crear_pool(cfg_db)
    pool_memoria = None
    adaptador_memoria = None
    if cfg_memoria.habilitada:
        try:
            pool_memoria, lector_memoria = await crear_pool_memoria_prueba(cfg_memoria)
            adaptador_memoria = AdaptadorMemoria(lector_memoria)
        except Exception:  # fail-soft: memoria es opcional; la herramienta queda cerrada y el Puerto sigue seguro
            # El servicio puede seguir atendiendo el resto del Puerto, pero la
            # herramienta falla cerrado con «memoria no disponible».
            logger.exception("memoria no disponible: no se abrió la base de prueba")
    avisador = Avisador(cfg_aviso, credenciales)
    try:
        emisor = EmisorTabla(pool)
        # El avisador es OBSERVADOR: ve cada registro antes que los emisores y aunque la tabla este caida.
        bitacora = Bitacora(emisores=[emisor, emisor_logger], observadores=[avisador])
        await bitacora.registrar("servicio_iniciado", sha_paquete=paquete.sha, pid=os.getpid())   # la sonda
        await avisador.iniciar()
    except BaseException:
        await avisador.cerrar()
        if pool_memoria is not None:
            pool_memoria.close()
            await pool_memoria.wait_closed()
        pool.close()
        await pool.wait_closed()
        raise
    return Servicio(cfg_faro, cfg_puerto, cfg_db, paquete, pool, emisor, bitacora,
                    PresupuestoBytes(cfg_puerto.presupuesto_bytes), solo_pruebas_mismo_uid,
                    cfg_aviso, cfg_control, avisador, cfg_memoria, pool_memoria, adaptador_memoria)


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
