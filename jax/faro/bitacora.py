"""La bitacora del Puerto: una linea por llamada, con todo lo que Block 7 va a necesitar.

Spec §4 «Auditoria»: id de correlacion y `entry_point` propagados de punta a punta, SHA del
paquete, herramienta, hash de los argumentos y del resultado. Esta fase entrega el registro
estructurado y su salida saneada a log; el emisor hacia Block 7 / bitacora encadenada con
usuario de solo `INSERT` es un emisor mas (`emisores=`), paso posterior del plan.

SANEADO: todo campo de texto de la linea de log pasa por `cli_sandbox._campo_log` (MINOR-22 y
MINOR-32 de la suscripcion fase 1, ya en `master`): colapsa saltos de linea y escapa `=`, el
espacio, la barra invertida y los caracteres de control, de modo que un valor no puede fabricar una
segunda linea ni un `clave=valor` dentro de la suya. No hacen falta comillas propias.

FALLA CERRADO: si un emisor lanza, la excepcion SUBE. La llamada se responde con error y no
se entrega el resultado: no hay accion sin bitacora. (Una DENEGACION sigue denegando aunque no
se pueda registrar: lo que no falla abierto es el acceso, no la anotacion.)
"""
from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable, Iterable

from cli_sandbox import _campo_log

logger = logging.getLogger(__name__)

def emisor_logger(registro: dict) -> None:
    campos = " ".join(f"{k}={_campo_log(v)}" for k, v in registro.items() if k != "evento")
    logger.info("faro_%s %s", _campo_log(registro.get("evento", "?")), campos)


class Bitacora:
    def __init__(self, emisores: Iterable[Callable[[dict], None]] | None = None):
        self._emisores = (emisor_logger,) if emisores is None else tuple(emisores)

    @property
    def emisores(self) -> tuple:
        return self._emisores

    async def registrar(self, evento: str, **campos) -> dict:
        """Entrega el registro a TODOS los emisores, en orden; un emisor puede ser sincrono o
        `async` (el de la tabla encadenada lo es). Si uno lanza, la excepcion sube."""
        registro = {"evento": evento, "momento": round(time.time(), 6), **campos}
        for emisor in self._emisores:
            resultado = emisor(registro)
            if inspect.isawaitable(resultado):
                await resultado
        return registro
