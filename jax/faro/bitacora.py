"""La bitacora del Puerto: una linea por llamada, con todo lo que Block 7 va a necesitar.

Spec §4 «Auditoria»: id de correlacion y `entry_point` propagados de punta a punta, SHA del
paquete, herramienta, hash de los argumentos y del resultado. Esta fase entrega el registro
estructurado y su salida saneada a log; el emisor hacia Block 7 / bitacora encadenada con
usuario de solo `INSERT` es un emisor mas (`emisores=`), paso posterior del plan.

SANEADO: `_campo_log` es el mismo patron de `cli_sandbox._campo_log` (MINOR-22 de la
suscripcion fase 1): todo campo de texto de la linea de log pasa por el, porque un valor con
saltos de linea puede fabricar una SEGUNDA linea de log con un `decision=permitido` falso.
Esa funcion vive hoy solo en la rama `feat/suscripcion-fase1-minors` (no esta en `master`);
esta es su copia minima, con la misma regex y la misma semantica, para unificarla cuando esa
rama se integre. Ademas, un valor con espacios, `=` o comillas sale entre comillas JSON, de modo que
no pueda fingir un campo `clave=valor` dentro de su propia linea.

FALLA CERRADO: si un emisor lanza, la excepcion SUBE. La llamada se responde con error y no
se entrega el resultado: no hay accion sin bitacora.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)

_RE_BLANCOS = re.compile(r"\s+")
_RE_NECESITA_COMILLAS = re.compile(r'[\s="]')


def _campo_log(valor, tope: int = 200) -> str:
    """Un valor para la linea de log: texto de UNA sola linea y de largo acotado. Los espacios y
    saltos (\\n, \\r, tabs, separadores unicode) se colapsan en uno solo."""
    return _RE_BLANCOS.sub(" ", str(valor))[:tope]


def _valor_de_linea(valor) -> str:
    limpio = _campo_log(valor)
    if limpio == "" or _RE_NECESITA_COMILLAS.search(limpio):
        return json.dumps(limpio, ensure_ascii=False)
    return limpio


def emisor_logger(registro: dict) -> None:
    campos = " ".join(f"{k}={_valor_de_linea(v)}" for k, v in registro.items() if k != "evento")
    logger.info("faro_%s %s", _campo_log(registro.get("evento", "?")), campos)


class Bitacora:
    def __init__(self, emisores: Iterable[Callable[[dict], None]] | None = None):
        self._emisores = (emisor_logger,) if emisores is None else tuple(emisores)

    def registrar(self, evento: str, **campos) -> dict:
        registro = {"evento": evento, "momento": round(time.time(), 6), **campos}
        for emisor in self._emisores:
            emisor(registro)
        return registro
