"""Los topes del Faro (plan 0.3b, P-4): conteo ATOMICO y la politica de que hacer al llegar.

REGLAS (spec §6 y decisiones de Fernando)
- **Sin regla de tope = sin tope de cantidad, pero con medicion y aviso: NO se niega.** El consumo se cuenta
  igual y la primera vez que un (tenant, recurso, periodo) consume sin regla se anota `tope_sin_regla` (que el
  `Avisador` manda a Telegram, con su tasa).
- **Con tope** (una regla vigente de `policy/faro/*.yaml`, evaluada por El Faro, nunca por quien consume): al
  llegar **falla cerrado**. Este modulo solo recibe el tope YA RESUELTO (`tope: int | None`); no lee politica ni
  ratifica nada.
- **D-4: no hay tope de agentes** (ni de concurrencia ni de profundidad) **ni de conexiones**. Pedir un tope
  para un recurso con esos prefijos es un error (`TopeProhibido`); medirlos sigue permitido.
- Si el almacen no se puede consultar: con tope se niega (no se sabe si cabe: `tope_no_verificable`); sin tope
  se deja pasar (no hay regla que niegue) y el resultado dice que no se pudo medir.
- Una anotacion de bitacora que falla NO cambia la decision (como la guardia: lo que no falla abierto es el
  acceso, no la anotacion).

ATOMICIDAD. `AlmacenMariaDB.sumar` hace un solo `UPDATE ... SET usado = LAST_INSERT_ID(usado + x) WHERE ...
AND usado + x <= tope`: el motor serializa los UPDATE de una fila, asi que N peticiones contra un tope de N-1
dejan pasar exactamente N-1. Nunca leer, comparar y escribir por separado. Usuario de base propio (migracion
004): lee, inserta y actualiza `faro_topes`; no toca la bitacora.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Protocol

from .bitacora import Bitacora
from .config import ConfigFaroInvalida
from .identidad import RE_TENANT

logger = logging.getLogger(__name__)

TABLA = "faro_topes"
MAX_CANTIDAD = 2 ** 53                      # lo que un entero de JSON/float representa sin perdida
# D-4, SEMANTICO: ningun recurso cuya palabra sea de agentes ni de conexiones lleva tope, se llame como se llame
# (`agentes`, `subagentes`, `agentes_concurrentes`, `agente.hijos`, `enjambre.agentes`, `swarm.agents`,
# `tokens.por_agente`...). Se mira cada palabra del nombre (separadas por `.`, `_` o `-`).
PALABRAS_SIN_TOPE = ("agent", "subagent", "enjambre", "swarm", "conexion")
_RE_PALABRAS = re.compile(r"[._-]+")


def es_recurso_sin_tope(recurso: str) -> bool:
    return any(p.startswith(PALABRAS_SIN_TOPE) for p in _RE_PALABRAS.split(recurso))

_RE_TENANT = RE_TENANT      # el mismo que valida el canal de control: un tenant aceptado alli nunca rompe aqui
_RE_RECURSO = re.compile(r"^[a-z0-9_.-]{1,48}$")
_RE_PERIODO = re.compile(r"^[A-Za-z0-9_.:-]{1,32}$")
_RESERVADOS = frozenset({"evento", "momento", "decision", "motivo", "tenant", "recurso", "cantidad", "usado", "tope",
                         "periodo"})
_MAX_CONTEXTO = 16
_MAX_VISTOS = 4096


class TopeProhibido(ConfigFaroInvalida):
    """D-4: no se pone tope a los agentes ni a las conexiones."""


class ResultadoDesconocido(Exception):
    """El UPDATE del conteo se ENVIO y no se sabe si se confirmo (plazo vencido o conexion cortada despues de
    mandarlo). No es lo mismo que «no se pudo consultar»: la cuenta pudo cambiar."""


@dataclass(frozen=True)
class ResultadoTope:
    permitido: bool
    usado: int | None       # lo contado tras la llamada (None si no se pudo consultar)
    tope: int | None
    medido: bool            # ¿se pudo contar?
    motivo: str             # "", "tope_alcanzado", "almacen_no_disponible" o "resultado_desconocido"


class AlmacenTopes(Protocol):
    async def sumar(self, clave: str, periodo: str, cantidad: int, tope: int | None) -> tuple[bool, int]:
        """(aplicado, usado). Con `tope`, aplica SOLO si `usado + cantidad <= tope`, en una operacion atomica."""


class AlmacenMariaDB:
    """El almacen real. `pool` es un `aiomysql.Pool` en autocommit (`crear_pool`) del usuario de los topes."""

    def __init__(self, pool, *, plazo_s: float = 4.0):
        """`plazo_s` tiene que ser MENOR que el de `Topes` (5 s): asi el plazo del almacen vence primero y sabe si el
        UPDATE ya se habia enviado (`ResultadoDesconocido`)."""
        self._pool = pool
        self._plazo_s = plazo_s

    async def leer(self, clave: str, periodo: str) -> int:
        async with self._pool.acquire() as con, con.cursor() as cur:
            await cur.execute(f"SELECT usado FROM {TABLA} WHERE clave = %s AND periodo = %s", (clave, periodo))
            fila = await cur.fetchone()
            return int(fila[0]) if fila else 0

    async def _sumar(self, clave: str, periodo: str, cantidad: int, tope: int | None, estado: dict) -> tuple[bool, int]:
        async with self._pool.acquire() as con, con.cursor() as cur:
            # La fila existe antes de contar (no-op si ya estaba): el UPDATE de abajo es el que cuenta.
            await cur.execute(f"INSERT INTO {TABLA} (clave, periodo, usado) VALUES (%s, %s, 0) "
                              "ON DUPLICATE KEY UPDATE usado = usado", (clave, periodo))
            estado["update_enviado"] = True          # desde aqui, un fallo es de resultado DESCONOCIDO
            if tope is None:
                await cur.execute(f"UPDATE {TABLA} SET usado = LAST_INSERT_ID(usado + %s) WHERE clave = %s AND periodo = %s",
                                  (cantidad, clave, periodo))
                aplicado = True
            else:
                await cur.execute(f"UPDATE {TABLA} SET usado = LAST_INSERT_ID(usado + %s) "
                                  "WHERE clave = %s AND periodo = %s AND usado + %s <= %s",
                                  (cantidad, clave, periodo, cantidad, tope))
                aplicado = cur.rowcount == 1
            if aplicado:
                await cur.execute("SELECT LAST_INSERT_ID()")
            else:
                await cur.execute(f"SELECT usado FROM {TABLA} WHERE clave = %s AND periodo = %s", (clave, periodo))
            fila = await cur.fetchone()
            return aplicado, int(fila[0])

    async def sumar(self, clave: str, periodo: str, cantidad: int, tope: int | None) -> tuple[bool, int]:
        estado = {"update_enviado": False}
        try:
            return await asyncio.wait_for(self._sumar(clave, periodo, cantidad, tope, estado), self._plazo_s)
        except Exception as exc:  # el motivo exacto no importa: importa si el UPDATE ya se habia enviado
            if estado["update_enviado"]:
                raise ResultadoDesconocido(f"el UPDATE se envio y fallo despues ({type(exc).__name__})") from None
            raise


def _entero(valor: object, nombre: str, minimo: int, maximo: int) -> int:
    if isinstance(valor, bool) or not isinstance(valor, int) or not minimo <= valor <= maximo:
        raise ValueError(f"{nombre} tiene que ser un entero entre {minimo} y {maximo}")
    return valor


class Topes:
    def __init__(self, almacen: AlmacenTopes, bitacora: Bitacora, *, plazo_s: float = 5.0):
        self._almacen = almacen
        self._bitacora = bitacora
        self._plazo_s = plazo_s
        self._vistos_sin_regla: set[tuple[str, str, str]] = set()
        self.inciertos: dict[tuple[str, str], int] = {}       # (clave, periodo) -> cantidad de conteos de resultado desconocido

    async def _anotar(self, evento: str, **campos) -> None:
        try:
            await self._bitacora.registrar(evento, **campos)
        except Exception:  # fail-closed: la decision ya se tomo y no cambia; solo la anotacion fallo y queda en el log
            logger.exception("no se pudo registrar %s", evento)

    async def consumir(self, *, tenant: str, recurso: str, cantidad: int = 1, tope: int | None = None,
                       periodo: str = "total", **contexto) -> ResultadoTope:
        if not isinstance(tenant, str) or not _RE_TENANT.fullmatch(tenant):
            raise ValueError("tenant invalido")
        if not isinstance(recurso, str) or not _RE_RECURSO.fullmatch(recurso):
            raise ValueError("recurso invalido")
        if not isinstance(periodo, str) or not _RE_PERIODO.fullmatch(periodo):
            raise ValueError("periodo invalido")
        cantidad = _entero(cantidad, "cantidad", 1, MAX_CANTIDAD)
        if tope is not None:
            tope = _entero(tope, "tope", 0, MAX_CANTIDAD)
            if es_recurso_sin_tope(recurso):
                raise TopeProhibido(f"D-4: no hay tope de agentes ni de conexiones ({recurso!r}); solo se mide")
        extra = {k: (v if isinstance(v, (int, float, bool)) or v is None else str(v)[:200])
                 for k, v in list(contexto.items())[:_MAX_CONTEXTO] if k not in _RESERVADOS}
        base = {"tenant": tenant, "recurso": recurso, "cantidad": cantidad, "periodo": periodo, "tope": tope}
        clave = f"{tenant}|{recurso}"
        try:
            aplicado, usado = await asyncio.wait_for(self._almacen.sumar(clave, periodo, cantidad, tope), self._plazo_s)
        except ResultadoDesconocido:
            # Se envio el UPDATE y no se sabe si cuenta. Con tope se NIEGA (no se sabe si cabia); sin tope no hay regla
            # que niegue. La cantidad queda como incierta hasta `reconciliar`.
            self.inciertos[(clave, periodo)] = self.inciertos.get((clave, periodo), 0) + cantidad
            await self._anotar("tope_resultado_desconocido", **extra, **base,
                               decision="permitido" if tope is None else "denegado", motivo="resultado_desconocido")
            return ResultadoTope(tope is None, None, tope, False, "resultado_desconocido")
        except Exception as exc:  # fail-soft: con tope NIEGA (fail-closed); sin tope no hay regla que niegue (D-4), se registra y avisa
            logger.warning("almacen de topes no disponible (%s)", type(exc).__name__)
            if tope is None:
                return ResultadoTope(True, None, None, False, "almacen_no_disponible")
            await self._anotar("tope_no_verificable", **extra, **base, decision="denegado", motivo="almacen_no_disponible")
            return ResultadoTope(False, None, tope, False, "almacen_no_disponible")
        if not aplicado:
            await self._anotar("tope_superado", **extra, **base, usado=usado, decision="denegado", motivo="tope_alcanzado")
            return ResultadoTope(False, usado, tope, True, "tope_alcanzado")
        if tope is None:
            visto = (tenant, recurso, periodo)
            if visto not in self._vistos_sin_regla:
                if len(self._vistos_sin_regla) >= _MAX_VISTOS:
                    self._vistos_sin_regla.clear()
                self._vistos_sin_regla.add(visto)
                await self._anotar("tope_sin_regla", **extra, **base, usado=usado, decision="permitido",
                                   motivo="sin_regla_de_tope")
        return ResultadoTope(True, usado, tope, True, "")

    async def reconciliar(self, *, tenant: str, recurso: str, periodo: str = "total") -> dict:
        """Lee el contador REAL y cierra lo incierto de (tenant, recurso, periodo): anota `tope_reconciliado` con lo leido
        y con la cantidad que estaba en duda, para que quien audita compare. No corrige el contador: un conteo
        confirmado cuya peticion se nego sobra en la cuenta (el lado seguro) y descontarlo es una decision de
        operacion (0.9), no de este modulo. Si la lectura falla, lo incierto se conserva y el error sube."""
        if not isinstance(tenant, str) or not _RE_TENANT.fullmatch(tenant) or not _RE_RECURSO.fullmatch(recurso) or not _RE_PERIODO.fullmatch(periodo):
            raise ValueError("tenant, recurso o periodo invalidos")
        clave = f"{tenant}|{recurso}"
        incierto = self.inciertos.pop((clave, periodo), 0)
        try:
            usado = await asyncio.wait_for(self._almacen.leer(clave, periodo), self._plazo_s)
        except BaseException:
            if incierto:
                self.inciertos[(clave, periodo)] = self.inciertos.get((clave, periodo), 0) + incierto
            raise
        await self._anotar("tope_reconciliado", tenant=tenant, recurso=recurso, periodo=periodo, usado=usado, incierto=incierto,
                           motivo="reconciliacion")
        return {"usado": usado, "incierto": incierto}
