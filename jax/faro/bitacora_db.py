"""La bitacora DURABLE del Faro: una tabla encadenada (`faro_bitacora`) escrita por un usuario de base que
SOLO tiene INSERT (auditoria MAJOR-5). Migraciones: `ops/faro/migrations/`.

Cada fila lleva `hash = sha256(hash_previo | cadena_id | seq | registro)`. El proceso lleva su cadena: como el
usuario de la aplicacion no puede LEER la tabla, no conoce el ultimo hash de la cadena anterior, asi que cada
arranque abre una cadena nueva (`cadena_id` aleatorio, `seq` desde 0, primer `hash_previo` en cero) cuya primera
fila es `inicio_cadena`. `verificar_cadena` (lo corre un auditor con SELECT) detecta una fila cambiada, una
borrada (hueco), una reordenada o una cadena sin inicio. Un fallo al insertar es ambiguo (puede haberse
confirmado) y SUBE: el Puerto no entrega resultados sin bitacora; el emisor se recupera abriendo una cadena nueva.

ANCLA EXTERNA. Quien puede borrar la COLA de una cadena la deja valida por si sola. Por eso el emisor expone su
ultima posicion (`ancla()`: cadena_id, seq, hash) y `publicar_anclas_periodicamente` la entrega a un publicador
(un archivo fuera de la base, un aviso, otro host: lo elige quien opera) cada `intervalo_s` mientras cambie;
`verificar_cadena(filas, anclas=...)` detecta `cola_truncada`, `ancla_no_coincide` y `cadena_ausente`.

PLAZO. Cada insercion (y la espera del Lock) tiene un plazo (`plazo_s`): una base colgada no deja a todos detras
del Lock; vence con `TimeoutError`, que SUBE (fallo cerrado), y la cadena sigue en otra nueva.

Es un emisor `async` de `jax.faro.bitacora.Bitacora` (`Bitacora(emisores=[EmisorTabla(pool)])`).
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field

import aiomysql

from .config import ConfigFaroInvalida

logger = logging.getLogger(__name__)
GENESIS = "0" * 64
_RE_IDENT = re.compile(r"^[A-Za-z0-9_]{1,64}$")
TABLA = "faro_bitacora"


@dataclass(frozen=True)
class ConfigBitacoraDB:
    host: str
    port: int
    usuario: str
    clave: str = field(repr=False)     # nunca se imprime
    base: str = ""
    connect_timeout_s: float = 5.0     # plazo de conexion: una base que no contesta no cuelga el arranque

    def __post_init__(self) -> None:
        if not _RE_IDENT.fullmatch(self.base):
            raise ConfigFaroInvalida("JAX_FARO_BITACORA_DB_NAME no es un nombre de base valido")
        if not (0 < self.connect_timeout_s <= 60):
            raise ConfigFaroInvalida("JAX_FARO_BITACORA_DB_CONNECT_TIMEOUT_S fuera de rango (0, 60]")

    @classmethod
    def desde_entorno(cls, env: Mapping[str, str]) -> "ConfigBitacoraDB":
        def pedir(nombre: str) -> str:
            valor = (env.get(nombre) or "").strip()
            if not valor:
                raise ConfigFaroInvalida(f"{nombre} no esta definida: sin la bitacora durable el Puerto no arranca")
            return valor
        try:
            puerto = int(pedir("JAX_FARO_BITACORA_DB_PORT"))
        except ValueError as exc:
            raise ConfigFaroInvalida("JAX_FARO_BITACORA_DB_PORT no es un entero") from exc
        try:
            plazo = float((env.get("JAX_FARO_BITACORA_DB_CONNECT_TIMEOUT_S") or "5").strip())
        except ValueError as exc:
            raise ConfigFaroInvalida("JAX_FARO_BITACORA_DB_CONNECT_TIMEOUT_S no es un numero") from exc
        return cls(host=pedir("JAX_FARO_BITACORA_DB_HOST"), port=puerto, usuario=pedir("JAX_FARO_BITACORA_DB_USER"),
                   clave=pedir("JAX_FARO_BITACORA_DB_PASSWORD"), base=pedir("JAX_FARO_BITACORA_DB_NAME"),
                   connect_timeout_s=plazo)


async def crear_pool(cfg: ConfigBitacoraDB) -> aiomysql.Pool:
    return await aiomysql.create_pool(host=cfg.host, port=cfg.port, user=cfg.usuario, password=cfg.clave, db=cfg.base,
                                      minsize=1, maxsize=2, autocommit=True, charset="utf8mb4",
                                      connect_timeout=cfg.connect_timeout_s)


def _canonico(registro: Mapping) -> str:
    return json.dumps(registro, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def calcular_hash(hash_previo: str, cadena_id: str, seq: int, registro: str) -> str:
    return hashlib.sha256(f"{hash_previo}|{cadena_id}|{seq}|{registro}".encode()).hexdigest()


class EmisorTabla:
    """Emisor async de la bitacora. Serializa (un `Lock`): el orden de la cadena es el orden de los INSERT."""

    def __init__(self, pool: aiomysql.Pool, *, plazo_s: float = 5.0):
        self._pool = pool
        self._plazo_s = plazo_s
        self._lock = asyncio.Lock()
        self._cadena_id: str | None = None
        self._seq = 0
        self._previo = GENESIS

    def _abrir_cadena(self) -> None:
        self._cadena_id, self._seq, self._previo = secrets.token_hex(16), 0, GENESIS

    async def _insertar(self, registro: Mapping) -> None:
        cuerpo = _canonico(registro)
        h = calcular_hash(self._previo, self._cadena_id, self._seq, cuerpo)
        async with self._pool.acquire() as con, con.cursor() as cur:
            await cur.execute(
                f"INSERT INTO {TABLA} (cadena_id, seq, momento, evento, run_id, id_correlacion, decision, registro, hash_previo, hash) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (self._cadena_id, self._seq, float(registro.get("momento", 0.0)), str(registro.get("evento", ""))[:32],
                 _corto(registro.get("run_id"), 64), _corto(registro.get("id_correlacion"), 128),
                 _corto(registro.get("decision"), 16), cuerpo, self._previo, h))
        self._previo, self._seq = h, self._seq + 1

    def ancla(self) -> dict | None:
        """La ultima posicion escrita de la cadena actual (o None si no hay): lo que se publica fuera."""
        if self._cadena_id is None or self._seq == 0:
            return None
        return {"cadena_id": self._cadena_id, "seq": self._seq - 1, "hash": self._previo}

    async def __call__(self, registro: Mapping) -> None:
        try:
            await asyncio.wait_for(self._emitir(registro), self._plazo_s)
        except TimeoutError:
            self._cadena_id = None          # la insercion cancelada pudo haberse confirmado: cadena nueva
            raise

    async def _emitir(self, registro: Mapping) -> None:
        async with self._lock:
            try:
                if self._cadena_id is None:
                    self._abrir_cadena()
                    await self._insertar({"evento": "inicio_cadena", "momento": registro.get("momento", 0.0),
                                          "cadena_id": self._cadena_id, "pid": os.getpid()})
                await self._insertar(registro)
            except BaseException:
                # La insercion pudo haberse confirmado aunque fallara: la cadena sigue en otra nueva.
                self._cadena_id = None
                raise


async def publicar_anclas_periodicamente(emisor: EmisorTabla, publicar, *, intervalo_s: float = 60.0) -> None:
    """Cada `intervalo_s` entrega al `publicar` (sincrono o async) el ancla del emisor SI cambio desde la ultima
    publicada. Un publicador que falla no mata el bucle: se reintenta en el intervalo siguiente."""
    ultima = None
    while True:
        await asyncio.sleep(intervalo_s)
        ancla = emisor.ancla()
        if ancla is None or ancla == ultima:
            continue
        try:
            r = publicar(ancla)
            if inspect.isawaitable(r):
                await r
            ultima = ancla
        except Exception:  # fail-soft: el destino de la ancla esta caido; se reintenta en el intervalo siguiente y no se cuenta como publicada
            logger.exception("no se pudo publicar el ancla de la bitacora")


def _corto(valor, tope: int):
    return None if valor is None else str(valor)[:tope]


@dataclass(frozen=True)
class Problema:
    codigo: str
    cadena_id: str
    seq: int | None = None


def verificar_cadena(filas: list[Mapping], anclas: list[Mapping] | tuple = ()) -> list[Problema]:
    """Lista de problemas (vacia = integra). `filas` son las filas de la tabla (con `cadena_id`, `seq`,
    `registro`, `hash_previo`, `hash`), en cualquier orden. `anclas` (publicadas fuera de la base) permiten
    notar que se trunco la cola: `cola_truncada`, `ancla_no_coincide`, `cadena_ausente`."""
    problemas: list[Problema] = []
    por_cadena: dict[str, list[Mapping]] = {}
    for f in filas:
        por_cadena.setdefault(f["cadena_id"], []).append(f)
    for cadena, rows in por_cadena.items():
        rows = sorted(rows, key=lambda f: f["seq"])
        if rows[0]["seq"] != 0:
            problemas.append(Problema("cadena_sin_inicio", cadena, rows[0]["seq"]))
        previo_esperado = GENESIS
        for i, f in enumerate(rows):
            if i > 0 and f["seq"] != rows[i - 1]["seq"] + 1:
                problemas.append(Problema("hueco", cadena, f["seq"]))
            if f["seq"] == 0 and f["hash_previo"] != GENESIS:
                problemas.append(Problema("inicio_no_es_genesis", cadena, 0))
            if i > 0 and f["hash_previo"] != rows[i - 1]["hash"]:
                problemas.append(Problema("encadenado_roto", cadena, f["seq"]))
            if f["hash"] != calcular_hash(f["hash_previo"], cadena, f["seq"], f["registro"]):
                problemas.append(Problema("hash_no_coincide", cadena, f["seq"]))
        del previo_esperado
    for a in anclas:
        rows = {f["seq"]: f for f in por_cadena.get(a["cadena_id"], [])}
        if not rows:
            problemas.append(Problema("cadena_ausente", a["cadena_id"], a["seq"]))
        elif a["seq"] not in rows:
            problemas.append(Problema("cola_truncada", a["cadena_id"], a["seq"]))
        elif rows[a["seq"]]["hash"] != a["hash"]:
            problemas.append(Problema("ancla_no_coincide", a["cadena_id"], a["seq"]))
    return problemas
