# jax/ejecutor/contratos/registro.py
"""Registro intocable del Ejecutor (C3). Lo escribe el PROXY, como `fruiz`, fuera de la jaula.

Spec 2026-09-15 §4 C3; decisión D-SP1-2 del índice de SP1: no lo alimenta la jaula
(podría no llamarlo, o llamarlo con lo que quisiera); lo alimenta lo que el cerebro
pide y la jaula devuelve, visto al pasar por el proxy.

Intocable por tres capas:
1. otra cuenta: directorio 0750 de `fruiz`; `axioma` ni lo lee;
2. `chattr +a` (root, al instalar): ni `fruiz` lo trunca ni lo reescribe;
3. cadena: cada línea lleva `n` (1, 2, 3…) y `prev`, el sha256 de los bytes de la
   línea anterior (GENESIS para la primera). `verificar_cadena` da la primera línea
   donde se rompe: una línea editada se ve en la SIGUIENTE (`prev_no_cuadra`); una
   borrada o insertada, donde el número salta (`n_no_cuadra`).

`anotar` es BLOQUEANTE (write + fsync): desde async, con `asyncio.to_thread`. Un lock
lo serializa: la cadena exige un escritor ordenado. Una escritura parcial deja el
registro marcado como roto y toda escritura posterior falla: un registro con un hueco
no sigue aceptando como si nada.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64
_LEER_COLA = 1 << 20


class RegistroCorrupto(RuntimeError):
    def __init__(self, codigo: str):
        super().__init__(codigo)
        self.codigo = codigo


@dataclass(frozen=True)
class Verificacion:
    ok: bool
    lineas: int
    primer_error: int | None
    codigo: str | None


def _hash(linea: bytes) -> str:
    return hashlib.sha256(linea).hexdigest()


def _ultima_linea(fd: int) -> bytes | None:
    tamanio = os.fstat(fd).st_size
    if tamanio == 0:
        return None
    desde = max(0, tamanio - _LEER_COLA)
    datos = os.pread(fd, tamanio - desde, desde)
    if not datos.endswith(b"\n"):
        raise RegistroCorrupto("ultima_linea_incompleta")
    cuerpo = datos[:-1]
    corte = cuerpo.rfind(b"\n")
    if corte < 0 and desde > 0:
        raise RegistroCorrupto("linea_demasiado_larga")
    return cuerpo[corte + 1:]


class Registro:
    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self._lock = threading.Lock()
        self._roto = False
        self._fd = os.open(self.ruta, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o640)
        try:
            lector = os.open(self.ruta, os.O_RDONLY | os.O_CLOEXEC)
            try:
                ultima = _ultima_linea(lector)
            finally:
                os.close(lector)
            if ultima is None:
                self._n, self._prev = 0, GENESIS
            else:
                try:
                    doc = json.loads(ultima)
                except ValueError:
                    raise RegistroCorrupto("ultima_linea_ilegible") from None
                if not isinstance(doc, dict) or isinstance(doc.get("n"), bool) or not isinstance(doc.get("n"), int):
                    raise RegistroCorrupto("ultima_linea_ilegible")
                self._n, self._prev = doc["n"], _hash(ultima)
        except BaseException:
            os.close(self._fd)
            raise

    def anotar(self, evento: dict) -> int:
        with self._lock:
            if self._roto:
                raise OSError("registro_roto")
            n = self._n + 1
            doc = {**evento, "n": n, "prev": self._prev, "momento": datetime.now(timezone.utc).isoformat()}
            linea = json.dumps(doc, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
            datos = linea + b"\n"
            if os.write(self._fd, datos) != len(datos):
                self._roto = True
                raise OSError("escritura_parcial")
            os.fsync(self._fd)
            self._n, self._prev = n, _hash(linea)
            return n

    def cerrar(self) -> None:
        os.close(self._fd)


def verificar_cadena(ruta) -> Verificacion:
    prev, n = GENESIS, 0
    with open(ruta, "rb") as f:
        for n, cruda in enumerate(f, start=1):
            if not cruda.endswith(b"\n"):
                return Verificacion(False, n, n, "ultima_linea_incompleta")
            linea = cruda[:-1]
            try:
                doc = json.loads(linea)
            except ValueError:
                return Verificacion(False, n, n, "linea_ilegible")
            if not isinstance(doc, dict) or doc.get("n") != n:
                return Verificacion(False, n, n, "n_no_cuadra")
            if doc.get("prev") != prev:
                return Verificacion(False, n, n, "prev_no_cuadra")
            prev = _hash(linea)
    return Verificacion(True, n, None, None)
