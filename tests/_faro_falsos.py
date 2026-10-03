"""Un MariaDB de mentira para las pruebas de la bitacora y del servicio que no necesitan una base real:
`FalsoPool` imita lo que `EmisorTabla` usa de aiomysql (`acquire()` -> `cursor()` -> `execute`)."""
from __future__ import annotations

import asyncio

COLUMNAS = ("cadena_id", "seq", "momento", "evento", "run_id", "id_correlacion", "decision", "registro", "hash_previo", "hash")


class _Cursor:
    def __init__(self, pool): self.pool = pool
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def execute(self, sql, params=None):
        if self.pool.modo == "colgar":
            await asyncio.sleep(3600)
        if self.pool.modo == "fallar":
            raise OSError("MariaDB caida (falso)")
        self.pool.filas.append(dict(zip(COLUMNAS, params)))


class _Con:
    def __init__(self, pool): self.pool = pool
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    def cursor(self): return _Cursor(self.pool)


class _Adquirir:
    def __init__(self, pool): self.pool = pool

    async def __aenter__(self):
        if self.pool.modo == "colgar_acquire":
            await asyncio.sleep(3600)
        return _Con(self.pool)

    async def __aexit__(self, *a): return False


class FalsoPool:
    def __init__(self, modo="ok"):
        self.modo, self.filas, self.cerrado = modo, [], False
    def acquire(self): return _Adquirir(self)
    def close(self): self.cerrado = True
    async def wait_closed(self): return None


# --------------------------------------------------------------------------- #
# Un Telegram de mentira: HTTP REAL en 127.0.0.1, nunca el de verdad.         #
# --------------------------------------------------------------------------- #
import json as _json
import threading as _threading
import time as _time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs


class FalsoTelegram:
    """Servidor HTTP local que imita `POST /bot<token>/sendMessage`. Guarda lo recibido en `.recibidos`
    (cada uno: `{"ruta", "chat_id", "text"}`). `estado` y `retraso_s` se pueden cambiar durante la prueba."""

    def __init__(self, estado: int = 200, retraso_s: float = 0.0):
        self.estado, self.retraso_s = estado, retraso_s
        self.recibidos: list[dict] = []
        self._cerrojo = _threading.Lock()
        falso = self

        class _Manejador(BaseHTTPRequestHandler):
            def do_POST(self):          # noqa: N802 (API de http.server)
                largo = int(self.headers.get("Content-Length", "0"))
                cuerpo = parse_qs(self.rfile.read(largo).decode())
                if falso.retraso_s:
                    _time.sleep(falso.retraso_s)
                with falso._cerrojo:
                    falso.recibidos.append({"ruta": self.path, "chat_id": cuerpo.get("chat_id", [""])[0],
                                            "text": cuerpo.get("text", [""])[0]})
                respuesta = _json.dumps({"ok": falso.estado == 200}).encode()
                self.send_response(falso.estado)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(respuesta)))
                self.end_headers()
                self.wfile.write(respuesta)

            def log_message(self, *a):  # silencio
                pass

        self._servidor = ThreadingHTTPServer(("127.0.0.1", 0), _Manejador)
        self._servidor.daemon_threads = True
        self._hilo = _threading.Thread(target=self._servidor.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._servidor.server_address[1]}"

    def __enter__(self):
        self._hilo.start()
        return self

    def __exit__(self, *a):
        self._servidor.shutdown()
        self._servidor.server_close()

    def textos(self) -> list[str]:
        with self._cerrojo:
            return [r["text"] for r in self.recibidos]

    def esperar(self, n: int, plazo_s: float = 5.0) -> bool:
        """Bloqueante: para usar desde `asyncio.to_thread` o fuera del bucle."""
        limite = _time.monotonic() + plazo_s
        while _time.monotonic() < limite:
            if len(self.textos()) >= n:
                return True
            _time.sleep(0.01)
        return len(self.textos()) >= n


# --------------------------------------------------------------------------- #
# Un almacen de topes en memoria (para las pruebas de la POLITICA de topes;   #
# la atomicidad del conteo se prueba contra MariaDB, no contra esto).         #
# --------------------------------------------------------------------------- #
class AlmacenMemoria:
    """Imita `AlmacenMariaDB.sumar(clave, periodo, cantidad, tope) -> (aplicado, usado)`."""

    def __init__(self, modo="ok"):
        self.modo, self.usado, self.llamadas = modo, {}, []

    async def sumar(self, clave, periodo, cantidad, tope):
        self.llamadas.append((clave, periodo, cantidad, tope))
        if self.modo == "colgar":
            await asyncio.sleep(3600)
        if self.modo == "fallar":
            raise OSError("almacen de topes caido (falso)")
        actual = self.usado.get((clave, periodo), 0)
        if tope is not None and actual + cantidad > tope:
            return False, actual
        self.usado[(clave, periodo)] = actual + cantidad
        return True, actual + cantidad
