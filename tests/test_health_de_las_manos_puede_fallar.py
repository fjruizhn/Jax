#!/usr/bin/env python3
"""El /health de LAS MANOS tiene que poder ponerse ROJO.

POR QUE EXISTE. Hasta el 2026-09-16 el endpoint devolvia `{"status": "alive"}`
FIJO: respondia «vivo» por el mero hecho de poder responder. Un control que no
puede fallar no valida nada —— y no es un endpoint cualquiera:

  - `loadtest/health.js` lo usa para las pruebas de carga de LAS CUATRO DEL
    RENDIMIENTO. Un p95 sobre un literal mide FastAPI, no el servicio.
  - La Mesa lo mira para saber si LAS MANOS esta vivo.
  - El dashboard lo muestra.

Con la base caida, los tres seguian en verde.

POR QUE ESTE ARCHIVO NO IMPORTA `server.py`. La primera version si, y en CI el
import fallaba: los cinco casos se SALTABAN en silencio, o sea que el test que
demuestra que /health puede ponerse rojo no corria justo donde importa. Un test
que nadie ejecuta afirma un estado que no existe y nadie lo contradice. La
logica se extrajo a `las_manos/salud.py`, que no arrastra el servidor entero.
"""
from __future__ import annotations

import asyncio
import contextlib
import sys
import unittest
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "las_manos"))

from salud import Salud, comprobar_audit, comprobar_base  # noqa: E402


def _conn_ok():
    cur = mock.AsyncMock()
    cur.execute = mock.AsyncMock()
    cur.fetchone = mock.AsyncMock(return_value=(1,))
    ctx = mock.MagicMock()
    ctx.__aenter__ = mock.AsyncMock(return_value=cur)
    ctx.__aexit__ = mock.AsyncMock(return_value=False)
    conn = mock.MagicMock()
    conn.cursor = mock.MagicMock(return_value=ctx)
    return conn


class _Abridor:
    """Doble de `jacobs.store.conexion`: context manager asincrono que entrega
    una conexion sana o explota al abrir. Cuenta cuantas veces se abrio."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.llamadas = 0

    def __call__(self):
        self.llamadas += 1

        @contextlib.asynccontextmanager
        async def _ctx():
            if self.error is not None:
                raise self.error
            yield _conn_ok()

        return _ctx()


class SaludTest(unittest.TestCase):
    def test_con_la_base_caida_NO_esta_sano(self):
        abrir = _Abridor(OSError("connection refused"))
        s = Salud({"base de datos": lambda: comprobar_base(abrir)})
        estado = asyncio.run(s.estado())
        self.assertFalse(estado["ok"], "el servicio se declaro sano con la base caida")
        self.assertTrue(any("base de datos" in p for p in estado["fallos"]))
        self.assertIn("connection refused", "".join(estado["fallos"]))

    def test_con_el_log_forense_no_escribible_NO_esta_sano(self):
        """Sin audit, ejecutar algo seria dejarlo sin rastro."""
        ruta = mock.MagicMock(spec=Path)
        ruta.parent.mkdir = mock.MagicMock()
        ruta.open = mock.MagicMock(side_effect=PermissionError("solo lectura"))
        s = Salud({"log de auditoria": lambda: comprobar_audit(ruta)})
        estado = asyncio.run(s.estado())
        self.assertFalse(estado["ok"])
        self.assertTrue(any("log de auditoria" in p for p in estado["fallos"]))

    def test_con_todo_sano_esta_sano(self):
        """Control del control: el caso bueno sigue siendo verde."""
        s = Salud({"base de datos": lambda: comprobar_base(_Abridor())})
        estado = asyncio.run(s.estado())
        self.assertTrue(estado["ok"])
        self.assertEqual(estado["fallos"], [])

    def test_una_dependencia_caida_no_tapa_a_las_otras(self):
        """Se reportan TODAS las que fallan, no la primera."""
        rota = _Abridor(OSError("no responde"))
        s = Salud({"uno": lambda: comprobar_base(rota), "dos": lambda: comprobar_base(rota)})
        estado = asyncio.run(s.estado())
        self.assertEqual(len(estado["fallos"]), 2)

    def test_la_cache_evita_una_consulta_por_peticion(self):
        """20.000 req/s en las pruebas de carga: sin cache, el remedio tumbaria
        la base que el chequeo quiere vigilar."""
        abrir = _Abridor()
        s = Salud({"base de datos": lambda: comprobar_base(abrir)}, ttl=5.0)

        async def veinticinco():
            for _ in range(25):
                await s.estado()
        asyncio.run(veinticinco())
        self.assertEqual(abrir.llamadas, 1,
                         f"se consultó la base {abrir.llamadas} veces en 25 peticiones")

    def test_vencido_el_ttl_se_vuelve_a_medir(self):
        """La cache no puede volverse ciega: si el TTL vence, se mide de nuevo."""
        reloj = {"t": 0.0}
        abrir = _Abridor()
        s = Salud({"base de datos": lambda: comprobar_base(abrir)},
                  ttl=5.0, reloj=lambda: reloj["t"])

        async def dos_ventanas():
            await s.estado()
            reloj["t"] = 6.0
            await s.estado()
        asyncio.run(dos_ventanas())
        self.assertEqual(abrir.llamadas, 2, "la cache no expiró nunca")

    def test_una_caida_posterior_se_nota_al_vencer_el_ttl(self):
        """El caso que de verdad importa: sano primero, caido despues."""
        reloj = {"t": 0.0}
        abrir = _Abridor()
        s = Salud({"base de datos": lambda: comprobar_base(abrir)},
                  ttl=5.0, reloj=lambda: reloj["t"])

        async def sano_y_luego_caido():
            primero = await s.estado()
            abrir.error = OSError("se cayo")
            reloj["t"] = 6.0
            return primero, await s.estado()
        primero, segundo = asyncio.run(sano_y_luego_caido())
        self.assertTrue(primero["ok"])
        self.assertFalse(segundo["ok"], "la caida posterior no se noto")


class EndpointTest(unittest.TestCase):
    """El endpoint es una capa fina sobre `Salud`. Lo unico suyo es el 503."""

    def test_el_endpoint_traduce_no_sano_a_503(self):
        fuente = (RAIZ / "las_manos" / "server.py").read_text(encoding="utf-8")
        self.assertIn('estado = await _salud.estado()', fuente)
        self.assertIn("response.status_code = 503", fuente)
        self.assertIn('"status": "alive" if estado["ok"] else "degraded"', fuente)

    def test_el_kill_switch_se_reporta_pero_NO_degrada(self):
        """Estar frenado a proposito es una decision, no una averia: no puede
        aparecer en la condicion del 503."""
        fuente = (RAIZ / "las_manos" / "server.py").read_text(encoding="utf-8")
        cuerpo = fuente[fuente.index('async def health('):fuente.index('@app.post("/human_gate/token")')]
        self.assertIn('"kill_switch_active": _kill_switch_active()', cuerpo)
        condicion = cuerpo[cuerpo.index('if not estado["ok"]'):cuerpo.index("return {")]
        self.assertNotIn("kill_switch", condicion)


if __name__ == "__main__":
    unittest.main()
