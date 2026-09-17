#!/usr/bin/env python3
"""Tests del arnes de prueba de carga (`scripts/load_test.py`).

Se testea la ARITMETICA, que es donde una prueba de carga miente sin avisar: un
p95 mal calculado da un numero tranquilizador y la app se cae igual. La parte
de red no se testea aca -- eso lo prueba usarla.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from load_test import campo_vacio, percentil, resumen  # noqa: E402


class PercentilTest(unittest.TestCase):
    def test_p95_de_100_valores_es_el_valor_95(self):
        # 1..100: el p95 tiene que ser 95, no 94 ni 96. El off-by-one de
        # percentiles es EL error clasico de un arnes casero.
        self.assertEqual(percentil(list(range(1, 101)), 95), 95)

    def test_p50_es_la_mediana(self):
        self.assertEqual(percentil([10, 20, 30], 50), 20)

    def test_una_sola_muestra_no_revienta(self):
        self.assertEqual(percentil([7.5], 99), 7.5)

    def test_sin_muestras_devuelve_none_en_vez_de_inventar_un_cero(self):
        # Devolver 0.0 seria peor que None: un p95 de 0 ms se lee como
        # "buenisimo" cuando en realidad no se midio nada.
        self.assertIsNone(percentil([], 95))

    def test_no_asume_la_lista_ordenada(self):
        self.assertEqual(percentil([100, 1, 50], 50), 50)


class SoloStdlibTest(unittest.TestCase):
    """El arnes no puede depender de nada que haya que instalar.

    Medido el 2026-09-11: la primera version importaba `httpx` y en atem-ai
    (.11) no existe -- ni en el python del sistema ni en ningun venv, porque esa
    maquina es PHP/Laravel. O sea que la politica "nada se lanza sin medir bajo
    carga" era inejecutable justo en la maquina donde vive AteneaERP, que es el
    proyecto con mas deuda de rendimiento del ecosistema.

    Una herramienta de politica que solo corre en una maquina no es una
    herramienta de politica. Este test lo fija: si alguien agrega una
    dependencia de terceros --arriba o dentro de una funcion-- se pone rojo.
    """

    def test_load_test_solo_importa_stdlib(self):
        import ast
        import sys

        ruta = Path(__file__).resolve().parents[1] / "scripts" / "load_test.py"
        arbol = ast.parse(ruta.read_text(encoding="utf-8"))
        modulos = set()
        for nodo in ast.walk(arbol):  # ast.walk: tambien los imports locales
            if isinstance(nodo, ast.Import):
                modulos.update(a.name.split(".")[0] for a in nodo.names)
            elif isinstance(nodo, ast.ImportFrom) and nodo.level == 0 and nodo.module:
                modulos.add(nodo.module.split(".")[0])
        ajenos = sorted(m for m in modulos if m not in sys.stdlib_module_names)
        self.assertEqual(
            ajenos, [],
            f"el arnes importa dependencias de terceros: {ajenos}. Tiene que correr "
            "en cualquier maquina del ecosistema sin instalar nada -- en atem-ai no "
            "hay httpx ni venv de python.",
        )


class ResumenTest(unittest.TestCase):
    def test_los_errores_no_entran_en_las_latencias_pero_si_en_la_tasa(self):
        # Una peticion que fallo rapido NO puede bajar el p95: es el modo en que
        # un arnes casero declara "todo bien" mientras la app rechaza la mitad
        # del trafico.
        r = resumen(latencias_ok=[100.0, 100.0], n_errores=2, segundos=1.0)
        self.assertEqual(r["p95_ms"], 100.0)
        self.assertEqual(r["errores"], 2)
        self.assertEqual(r["tasa_error"], 0.5)
        self.assertEqual(r["rps"], 4.0, "el rps cuenta TODAS las peticiones, fallidas incluidas")

    def test_todo_error_no_finge_latencia(self):
        r = resumen(latencias_ok=[], n_errores=5, segundos=2.0)
        self.assertIsNone(r["p95_ms"])
        self.assertEqual(r["tasa_error"], 1.0)


class CampoVacioAlFinalTest(unittest.TestCase):
    """La precondicion se vuelve a mirar AL FINAL (2026-09-17).

    ROJO CONTRA `44250a1`: `campo_vacio` no existia. La re-medicion del
    pre-vuelo de ese dia se declaro "sin sondas" mirando `sondeadas: []` UNA
    vez, antes de medir; a mitad de la corrida otra sesion borro las filas de
    salud sembradas, el pre-vuelo sondeo y salieron TRES llamadas reales a
    proveedores pagos que nadie vio hasta revisar `axioma_usage` despues.
    """

    def _servidor(self, cuerpo: bytes, codigo: int = 200):
        import http.server
        import threading

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(codigo)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(cuerpo)))
                self.end_headers()
                self.wfile.write(cuerpo)

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_address[1]}/"

    def test_campo_vacio_es_vacio(self):
        url = self._servidor(b'{"sondeadas": []}')
        vacio, valor = campo_vacio(url, "POST", "{}", {}, 5.0, "sondeadas")
        self.assertTrue(vacio)
        self.assertEqual(valor, [])

    def test_campo_con_contenido_no_es_vacio(self):
        """El caso real: la sonda se disparo durante la medicion."""
        url = self._servidor(b'{"sondeadas": ["hipatia", "jekyll", "kimi"]}')
        vacio, valor = campo_vacio(url, "POST", "{}", {}, 5.0, "sondeadas")
        self.assertFalse(vacio)
        self.assertEqual(valor, ["hipatia", "jekyll", "kimi"])

    def test_campo_ausente_no_se_lee_como_vacio(self):
        """Fail-closed: "no pude mirar" no es "esta limpio"."""
        url = self._servidor(b'{"ok": true}')
        vacio, valor = campo_vacio(url, "POST", "{}", {}, 5.0, "sondeadas")
        self.assertFalse(vacio)
        self.assertIn("no verificable", str(valor))

    def test_respuesta_ilegible_no_se_lee_como_vacio(self):
        url = self._servidor(b'no soy json')
        vacio, valor = campo_vacio(url, "POST", "{}", {}, 5.0, "sondeadas")
        self.assertFalse(vacio)
        self.assertIn("no verificable", str(valor))


if __name__ == "__main__":
    unittest.main()
