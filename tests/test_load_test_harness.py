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

from load_test import percentil, resumen  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
