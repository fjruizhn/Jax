#!/usr/bin/env python3
"""T3 (2026-08-22, auditoria usage_writer): chequeo de reconciliación
motor_jobs.jsonl vs axioma_usage. Réplica de la lógica de cálculo pura
(mismo patrón que tests/test_jacobs_director.py) -- sin DB ni filesystem
real, eso lo cubre check_usage_reconciliation() en vivo (reaper.py).

Corre con:
  PYTHONPATH=/home/fruiz/jax/las_manos .venv/bin/python jacobs/_usage_reconciliation_test.py
"""
from __future__ import annotations

import unittest

from jacobs.reaper import _compute_http_direct_gap, _compute_reconciliation_gap


class UsageReconciliationTest(unittest.TestCase):
    def test_todo_reconciliado_gap_cero(self):
        jobs = [{"job_id": "a", "has_usage": True}, {"job_id": "b", "has_usage": True}]
        result = _compute_reconciliation_gap(jobs, {"a", "b"})
        self.assertEqual(result["gap_pct"], 0.0)
        self.assertEqual(result["expected"], 2)
        self.assertEqual(result["missing"], [])

    def test_nada_reconciliado_gap_100(self):
        """El caso real que disparó esta auditoría: 0/9 dispatches
        reconciliados."""
        jobs = [{"job_id": f"j{i}", "has_usage": True} for i in range(9)]
        result = _compute_reconciliation_gap(jobs, set())
        self.assertEqual(result["gap_pct"], 100.0)
        self.assertEqual(result["expected"], 9)
        self.assertEqual(len(result["missing"]), 9)

    def test_jobs_sin_usage_no_cuentan_como_esperados(self):
        """Un job que falló ANTES de llamar al LLM (0 tokens) no debe
        contar como 'esperado' -- no hay fila que buscar."""
        jobs = [{"job_id": "a", "has_usage": True}, {"job_id": "b", "has_usage": False}]
        result = _compute_reconciliation_gap(jobs, {"a"})
        self.assertEqual(result["expected"], 1)
        self.assertEqual(result["gap_pct"], 0.0)

    def test_sin_jobs_esperados_gap_cero_no_division_por_cero(self):
        result = _compute_reconciliation_gap([], set())
        self.assertEqual(result["gap_pct"], 0.0)
        self.assertEqual(result["expected"], 0)

    def test_parcial_reporta_porcentaje_correcto(self):
        jobs = [{"job_id": f"j{i}", "has_usage": True} for i in range(4)]
        result = _compute_reconciliation_gap(jobs, {"j0", "j1", "j2"})
        self.assertEqual(result["gap_pct"], 25.0)
        self.assertEqual(result["missing"], ["j3"])


class HttpDirectReconciliationGapTest(unittest.TestCase):
    """T5 (2026-08-21, punto ciego): check_usage_reconciliation() solo
    cubría Motor Registry. Estos tests cubren la lógica pura del camino
    HTTP-directo (hipatia/jekyll/thot/ada) -- aproximada por diseño
    (conteo por facet, sin job_id/step_id), ver _compute_http_direct_gap."""

    def test_todo_reconciliado_gap_cero(self):
        expected = {"jekyll": 2, "hipatia": 1}
        actual = {"jekyll": 2, "hipatia": 1}
        result = _compute_http_direct_gap(expected, actual)
        self.assertEqual(result["gap_pct"], 0.0)
        self.assertEqual(result["expected"], 3)
        self.assertEqual(result["missing_by_facet"], {})

    def test_nada_reconciliado_gap_100(self):
        """El caso real encontrado auditando: el pipeline
        8d02047d-9c1b-4da0-9586-db643fd7472d completó 4 steps
        HTTP-directos (jekyll/hipatia/thot/ada) y ninguno tiene fila."""
        expected = {"jekyll": 1, "hipatia": 1, "thot": 1, "ada": 1}
        actual: dict[str, int] = {}
        result = _compute_http_direct_gap(expected, actual)
        self.assertEqual(result["gap_pct"], 100.0)
        self.assertEqual(result["expected"], 4)
        self.assertEqual(
            result["missing_by_facet"], {"ada": 1, "hipatia": 1, "jekyll": 1, "thot": 1}
        )

    def test_exceso_en_un_facet_no_tapa_gap_real_en_otro(self):
        """Sin capear por facet, sumar todo antes de comparar dejaría
        que 5 filas de jekyll de más (otra ventana, otro escritor)
        tapen que hipatia perdió su única fila -- gap_pct daría 0%
        cuando en realidad hipatia está en 100%. El cap por facet evita
        exactamente esto."""
        expected = {"jekyll": 1, "hipatia": 1}
        actual = {"jekyll": 6, "hipatia": 0}
        result = _compute_http_direct_gap(expected, actual)
        self.assertEqual(result["missing_by_facet"], {"hipatia": 1})
        self.assertEqual(result["gap_pct"], 50.0)

    def test_sin_expected_gap_cero_no_division_por_cero(self):
        result = _compute_http_direct_gap({}, {})
        self.assertEqual(result["gap_pct"], 0.0)
        self.assertEqual(result["expected"], 0)

    def test_parcial_reporta_porcentaje_y_missing_by_facet_correcto(self):
        expected = {"jekyll": 3, "ada": 1}
        actual = {"jekyll": 2, "ada": 1}
        result = _compute_http_direct_gap(expected, actual)
        self.assertEqual(result["expected"], 4)
        self.assertEqual(result["reconciled"], 3)
        self.assertEqual(result["gap_pct"], 25.0)
        self.assertEqual(result["missing_by_facet"], {"jekyll": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
