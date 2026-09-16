#!/usr/bin/env python3
"""Un monitor que no pudo medir NO puede decir OK.

POR QUÉ EXISTE. `check_usage_reconciliation()` atrapa el fallo de cada camino
y deja `{"expected": 0, ..., "error": True}`. La condición de alerta es
`expected > 0 and gap > umbral`, así que con el chequeo caído `expected` vale 0,
la condición es falsa y —hasta el 2026-09-16— caía en el `else`, que escribía
«reconciliación de usage OK -- 0/0 dispatches con fila (0.0% gap)».

`error: True` se escribía y no lo leía nadie. El vigilante reportaba verde
justo cuando se había quedado ciego, que es peor que no tener vigilante: apaga
la sospecha en vez de encenderla. Misma familia que «un control que no falla no
valida nada», en la capa del monitor.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jacobs import reaper  # noqa: E402


class ReaperNoReportaVerdeSiFalloTest(unittest.TestCase):
    def _correr_con_todo_roto(self):
        """Los dos caminos del chequeo fallan; se capturan los logs."""
        boom = mock.AsyncMock(side_effect=RuntimeError("la base no responde"))
        with mock.patch.object(reaper, "_fetch_reconciled_job_ids", boom), \
             mock.patch.object(reaper, "_fetch_http_direct_expected", boom), \
             mock.patch.object(reaper, "_fetch_http_direct_actual", boom), \
             mock.patch.object(reaper, "_load_terminal_motor_jobs",
                               mock.Mock(side_effect=RuntimeError("log ilegible"))), \
             mock.patch.object(reaper, "send_telegram_alert", mock.AsyncMock()):
            # INFO, no WARNING: el mensaje que hay que cazar («OK -- 0/0
            # dispatches») se emite con logger.info, así que capturar desde
            # WARNING lo dejaba fuera y el test pasaba contra el código
            # defectuoso. Detectado al verificar que el test fallaba antes del
            # arreglo —— el control del control.
            with self.assertLogs(reaper.logger, level=logging.INFO) as capturado:
                resultado = asyncio.run(reaper.check_usage_reconciliation())
        return resultado, "\n".join(capturado.output)

    def test_no_dice_OK_cuando_el_chequeo_fallo(self):
        resultado, logs = self._correr_con_todo_roto()
        self.assertTrue(resultado["motor_registry"].get("error"))
        self.assertTrue(resultado["http_direct"].get("error"))
        self.assertNotIn(
            "OK --", logs,
            "el monitor reportó OK con el chequeo caído:\n" + logs)

    def test_dice_explicitamente_que_no_hay_veredicto(self):
        _, logs = self._correr_con_todo_roto()
        self.assertIn("SIN VEREDICTO", logs)
        for camino in ("Motor Registry", "HTTP-directo"):
            self.assertIn(camino, logs, f"no se declaró el fallo del camino {camino}")

    def test_el_caso_sano_sigue_diciendo_OK(self):
        """Control del control: si no hay error, el mensaje de siempre no cambia."""
        with mock.patch.object(reaper, "_load_terminal_motor_jobs", mock.Mock(return_value=[])), \
             mock.patch.object(reaper, "_fetch_reconciled_job_ids", mock.AsyncMock(return_value=[])), \
             mock.patch.object(reaper, "_fetch_http_direct_expected", mock.AsyncMock(return_value={})), \
             mock.patch.object(reaper, "_fetch_http_direct_actual", mock.AsyncMock(return_value={})), \
             mock.patch.object(reaper, "send_telegram_alert", mock.AsyncMock()):
            with self.assertLogs(reaper.logger, level=logging.INFO) as capturado:
                resultado = asyncio.run(reaper.check_usage_reconciliation())
        logs = "\n".join(capturado.output)
        self.assertNotIn("SIN VEREDICTO", logs)
        self.assertIn("OK --", logs)
        self.assertFalse(resultado["motor_registry"].get("error"))


if __name__ == "__main__":
    unittest.main()
