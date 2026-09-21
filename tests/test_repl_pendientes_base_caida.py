"""`handle_pendientes_command` (jax/core/main.py) trataba `None` (base caida)
igual que `False` (el pendiente no existe) en la rama `/pendientes done`:

    ok = await db.mark_action_item_done(int(arg))
    return f"..." if ok else f"No encontre el pendiente #{arg}."

`if ok` es falsy para los dos casos. Es EXACTAMENTE el mismo aplastamiento de
borde que ya se arreglo en `handle_fact_command`/`verify_fact`
(tests/test_repl_fact_verify_autoria.py, `OtrasRamasSinCambiarTest` y
`VerifyConAutoriaTest::test_base_caida_no_se_reporta_como_hecho_inexistente`)
-- pendiente notado el 2026-09-20, arreglado el 2026-09-21 (recordatorio
agendado en `claude-skills/PENDIENTES.md` para el 2026-09-27: esa fecha es
de AVISO, no de arreglo -- ver el hallazgo #4 de la auditoria adversarial
de jax#247). El arreglo de la capa de datos (`mark_action_item_done`
distingue `True`/`False`/`None`, ver
tests/test_accion_persona_retorno_no_ambiguo.py) no sirve de nada si el
consumidor lo vuelve a aplastar en el borde.

Sin base de datos a proposito: es un contrato de logica sobre el llamador
REPL, no sobre MemoryDB en si.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jax.core.main import handle_pendientes_command  # noqa: E402


def _db_falso():
    db = mock.MagicMock()
    db.mark_action_item_done = mock.AsyncMock(return_value=True)
    db.get_action_items = mock.AsyncMock(return_value=[])
    return db


class PendientesDoneTest(unittest.IsolatedAsyncioTestCase):
    async def test_caso_feliz_dice_marcado(self):
        db = _db_falso()
        salida = await handle_pendientes_command(db, "/pendientes done 7")
        db.mark_action_item_done.assert_awaited_once_with(7)
        self.assertIn("marcado como hecho", salida)

    async def test_no_existe_dice_no_encontre(self):
        db = _db_falso()
        db.mark_action_item_done = mock.AsyncMock(return_value=False)
        salida = await handle_pendientes_command(db, "/pendientes done 7")
        self.assertIn("No encontre", salida)

    async def test_base_caida_no_se_reporta_como_pendiente_inexistente(self):
        """El defecto: con `None` (base caida), la rama vieja caia en el
        `else` y afirmaba 'No encontre el pendiente #7' -- una mentira: el
        pendiente puede existir perfectamente, lo unico que fallo fue la
        conexion."""
        db = _db_falso()
        db.mark_action_item_done = mock.AsyncMock(return_value=None)
        salida = await handle_pendientes_command(db, "/pendientes done 7")
        self.assertNotIn("No encontre", salida, (
            "un None (base caida) no puede leerse como 'el pendiente no existe'"))


if __name__ == "__main__":
    unittest.main()
