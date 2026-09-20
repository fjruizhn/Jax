"""El comando REPL /fact verify tiene que saber quien aprueba.

Hallazgo reportado en la ronda 2026-09-20 (plan memoria-admin, Task 2):
`MemoryDB.verify_fact` cambio de `(fact_id)` a `(fact_id, verified_by)`, SIN
default a proposito -- un aprobador implicito es un aprobador inventado.
`handle_fact_command` (jax/core/main.py) seguia llamando a
`db.verify_fact(int(arg))` con UN solo argumento.

Por como esta decorado `verify_fact` (`db_error_handler`, que atrapa
CUALQUIER excepcion y la convierte en `None` + un log de ERROR), ese
`TypeError` de aridad NO llegaba a nadie: se tragaba en silencio y el REPL
respondia "No encontre el hecho #N." -- una MENTIRA (el hecho existe; lo que
fallo fue la llamada). Confirmado a mano contra el codigo de esta ronda:

    DB error en verify_fact: MemoryDB.verify_fact() missing 1 required
    positional argument: 'verified_by'
    ...
    'No encontre el hecho #7.'

La correccion (decision de Fernando, no se reabre): `handle_fact_command`
recibe `repl_uid` OBLIGATORIO y sin default -- mismo criterio que
`verify_fact`. Hay UN solo llamador (jax/core/main.py:713), asi que no hay
excusa para un default. Si `repl_uid` es falsy (None o 0, JAX_REPL_USER_ID
sin configurar), NO se llama a `verify_fact` -- mismo patron que el candado
de `save_fact` (Task 2 Step 4 del plan de memoria-admin).

Sin base de datos a proposito: son fallos/contratos de logica sobre el
llamador REPL, no sobre MemoryDB en si (esa parte ya la cubre
tests/test_memoria_gobernanza.py).
"""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jax.core.main import handle_fact_command  # noqa: E402


def _db_falso():
    """Un `db` mockeado, sin tocar MemoryDB real ni la red."""
    db = mock.MagicMock()
    db.verify_fact = mock.AsyncMock(return_value=True)
    db.get_facts = mock.AsyncMock(return_value=[])
    db.get_fact_text = mock.AsyncMock(return_value=None)
    db.delete_fact = mock.AsyncMock(return_value=True)
    return db


class ContratoDeLaFirmaTest(unittest.TestCase):
    def test_handle_fact_command_exige_repl_uid_sin_default(self):
        """Mismo criterio que MemoryDB.verify_fact: el parametro es
        OBLIGATORIO (no puede tener default), aunque su VALOR legitimamente
        pueda ser None (JAX_REPL_USER_ID sin configurar) -- eso lo decide la
        rama 'verify', no la firma."""
        firma = inspect.signature(handle_fact_command)
        assert "repl_uid" in firma.parameters, (
            "handle_fact_command no recibe repl_uid -- sigue sin poder "
            "atribuir quien aprueba en /fact verify")
        assert firma.parameters["repl_uid"].default is inspect.Parameter.empty, (
            "repl_uid tiene un default: un aprobador implicito es un "
            "aprobador inventado, igual que en verify_fact")


class VerifyConAutoriaTest(unittest.IsolatedAsyncioTestCase):
    async def test_llamada_vieja_de_tres_argumentos_ya_no_alcanza(self):
        """El TypeError que ANTES de este arreglo `db_error_handler` tragaba
        en silencio (dentro de verify_fact) ahora se ve ACA, en el limite de
        handle_fact_command, apenas alguien intenta llamarlo sin decir quien
        es -- Python lo rechaza antes de que el bug llegue a esconderse en
        un log que nadie lee."""
        db = _db_falso()
        with self.assertRaises(TypeError):
            await handle_fact_command(db, "/fact verify 7", {})  # sin repl_uid

    async def test_con_repl_uid_valido_verify_fact_recibe_los_dos_argumentos(self):
        """No alcanza con que no explote: verify_fact tiene que recibir el
        fact_id Y el repl_uid, en ese orden."""
        db = _db_falso()
        salida = await handle_fact_command(db, "/fact verify 7", {}, repl_uid=42)
        db.verify_fact.assert_awaited_once_with(7, 42)
        self.assertIn("verificado", salida)

    async def test_sin_repl_uid_no_se_llama_a_verify_fact(self):
        """JAX_REPL_USER_ID sin configurar -> repl_uid=None. Aprobar sin
        dueno es un sello vacio (mismo criterio que el candado de
        save_fact): NI SE INTENTA."""
        db = _db_falso()
        salida = await handle_fact_command(db, "/fact verify 7", {}, repl_uid=None)
        db.verify_fact.assert_not_awaited()
        self.assertIn("JAX_REPL_USER_ID", salida)

    async def test_repl_uid_cero_tampoco_llama_a_verify_fact(self):
        """0 no es un id de usuario valido en este sistema -- cuenta como
        'no se sabe', igual que en el candado de save_fact."""
        db = _db_falso()
        salida = await handle_fact_command(db, "/fact verify 7", {}, repl_uid=0)
        db.verify_fact.assert_not_awaited()
        self.assertIn("JAX_REPL_USER_ID", salida)

    async def test_uso_invalido_sigue_sin_tocar_verify_fact(self):
        """Control: /fact verify sin id numerico sigue devolviendo el
        mensaje de uso, con o sin repl_uid -- este camino no cambio."""
        db = _db_falso()
        salida = await handle_fact_command(db, "/fact verify", {}, repl_uid=42)
        db.verify_fact.assert_not_awaited()
        self.assertIn("Uso:", salida)


class OtrasRamasSinCambiarTest(unittest.IsolatedAsyncioTestCase):
    """Las ramas que el hallazgo NO senalo (list, delete, confirm, help)
    siguen aceptando repl_uid (ahora obligatorio para TODA la funcion) sin
    que su comportamiento propio cambie -- ninguna de las tres toca un
    metodo cuya aridad se movio en esta ronda."""

    async def test_list_no_usa_repl_uid_para_nada(self):
        db = _db_falso()
        db.get_facts = mock.AsyncMock(return_value=[])
        await handle_fact_command(db, "/fact list", {}, repl_uid=None)
        db.get_facts.assert_awaited_once()

    async def test_delete_no_usa_repl_uid_para_nada(self):
        db = _db_falso()
        db.get_fact_text = mock.AsyncMock(return_value="un hecho cualquiera")
        salida = await handle_fact_command(db, "/fact delete 7", {}, repl_uid=None)
        self.assertIn("Vas a borrar", salida)

    async def test_confirm_no_usa_repl_uid_para_nada(self):
        db = _db_falso()
        db.delete_fact = mock.AsyncMock(return_value=True)
        salida = await handle_fact_command(
            db, "/fact confirm", {"id": 7, "text": "x"}, repl_uid=None)
        self.assertIn("borrado", salida)

    async def test_ayuda_no_usa_repl_uid_para_nada(self):
        db = _db_falso()
        salida = await handle_fact_command(db, "/fact", {}, repl_uid=None)
        self.assertIn("Comandos de memoria", salida)


if __name__ == "__main__":
    unittest.main()
