"""Descartar pipelines: columnas e índices contra la base de TEST (spec
2026-09-22-descartar-pipelines §3). `store.get_pool`/`store.init_schema`
del brief no existen en este repo: los nombres reales son
`store.conexion()`/`store.init_tables()`, los mismos que usa
`tests/test_jacobs_reaper_cas_db.py`.

Fix round 1 (2026-09-22, revisión del coordinador): "strengthen the reaper
guard" -- `tests/test_jacobs_descarte.py::test_el_reaper_solo_cosecha_no_terminales`
solo mira el texto fuente de `jacobs/reaper.py` (grep de la lista de estados
no-terminales). Acá se agrega la prueba de comportamiento real: un pipeline
`discarded` y uno `hidden`, los dos con `updated_at`/`created_at` MUY viejos
(muy por encima de cualquier umbral del reaper), sobreviven a un barrido
REAL de `reaper.reap_orphaned_pipelines()` contra la base de TEST -- no un
mock del barrido, la función que corre en producción."""
from __future__ import annotations

import json
import time
import unittest
import uuid

from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

from jacobs import descarte, reaper, store  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus  # noqa: E402

#: Mismo mapeo que jacobs/routes.py::_EVENTO_DE -- Task 3, fix round 1
#: (Ruling 9): `store.pipeline_transicion_descarte` ahora recibe el evento y
#: lo escribe en la MISMA transacción que el CAS. Este helper arma esos dos
#: kwargs para no repetir el payload en cada llamada de este archivo.
_EVENTO_DE = {
    "discard": "PIPELINE_DISCARDED", "recover": "PIPELINE_RECOVERED",
    "hide": "PIPELINE_HIDDEN", "restore": "PIPELINE_RESTORED",
}


async def _transicion(pid: str, epoca: int, accion: str, *,
                       desde: PipelineStatus, a: PipelineStatus, user_id: str) -> bool:
    return await store.pipeline_transicion_descarte(
        pid, epoca, accion, desde=desde, a=a, user_id=user_id,
        evento_tipo=_EVENTO_DE[accion],
        evento_payload={"user_id": user_id, "desde": desde.value, "a": a.value},
    )


class DescarteColumnasEIndicesDBTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)

    async def _columnas(self) -> dict[str, str]:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SHOW COLUMNS FROM jacobs_pipelines")
                return {fila[0]: fila[1] for fila in await cur.fetchall()}

    async def test_columnas_del_descarte_existen(self):
        await store.init_tables()
        cols = await self._columnas()
        self.assertEqual(cols["status_previo"].lower(), "varchar(20)")
        self.assertEqual(cols["descartado_por"].lower(), "varchar(50)")
        self.assertEqual(cols["descartado_at"].lower(), "double")

    async def test_indices_del_descarte_existen(self):
        await store.init_tables()
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SHOW INDEX FROM jacobs_pipelines")
                filas = await cur.fetchall()
        por_indice: dict[str, list[tuple[int, str]]] = {}
        for f in filas:
            por_indice.setdefault(f[2], []).append((f[3], f[4]))
        cols = {k: [c for _, c in sorted(v)] for k, v in por_indice.items()}
        self.assertEqual(
            cols["idx_pipelines_descartados"],
            ["user_id", "tenant_id", "status", "descartado_at"],
        )
        self.assertEqual(cols["idx_pipelines_ocultos"], ["status", "descartado_at"])


class ReaperNoTocaDescartadosNiOcultosDBTest(unittest.IsolatedAsyncioTestCase):
    """Comportamiento real (no grep): el reaper NO cosecha discarded/hidden,
    ni siquiera con updated_at/created_at extremadamente stale."""

    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)
        self._pids: list[str] = []

    async def asyncTearDown(self):
        for pid in self._pids:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("DELETE FROM jacobs_events WHERE pipeline_id=%s", (pid,))
                    await cur.execute("DELETE FROM jacobs_steps WHERE pipeline_id=%s", (pid,))
                    await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
                await conn.commit()

    async def _crear_stale(self, status: PipelineStatus) -> str:
        pid = str(uuid.uuid4())
        # Muy por encima de PENDING_MAX_AGE_SECONDS (300s), RUNNING_STALE_SECONDS
        # (1800s) e INTERRUPTED_NO_OWNER_MAX_AGE_SECONDS (600s) -- si el reaper
        # clasificara mal a discarded/hidden como "candidato", esta antigüedad
        # los cosecharía con cualquiera de los tres umbrales.
        viejo = time.time() - 10 * reaper.RUNNING_STALE_SECONDS
        await store.init_tables()
        await store.pipeline_create(Pipeline(
            pipeline_id=pid, name="t-reaper-descarte", invoked_by="plataforma",
            mode="autonomous", status=status, created_at=viejo, updated_at=viejo,
            run_epoch=0,
        ))
        self._pids.append(pid)
        return pid

    async def test_el_reaper_no_toca_un_discarded_stale(self):
        pid = await self._crear_stale(PipelineStatus.discarded)
        cosechados = await reaper.reap_orphaned_pipelines()
        self.assertNotIn(pid, [c["pipeline_id"] for c in cosechados])
        fila = await store.pipeline_get(pid)
        self.assertEqual(fila.status, PipelineStatus.discarded)

    async def test_el_reaper_no_toca_un_hidden_stale(self):
        pid = await self._crear_stale(PipelineStatus.hidden)
        cosechados = await reaper.reap_orphaned_pipelines()
        self.assertNotIn(pid, [c["pipeline_id"] for c in cosechados])
        fila = await store.pipeline_get(pid)
        self.assertEqual(fila.status, PipelineStatus.hidden)


class TransicionDescarteCasDBTest(unittest.IsolatedAsyncioTestCase):
    """Task 2 (spec §3): `store.pipeline_transicion_descarte` contra MariaDB
    real. Mismo `Pipeline(...)` que `tests/test_jacobs_reaper_cas_db.py`
    (`user_id="u1"`, `tenant_id="1"`, `run_epoch=3`), pero `status=aborted`:
    el pipeline detenido del que arranca el descarte. `store.pipeline_create`
    SÍ acepta `run_epoch` como campo del modelo (lo mismo que usa la fixture
    `_running` del reaper) -- no hace falta ningún UPDATE aparte."""

    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)
        await store.init_tables()
        self._pids: list[str] = []
        self.pid = await self._crear(PipelineStatus.aborted)
        self.addAsyncCleanup(self._borrar)

    async def _crear(self, status: PipelineStatus) -> str:
        """Fix round 1 (I-1): algunas de las pruebas de `validar_transicion`
        necesitan una fila que arranque en un estado DISTINTO de `aborted`
        (p.ej. `running`, `expired`) -- `self.pid` solo cubre el caso base."""
        pid = str(uuid.uuid4())
        await store.pipeline_create(Pipeline(
            pipeline_id=pid, name="t-descarte-cas", invoked_by="plataforma",
            mode="autonomous", status=status,
            user_id="u1", tenant_id="1", run_epoch=3,
        ))
        self._pids.append(pid)
        return pid

    async def _borrar(self):
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                for pid in self._pids:
                    await cur.execute("DELETE FROM jacobs_events WHERE pipeline_id=%s", (pid,))
                    await cur.execute("DELETE FROM jacobs_steps WHERE pipeline_id=%s", (pid,))
                    await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))

    async def _fila(self, pid: str | None = None) -> tuple:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT status, status_previo, descartado_por, descartado_at "
                    "FROM jacobs_pipelines WHERE pipeline_id=%s",
                    (pid or self.pid,),
                )
                return await cur.fetchone()

    async def test_descartar_escribe_estado_y_columnas_en_la_misma_escritura(self):
        ok = await _transicion(
            self.pid, 3, "discard",
            desde=PipelineStatus.aborted, a=PipelineStatus.discarded, user_id="u1")
        self.assertTrue(ok)
        fila = await self._fila()
        self.assertEqual(fila[:3], ("discarded", "aborted", "u1"))
        self.assertIsNotNone(fila[3])

    async def test_descartar_con_epoca_vieja_no_escribe(self):
        ok = await _transicion(
            self.pid, 2, "discard",
            desde=PipelineStatus.aborted, a=PipelineStatus.discarded, user_id="u1")
        self.assertFalse(ok)
        self.assertEqual((await self._fila())[0], "aborted")

    async def test_descartar_desde_otro_estado_no_escribe(self):
        # M-1 (fix round 1): `expired` SÍ está permitido para "discard" en
        # general (TRANSICIONES["discard"]) -- lo que falla acá es el CAS
        # (la fila real está en `aborted`, no en `expired`), no la validación.
        ok = await _transicion(
            self.pid, 3, "discard",
            desde=PipelineStatus.expired, a=PipelineStatus.discarded, user_id="u1")
        self.assertFalse(ok)
        self.assertEqual((await self._fila())[0], "aborted")

    async def test_recuperar_limpia_las_tres_columnas(self):
        await _transicion(
            self.pid, 3, "discard",
            desde=PipelineStatus.aborted, a=PipelineStatus.discarded, user_id="u1")
        ok = await _transicion(
            self.pid, 3, "recover",
            desde=PipelineStatus.discarded, a=PipelineStatus.aborted, user_id="u1")
        self.assertTrue(ok)
        self.assertEqual(await self._fila(), ("aborted", None, None, None))

    async def test_ocultar_y_restaurar_conservan_las_columnas(self):
        await _transicion(
            self.pid, 3, "discard",
            desde=PipelineStatus.aborted, a=PipelineStatus.discarded, user_id="u1")
        antes = await self._fila()
        ok_hide = await _transicion(
            self.pid, 3, "hide",
            desde=PipelineStatus.discarded, a=PipelineStatus.hidden, user_id="admin")
        self.assertTrue(ok_hide)
        self.assertEqual((await self._fila())[1:], antes[1:])
        ok_restore = await _transicion(
            self.pid, 3, "restore",
            desde=PipelineStatus.hidden, a=PipelineStatus.discarded, user_id="admin")
        self.assertTrue(ok_restore)
        self.assertEqual(await self._fila(), ("discarded",) + antes[1:])

    # Fix round 1 (2026-09-22, Ruling 7, I-1): la escritura tiene que exigir
    # la transición permitida (descarte.TRANSICIONES), no solo la época y el
    # estado exacto. Sin esto, `discard` desde `running` liberaba el cupo de
    # un pipeline que sigue ejecutando y lo dejaba huérfano para siempre.

    async def test_descartar_desde_running_rechaza_y_no_escribe(self):
        pid = await self._crear(PipelineStatus.running)
        with self.assertRaises(descarte.TransicionDescarteInvalida):
            await _transicion(
                pid, 3, "discard",
                desde=PipelineStatus.running, a=PipelineStatus.discarded, user_id="u1")
        self.assertEqual((await self._fila(pid))[0], "running")

    async def test_ocultar_desde_aborted_rechaza_y_no_escribe(self):
        # self.pid arranca en `aborted` -- "hide" solo está permitido desde
        # `discarded` (la antesala obligatoria, spec §2: "nunca se oculta en
        # un paso").
        with self.assertRaises(descarte.TransicionDescarteInvalida):
            await _transicion(
                self.pid, 3, "hide",
                desde=PipelineStatus.aborted, a=PipelineStatus.hidden, user_id="admin")
        self.assertEqual((await self._fila())[0], "aborted")

    async def test_recuperar_con_a_que_no_coincide_con_status_previo_no_escribe(self):
        # `a=expired` es un destino VÁLIDO de recover en general (está en
        # TRANSICIONES["discard"]), pero esta fila se descartó desde
        # `aborted` -- no levanta (no es un error de contrato), simplemente
        # no hay fila que matchee el WHERE (`status_previo='expired'`).
        await _transicion(
            self.pid, 3, "discard",
            desde=PipelineStatus.aborted, a=PipelineStatus.discarded, user_id="u1")
        ok = await _transicion(
            self.pid, 3, "recover",
            desde=PipelineStatus.discarded, a=PipelineStatus.expired, user_id="u1")
        self.assertFalse(ok)
        fila = await self._fila()
        self.assertEqual(fila[0], "discarded")
        self.assertEqual(fila[1], "aborted")

    # M-2 (fix round 1): el ciclo completo con `expired`, no solo `aborted`
    # -- recuperar tiene que devolver al estado EXACTO previo.
    async def test_ciclo_completo_de_expired_descartar_y_recuperar(self):
        pid = await self._crear(PipelineStatus.expired)
        ok_discard = await _transicion(
            pid, 3, "discard",
            desde=PipelineStatus.expired, a=PipelineStatus.discarded, user_id="u1")
        self.assertTrue(ok_discard)
        self.assertEqual((await self._fila(pid))[:3], ("discarded", "expired", "u1"))
        ok_recover = await _transicion(
            pid, 3, "recover",
            desde=PipelineStatus.discarded, a=PipelineStatus.expired, user_id="u1")
        self.assertTrue(ok_recover)
        self.assertEqual(await self._fila(pid), ("expired", None, None, None))

    # Task 3 (2026-09-22-descartar-pipelines): `store.pipeline_status_previo`
    # es lo que lee la ruta de `/recover` para calcular `a` ANTES del CAS
    # (jacobs/routes.py::transicion_descarte). Contra la base real, no un
    # mock -- lo que se mockea en tests/test_jacobs_descarte.py es esta
    # misma función.
    async def test_status_previo_de_un_descartado(self):
        await _transicion(
            self.pid, 3, "discard",
            desde=PipelineStatus.aborted, a=PipelineStatus.discarded, user_id="u1")
        self.assertEqual(await store.pipeline_status_previo(self.pid), "aborted")

    async def test_status_previo_de_uno_que_nunca_se_descarto_es_none(self):
        self.assertIsNone(await store.pipeline_status_previo(self.pid))

    async def test_status_previo_de_un_pipeline_inexistente_es_none(self):
        self.assertIsNone(await store.pipeline_status_previo(str(uuid.uuid4())))


class TransicionDescarteAtomicaDBTest(unittest.IsolatedAsyncioTestCase):
    """Task 3, fix round 1 (2026-09-22, Ruling 9): el CAS y el evento de
    auditoría van en la MISMA transacción, sobre la MISMA conexión
    (`conexion_dedicada(found_rows=True)` + `transaccion()`, reutilizados de
    lo que el store ya usaba en otro lado -- no un mecanismo nuevo). Antes,
    la ruta llamaba a `store.event_append` en una SEGUNDA conexión, después
    del CAS: si esa escritura fallaba, la transición quedaba hecha SIN
    auditoría, y un reintento del llamador ya no la repetía (la fila dejó
    de estar en `desde`, así que el CAS siguiente da 409 antes de llegar al
    evento). En `recover`/`hide`/`restore` ese evento es el ÚNICO registro
    de quién hizo la transición -- `recover` además BORRA `descartado_por`."""

    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)
        await store.init_tables()
        self.pid = str(uuid.uuid4())
        await store.pipeline_create(Pipeline(
            pipeline_id=self.pid, name="t-descarte-atomico", invoked_by="plataforma",
            mode="autonomous", status=PipelineStatus.aborted,
            user_id="u1", tenant_id="1", run_epoch=3,
        ))
        self.addAsyncCleanup(self._borrar)

    async def _borrar(self):
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM jacobs_events WHERE pipeline_id=%s", (self.pid,))
                await cur.execute("DELETE FROM jacobs_steps WHERE pipeline_id=%s", (self.pid,))
                await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (self.pid,))

    async def _eventos(self) -> list[tuple]:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT event_type, payload FROM jacobs_events WHERE pipeline_id=%s",
                    (self.pid,))
                return list(await cur.fetchall())

    async def test_si_el_insert_del_evento_falla_el_update_no_queda(self):
        """El INSERT del evento revienta con un error REAL de MariaDB, no un
        mock.

        Fix round 2 (MAJOR de la revisión): un `mock.patch.object(store,
        "event_append", side_effect=...)` que levanta SIN mirar sus
        argumentos no distingue "el INSERT se intentó en la conexión
        correcta, dentro de la transacción, y falló" de "ni siquiera se
        llegó a intentar" -- si alguien saca `conexion=conn` en
        `store.py::pipeline_transicion_descarte`, el mock ciego de la
        versión anterior seguía en verde igual, porque de todos modos iba a
        levantar. Reemplazado por un fallo que la base MISMA produce: el
        `evento_payload` de este test lleva un `float("nan")` -- Python
        serializa `NaN`/`Infinity` por defecto (`json.dumps(..., allow_nan=True)`
        es el default), pero la gramática JSON estricta no los admite. La
        columna `payload JSON` de `jacobs_events` es, en MariaDB, un alias
        de `LONGTEXT` con un `CHECK (JSON_VALID(payload))` automático desde
        10.2.7 -- así que el INSERT choca con ese CHECK. Confirmado contra
        la base real antes de escribir esta versión:
        `(4025, "CONSTRAINT \\`jacobs_events.payload\\` failed for ...")`.

        Con el error viniendo de la base y no de un doble, la aserción de
        abajo (`fila.status == aborted`) sí depende de qué conexión recibió
        el INSERT: si el INSERT fallara en una conexión DISTINTA a la del
        UPDATE (la mutación de "volver a dos conexiones"), el UPDATE ya
        habría quedado autocommiteado en su propia conexión ANTES de que el
        evento fallara, y la fila terminaría en `discarded` -- la aserción
        cae. Con las dos en la MISMA transacción, el error del INSERT
        descarta también el UPDATE con ella (`transaccion()` cierra la
        conexión en vez de mandar `ROLLBACK`), y la fila queda en `aborted`."""
        payload_invalido = {
            "user_id": "u1", "desde": "aborted", "a": "discarded", "x": float("nan"),
        }
        with self.assertRaises(Exception) as ctx:
            await store.pipeline_transicion_descarte(
                self.pid, 3, "discard",
                desde=PipelineStatus.aborted, a=PipelineStatus.discarded, user_id="u1",
                evento_tipo="PIPELINE_DISCARDED", evento_payload=payload_invalido,
            )
        self.assertIn("4025", str(ctx.exception))
        fila = await store.pipeline_get(self.pid)
        self.assertEqual(fila.status, PipelineStatus.aborted)
        self.assertEqual(await self._eventos(), [])

    async def test_transicion_exitosa_deja_exactamente_un_evento_con_el_payload(self):
        ok = await _transicion(
            self.pid, 3, "discard",
            desde=PipelineStatus.aborted, a=PipelineStatus.discarded, user_id="u1")
        self.assertTrue(ok)
        eventos = await self._eventos()
        self.assertEqual(len(eventos), 1)
        tipo, payload_crudo = eventos[0]
        self.assertEqual(tipo, "PIPELINE_DISCARDED")
        self.assertEqual(
            json.loads(payload_crudo),
            {"user_id": "u1", "desde": "aborted", "a": "discarded"},
        )
