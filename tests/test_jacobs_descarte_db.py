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
mock del barrido, la función que corre en producción.

Task 1-bis (2026-09-22, Ruling 18): `visible` -- columna GENERATED VIRTUAL
(1 si status NOT IN ('discarded','hidden'), 0 si no) -- e
idx_pipelines_visibles (user_id, tenant_id, visible, created_at). El
listado principal de jax-platform hoy paga un costo LINEAL con el
histórico de descartados de un dueño (medido en jax-platform,
docs/carga-sql-pipelines-del-usuario-indice-2026-09-22.md: forma extrema,
4,2-4,4 ms con 5000 descartados y 3 vivos, recorriendo el tenant casi
completo antes de poder cortar en el LIMIT) -- este índice deja el filtro
DENTRO del índice, así que el costo queda acotado por el LIMIT sin
importar cuántas filas descartadas tenga el dueño. Ver
VisibleSigueAlEstadoDBTest (el valor de la columna sigue a `status`, en
creación y en cada transición) y CostoAcotadoPorVisibleDBTest (EXPLAIN +
contadores `Handler_read%` reales contra dos formas sembradas: la que
antes pagaba el costo lineal, y un historial largo de filas vivas)."""
from __future__ import annotations

import json
import time
import unittest
import uuid
from unittest import mock

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

    # Task 1-bis (2026-09-22, Ruling 18): `visible` -- GENERATED VIRTUAL,
    # TINYINT(1) -- e idx_pipelines_visibles, en ese orden EXACTO de
    # columnas (no cualquier orden: un índice con `visible` DESPUÉS de
    # `created_at` no serviría de nada para el filtro -- el motor sólo
    # puede usar un prefijo continuo del índice para IGUALDAD antes de
    # entrar al rango/orden).
    async def test_columna_visible_generada_existe(self):
        await store.init_tables()
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SHOW COLUMNS FROM jacobs_pipelines WHERE Field='visible'")
                (campo, tipo, nulo, clave, default, extra) = await cur.fetchone()
        self.assertEqual(tipo.lower(), "tinyint(1)")
        self.assertEqual(extra.upper(), "VIRTUAL GENERATED")

    async def test_indice_visibles_existe_en_orden(self):
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
            cols["idx_pipelines_visibles"],
            ["user_id", "tenant_id", "visible", "created_at"],
        )


# Task 1-bis (2026-09-22, Ruling 18/19d, MINOR-4): mapeo EXPLÍCITO
# PipelineStatus -> visibilidad esperada (con owner_ack_at YA puesto --
# Ruling 19a añade "AND owner_ack_at IS NOT NULL" a la expresión de
# `visible`; el mapeo de acá es sobre el caso "acked", el caso sin ack se
# prueba aparte). A propósito NO es una fórmula (`s not in (discarded,
# hidden)`) que un estado nuevo pasaría a integrar SOLO -- es un diccionario
# que alguien tiene que tocar a mano. `VisibilidadExhaustivaTest` de abajo
# exige que cubra TODO PipelineStatus; si el enum suma un estado nuevo sin
# decidir acá si es visible u oculto, ese test cae.
_VISIBILIDAD_ESPERADA: dict[PipelineStatus, bool] = {
    PipelineStatus.pending: True,
    PipelineStatus.running: True,
    PipelineStatus.completed: True,
    PipelineStatus.failed: True,
    PipelineStatus.aborted: True,
    PipelineStatus.interrupted: True,
    PipelineStatus.expired: True,
    PipelineStatus.disputed: True,
    PipelineStatus.discarded: False,
    PipelineStatus.hidden: False,
}
_ESTADOS_VISIBLES = tuple(s for s, v in _VISIBILIDAD_ESPERADA.items() if v)
_ESTADOS_NO_VISIBLES = tuple(s for s, v in _VISIBILIDAD_ESPERADA.items() if not v)


class VisibilidadExhaustivaTest(unittest.TestCase):
    """MINOR-4 (fix round 1, Ruling 19d): pura, sin DB -- si `PipelineStatus`
    suma un valor nuevo sin decidir su visibilidad en `_VISIBILIDAD_ESPERADA`,
    este test cae. Es la baranda que fuerza la decisión; DB real prueba que
    el motor calcula lo mismo (`VisibleSigueAlEstadoDBTest`)."""

    def test_todo_estado_del_enum_tiene_mapeo_de_visibilidad(self):
        faltan = set(PipelineStatus) - set(_VISIBILIDAD_ESPERADA)
        self.assertEqual(
            faltan, set(),
            f"PipelineStatus sin decisión de visibilidad en _VISIBILIDAD_ESPERADA: "
            f"{faltan} -- agregalo al mapeo (True/False) antes de seguir."
        )


class VisibleSigueAlEstadoDBTest(unittest.IsolatedAsyncioTestCase):
    """`visible` es GENERATED a partir de `status` Y `owner_ack_at` (Ruling
    19a) -- se prueba contra MariaDB real, no contra la expresión en
    abstracto: la fila se crea con cada `status` del enum y se lee la
    columna calculada por el motor."""

    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)
        await store.init_tables()
        self._pids: list[str] = []
        self.addAsyncCleanup(self._borrar)

    async def _borrar(self):
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                for pid in self._pids:
                    await cur.execute("DELETE FROM jacobs_events WHERE pipeline_id=%s", (pid,))
                    await cur.execute("DELETE FROM jacobs_steps WHERE pipeline_id=%s", (pid,))
                    await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))

    async def _crear(self, status: PipelineStatus, *, con_ack: bool = True) -> str:
        """`con_ack=True` (default): representa el caso normal, "el dueño
        reconoció este pipeline en la Mesa" (T6-5a). `con_ack=False`
        simula un hijo de Ada (jacobs/subpipelines.py) que todavía no tiene
        `owner_ack_at` -- `store.pipeline_create` NUNCA lo acepta como
        columna (siempre nace NULL, ver jacobs/store.py::pipeline_create),
        así que "con ack" necesita un UPDATE aparte después de crear."""
        pid = str(uuid.uuid4())
        await store.pipeline_create(Pipeline(
            pipeline_id=pid, name="t-visible", invoked_by="plataforma",
            mode="autonomous", status=status, user_id="u1", tenant_id="1",
        ))
        self._pids.append(pid)
        if con_ack:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "UPDATE jacobs_pipelines SET owner_ack_at=%s WHERE pipeline_id=%s",
                        (time.time(), pid),
                    )
                await conn.commit()
        return pid

    async def _visible(self, pid: str) -> int:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT visible FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
                (v,) = await cur.fetchone()
                return v

    async def test_visible_es_1_para_cada_estado_visible(self):
        for status in _ESTADOS_VISIBLES:
            with self.subTest(status=status):
                pid = await self._crear(status)
                self.assertEqual(await self._visible(pid), 1)

    async def test_visible_es_0_para_discarded_y_hidden(self):
        for status in _ESTADOS_NO_VISIBLES:
            with self.subTest(status=status):
                pid = await self._crear(status)
                self.assertEqual(await self._visible(pid), 0)

    async def test_visible_es_0_sin_ack_aunque_el_estado_sea_visible(self):
        """Ruling 19a: un hijo de Ada sin `owner_ack_at` (con_ack=False) es
        `visible=0` aunque su `status` sea uno de los ocho visibles -- el
        AND de la expresión, no sólo el NOT IN. Sin este AND, muchos hijos
        de Ada sin ack le devuelven a idx_pipelines_visibles el mismo costo
        sin techo que Ruling 18 quería evitar para los descartados."""
        for status in _ESTADOS_VISIBLES:
            with self.subTest(status=status):
                pid = await self._crear(status, con_ack=False)
                self.assertEqual(await self._visible(pid), 0)

    async def test_visible_es_1_con_ack_para_cada_estado_del_mapeo(self):
        """Cruza `_VISIBILIDAD_ESPERADA` completo (MINOR-4) contra MariaDB
        real, no sólo los dos grupos por separado."""
        for status, esperado in _VISIBILIDAD_ESPERADA.items():
            with self.subTest(status=status):
                pid = await self._crear(status, con_ack=True)
                self.assertEqual(await self._visible(pid), int(esperado))

    async def test_visible_cambia_a_0_al_descartar_y_a_1_al_recuperar(self):
        pid = await self._crear(PipelineStatus.aborted)
        self.assertEqual(await self._visible(pid), 1)
        ok = await _transicion(
            pid, 0, "discard",
            desde=PipelineStatus.aborted, a=PipelineStatus.discarded, user_id="u1")
        self.assertTrue(ok)
        self.assertEqual(await self._visible(pid), 0)
        ok = await _transicion(
            pid, 0, "recover",
            desde=PipelineStatus.discarded, a=PipelineStatus.aborted, user_id="u1")
        self.assertTrue(ok)
        self.assertEqual(await self._visible(pid), 1)

    async def test_visible_sigue_en_0_al_ocultar_un_descartado(self):
        """hide/restore van discarded<->hidden -- las dos puntas de ese ciclo
        son NO visibles; la propiedad interesante acá es que `visible` NO
        vuelve a 1 a mitad de camino."""
        pid = await self._crear(PipelineStatus.aborted)
        await _transicion(
            pid, 0, "discard",
            desde=PipelineStatus.aborted, a=PipelineStatus.discarded, user_id="u1")
        self.assertEqual(await self._visible(pid), 0)
        ok = await _transicion(
            pid, 0, "hide",
            desde=PipelineStatus.discarded, a=PipelineStatus.hidden, user_id="admin")
        self.assertTrue(ok)
        self.assertEqual(await self._visible(pid), 0)
        ok = await _transicion(
            pid, 0, "restore",
            desde=PipelineStatus.hidden, a=PipelineStatus.discarded, user_id="admin")
        self.assertTrue(ok)
        self.assertEqual(await self._visible(pid), 0)

    async def _aparece_por_el_indice(self, pid: str) -> bool:
        """MINOR-1 (fix round 1): no alcanza con leer la columna `visible`
        (`_visible()` de arriba) -- eso prueba que la EXPRESIÓN calcula
        bien, no que MariaDB mantenga el ÍNDICE consistente con la columna
        generada en cada UPDATE de `status`. `FORCE INDEX` obliga a leer
        POR el índice, no por un scan que ignoraría una inconsistencia."""
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT pipeline_id FROM jacobs_pipelines "
                    "FORCE INDEX (idx_pipelines_visibles) "
                    "WHERE user_id=%s AND tenant_id=%s AND visible=1",
                    ("u1", "1"),
                )
                vistos = {fila[0] for fila in await cur.fetchall()}
        return pid in vistos

    async def test_el_indice_refleja_el_descarte_y_la_recuperacion(self):
        pid = await self._crear(PipelineStatus.aborted)
        self.assertTrue(await self._aparece_por_el_indice(pid))
        ok = await _transicion(
            pid, 0, "discard",
            desde=PipelineStatus.aborted, a=PipelineStatus.discarded, user_id="u1")
        self.assertTrue(ok)
        self.assertFalse(await self._aparece_por_el_indice(pid))
        ok = await _transicion(
            pid, 0, "recover",
            desde=PipelineStatus.discarded, a=PipelineStatus.aborted, user_id="u1")
        self.assertTrue(ok)
        self.assertTrue(await self._aparece_por_el_indice(pid))


class CanarioTrampaInstantDBTest(unittest.IsolatedAsyncioTestCase):
    """MAJOR-1 (fix round 1, Ruling 19b, 2026-09-22): canario vivo contra
    MariaDB real de la trampa del 1845 -- ver
    tests/test_store_columna_descarte_acotada.py::test_visible_es_la_ultima_columna_del_loop
    y el comentario de `idx_pipelines_visibles` en jacobs/store.py."""

    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)
        await store.init_tables()  # deja idx_pipelines_visibles creado

    async def test_add_column_instant_sigue_fallando_con_visible_indexada(self):
        """Si este test se pone en ROJO, MariaDB dejó de rechazar
        ALGORITHM=INSTANT sobre una tabla con un índice sobre columna
        VIRTUAL -- la barrera de Ruling 19b (la regla de "visible al
        final" en test_store_columna_descarte_acotada.py) ya no hace
        falta contra ESTA versión de MariaDB, y hay que revisar las dos
        juntas antes de retirarlas."""
        try:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "ALTER TABLE jacobs_pipelines ADD COLUMN tmp_canario INT NULL, "
                        "ALGORITHM=INSTANT"
                    )
            aplico = True
        except Exception as e:  # fail-soft: no se sabe de antemano la clase exacta que MariaDB levanta (1845 hoy) -- self.assertIn("1845", mensaje) de abajo sigue exigiendo el mensaje correcto, así que un error DISTINTO (p.ej. de conexión) también pone este test en rojo
            aplico = False
            mensaje = str(e)
        if aplico:
            # No dejar la base de sesión con la columna de sobra ANTES de
            # fallar el test -- limpieza primero, fallo después.
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "ALTER TABLE jacobs_pipelines DROP COLUMN tmp_canario, ALGORITHM=COPY"
                    )
            self.fail(
                "ALGORITHM=INSTANT YA NO falla con idx_pipelines_visibles presente "
                "-- la barrera de Ruling 19b (MAJOR-1, Task 1-bis fix round 1) se "
                "puede retirar: revisar test_visible_es_la_ultima_columna_del_loop "
                "en tests/test_store_columna_descarte_acotada.py y el comentario de "
                "idx_pipelines_visibles en jacobs/store.py antes de sacarlos."
            )
        self.assertIn("1845", mensaje)


class DriftDeExpresionVisibleDBTest(unittest.IsolatedAsyncioTestCase):
    """MINOR-A (fix round 2, revisión del coordinador, 2026-09-22): el
    chequeo de existencia del loop de columnas ("existe" -> `continue`) no
    alcanza para una columna GENERATED -- una base con `visible` YA creada
    pero con una expresión VIEJA (p.ej. la del commit `02fbaed`, sin "AND
    owner_ack_at IS NOT NULL") pasaría ese chequeo en silencio.
    `store._verificar_expresion_visible` compara
    `information_schema.COLUMNS.GENERATION_EXPRESSION` contra
    `store._EXPRESION_VISIBLE`, normalizada, y `init_tables()` se aborta si
    difieren."""

    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)
        await store.init_tables()

    async def test_la_expresion_esperada_pasa_sin_error(self):
        """Caso normal (lo que corre en CADA test de este archivo, ya
        implícito) -- acá EXPLÍCITO: dos `init_tables()` seguidos, sin
        tocar nada entre medio, no levantan."""
        await store.init_tables()  # segunda llamada -- visible ya existe

    async def test_una_expresion_vieja_frena_init_tables(self):
        """La expresión de ANTES de Ruling 19a (commit 02fbaed): sin el AND
        de ack. Se crea a mano contra la base de TEST -- nunca revirtiendo
        store.py -- para simular una base que quedó atrás."""
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "ALTER TABLE jacobs_pipelines DROP INDEX idx_pipelines_visibles, "
                    "DROP COLUMN visible, ALGORITHM=COPY"
                )
                await cur.execute(
                    "ALTER TABLE jacobs_pipelines ADD COLUMN visible TINYINT(1) "
                    "GENERATED ALWAYS AS (status NOT IN ('discarded','hidden')) VIRTUAL, "
                    "ALGORITHM=INSTANT"
                )
        try:
            with self.assertRaises(RuntimeError) as ctx:
                await store.init_tables()
            mensaje = str(ctx.exception)
            self.assertIn("expresión DISTINTA", mensaje)
            self.assertIn("owner_ack_at", mensaje)
            self.assertIn("02fbaed", mensaje)
        finally:
            # Restaurar ANTES de que termine el test -- otros tests de la
            # misma sesión asumen la expresión correcta.
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "ALTER TABLE jacobs_pipelines DROP COLUMN visible, ALGORITHM=COPY"
                    )
            await store.init_tables()

    async def test_visible_no_generada_frena_init_tables(self):
        """Fix round 3 (revisión del coordinador, 2026-09-22): distinto del
        caso de arriba -- acá `visible` existe pero NO es GENERATED en
        absoluto (`GENERATION_EXPRESSION` es NULL). El código viejo
        trataba "fila con NULL" igual que "sin fila" (columna inexistente)
        y devolvía en silencio -- fail-OPEN: con una columna COMÚN en vez
        de generada, TODAS las filas leerían visible=1 sin importar su
        status/owner_ack_at. Se crea a mano contra la base de TEST --
        nunca revirtiendo store.py -- para simular ese error operativo."""
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "ALTER TABLE jacobs_pipelines DROP INDEX idx_pipelines_visibles, "
                    "DROP COLUMN visible, ALGORITHM=COPY"
                )
                await cur.execute(
                    "ALTER TABLE jacobs_pipelines ADD COLUMN visible TINYINT(1) "
                    "DEFAULT 1, ALGORITHM=INSTANT"
                )
        try:
            with self.assertRaises(RuntimeError) as ctx:
                await store.init_tables()
            mensaje = str(ctx.exception)
            self.assertIn("no es una columna generada", mensaje)
        finally:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "ALTER TABLE jacobs_pipelines DROP COLUMN visible, ALGORITHM=COPY"
                    )
            await store.init_tables()

    async def test_normalizar_no_da_falsa_alarma_con_lo_que_devuelve_mariadb(self):
        """La forma REAL que guarda MariaDB (con backticks y minúsculas,
        distinta carácter por carácter del DDL fuente) no puede disparar el
        drift check -- si esto fallara, CADA arranque normal levantaría."""
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT GENERATION_EXPRESSION FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='jacobs_pipelines' "
                    "AND COLUMN_NAME='visible'"
                )
                (encontrada,) = await cur.fetchone()
        self.assertNotEqual(
            encontrada, store._EXPRESION_VISIBLE,
            "MariaDB devolvió la expresión BYTE A BYTE igual al DDL fuente -- "
            "este test dejó de probar la normalización (el punto es que son "
            "distintas en forma pero iguales en significado)."
        )
        self.assertEqual(
            store._normalizar_expresion_generada(encontrada),
            store._normalizar_expresion_generada(store._EXPRESION_VISIBLE),
        )
        await store.init_tables()  # no debe levantar


class Escenario1846DBTest(unittest.IsolatedAsyncioTestCase):
    """Ruling 19c (fix round 1, 2026-09-22): DROP INDEX
    idx_pipelines_descartados y volver a llamar init_tables() tiene que
    recrearlo sin reventar -- CREATE INDEX (no ADD COLUMN) sobre una tabla
    que ya tiene `visible` indexada. Medido contra MariaDB 12.3.3 real
    ANTES de escribir este test: las tres formas (recrear
    idx_pipelines_descartados, recrear idx_jacobs_pipelines_duenio, y
    recrear el propio idx_pipelines_visibles) pasan limpio -- la trampa del
    1845/1846 es específica de `ADD COLUMN` (que cambia el row format),
    no de `CREATE INDEX` (que no lo cambia). No hizo falta tocar
    `_crear_indice_acotado`: ya sube cualquier error que no sea 1205, y acá
    no aparece ninguno. Este test es la baranda de regresión, no la
    corrección de un bug -- si algún día una versión de MariaDB empieza a
    pedir LOCK=SHARED también para CREATE INDEX en esta situación, cae acá
    primero."""

    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)
        await store.init_tables()

    async def test_recrear_idx_pipelines_descartados_no_revienta(self):
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP INDEX idx_pipelines_descartados ON jacobs_pipelines")
        await store.init_tables()
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SHOW INDEX FROM jacobs_pipelines WHERE Key_name='idx_pipelines_descartados'"
                )
                self.assertEqual(len(await cur.fetchall()), 4)


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

    async def test_si_el_insert_del_evento_falla_la_transicion_se_deshace(self):
        """El INSERT del evento revienta con un error REAL de MariaDB, no un
        mock.

        **Lo que esta prueba SÍ demuestra** (fix round 3, corrección de la
        revisión): que un fallo en la escritura de auditoría propaga la
        excepción y DESHACE la transición completa -- la fila queda en
        `aborted` (no en `discarded`) y no queda ningún evento. Eso es
        cierto pase lo que pase con `conexion=conn`: si alguien lo sacara,
        el INSERT fallido en la conexión del pool también levantaría, la
        excepción también saldría de `async with transaccion(conn):`, y el
        UPDATE (que sigue sin commitear en `conn` en ESE momento) también se
        descartaría igual -- esta prueba NO distingue esos dos casos, así
        que NO prueba que el INSERT haya corrido en la MISMA conexión que
        el UPDATE. Esa propiedad (la que de verdad exige `conexion=conn`)
        la prueba `test_el_evento_no_es_visible_para_otra_conexion_antes_del_commit`,
        de abajo, con una espía y una lectura desde OTRA sesión mientras la
        transacción sigue abierta.

        El fallo: el `evento_payload` de este test lleva un `float("nan")`
        -- Python serializa `NaN`/`Infinity` por defecto
        (`json.dumps(..., allow_nan=True)` es el default), pero la
        gramática JSON estricta no los admite. La columna `payload JSON` de
        `jacobs_events` es, en MariaDB, un alias de `LONGTEXT` con un
        `CHECK (JSON_VALID(payload))` automático **desde 10.4.3** (no
        10.2.7 -- esa fue la versión que agregó el TIPO `JSON` como alias;
        el CHECK automático es dos años después) -- así que el INSERT choca
        con ese CHECK. Confirmado contra la base real antes de escribir
        esta versión: `(4025, "CONSTRAINT \\`jacobs_events.payload\\`
        failed for ...")`."""
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

    async def test_el_evento_no_es_visible_para_otra_conexion_antes_del_commit(self):
        """Fix round 3 (MAJOR, la revisión mostró por qué el test anterior
        no alcanzaba): la única forma de probar "el INSERT del evento corrió
        en la MISMA conexión/transacción que el UPDATE" es de comportamiento,
        no de que un error se propague -- eso último es cierto tanto si
        comparten conexión como si no.

        La prueba real de aislamiento transaccional: mientras la
        transacción de `pipeline_transicion_descarte` sigue ABIERTA (no
        confirmada todavía), una conexión DISTINTA no puede ver el evento
        recién insertado -- MVCC nunca deja ver filas de una transacción sin
        confirmar, sea cual sea el nivel de aislamiento. Eso sólo es
        observable si el INSERT corrió DENTRO de esa transacción abierta; si
        `event_append` escribiera por el pool en autocommit (la mutación de
        sacar `conexion=conn`, store.py:1722), el evento se confirmaría SOLO
        (fuera de la transacción del UPDATE) y sería visible de inmediato --
        la cuenta de abajo daría 1, no 0, y la aserción cae.

        Mecanismo: una espía que ENVUELVE (no reemplaza) el
        `store.event_append` real -- llama al original de verdad (así el
        INSERT ocurre) y, ANTES de devolver el control a
        `pipeline_transicion_descarte` (que todavía no llegó al `commit()`
        de `transaccion()`), abre una conexión NUEVA del pool y cuenta las
        filas de `jacobs_events` para este pipeline. Tiene que dar 0."""
        orig_event_append = store.event_append
        vistos_durante_la_transaccion: list[int] = []

        async def espia(*args, **kwargs):
            self.assertIsNotNone(
                kwargs.get("conexion"),
                "pipeline_transicion_descarte tiene que pasar conexion= a event_append")
            await orig_event_append(*args, **kwargs)
            # Todavía dentro de `async with transaccion(conn):` en el llamador:
            # el commit del UPDATE+evento no corrió todavía. Otra conexión NO
            # tiene que poder ver el evento recién insertado.
            async with store.conexion() as otra_conexion:
                async with otra_conexion.cursor() as cur:
                    await cur.execute(
                        "SELECT COUNT(*) FROM jacobs_events WHERE pipeline_id=%s", (self.pid,))
                    vistos_durante_la_transaccion.append((await cur.fetchone())[0])

        with mock.patch.object(store, "event_append", new=espia):
            ok = await store.pipeline_transicion_descarte(
                self.pid, 3, "discard",
                desde=PipelineStatus.aborted, a=PipelineStatus.discarded, user_id="u1",
                evento_tipo="PIPELINE_DISCARDED",
                evento_payload={"user_id": "u1", "desde": "aborted", "a": "discarded"},
            )
        self.assertTrue(ok)
        self.assertEqual(
            vistos_durante_la_transaccion, [0],
            "el evento ya era visible para OTRA conexión antes de que la "
            "transacción del CAS confirmara -- el INSERT no corrió en la "
            "misma conexión/transacción que el UPDATE",
        )
        # Después del commit (pipeline_transicion_descarte ya retornó):
        # ahora SÍ tiene que estar, y ser el único.
        eventos = await self._eventos()
        self.assertEqual(len(eventos), 1)

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


# Task 1-bis (2026-09-22, Ruling 18): el listado principal de jax-platform
# (SQL_PIPELINES_DEL_USUARIO) filtra "status NOT IN ('discarded','hidden')".
# Antes de idx_pipelines_visibles, ese filtro NO estaba en ningún índice de
# jacobs_pipelines con ese orden -- el plan medido en jax-platform (Ruling
# 17, docs/carga-sql-pipelines-del-usuario-indice-2026-09-22.md) tenía que
# recorrer el histórico completo del dueño para descartar filas antes de
# cortar en el LIMIT (forma extrema: 4,2-4,4 ms con 5000 descartados y 3
# vivas). Acá se prueba la propiedad DIRECTO contra MariaDB real: el
# EXPLAIN (la forma del plan) Y los contadores Handler_read (lo que el
# motor leyó DE VERDAD, no la estimación de `rows`) -- las dos formas
# sembradas por el controlador.
_LIMITE = 50

#: Ruling 19a (fix round 1, 2026-09-22): `AND owner_ack_at IS NOT NULL` ya NO
#: va acá -- la expresión de `visible` (jacobs/store.py::init_tables()) lo
#: incluye. Dejarlo acá TAMBIÉN sería redundante, no incorrecto, pero esta
#: consulta es el stand-in de lo que jax-platform va a terminar usando
#: (SQL_PIPELINES_DEL_USUARIO), y ESA versión futura tampoco lo va a
#: necesitar aparte.
_SQL_VISIBLES = (
    "SELECT pipeline_id, name, status, created_at, updated_at FROM jacobs_pipelines "
    "FORCE INDEX (idx_pipelines_visibles) "
    "WHERE user_id=%s AND tenant_id=%s AND visible=1 "
    "ORDER BY created_at DESC LIMIT %s OFFSET 0"
)


class CostoAcotadoPorVisibleDBTest(unittest.IsolatedAsyncioTestCase):
    """FORCE INDEX (idx_pipelines_visibles), no `IGNORE INDEX` de la lección
    de jax-platform (Ruling 17): acá el índice SIEMPRE existe -- lo crea
    `init_tables()` de este mismo repo en `asyncSetUp` -- así que nombrarlo
    directo no tiene el riesgo de acoplamiento de deploy que tenía
    `IGNORE INDEX` contra un índice de OTRO repo. Nombrarlo directo es
    además lo que hace que la mutación "sin el índice" falle FUERTE (ERROR
    1176 de MariaDB) en vez de caer en silencio a un plan peor."""

    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)
        await store.init_tables()
        self._tenants: list[str] = []
        self.addAsyncCleanup(self._borrar)

    async def _borrar(self):
        # BLOCK-1 (fix round 1, revisión del coordinador, 2026-09-22):
        # tests/test_delete_de_tablas_compartidas.py exige que un DELETE
        # sobre una tabla COMPARTIDA (jacobs_pipelines lo es) filtre por un
        # marcador PROPIO de este test -- `tenant_id` acá SIEMPRE lleva un
        # uuid generado por `_sembrar()` (nunca un literal fijo como
        # "tA-muchos-descartados"), así que el DELETE es seguro: no puede
        # tocar la fila de otra sesión ni de otro test. El pragma deja
        # explícito POR QUÉ para el detector estático (no reconoce
        # "tenant_id" como nombre de marcador por sí solo).
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                for tenant_id in self._tenants:
                    await cur.execute(
                        "DELETE FROM jacobs_pipelines WHERE tenant_id=%s", (tenant_id,))  # marcador-propio: tenant_id lleva un uuid propio de este test (ver _sembrar)
            await conn.commit()

    async def _sembrar(self, user_id: str, tenant_id: str, *,
                        n_no_visibles: int, n_visibles: int, status_no_visible: str,
                        n_sin_ack: int = 0) -> None:
        """`cur.executemany`, no un INSERT por fila -- miles de INSERT
        individuales tardan minutos (misma lección que
        docs/carga-sql-pipelines-del-usuario-indice-2026-09-22.md en
        jax-platform). `ANALYZE TABLE` al final: sin estadísticas frescas el
        plan puede depender de qué otros tests corrieron antes en la misma
        base de sesión.

        `n_sin_ack` (Ruling 19a, fix round 1): filas con `status` VISIBLE
        pero `owner_ack_at=NULL` -- hijos de Ada sin ack todavía. Tienen
        que quedar AFUERA del rango del índice igual que las
        descartadas/ocultas; si no, un dueño con muchos hijos sin ack
        vuelve a pagar el costo lineal que Ruling 18 quería evitar."""
        self._tenants.append(tenant_id)
        base = time.time() - 1_000_000
        filas = [
            (str(uuid.uuid4()), "x", "plataforma", "autonomous", status_no_visible,
             base + i, base + i, user_id, tenant_id, base)
            for i in range(n_no_visibles)
        ] + [
            (str(uuid.uuid4()), "x", "plataforma", "autonomous", "completed",
             base + n_no_visibles + i, base + n_no_visibles + i, user_id, tenant_id, base)
            for i in range(n_visibles)
        ] + [
            (str(uuid.uuid4()), "x", "plataforma", "autonomous", "running",
             base + n_no_visibles + n_visibles + i, base + n_no_visibles + n_visibles + i,
             user_id, tenant_id, None)
            for i in range(n_sin_ack)
        ]
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO jacobs_pipelines (pipeline_id,name,invoked_by,mode,status,"
                    "created_at,updated_at,user_id,tenant_id,owner_ack_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    filas,
                )
                await cur.execute("ANALYZE TABLE jacobs_pipelines")
            await conn.commit()

    async def _explain_y_handler_read(self, user_id: str, tenant_id: str) -> tuple[dict, dict, list]:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("EXPLAIN " + _SQL_VISIBLES, (user_id, tenant_id, _LIMITE))
                cols = [d[0] for d in cur.description]
                (explain_fila,) = await cur.fetchall()
                explain = dict(zip(cols, explain_fila))

                await cur.execute("FLUSH STATUS")
                await cur.execute(_SQL_VISIBLES, (user_id, tenant_id, _LIMITE))
                filas = await cur.fetchall()
                await cur.execute("SHOW SESSION STATUS LIKE 'Handler_read%'")
                handler = {k: int(v) for k, v in await cur.fetchall()}
        return explain, handler, filas

    @staticmethod
    def _tenant_unico(prefijo: str) -> str:
        """BLOCK-1 (fix round 1): un tenant por TEST, nunca un literal fijo
        -- la base de sesión la comparten otras corridas/tests, y un
        tenant_id repetido entre dos sesiones concurrentes sembraría (o
        borraría, ver `_borrar`) filas ajenas."""
        return f"{prefijo}-{uuid.uuid4().hex[:12]}"

    async def test_muchos_descartados_el_plan_usa_el_indice_sin_filesort(self):
        """Forma (b) del controlador: 5000 descartados + 2000 sin ack + 3
        vivas. Ésta es la forma que ANTES pagaba el costo lineal (medido en
        jax-platform: FORCE INDEX (idx_jacobs_pipelines_duenio) da 4,2-4,4
        ms recorriendo casi el tenant completo). Ruling 19a (fix round 1):
        se suman 2000 hijos de Ada sin ack -- mismo status VISIBLE que las
        3 vivas, pero sin owner_ack_at -- para confirmar que el plan sigue
        sin filesort con las tres categorías presentes."""
        tenant = self._tenant_unico("tA-muchos-descartados")
        await self._sembrar("userA", tenant,
                             n_no_visibles=5000, n_visibles=3, status_no_visible="discarded",
                             n_sin_ack=2000)
        explain, _handler, _filas = await self._explain_y_handler_read("userA", tenant)
        self.assertEqual(explain["key"], "idx_pipelines_visibles")
        self.assertNotEqual(explain["type"], "ALL")
        extra = (explain["Extra"] or "").lower()
        self.assertNotIn("filesort", extra)
        self.assertNotIn("temporary", extra)

    async def test_muchos_descartados_el_motor_no_lee_las_descartadas(self):
        """La propiedad real (no sólo la clave del plan): con 5000
        descartadas + 2000 sin ack (Ruling 19a) y 3 vivas, el total de
        Handler_read tiene que quedar del orden de las 3 vivas -- NO de
        las 7000 filas que `visible=1` deja afuera del rango del índice
        (5000 por status, 2000 por falta de ack)."""
        tenant = self._tenant_unico("tA-muchos-descartados")
        await self._sembrar("userA", tenant,
                             n_no_visibles=5000, n_visibles=3, status_no_visible="discarded",
                             n_sin_ack=2000)
        _explain, handler, filas = await self._explain_y_handler_read("userA", tenant)
        self.assertEqual(len(filas), 3)
        total = sum(handler.values())
        self.assertLessEqual(
            total, len(filas) + 10,
            f"{total} lecturas Handler_read para 3 filas vivas -- huele a que "
            f"el motor está tocando las 7000 no-visibles (descartadas o sin "
            f"ack): {handler}",
        )

    async def test_historial_largo_el_plan_usa_el_indice_sin_filesort(self):
        """Forma (a) del controlador: 5000 vivas + 50 descartadas + 2000
        sin ack -- un historial largo de filas VISIBLES, no de descartadas.
        El plan tiene que seguir siendo el mismo índice, sin filesort."""
        tenant = self._tenant_unico("tB-historial-largo")
        await self._sembrar("userB", tenant,
                             n_no_visibles=50, n_visibles=5000, status_no_visible="discarded",
                             n_sin_ack=2000)
        explain, _handler, _filas = await self._explain_y_handler_read("userB", tenant)
        self.assertEqual(explain["key"], "idx_pipelines_visibles")
        self.assertNotEqual(explain["type"], "ALL")
        extra = (explain["Extra"] or "").lower()
        self.assertNotIn("filesort", extra)
        self.assertNotIn("temporary", extra)

    async def test_historial_largo_el_motor_lee_aprox_el_limite_no_las_5000(self):
        """La propiedad central de Ruling 18/19a: con 5000 filas VISIBLES y
        2000 hijos de Ada sin ack, el costo del LIMIT no depende de cuántas
        haya en total -- el `EXPLAIN.rows` estimado para esta forma es
        ~5000 (la cardinalidad del rango completo de `visible=1` para el
        tenant), pero el motor CORTA apenas junta el LIMIT -- por eso el
        número que hay que mirar es Handler_read, no `rows` (el propio
        pedido del controlador)."""
        tenant = self._tenant_unico("tB-historial-largo")
        await self._sembrar("userB", tenant,
                             n_no_visibles=50, n_visibles=5000, status_no_visible="discarded",
                             n_sin_ack=2000)
        explain, handler, filas = await self._explain_y_handler_read("userB", tenant)
        self.assertEqual(len(filas), _LIMITE)
        # La propiedad que EXPLAIN.rows por sí solo NO prueba: la estimación
        # de cardinalidad del rango es del orden de las 5000 visibles del
        # tenant, muy por encima de lo que el motor termina leyendo de
        # verdad con el LIMIT.
        self.assertGreater(int(explain["rows"]), _LIMITE * 5)
        total = sum(handler.values())
        self.assertLessEqual(
            total, _LIMITE + 10,
            f"{total} lecturas Handler_read para LIMIT={_LIMITE} -- huele a "
            f"que el motor está leyendo de más de las 5000 vivas o de los "
            f"2000 sin ack: {handler}",
        )

    async def test_muchos_descartados_lee_bastante_menos_que_el_total_sembrado(self):
        """Cota independiente de la anterior, contra el TOTAL sembrado
        (7003 filas: 5000 descartadas + 2000 sin ack + 3 vivas) en vez de
        un número fijo -- por si el mecanismo de Handler_read cambiara de
        forma en una versión futura de MariaDB, esta cota sigue siendo
        significativa."""
        tenant = self._tenant_unico("tA-muchos-descartados")
        await self._sembrar("userA", tenant,
                             n_no_visibles=5000, n_visibles=3, status_no_visible="discarded",
                             n_sin_ack=2000)
        _explain, handler, _filas = await self._explain_y_handler_read("userA", tenant)
        total = sum(handler.values())
        self.assertLess(total, 7003 / 10)
