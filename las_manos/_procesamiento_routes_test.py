"""
LAS MANOS -- endpoint de Procesamiento de Archivos (2026-09-21, ronda 3 de
arreglo tras revisión adversarial -- ver
`.superpowers/sdd/2026-09-20-procesamiento-archivos-nucleo/endpoint-hallazgos.md`
y `endpoint-hallazgos-r2.md`).

Ronda 2 cerró bien B-1 (NUL), B-3 (reconciliación) y el test del loop
(B-5) -- esos no se tocan acá. Esta ronda ataca el defecto estructural que
esa ronda dejó sin nombrar: **el semáforo cuenta TRABAJOS y el pool cuenta
ARCHIVOS**. Cada grupo de tests de abajo corresponde a un punto del ruling:

  - N-1 (bloqueante): el permiso del semáforo se filtraba si `_STORE.
    create()` reventaba (ej. un `usuario` con un surrogate solitario).
  - N-2 (bloqueante): un nombre no codificable se rechaza ANTES del
    executor, y `write_result` no puede tumbar el lote entero por eso.
  - N-3: cancelar es honesto -- no mata hilos, sólo deja de programar
    trabajo nuevo y espera a que lo que ya corría termine solo.
  - N-4 / B-2: executor de E/S SEPARADO del de OCR; `RUNNING` se marca
    cuando el PRIMER archivo arranca de verdad, no al admitir.
  - N-5: la mutación que de verdad importa -- `run_in_executor(executor,
    …)` → `run_in_executor(None, …)` -- con un espía sobre el executor
    real, no con un test de retraso del loop (que no la distingue).
  - B-6: `usuario` no vacío, con tope de largo.
  - N-6: el arranque de LAS MANOS llama a `reconciliar_trabajos_huerfanos`.

Corre con:
  PYTHONPATH=.:las_manos python -m pytest -v las_manos/_procesamiento_routes_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import asyncio
import json
import secrets
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

import procesamiento_routes as rutas_mod
from auth_servicio import ENCABEZADO, IDENTIDAD_JACOBS, IDENTIDAD_PLATAFORMA, proteger
from motor_registry import tool_authority
from motor_registry.job_store import JobStore
from motor_registry.models import JobStatus
from processing_job_store import ProcessingJobStore
from processing_ownership import ProcessingOwnershipContext
from procesamiento.ficha import Ficha
from procesamiento_routes import router


#: E2a: LAS MANOS trabaja por `project_uuid` (UUID canónico, 36 caracteres).
UUID_PRUEBA = "0192f1d2-7c3a-7b4e-9a10-3f5e2d1c0b9a"
OWNER = ProcessingOwnershipContext("processing-owner.1", "1", "2", "3")


class _ProcessingJobStoreFixture(ProcessingJobStore):
    """Legacy worker fixtures have no HTTP envelope; keep their test data owned."""
    def create(self, **kwargs):
        return super().create(ownership=kwargs.pop("ownership", OWNER), **kwargs)


def _ficha(sha256: str, estado: str = "ok", extractor: str = "pdf") -> Ficha:
    return Ficha(
        sha256=sha256, origen="fuente/x", extractor=extractor,
        extractor_version="1", fecha="2026-09-21T00:00:00+00:00",
        estado=estado, detalle={},
    )


class _ExecutorEspia(ThreadPoolExecutor):
    """N-5: espía sobre un `ThreadPoolExecutor` REAL -- confirma que el
    trabajo llegó a ESTE objeto puntual, no a "algún executor que evitó el
    loop" (un test de retraso del loop no distingue el executor dedicado
    del executor por defecto: los dos evitan el loop igual)."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.llamado = False

    def submit(self, fn, /, *args, **kwargs):
        self.llamado = True
        return super().submit(fn, *args, **kwargs)


# ===========================================================================
#  Grupo 1 -- el worker, sin HTTP
# ===========================================================================
class TrabajoWorkerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmpdir.name) / "workspace"
        self.workspace.mkdir()
        self.store = _ProcessingJobStoreFixture(str(Path(self._tmpdir.name) / "jobs.jsonl"))
        self._parche_workspace = patch.object(
            tool_authority, "WORKSPACE_ROOT", self.workspace.resolve()
        )
        self._parche_workspace.start()
        self.addCleanup(self._parche_workspace.stop)
        self.addCleanup(self._tmpdir.cleanup)

        # Executores/semáforo PROPIOS de este test -- nunca los
        # módulo-globales de procesamiento_routes (aislamiento).
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-ocr")
        self.addCleanup(self.executor.shutdown)
        self.executor_io = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-io")
        self.addCleanup(self.executor_io.shutdown)
        self.semaforo = asyncio.Semaphore(2)

    def _archivo_en_workspace(self, nombre: str, contenido: bytes = b"x") -> str:
        (self.workspace / nombre).write_bytes(contenido)
        return nombre  # ruta RELATIVA -- contrato de resolve_jailed_path

    async def _ejecutar(self, job_id, proyecto, rutas, *, semaforo=None, executor=None, executor_io=None):
        """Mismo contrato que `crear_trabajo()`: adquiere el semáforo ANTES
        de llamar -- `_ejecutar_trabajo` lo libera en su `finally`."""
        semaforo = semaforo if semaforo is not None else self.semaforo
        executor = executor if executor is not None else self.executor
        executor_io = executor_io if executor_io is not None else self.executor_io
        await semaforo.acquire()
        await rutas_mod._ejecutar_trabajo(
            job_id, proyecto, rutas, store=self.store,
            executor=executor, executor_io=executor_io, semaforo=semaforo,
        )

    def _crear_job(self):
        return self.store.create(
            caller="x", capability="y", motor="z", trace_id="t", prompt="p",
            recursion_depth=0,
        )

    # -- caso feliz -----------------------------------------------------
    async def test_trabajo_completo_reporta_resultado_por_archivo_sin_contenido(self):
        self._archivo_en_workspace("doc.pdf")

        def _ingerir_falso(origen, trabajo, *, subruta=None):
            ficha = _ficha("a" * 64)
            carpeta = rutas_mod.ingesta.ruta_procesado(trabajo, ficha.sha256)
            carpeta.mkdir(parents=True)
            (carpeta / "extracto.txt").write_text("contenido secreto del cliente")
            return ficha

        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", side_effect=_ingerir_falso):
            await self._ejecutar(job_id, UUID_PRUEBA, ["doc.pdf"])

        job = self.store.get(job_id)
        assert job.status == JobStatus.COMPLETED, job
        resultados = json.loads(Path(job.result_path).read_text())
        assert len(resultados) == 1
        r = resultados[0]
        assert r["estado"] == "ok"
        assert r["extractor"] == "pdf"
        assert r["extracto_bytes"] > 0
        assert r["carpeta_procesado"] == f"proyectos/{UUID_PRUEBA}/procesado/{'a' * 64}"
        assert "contenido secreto" not in json.dumps(r)
        assert set(r) == {"archivo", "estado", "extractor", "extracto_bytes", "carpeta_procesado", "error"}

    async def test_M1_ruta_fuera_del_jail_se_rechaza_sin_llamar_a_ingerir(self):
        ingerir_mock = AsyncMock()
        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", ingerir_mock):
            await self._ejecutar(
                job_id, UUID_PRUEBA, ["/etc/passwd", "../fuera-del-workspace.txt"],
            )
        job = self.store.get(job_id)
        assert job.status == JobStatus.COMPLETED, job
        resultados = json.loads(Path(job.result_path).read_text())
        assert len(resultados) == 2
        for r in resultados:
            assert r["estado"] == "rechazado", r
            assert r["error"], "el rechazo tiene que traer la razón del jail"
        ingerir_mock.assert_not_called()

    async def test_B1_ruta_con_byte_nul_entre_sanas_no_tumba_el_lote(self):
        """Un byte NUL (`\\x00`) es codificable a UTF-8 sin problema (es
        `resolve()`, no la codificación, lo que se atraganta con él -- ver
        _tool_authority_test.py::test_3b) -- este caso es DISTINTO de N-2
        (surrogates solitarios), y sigue sin tumbar el lote."""
        self._archivo_en_workspace("sano1.pdf")
        self._archivo_en_workspace("sano2.pdf")

        def _ingerir_falso(origen, trabajo, *, subruta=None):
            return _ficha("b" * 64)

        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", side_effect=_ingerir_falso):
            await self._ejecutar(
                job_id, UUID_PRUEBA, ["sano1.pdf", "archivo\x00malo.pdf", "sano2.pdf"],
            )

        job = self.store.get(job_id)
        assert job.status == JobStatus.COMPLETED, f"un byte NUL tumbó el lote entero: {job}"
        resultados = json.loads(Path(job.result_path).read_text())
        assert len(resultados) == 3
        por_archivo = {r["archivo"]: r for r in resultados}
        assert por_archivo["sano1.pdf"]["estado"] == "ok"
        assert por_archivo["sano2.pdf"]["estado"] == "ok"
        assert por_archivo["archivo\x00malo.pdf"]["estado"] == "rechazado"

    async def test_M2_un_archivo_roto_no_tumba_el_lote(self):
        for nombre in ("uno.pdf", "dos.pdf", "tres.pdf"):
            self._archivo_en_workspace(nombre)

        respuestas_por_nombre = {
            "uno.pdf": _ficha("1" * 64),
            "dos.pdf": RuntimeError("pdftoppm: timeout"),
            "tres.pdf": _ficha("3" * 64),
        }

        def _ingerir_falso(origen, trabajo, *, subruta=None):
            resultado = respuestas_por_nombre[Path(origen).name]
            if isinstance(resultado, Exception):
                raise resultado
            return resultado

        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", side_effect=_ingerir_falso):
            await self._ejecutar(job_id, UUID_PRUEBA, ["uno.pdf", "dos.pdf", "tres.pdf"])

        job = self.store.get(job_id)
        assert job.status == JobStatus.COMPLETED, f"un archivo roto tumbó el lote entero: {job}"
        resultados = json.loads(Path(job.result_path).read_text())
        assert len(resultados) == 3
        por_archivo = {r["archivo"]: r for r in resultados}
        assert por_archivo["uno.pdf"]["estado"] == "ok"
        assert por_archivo["dos.pdf"]["estado"] == "error"
        assert "pdftoppm: timeout" in por_archivo["dos.pdf"]["error"]
        assert por_archivo["tres.pdf"]["estado"] == "ok"

    # -- N-2: nombre no codificable -----------------------------------------
    async def test_N2_ruta_no_codificable_se_rechaza_sin_tumbar_el_lote(self):
        """`\\udcff` es un surrogate solitario -- típico de un nombre de
        archivo en cp1252/latin-1 que Python re-expone así al leerlo del
        filesystem (`surrogateescape`). Antes: `resolve_jailed_path` lo
        aceptaba bien (es un path POSIX válido, sigue symlinks vía
        surrogateescape sin problema), `_procesar_una_ruta` lo reportaba
        'rechazado' bien -- pero ESE string crudo viajaba hasta
        `write_result`, que hace `Path.write_text(..., encoding='utf-8')`
        ESTRICTO: revienta, y el job entero (incluidos los dos archivos
        sanos) terminaba 'failed', con los resultados de los sanos
        PERDIDOS."""
        self._archivo_en_workspace("sano1.pdf")
        self._archivo_en_workspace("sano2.pdf")
        ruta_mala = "malo\udcff.pdf"

        def _ingerir_falso(origen, trabajo, *, subruta=None):
            return _ficha("c" * 64)

        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", side_effect=_ingerir_falso) as ingerir_mock:
            await self._ejecutar(job_id, UUID_PRUEBA, ["sano1.pdf", ruta_mala, "sano2.pdf"])

        job = self.store.get(job_id)
        assert job.status == JobStatus.COMPLETED, (
            f"un nombre no codificable tumbó el lote entero (los sanos se perdieron): {job}"
        )
        resultados = json.loads(Path(job.result_path).read_text())
        assert len(resultados) == 3
        por_archivo = {r["archivo"]: r for r in resultados}
        assert por_archivo["sano1.pdf"]["estado"] == "ok"
        assert por_archivo["sano2.pdf"]["estado"] == "ok"
        # el nombre malo NO aparece literal (no se puede codificar) -- pero
        # SÍ hay una entrada rechazada por él, y viene saneada.
        assert ruta_mala not in resultados[0] and ruta_mala not in json.dumps(resultados)
        rechazados = [r for r in resultados if r["estado"] == "rechazado"]
        assert len(rechazados) == 1
        assert "no se puede codificar" in rechazados[0]["error"]

        # nunca se le pagó un hilo real a un nombre que ni se puede reportar
        llamados = {c.args[1] for c in ingerir_mock.call_args_list}
        assert ruta_mala not in llamados, "se llegó a llamar ingerir() con la ruta no codificable"

    async def test_N2_guardar_resultado_reintenta_con_ensure_ascii_si_algo_se_cuela(self):
        """Defensa de última línea de `_guardar_resultado`: si por CUALQUIER
        otro motivo el JSON no es codificable a UTF-8 estricto, reintenta
        con `ensure_ascii=True` en vez de perder el resultado entero."""
        resultados = [{"archivo": "x\udcff", "estado": "ok"}]  # simula algo que se coló
        with tempfile.TemporaryDirectory() as d:
            store = JobStore(str(Path(d) / "jobs.jsonl"))
            job_id = store.create(caller="x", capability="y", motor="z", trace_id="t", prompt="p", recursion_depth=0)
            ruta = rutas_mod._guardar_resultado(store, job_id, resultados)
            contenido = Path(ruta).read_text(encoding="utf-8")  # no debe lanzar
            assert "\\udcff" in contenido  # escapado como \uXXXX (ASCII puro)

    def test_N2_resultado_no_codificable_sanea_el_nombre(self):
        """Unitario y directo: el `archivo` que `_resultado_no_codificable`
        produce tiene que ser ASCII puro -- sin esto, el fallback de
        `_guardar_resultado` (ensure_ascii=True) terminaría rescatando el
        lote de todos modos, y un test end-to-end no notaría que ESTA
        sanitización puntual desapareció."""
        r = rutas_mod._resultado_no_codificable("malo\udcff.pdf")
        r.archivo.encode("ascii")  # no debe lanzar -- si lanza, no es ASCII puro
        assert "\udcff" not in r.archivo

    def test_MINORC_leer_resultado_sanea_un_error_con_surrogate(self):
        """MINOR-C (ronda 4): N-2 defiende la ESCRITURA del campo
        `archivo` -- pero un `error` (mensaje de excepción, no derivado
        de `_resultado_no_codificable`) puede traer un surrogate
        solitario por otro camino, y `_guardar_resultado` lo rescata al
        ESCRIBIR (`ensure_ascii=True`). El problema es que `json.loads`
        DESESCAPA ese surrogate de vuelta al leer -- si nadie sanea del
        lado de LECTURA, la respuesta HTTP (bug real de Starlette, ver
        N-1) daría 500 en CADA `GET` futuro sobre ese job, dejando los
        archivos SANOS del mismo lote inaccesibles para siempre."""
        resultados_crudos = [
            {
                "archivo": "sano.pdf", "estado": "error",
                "error": "no se pudo leer 'archivo\udcff.tmp'",
                "extractor": None, "extracto_bytes": 0, "carpeta_procesado": None,
            },
        ]
        with tempfile.TemporaryDirectory() as d:
            result_path = Path(d) / "resultado.json"
            # Mismo camino que produce `_guardar_resultado`: el fallback
            # `ensure_ascii=True` es justo lo que permite que ESTO llegue
            # a existir en disco sin reventar la escritura.
            result_path.write_text(
                json.dumps(resultados_crudos, ensure_ascii=True), encoding="utf-8",
            )
            leidos = rutas_mod._leer_resultados_de_disco(str(result_path))

        leidos[0]["error"].encode("ascii")  # no debe lanzar -- si lanza, no es ASCII puro
        assert "\udcff" not in leidos[0]["error"]
        # y se puede construir la respuesta sin reventar
        respuesta = rutas_mod.ResultadoArchivo(**leidos[0])
        assert respuesta.error

    # -- N-4: RUNNING no se adelanta a que un hilo REAL arranque -----------
    async def test_N4_running_no_se_marca_mientras_el_archivo_sigue_en_cola(self):
        """Pool de UN hilo, ocupado con otra cosa (bloqueado a propósito):
        el archivo del trabajo bajo prueba queda EN COLA, sin arrancar
        todavía -- el job tiene que seguir en `pending`, nunca `running`,
        mientras eso dure."""
        executor_1_hilo = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-running-tardio")
        self.addCleanup(executor_1_hilo.shutdown)
        loop = asyncio.get_running_loop()
        bloqueo = threading.Event()
        # addCleanup es LIFO: registrado DESPUÉS del shutdown de arriba,
        # corre ANTES -- si un assert de este test falla antes de la
        # línea `bloqueo.set()` de más abajo, el hilo quedaría esperando
        # PARA SIEMPRE y `executor.shutdown(wait=True)` colgaría el
        # proceso entero (encontrado armando esta misma mutación: un
        # `assert` que falla ANTES de liberar el hilo cuelga el test
        # runner completo, no reporta un fallo limpio).
        self.addCleanup(bloqueo.set)
        ocupa = loop.run_in_executor(executor_1_hilo, bloqueo.wait)

        self._archivo_en_workspace("a.pdf")
        semaforo = asyncio.Semaphore(1)
        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", return_value=_ficha("6" * 64)):
            await semaforo.acquire()
            tarea = asyncio.create_task(
                rutas_mod._ejecutar_trabajo(
                    job_id, UUID_PRUEBA, ["a.pdf"], store=self.store,
                    executor=executor_1_hilo, executor_io=self.executor_io, semaforo=semaforo,
                )
            )
            await asyncio.sleep(0.05)  # "a.pdf" quedó EN COLA -- el único hilo está ocupado
            assert self.store.get(job_id).status == JobStatus.PENDING, (
                f"RUNNING se marcó sin que ningún hilo real hubiera arrancado: {self.store.get(job_id)}"
            )
            bloqueo.set()
            await tarea
            await ocupa

        assert self.store.get(job_id).status == JobStatus.COMPLETED

    async def test_marca_running_antes_de_completed(self):
        vistos: list[str] = []
        original_update = self.store.update

        def _update_espiado(job_id, **kwargs):
            if "status" in kwargs:
                vistos.append(kwargs["status"])
            return original_update(job_id, **kwargs)

        job_id = self._crear_job()
        with patch.object(self.store, "update", side_effect=_update_espiado), \
             patch.object(rutas_mod.ingesta, "ingerir", return_value=_ficha("f" * 64)):
            self._archivo_en_workspace("a.pdf")
            await self._ejecutar(job_id, UUID_PRUEBA, ["a.pdf"])

        assert vistos == [JobStatus.RUNNING.value, JobStatus.COMPLETED.value], vistos

    async def test_ruta_ausente_dentro_del_jail_se_rechaza_no_revienta(self):
        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir") as ingerir_mock:
            await self._ejecutar(job_id, UUID_PRUEBA, ["no-existe.pdf"])
        job = self.store.get(job_id)
        assert job.status == JobStatus.COMPLETED
        resultados = json.loads(Path(job.result_path).read_text())
        assert resultados[0]["estado"] == "rechazado"
        ingerir_mock.assert_not_called()

    # -- B-5: la mutación que más importa -- medir el LOOP, no el POST ---
    async def test_B5_el_trabajo_no_bloquea_el_loop_de_eventos(self):
        self._archivo_en_workspace("lento.pdf")

        def _ingerir_lento(origen, trabajo, *, subruta=None):
            time.sleep(0.4)  # bloqueante DE VERDAD -- simula OCR real
            return _ficha("z" * 64)

        job_id = self._crear_job()
        retrasos: list[float] = []

        async def _sondear_loop():
            for _ in range(25):
                t0 = time.perf_counter()
                await asyncio.sleep(0.01)
                retrasos.append(time.perf_counter() - t0 - 0.01)

        with patch.object(rutas_mod.ingesta, "ingerir", side_effect=_ingerir_lento):
            # El sondeo se crea PRIMERO y se le da tiempo real de arrancar
            # y quedar DENTRO de su `sleep(0.01)` antes de lanzar el
            # trabajo -- si no, un trabajo que bloquea antes de su primer
            # `await` puede terminar ANTES de que el sondeo arranque, y el
            # test pasa igual con una mutación que saca el `run_in_executor`
            # (falso negativo, confirmado a mano en la ronda anterior).
            sondeo = asyncio.create_task(_sondear_loop())
            await asyncio.sleep(0.02)
            await self._ejecutar(job_id, UUID_PRUEBA, ["lento.pdf"])
            await sondeo

        peor = max(retrasos)
        assert peor < 0.15, (
            f"el loop de eventos se retrasó {peor:.3f}s durante el trabajo -- "
            "el OCR no está corriendo fuera del loop"
        )
        assert self.store.get(job_id).status == JobStatus.COMPLETED

    async def test_B2_archivos_del_mismo_trabajo_se_reparten_entre_hilos(self):
        self._archivo_en_workspace("p1.pdf")
        self._archivo_en_workspace("p2.pdf")

        def _ingerir_lento(origen, trabajo, *, subruta=None):
            time.sleep(0.3)
            return _ficha("e" * 64)

        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", side_effect=_ingerir_lento):
            t0 = time.perf_counter()
            await self._ejecutar(job_id, UUID_PRUEBA, ["p1.pdf", "p2.pdf"])
            dt = time.perf_counter() - t0

        assert dt < 0.5, f"tardó {dt:.2f}s -- no se repartió entre los dos hilos disponibles"

    # -- outer except: nunca deja running huérfano -----------------------
    async def test_una_excepcion_catastrofica_marca_failed_no_deja_running(self):
        self._archivo_en_workspace("a.pdf")
        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", return_value=_ficha("9" * 64)), \
             patch.object(self.store, "write_result", side_effect=RuntimeError("disco lleno")):
            await self._ejecutar(job_id, UUID_PRUEBA, ["a.pdf"])

        job = self.store.get(job_id)
        assert job.status == JobStatus.FAILED, job
        assert "disco lleno" in job.error

    # -- el semáforo se libera SIEMPRE ------------------------------------
    async def test_semaforo_se_libera_aunque_el_trabajo_falle(self):
        semaforo = asyncio.Semaphore(1)
        self._archivo_en_workspace("a.pdf")
        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", return_value=_ficha("1" * 64)), \
             patch.object(self.store, "write_result", side_effect=RuntimeError("boom")):
            await self._ejecutar(job_id, UUID_PRUEBA, ["a.pdf"], semaforo=semaforo)

        assert not semaforo.locked(), "el semáforo quedó tomado tras un trabajo que falló"

    async def test_semaforo_se_libera_al_completar(self):
        semaforo = asyncio.Semaphore(1)
        self._archivo_en_workspace("a.pdf")
        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", return_value=_ficha("2" * 64)):
            await self._ejecutar(job_id, UUID_PRUEBA, ["a.pdf"], semaforo=semaforo)

        assert not semaforo.locked()

    # -- B-6: el principal se registra ------------------------------------
    async def test_B6_owner_user_is_the_immutable_job_caller(self):
        job_id = self.store.create(
            caller="ana@cliente.com", capability="ingesta_archivos",
            motor="n/a", trace_id="t", prompt="n/a", recursion_depth=0,
        )
        with patch.object(rutas_mod.ingesta, "ingerir", return_value=_ficha("3" * 64)):
            self._archivo_en_workspace("a.pdf")
            await self._ejecutar(job_id, UUID_PRUEBA, ["a.pdf"])
        assert self.store.get(job_id).caller == "user:2"

    # -- B-3: reconciliación al arrancar (ronda anterior, sin cambios) -----
    def test_B3_reconciliar_marca_failed_lo_que_quedo_pending_running_o_cancelling(self):
        j_pending = self._crear_job()
        j_running = self._crear_job()
        self.store.update(j_running, status=JobStatus.RUNNING.value)
        j_cancelling = self._crear_job()
        self.store.update(j_cancelling, status=JobStatus.CANCELLING.value)
        j_completado = self._crear_job()
        self.store.update(j_completado, status=JobStatus.COMPLETED.value, finished_at=time.time())

        n = rutas_mod.reconciliar_trabajos_huerfanos(self.store)

        assert n == 3, n
        for j in (j_pending, j_running, j_cancelling):
            assert self.store.get(j).status == JobStatus.FAILED
            assert self.store.get(j).error
        assert self.store.get(j_completado).status == JobStatus.COMPLETED

    def test_B3_reconciliar_no_hace_nada_sin_huerfanos(self):
        j = self._crear_job()
        self.store.update(j, status=JobStatus.COMPLETED.value, finished_at=time.time())
        assert rutas_mod.reconciliar_trabajos_huerfanos(self.store) == 0
        assert self.store.get(j).status == JobStatus.COMPLETED

    # -- B-4/N-4: GET no bloquea el loop leyendo el resultado ---------------
    async def test_B4_leer_resultado_no_bloquea_el_loop(self):
        def _lectura_lenta(result_path):
            time.sleep(0.3)
            return [{
                "archivo": "x", "estado": "ok", "extractor": "pdf",
                "extracto_bytes": 1, "carpeta_procesado": "c", "error": None,
            }]

        class _VistaFalsa:
            status = JobStatus.COMPLETED
            error = None
            result_path = "irrelevante-para-el-mock"

        retrasos: list[float] = []

        async def _sondear_loop():
            for _ in range(20):
                t0 = time.perf_counter()
                await asyncio.sleep(0.01)
                retrasos.append(time.perf_counter() - t0 - 0.01)

        with patch.object(rutas_mod, "_leer_resultados_de_disco", side_effect=_lectura_lenta):
            sondeo = asyncio.create_task(_sondear_loop())
            await asyncio.sleep(0.02)
            await rutas_mod._construir_respuesta_estado("job-x", _VistaFalsa())
            await sondeo

        peor = max(retrasos)
        assert peor < 0.15, (
            f"el loop se retrasó {peor:.3f}s leyendo el resultado -- "
            "la lectura no está corriendo fuera del loop"
        )

    async def test_N4_get_no_espera_detras_del_pool_de_ocr_saturado(self):
        """Medido en la revisión con tesseract real: consultar un trabajo
        YA TERMINADO tardó 37s porque la lectura compartía pool con el
        OCR. Acá se satura el pool de OCR REAL del módulo
        (`_EXECUTOR_OCR`) con trabajo lento de verdad, y se confirma que
        leer un resultado (que usa `_EXECUTOR_IO`, un pool DISTINTO)
        sigue respondiendo rápido."""
        loop = asyncio.get_running_loop()
        ocupando = [
            loop.run_in_executor(rutas_mod._EXECUTOR_OCR, time.sleep, 0.6)
            for _ in range(rutas_mod._MAX_WORKERS)
        ]
        await asyncio.sleep(0.05)  # deja que los hilos del pool de OCR arranquen de verdad

        job_id = self._crear_job()
        resultados = [{
            "archivo": "a.pdf", "estado": "ok", "extractor": "pdf",
            "extracto_bytes": 1, "carpeta_procesado": "c", "error": None,
        }]
        result_path = self.store.write_result(job_id, json.dumps(resultados))
        self.store.update(
            job_id, status=JobStatus.COMPLETED.value, finished_at=time.time(),
            result_path=result_path,
        )
        view = self.store.get(job_id)

        t0 = time.perf_counter()
        respuesta = await rutas_mod._construir_respuesta_estado(job_id, view)
        dt = time.perf_counter() - t0

        assert dt < 0.3, (
            f"el GET tardó {dt:.2f}s -- esperó detrás del pool de OCR saturado "
            "(¿la lectura volvió a compartir executor con el OCR?)"
        )
        assert respuesta.resultados[0].estado == "ok"
        await asyncio.gather(*ocupando)

    # -- N-5: la mutación que más importa ------------------------------------
    async def test_N5_procesar_usa_el_executor_ocr_dedicado(self):
        espia = _ExecutorEspia(max_workers=2, thread_name_prefix="espia-ocr")
        self.addCleanup(espia.shutdown)
        self._archivo_en_workspace("a.pdf")
        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", return_value=_ficha("7" * 64)):
            await self._ejecutar(job_id, UUID_PRUEBA, ["a.pdf"], executor=espia)
        assert espia.llamado, (
            "el trabajo no pasó por el executor de OCR dedicado -- "
            "¿se cambió run_in_executor(executor, …) por run_in_executor(None, …)?"
        )

    async def test_N5_guardar_resultado_usa_el_executor_io_dedicado(self):
        espia = _ExecutorEspia(max_workers=2, thread_name_prefix="espia-io")
        self.addCleanup(espia.shutdown)
        self._archivo_en_workspace("a.pdf")
        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", return_value=_ficha("8" * 64)):
            await self._ejecutar(job_id, UUID_PRUEBA, ["a.pdf"], executor_io=espia)
        assert espia.llamado, (
            "escribir el resultado no pasó por el executor de E/S dedicado"
        )

    async def test_N5_leer_resultado_usa_el_executor_io_dedicado(self):
        espia = _ExecutorEspia(max_workers=2, thread_name_prefix="espia-io-lectura")
        self.addCleanup(espia.shutdown)

        class _VistaFalsa:
            status = JobStatus.COMPLETED
            error = None
            result_path = "irrelevante"

        with patch.object(rutas_mod, "_EXECUTOR_IO", espia), \
             patch.object(rutas_mod, "_leer_resultados_de_disco", return_value=[]):
            await rutas_mod._construir_respuesta_estado("job-y", _VistaFalsa())

        assert espia.llamado, "leer el resultado no pasó por _EXECUTOR_IO"

    # -- MINOR-D: la carrera entre "decidir" y "marcar RUNNING" -------------
    def test_MINORD_cancelar_espera_a_que_termine_de_marcar_running(self):
        """MINOR-D (ronda 4): antes, `arrancar_o_saltar()` (bajo un lock)
        y `_marcar_running_una_vez()` (bajo OTRO, un `threading.Event`)
        eran DOS operaciones separadas -- un `cancelar()` que llegaba
        justo ENTRE medio dejaba escrito `running` DESPUÉS de
        `cancelling`: el estado iba PARA ATRÁS (visto en rojo 3 de 3
        corridas). Este test no depende de la suerte del GIL para
        reproducirlo: bloquea DE VERDAD, con un `threading.Event`, DENTRO
        de la escritura de `running` (que ahora corre bajo el MISMO lock
        que `cancelar()`) y confirma que `cancelar()` -- llamado desde
        OTRO hilo -- queda ESPERANDO ese lock, no se cuela antes."""
        orden: list[str] = []
        adentro = threading.Event()
        seguir = threading.Event()

        class _StoreLento:
            def update(self, job_id, **kwargs):
                estado = kwargs.get("status")
                if estado == JobStatus.RUNNING.value:
                    orden.append("running:entrando")
                    adentro.set()
                    assert seguir.wait(timeout=2), "seguir nunca se marcó -- deadlock"
                    orden.append("running:saliendo")
                else:
                    orden.append(estado)

        control = rutas_mod._ControlTrabajo(_StoreLento(), "job-x")

        hilo_arranca = threading.Thread(target=control.arrancar_o_marcar_running)
        hilo_arranca.start()
        self.addCleanup(lambda: (seguir.set(), hilo_arranca.join(timeout=2)))
        assert adentro.wait(timeout=2), "arrancar_o_marcar_running nunca llegó a marcar running"

        hilo_cancela = threading.Thread(target=control.cancelar)
        hilo_cancela.start()
        time.sleep(0.1)  # tiempo de sobra: si pudiera colarse, ya habría terminado
        assert "cancelling" not in orden, (
            f"cancelar() terminó mientras arrancar_o_marcar_running seguía "
            f"DENTRO de su sección crítica -- no comparten el lock de verdad: {orden}"
        )

        seguir.set()
        hilo_arranca.join(timeout=2)
        hilo_cancela.join(timeout=2)

        assert orden == ["running:entrando", "running:saliendo", "cancelling"], orden

    # -- N-3: cancelación honesta, con el executor REAL --------------------
    async def test_N3_cancelar_deja_terminar_lo_que_ya_arranco_y_corta_lo_que_esperaba(self):
        """Pool de UN hilo, dos archivos: el primero arranca YA (ocupa el
        único hilo real), el segundo queda en cola. Se cancela mientras el
        primero sigue corriendo de VERDAD (`time.sleep` real, no un
        `asyncio.sleep` que se pueda saltear) -- el que ya arrancó TERMINA
        SOLO (Python no mata hilos), el que esperaba nunca arranca. Estado
        final CANCELLED, nunca COMPLETED. El semáforo NO se libera hasta
        que el hilo en vuelo termina de verdad."""
        executor_1_hilo = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-1-hilo")
        self.addCleanup(executor_1_hilo.shutdown)
        semaforo = asyncio.Semaphore(1)

        self._archivo_en_workspace("primero.pdf")
        self._archivo_en_workspace("segundo.pdf")

        terminados: list[str] = []

        def _ingerir_lento(origen, trabajo, *, subruta=None):
            time.sleep(0.3)  # bloqueante DE VERDAD, en el hilo real
            terminados.append(Path(origen).name)
            return _ficha("d" * 64)

        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", side_effect=_ingerir_lento):
            await semaforo.acquire()
            tarea = asyncio.create_task(
                rutas_mod._ejecutar_trabajo(
                    job_id, UUID_PRUEBA, ["primero.pdf", "segundo.pdf"], store=self.store,
                    executor=executor_1_hilo, executor_io=self.executor_io, semaforo=semaforo,
                )
            )
            await asyncio.sleep(0.08)  # deja que "primero" arranque de VERDAD en el hilo
            assert self.store.get(job_id).status == JobStatus.RUNNING
            assert terminados == [], "no debería haber terminado todavía"

            # Mismo mecanismo que el endpoint de cancelación, sin pasar por
            # HTTP: `cancelar()` es quien escribe CANCELLING (N1, ronda 5).
            control = rutas_mod._CONTROLES[job_id]
            control.cancelar()

            # El semáforo TODAVÍA no se liberó -- "primero" sigue corriendo.
            assert semaforo.locked(), "el semáforo se liberó ANTES de que terminara el hilo en vuelo"
            assert self.store.get(job_id).status == JobStatus.CANCELLING

            await tarea  # espera a que "primero" termine DE VERDAD

        assert terminados == ["primero.pdf"], (
            f"'segundo.pdf' no debería haber arrancado nunca: {terminados}"
        )
        job = self.store.get(job_id)
        assert job.status == JobStatus.CANCELLED
        resultados = json.loads(Path(job.result_path).read_text())
        por_archivo = {r["archivo"]: r for r in resultados}
        assert por_archivo["primero.pdf"]["estado"] == "ok"
        assert por_archivo["segundo.pdf"]["estado"] == "cancelado"
        assert not semaforo.locked(), "el semáforo sigue tomado después de terminar"

    async def test_N3_cancelar_un_trabajo_sin_ningun_hilo_arrancado_aun(self):
        """Caso borde: se cancela ANTES de que el primer archivo llegue a
        arrancar en el hilo (pool ocupado por completo con otra cosa) --
        el archivo nunca corre, el job pasa directo a CANCELLED."""
        executor_1_hilo = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-1-hilo-b")
        self.addCleanup(executor_1_hilo.shutdown)
        loop = asyncio.get_running_loop()
        bloqueo = threading.Event()
        self.addCleanup(bloqueo.set)  # mismo motivo que en el otro test con `bloqueo` -- ver ahí
        ocupa = loop.run_in_executor(executor_1_hilo, bloqueo.wait)  # ocupa el único hilo hasta que se libere a mano

        semaforo = asyncio.Semaphore(1)
        self._archivo_en_workspace("nunca.pdf")
        ingerir_mock = AsyncMock()
        job_id = self._crear_job()

        with patch.object(rutas_mod.ingesta, "ingerir", ingerir_mock):
            await semaforo.acquire()
            tarea = asyncio.create_task(
                rutas_mod._ejecutar_trabajo(
                    job_id, UUID_PRUEBA, ["nunca.pdf"], store=self.store,
                    executor=executor_1_hilo, executor_io=self.executor_io, semaforo=semaforo,
                )
            )
            await asyncio.sleep(0.05)  # el futuro de "nunca.pdf" quedó EN COLA, no arrancó
            control = rutas_mod._CONTROLES[job_id]
            control.cancelar()
            bloqueo.set()  # libera el hilo que lo tenía ocupado
            await tarea
            await ocupa

        ingerir_mock.assert_not_called()
        job = self.store.get(job_id)
        assert job.status == JobStatus.CANCELLED
        resultados = json.loads(Path(job.result_path).read_text())
        assert resultados[0]["estado"] == "cancelado"

    # -- Ronda 5 --------------------------------------------------------------
    def _historial_de_estados(self, job_id: str, *, crudo: bool = False) -> list[str]:
        """El historial REAL del job, leído del JSONL append-only (cada
        línea es el estado completo tras un `create`/`update`), con los
        estados repetidos consecutivos colapsados -- `update()` sin
        `status` (ej. `result_path`) re-escribe el mismo estado."""
        estados: list[str] = []
        for linea in self.store._path.read_text(encoding="utf-8").splitlines():
            evento = json.loads(linea)
            if evento.get("job_id") != job_id:
                continue
            if crudo or not estados or estados[-1] != evento["status"]:
                estados.append(evento["status"])
        return estados

    async def test_N1r5_cancelar_por_la_ruta_real_no_hace_ir_el_historial_para_atras(self):
        """N1 (ronda 5): `cancelar_trabajo()` escribía `CANCELLING` FUERA del
        lock de `_ControlTrabajo` y recién después llamaba a
        `control.cancelar()`. Si un hilo estaba DENTRO de
        `arrancar_o_marcar_running()` escribiendo `running` en ese momento,
        el historial quedaba `cancelling, running, cancelling` -- para
        atrás. Y además el loop de eventos quedaba congelado esperando un
        `threading.Lock` que el hilo retenía durante una escritura a disco.

        Pasa por `cancelar_trabajo()` REAL (no por `_ControlTrabajo` suelto,
        que es por qué el test de MINOR-D no lo vio): la escritura de
        `running` se bloquea DE VERDAD (`threading.Event`) con el lock
        tomado, se cancela por la ruta, y se exige (a) historial
        monótono en el JSONL y (b) que el loop siga respondiendo mientras
        la cancelación espera ese lock."""
        executor_1_hilo = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-n1r5")
        self.addCleanup(executor_1_hilo.shutdown)
        adentro = threading.Event()
        seguir = threading.Event()
        self.addCleanup(seguir.set)  # LIFO: corre antes del shutdown -- ver test_N4_running_no_se_marca...
        original_update = self.store.update

        def _update_que_retiene_running(job_id, **kwargs):
            if kwargs.get("status") == JobStatus.RUNNING.value:
                adentro.set()
                assert seguir.wait(timeout=5), "seguir nunca se marcó"
            return original_update(job_id, **kwargs)

        self._archivo_en_workspace("a.pdf")
        semaforo = asyncio.Semaphore(1)
        job_id = self._crear_job()

        retrasos: list[float] = []
        sondeando = True

        async def _sondeo():
            while sondeando:
                t0 = time.perf_counter()
                await asyncio.sleep(0.01)
                retrasos.append(time.perf_counter() - t0 - 0.01)

        with patch.object(self.store, "update", side_effect=_update_que_retiene_running), \
             patch.object(rutas_mod, "_STORE", self.store), \
             patch.object(rutas_mod, "_EXECUTOR_IO", self.executor_io), \
             patch.object(rutas_mod.ingesta, "ingerir", return_value=_ficha("7" * 64)):
            await semaforo.acquire()
            tarea = asyncio.create_task(
                rutas_mod._ejecutar_trabajo(
                    job_id, UUID_PRUEBA, ["a.pdf"], store=self.store,
                    executor=executor_1_hilo, executor_io=self.executor_io, semaforo=semaforo,
                )
            )
            assert await asyncio.to_thread(adentro.wait, 2), (
                "el hilo nunca llegó a escribir running"
            )

            sondeo = asyncio.create_task(_sondeo())
            await asyncio.sleep(0.02)  # el sondeo ya está DENTRO de su sleep (lección de B-5)
            liberador = threading.Timer(0.3, seguir.set)  # suelta el lock en 0,3 s, desde otro hilo
            liberador.start()
            self.addCleanup(liberador.cancel)
            respuesta = await rutas_mod.cancelar_trabajo(job_id, _owned_request())
            sondeando = False
            await sondeo
            await tarea

        # Once the lock is released, the worker can finish and persist
        # CANCELLED before this coroutine serializes its response. Both are
        # valid snapshots; the monotone history below is the cancellation
        # invariant this test is intended to protect.
        assert respuesta.estado in (
            JobStatus.CANCELLING.value, JobStatus.CANCELLED.value,
        ), respuesta
        historial = self._historial_de_estados(job_id)
        assert historial == [
            JobStatus.PENDING.value, JobStatus.RUNNING.value,
            JobStatus.CANCELLING.value, JobStatus.CANCELLED.value,
        ], f"el historial fue para atrás (o se salteó un paso): {historial}"
        assert max(retrasos) < 0.15, (
            f"el loop de eventos se congeló {max(retrasos):.3f}s esperando el lock "
            f"de _ControlTrabajo desde cancelar_trabajo()"
        )

    async def test_N3r5_cancelado_antes_de_arrancar_nunca_escribe_running_ni_started_at(self):
        """N3 (ronda 5): un archivo cancelado mientras esperaba en la cola
        del pool (ocupado por otro trabajo) nunca corrió nada -- el job no
        puede haber pasado por `running` ni tener `started_at`. Sobrevivía
        la mutación que escribe `RUNNING` ANTES de mirar `self.cancelado`:
        el test de N-3 de arriba sólo mira el estado FINAL."""
        executor_1_hilo = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-n3r5")
        self.addCleanup(executor_1_hilo.shutdown)
        loop = asyncio.get_running_loop()
        bloqueo = threading.Event()
        self.addCleanup(bloqueo.set)  # mismo motivo que en test_N4_running_no_se_marca...
        ocupa = loop.run_in_executor(executor_1_hilo, bloqueo.wait)  # "otro trabajo" ocupa el pool

        semaforo = asyncio.Semaphore(1)
        self._archivo_en_workspace("en-cola.pdf")
        ingerir_mock = AsyncMock()
        job_id = self._crear_job()

        with patch.object(rutas_mod.ingesta, "ingerir", ingerir_mock):
            await semaforo.acquire()
            tarea = asyncio.create_task(
                rutas_mod._ejecutar_trabajo(
                    job_id, UUID_PRUEBA, ["en-cola.pdf"], store=self.store,
                    executor=executor_1_hilo, executor_io=self.executor_io, semaforo=semaforo,
                )
            )
            await asyncio.sleep(0.05)  # "en-cola.pdf" quedó esperando turno
            rutas_mod._CONTROLES[job_id].cancelar()
            bloqueo.set()  # el pool se libera; ahora le toca el turno al cancelado
            await tarea
            await ocupa

        ingerir_mock.assert_not_called()
        job = self.store.get(job_id)
        assert job.status == JobStatus.CANCELLED, job
        historial = self._historial_de_estados(job_id)
        assert JobStatus.RUNNING.value not in historial, (
            f"un trabajo que nunca arrancó quedó registrado como running: {historial}"
        )
        assert job.started_at is None, (
            f"un trabajo que nunca arrancó tiene started_at={job.started_at}"
        )

    # -- N2 (ronda 5): los nombres con tilde son el caso NORMAL -------------
    _NOMBRES_ACENTUADOS = (
        "Constitución de sociedad.pdf",
        "factura_año.pdf",
        "Remodelación/Crédito Ñandú.xlsx",
        "Diseño – Etapa Única.docx",
    )

    def _resultados_acentuados(self) -> list[dict]:
        return [
            {
                "archivo": nombre, "estado": "error" if i == 1 else "ok",
                "extractor": None if i == 1 else "pdf", "extracto_bytes": 10 * i,
                "carpeta_procesado": None if i == 1 else f"proyectos/remodelación/procesado/{i}",
                "error": "no se pudo abrir 'Crédito Ñandú.xlsx'" if i == 1 else None,
            }
            for i, nombre in enumerate(self._NOMBRES_ACENTUADOS)
        ]

    def test_N2r5_leer_resultado_devuelve_los_nombres_acentuados_intactos(self):
        """N2 (ronda 5): el saneo de lectura de MINOR-C tiene que tocar
        SÓLO lo que no se puede codificar a UTF-8. Los documentos de este
        sistema son de clientes hondureños: casi todo nombre lleva tilde o
        eñe. Sanear todo `str` convertiría `factura_año.pdf` en
        `factura_a\\xf1o.pdf` en cada respuesta -- y hasta la ronda 4
        ningún test lo notaba. Se prueban los DOS formatos que
        `_guardar_resultado` puede dejar en disco (UTF-8 directo y el
        fallback `ensure_ascii=True`)."""
        esperados = self._resultados_acentuados()
        with tempfile.TemporaryDirectory() as d:
            store = JobStore(str(Path(d) / "jobs.jsonl"))
            job_id = store.create(caller="x", capability="y", motor="z", trace_id="t", prompt="p", recursion_depth=0)
            ruta_utf8 = rutas_mod._guardar_resultado(store, job_id, esperados)
            leidos_utf8 = rutas_mod._leer_resultados_de_disco(ruta_utf8)

            ruta_ascii = Path(d) / "fallback.json"
            ruta_ascii.write_text(json.dumps(esperados, ensure_ascii=True), encoding="utf-8")
            leidos_ascii = rutas_mod._leer_resultados_de_disco(str(ruta_ascii))

        assert leidos_utf8 == esperados, leidos_utf8
        assert leidos_ascii == esperados, leidos_ascii
        assert leidos_utf8[1]["archivo"] == "factura_año.pdf"
    # -- Ronda 6 --------------------------------------------------------------
    async def test_R6a_tilde_mas_surrogate_no_tumba_el_lote(self):
        """Ronda 6, defecto 1: `_saneado_ascii` hacía
        `encode("utf-8", "backslashreplace").decode("ascii")` --
        `backslashreplace` no toca una tilde VÁLIDA, la deja en bytes UTF-8,
        y el `decode("ascii")` revienta. Un nombre con tilde Y un surrogate
        (`Crédito\\udcff.pdf`) tumbaba el lote entero: `failed`, sin
        resultados, y el sano perdido."""
        self._archivo_en_workspace("sano.pdf")
        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", return_value=_ficha("8" * 64)) as ingerir_mock:
            await self._ejecutar(job_id, UUID_PRUEBA, ["sano.pdf", "Crédito\udcff.pdf"])

        job = self.store.get(job_id)
        assert job.status == JobStatus.COMPLETED, f"tilde + surrogate tumbó el lote: {job}"
        resultados = json.loads(Path(job.result_path).read_text(encoding="utf-8"))
        assert [r["estado"] for r in resultados] == ["ok", "rechazado"], resultados
        assert resultados[0]["archivo"] == "sano.pdf"
        assert resultados[1]["archivo"] == "Cr\\xe9dito\\udcff.pdf", resultados[1]
        assert ingerir_mock.call_count == 1

    def test_R6a_saneado_ascii_es_ascii_con_tilde_y_surrogate(self):
        for s in ("malo\udcff.pdf", "Crédito\udcff.pdf", "factura_año.pdf"):
            saneado = rutas_mod._saneado_ascii(s)
            saneado.encode("ascii")  # no debe lanzar
        assert rutas_mod._saneado_ascii("Crédito\udcff.pdf") == "Cr\\xe9dito\\udcff.pdf"

    async def test_R6b_cancelar_antes_de_que_arranque_el_worker_termina_cancelled(self):
        """Ronda 6, defecto 2: el `_ControlTrabajo` lo creaba el worker al
        arrancar. Un `POST .../cancel` que llegaba entre `crear_trabajo()` y
        esa primera vuelta del loop no encontraba control, escribía
        `CANCELLING` a pelo, y el worker arrancaba después con
        `cancelado=False`: historial `pending, pending, cancelling,
        running, completed` -- la cancelación perdida. Ahora el control
        existe desde `crear_trabajo()`, antes de programar la tarea: la
        ventana no existe."""
        self._archivo_en_workspace("a.pdf")
        ingerir_mock = AsyncMock()
        with patch.object(rutas_mod, "_STORE", self.store), \
             patch.object(rutas_mod, "_EXECUTOR_OCR", self.executor), \
             patch.object(rutas_mod, "_EXECUTOR_IO", self.executor_io), \
             patch.object(rutas_mod, "_SEMAFORO_TRABAJOS", asyncio.Semaphore(2)), \
             patch.object(rutas_mod.proyecto_activo, "identidad_activa_del_proyecto", AsyncMock(return_value=True)), \
             patch.object(rutas_mod.ingesta, "ingerir", ingerir_mock):
            creado = await rutas_mod.crear_trabajo(
                rutas_mod.TrabajoRequest(project_uuid=UUID_PRUEBA, rutas=["a.pdf"]), _owned_request()
            )
            # SIN ceder el loop: el worker todavía no dio su primera vuelta.
            respuesta = await rutas_mod.cancelar_trabajo(creado.job_id, _owned_request())
            for _ in range(200):
                if self.store.get(creado.job_id).status in (
                    JobStatus.CANCELLED, JobStatus.COMPLETED, JobStatus.FAILED,
                ):
                    break
                await asyncio.sleep(0.01)

        assert respuesta.estado == JobStatus.CANCELLING.value, respuesta
        # CRUDO (una entrada por línea del JSONL): el código de 146e0a5/f1d4298
        # da exactamente ['pending', 'pending', 'cancelling', 'running', 'completed'].
        historial = self._historial_de_estados(creado.job_id, crudo=True)
        assert historial == [
            JobStatus.PENDING.value, JobStatus.PENDING.value,
            JobStatus.CANCELLING.value, JobStatus.CANCELLED.value,
        ], f"la cancelación se perdió o el historial fue para atrás: {historial}"
        assert self.store.get(creado.job_id).started_at is None
        ingerir_mock.assert_not_called()

# ===========================================================================
#  Grupo 2 -- HTTP: admisión, cancelación, no-bloqueo, autenticación
# ===========================================================================
CRED = {
    IDENTIDAD_PLATAFORMA: secrets.token_urlsafe(32),
    IDENTIDAD_JACOBS: secrets.token_urlsafe(32),
}


def _credenciales() -> dict[str, bytes]:
    return {k: v.encode("ascii") for k, v in CRED.items()}


def _app(**kw) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    proteger(app, _credenciales())
    return app


def _h(identidad: str | None) -> dict[str, str]:
    headers = {ENCABEZADO: CRED[identidad]} if identidad else {}
    if identidad == IDENTIDAD_PLATAFORMA:
        headers.update({
            "X-Jax-Processing-Owner-Version": OWNER.version,
            "X-Jax-Processing-Tenant-Id": OWNER.tenant_id,
            "X-Jax-Processing-User-Id": OWNER.user_id,
            "X-Jax-Processing-Project-Id": OWNER.project_id,
        })
    return headers


def _owned_request() -> Request:
    return Request({"type": "http", "state": {"processing_ownership": OWNER}})


import contextlib
import importlib
import os

import pytest

from procesamiento import dependencias
import aviso_extractores

_ESTADO_REAL = dependencias.estado


@pytest.fixture(autouse=True)
def _extractores_instalados_por_defecto(monkeypatch):
    """jax-14: el job `governance` de CI NO instala requirements-archivos.txt, asi
    que `dependencias.estado()` real daria 503 en TODO POST. Salvo las pruebas
    del freno (que fijan su propio estado con `_estado_sin_modulos`/`_estado_falta`),
    el estado es "todo ok" -- estas pruebas miden el endpoint, no el venv."""
    monkeypatch.setattr(dependencias, "estado", lambda *a, **k: {"ok": True, "faltan": [], "motivos": {}})
    aviso_extractores._reiniciar()
    yield
    aviso_extractores._reiniciar()


@contextlib.contextmanager
def _estado_sin_modulos(*modulos, error=ModuleNotFoundError):
    """El `estado()` REAL, con un importador controlado: solo `modulos` fallan; el
    resto "se importa". Asi las pruebas valen igual con o sin las dependencias
    en el interprete (el job `governance` no las instala)."""
    def importador(nombre):
        if nombre in modulos:
            raise error(f"simulado: {nombre}")
        return object()

    with patch.object(dependencias, "estado", lambda *a, **k: _ESTADO_REAL(importador=importador)):
        yield


@contextlib.contextmanager
def _estado_falta(*paquetes):
    estado = {"ok": False, "faltan": sorted(paquetes), "motivos": {p: "ModuleNotFoundError" for p in paquetes}}
    with patch.object(dependencias, "estado", lambda *a, **k: estado):
        yield


class TrabajoHTTPTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.store = _ProcessingJobStoreFixture(str(Path(self._tmpdir.name) / "jobs.jsonl"))
        self._parche_store = patch.object(rutas_mod, "_STORE", self.store)
        self._parche_store.start()
        self.addCleanup(self._parche_store.stop)

        self._semaforo_test = asyncio.Semaphore(2)
        self._parche_semaforo = patch.object(rutas_mod, "_SEMAFORO_TRABAJOS", self._semaforo_test)
        self._parche_semaforo.start()
        self.addCleanup(self._parche_semaforo.stop)

        # Ronda 6: `crear_trabajo()` registra el control en `_CONTROLES`, y
        # los workers falsos de estos tests nunca lo sacan. Cada test arranca
        # y termina con el registro vacío.
        self._parche_controles = patch.dict(rutas_mod._CONTROLES, clear=True)
        self._parche_controles.start()
        self.addCleanup(self._parche_controles.stop)

        # E2a: el proyecto tiene que existir y estar ACTIVE (si no, 422).
        self._parche_activo = patch.object(
            rutas_mod.proyecto_activo, "identidad_activa_del_proyecto", AsyncMock(return_value=True))
        self._parche_activo.start()
        self.addCleanup(self._parche_activo.stop)

    def _post(self, c, **overrides):
        cuerpo = {"project_uuid": UUID_PRUEBA, "rutas": ["a.pdf"]}
        cuerpo.update(overrides)
        return c.post("/procesamiento/trabajos", json=cuerpo, headers=_h(IDENTIDAD_PLATAFORMA))

    def test_processing_ownership_rejections_are_retryable_and_do_not_append(self):
        body = {"project_uuid": UUID_PRUEBA, "rutas": [], "usuario": "legacy"}
        invalid_headers = [
            {ENCABEZADO: CRED[IDENTIDAD_PLATAFORMA]},
            {**_h(IDENTIDAD_PLATAFORMA), "X-Jax-Processing-Tenant-Id": "03"},
            {**_h(IDENTIDAD_PLATAFORMA), "X-Jax-Processing-Unknown": "x"},
        ]
        with TestClient(_app()) as client:
            for headers in invalid_headers:
                response = client.post("/procesamiento/trabajos", json=body, headers=headers)
                assert response.status_code == 403, response.text
                assert response.json()["detail"]["code"] == "processing_ownership_invalid"
                assert self.store._index == {}

            # Starlette preserves raw duplicate headers through the ASGI
            # scope, so this proves the middleware rejects the ambiguous
            # envelope before Pydantic or the store runs.
            duplicate_headers = list(_h(IDENTIDAD_PLATAFORMA).items()) + [
                ("X-Jax-Processing-Tenant-Id", OWNER.tenant_id),
            ]
            response = client.send(client.build_request(
                "POST", "/procesamiento/trabajos", json=body, headers=duplicate_headers,
            ))
            assert response.status_code == 403, response.text
            assert response.json()["detail"]["code"] == "processing_ownership_invalid"
            assert self.store._index == {}

    def test_body_usuario_with_valid_owner_is_extra_forbidden(self):
        with TestClient(_app()) as client:
            response = self._post(client, usuario="legacy")
        assert response.status_code == 422, response.text
        assert self.store._index == {}

    # -- jax-14 (2026-10-03): freno de dependencias de extraccion ----------
    # El fixture autouse del modulo fija `dependencias.estado` en "todo ok"; estas
    # pruebas fijan el suyo: `_estado_sin_modulos(...)` es el estado REAL con un importador
    # controlado (vale con o sin las dependencias en el interprete), `_estado_falta(...)` uno sintetico.
    def test_post_con_pdf_y_pdfplumber_ausente_da_503_con_faltan_y_tipos_y_no_crea_trabajo(self):
        with _estado_sin_modulos("pdfplumber"), TestClient(_app()) as client:
            response = self._post(client, rutas=["a.pdf"])
        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "extractores_no_disponibles"
        assert detail["faltan"] == ["pdfplumber"]
        assert detail["tipos"] == [".pdf"]
        assert self.store._index == {}

    def test_post_503_por_cada_tipo_afectado(self):
        casos = {"docx": ("python-docx", "b.docx"), "openpyxl": ("openpyxl", "c.xlsx")}
        for modulo, (paquete, ruta) in casos.items():
            with _estado_sin_modulos(modulo), TestClient(_app()) as client:
                response = self._post(client, rutas=[ruta])
            assert response.status_code == 503, (modulo, response.text)
            assert response.json()["detail"]["faltan"] == [paquete]

    def test_post_lote_solo_de_imagenes_con_pdfplumber_ausente_sigue_admitiendo(self):
        async def _noop(*args, **kwargs):
            return None

        with _estado_falta("pdfplumber"), patch.object(rutas_mod, "_ejecutar_trabajo", _noop), \
             TestClient(_app()) as client:
            response = self._post(client, rutas=["a.png", "b.jpg"])
        assert response.status_code == 202, response.text

    def test_post_lote_mixto_con_un_pdf_y_pdfplumber_ausente_da_503(self):
        with _estado_falta("pdfplumber"), TestClient(_app()) as client:
            response = self._post(client, rutas=["a.png", "b.pdf"])
        assert response.status_code == 503, response.text

    def test_post_sin_pillow_una_imagen_con_firma_llamada_foto_da_503(self):
        """Jax#338 ronda 18 (MAJOR de Sol r17): el freno decide como la compuerta,
        por la FIRMA del contenido (leido a traves del jail); un PNG llamado
        `foto` frena el lote igual que un `.png`. Sin Pillow en la prueba: el job
        `governance` no la instala, y el freno solo lee la firma."""
        foto = Path(self._tmpdir.name) / "foto"
        foto.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

        async def _noop(*args, **kwargs):
            return None

        with _estado_falta("pillow"), patch.object(rutas_mod, "_ejecutar_trabajo", _noop), \
             patch.object(rutas_mod.tool_authority, "resolve_jailed_path", return_value=(foto, "")), \
             TestClient(_app()) as client:
            response = self._post(client, rutas=["foto"])
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["faltan"] == ["pillow"]
        assert self.store._index == {}

    def test_post_lote_sobre_el_maximo_con_dependencia_faltante_da_422_sin_leer_cabeceras(self):
        """Jax#338 ronda 18: las validaciones baratas (cantidad maxima de rutas,
        formato del project_uuid) van ANTES de que el freno lea cabeceras."""
        leidas: list = []

        def abrir_espia(ruta):
            leidas.append(ruta)
            return None

        with _estado_falta("pillow"), patch.object(rutas_mod, "_ruta_del_jail", abrir_espia), \
             patch.object(rutas_mod, "_MAX_RUTAS_POR_TRABAJO", 2), TestClient(_app()) as client:
            response = self._post(client, rutas=["a.png", "b.png", "c.png"])
        assert response.status_code == 422, response.text
        assert leidas == [], "el freno leyo cabeceras de un lote que se rechaza por tamano"
        assert self.store._index == {}

    def test_post_project_uuid_invalido_con_dependencia_faltante_da_422_sin_leer_cabeceras(self):
        leidas: list = []

        def abrir_espia(ruta):
            leidas.append(ruta)
            return None

        with _estado_falta("pillow"), patch.object(rutas_mod, "_ruta_del_jail", abrir_espia), \
             TestClient(_app()) as client:
            response = self._post(client, project_uuid="no-es-un-uuid", rutas=["a.png"])
        assert response.status_code == 422, response.text
        assert response.json()["detail"]["code"] == "project_uuid_invalido"
        assert leidas == []

    def test_post_paquete_faltante_sin_mapa_bloquea_todos_los_tipos(self):
        with _estado_falta("linea-no-reconocida:3"), TestClient(_app()) as client:
            response = self._post(client, rutas=["a.png"])
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["tipos"] == ["*"]

    def test_post_503_por_extractores_ocurre_antes_del_acquire_del_semaforo(self):
        """MINOR-4: un espia sobre el semaforo. Si el 503 se mueve DESPUES del
        acquire, `acquire` se llama y esta prueba falla (cupo tomado y, segun
        el camino, filtrado)."""
        espia = Mock()
        espia.locked = Mock(return_value=False)
        espia.acquire = AsyncMock()
        espia.release = Mock()
        with _estado_falta("pdfplumber"), patch.object(rutas_mod, "_SEMAFORO_TRABAJOS", espia), \
             TestClient(_app()) as client:
            for _ in range(3):
                assert self._post(client, rutas=["a.pdf"]).status_code == 503
        espia.acquire.assert_not_called()

    def test_post_con_todo_instalado_sigue_admitiendo(self):
        async def _noop(*args, **kwargs):
            return None

        with patch.object(rutas_mod, "_ejecutar_trabajo", _noop), TestClient(_app()) as client:
            response = self._post(client)
        assert response.status_code == 202, response.text

    def test_un_importador_que_lanza_attributeerror_da_503_y_health_200_y_el_arranque_no_cae(self):
        import server

        def importador(nombre, *a, **k):
            if nombre == "pdfplumber":
                raise AttributeError("a medio instalar")
            return object()

        with patch.object(dependencias, "estado", lambda *a, **k: _ESTADO_REAL(importador=importador)):
            with TestClient(_app()) as client:
                post = self._post(client, rutas=["a.pdf"])
            with patch.object(server._salud, "estado", AsyncMock(return_value={"ok": True, "fallos": []})):
                health = TestClient(server.app).get("/health")
            with patch("jacobs.reaper.send_telegram_alert", AsyncMock(return_value={"ok": True})):
                self._arrancar_servidor()  # no lanza: el servicio arranca igual
        assert post.status_code == 503, post.text
        assert post.json()["detail"]["code"] == "extractores_no_disponibles"
        assert health.status_code == 200, health.text
        ext = health.json()["extractores"]
        assert ext["ok"] is False and ext["faltan"] == ["pdfplumber"]
        assert ext["motivos"] == {"pdfplumber": "AttributeError"}

    def test_health_informa_el_estado_de_los_extractores_sin_dejar_de_dar_200(self):
        import server

        with patch.object(server._salud, "estado", AsyncMock(return_value={"ok": True, "fallos": []})):
            with _estado_sin_modulos("docx"):
                # sin `with`: no corre el startup (necesita MariaDB real)
                roto = TestClient(server.app).get("/health")
            sano = TestClient(server.app).get("/health")
        assert roto.status_code == 200, roto.text
        assert roto.json()["extractores"]["ok"] is False
        assert roto.json()["extractores"]["faltan"] == ["python-docx"]
        assert sano.status_code == 200, sano.text
        assert sano.json()["extractores"] == {"ok": True, "faltan": [], "motivos": {}}

    def _arrancar_servidor(self):
        import server

        async def _arrancar():
            for handler in server.app.router.on_startup:
                await handler()
            # deja correr la tarea fire-and-forget del aviso
            await asyncio.sleep(0)
            await asyncio.gather(*list(aviso_extractores._TAREAS))

        with patch.object(server, "_configure_b7_trusted_runtime"), \
             patch("jacobs.subpipelines.config_subpipelines"), \
             patch.object(server.jacobs_store, "tamanio_pool"), \
             patch.object(server.jacobs_store, "init_tables", AsyncMock()), \
             patch("procesamiento_idempotencia.init_tabla", AsyncMock()), \
             patch("motor_registry.routes.init_motor_catalog", AsyncMock()), \
             patch("jacobs.reaper.reap_orphaned_pipelines", AsyncMock()), \
             patch("jacobs.reaper.start_reaper_loop", AsyncMock()), \
             patch.object(rutas_mod, "reconciliar_trabajos_huerfanos", Mock(return_value=0)):
            asyncio.run(_arrancar())

    def test_el_arranque_loguea_error_y_avisa_si_faltan_extractores_y_no_se_cae(self):
        enviar = AsyncMock(return_value={"ok": True, "message_id": 1, "error": None})
        with _estado_falta("pdfplumber"), patch("jacobs.reaper.send_telegram_alert", enviar):
            with self.assertLogs(level="ERROR") as capturado:
                self._arrancar_servidor()
        mensajes = [r.getMessage() for r in capturado.records]
        assert any("pdfplumber" in m and "extractores" in m for m in mensajes), mensajes
        enviar.assert_awaited_once()
        assert "pdfplumber" in enviar.await_args.args[0]

    def test_el_arranque_con_todo_ok_no_avisa(self):
        enviar = AsyncMock()
        with patch("jacobs.reaper.send_telegram_alert", enviar):
            self._arrancar_servidor()
        enviar.assert_not_awaited()

    def test_el_aviso_desde_el_post_sale_una_vez_por_hora(self):
        enviar = AsyncMock(return_value={"ok": True, "message_id": 1, "error": None})
        reloj = [1000.0]
        with _estado_falta("pdfplumber"), patch("jacobs.reaper.send_telegram_alert", enviar), \
             patch.object(aviso_extractores, "_reloj", lambda: reloj[0]), TestClient(_app()) as client:
            assert self._post(client, rutas=["a.pdf"]).status_code == 503
            time.sleep(0.2)
            reloj[0] += 1800
            assert self._post(client, rutas=["a.pdf"]).status_code == 503
            time.sleep(0.2)
            primera = enviar.await_count
            reloj[0] += 1801  # ya pasó más de una hora desde el primero
            assert self._post(client, rutas=["a.pdf"]).status_code == 503
            time.sleep(0.2)
        assert primera == 1, "la segunda dentro de la hora NO avisa"
        assert enviar.await_count == 2

    def test_el_texto_realmente_enviado_no_lleva_las_rutas_del_lote(self):
        """MINOR-N4: se afirma sobre lo que recibio `send_telegram_alert`, no sobre
        un texto armado por la propia prueba. Si `mensaje()` metiera rutas, falla."""
        enviar = AsyncMock(return_value={"ok": True, "message_id": 1, "error": None})
        rutas = ["secreto-del-cliente.pdf", "carpeta/balance-confidencial.pdf"]
        with _estado_falta("pdfplumber"), patch("jacobs.reaper.send_telegram_alert", enviar), \
             TestClient(_app()) as client:
            r = self._post(client, rutas=rutas)
            time.sleep(0.2)
        assert r.status_code == 503, r.text
        enviar.assert_awaited_once()
        enviado = enviar.await_args.args[0]
        assert "pdfplumber" in enviado
        for ruta in rutas:
            assert ruta not in enviado and Path(ruta).stem not in enviado

    def test_sin_credenciales_de_telegram_el_503_no_rompe(self):
        with _estado_falta("pdfplumber"), \
             patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}), \
             TestClient(_app()) as client:
            r = self._post(client, rutas=["a.pdf"])
            time.sleep(0.2)
        assert r.status_code == 503, r.text

    def test_el_log_del_503_sale_como_mucho_una_vez_por_minuto(self):
        reloj = [9000.0]
        with _estado_falta("pdfplumber"), patch("jacobs.reaper.send_telegram_alert", AsyncMock()), \
             patch.object(aviso_extractores, "_reloj", lambda: reloj[0]), TestClient(_app()) as client:
            with self.assertLogs(rutas_mod.logger, level="ERROR") as capturado:
                for _ in range(5):
                    self._post(client, rutas=["a.pdf"])
                reloj[0] += 61
                self._post(client, rutas=["a.pdf"])
        rechazos = [r for r in capturado.records if "rechazado" in r.getMessage()]
        assert len(rechazos) == 2

    def test_el_error_con_codigo_de_la_ficha_llega_en_el_resultado_del_archivo(self):
        ficha = Mock(estado="error", extractor="pdf", sha256="a" * 64, detalle={"codigo": "dependencia_no_instalada"})
        with tempfile.TemporaryDirectory() as tmp:
            archivo = Path(tmp) / "x.pdf"
            archivo.write_bytes(b"%PDF-1.4")
            with patch.object(rutas_mod.tool_authority, "resolve_jailed_path", return_value=(archivo, "")), \
                 patch.object(rutas_mod.ingesta, "ingerir", return_value=ficha), \
                 patch.object(rutas_mod.ingesta, "ruta_procesado", return_value=Path(tmp) / "nada"), \
                 patch.object(rutas_mod.tool_authority, "WORKSPACE_ROOT", Path(tmp)):
                r = rutas_mod._procesar_una_ruta(Path(tmp), "x.pdf")
        assert r.estado == "error"
        assert r.error == "dependencia_no_instalada"

    def _error_de_ficha(self, detalle):
        ficha = Mock(estado="error", extractor="tesseract", sha256="a" * 64, detalle=detalle)
        with tempfile.TemporaryDirectory() as tmp:
            archivo = Path(tmp) / "x.gif"
            archivo.write_bytes(b"GIF89a")
            with patch.object(rutas_mod.tool_authority, "resolve_jailed_path", return_value=(archivo, "")), \
                 patch.object(rutas_mod.ingesta, "ingerir", return_value=ficha), \
                 patch.object(rutas_mod.ingesta, "ruta_procesado", return_value=Path(tmp) / "nada"), \
                 patch.object(rutas_mod.tool_authority, "WORKSPACE_ROOT", Path(tmp)):
                return rutas_mod._procesar_una_ruta(Path(tmp), "x.gif")

    def test_formato_no_soportado_viaja_con_su_formato_en_el_error(self):
        r = self._error_de_ficha({"codigo": "formato_no_soportado", "formato": "gif_animado"})
        assert r.estado == "error"
        assert r.error == "formato_no_soportado:gif_animado"

    def test_png_animado_es_un_formato_estable_del_contrato(self):
        """Jax#338 ronda 18: `png_animado` viaja como los otros formatos (pasa la
        validacion `[a-z0-9_]{1,40}`)."""
        r = self._error_de_ficha({"codigo": "formato_no_soportado", "formato": "png_animado"})
        assert r.error == "formato_no_soportado:png_animado"

    def test_formato_no_soportado_sin_formato_viaja_solo_el_codigo(self):
        r = self._error_de_ficha({"codigo": "formato_no_soportado"})
        assert r.error == "formato_no_soportado"

    def test_el_formato_solo_se_agrega_a_formato_no_soportado_y_se_valida(self):
        assert self._error_de_ficha({"codigo": "archivo_ilegible", "formato": "gif_animado"}).error == "archivo_ilegible"
        # una ficha cacheada con un `formato` raro no mete texto libre en el error
        assert self._error_de_ficha(
            {"codigo": "formato_no_soportado", "formato": "x; DROP\nTABLE"}
        ).error == "formato_no_soportado"

    def test_n29_un_salto_de_linea_al_final_del_formato_no_se_acepta(self):
        """`$` coincide antes de un `\\n` final: con `fullmatch` no."""
        for valor in ("gif_animado\n", "gif_animado\r\n", "\ngif_animado", "gif animado"):
            r = self._error_de_ficha({"codigo": "formato_no_soportado", "formato": valor})
            assert r.error == "formato_no_soportado", repr(valor)
        assert self._error_de_ficha(
            {"codigo": "formato_no_soportado", "formato": "gif_animado"}).error == "formato_no_soportado:gif_animado"

    def test_b6_persists_authenticated_human_uploader_as_caller(self):
        async def _noop(*args, **kwargs):
            return None

        with patch.object(rutas_mod, "_ejecutar_trabajo", _noop), TestClient(_app()) as client:
            response = self._post(client)
        assert response.status_code == 202, response.text
        assert self.store.get(response.json()["job_id"]).caller == "user:2"

    def test_get_and_cancel_require_each_exact_owner_dimension_without_mutation(self):
        job_id = self.store.create(
            caller="x", capability="x", motor="n/a", trace_id="t", prompt="", recursion_depth=0,
        )
        with TestClient(_app()) as client:
            # The exact authenticated owner retains normal read access.
            control = client.get(f"/procesamiento/trabajos/{job_id}", headers=_h(IDENTIDAD_PLATAFORMA))
            assert control.status_code == 200, control.text

            for header, wrong_value in (
                ("X-Jax-Processing-Tenant-Id", "9"),
                ("X-Jax-Processing-User-Id", "9"),
                ("X-Jax-Processing-Project-Id", "9"),
            ):
                wrong_owner = {**_h(IDENTIDAD_PLATAFORMA), header: wrong_value}
                before = len(Path(self.store._path).read_text().splitlines())
                get_response = client.get(f"/procesamiento/trabajos/{job_id}", headers=wrong_owner)
                cancel_response = client.post(f"/procesamiento/trabajos/{job_id}/cancel", headers=wrong_owner)
                assert get_response.status_code == 404, get_response.text
                assert cancel_response.status_code == 404, cancel_response.text
                assert len(Path(self.store._path).read_text().splitlines()) == before
                assert self.store.get(job_id).status == JobStatus.PENDING

    # -- N-1: el permiso siempre vuelve --------------------------------------
    def test_N1_falla_al_crear_el_job_no_deja_el_semaforo_agotado(self):
        """El `try/finally` de N-1 protege CUALQUIER falla en el tramo
        `acquire()` -> tarea registrada -- no sólo la de `proyecto`
        (que MINOR-B cierra en el ORIGEN, antes de tocar el semáforo
        siquiera -- ver `test_MINORB_proyecto_no_codificable_se_rechaza_
        en_la_admision`, más abajo). Se simula con `_STORE.create()`
        mockeado para reventar -- así este test sigue siendo válido pase
        lo que pase con las validaciones de campos puntuales, presentes o
        futuras."""
        async def _correr():
            with patch.object(self.store, "create", side_effect=RuntimeError("boom -- create() reventó")):
                for _ in range(2):  # tamaño real del semáforo de prueba
                    req = rutas_mod.TrabajoRequest(project_uuid=UUID_PRUEBA, rutas=[])
                    with self.assertRaises(RuntimeError):
                        await rutas_mod.crear_trabajo(req, _owned_request())

            assert not self._semaforo_test.locked(), (
                "el semáforo quedó agotado tras dos pedidos rotos -- DoS de N requests"
            )
            # y un pedido LIMPIO subsiguiente se admite normalmente
            req_limpio = rutas_mod.TrabajoRequest(project_uuid=UUID_PRUEBA, rutas=[])
            with patch.object(rutas_mod, "_ejecutar_trabajo", AsyncMock()):
                respuesta = await rutas_mod.crear_trabajo(req_limpio, _owned_request())
            assert respuesta.job_id

        asyncio.run(_correr())

    def test_MINORB_proyecto_no_codificable_se_rechaza_en_la_admision(self):
        """MINOR-B (ronda 4): antes, un `proyecto` con un surrogate
        solitario (`\\udcff`) llegaba hasta `_STORE.update()` -- que
        revienta DESPUÉS de que `create()` YA escribió un registro
        `pending` que nunca avanza (N-1 evitaba que el semáforo quedara
        agotado, pero no evitaba el registro huérfano). Ahora se rechaza
        EN LA ADMISIÓN, antes de tocar el semáforo siquiera: 422 limpio,
        ningún job creado."""
        # E2a: el `project_uuid` con un surrogate lo rechaza la regex del UUID
        # canónico, en la admisión: 422 `project_uuid_invalido`, nunca 500.
        proyecto_malo = UUID_PRUEBA[:-1] + "\udcff"
        cuerpo_json = json.dumps(
            {"project_uuid": proyecto_malo, "rutas": []},
            ensure_ascii=True,
        ).encode("ascii")
        with TestClient(_app(), raise_server_exceptions=False) as c:
            r = c.post(
                "/procesamiento/trabajos", content=cuerpo_json,
                headers={**_h(IDENTIDAD_PLATAFORMA), "content-type": "application/json"},
            )
            assert r.status_code == 422, r.text
            assert r.json()["detail"]["code"] == "project_uuid_invalido", r.text
            assert not self._semaforo_test.locked(), (
                "el semáforo se tocó aunque el proyecto se rechazó en la admisión"
            )
            assert len(self.store._index) == 0, "no debería haberse creado ningún job"

    # -- B-5 / no bloqueo --------------------------------------------------
    def test_post_no_espera_a_que_el_trabajo_termine(self):
        async def _lento(job_id, proyecto, rutas, *, store, executor=None, executor_io=None, semaforo=None, control=None):
            await asyncio.sleep(2)
            store.update(job_id, status=JobStatus.COMPLETED.value, finished_at=time.time())

        with patch.object(rutas_mod, "_ejecutar_trabajo", _lento), \
             TestClient(_app()) as c:
            t0 = time.perf_counter()
            r = self._post(c)
            elapsed = time.perf_counter() - t0

        assert r.status_code == 202, r.text
        assert "job_id" in r.json()
        assert elapsed < 1.0, f"el POST esperó al trabajo ({elapsed:.2f}s) -- no es asíncrono"

    # -- 404 / forma de la respuesta -----------------------------------
    def test_get_job_desconocido_404(self):
        with TestClient(_app()) as c:
            r = c.get("/procesamiento/trabajos/no-existe", headers=_h(IDENTIDAD_PLATAFORMA))
        assert r.status_code == 404, r.text

    def test_get_trae_estado_y_resultados_del_job_completado(self):
        job_id = self.store.create(
            caller="x", capability="y", motor="z", trace_id="t", prompt="p",
            recursion_depth=0,
        )
        resultados = [{
            "archivo": "a.pdf", "estado": "ok", "extractor": "pdf",
            "extracto_bytes": 42, "carpeta_procesado": "proyectos/p/procesado/abc",
            "error": None,
        }]
        result_path = self.store.write_result(job_id, json.dumps(resultados))
        self.store.update(
            job_id, status=JobStatus.COMPLETED.value, finished_at=time.time(),
            result_path=result_path,
        )

        with TestClient(_app()) as c:
            r = c.get(f"/procesamiento/trabajos/{job_id}", headers=_h(IDENTIDAD_PLATAFORMA))

        assert r.status_code == 200, r.text
        cuerpo = r.json()
        assert cuerpo["job_id"] == job_id
        assert cuerpo["estado"] == "completed"
        assert cuerpo["resultados"] == resultados

    # -- B-6: principal obligatorio, no vacío, con tope --------------------
    def test_B6_usuario_no_es_requerido(self):
        with TestClient(_app()) as c:
            r = c.post(
                "/procesamiento/trabajos", json={"project_uuid": UUID_PRUEBA, "rutas": []},
                headers=_h(IDENTIDAD_PLATAFORMA),
            )
        assert r.status_code == 202, r.text

    def test_B6_usuario_vacio_es_extra_422(self):
        with TestClient(_app()) as c:
            r = self._post(c, usuario="")
        assert r.status_code == 422, r.text

    def test_B6_usuario_muy_largo_es_extra_422(self):
        with TestClient(_app()) as c:
            r = self._post(c, usuario="x" * 300)
        assert r.status_code == 422, r.text

    def test_B6_usuario_no_puede_elegir_caller(self):
        async def _noop(job_id, proyecto, rutas, *, store, executor=None, executor_io=None, semaforo=None, control=None):
            return None

        with patch.object(rutas_mod, "_ejecutar_trabajo", _noop), TestClient(_app()) as c:
            r = self._post(c, usuario="carlos@cliente.com")

        assert r.status_code == 422, r.text

    # -- B-2: admisión ------------------------------------------------------
    def test_B2_demasiadas_rutas_es_422_y_no_crea_job(self):
        with patch.object(rutas_mod, "_MAX_RUTAS_POR_TRABAJO", 3), TestClient(_app()) as c:
            r = self._post(c, rutas=["a.pdf", "b.pdf", "c.pdf", "d.pdf"])
        assert r.status_code == 422, r.text
        assert len(self.store._index) == 0, "no debería haberse creado ningún job"

    def test_B2_sin_capacidad_es_429_y_no_crea_job(self):
        with TestClient(_app()) as c:
            asyncio.run(self._semaforo_test.acquire())
            asyncio.run(self._semaforo_test.acquire())  # agota los 2 permisos
            r = self._post(c)
        assert r.status_code == 429, r.text
        assert len(self.store._index) == 0, "no debería haberse creado ningún job sin lugar"

    def test_B2_con_capacidad_libre_admite(self):
        async def _noop(job_id, proyecto, rutas, *, store, executor=None, executor_io=None, semaforo=None, control=None):
            return None

        with patch.object(rutas_mod, "_ejecutar_trabajo", _noop), TestClient(_app()) as c:
            r = self._post(c)
        assert r.status_code == 202, r.text

    def test_MINORA_bandera_no_libera_dos_veces_si_falla_el_registro(self):
        """MINOR-A (ronda 4): la bandera `permiso_transferido` tiene que
        quedar en `True` apenas se crea la tarea -- NO después de
        `add_done_callback`/`job_tasks.register`. Si alguna de esas dos
        llamadas lanza DESPUÉS de crear la tarea, la tarea YA creada va a
        liberar el semáforo ella sola (su propio `finally`) -- si la
        bandera seguía en `False` en ese momento, el `finally` de
        `crear_trabajo()` la liberaría OTRA VEZ: el semáforo sube por
        encima de su capacidad real (medido con la mutación: 2 -> 3)."""
        async def _worker_que_libera(job_id, proyecto, rutas, *, store, executor=None, executor_io=None, semaforo=None, control=None):
            # Simula el contrato real de `_ejecutar_trabajo`: quien lo
            # llama YA adquirió el semáforo, así que este worker (aunque
            # sea un doble falso) lo libera él mismo al terminar.
            (semaforo if semaforo is not None else rutas_mod._SEMAFORO_TRABAJOS).release()

        async def _correr():
            with patch.object(rutas_mod.job_tasks, "register", side_effect=RuntimeError("boom -- register() reventó")), \
                 patch.object(rutas_mod, "_ejecutar_trabajo", _worker_que_libera):
                req = rutas_mod.TrabajoRequest(project_uuid=UUID_PRUEBA, rutas=[])
                with self.assertRaises(RuntimeError):
                    await rutas_mod.crear_trabajo(req, _owned_request())
                await asyncio.sleep(0.05)  # deja correr la tarea ya creada (que se libera sola)

        asyncio.run(_correr())
        assert self._semaforo_test._value <= 2, (
            f"el semáforo subió por encima de su capacidad real (liberado dos veces): "
            f"value={self._semaforo_test._value}"
        )
    def test_R6b_si_falla_programar_la_tarea_no_queda_control_huerfano(self):
        """Ronda 6: el control se registra ANTES de `create_task`. Si
        programar la tarea falla, ningún worker va a correr su `finally`,
        así que `crear_trabajo()` tiene que sacar el control él mismo (y
        devolver el permiso, que ya cubre N-1)."""
        def _create_task_que_falla(coro, *a, **kw):
            coro.close()  # nunca se va a esperar: se cierra para no dejar un "never awaited"
            raise RuntimeError("boom -- create_task")

        async def _correr():
            req = rutas_mod.TrabajoRequest(project_uuid=UUID_PRUEBA, rutas=[])
            with patch.object(rutas_mod.asyncio, "create_task", side_effect=_create_task_que_falla):
                with self.assertRaises(RuntimeError):
                    await rutas_mod.crear_trabajo(req, _owned_request())

        asyncio.run(_correr())
        assert len(self.store._index) == 1, "el job tendría que haberse creado antes del fallo"
        assert rutas_mod._CONTROLES == {}, f"quedó un control huérfano: {rutas_mod._CONTROLES}"
        # `_value`, no `locked()`: con 2 permisos, uno filtrado no lo agota.
        assert self._semaforo_test._value == 2, f"el permiso no volvió: value={self._semaforo_test._value}"

    # -- N-3: cancelación -- forma HTTP (404/409), el executor real va en
    #    el Grupo 1 (test_N3_cancelar_deja_terminar...) --------------------
    def test_N3_cancelar_job_desconocido_404(self):
        with TestClient(_app()) as c:
            r = c.post("/procesamiento/trabajos/no-existe/cancel", headers=_h(IDENTIDAD_PLATAFORMA))
        assert r.status_code == 404, r.text

    def test_N3_cancelar_job_terminal_da_409(self):
        job_id = self.store.create(
            caller="x", capability="y", motor="z", trace_id="t", prompt="p",
            recursion_depth=0,
        )
        self.store.update(job_id, status=JobStatus.COMPLETED.value, finished_at=time.time())
        with TestClient(_app()) as c:
            r = c.post(f"/procesamiento/trabajos/{job_id}/cancel", headers=_h(IDENTIDAD_PLATAFORMA))
        assert r.status_code == 409, r.text

    def test_N3_cancelar_marca_cancelling_no_cancelled_de_una(self):
        """La ruta HTTP en sí (sin executor real detrás, `_ejecutar_trabajo`
        mockeado como algo que nunca termina) tiene que pasar por
        `CANCELLING` -- nunca saltar directo a `CANCELLED` (eso mentiría
        sobre hilos que todavía no terminaron)."""
        async def _nunca_termina(job_id, proyecto, rutas, *, store, executor=None, executor_io=None, semaforo=None, control=None):
            await asyncio.sleep(10)

        with patch.object(rutas_mod, "_ejecutar_trabajo", _nunca_termina), TestClient(_app()) as c:
            job_id = self._post(c).json()["job_id"]
            time.sleep(0.05)
            r = c.post(f"/procesamiento/trabajos/{job_id}/cancel", headers=_h(IDENTIDAD_PLATAFORMA))
            assert r.status_code == 200, r.text
            assert r.json()["estado"] == "cancelling", r.json()

    def test_MAJOR1_cancelar_por_http_con_worker_real_termina_cancelled(self):
        """MAJOR-1 (ronda 4): ninguno de los tests anteriores prueba la
        cancelación de punta a punta -- los de N-3 llaman a
        `control.cancelar()` a mano (nunca pasan por la ruta HTTP), y el
        de arriba mockea `_ejecutar_trabajo` entero (nunca hay un
        `_ControlTrabajo` real registrado). Si `cancelar_trabajo()`
        cambiara `control.cancelar()` por un `pass`, TODOS esos tests
        seguirían en verde -- este es el único que pasa por la ruta HTTP
        real CON el worker real corriendo sobre un executor de un hilo
        real (`time.sleep` bloqueante de verdad, no un `asyncio.sleep`
        que se pueda saltear -- mismo criterio que N-3)."""
        with tempfile.TemporaryDirectory() as d:
            workspace = Path(d) / "workspace"
            workspace.mkdir()
            (workspace / "lento.pdf").write_bytes(b"x")

            executor_1_hilo = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-major1-ocr")
            executor_io = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-major1-io")

            def _ingerir_lento(origen, trabajo, *, subruta=None):
                time.sleep(0.3)  # bloqueante DE VERDAD, en el hilo real
                return Ficha(
                    sha256="9" * 64, origen="fuente/x", extractor="pdf",
                    extractor_version="1", fecha="2026-09-21T00:00:00+00:00",
                    estado="ok", detalle={},
                )

            with patch.object(tool_authority, "WORKSPACE_ROOT", workspace.resolve()), \
                 patch.object(rutas_mod, "_EXECUTOR_OCR", executor_1_hilo), \
                 patch.object(rutas_mod, "_EXECUTOR_IO", executor_io), \
                 patch.object(rutas_mod.ingesta, "ingerir", side_effect=_ingerir_lento), \
                 TestClient(_app()) as c:
                job_id = self._post(c, rutas=["lento.pdf"]).json()["job_id"]
                time.sleep(0.08)  # deja que el hilo arranque de verdad
                r = c.post(f"/procesamiento/trabajos/{job_id}/cancel", headers=_h(IDENTIDAD_PLATAFORMA))
                assert r.status_code == 200, r.text
                assert r.json()["estado"] == "cancelling", r.json()

                # espera a que el worker real termine (el hilo en vuelo
                # sigue solo -- ver N-3) sin tocar `asyncio.sleep`
                for _ in range(100):
                    if self.store.get(job_id).status in (
                        JobStatus.CANCELLED, JobStatus.COMPLETED, JobStatus.FAILED,
                    ):
                        break
                    time.sleep(0.02)

            executor_1_hilo.shutdown(wait=True)
            executor_io.shutdown(wait=True)

        job = self.store.get(job_id)
        assert job.status == JobStatus.CANCELLED, (
            f"cancelar por HTTP con el worker real no terminó en 'cancelled': {job}"
        )

    # -- N-6: el arranque reconcilia -----------------------------------------
    def test_N6_arranque_llama_a_reconciliar_trabajos_huerfanos(self):
        """MINOR-E (ronda 4): no alcanza con que la llamada EXISTA en
        algún lado del archivo -- tiene que estar DENTRO de una función
        decorada con `@app.on_event("startup")`. El chequeo viejo
        (`ast.walk` sobre TODO el árbol) seguía en verde si la llamada se
        movía a un hook de `shutdown` -- visto en rojo reproduciendo
        exactamente ese movimiento a mano."""
        server_py = Path(__file__).resolve().parent / "server.py"
        arbol = ast.parse(server_py.read_text(encoding="utf-8"))

        def _es_on_event(dec: ast.expr, evento: str) -> bool:
            return (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute) and dec.func.attr == "on_event"
                and len(dec.args) == 1 and isinstance(dec.args[0], ast.Constant)
                and dec.args[0].value == evento
            )

        def _llama_a_reconciliar(func: ast.AST) -> bool:
            return any(
                isinstance(n, ast.Call) and getattr(n.func, "id", None) == "reconciliar_trabajos_huerfanos"
                for n in ast.walk(func)
            )

        def _funciones_con_hook(evento: str) -> list[ast.AST]:
            return [
                n for n in ast.walk(arbol)
                if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
                and any(_es_on_event(d, evento) for d in n.decorator_list)
            ]

        funciones_startup = _funciones_con_hook("startup")
        funciones_shutdown = _funciones_con_hook("shutdown")

        assert funciones_startup, "server.py no tiene ningún hook @app.on_event('startup')"
        assert any(_llama_a_reconciliar(f) for f in funciones_startup), (
            "reconciliar_trabajos_huerfanos() no está dentro de ningún hook de ARRANQUE"
        )
        assert not any(_llama_a_reconciliar(f) for f in funciones_shutdown), (
            "reconciliar_trabajos_huerfanos() está en un hook de APAGADO, no de arranque"
        )
    def test_N2r5_get_devuelve_los_nombres_acentuados_intactos(self):
        """N2 (ronda 5), por HTTP: lo que ve jax-platform. `factura_año.pdf`
        tiene que volver como `factura_año.pdf`, no como
        `factura_a\\xf1o.pdf`."""
        job_id = self.store.create(
            caller="x", capability="y", motor="z", trace_id="t", prompt="p",
            recursion_depth=0,
        )
        resultados = [
            {"archivo": "Constitución de sociedad.pdf", "estado": "ok", "extractor": "pdf",
             "extracto_bytes": 42, "carpeta_procesado": "proyectos/remodelacion/procesado/abc",
             "error": None},
            {"archivo": "factura_año.pdf", "estado": "error", "extractor": None,
             "extracto_bytes": 0, "carpeta_procesado": None,
             "error": "no se pudo abrir 'Crédito Ñandú.xlsx'"},
        ]
        result_path = rutas_mod._guardar_resultado(self.store, job_id, resultados)
        self.store.update(
            job_id, status=JobStatus.COMPLETED.value, finished_at=time.time(),
            result_path=result_path,
        )

        with TestClient(_app()) as c:
            r = c.get(f"/procesamiento/trabajos/{job_id}", headers=_h(IDENTIDAD_PLATAFORMA))

        assert r.status_code == 200, r.text
        assert r.json()["resultados"] == resultados, r.json()["resultados"]

    # -- N5 (ronda 5): el arranque EJECUTA la reconciliación ----------------
    def test_N5r5_el_arranque_ejecuta_reconciliar_trabajos_huerfanos(self):
        """N5 (ronda 5): el test AST de N-6 acepta la llamada dentro de un
        `if False:` -- aparece en el árbol, dentro del hook de arranque, y
        nunca corre. Este test corre DE VERDAD todos los handlers de
        `startup` que `server.app` registra (con lo que tocaría la base
        reemplazado por dobles) y exige que la reconciliación se haya
        EJECUTADO una vez.

        `_configure_b7_trusted_runtime()` corre en el MISMO handler de
        arranque (jax#250) y exige `JAX_DEPLOYMENT_ID`/`JAX_DB_HOST`/
        `JAX_DB_PORT` y MariaDB real -- infraestructura ajena a lo que este
        test mide. Se neutraliza esa exigencia puntual (no el chequeo B7 en
        sí, que sigue intacto en producción) para poder seguir ejecutando el
        resto de los handlers reales, reconciliar_trabajos_huerfanos()
        incluido."""
        import server

        reconciliar = Mock(return_value=0)

        async def _arrancar():
            assert server.app.router.on_startup, "LAS MANOS no registra ningún handler de startup"
            for handler in server.app.router.on_startup:
                await handler()

        with patch.object(server, "_configure_b7_trusted_runtime"), \
             patch("jacobs.subpipelines.config_subpipelines"), \
             patch.object(server.jacobs_store, "tamanio_pool"), \
             patch.object(server.jacobs_store, "init_tables", AsyncMock()), \
             patch("procesamiento_idempotencia.init_tabla", AsyncMock()), \
             patch("motor_registry.routes.init_motor_catalog", AsyncMock()), \
             patch("jacobs.reaper.reap_orphaned_pipelines", AsyncMock()), \
             patch("jacobs.reaper.start_reaper_loop", AsyncMock()), \
             patch.object(rutas_mod, "reconciliar_trabajos_huerfanos", reconciliar):
            asyncio.run(_arrancar())

        reconciliar.assert_called_once_with()

    # -- autenticación --------------------------------------------------
    def test_credencial_plataforma_accede_jacobs_no(self):
        with TestClient(_app()) as c:
            r_plat = self._post(c)
            r_jacobs = c.post(
                "/procesamiento/trabajos",
                json={"project_uuid": UUID_PRUEBA, "rutas": [], "usuario": "a"},
                headers=_h(IDENTIDAD_JACOBS),
            )
            r_sin_credencial = c.post(
                "/procesamiento/trabajos", json={"project_uuid": UUID_PRUEBA, "rutas": [], "usuario": "a"},
            )
        assert r_plat.status_code == 202, r_plat.text
        assert r_jacobs.status_code == 403, r_jacobs.text
        assert r_sin_credencial.status_code == 401, r_sin_credencial.text

    def test_credencial_jacobs_no_puede_cancelar(self):
        job_id = self.store.create(
            caller="x", capability="y", motor="z", trace_id="t", prompt="p",
            recursion_depth=0,
        )
        with TestClient(_app()) as c:
            r = c.post(f"/procesamiento/trabajos/{job_id}/cancel", headers=_h(IDENTIDAD_JACOBS))
        assert r.status_code == 403, r.text


# ===========================================================================
#  Grupo 3 -- E2a: LAS MANOS trabaja por `project_uuid`, solo proyectos ACTIVE
# ===========================================================================
class ProjectUuidTest(unittest.TestCase):
    UUID = UUID_PRUEBA

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.store = _ProcessingJobStoreFixture(str(Path(self._tmpdir.name) / "jobs.jsonl"))
        self.semaforo = asyncio.Semaphore(2)
        for parche in (
            patch.object(rutas_mod, "_STORE", self.store),
            patch.object(rutas_mod, "_SEMAFORO_TRABAJOS", self.semaforo),
            patch.dict(rutas_mod._CONTROLES, clear=True),
        ):
            parche.start()
            self.addCleanup(parche.stop)
        self.cabeceras_plataforma = _h(IDENTIDAD_PLATAFORMA)

    def _post(self, cuerpo, estado="ACTIVE"):
        with patch.object(rutas_mod.proyecto_activo, "identidad_activa_del_proyecto", AsyncMock(return_value=estado == "ACTIVE")), \
             patch.object(rutas_mod, "_ejecutar_trabajo", AsyncMock()), \
             TestClient(_app()) as c:
            return c.post("/procesamiento/trabajos", json=cuerpo, headers=self.cabeceras_plataforma)

    def test_proyecto_viejo_por_nombre_da_422(self):
        r = self._post({"proyecto": "lacteos", "rutas": ["a.pdf"]})
        self.assertEqual(r.status_code, 422)

    def test_uuid_no_canonico_da_422_con_codigo(self):
        # 36 caracteres, pasa el largo de pydantic, no es un UUID.
        for malo in ("../../etc" + "x" * 27, self.UUID.upper(), "z" * 36):
            r = self._post({"project_uuid": malo, "rutas": ["a.pdf"]})
            self.assertEqual(r.status_code, 422, malo)
            self.assertEqual(r.json()["detail"]["code"], "project_uuid_invalido", malo)

    def test_uuid_de_otro_largo_da_422(self):
        r = self._post({"project_uuid": "../../etc", "rutas": ["a.pdf"]})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["detail"]["code"], "project_uuid_invalido")

    def test_base_caida_da_503_sin_job_ni_cupo(self):
        # Si `estado_del_proyecto` lanza (base caida, timeout) no es un 500 anonimo ni un
        # proyecto "no activo": es 503 con codigo estable, sin job creado y sin tomar cupo.
        with patch.object(rutas_mod.proyecto_activo, "identidad_activa_del_proyecto",
                          AsyncMock(side_effect=ConnectionError("base caida"))), \
             patch.object(rutas_mod, "_ejecutar_trabajo", AsyncMock()) as ejecutar, \
             TestClient(_app()) as c:
            r = c.post("/procesamiento/trabajos",
                       json={"project_uuid": self.UUID, "rutas": ["a.pdf"]},
                       headers=self.cabeceras_plataforma)
        self.assertEqual(r.status_code, 503, r.text)
        self.assertEqual(r.json()["detail"]["code"], "base_no_disponible")
        self.assertFalse(self.semaforo.locked())
        self.assertEqual(self.semaforo._value, 2)      # no tomo ningun permiso
        self.assertEqual(self.store._index, {})        # no creo job
        ejecutar.assert_not_called()

    def test_trabajo_de_rechaza_lo_que_no_es_uuid_canonico(self):
        for malo in ("p", "../../etc", self.UUID.upper(), "medicion-aislamiento", ""):
            with self.assertRaises(ValueError, msg=malo):
                rutas_mod._trabajo_de(malo)

    def test_proyecto_archivado_da_422_y_no_toma_cupo(self):
        r = self._post({"project_uuid": self.UUID, "rutas": ["a.pdf"]}, estado="ARCHIVED")
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["detail"]["code"], "proyecto_no_activo")
        self.assertFalse(self.semaforo.locked())
        self.assertEqual(self.semaforo._value, 2)
        self.assertEqual(len(self.store._index), 0, "no debería haberse creado ningún job")

    def test_proyecto_oculto_o_deshabilitado_da_422(self):
        for estado in ("HIDDEN", "DISABLED"):
            r = self._post({"project_uuid": self.UUID, "rutas": ["a.pdf"]}, estado=estado)
            self.assertEqual(r.status_code, 422, estado)
            self.assertEqual(r.json()["detail"]["code"], "proyecto_no_activo", estado)

    def test_proyecto_inexistente_da_422(self):
        r = self._post({"project_uuid": self.UUID, "rutas": ["a.pdf"]}, estado=None)
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["detail"]["code"], "proyecto_no_activo")

    def test_proyecto_activo_admite_y_guarda_el_uuid(self):
        r = self._post({"project_uuid": self.UUID, "rutas": ["a.pdf"]})
        self.assertEqual(r.status_code, 202, r.text)
        self.assertEqual(self.store._index[r.json()["job_id"]]["proyecto"], self.UUID)

    def test_carpeta_del_trabajo_es_el_uuid(self):
        self.assertEqual(rutas_mod._trabajo_de(self.UUID),
                         tool_authority.WORKSPACE_ROOT / "proyectos" / self.UUID)


# -- jax-14 ronda 2, MINOR-N2: las variables de intervalo nunca tumban el import ----------
def _recargar_aviso(**entorno):
    with patch.dict(os.environ, entorno):
        return importlib.reload(aviso_extractores)


@pytest.fixture
def _aviso_restaurado():
    yield
    importlib.reload(aviso_extractores)  # entorno limpio: vuelven los valores por defecto


def test_valor_no_numerico_usa_el_default_sin_lanzar_y_el_import_no_cae(_aviso_restaurado, caplog):
    with caplog.at_level("WARNING", logger="las_manos.aviso_extractores"):
        mod = _recargar_aviso(JAX_PROCESAMIENTO_AVISO_EXTRACTORES_S="1h", JAX_PROCESAMIENTO_LOG_EXTRACTORES_S="abc")
    assert mod.INTERVALO_AVISO_S == 3600.0
    assert mod.INTERVALO_LOG_S == 60.0
    avisos = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("JAX_PROCESAMIENTO_AVISO_EXTRACTORES_S" in a for a in avisos), avisos


@pytest.mark.parametrize("valor", ["0", "-5", "nan", "inf", "-inf"])
def test_cero_negativo_o_no_finito_se_acota_al_minimo_o_al_default(_aviso_restaurado, valor):
    mod = _recargar_aviso(JAX_PROCESAMIENTO_AVISO_EXTRACTORES_S=valor, JAX_PROCESAMIENTO_LOG_EXTRACTORES_S=valor)
    assert mod.INTERVALO_AVISO_S >= 60.0
    assert mod.INTERVALO_LOG_S >= 10.0
    assert mod.INTERVALO_AVISO_S < float("inf") and mod.INTERVALO_LOG_S < float("inf")


def test_los_minimos_son_60_para_el_aviso_y_10_para_el_log(_aviso_restaurado):
    mod = _recargar_aviso(JAX_PROCESAMIENTO_AVISO_EXTRACTORES_S="1", JAX_PROCESAMIENTO_LOG_EXTRACTORES_S="1")
    assert (mod.INTERVALO_AVISO_S, mod.INTERVALO_LOG_S) == (60.0, 10.0)


def test_un_valor_enorme_se_baja_al_techo_con_warning(_aviso_restaurado, caplog):
    """MINOR-N5: sin techo, 1e28 (o una confusion de unidades) apagaria el aviso del POST
    mientras viva el proceso."""
    with caplog.at_level("WARNING", logger="las_manos.aviso_extractores"):
        mod = _recargar_aviso(JAX_PROCESAMIENTO_AVISO_EXTRACTORES_S="1e28", JAX_PROCESAMIENTO_LOG_EXTRACTORES_S="1e28")
    assert (mod.INTERVALO_AVISO_S, mod.INTERVALO_LOG_S) == (86400.0, 3600.0)
    avisos = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("JAX_PROCESAMIENTO_AVISO_EXTRACTORES_S" in a for a in avisos), avisos
    assert any("JAX_PROCESAMIENTO_LOG_EXTRACTORES_S" in a for a in avisos), avisos


def test_un_valor_valido_por_encima_del_minimo_se_respeta(_aviso_restaurado):
    mod = _recargar_aviso(JAX_PROCESAMIENTO_AVISO_EXTRACTORES_S="7200", JAX_PROCESAMIENTO_LOG_EXTRACTORES_S="30")
    assert (mod.INTERVALO_AVISO_S, mod.INTERVALO_LOG_S) == (7200.0, 30.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
