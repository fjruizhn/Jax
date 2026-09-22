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
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import procesamiento_routes as rutas_mod
from auth_servicio import ENCABEZADO, IDENTIDAD_JACOBS, IDENTIDAD_PLATAFORMA, proteger
from motor_registry import tool_authority
from motor_registry.job_store import JobStore
from motor_registry.models import JobStatus
from procesamiento.ficha import Ficha
from procesamiento_routes import router


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
        self.store = JobStore(str(Path(self._tmpdir.name) / "jobs.jsonl"))
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
            await self._ejecutar(job_id, "Proyecto Uno", ["doc.pdf"])

        job = self.store.get(job_id)
        assert job.status == JobStatus.COMPLETED, job
        resultados = json.loads(Path(job.result_path).read_text())
        assert len(resultados) == 1
        r = resultados[0]
        assert r["estado"] == "ok"
        assert r["extractor"] == "pdf"
        assert r["extracto_bytes"] > 0
        assert r["carpeta_procesado"] == f"proyectos/proyecto-uno/procesado/{'a' * 64}"
        assert "contenido secreto" not in json.dumps(r)
        assert set(r) == {"archivo", "estado", "extractor", "extracto_bytes", "carpeta_procesado", "error"}

    async def test_M1_ruta_fuera_del_jail_se_rechaza_sin_llamar_a_ingerir(self):
        ingerir_mock = AsyncMock()
        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", ingerir_mock):
            await self._ejecutar(
                job_id, "proy", ["/etc/passwd", "../fuera-del-workspace.txt"],
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
                job_id, "proy", ["sano1.pdf", "archivo\x00malo.pdf", "sano2.pdf"],
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
            await self._ejecutar(job_id, "proy", ["uno.pdf", "dos.pdf", "tres.pdf"])

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
            await self._ejecutar(job_id, "proy", ["sano1.pdf", ruta_mala, "sano2.pdf"])

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
                    job_id, "p", ["a.pdf"], store=self.store,
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
            await self._ejecutar(job_id, "p", ["a.pdf"])

        assert vistos == [JobStatus.RUNNING.value, JobStatus.COMPLETED.value], vistos

    async def test_ruta_ausente_dentro_del_jail_se_rechaza_no_revienta(self):
        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir") as ingerir_mock:
            await self._ejecutar(job_id, "proy", ["no-existe.pdf"])
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
            await self._ejecutar(job_id, "p", ["lento.pdf"])
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
            await self._ejecutar(job_id, "p", ["p1.pdf", "p2.pdf"])
            dt = time.perf_counter() - t0

        assert dt < 0.5, f"tardó {dt:.2f}s -- no se repartió entre los dos hilos disponibles"

    # -- outer except: nunca deja running huérfano -----------------------
    async def test_una_excepcion_catastrofica_marca_failed_no_deja_running(self):
        self._archivo_en_workspace("a.pdf")
        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", return_value=_ficha("9" * 64)), \
             patch.object(self.store, "write_result", side_effect=RuntimeError("disco lleno")):
            await self._ejecutar(job_id, "p", ["a.pdf"])

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
            await self._ejecutar(job_id, "p", ["a.pdf"], semaforo=semaforo)

        assert not semaforo.locked(), "el semáforo quedó tomado tras un trabajo que falló"

    async def test_semaforo_se_libera_al_completar(self):
        semaforo = asyncio.Semaphore(1)
        self._archivo_en_workspace("a.pdf")
        job_id = self._crear_job()
        with patch.object(rutas_mod.ingesta, "ingerir", return_value=_ficha("2" * 64)):
            await self._ejecutar(job_id, "p", ["a.pdf"], semaforo=semaforo)

        assert not semaforo.locked()

    # -- B-6: el principal se registra ------------------------------------
    async def test_B6_usuario_llega_al_job_como_caller(self):
        job_id = self.store.create(
            caller="ana@cliente.com", capability="ingesta_archivos",
            motor="n/a", trace_id="t", prompt="n/a", recursion_depth=0,
        )
        with patch.object(rutas_mod.ingesta, "ingerir", return_value=_ficha("3" * 64)):
            self._archivo_en_workspace("a.pdf")
            await self._ejecutar(job_id, "p", ["a.pdf"])
        assert self.store.get(job_id).caller == "ana@cliente.com"

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
            await self._ejecutar(job_id, "p", ["a.pdf"], executor=espia)
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
            await self._ejecutar(job_id, "p", ["a.pdf"], executor_io=espia)
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
                    job_id, "p", ["primero.pdf", "segundo.pdf"], store=self.store,
                    executor=executor_1_hilo, executor_io=self.executor_io, semaforo=semaforo,
                )
            )
            await asyncio.sleep(0.08)  # deja que "primero" arranque de VERDAD en el hilo
            assert self.store.get(job_id).status == JobStatus.RUNNING
            assert terminados == [], "no debería haber terminado todavía"

            # Mismo mecanismo que el endpoint de cancelación, sin pasar por HTTP.
            control = rutas_mod._CONTROLES[job_id]
            self.store.update(job_id, status=JobStatus.CANCELLING.value)
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
                    job_id, "p", ["nunca.pdf"], store=self.store,
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
    return {ENCABEZADO: CRED[identidad]} if identidad else {}


class TrabajoHTTPTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.store = JobStore(str(Path(self._tmpdir.name) / "jobs.jsonl"))
        self._parche_store = patch.object(rutas_mod, "_STORE", self.store)
        self._parche_store.start()
        self.addCleanup(self._parche_store.stop)

        self._semaforo_test = asyncio.Semaphore(2)
        self._parche_semaforo = patch.object(rutas_mod, "_SEMAFORO_TRABAJOS", self._semaforo_test)
        self._parche_semaforo.start()
        self.addCleanup(self._parche_semaforo.stop)

    def _post(self, c, **overrides):
        cuerpo = {"proyecto": "p", "rutas": ["a.pdf"], "usuario": "ana@cliente.com"}
        cuerpo.update(overrides)
        return c.post("/procesamiento/trabajos", json=cuerpo, headers=_h(IDENTIDAD_PLATAFORMA))

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
                    req = rutas_mod.TrabajoRequest(proyecto="p", rutas=[], usuario="ana@cliente.com")
                    with self.assertRaises(RuntimeError):
                        await rutas_mod.crear_trabajo(req)

            assert not self._semaforo_test.locked(), (
                "el semáforo quedó agotado tras dos pedidos rotos -- DoS de N requests"
            )
            # y un pedido LIMPIO subsiguiente se admite normalmente
            req_limpio = rutas_mod.TrabajoRequest(proyecto="p", rutas=[], usuario="ana@cliente.com")
            with patch.object(rutas_mod, "_ejecutar_trabajo", AsyncMock()):
                respuesta = await rutas_mod.crear_trabajo(req_limpio)
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
        proyecto_malo = "p\udcff"
        cuerpo_json = json.dumps(
            {"proyecto": proyecto_malo, "rutas": [], "usuario": "ana@cliente.com"},
            ensure_ascii=True,
        ).encode("ascii")
        with TestClient(_app(), raise_server_exceptions=False) as c:
            r = c.post(
                "/procesamiento/trabajos", content=cuerpo_json,
                headers={**_h(IDENTIDAD_PLATAFORMA), "content-type": "application/json"},
            )
            assert r.status_code == 422, r.text
            assert not self._semaforo_test.locked(), (
                "el semáforo se tocó aunque el proyecto se rechazó en la admisión"
            )
            assert len(self.store._index) == 0, "no debería haberse creado ningún job"

    # -- B-5 / no bloqueo --------------------------------------------------
    def test_post_no_espera_a_que_el_trabajo_termine(self):
        async def _lento(job_id, proyecto, rutas, *, store, executor=None, executor_io=None, semaforo=None):
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
    def test_B6_usuario_obligatorio_falta_es_422(self):
        with TestClient(_app()) as c:
            r = c.post(
                "/procesamiento/trabajos", json={"proyecto": "p", "rutas": []},
                headers=_h(IDENTIDAD_PLATAFORMA),
            )
        assert r.status_code == 422, r.text

    def test_B6_usuario_vacio_es_422(self):
        with TestClient(_app()) as c:
            r = self._post(c, usuario="")
        assert r.status_code == 422, r.text

    def test_B6_usuario_muy_largo_es_422(self):
        with TestClient(_app()) as c:
            r = self._post(c, usuario="x" * 300)
        assert r.status_code == 422, r.text

    def test_B6_usuario_se_registra_como_caller_del_job(self):
        async def _noop(job_id, proyecto, rutas, *, store, executor=None, executor_io=None, semaforo=None):
            return None

        with patch.object(rutas_mod, "_ejecutar_trabajo", _noop), TestClient(_app()) as c:
            r = self._post(c, usuario="carlos@cliente.com")
            job_id = r.json()["job_id"]

        assert self.store.get(job_id).caller == "carlos@cliente.com"

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
        async def _noop(job_id, proyecto, rutas, *, store, executor=None, executor_io=None, semaforo=None):
            return None

        with patch.object(rutas_mod, "_ejecutar_trabajo", _noop), TestClient(_app()) as c:
            r = self._post(c)
        assert r.status_code == 202, r.text

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
        async def _nunca_termina(job_id, proyecto, rutas, *, store, executor=None, executor_io=None, semaforo=None):
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

    # -- autenticación --------------------------------------------------
    def test_credencial_plataforma_accede_jacobs_no(self):
        with TestClient(_app()) as c:
            r_plat = self._post(c)
            r_jacobs = c.post(
                "/procesamiento/trabajos",
                json={"proyecto": "p", "rutas": [], "usuario": "a"},
                headers=_h(IDENTIDAD_JACOBS),
            )
            r_sin_credencial = c.post(
                "/procesamiento/trabajos", json={"proyecto": "p", "rutas": [], "usuario": "a"},
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
