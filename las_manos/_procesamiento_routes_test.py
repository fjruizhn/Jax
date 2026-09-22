"""
LAS MANOS -- endpoint de Procesamiento de Archivos (2026-09-21, ronda de
arreglo tras revisión adversarial -- ver
`.superpowers/sdd/2026-09-20-procesamiento-archivos-nucleo/endpoint-hallazgos.md`
y `endpoint-report.md`).

Corre en "tests-puros": `procesamiento.ingesta.ingerir` va SUSTITUIDO acá --
sin pdftoppm/tesseract/libreoffice reales (esos ya los cubre
`procesamiento/_ingesta_test.py`, jax#252). Cada archivo de esta suite es
la evidencia de UN hallazgo de la ronda anterior:

  - B-1: una ruta con byte NUL entre sanas no tumba el lote (la causa raíz
    se arregló en `las_manos/_tool_authority_test.py`, esto prueba el
    efecto en el endpoint).
  - B-2: admisión (429/422), y el trabajo NO monopoliza el executor por
    defecto (se mide el RETRASO DEL LOOP mientras el trabajo corre real, no
    el retorno del POST -- B-5).
  - B-3: reconciliación al arrancar + cancelación en vuelo.
  - B-4: `GET` no bloquea el loop leyendo el resultado.
  - B-5: las cuatro mutaciones "fire-and-forget sin red" tienen que morir:
    sacar el `run_in_executor`, sacar el `update(FAILED)` del except, sacar
    `job_tasks.register()`, sacar el `add_done_callback`.
  - B-6: `usuario` es obligatorio y se registra como `caller` del job.

Dos grupos:

  - `TrabajoWorkerTest` (asíncrono, `IsolatedAsyncioTestCase`): llama
    `_ejecutar_trabajo`/`reconciliar_trabajos_huerfanos` DIRECTO, con un
    `JobStore` de tempdir propio y un executor/semáforo PEQUEÑOS y propios
    (nunca los módulo-globales -- aislamiento entre tests).
  - `TrabajoHTTPTest` (síncrono, `TestClient`): la forma HTTP -- 202/404/
    409/422/429, y la credencial de servicio.

Corre con:
  PYTHONPATH=.:las_manos python -m pytest -v las_manos/_procesamiento_routes_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import tempfile
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

        # Executor/semáforo PROPIOS de este test -- nunca los
        # módulo-globales de procesamiento_routes (aislamiento).
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test")
        self.addCleanup(self.executor.shutdown)
        self.semaforo = asyncio.Semaphore(2)

    def _archivo_en_workspace(self, nombre: str, contenido: bytes = b"x") -> str:
        (self.workspace / nombre).write_bytes(contenido)
        return nombre  # ruta RELATIVA -- contrato de resolve_jailed_path

    async def _ejecutar(self, job_id, proyecto, rutas, *, semaforo=None, executor=None):
        """Mismo contrato que `crear_trabajo()`: adquiere el semáforo ANTES
        de llamar -- `_ejecutar_trabajo` lo libera en su `finally`."""
        semaforo = semaforo if semaforo is not None else self.semaforo
        executor = executor if executor is not None else self.executor
        await semaforo.acquire()
        await rutas_mod._ejecutar_trabajo(
            job_id, proyecto, rutas, store=self.store, executor=executor, semaforo=semaforo,
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

    # -- B-1 (raíz en tool_authority; esto prueba el EFECTO en el endpoint) --
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
        """Evidencia exigida para B-1 a nivel endpoint: antes del fix en
        `tool_authority.py`, un byte NUL en CUALQUIER ruta del lote hacía
        que `resolve_jailed_path` dejara escapar `ValueError`, y esa
        excepción no la atrapaba nada en `_procesar_una_ruta` -- el job
        entero terminaba `failed`, con CERO resultados, sin decir cuál
        archivo fue."""
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
        assert job.status == JobStatus.COMPLETED, (
            f"un byte NUL tumbó el lote entero: {job}"
        )
        resultados = json.loads(Path(job.result_path).read_text())
        assert len(resultados) == 3, "el lote no siguió con los demás archivos"
        por_archivo = {r["archivo"]: r for r in resultados}
        assert por_archivo["sano1.pdf"]["estado"] == "ok"
        assert por_archivo["sano2.pdf"]["estado"] == "ok"
        assert por_archivo["archivo\x00malo.pdf"]["estado"] == "rechazado"

    # -- M2: un archivo roto no tumba el lote ----------------------------
    async def test_M2_un_archivo_roto_no_tumba_el_lote(self):
        for nombre in ("uno.pdf", "dos.pdf", "tres.pdf"):
            self._archivo_en_workspace(nombre)

        # B-2: los archivos ahora se procesan EN PARALELO (cada uno en su
        # propio hilo del executor) -- un side_effect que hace `pop(0)` de
        # una lista compartida asume orden secuencial y es una carrera de
        # verdad entre hilos. Se decide por NOMBRE, no por orden de
        # llamada.
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
        assert len(resultados) == 3, "el lote no siguió con los demás archivos"
        por_archivo = {r["archivo"]: r for r in resultados}
        assert por_archivo["uno.pdf"]["estado"] == "ok"
        assert por_archivo["dos.pdf"]["estado"] == "error"
        assert "pdftoppm: timeout" in por_archivo["dos.pdf"]["error"]
        assert por_archivo["tres.pdf"]["estado"] == "ok"

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
        """La mutación que sacaría `run_in_executor` (o lo cambiara por una
        llamada directa) NO rompía ningún test viejo: medían el retorno del
        POST, y `create_task` difiere el cuerpo del trabajo hasta el
        siguiente `await`, así que el POST volvía rápido aunque el OCR
        corriera entero DENTRO del loop. Este test mide el loop mismo,
        mientras el trabajo (un `time.sleep` REAL, bloqueante, en el hilo)
        está en vuelo."""
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
            # IMPORTANTE (aprendido de B-5 en la revisión): si el sondeo se
            # arranca con `asyncio.gather(trabajo, sondeo)` y el trabajo
            # bloquea ANTES de su primer `await`, el sondeo ni siquiera
            # llegó a arrancar su primer `sleep(0.01)` cuando el bloqueo ya
            # pasó -- el test "medía" un intervalo que nunca coincidió con
            # el bloqueo real y pasaba igual con la mutación que saca el
            # `run_in_executor` (falso negativo, confirmado a mano). Por
            # eso el sondeo se crea PRIMERO, como tarea aparte, y se le da
            # tiempo real de arrancar y quedar DENTRO de su `sleep(0.01)`
            # antes de lanzar el trabajo.
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
        """B-2: antes, `_procesar_rutas` metía el LOTE ENTERO en un único
        `to_thread` -- un job de muchos archivos no se podía repartir ni
        con un pool más grande. Con dos archivos de 0,3s cada uno y dos
        hilos disponibles, el trabajo tiene que tardar ~0,3s (en paralelo),
        no ~0,6s (uno detrás del otro en el mismo hilo)."""
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
        """Ataca el `except Exception` de `_ejecutar_trabajo` (no el de
        `_procesar_una_ruta`): si `store.write_result` revienta, el job
        tiene que quedar `failed` con el motivo -- si se borrara el
        `update(FAILED, ...)` de ese except, este test vería el job
        atascado en `running` para siempre."""
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
        """No se ejercita acá el HTTP -- ver `TrabajoHTTPTest` para el
        camino completo del POST -- esto confirma que `_ejecutar_trabajo`
        no toca el `caller` que ya viene puesto en `create()` (lo pone
        `crear_trabajo()`, que es lo que se verifica del lado HTTP)."""
        job_id = self.store.create(
            caller="ana@cliente.com", capability="ingesta_archivos",
            motor="n/a", trace_id="t", prompt="n/a", recursion_depth=0,
        )
        with patch.object(rutas_mod.ingesta, "ingerir", return_value=_ficha("3" * 64)):
            self._archivo_en_workspace("a.pdf")
            await self._ejecutar(job_id, "p", ["a.pdf"])
        assert self.store.get(job_id).caller == "ana@cliente.com"

    # -- B-3: reconciliación al arrancar -----------------------------------
    def test_B3_reconciliar_marca_failed_lo_que_quedo_pending_o_running(self):
        j_pending = self._crear_job()  # create() deja status=pending
        j_running = self._crear_job()
        self.store.update(j_running, status=JobStatus.RUNNING.value)
        j_completado = self._crear_job()
        self.store.update(j_completado, status=JobStatus.COMPLETED.value, finished_at=time.time())

        n = rutas_mod.reconciliar_trabajos_huerfanos(self.store)

        assert n == 2, n
        assert self.store.get(j_pending).status == JobStatus.FAILED
        assert self.store.get(j_pending).error
        assert self.store.get(j_running).status == JobStatus.FAILED
        assert self.store.get(j_completado).status == JobStatus.COMPLETED, (
            "reconciliar tocó un trabajo que ya había terminado"
        )

    def test_B3_reconciliar_no_hace_nada_sin_huerfanos(self):
        j = self._crear_job()
        self.store.update(j, status=JobStatus.COMPLETED.value, finished_at=time.time())
        assert rutas_mod.reconciliar_trabajos_huerfanos(self.store) == 0
        assert self.store.get(j).status == JobStatus.COMPLETED

    # -- B-4: GET no bloquea el loop leyendo el resultado -------------------
    async def test_B4_leer_resultado_no_bloquea_el_loop(self):
        """Medido en la revisión con 200.000 rutas: 0,39s con LAS MANOS
        entero congelado. Acá se simula el mismo bloqueo con un
        `time.sleep` real y se mide el loop mientras `_construir_respuesta_
        estado` corre -- tiene que seguir respondiendo."""
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
            # Mismo motivo que en test_B5_el_trabajo_no_bloquea_el_loop_de_
            # eventos: el sondeo tiene que estar YA dentro de su primer
            # `sleep(0.01)` antes de disparar la lectura -- si no, una
            # lectura que bloquea ANTES de su primer `await` termina antes
            # de que el sondeo llegue a arrancar, y el test pasa igual con
            # la mutación que saca el `run_in_executor` (falso negativo).
            sondeo = asyncio.create_task(_sondear_loop())
            await asyncio.sleep(0.02)
            await rutas_mod._construir_respuesta_estado("job-x", _VistaFalsa())
            await sondeo

        peor = max(retrasos)
        assert peor < 0.15, (
            f"el loop se retrasó {peor:.3f}s leyendo el resultado -- "
            "la lectura no está corriendo fuera del loop"
        )


# ===========================================================================
#  Grupo 2 -- HTTP: admisión, cancelación, no-bloqueo, autenticación
# ===========================================================================
CRED = {
    IDENTIDAD_PLATAFORMA: secrets.token_urlsafe(32),
    IDENTIDAD_JACOBS: secrets.token_urlsafe(32),
}


def _credenciales() -> dict[str, bytes]:
    return {k: v.encode("ascii") for k, v in CRED.items()}


def _app() -> FastAPI:
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

        # Semáforo PROPIO por test -- el módulo-global se compartiría entre
        # TODOS los tests de esta clase (y entre corridas), y un test que
        # reemplaza `_ejecutar_trabajo` por un mock que nunca libera de
        # verdad dejaría el semáforo agotado para el resto.
        self._semaforo_test = asyncio.Semaphore(2)
        self._parche_semaforo = patch.object(rutas_mod, "_SEMAFORO_TRABAJOS", self._semaforo_test)
        self._parche_semaforo.start()
        self.addCleanup(self._parche_semaforo.stop)

    def _post(self, c, **overrides):
        cuerpo = {"proyecto": "p", "rutas": ["a.pdf"], "usuario": "ana@cliente.com"}
        cuerpo.update(overrides)
        return c.post("/procesamiento/trabajos", json=cuerpo, headers=_h(IDENTIDAD_PLATAFORMA))

    # -- B-5 / no bloqueo --------------------------------------------------
    def test_post_no_espera_a_que_el_trabajo_termine(self):
        async def _lento(job_id, proyecto, rutas, *, store, executor=None, semaforo=None):
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

    # -- B-6: principal obligatorio y registrado -------------------------
    def test_B6_usuario_obligatorio_falta_es_422(self):
        with TestClient(_app()) as c:
            r = c.post(
                "/procesamiento/trabajos", json={"proyecto": "p", "rutas": []},
                headers=_h(IDENTIDAD_PLATAFORMA),
            )
        assert r.status_code == 422, r.text

    def test_B6_usuario_se_registra_como_caller_del_job(self):
        async def _noop(job_id, proyecto, rutas, *, store, executor=None, semaforo=None):
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
        async def _noop(job_id, proyecto, rutas, *, store, executor=None, semaforo=None):
            return None

        with patch.object(rutas_mod, "_ejecutar_trabajo", _noop), TestClient(_app()) as c:
            r = self._post(c)
        assert r.status_code == 202, r.text

    # -- B-3: cancelación --------------------------------------------------
    def test_B3_cancelar_job_desconocido_404(self):
        with TestClient(_app()) as c:
            r = c.post("/procesamiento/trabajos/no-existe/cancel", headers=_h(IDENTIDAD_PLATAFORMA))
        assert r.status_code == 404, r.text

    def test_B3_cancelar_job_terminal_da_409(self):
        job_id = self.store.create(
            caller="x", capability="y", motor="z", trace_id="t", prompt="p",
            recursion_depth=0,
        )
        self.store.update(job_id, status=JobStatus.COMPLETED.value, finished_at=time.time())
        with TestClient(_app()) as c:
            r = c.post(f"/procesamiento/trabajos/{job_id}/cancel", headers=_h(IDENTIDAD_PLATAFORMA))
        assert r.status_code == 409, r.text

    def test_B3_cancelar_corta_la_tarea_en_vuelo_no_solo_el_registro(self):
        """Catches la mutación 'borrar `job_tasks.register()`': sin
        registrar la tarea, `job_tasks.cancel(job_id)` no encuentra nada
        que cortar y la tarea simulada llegaría a 'completo' igual, aunque
        el store ya diga 'cancelled'."""
        marca: list[str] = []

        async def _lento(job_id, proyecto, rutas, *, store, executor=None, semaforo=None):
            try:
                await asyncio.sleep(5)
                marca.append("completo")
            except asyncio.CancelledError:
                marca.append("cancelado")
                raise

        with patch.object(rutas_mod, "_ejecutar_trabajo", _lento), TestClient(_app()) as c:
            job_id = self._post(c).json()["job_id"]
            time.sleep(0.05)  # deja que la tarea arranque y quede en el sleep(5)
            r = c.post(f"/procesamiento/trabajos/{job_id}/cancel", headers=_h(IDENTIDAD_PLATAFORMA))
            assert r.status_code == 200, r.text
            assert r.json()["estado"] == "cancelled"
            time.sleep(0.05)  # deja que el CancelledError se propague

            # La aserción va DENTRO del `with` -- al cerrar el bloque,
            # `TestClient` apaga su loop y eso por sí solo puede cancelar
            # cualquier tarea pendiente (incluida ésta), sin que tenga nada
            # que ver con `job_tasks.cancel()`. Afuera del `with`, este test
            # no distinguía "se canceló porque se lo pedí" de "se canceló
            # porque el runner cerró el loop" -- confirmado a mano: con
            # `job_tasks.register()` borrado, este assert acá adentro
            # falla (`marca == []`, la tarea sigue en su `sleep(5)`).
            assert marca == ["cancelado"], (
                f"la tarea no se cortó de verdad (job_tasks.register() sin efecto): {marca}"
            )

    # -- B-5: el done_callback de excepciones sigue wireado -----------------
    def test_B5_done_callback_registra_la_excepcion_no_capturada(self):
        async def _rompe(job_id, proyecto, rutas, *, store, executor=None, semaforo=None):
            raise RuntimeError("boom-catastrofico")

        with patch.object(rutas_mod, "_ejecutar_trabajo", _rompe), \
             patch.object(
                 rutas_mod, "_log_worker_exception",
                 wraps=rutas_mod._log_worker_exception,
             ) as espia, \
             TestClient(_app()) as c:
            r = self._post(c)
            assert r.status_code == 202, r.text
            time.sleep(0.05)

        espia.assert_called_once()

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
