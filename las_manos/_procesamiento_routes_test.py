"""
LAS MANOS -- endpoint de Procesamiento de Archivos (2026-09-21).

`POST /procesamiento/trabajos` + `GET /procesamiento/trabajos/{id}`: encola
un trabajo de ingesta y lo consulta después -- nunca síncrono (un PDF
escaneado de 30 páginas tarda ~80s medidos, y una petición HTTP no puede
esperar eso).

Corre en "tests-puros": `procesamiento.ingesta.ingerir` va SUSTITUIDO acá --
sin pdftoppm/tesseract/libreoffice reales. La extracción de verdad ya la
cubre `procesamiento/_ingesta_test.py` (jax#252); esto prueba el CABLEADO
nuevo: cola async, jail sobre las rutas de entrada, aislamiento de fallos
por archivo, forma de la respuesta, y la autenticación de servicio.

Dos grupos:

  - `TrabajoWorkerTest` (asíncrono, `IsolatedAsyncioTestCase`): llama
    `_ejecutar_trabajo`/`_procesar_una_ruta` DIRECTO, con un `JobStore` de
    tempdir propio (mismo patrón que `_motor_job_model_test.py` para
    `worker.run`) y `tool_authority.WORKSPACE_ROOT` parchado a un tempdir
    (mismo patrón que `_tool_authority_test.py`).
  - `TrabajoHTTPTest` (síncrono, `TestClient`): la forma HTTP -- 202
    inmediato, 404, y la credencial de servicio (mismo patrón que
    `tests/test_las_manos_auth_servicio.py`).

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

    def _archivo_en_workspace(self, nombre: str, contenido: bytes = b"x") -> str:
        (self.workspace / nombre).write_bytes(contenido)
        return nombre  # ruta RELATIVA -- contrato de resolve_jailed_path

    async def test_trabajo_completo_reporta_resultado_por_archivo_sin_contenido(self):
        """Caso feliz: un archivo válido produce un resultado con estado,
        extractor, tamaño del extracto y la carpeta -- nunca el contenido."""
        self._archivo_en_workspace("doc.pdf")

        def _ingerir_falso(origen, trabajo, *, subruta=None):
            ficha = _ficha("a" * 64)
            carpeta = rutas_mod.ingesta.ruta_procesado(trabajo, ficha.sha256)
            carpeta.mkdir(parents=True)
            (carpeta / "extracto.txt").write_text("contenido secreto del cliente")
            return ficha

        job_id = self.store.create(
            caller="x", capability="y", motor="z", trace_id="t", prompt="p",
            recursion_depth=0,
        )
        with patch.object(rutas_mod.ingesta, "ingerir", side_effect=_ingerir_falso):
            await rutas_mod._ejecutar_trabajo(job_id, "Proyecto Uno", ["doc.pdf"], store=self.store)

        job = self.store.get(job_id)
        assert job.status == JobStatus.COMPLETED, job
        assert job.result_path is not None
        resultados = json.loads(Path(job.result_path).read_text())
        assert len(resultados) == 1
        r = resultados[0]
        assert r["estado"] == "ok"
        assert r["extractor"] == "pdf"
        assert r["extracto_bytes"] > 0
        assert r["carpeta_procesado"] == f"proyectos/proyecto-uno/procesado/{'a' * 64}"
        # el endpoint NUNCA devuelve el contenido del extracto
        assert "contenido secreto" not in json.dumps(r)
        assert set(r) == {"archivo", "estado", "extractor", "extracto_bytes", "carpeta_procesado", "error"}

    async def test_ruta_fuera_del_jail_se_rechaza_sin_llamar_a_ingerir(self):
        """MUTACIÓN 1: una ruta que escapa del workspace (absoluta, o '..')
        se reporta 'rechazado' y jamás llega a `ingesta.ingerir` -- si el
        jail se salteara, este test lo vería llamado y no fallaría."""
        ingerir_mock = AsyncMock()
        job_id = self.store.create(
            caller="x", capability="y", motor="z", trace_id="t", prompt="p",
            recursion_depth=0,
        )
        with patch.object(rutas_mod.ingesta, "ingerir", ingerir_mock):
            await rutas_mod._ejecutar_trabajo(
                job_id, "proy", ["/etc/passwd", "../fuera-del-workspace.txt"],
                store=self.store,
            )

        job = self.store.get(job_id)
        assert job.status == JobStatus.COMPLETED, job
        resultados = json.loads(Path(job.result_path).read_text())
        assert len(resultados) == 2
        for r in resultados:
            assert r["estado"] == "rechazado", r
            assert r["error"], "el rechazo tiene que traer la razón del jail"
            assert r["carpeta_procesado"] is None
        ingerir_mock.assert_not_called()

    async def test_un_archivo_roto_no_tumba_el_lote(self):
        """MUTACIÓN 2: tres archivos, el del medio revienta `ingerir()` --
        el trabajo tiene que terminar COMPLETED con los tres resultados,
        no FAILED con cero (que es lo que pasaría si la excepción del
        archivo roto escapara del bucle)."""
        for nombre in ("uno.pdf", "dos.pdf", "tres.pdf"):
            self._archivo_en_workspace(nombre)

        respuestas = [_ficha("1" * 64), RuntimeError("pdftoppm: timeout"), _ficha("3" * 64)]

        def _ingerir_falso(origen, trabajo, *, subruta=None):
            resultado = respuestas.pop(0)
            if isinstance(resultado, Exception):
                raise resultado
            return resultado

        job_id = self.store.create(
            caller="x", capability="y", motor="z", trace_id="t", prompt="p",
            recursion_depth=0,
        )
        with patch.object(rutas_mod.ingesta, "ingerir", side_effect=_ingerir_falso):
            await rutas_mod._ejecutar_trabajo(
                job_id, "proy", ["uno.pdf", "dos.pdf", "tres.pdf"], store=self.store,
            )

        job = self.store.get(job_id)
        assert job.status == JobStatus.COMPLETED, (
            f"un archivo roto tumbó el lote entero: {job}"
        )
        resultados = json.loads(Path(job.result_path).read_text())
        assert len(resultados) == 3, "el lote no siguió con los demás archivos"
        assert resultados[0]["estado"] == "ok"
        assert resultados[1]["estado"] == "error"
        assert "pdftoppm: timeout" in resultados[1]["error"]
        assert resultados[2]["estado"] == "ok"

    async def test_marca_running_antes_de_completed(self):
        """El job pasa por RUNNING -- si esto se saltea, un caller que
        consulta justo después del POST vería un estado que no existió."""
        vistos: list[str] = []
        original_update = self.store.update

        def _update_espiado(job_id, **kwargs):
            if "status" in kwargs:
                vistos.append(kwargs["status"])
            return original_update(job_id, **kwargs)

        job_id = self.store.create(
            caller="x", capability="y", motor="z", trace_id="t", prompt="p",
            recursion_depth=0,
        )
        with patch.object(self.store, "update", side_effect=_update_espiado), \
             patch.object(rutas_mod.ingesta, "ingerir", return_value=_ficha("f" * 64)):
            self._archivo_en_workspace("a.pdf")
            await rutas_mod._ejecutar_trabajo(job_id, "p", ["a.pdf"], store=self.store)

        assert vistos == [JobStatus.RUNNING.value, JobStatus.COMPLETED.value], vistos

    async def test_ruta_ausente_dentro_del_jail_se_rechaza_no_revienta(self):
        job_id = self.store.create(
            caller="x", capability="y", motor="z", trace_id="t", prompt="p",
            recursion_depth=0,
        )
        with patch.object(rutas_mod.ingesta, "ingerir") as ingerir_mock:
            await rutas_mod._ejecutar_trabajo(
                job_id, "proy", ["no-existe.pdf"], store=self.store,
            )
        job = self.store.get(job_id)
        assert job.status == JobStatus.COMPLETED
        resultados = json.loads(Path(job.result_path).read_text())
        assert resultados[0]["estado"] == "rechazado"
        ingerir_mock.assert_not_called()


# ===========================================================================
#  Grupo 2 -- HTTP: forma de la respuesta, no-bloqueo, autenticación
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

    def test_post_no_espera_a_que_el_trabajo_termine(self):
        """MUTACIÓN implícita: si `crear_trabajo` hiciera `await
        _ejecutar_trabajo(...)` en vez de `asyncio.create_task(...)`, el
        POST tardaría los 2s completos del trabajo simulado -- este test
        exige que tarde una fracción de eso."""
        async def _lento(job_id, proyecto, rutas, *, store):
            await asyncio.sleep(2)
            store.update(job_id, status=JobStatus.COMPLETED.value, finished_at=time.time())

        with patch.object(rutas_mod, "_ejecutar_trabajo", _lento), \
             TestClient(_app()) as c:
            t0 = time.perf_counter()
            r = c.post(
                "/procesamiento/trabajos",
                json={"proyecto": "p", "rutas": ["a.pdf"]},
                headers=_h(IDENTIDAD_PLATAFORMA),
            )
            elapsed = time.perf_counter() - t0

        assert r.status_code == 202, r.text
        assert "job_id" in r.json()
        assert elapsed < 1.0, f"el POST esperó al trabajo ({elapsed:.2f}s) -- no es asíncrono"

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

    def test_credencial_plataforma_accede_jacobs_no(self):
        with TestClient(_app()) as c:
            r_plat = c.post(
                "/procesamiento/trabajos", json={"proyecto": "p", "rutas": []},
                headers=_h(IDENTIDAD_PLATAFORMA),
            )
            r_jacobs = c.post(
                "/procesamiento/trabajos", json={"proyecto": "p", "rutas": []},
                headers=_h(IDENTIDAD_JACOBS),
            )
            r_sin_credencial = c.post(
                "/procesamiento/trabajos", json={"proyecto": "p", "rutas": []},
            )
        assert r_plat.status_code == 202, r_plat.text
        assert r_jacobs.status_code == 403, r_jacobs.text
        assert r_sin_credencial.status_code == 401, r_sin_credencial.text


if __name__ == "__main__":
    unittest.main(verbosity=2)
