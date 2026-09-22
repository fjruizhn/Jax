"""
LAS MANOS -- Endpoint de Procesamiento de Archivos (2026-09-21).

`procesamiento/` (jax#252, 252 tests, en verde) ya sabe ingerir un documento
-- lo que faltaba era quien lo llamara desde HTTP sin bloquear el request:
un PDF escaneado de 30 paginas tarda ~80s (2,7s/pagina medidos), y una
peticion HTTP no puede quedarse esperando eso.

    POST /procesamiento/trabajos      {proyecto, rutas[]} -> {job_id} (202)
    GET  /procesamiento/trabajos/{id} -> estado + resultados por archivo

Maquinaria REUSADA, no nueva:

  - `motor_registry.job_store.JobStore` + `JobStatus` -- el mismo append-only
    JSONL y el mismo enum de estados que usan los jobs de motor. INSTANCIA
    PROPIA (JSONL propio, `logs/procesamiento_jobs.jsonl`), no la `_STORE`
    de `motor_registry/routes.py`: `JobStore.create()` exige motor/
    capability/prompt/recursion_depth -- vocabulario del dominio de motores
    LLM. Forzar "proyecto"/"rutas" en esos campos hubiera sido peor que una
    segunda instancia de la MISMA clase, en el MISMO archivo append-only,
    sin reescribir una sola linea de la maquinaria (ver el Informe).
  - `JobStore.write_result()` (ya existente: guarda contenido arbitrario en
    un archivo propio junto al JSONL, devuelve la ruta) guarda el detalle
    por archivo -- `MotorJobView` no tiene un campo para "resultados por
    archivo" y este modulo no lo toca; el endpoint arma su propia respuesta
    leyendo `result_path`.
  - `motor_registry.job_tasks.register()` -- mismo mecanismo que evita que
    el `asyncio.create_task` del dispatch de motores se pierda por falta de
    referencia fuerte (ver worker.py / GAP2).

Jail: cada ruta de entrada pasa por
`motor_registry.tool_authority.resolve_jailed_path` -- el MISMO jail que
protege `read_file`/`write_file` (GAP2 Fase 2). Una ruta fuera de
`JAX_WORKSPACE_DIR` NUNCA llega a `ingesta.ingerir()`: se rechaza para ESE
archivo y el resto del lote sigue (fallo cerrado, no una excepcion que
tumbe el trabajo entero).

Autenticacion: ninguna nueva. Las dos rutas se suman a
`auth_servicio.PERMISOS[IDENTIDAD_PLATAFORMA]` -- el MISMO mecanismo que ya
protege `/jacobs/*` y `/motor/authorize-facet` (jax-platform es quien habla
HTTP con LAS MANOS para esto, ver `server.py`).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import unicodedata
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from motor_registry import job_tasks, tool_authority
from motor_registry.job_store import JobStore
from motor_registry.models import JobStatus
from procesamiento import ingesta

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
_STORE = JobStore(str(BASE_DIR / "logs" / "procesamiento_jobs.jsonl"))

#: nombre del archivo de metadatos que NO cuenta como salida del extractor
#: (mismo criterio que `scripts/procesar_archivos.py::_tamano_extracto`).
_NOMBRE_FICHA = "ficha.json"

router = APIRouter(prefix="/procesamiento", tags=["procesamiento"])


# ---------------------------------------------------------------------------
#  Modelos
# ---------------------------------------------------------------------------
class TrabajoRequest(BaseModel):
    proyecto: str
    rutas: list[str]
    model_config = ConfigDict(extra="forbid")


class TrabajoCreadoResponse(BaseModel):
    job_id: str


class ResultadoArchivo(BaseModel):
    """Nunca lleva el contenido del extracto -- para eso está `file_read`
    (jail de `tool_authority`), no este endpoint."""
    archivo: str
    estado: str
    extractor: str | None = None
    extracto_bytes: int = 0
    carpeta_procesado: str | None = None
    error: str | None = None


class TrabajoEstadoResponse(BaseModel):
    job_id: str
    estado: str
    error: str | None = None
    resultados: list[ResultadoArchivo] = []


# ---------------------------------------------------------------------------
#  Lógica pura (sin store, sin red) -- fácil de probar y de razonar
# ---------------------------------------------------------------------------
def _slug(texto: str) -> str:
    """Minúsculas, sin acentos, separado por guiones -- MISMA regla que
    `scripts/procesar_archivos.py::_slug` (duplicada a propósito: `scripts/`
    no es un paquete importable -- sin `__init__.py`, a diferencia de
    `procesamiento/`, `las_manos/`, `jacobs/` -- importar desde ahí hubiera
    invertido la dirección de dependencia; ver el Informe)."""
    sin_acentos = (
        unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    )
    normalizado = re.sub(r"[^a-z0-9]+", "-", sin_acentos.lower()).strip("-")
    return normalizado or "carpeta"


def _tamano_extracto(carpeta: Path) -> int:
    if not carpeta.is_dir():
        return 0
    return sum(
        p.stat().st_size
        for p in carpeta.iterdir()
        if p.is_file() and p.name != _NOMBRE_FICHA
    )


def _trabajo_de(proyecto: str) -> Path:
    return tool_authority.WORKSPACE_ROOT / "proyectos" / _slug(proyecto)


def _procesar_una_ruta(trabajo: Path, ruta: str) -> ResultadoArchivo:
    """Nunca lanza: una ruta fuera del jail, ausente, o cuya ingesta falla
    queda REPORTADA, no propagada -- fallo cerrado por archivo, para que un
    documento roto no tumbe el resto del lote."""
    resolved, razon = tool_authority.resolve_jailed_path(ruta, [])
    if resolved is None:
        return ResultadoArchivo(archivo=ruta, estado="rechazado", error=razon)
    if not resolved.is_file():
        return ResultadoArchivo(
            archivo=ruta, estado="rechazado",
            error=f"no es un archivo regular dentro del workspace: '{ruta}'",
        )

    try:
        ficha = ingesta.ingerir(resolved, trabajo)
    except Exception as exc:  # fail cerrado: un archivo roto no tumba el lote
        logger.error("procesamiento: fallo ingiriendo '%s': %r", ruta, exc)
        return ResultadoArchivo(archivo=ruta, estado="error", error=str(exc))

    carpeta = ingesta.ruta_procesado(trabajo, ficha.sha256)
    return ResultadoArchivo(
        archivo=ruta,
        estado=ficha.estado,
        extractor=ficha.extractor,
        extracto_bytes=_tamano_extracto(carpeta),
        carpeta_procesado=str(carpeta.relative_to(tool_authority.WORKSPACE_ROOT)),
    )


def _procesar_rutas(proyecto: str, rutas: list[str]) -> list[dict]:
    """Corre en un hilo (`asyncio.to_thread`, ver `_ejecutar_trabajo`):
    `ingesta.ingerir` es E/S de disco y subprocess bloqueantes (pdftoppm,
    tesseract), nunca dentro del loop de eventos (LAS CUATRO DEL RENDIMIENTO,
    regla 3 -- async)."""
    trabajo = _trabajo_de(proyecto)
    return [
        _procesar_una_ruta(trabajo, ruta).model_dump()
        for ruta in rutas
    ]


# ---------------------------------------------------------------------------
#  El worker del job -- toma `store` explícito (mismo patrón testable que
#  `motor_registry.worker.run`, no el closure sobre un global de routes.py)
# ---------------------------------------------------------------------------
async def _ejecutar_trabajo(
    job_id: str, proyecto: str, rutas: list[str], *, store: JobStore,
) -> None:
    store.update(job_id, status=JobStatus.RUNNING.value, started_at=time.time())
    try:
        resultados = await asyncio.to_thread(_procesar_rutas, proyecto, rutas)
        result_path = await asyncio.to_thread(
            store.write_result, job_id, json.dumps(resultados, ensure_ascii=False),
        )
        por_estado: dict[str, int] = {}
        for r in resultados:
            por_estado[r["estado"]] = por_estado.get(r["estado"], 0) + 1
        store.update(
            job_id, status=JobStatus.COMPLETED.value, finished_at=time.time(),
            result_path=result_path,
            result_summary=f"{len(resultados)} archivo(s): {por_estado}"[:200],
        )
    except Exception as exc:  # nunca deja un job pending/running huérfano
        logger.error("procesamiento: job %s falló: %r", job_id, exc)
        store.update(
            job_id, status=JobStatus.FAILED.value, finished_at=time.time(),
            error=str(exc),
        )


def _log_worker_exception(task: asyncio.Task, *, job_id: str) -> None:
    """Mismo patrón que motor_registry/routes.py::_log_worker_exception:
    cierra el punto ciego del create_task fire-and-forget."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "procesamiento: excepción no capturada en el worker del job %s: %r",
            job_id, exc,
        )


# ---------------------------------------------------------------------------
#  Rutas HTTP
# ---------------------------------------------------------------------------
@router.post("/trabajos", response_model=TrabajoCreadoResponse, status_code=202)
async def crear_trabajo(req: TrabajoRequest) -> TrabajoCreadoResponse:
    job_id = _STORE.create(
        caller="las_manos.procesamiento",
        capability="ingesta_archivos",
        motor="procesamiento",
        trace_id=str(uuid.uuid4()),
        prompt=req.proyecto,
        recursion_depth=0,
    )
    task = asyncio.create_task(
        _ejecutar_trabajo(job_id, req.proyecto, req.rutas, store=_STORE)
    )
    task.add_done_callback(lambda t: _log_worker_exception(t, job_id=job_id))
    job_tasks.register(job_id, task)
    return TrabajoCreadoResponse(job_id=job_id)


@router.get("/trabajos/{job_id}", response_model=TrabajoEstadoResponse)
async def estado_trabajo(job_id: str) -> TrabajoEstadoResponse:
    view = _STORE.get(job_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"Trabajo '{job_id}' no encontrado")

    resultados: list[ResultadoArchivo] = []
    if view.result_path:
        try:
            crudos = json.loads(Path(view.result_path).read_text(encoding="utf-8"))
            resultados = [ResultadoArchivo(**r) for r in crudos]
        except (OSError, ValueError) as exc:  # fail-soft: un resultado ilegible no tumba la consulta de estado -- el job YA terminó, esto es sólo su detalle
            logger.error("procesamiento: no se pudo leer el resultado de %s: %r", job_id, exc)

    return TrabajoEstadoResponse(
        job_id=job_id, estado=view.status.value, error=view.error, resultados=resultados,
    )
