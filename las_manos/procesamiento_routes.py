"""
LAS MANOS -- Endpoint de Procesamiento de Archivos (2026-09-21, ronda de
arreglo tras revisión adversarial -- ver
`.superpowers/sdd/2026-09-20-procesamiento-archivos-nucleo/endpoint-hallazgos.md`).

`procesamiento/` (jax#252, 252 tests, en verde) ya sabe ingerir un documento
-- lo que faltaba era quien lo llamara desde HTTP sin bloquear el request:
un PDF escaneado de 30 páginas tarda ~80s (2,7s/página medidos), y una
petición HTTP no puede quedarse esperando eso.

    POST /procesamiento/trabajos             {proyecto, rutas[], usuario} -> {job_id} (202)
    GET  /procesamiento/trabajos/{id}        -> estado + resultados por archivo
    POST /procesamiento/trabajos/{id}/cancel -> corta el trabajo en vuelo

Maquinaria REUSADA de `motor_registry` (job_store, job_tasks, tool_authority),
instancia propia de `JobStore` -- ver el resto del razonamiento en el
Informe (`endpoint-report.md`). Esta ronda corrige seis hallazgos de la
revisión adversarial:

- **B-1** (raíz, en `tool_authority.py`): un byte NUL en una ruta dejaba
  escapar `ValueError` fuera de `resolve_jailed_path` -- arreglado ahí, no
  acá.
- **B-2**: el trabajo corre en un `ThreadPoolExecutor` PROPIO
  (`_EXECUTOR`), nunca el executor por defecto que usa `worker.py` para
  despachar motores (`git show`/`git reset` vía `asyncio.to_thread`). Cada
  ARCHIVO es su propia unidad de trabajo (no el lote entero en un solo
  hilo) -- así un job con muchos archivos se reparte entre los hilos del
  pool en vez de monopolizar uno solo por horas. Admisión: un semáforo
  (`_SEMAFORO_TRABAJOS`, del mismo tamaño que el pool) y un tope de
  `len(rutas)` por pedido (`_MAX_RUTAS_POR_TRABAJO`); sin lugar, 429 ANTES
  de crear ningún registro.
- **B-3**: `reconciliar_trabajos_huerfanos()` (llamada desde el startup
  hook de `server.py`) marca `failed` a lo que haya quedado
  `pending`/`running` de una corrida anterior -- un reinicio ya no deja un
  trabajo "corriendo" para siempre. `POST .../cancel` corta uno en vuelo.
- **B-4**: `GET` ya no lee ni parsea el resultado dentro del loop de
  eventos -- va al mismo executor dedicado.
- **B-5**: ver `_procesamiento_routes_test.py` -- la mutación que importa
  (sacar el `run_in_executor`) ahora se prueba MIDIENDO EL RETRASO DEL LOOP
  durante el trabajo, no el retorno del POST.
- **B-6**: `usuario` es obligatorio en el pedido y se registra como
  `caller` del job (antes era una constante que no identificaba a nadie).
  No es autorización completa (eso es otra ronda) -- es lo mínimo para que
  un incidente sea investigable.

**Limitación documentada, diferida a propósito** (no se arregla en esta
ronda): `JobStore.create()` exige vocabulario de motores LLM
(`motor`/`capability`/`prompt`) que este dominio no tiene. En vez de forzar
`proyecto` dentro de `prompt` (mentir), esos campos llevan un valor que
declara explícitamente "no aplica", y el proyecto se guarda en un campo
PROPIO (`proyecto`, vía `JobStore.update()`, que acepta kwargs arbitrarios)
-- el arreglo de raíz (que `JobStore` deje de exigir ese vocabulario) es
refactor de código compartido con los jobs de motor y no es una decisión de
esta ronda.

**MINOR-7, anotado y no explotable hoy:** esta instancia de `JobStore` y la
de `motor_registry/routes.py` comparten `logs/motor_results/` (mismo
directorio padre, nombres de archivo por `job_id` -- un UUID4, así que la
probabilidad de colisión real es nula) y `job_tasks._RUNNING` (mismo
diccionario global, misma razón).

Jail: cada ruta de entrada pasa por
`motor_registry.tool_authority.resolve_jailed_path` -- el MISMO jail que
protege `read_file`/`write_file` (GAP2 Fase 2).

Autenticación: ninguna nueva. Las tres rutas se suman a
`auth_servicio.PERMISOS[IDENTIDAD_PLATAFORMA]`.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
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

# ---------------------------------------------------------------------------
#  B-2: executor propio + admisión
# ---------------------------------------------------------------------------
# Medido con tesseract REAL (2026-09-21, ver endpoint-report.md): 1 página
# sintética de densidad realista, 1,97s secuencial (consistente con los
# "2,7s/página" ya documentados en el proyecto para un PDF escaneado real).
# 4 páginas EN PARALELO con este mismo diseño (executor dedicado,
# max_workers=4): 2,54s totales -- contra 7,90s si fuera secuencial. Con el
# executor dedicado SATURADO por esas 4 páginas, un `to_thread` ajeno
# (representando un `git show`/`git reset` de worker.py, que usa el
# executor POR DEFECTO) tardó 0,4 ms -- el aislamiento es real, no teórico.
# Configurable porque el número correcto depende del hardware real de
# producción, nunca hardcodeado sin escape (Principio IV).
_MAX_WORKERS = int(os.getenv("JAX_PROCESAMIENTO_MAX_WORKERS", "4"))
_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="procesamiento")

# Como máximo tantos TRABAJOS corriendo a la vez como hilos reales tiene el
# executor -- un trabajo admitido siempre tiene, por construcción, un hilo
# real disponible en el mismo pool (nunca "running" mintiendo detrás de una
# cola sin fondo). Sin lugar -> 429, ANTES de crear ningún registro: nunca
# queda un job a medias por falta de capacidad.
_SEMAFORO_TRABAJOS = asyncio.Semaphore(_MAX_WORKERS)

# Tope estructural de archivos por pedido: un solo request no puede
# monopolizar el pool durante horas. Con 4 hilos y documentos de ~30
# páginas (~80s cada uno, 2,7s/página), 50 archivos son, en el peor caso,
# (50/4)*80s ≈ 1000s (~17 min) -- contra las 2,2h que ocupaba UN hilo antes
# de esta ronda. Sigue siendo mucho, y por eso el pedido puede partirse en
# varios `POST` -- pero ya no puede tumbar el pool compartido con los
# motores, que es lo que B-2 vino a cerrar.
_MAX_RUTAS_POR_TRABAJO = int(os.getenv("JAX_PROCESAMIENTO_MAX_RUTAS", "50"))

# MINOR-6: tope de longitud de `proyecto` -- sin esto, un `proyecto`
# arbitrariamente largo infla el JSONL append-only sin límite (cada
# `update()` re-esparce el estado ENTERO, así que un campo largo se
# duplica en cada línea).
_MAX_PROYECTO_LEN = 200

router = APIRouter(prefix="/procesamiento", tags=["procesamiento"])


# ---------------------------------------------------------------------------
#  Modelos
# ---------------------------------------------------------------------------
class TrabajoRequest(BaseModel):
    proyecto: str
    rutas: list[str]
    # B-6: principal obligatorio -- sin esto, todo trabajo tenía el mismo
    # `caller` constante y ningún incidente era atribuible. jax-platform ya
    # tiene el JWT del usuario; que lo pase. No es autorización completa
    # (eso es otra ronda): es lo mínimo para que un IDOR sea investigable.
    usuario: str
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
    documento roto no tumbe el resto del lote.

    B-1: antes, una ruta con un byte NUL hacía que `resolve_jailed_path`
    dejara escapar `ValueError` -- arreglado en la raíz
    (`tool_authority.py`), no acá: esta función sigue confiando en que
    `resolve_jailed_path` nunca lanza, y ahora es cierto."""
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


async def _procesar_rutas_paralelo(
    executor: ThreadPoolExecutor, proyecto: str, rutas: list[str],
) -> list[dict]:
    """B-2: cada RUTA es su propia unidad de trabajo en el executor
    dedicado -- antes, `_procesar_rutas` metía el lote ENTERO en un único
    `asyncio.to_thread`, así que ni con un pool más grande se podía
    repartir un job de muchos archivos entre varios hilos. `gather`
    conserva el orden de `rutas` en el resultado."""
    trabajo = _trabajo_de(proyecto)
    loop = asyncio.get_running_loop()
    futuros = [
        loop.run_in_executor(executor, _procesar_una_ruta, trabajo, ruta)
        for ruta in rutas
    ]
    resultados = await asyncio.gather(*futuros)
    return [r.model_dump() for r in resultados]


# ---------------------------------------------------------------------------
#  El worker del job -- toma `store`/`executor`/`semaforo` explícitos
#  (mismo patrón testable que `motor_registry.worker.run`).
#
#  CONTRATO: quien llama a esta función YA tiene que haber adquirido
#  `semaforo` (una unidad) ANTES de crear la tarea -- esta función lo
#  libera en un `finally`, pase lo que pase (éxito, excepción, o
#  cancelación). `crear_trabajo()` es el único caller real; los tests que
#  la llaman directo tienen que adquirir el semáforo ellos mismos primero.
# ---------------------------------------------------------------------------
async def _ejecutar_trabajo(
    job_id: str, proyecto: str, rutas: list[str], *, store: JobStore,
    executor: ThreadPoolExecutor | None = None,
    semaforo: asyncio.Semaphore | None = None,
) -> None:
    executor = executor if executor is not None else _EXECUTOR
    semaforo = semaforo if semaforo is not None else _SEMAFORO_TRABAJOS
    try:
        # B-2 (ruling): "movés el store.update(RUNNING) a después de tomar
        # el hilo". Como la admisión (el semáforo, tomado por el caller
        # ANTES de crear esta tarea) tiene el MISMO tamaño que el pool de
        # hilos, que esta corrutina esté corriendo YA significa que hay
        # capacidad real en el pool -- a diferencia de antes, donde
        # `RUNNING` se escribía apenas se creaba la tarea, sin importar
        # cuántos hilos ya estaban ocupados.
        store.update(job_id, status=JobStatus.RUNNING.value, started_at=time.time())
        try:
            resultados = await _procesar_rutas_paralelo(executor, proyecto, rutas)
            loop = asyncio.get_running_loop()
            result_path = await loop.run_in_executor(
                executor, store.write_result, job_id,
                json.dumps(resultados, ensure_ascii=False),
            )
            por_estado: dict[str, int] = {}
            for r in resultados:
                por_estado[r["estado"]] = por_estado.get(r["estado"], 0) + 1
            store.update(
                job_id, status=JobStatus.COMPLETED.value, finished_at=time.time(),
                result_path=result_path,
                result_summary=f"{len(resultados)} archivo(s): {por_estado}"[:200],
            )
        except Exception as exc:
            # `except Exception` -- NO `BaseException` -- a propósito:
            # `asyncio.CancelledError` hereda de `BaseException` desde
            # Python 3.8 y pasa de largo. Eso es correcto acá: cuando el
            # trabajo se cancela, `cancelar_trabajo()` YA marcó
            # `CANCELLED` en el store ANTES de cortar esta tarea -- si
            # este bloque atrapara la cancelación y escribiera `FAILED`
            # encima, pisaría ese estado con uno menos preciso.
            logger.error("procesamiento: job %s falló: %r", job_id, exc)
            store.update(
                job_id, status=JobStatus.FAILED.value, finished_at=time.time(),
                error=str(exc),
            )
    finally:
        # Se libera SIEMPRE -- éxito, excepción capturada arriba, o
        # cancelación (un `finally` corre igual cuando `CancelledError`
        # atraviesa la corrutina). Sin esto, un trabajo cancelado o
        # catastróficamente roto deja el permiso tomado para siempre y el
        # semáforo se agota en silencio.
        semaforo.release()


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


def _leer_resultados_de_disco(result_path: str) -> list[dict]:
    """B-4: corre en el executor dedicado, nunca en el loop de eventos.
    `read_text` + `json.loads` son bloqueantes; medido con 200.000 rutas en
    un solo resultado: 0,39s con LAS MANOS entero congelado (`/health`, los
    pipelines, todo) cuando esto corría directo en una `async def`."""
    try:
        crudos = json.loads(Path(result_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # fail-soft: el job YA terminó, un resultado ilegible no tumba la consulta de estado
        logger.error("procesamiento: no se pudo leer el resultado en '%s': %r", result_path, exc)
        return []
    if not isinstance(crudos, list):
        logger.error("procesamiento: resultado con forma inesperada en '%s' (no es una lista)", result_path)
        return []
    return crudos


async def _construir_respuesta_estado(job_id: str, view) -> TrabajoEstadoResponse:
    resultados: list[ResultadoArchivo] = []
    if view.result_path:
        loop = asyncio.get_running_loop()
        crudos = await loop.run_in_executor(_EXECUTOR, _leer_resultados_de_disco, view.result_path)
        try:
            resultados = [ResultadoArchivo(**r) for r in crudos]
        except (TypeError, ValueError) as exc:
            logger.error("procesamiento: resultado con forma inesperada para %s: %r", job_id, exc)
    return TrabajoEstadoResponse(
        job_id=job_id, estado=view.status.value, error=view.error, resultados=resultados,
    )


# ---------------------------------------------------------------------------
#  Reconciliación al arrancar (B-3)
# ---------------------------------------------------------------------------
def reconciliar_trabajos_huerfanos(store: JobStore | None = None) -> int:
    """Al arrancar LAS MANOS: cualquier job `pending`/`running` en el JSONL
    es, por definición, huérfano -- este proceso recién arrancó y
    `job_tasks._RUNNING` (in-memory) está vacío, así que NINGUNA tarea viva
    puede estar trabajando en él. `JobStore._load()` reconstruye el índice
    desde disco pero nunca reconcilia estados "en vuelo" contra tareas
    reales -- sin este paso, un reinicio del servicio deja el trabajo
    'running' PARA SIEMPRE. Se llama una sola vez desde el startup hook de
    `server.py` (mismo criterio que `jacobs.reaper.reap_orphaned_pipelines`
    para los pipelines de Jacobs)."""
    store = store if store is not None else _STORE
    huerfanos = store.ids_en_estado(JobStatus.PENDING.value, JobStatus.RUNNING.value)
    for job_id in huerfanos:
        store.update(
            job_id, status=JobStatus.FAILED.value, finished_at=time.time(),
            error="proceso de LAS MANOS reiniciado mientras el trabajo corría (recuperado al arrancar)",
        )
    if huerfanos:
        logger.warning(
            "procesamiento: %d trabajo(s) huérfano(s) reconciliados a 'failed' al arrancar: %s",
            len(huerfanos), huerfanos,
        )
    return len(huerfanos)


# ---------------------------------------------------------------------------
#  Rutas HTTP
# ---------------------------------------------------------------------------
@router.post("/trabajos", response_model=TrabajoCreadoResponse, status_code=202)
async def crear_trabajo(req: TrabajoRequest) -> TrabajoCreadoResponse:
    if len(req.rutas) > _MAX_RUTAS_POR_TRABAJO:
        raise HTTPException(
            status_code=422,
            detail=(
                f"demasiadas rutas en un solo trabajo: {len(req.rutas)} > "
                f"{_MAX_RUTAS_POR_TRABAJO} (JAX_PROCESAMIENTO_MAX_RUTAS) -- "
                "partilo en más de un pedido"
            ),
        )
    if _SEMAFORO_TRABAJOS.locked():
        raise HTTPException(
            status_code=429,
            detail=(
                f"sin capacidad de procesamiento libre ahora mismo "
                f"(máximo {_MAX_WORKERS} trabajos concurrentes) -- reintentar"
            ),
        )
    # Sin ningún `await` entre el chequeo de arriba y este acquire: en
    # asyncio (cooperativo, un solo hilo) eso significa que ningún otro
    # request puede colarse en el medio y robarse el permiso que `locked()`
    # vio libre -- atómico dentro de este turno del loop de eventos (B-2).
    await _SEMAFORO_TRABAJOS.acquire()

    proyecto = req.proyecto[:_MAX_PROYECTO_LEN]  # MINOR-6
    job_id = _STORE.create(
        # B-6: el principal REAL -- antes era la constante
        # "las_manos.procesamiento", que no identificaba a nadie.
        caller=req.usuario,
        capability="ingesta_archivos",
        # `motor`/`prompt` son vocabulario de JobStore para motores LLM
        # (create() los exige) -- este job no despacha ningún motor. En vez
        # de fingir (poner "procesamiento" como si fuera un motor real, o
        # el nombre del proyecto como si fuera un prompt), el valor declara
        # explícitamente que no aplica. El arreglo de raíz (que JobStore
        # deje de exigir este vocabulario) queda diferido -- ver el
        # docstring del módulo.
        motor="n/a_no_es_un_motor_llm",
        trace_id=str(uuid.uuid4()),
        prompt="n/a -- este job no despacha un motor LLM, ver el campo 'proyecto'",
        recursion_depth=0,
    )
    # El campo que de verdad significa "proyecto" -- `update()` acepta
    # kwargs arbitrarios (van tal cual al JSONL), así que no hace falta
    # forzarlo dentro de `prompt`.
    _STORE.update(job_id, proyecto=proyecto)

    # El semáforo se libera DENTRO de `_ejecutar_trabajo` (su propio
    # `finally`, contrato documentado en su docstring) -- no acá, para no
    # liberarlo dos veces.
    task = asyncio.create_task(
        _ejecutar_trabajo(job_id, proyecto, req.rutas, store=_STORE)
    )
    task.add_done_callback(lambda t: _log_worker_exception(t, job_id=job_id))
    job_tasks.register(job_id, task)
    return TrabajoCreadoResponse(job_id=job_id)


@router.get("/trabajos/{job_id}", response_model=TrabajoEstadoResponse)
async def estado_trabajo(job_id: str) -> TrabajoEstadoResponse:
    view = _STORE.get(job_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"Trabajo '{job_id}' no encontrado")
    return await _construir_respuesta_estado(job_id, view)


@router.post("/trabajos/{job_id}/cancel", response_model=TrabajoEstadoResponse)
async def cancelar_trabajo(job_id: str) -> TrabajoEstadoResponse:
    """Mismo patrón que `POST /motor/job/{id}/cancel`
    (motor_registry/routes.py::cancel_job): el store se marca `CANCELLED`
    ANTES de cortar la tarea -- si fuera al revés, `_ejecutar_trabajo`
    podría alcanzar a escribir `COMPLETED`/`FAILED` en la ventana entre el
    corte y el update(), y el cliente vería un estado que no refleja lo que
    pidió."""
    view = _STORE.get(job_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"Trabajo '{job_id}' no encontrado")
    if view.status in (
        JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.REJECTED,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Trabajo '{job_id}' ya está en estado terminal: {view.status.value}",
        )
    _STORE.update(job_id, status=JobStatus.CANCELLED.value, finished_at=time.time())
    job_tasks.cancel(job_id)
    return await _construir_respuesta_estado(job_id, _STORE.get(job_id))
