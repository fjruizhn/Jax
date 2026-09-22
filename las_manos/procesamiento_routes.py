"""
LAS MANOS -- Endpoint de Procesamiento de Archivos (2026-09-21, ronda 3 de
arreglo tras revisión adversarial -- ver
`.superpowers/sdd/2026-09-20-procesamiento-archivos-nucleo/endpoint-hallazgos.md`
y `endpoint-hallazgos-r2.md`).

`procesamiento/` (jax#252, 252 tests, en verde) ya sabe ingerir un documento
-- lo que faltaba era quien lo llamara desde HTTP sin bloquear el request:
un PDF escaneado de 30 páginas tarda ~80s (2,7s/página medidos), y una
petición HTTP no puede quedarse esperando eso.

    POST /procesamiento/trabajos             {proyecto, rutas[], usuario} -> {job_id} (202)
    GET  /procesamiento/trabajos/{id}        -> estado + resultados por archivo
    POST /procesamiento/trabajos/{id}/cancel -> deja de programar archivos nuevos

Ronda 2 (B-1..B-6) cerró el jail sobre bytes NUL, el executor propio, la
reconciliación al arrancar, el GET async y el principal obligatorio. La
ronda 3 corrige el defecto ESTRUCTURAL que esa ronda dejó sin nombrar,
señalado por el ruling del coordinador:

    "El semáforo cuenta TRABAJOS y el pool cuenta ARCHIVOS. Todo lo que
    miente sale de ahí: running sin hilos, el cupo que se libera antes de
    tiempo, el GET esperando 37s."

Siete arreglos, en el orden del ruling:

- **N-1 (bloqueante):** el semáforo se adquiría ANTES de `_STORE.create()`
  -- si `create()` lanzaba (ej. un `usuario` con un surrogate solitario,
  JSON válido que pydantic acepta pero que `json.dumps`+escritura UTF-8 no
  puede codificar), nadie lo liberaba. Cuatro pedidos así agotaban el
  semáforo PARA SIEMPRE (DoS con cuatro requests). Ahora todo el tramo
  entre el `acquire()` y que la tarea quede registrada corre bajo un
  `try/finally` que libera el permiso si algo falla ANTES de que la tarea
  tome la responsabilidad de liberarlo ella misma.
- **N-2 (bloqueante):** una ruta cuyo nombre no se puede codificar a UTF-8
  (ej. un byte perdido de cp1252 que Python re-expone como surrogate
  solitario, `\\udcXX`) se rechaza ANTES de tocar el executor -- nunca paga
  un hilo real por algo que no se puede ni reportar. Y el nombre que SÍ se
  guarda en el resultado va saneado (`backslashreplace`), así que ESA
  entrada nunca puede tumbar la escritura del LOTE ENTERO. `_guardar_
  resultado` además tiene una defensa de última línea: si de cualquier
  otra forma algo no codificable se cuela, reintenta con
  `ensure_ascii=True` (texto puro ASCII SIEMPRE es UTF-8 válido) en vez de
  perder los resultados de los archivos sanos.
- **N-3:** cancelar es honesto sobre lo que Python puede hacer -- NO se
  mata un hilo, y `Future.cancel()` NO sirve para simularlo: verificado a
  mano que, sobre un `run_in_executor` YA corriendo, `.cancel()` devuelve
  `True` de inmediato mintiendo -- el hilo real sigue solo en segundo
  plano. `POST .../cancel` marca `_ControlTrabajo.cancelado`; cada archivo
  se AUTOCONSULTA justo antes de arrancar de verdad y se salta sin tocar
  el disco si ya se pidió cancelar. Lo que ya arrancó sigue hasta
  terminar SOLO. El job pasa por `CANCELLING` mientras espera esos hilos,
  y sólo a `CANCELLED` (terminal) -- y sólo AHÍ se libera el semáforo --
  cuando el último hilo en vuelo de ESE trabajo terminó.
- **N-4 / B-2 (executor separado para I/O):** un `ThreadPoolExecutor`
  chico y propio (`_EXECUTOR_IO`) para leer/escribir el resultado del
  job -- nunca el mismo pool que hace OCR real. Antes, consultar un
  trabajo YA TERMINADO podía esperar detrás de horas de cola de OCR (medido
  en la revisión: 37s). Y `RUNNING` se escribe DESDE DENTRO DEL HILO, la
  primera vez que un archivo de ESE trabajo arranca de verdad -- no al
  admitir el pedido.
- **N-5:** la mutación que de verdad importa es cambiar el executor que se
  le pasa a `run_in_executor` por `None` (que usa el executor compartido
  con los motores, EXACTAMENTE lo que B-2 vino a evitar) -- un test de
  retraso del loop no la detecta (con `None` el trabajo SIGUE fuera del
  loop, sólo que en el pool equivocado). El test de esta ronda espía el
  objeto executor real y confirma que es a ÉL a quien le llega el trabajo.
- **B-6:** `usuario` no puede venir vacío ni arbitrariamente largo
  (`pydantic.Field(min_length=1, max_length=...)`) -- antes `""` y un
  string de 2 MB daban 202 los dos, y el de 2 MB inflaba el JSONL en 4 MB
  (se re-esparce el estado ENTERO en cada `update()`).
- **N-6:** test dedicado que confirma que el startup hook de `server.py`
  llama a `reconciliar_trabajos_huerfanos()`.

Jail: cada ruta de entrada pasa por
`motor_registry.tool_authority.resolve_jailed_path` -- el MISMO jail que
protege `read_file`/`write_file` (GAP2 Fase 2).

Autenticación: ninguna nueva. Las tres rutas se suman a
`auth_servicio.PERMISOS[IDENTIDAD_PLATAFORMA]`.

**Limitación documentada, diferida a propósito** (no se arregla en esta
ronda): `JobStore.create()` exige vocabulario de motores LLM
(`motor`/`capability`/`prompt`) que este dominio no tiene. `motor`/`prompt`
llevan un valor que declara explícitamente "no aplica", y `proyecto` se
guarda en un campo PROPIO vía `JobStore.update()` (acepta kwargs
arbitrarios). El arreglo de raíz (que `JobStore` deje de exigir ese
vocabulario) es refactor de código compartido con los jobs de motor.

**MINOR-7, anotado y no explotable hoy:** esta instancia de `JobStore` y la
de `motor_registry/routes.py` comparten `logs/motor_results/` y
`job_tasks._RUNNING` -- nombres de archivo/claves por `job_id` (UUID4),
colisión real nula.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

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
#  B-2 / N-4: DOS executors propios, con roles distintos
# ---------------------------------------------------------------------------
# `_EXECUTOR_OCR`: el trabajo pesado (ingesta.ingerir -- pdftoppm/tesseract
# reales). Medido con tesseract real (script completo, reproducible, en
# endpoint-report.md -- la ronda anterior dio un número sin dejar el
# script, y no se pudo reproducir; esta vez el script queda escrito):
# 1 página sintética de densidad realista ronda 1,9-2,4s según densidad de
# texto (dos imágenes distintas midieron 1,97s y 2,3s) -- el propio
# re-revisor midió 6,3s/página en OTRA imagen, más densa. El número
# correcto depende del contenido real, no es una constante -- por eso el
# tamaño del pool es una env var, no un literal.
_MAX_WORKERS = int(os.getenv("JAX_PROCESAMIENTO_MAX_WORKERS", "4"))
_EXECUTOR_OCR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="procesamiento-ocr")

# `_EXECUTOR_IO`: leer/escribir el JSON del resultado -- SEPARADO del OCR a
# propósito (N-4). Antes, `GET` sobre un trabajo YA TERMINADO podía esperar
# detrás de horas de cola de OCR real (medido en la revisión: 37s) porque
# la lectura usaba el MISMO pool que el procesamiento pesado. Chico porque
# es E/S liviana (un `read_text`/`write_text` de un JSON, nunca un
# subprocess), no CPU-bound.
_MAX_WORKERS_IO = int(os.getenv("JAX_PROCESAMIENTO_MAX_WORKERS_IO", "2"))
_EXECUTOR_IO = ThreadPoolExecutor(max_workers=_MAX_WORKERS_IO, thread_name_prefix="procesamiento-io")

# Como máximo tantos TRABAJOS corriendo a la vez como hilos reales tiene el
# pool de OCR -- sin lugar, 429 ANTES de crear ningún registro.
_SEMAFORO_TRABAJOS = asyncio.Semaphore(_MAX_WORKERS)

# Tope estructural de archivos por pedido: un solo request no puede
# monopolizar el pool durante horas. Con 4 hilos y documentos de ~30
# páginas (~80s cada uno a 2,7s/página), 50 archivos son, en el peor caso,
# (50/4)*80s ≈ 1000s (~17 min) -- contra las 2,2h que ocupaba UN hilo antes
# de B-2. El pedido puede partirse en varios `POST`.
_MAX_RUTAS_POR_TRABAJO = int(os.getenv("JAX_PROCESAMIENTO_MAX_RUTAS", "50"))

# MINOR-6: tope de longitud de `proyecto` -- sin esto, un `proyecto`
# arbitrariamente largo infla el JSONL append-only sin límite (cada
# `update()` re-esparce el estado ENTERO).
_MAX_PROYECTO_LEN = 200

#: B-6: `usuario` no vacío, tope de largo -- antes `""` y un string de 2MB
#: daban 202 los dos, y el de 2MB inflaba el JSONL en 4MB (se re-esparce
#: el estado ENTERO en cada `update()`, así que un campo largo se duplica
#: en cada línea).
_MAX_USUARIO_LEN = 200

router = APIRouter(prefix="/procesamiento", tags=["procesamiento"])


# ---------------------------------------------------------------------------
#  Modelos
# ---------------------------------------------------------------------------
class TrabajoRequest(BaseModel):
    proyecto: str
    rutas: list[str]
    # B-6: principal obligatorio, no vacío, con tope -- jax-platform ya
    # tiene el JWT del usuario; que lo pase. No es autorización completa
    # (eso es otra ronda): es lo mínimo para que un IDOR sea investigable.
    usuario: str = Field(min_length=1, max_length=_MAX_USUARIO_LEN)
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


def _codificable_utf8(s: str) -> bool:
    """N-2: `True` si `s` se puede codificar a UTF-8 sin pérdida. Un
    surrogate solitario (`\\udcXX`, típico de un nombre de archivo en una
    codificación distinta -- cp1252, latin-1 -- que Python re-expone así
    vía `surrogateescape` al leer el filesystem) es un `str` Python
    perfectamente válido, pero NINGÚN `.encode('utf-8')` estricto lo
    acepta -- y `JobStore`/`write_result` escriben JSON como UTF-8."""
    try:
        s.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


def _saneado_ascii(s: str) -> str:
    """Texto ASCII puro, legible y determinista: todo lo que no es ASCII
    -- una tilde válida (`é` -> `\\xe9`) o un surrogate solitario
    (`\\udcff`) -- sale como su escape. Usado del lado de ESCRITURA
    (`_resultado_no_codificable`, N-2) y del de LECTURA
    (`_leer_resultados_de_disco`, MINOR-C, ronda 4).

    Ronda 6: antes codificaba a UTF-8 con `backslashreplace` y decodificaba
    como ASCII, y este docstring decía "sin excepción posible". Era falso:
    `backslashreplace` sólo escapa lo que el codec NO puede codificar, así
    que una tilde válida quedaba como bytes UTF-8 y el `decode("ascii")`
    reventaba. `Crédito\\udcff.pdf` tumbaba el lote entero. Codificando a
    ASCII, `backslashreplace` escapa TODO lo no-ASCII y el resultado es
    ASCII por construcción. `surrogatepass` quedó descartado: produce UTF-8
    inválido que revienta más adelante, en otro lado."""
    return s.encode("ascii", errors="backslashreplace").decode("ascii")


def _resultado_no_codificable(ruta: str) -> ResultadoArchivo:
    """N-2: se rechaza ANTES de tocar el executor (nunca paga un hilo real
    por algo que ni se puede reportar), y el valor que se guarda en
    `archivo` va SANEADO -- para que ESTA entrada nunca pueda tumbar la
    escritura del resultado del LOTE ENTERO (`write_result` serializa
    TODOS los resultados juntos, un solo campo envenenado revienta el
    JSON de todos)."""
    return ResultadoArchivo(
        archivo=_saneado_ascii(ruta), estado="rechazado",
        error="el nombre de la ruta no se puede codificar a UTF-8 (caracteres inválidos)",
    )


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


# ---------------------------------------------------------------------------
#  N-3: control de cancelación cooperativo
# ---------------------------------------------------------------------------
class _ControlTrabajo:
    """Estado compartido entre `cancelar_trabajo()` (que corre en OTRA
    invocación HTTP, en paralelo) y el worker real de ESTE job.

    En Python no se mata un hilo del sistema operativo. Y `concurrent.
    futures.Future.cancel()` (vía la envoltura de `asyncio.run_in_executor`)
    **no sirve para detectar esto**: verificado a mano que, llamado sobre
    un `run_in_executor` que YA está corriendo de verdad en su hilo,
    `.cancel()` devuelve `True` y `.cancelled()` se pone en `True` de
    INMEDIATO -- mintiendo -- mientras el hilo real sigue corriendo en
    segundo plano y su resultado (que nadie espera ya) se descarta en
    silencio. Por eso acá NO se cancela ningún `Future`: cada archivo
    AUTOCONSULTA Y MARCA `RUNNING` en una sola operación atómica
    (`arrancar_o_marcar_running()`), en el instante justo antes de arrancar
    de verdad -- el que ya pasó ese chequeo sigue hasta el final pase lo
    que pase después; el que todavía no le tocó turno en el pool se salta
    sin tocar el disco.

    MINOR-D (ronda 4): "decidir si arranca" y "marcar RUNNING" eran DOS
    operaciones separadas -- `arrancar_o_saltar()` bajo el lock, después
    `_marcar_running_una_vez()` bajo OTRO lock (el `threading.Event`).
    Entre las dos había una ventana real: un `cancelar()` que llegaba justo
    ahí dejaba el store en `CANCELLING` y la escritura de `RUNNING`, que
    seguía de largo, lo pisaba -- el estado iba PARA ATRÁS (de `cancelling`
    a `running`), visto en rojo 3 de 3 corridas. `cancelar()` ahora también
    escribe `CANCELLING` en el store, bajo el MISMO lock que
    `arrancar_o_marcar_running()` -- las dos operaciones (decidir+marcar
    RUNNING, y cancelar+marcar CANCELLING) son mutuamente excluyentes."""
    __slots__ = ("cancelado", "marcado_running", "_lock", "_store", "_job_id")

    def __init__(self, store: JobStore, job_id: str) -> None:
        self.cancelado = False
        self.marcado_running = False
        self._lock = threading.Lock()
        self._store = store
        self._job_id = job_id

    def cancelar(self) -> None:
        """Llamada desde `cancelar_trabajo()` en un hilo de `_EXECUTOR_IO`,
        nunca desde el loop de eventos (tomar este lock puede significar
        esperar una escritura a disco del hilo que lo retiene -- N1, ronda
        5). Bajo el MISMO lock que `arrancar_o_marcar_running()` -- ver
        MINOR-D. Es el ÚNICO lugar que escribe `CANCELLING` mientras el
        trabajo tiene un control vivo."""
        with self._lock:
            if self.cancelado:
                return  # idempotente -- no reescribe CANCELLING de más
            self.cancelado = True
            self._store.update(self._job_id, status=JobStatus.CANCELLING.value)

    def arrancar_o_marcar_running(self) -> bool:
        """`True`: este archivo puede arrancar de verdad -- y, si es el
        primero de este trabajo, ya dejó `RUNNING` escrito en el store,
        bajo el mismo lock que decidió que podía arrancar (MINOR-D: sin
        esto, `cancelar()` podía colarse ENTRE decidir y marcar, y la
        escritura de `RUNNING` pisaba el `CANCELLING` que se acababa de
        pedir). `False`: el trabajo ya estaba cancelado ANTES de que le
        tocara el turno -- nunca se paga un hilo real por él, y nunca se
        escribe `RUNNING`."""
        with self._lock:
            if self.cancelado:
                return False
            if not self.marcado_running:
                self.marcado_running = True
                self._store.update(
                    self._job_id, status=JobStatus.RUNNING.value, started_at=time.time(),
                )
            return True


#: job_id -> control vivo, SOLO mientras `_ejecutar_trabajo` está corriendo
#: (se registra al empezar, se saca en el `finally`). Permite que
#: `cancelar_trabajo()`, que corre en otra invocación, encuentre el
#: control de ESE job para pedirle que deje de programar archivos nuevos.
_CONTROLES: dict[str, _ControlTrabajo] = {}


async def _procesar_rutas_paralelo(
    executor: ThreadPoolExecutor, proyecto: str, rutas: list[str],
    *, control: _ControlTrabajo,
) -> list[dict]:
    """Cada RUTA CODIFICABLE es su propia unidad de trabajo en el pool de
    OCR -- un job con muchos archivos se reparte entre los hilos del pool
    en vez de monopolizar uno solo (B-2). El orden del resultado sigue el
    orden de `rutas`, sin importar en qué orden terminen los hilos.
    `control` ya sabe a qué `store`/`job_id` pertenece (ver
    `_ControlTrabajo`), así que esta función no necesita esos dos
    parámetros aparte."""
    trabajo = _trabajo_de(proyecto)
    loop = asyncio.get_running_loop()

    resultados: list[ResultadoArchivo | None] = [None] * len(rutas)
    indices_pendientes: list[int] = []
    for i, ruta in enumerate(rutas):
        if not _codificable_utf8(ruta):  # N-2: rechazo ANTES del executor
            resultados[i] = _resultado_no_codificable(ruta)
        else:
            indices_pendientes.append(i)

    def _trabajo_de_un_archivo(ruta: str) -> ResultadoArchivo:
        # N-3/MINOR-D: decidir y marcar RUNNING son UNA sola operación
        # atómica -- ver `_ControlTrabajo.arrancar_o_marcar_running`.
        if not control.arrancar_o_marcar_running():
            return ResultadoArchivo(
                archivo=ruta, estado="cancelado",
                error="cancelado antes de empezar a procesarse",
            )
        return _procesar_una_ruta(trabajo, ruta)

    if indices_pendientes:
        futuros = [
            loop.run_in_executor(executor, _trabajo_de_un_archivo, rutas[i])
            for i in indices_pendientes
        ]
        completados = await asyncio.gather(*futuros, return_exceptions=True)
        for i, resultado in zip(indices_pendientes, completados):
            if isinstance(resultado, BaseException):
                # Defensivo: `_procesar_una_ruta` ya atrapa sus propias
                # excepciones -- esto no debería dispararse nunca, pero si
                # lo hace, no se pierde el resto del lote por eso.
                logger.error(
                    "procesamiento: resultado inesperado para '%s': %r",
                    rutas[i], resultado,
                )
                resultados[i] = ResultadoArchivo(
                    archivo=rutas[i], estado="error", error=str(resultado),
                )
            else:
                resultados[i] = resultado

    return [r.model_dump() for r in resultados]  # type: ignore[union-attr]


def _guardar_resultado(store: JobStore, job_id: str, resultados: list[dict]) -> str:
    """N-2, defensa de última línea: aunque los nombres de archivo YA
    vienen saneados (`_resultado_no_codificable`), esta escritura no puede
    tumbar el lote por un problema de codificación bajo NINGUNA
    circunstancia. `ensure_ascii=True` es la garantía dura -- texto ASCII
    puro SIEMPRE se puede escribir como UTF-8, sin excepción posible."""
    try:
        return store.write_result(job_id, json.dumps(resultados, ensure_ascii=False))
    except UnicodeEncodeError as exc:
        logger.error(
            "procesamiento: job %s -- el resultado no era codificable a UTF-8 "
            "(%r), reintentando con ensure_ascii=True", job_id, exc,
        )
        return store.write_result(job_id, json.dumps(resultados, ensure_ascii=True))


# ---------------------------------------------------------------------------
#  El worker del job -- toma `store`/`executor`/`executor_io`/`semaforo`
#  explícitos (mismo patrón testable que `motor_registry.worker.run`).
#
#  CONTRATO: quien llama a esta función YA tiene que haber adquirido
#  `semaforo` (una unidad) ANTES de crear la tarea -- esta función lo
#  libera en un `finally`, y SÓLO cuando el trabajo (incluida cualquier
#  cancelación en curso) terminó de verdad -- nunca antes de que los
#  hilos en vuelo de este job hayan terminado (N-3).
# ---------------------------------------------------------------------------
async def _ejecutar_trabajo(
    job_id: str, proyecto: str, rutas: list[str], *, store: JobStore,
    executor: ThreadPoolExecutor | None = None,
    executor_io: ThreadPoolExecutor | None = None,
    semaforo: asyncio.Semaphore | None = None,
    control: _ControlTrabajo | None = None,
) -> None:
    executor = executor if executor is not None else _EXECUTOR_OCR
    executor_io = executor_io if executor_io is not None else _EXECUTOR_IO
    semaforo = semaforo if semaforo is not None else _SEMAFORO_TRABAJOS
    # Ronda 6: en producción el control llega YA creado y registrado desde
    # `crear_trabajo()`, antes de programar esta tarea. Si lo creara el
    # worker al arrancar, un cancel que llegara antes de esa primera vuelta
    # del loop no lo encontraría y se perdería (ver `crear_trabajo`). Crearlo
    # acá queda sólo para quien llama al worker directo (tests, medición).
    if control is None:
        control = _ControlTrabajo(store, job_id)
    _CONTROLES[job_id] = control
    try:
        try:
            resultados = await _procesar_rutas_paralelo(
                executor, proyecto, rutas, control=control,
            )
            loop = asyncio.get_running_loop()
            result_path = await loop.run_in_executor(
                executor_io, _guardar_resultado, store, job_id, resultados,
            )
            por_estado: dict[str, int] = {}
            for r in resultados:
                por_estado[r["estado"]] = por_estado.get(r["estado"], 0) + 1
            # N-3: si se pidió cancelar en algún momento, el estado FINAL
            # es CANCELLED (terminal) -- nunca COMPLETED, aunque algunos
            # archivos hayan terminado bien antes del pedido.
            estado_final = (
                JobStatus.CANCELLED.value if control.cancelado else JobStatus.COMPLETED.value
            )
            store.update(
                job_id, status=estado_final, finished_at=time.time(),
                result_path=result_path,
                result_summary=f"{len(resultados)} archivo(s): {por_estado}"[:200],
            )
        except Exception as exc:
            logger.error("procesamiento: job %s falló: %r", job_id, exc)
            store.update(
                job_id, status=JobStatus.FAILED.value, finished_at=time.time(),
                error=str(exc),
            )
    finally:
        # El semáforo se libera SIEMPRE, y SÓLO ACÁ -- después de que
        # `_procesar_rutas_paralelo` (que espera a los hilos en vuelo, no
        # los abandona) haya terminado de verdad. Nunca antes: eso sería
        # exactamente la mentira que N-3 vino a cerrar.
        _CONTROLES.pop(job_id, None)
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
    """N-4: corre en `_EXECUTOR_IO`, nunca en el loop de eventos NI en el
    pool de OCR (antes compartía executor con el OCR -- medido en la
    revisión: consultar un trabajo YA TERMINADO tardó 37s porque la
    lectura esperaba detrás de la cola real de tesseract).

    MINOR-C (ronda 4): `_guardar_resultado` garantiza que lo que queda en
    disco SIEMPRE se pudo escribir (su fallback `ensure_ascii=True`
    escapa cualquier surrogate solitario como texto ASCII) -- pero
    `json.loads` DESESCAPA ese texto y reconstruye el surrogate original
    en memoria. Si ese surrogate llegó por un camino que N-2 no sanea (un
    mensaje de excepción con un nombre de archivo crudo, por ejemplo, no
    sólo `archivo`), la respuesta HTTP (Starlette `JSONResponse`, que
    revienta con esto -- ver el hallazgo de N-1 en la ronda anterior)
    daría 500 en CADA `GET` futuro sobre ese job, dejando los archivos
    SANOS del mismo lote inaccesibles para siempre. Se sanea acá, del
    lado de LECTURA, cualquier valor `str` de cualquier campo -- no sólo
    `archivo` -- antes de que llegue a construir la respuesta."""
    try:
        crudos = json.loads(Path(result_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # fail-soft: el job YA terminó, un resultado ilegible no tumba la consulta de estado
        logger.error("procesamiento: no se pudo leer el resultado en '%s': %r", result_path, exc)
        return []
    if not isinstance(crudos, list):
        logger.error("procesamiento: resultado con forma inesperada en '%s' (no es una lista)", result_path)
        return []
    return [
        {k: (_saneado_ascii(v) if isinstance(v, str) and not _codificable_utf8(v) else v)
         for k, v in item.items()}
        if isinstance(item, dict) else item
        for item in crudos
    ]


async def _construir_respuesta_estado(job_id: str, view) -> TrabajoEstadoResponse:
    resultados: list[ResultadoArchivo] = []
    if view.result_path:
        loop = asyncio.get_running_loop()
        crudos = await loop.run_in_executor(_EXECUTOR_IO, _leer_resultados_de_disco, view.result_path)
        try:
            resultados = [ResultadoArchivo(**r) for r in crudos]
        except (TypeError, ValueError) as exc:
            logger.error("procesamiento: resultado con forma inesperada para %s: %r", job_id, exc)
    return TrabajoEstadoResponse(
        job_id=job_id, estado=view.status.value, error=view.error, resultados=resultados,
    )


# ---------------------------------------------------------------------------
#  Reconciliación al arrancar (B-3, cerrado en la ronda anterior -- sin
#  cambios acá)
# ---------------------------------------------------------------------------
def reconciliar_trabajos_huerfanos(store: JobStore | None = None) -> int:
    """Al arrancar LAS MANOS: cualquier job `pending`/`running`/`cancelling`
    en el JSONL es, por definición, huérfano -- este proceso recién
    arrancó, así que ninguna tarea viva puede estar trabajando en él.
    `JobStore._load()` reconstruye el índice desde disco pero nunca
    reconcilia estados "en vuelo" contra tareas reales -- sin este paso, un
    reinicio del servicio deja el trabajo así PARA SIEMPRE. Se llama una
    sola vez desde el startup hook de `server.py` (mismo criterio que
    `jacobs.reaper.reap_orphaned_pipelines`)."""
    store = store if store is not None else _STORE
    huerfanos = store.ids_en_estado(
        JobStatus.PENDING.value, JobStatus.RUNNING.value, JobStatus.CANCELLING.value,
    )
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
    proyecto = req.proyecto[:_MAX_PROYECTO_LEN]  # MINOR-6
    # MINOR-B: validado ANTES del `acquire()` -- `_STORE.create()` corre
    # DESPUÉS de tomar el semáforo, y un `proyecto` no codificable a UTF-8
    # lo hace reventar ahí (ver N-1: el `try/finally` de más abajo absorbe
    # esa falla sin filtrar el permiso, pero un registro `pending` que
    # nunca avanza -- `create()` YA escribió, `update(proyecto=...)` es
    # quien revienta -- quedaba de todos modos). Rechazar acá cierra el
    # vector en el origen: ni se toca el semáforo, ni queda un registro a
    # medias.
    if not _codificable_utf8(proyecto):
        raise HTTPException(
            status_code=422,
            detail="el proyecto no se puede codificar a UTF-8 (caracteres inválidos)",
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
    # vio libre -- atómico dentro de este turno del loop de eventos.
    await _SEMAFORO_TRABAJOS.acquire()

    # N-1: desde acá hasta que la tarea quede registrada, CUALQUIER
    # excepción tiene que liberar el permiso -- antes, si `_STORE.create()`
    # lanzaba, el semáforo quedaba tomado PARA SIEMPRE. Cuatro pedidos así
    # agotaban los 4 permisos y TODO POST limpio recibía 429 hasta
    # reiniciar el proceso -- un DoS de cuatro requests. `proyecto` ya no
    # puede ser la causa (validado arriba), pero el `try/finally` se queda
    # como defensa general: cualquier otra falla en este tramo (ej. el
    # propio `usuario`, si algún día pierde su `Field`) tiene que seguir
    # liberando el permiso.
    permiso_transferido = False
    job_id: str | None = None
    try:
        job_id = _STORE.create(
            # B-6: el principal REAL -- antes era la constante
            # "las_manos.procesamiento", que no identificaba a nadie.
            caller=req.usuario,
            capability="ingesta_archivos",
            # `motor`/`prompt` son vocabulario de JobStore para motores LLM
            # -- este job no despacha ningún motor. Ver la limitación
            # documentada al principio del archivo.
            motor="n/a_no_es_un_motor_llm",
            trace_id=str(uuid.uuid4()),
            prompt="n/a -- este job no despacha un motor LLM, ver el campo 'proyecto'",
            recursion_depth=0,
        )
        # El campo que de verdad significa "proyecto" -- `update()` acepta
        # kwargs arbitrarios (van tal cual al JSONL).
        _STORE.update(job_id, proyecto=proyecto)

        # Ronda 6: el control se crea y se registra ANTES de programar la
        # tarea. Antes lo creaba el worker en su primera vuelta del loop; un
        # `POST .../cancel` que llegaba en ese hueco no lo encontraba,
        # escribía CANCELLING a pelo, y el worker arrancaba después con
        # `cancelado=False`. Historial medido: `pending, pending,
        # cancelling, running, completed`, con la cancelación perdida. Con
        # el control creado acá, ningún cancel puede llegar antes que él: la
        # ventana no existe.
        control = _ControlTrabajo(_STORE, job_id)
        _CONTROLES[job_id] = control
        task = asyncio.create_task(
            _ejecutar_trabajo(job_id, proyecto, req.rutas, store=_STORE, control=control)
        )
        # MINOR-A: la bandera va ACÁ, apenas se creó la tarea -- no
        # después de `add_done_callback`/`job_tasks.register`. Antes,
        # si CUALQUIERA de esas dos llamadas lanzaba, la tarea YA creada
        # iba a liberar el semáforo sola (su propio `finally`) Y este
        # `finally` de acá TAMBIÉN lo liberaba (`permiso_transferido`
        # seguía en `False`) -- doble liberación, el semáforo sube por
        # encima de su capacidad real (medido: 4 -> 7). A partir de esta
        # línea, `_ejecutar_trabajo` es la ÚNICA responsable.
        permiso_transferido = True
        task.add_done_callback(lambda t: _log_worker_exception(t, job_id=job_id))
        job_tasks.register(job_id, task)
        return TrabajoCreadoResponse(job_id=job_id)
    finally:
        if not permiso_transferido:
            # Nadie va a correr el `finally` del worker: el control
            # registrado arriba (si se llegó a registrar) se saca acá.
            if job_id is not None:
                _CONTROLES.pop(job_id, None)
            _SEMAFORO_TRABAJOS.release()


@router.get("/trabajos/{job_id}", response_model=TrabajoEstadoResponse)
async def estado_trabajo(job_id: str) -> TrabajoEstadoResponse:
    view = _STORE.get(job_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"Trabajo '{job_id}' no encontrado")
    return await _construir_respuesta_estado(job_id, view)


@router.post("/trabajos/{job_id}/cancel", response_model=TrabajoEstadoResponse)
async def cancelar_trabajo(job_id: str) -> TrabajoEstadoResponse:
    """N-3: cancelar es honesto sobre lo que Python puede hacer. NO mata
    ningún hilo -- marca `CANCELLING` y marca `control.cancelado`, que cada
    archivo AUTOCONSULTA justo antes de arrancar de verdad (ver
    `_ControlTrabajo`): el que todavía no le tocó turno en el pool se
    salta sin tocar el disco. Lo que YA está corriendo de verdad (un
    `tesseract`/`pdftoppm` en vuelo) sigue hasta terminar solo -- no se le
    llama `.cancel()` a su `Future`, porque eso miente (ver el docstring
    de `_ControlTrabajo`). Recién cuando el último hilo en vuelo de este
    job termina, `_ejecutar_trabajo` marca `CANCELLED` (terminal) y libera
    el permiso del semáforo -- nunca antes, o la admisión estaría
    mintiendo sobre cuánta capacidad real hay libre."""
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
    if view.status != JobStatus.CANCELLING:
        control = _CONTROLES.get(job_id)
        if control is not None:
            # N1 (ronda 5): `CANCELLING` lo escribe `control.cancelar()`,
            # BAJO el mismo lock que `arrancar_o_marcar_running()` -- antes
            # se escribía también acá, FUERA del lock, y con un hilo dentro
            # de la escritura de `running` el historial quedaba
            # `cancelling, running, cancelling` (para atrás). Y corre en
            # `_EXECUTOR_IO`, no en el loop: tomar ese `threading.Lock`
            # puede implicar esperar a que el hilo termine una escritura a
            # disco, y eso congelaba el loop de eventos entero.
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_EXECUTOR_IO, control.cancelar)
        else:
            # Sin control vivo no hay lock que compartir. Desde la ronda 6,
            # el control existe desde `crear_trabajo()` hasta el `finally`
            # del worker, así que esto sólo alcanza a un job no terminal sin
            # tarea viva (un huérfano que la reconciliación del arranque
            # todavía no marcó).
            _STORE.update(job_id, status=JobStatus.CANCELLING.value)
    return await _construir_respuesta_estado(job_id, _STORE.get(job_id))
