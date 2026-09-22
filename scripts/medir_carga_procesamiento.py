#!/usr/bin/env python3
"""Medición de carga del endpoint de Procesamiento de Archivos (B-2/N-4/N-5).

Corrección de método (ronda 4, hallazgo del coordinador sobre la ronda
anterior): el script vivía bajo `.superpowers/sdd/`, que un `.gitignore`
propio tapa -- "queda en el repo" era falso, se perdía al borrar el
worktree. Vive acá, en `scripts/`, fuera de cualquier `.gitignore`.

Y, más importante, **mide el módulo real**, no `ocr.extraer()` suelto:

  - Llama a `procesamiento_routes._ejecutar_trabajo()` de verdad, con un
    `JobStore` real (tempdir) y archivos reales en un workspace real
    (tempdir, `tool_authority.WORKSPACE_ROOT` parcheado) -- exactamente el
    camino que corre un `POST /procesamiento/trabajos` real, salvo la capa
    HTTP en sí.
  - La sección de aislamiento NO arma su propio `ThreadPoolExecutor` de
    control (como hacía la versión anterior, que por eso "no podía fallar"
    aunque el módulo usara `None`): el trabajo real se dispara SIN pasarle
    `executor=` -- usa el default real del módulo
    (`procesamiento_routes._EXECUTOR_OCR`) -- y el "ajeno" es
    `asyncio.to_thread(...)`, que es LITERALMENTE `run_in_executor(None,
    ...)` -- el mismo camino que usa `worker.py` para despachar motores.
    Si el módulo alguna vez cambia su `executor` real por `None` (la
    mutación de N-5), el trabajo de OCR terminaría en el MISMO pool que el
    ajeno, y este script lo mediría como demora real, no como aislamiento.

Corre con:
  PYTHONPATH=.:las_manos python3 scripts/medir_carga_procesamiento.py
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "las_manos"))

import procesamiento_routes as rutas_mod  # noqa: E402
from motor_registry import tool_authority  # noqa: E402
from motor_registry.job_store import JobStore  # noqa: E402
from motor_registry.models import JobStatus  # noqa: E402


def _fuente(tamano: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", tamano)
    except OSError:
        return None


def _pagina_sintetica(destino: Path, semilla: int, n_lineas: int) -> Path:
    """Página sintética -- texto genérico repetido con variación por
    línea, NUNCA contenido de cliente. `n_lineas` controla la densidad."""
    from PIL import Image, ImageDraw

    size = (1700, 2200)  # similar a 300 DPI de una hoja carta
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    fuente = _fuente(28)
    lineas = [
        f"Linea {semilla}-{i}: Estado financiero de prueba, cifra {i * 137 + semilla} "
        f"unidades monetarias, periodo {2020 + (i % 5)}."
        for i in range(n_lineas)
    ]
    y = 40
    paso = min(46, (size[1] - 80) // max(n_lineas, 1))
    for linea in lineas:
        d.text((40, y), linea, fill="black", font=fuente)
        y += paso
    img.save(destino)
    return destino


async def _medir_trabajo_real(n_archivos: int, n_lineas: int) -> tuple[float, str]:
    """Corre `_ejecutar_trabajo()` DE VERDAD -- el mismo camino que un
    `POST /procesamiento/trabajos` real, salvo la capa HTTP. Devuelve
    (segundos totales, estado final)."""
    with tempfile.TemporaryDirectory() as d:
        workspace = Path(d) / "workspace"
        workspace.mkdir()
        store = JobStore(str(Path(d) / "jobs.jsonl"))
        rutas = []
        for i in range(n_archivos):
            nombre = f"pagina-{i}.png"
            _pagina_sintetica(workspace / nombre, semilla=i, n_lineas=n_lineas)
            rutas.append(nombre)

        with patch.object(tool_authority, "WORKSPACE_ROOT", workspace.resolve()):
            job_id = store.create(
                caller="medicion", capability="ingesta_archivos", motor="n/a",
                trace_id="medicion", prompt="n/a", recursion_depth=0,
            )
            t0 = time.perf_counter()
            # Sin `executor=`/`executor_io=`: usa los defaults REALES del
            # módulo (`_EXECUTOR_OCR`/`_EXECUTOR_IO`) -- es la parte que la
            # versión anterior de este script no hacía.
            await rutas_mod._ejecutar_trabajo(
                job_id, "medicion-carga", rutas, store=store,
            )
            dt = time.perf_counter() - t0
        estado = store.get(job_id).status.value
        return dt, estado


async def _medir_aislamiento(n_archivos: int, n_lineas: int) -> tuple[float, float, str]:
    """Dispara un trabajo real (executor REAL del módulo) y, EN PARALELO,
    un `to_thread` ajeno (el mismo camino que usa `worker.py` para
    despachar motores -- `run_in_executor(None, ...)`). Mide cuánto espera
    el ajeno. Si el módulo alguna vez pasa `None` en vez de su executor
    dedicado (N-5), el trabajo real y el ajeno terminan en el MISMO pool
    y esto lo refleja como demora, no como aislamiento -- a propósito, sin
    armar ningún executor de control aparte para el trabajo en sí.

    El ÚNICO control que se arma es el executor POR DEFECTO de asyncio
    (`loop.set_default_executor`, API pública) fijado a un tamaño chico y
    conocido (2 hilos). Sin esto, en una máquina con muchos cores (este
    host: 32 -- el default de `ThreadPoolExecutor()` es `min(32,
    nproc+4)`), el pool por defecto tiene tanto margen que "aislado" y
    "no aislado" miden IGUAL aunque el módulo esté roto: sobra lugar para
    las 4 tareas de OCR igual. Con el default achicado, si el OCR cae ahí
    por error, compite de verdad por esos 2 cupos con el ajeno -- y esto
    SIGUE sin tocar el executor que el módulo usa cuando está bien
    (`_EXECUTOR_OCR`, propio, intacto)."""
    with tempfile.TemporaryDirectory() as d:
        workspace = Path(d) / "workspace"
        workspace.mkdir()
        store = JobStore(str(Path(d) / "jobs.jsonl"))
        rutas = []
        for i in range(n_archivos):
            nombre = f"pagina-{i}.png"
            _pagina_sintetica(workspace / nombre, semilla=i, n_lineas=n_lineas)
            rutas.append(nombre)

        with patch.object(tool_authority, "WORKSPACE_ROOT", workspace.resolve()):
            job_id = store.create(
                caller="medicion", capability="ingesta_archivos", motor="n/a",
                trace_id="medicion", prompt="n/a", recursion_depth=0,
            )
            loop = asyncio.get_running_loop()
            default_chico = ThreadPoolExecutor(max_workers=2, thread_name_prefix="medicion-default")
            loop.set_default_executor(default_chico)

            async def _ajeno() -> float:
                t0 = time.perf_counter()
                await asyncio.to_thread(time.sleep, 0)  # simula un git show/reset corto de worker.py
                return time.perf_counter() - t0

            t0 = time.perf_counter()
            trabajo_task = asyncio.create_task(
                rutas_mod._ejecutar_trabajo(job_id, "medicion-aislamiento", rutas, store=store)
            )
            await asyncio.sleep(0.05)  # deja que el trabajo real arranque de verdad
            ajeno_dt = await _ajeno()
            await trabajo_task
            total_dt = time.perf_counter() - t0
            default_chico.shutdown(wait=False)
        return ajeno_dt, total_dt, store.get(job_id).status.value


if __name__ == "__main__":
    version = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    print("=== Endpoint de Procesamiento -- medición de carga real (B-2/N-4/N-5) ===")
    print(f"tesseract: {version}")
    print(f"nproc: {__import__('os').cpu_count()}")
    print(f"_MAX_WORKERS (OCR): {rutas_mod._MAX_WORKERS}, _MAX_WORKERS_IO: {rutas_mod._MAX_WORKERS_IO}")

    print("\n1) Un trabajo real (_ejecutar_trabajo), 1 archivo, densidad ALTA (45 líneas):")
    dt, estado = asyncio.run(_medir_trabajo_real(1, 45))
    print(f"  {dt:.2f}s, estado final={estado}")

    print(f"\n2) Un trabajo real, {rutas_mod._MAX_WORKERS} archivos (densidad alta) EN PARALELO "
          f"(pool real, max_workers={rutas_mod._MAX_WORKERS}):")
    dt_paralelo, estado = asyncio.run(_medir_trabajo_real(rutas_mod._MAX_WORKERS, 45))
    print(f"  total: {dt_paralelo:.2f}s (vs {dt * rutas_mod._MAX_WORKERS:.2f}s si fuera secuencial), "
          f"estado final={estado}")

    print(f"\n3) Aislamiento: un trabajo real de {rutas_mod._MAX_WORKERS} archivos corriendo en el "
          "pool REAL del módulo -- ¿espera un to_thread ajeno (el mismo camino que worker.py)?")
    ajeno_dt, total_dt, estado = asyncio.run(_medir_aislamiento(rutas_mod._MAX_WORKERS, 45))
    print(f"  to_thread ajeno: {ajeno_dt * 1000:.1f} ms")
    print(f"  trabajo real total: {total_dt:.2f}s, estado final={estado}")
    print(f"  {'AISLADO -- el ajeno no esperó' if ajeno_dt < 0.2 else 'NO AISLADO -- el ajeno esperó detrás del trabajo real'}")
