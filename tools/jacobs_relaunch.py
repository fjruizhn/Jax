#!/usr/bin/env python3
"""
Jacobs — relanzador de pipelines abortados o vencidos (CLI).

Desde 2026-09-17 (spec prevuelo-y-continuar §5.4) NO tiene lógica propia:
llama a jacobs/continuar.py::continuar, la MISMA función que
POST /jacobs/pipeline/{id}/continue, y después corre la corrida en este
proceso. Los tres agujeros de la versión anterior se cierran por construcción:
  1. usaba la foto de `plan` de la creación -> continuar lee jacobs_steps;
  2. no borraba las refs de los pasos a rehacer -> continuar las quita;
  3. se saltaba el candado y el límite -> continuar los aplica. El candado es
     de proceso: entre este CLI y LAS MANOS serializa la transacción con
     SELECT ... FOR UPDATE y la época (desvío 7 del plan).

`--from-step` ya no existe: qué se reusa lo decide la ref legible de cada
paso, no un número elegido a mano.

Uso:
    python tools/jacobs_relaunch.py <pipeline_id> [--reasignar 4=ada ...] [--costo-max-aceptado 1.50]

Código de salida:
    0 pipeline completado
    1 continuar() lo rechazó (403/404/409/422/423/429; el motivo se imprime)
    2 error inesperado en el análisis/pre-vuelo (DB caída, faceta o credencial
      no disponible): mismo criterio de fail-closed que jacobs/routes.py::
      _no_disponible (503 prevuelo_no_disponible) -- acá no hay un pipeline
      corriendo, así que no hay un status HTTP que devolver: el motivo se
      imprime redactado y recortado, y NO se corre nada
    3 la corrida terminó, pero el pipeline no quedó `completed`

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ENV_PATH = "/etc/jax/.env"


def _cargar_env(ruta: str = ENV_PATH) -> None:
    """/etc/jax/.env -> os.environ sin pisar lo ya seteado. SOLO desde main():
    antes corría al importar el módulo, y cualquier import (un test) quedaba
    con el entorno de producción cargado."""
    if not os.path.exists(ruta):
        return
    with open(ruta, encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            clave, valor = linea.split("=", 1)
            os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


def _asegurar_rutas() -> None:
    """Raíz del repo (jacobs) y las_manos (facet_resolver, contrato_dispatch,
    motor_registry), como corre LAS MANOS. Antes era ~/jax fijo: desde un
    worktree importaba el código del checkout de producción."""
    for ruta in (str(RAIZ), str(RAIZ / "las_manos")):
        if ruta not in sys.path:
            sys.path.insert(0, ruta)


def parsear_reasignar(valores: list[str]) -> dict[str, str]:
    salida: dict[str, str] = {}
    for valor in valores:
        indice, separador, faceta = valor.partition("=")
        if not separador or not indice.strip().isdigit() or not faceta.strip():
            raise argparse.ArgumentTypeError(
                f"--reasignar espera PASO=FACETA (por ejemplo 4=ada); llegó {valor!r}")
        salida[indice.strip()] = faceta.strip()
    return salida


def _costo(valor: str) -> Decimal:
    try:
        costo = Decimal(valor)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"--costo-max-aceptado no es un número: {valor!r}") from exc
    if costo < 0:
        raise argparse.ArgumentTypeError("--costo-max-aceptado no puede ser negativo")
    return costo


async def relanzar(pipeline_id: str, reasignar: dict[str, str], costo_max_aceptado: Decimal | None) -> int:
    from jacobs import continuar, store
    from jacobs.executor import run_pipeline
    from jacobs.models import INVOKER_PLATAFORMA, PipelineStatus
    from jax.core.redaccion import recortar_redactado

    try:
        respuesta, pipeline = await continuar.continuar(
            pipeline_id, INVOKER_PLATAFORMA,
            reasignar=reasignar or None, costo_max_aceptado_usd=costo_max_aceptado,
        )
    except continuar.ContinuarRechazado as rechazo:
        print(f"✗ {rechazo.status_code} {json.dumps(rechazo.cuerpo(), ensure_ascii=False)}")
        return 1
    except Exception as exc:  # fail-closed: sin análisis/pre-vuelo no se continúa (mismo criterio que jacobs/routes.py::_no_disponible); no se corre nada
        motivo = recortar_redactado(f"{type(exc).__name__}: {exc}", 300)
        print(f"✗ error inesperado, no se continuó: {motivo}")
        return 2

    print(
        f"▶ Continuando {pipeline_id} (época {respuesta['run_epoch']}): "
        f"reusados={respuesta['pasos_reusados']} a correr={respuesta['pasos_a_correr']} "
        f"costo máximo=US$ {respuesta['costo_max_usd']}"
    )
    await run_pipeline(pipeline)

    final = await store.pipeline_get(pipeline_id)
    for s in await store.steps_by_pipeline(pipeline_id):
        duracion = round(s.finished_at - s.started_at, 1) if s.started_at and s.finished_at else None
        print(f"  idx={s.step_index} {s.facet}/{s.capability} status={s.status.value} "
              f"timeout={s.timeout_seconds} out={(s.output_ref or '')[:11]} dur={duracion} err={s.error}")
    print(f"=== FINAL: pipeline status={final.status.value} ===")
    return 0 if final.status == PipelineStatus.completed else 3


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Continúa un pipeline de Jacobs abortado o vencido (misma función que el endpoint).")
    parser.add_argument("pipeline_id")
    parser.add_argument("--reasignar", action="append", default=[], metavar="PASO=FACETA")
    parser.add_argument("--costo-max-aceptado", type=_costo, default=None, metavar="USD")
    args = parser.parse_args(argv)
    try:
        reasignar = parsear_reasignar(args.reasignar)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    _cargar_env()
    _asegurar_rutas()
    sys.exit(asyncio.run(relanzar(args.pipeline_id, reasignar, args.costo_max_aceptado)))


if __name__ == "__main__":
    main()
