#!/usr/bin/env python3
"""
Jacobs — Relanzador de pipelines desde un step dado.

Reanuda un pipeline que quedó en aborted/failed re-ejecutando desde --from-step,
SIN repetir los steps anteriores (sus outputs ya están en context_refs / jacobs_steps).
Re-aplica el timeout por capability definido en jacobs.plan (fuente única de verdad),
así un pipeline viejo creado antes del fix hereda el timeout corregido.

Uso:
    python tools/jacobs_relaunch.py <pipeline_id> --from-step 8

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

# /etc/jax/.env -> os.environ (sin pisar lo ya seteado por systemd)
_ENV_PATH = "/etc/jax/.env"
if os.path.exists(_ENV_PATH):
    for _line in open(_ENV_PATH):
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

sys.path.insert(0, os.path.expanduser("~/jax"))

from jacobs import store  # noqa: E402
from jacobs.executor import run_pipeline  # noqa: E402
from jacobs.models import PipelineStatus, StepStatus  # noqa: E402
from jacobs.plan import _techo_segundos  # noqa: E402
from jacobs.policy import check_kill_switch  # noqa: E402


async def relaunch(pipeline_id: str, from_step: int) -> int:
    if check_kill_switch():
        print("✗ Kill switch activo (/etc/jax/PAUSE). Abortando.")
        return 2

    pipeline = await store.pipeline_get(pipeline_id)
    if pipeline is None:
        print(f"✗ Pipeline '{pipeline_id}' no encontrado.")
        return 1

    print(f"Pipeline: {pipeline.name}  status={pipeline.status.value}  steps={len(pipeline.plan)}")
    if from_step < 0 or from_step >= len(pipeline.plan):
        print(f"✗ --from-step {from_step} fuera de rango [0, {len(pipeline.plan)-1}].")
        return 1

    # Verificar que las dependencias de los steps a re-ejecutar tienen su ref en context.
    missing = []
    for step in pipeline.plan[from_step:]:
        for dep in step.depends_on:
            if dep < from_step and not pipeline.context.get(f"step_{dep}_ref"):
                missing.append((step.step_index, dep))
    if missing:
        print(f"✗ Faltan refs de dependencias ya completadas: {missing}. No se puede reanudar sin re-ejecutar.")
        return 1

    # Re-aplicar timeout por capability (corrige planes viejos) y resetear estado
    # de los steps a re-ejecutar (from_step en adelante).
    # El timeout por capability sale de la DB (fuente unica desde 2026-09-01);
    # antes salia de un dict en jacobs/plan.py que duplicaba esa misma columna.
    caps = (await store.get_motor_governance())["capabilities"]
    for step in pipeline.plan:
        if step.step_index < from_step:
            continue
        new_timeout, _ = _techo_segundos(caps, step.capability)
        if new_timeout and step.timeout_seconds != new_timeout:
            print(f"  step {step.step_index} ({step.capability}): timeout {step.timeout_seconds} -> {new_timeout}")
            step.timeout_seconds = new_timeout
        step.status = StepStatus.pending
        step.error = None
        step.started_at = None
        step.finished_at = None
        step.output_ref = None
        await store.step_upsert(step)

    pipeline.current_step_index = from_step
    pipeline.status = PipelineStatus.running
    await store.pipeline_update_status(
        pipeline_id, PipelineStatus.running, from_step, pipeline.context
    )
    await store.event_append(
        pipeline_id, "PIPELINE_RELAUNCHED",
        {"from_step": from_step, "by": "tools/jacobs_relaunch.py"},
    )

    print(f"▶ Reanudando desde step {from_step}...")
    await run_pipeline(pipeline)

    # Estado final
    final = await store.pipeline_get(pipeline_id)
    steps = await store.steps_by_pipeline(pipeline_id)
    print(f"\n=== FINAL: pipeline status={final.status.value} ===")
    for s in steps:
        dur = round(s.finished_at - s.started_at, 1) if s.started_at and s.finished_at else None
        kind = (s.output_ref or "")[:11]
        print(f"  idx={s.step_index} {s.facet}/{s.capability} status={s.status.value} "
              f"timeout={s.timeout_seconds} out={kind} dur={dur} err={s.error}")
    return 0 if final.status == PipelineStatus.completed else 3


def main() -> None:
    ap = argparse.ArgumentParser(description="Relanza un pipeline de Jacobs desde un step.")
    ap.add_argument("pipeline_id")
    ap.add_argument("--from-step", type=int, required=True)
    args = ap.parse_args()
    rc = asyncio.run(relaunch(args.pipeline_id, args.from_step))
    sys.exit(rc)


if __name__ == "__main__":
    main()
