# ============================================================================
#  REEMPLAZO PARA executor.py
#
#  Sustituye la función run_pipeline() COMPLETA (desde "async def run_pipeline"
#  hasta justo antes de "async def _persist_step_to_repo") por este bloque.
#
#  AÑADE además dos funciones nuevas: _compute_waves() y _run_one_step().
#  Colocalas ANTES de run_pipeline (después de _dispatch_step está bien).
#
#  Las funciones _fail_step, _persist_step_to_repo, _dispatch_step,
#  _assemble_mechanical, etc. NO cambian: se siguen usando tal cual.
# ============================================================================


# ----------------------------------------------------------------
#  Cálculo de olas topológicas a partir del DAG (depends_on)
# ----------------------------------------------------------------

def _compute_waves(plan: list[Step], done: set[int]) -> list[list[int]]:
    """Particiona los step_index pendientes en olas topológicas.

    Una ola = todos los steps cuyas dependencias ya están satisfechas (en `done`
    o completadas en olas previas). Steps sin deps van en la ola 0.

    Respeta SOLO depends_on, no el orden del plan. Si hay ciclo o dep inexistente
    (plan.py valida 0 <= dep < idx, así que no debería), los steps irresolubles
    quedan fuera y se loguean — nunca se cuelga. "El que supone se equivoca."
    """
    pending = {s.step_index for s in plan if s.step_index not in done}
    deps_by_idx = {s.step_index: set(s.depends_on or []) for s in plan}

    waves: list[list[int]] = []
    satisfied = set(done)

    while pending:
        ready = sorted(
            idx for idx in pending
            if deps_by_idx.get(idx, set()) <= satisfied
        )
        if not ready:
            logger.error(
                "Jacobs: %d steps sin dependencias resolubles (posible ciclo): %s",
                len(pending), sorted(pending),
            )
            break
        waves.append(ready)
        for idx in ready:
            pending.discard(idx)
            satisfied.add(idx)

    return waves


# ----------------------------------------------------------------
#  Ejecución de UN step (cuerpo del antiguo try/except, extraído)
# ----------------------------------------------------------------

async def _run_one_step(step: Step, i: int, pipeline: Pipeline) -> bool:
    """Ejecuta un step individual. Devuelve True si completó, False si falló.

    Es el cuerpo del antiguo bloque try/except del loop secuencial, extraído sin
    cambios de lógica para poder lanzarlo en paralelo vía asyncio.gather.
    Artifacts, persistencia al repo y eventos: idénticos al original.
    """
    step.status     = StepStatus.running
    step.started_at = time.time()
    await store.step_upsert(step)
    await store.event_append(
        pipeline.pipeline_id, "STEP_STARTED",
        {"step_index": i, "facet": step.facet, "capability": step.capability},
        step.step_id,
    )

    try:
        raw_output = await asyncio.wait_for(
            _dispatch_step(step, pipeline),
            timeout=step.timeout_seconds,
        )

        ref, inline = save_if_large(pipeline.pipeline_id, step.step_id, raw_output)
        if ref:
            step.output_ref = ref
            pipeline.context[f"step_{i}_ref"] = ref
        else:
            inline_ref = f"inline:{json.dumps(inline, ensure_ascii=False)}"
            step.output_ref = inline_ref
            pipeline.context[f"step_{i}_ref"] = inline_ref

        step.status      = StepStatus.completed
        step.finished_at = time.time()
        await store.step_upsert(step)
        await store.event_append(
            pipeline.pipeline_id, "STEP_COMPLETED",
            {"step_index": i, "output_ref": step.output_ref},
            step.step_id,
        )
        try:
            await _persist_step_to_repo(
                pipeline_id=pipeline.pipeline_id,
                pipeline_name=pipeline.name,
                step_index=i,
                facet=step.facet,
                capability=step.capability,
                raw_output=raw_output,
            )
        except Exception as _persist_err:  # noqa: BLE001
            logger.warning("No se pudo persistir step %d al repo: %s", i, _persist_err)
        return True

    except asyncio.TimeoutError:
        await _fail_step(pipeline, step, i, f"Timeout ({step.timeout_seconds}s)")
        return False
    except Exception as exc:  # noqa: BLE001
        await _fail_step(pipeline, step, i, str(exc))
        return False


# ----------------------------------------------------------------
#  Pipeline runner — DIRECTOR DE ORQUESTA (ejecución por olas)
# ----------------------------------------------------------------

async def run_pipeline(pipeline: Pipeline) -> None:
    """
    Ejecuta el pipeline por OLAS topológicas. Dentro de cada ola, los steps
    corren EN PARALELO (asyncio.gather). El orden entre olas respeta depends_on.

    Modos:
      dry_run    — no ejecuta nada, completa inmediatamente.
      supervised — ejecuta UNA ola y pausa (status=interrupted); espera /resume.
                   La granularidad de aprobación es la OLA, no el step.
    Hyde: si un step de la ola es hyde sin aprobar, la ola NO se ejecuta y el
    pipeline se interrumpe hasta /approve-step.
    """
    pipeline_id = pipeline.pipeline_id

    if pipeline.mode == "dry_run":
        await store.pipeline_update_status(pipeline_id, PipelineStatus.completed)
        await store.event_append(pipeline_id, "DRY_RUN_COMPLETE", {"steps": len(pipeline.plan)})
        return

    await store.pipeline_update_status(
        pipeline_id, PipelineStatus.running, pipeline.current_step_index, pipeline.context
    )
    await store.event_append(pipeline_id, "PIPELINE_STARTED")

    # Estado derivado del DAG, no de un cursor lineal: un step está "hecho" si
    # tiene su ref en context (sobrevive a /resume y al relanzador).
    done = {
        i for i in range(len(pipeline.plan))
        if pipeline.context.get(f"step_{i}_ref")
    }

    waves = _compute_waves(pipeline.plan, done)
    logger.info(
        "Jacobs director: %d olas, tamaños=%s (ya completos: %s)",
        len(waves), [len(w) for w in waves], sorted(done),
    )

    for wave_num, wave in enumerate(waves):
        # ---- Kill switch: antes de cada ola ----
        if check_kill_switch():
            for i in wave:
                step = pipeline.plan[i]
                step.status = StepStatus.failed
                step.error  = "Kill switch activo"
                await store.step_upsert(step)
            await store.event_append(
                pipeline_id, "KILL_SWITCH_ABORTED", {"wave": wave_num, "steps": wave}
            )
            await store.pipeline_update_status(pipeline_id, PipelineStatus.aborted)
            return

        # ---- Hyde gate: si algún step de la ola es hyde sin aprobar, interrumpir ----
        hyde_pending = [
            i for i in wave
            if pipeline.plan[i].facet == "hyde"
            and not pipeline.context.get(f"hyde_approved_{pipeline.plan[i].step_id}")
        ]
        if hyde_pending:
            for i in hyde_pending:
                step = pipeline.plan[i]
                step.status = StepStatus.blocked_human_gate
                await store.step_upsert(step)
                await store.event_append(
                    pipeline_id, "STEP_BLOCKED_HUMAN_GATE",
                    {"step_index": i, "facet": "hyde"}, step.step_id,
                )
            await store.pipeline_update_status(
                pipeline_id, PipelineStatus.interrupted, wave[0], pipeline.context
            )
            await store.event_append(
                pipeline_id, "PIPELINE_INTERRUPTED",
                {"at_wave": wave_num, "hyde_steps": hyde_pending,
                 "reason": "hyde — requiere /approve-step"},
            )
            return

        # ---- EJECUTAR LA OLA EN PARALELO ----
        await store.event_append(
            pipeline_id, "WAVE_STARTED",
            {"wave": wave_num, "steps": wave, "parallel": len(wave)},
        )
        results = await asyncio.gather(*[
            _run_one_step(pipeline.plan[i], i, pipeline)
            for i in wave
        ])

        # Persistir avance del context tras la ola completa.
        # current_step_index = primer índice NO completado (informativo).
        next_idx = max(wave) + 1
        await store.pipeline_update_status(
            pipeline_id, PipelineStatus.running, next_idx, pipeline.context
        )

        # ---- ¿Algún step falló sin skip_on_fail? → abortar ----
        failed = [
            i for i, ok in zip(wave, results)
            if not ok and not pipeline.plan[i].skip_on_fail
        ]
        if failed:
            await store.pipeline_update_status(pipeline_id, PipelineStatus.aborted)
            await store.event_append(
                pipeline_id, "PIPELINE_ABORTED",
                {"at_wave": wave_num, "failed_steps": failed},
            )
            return

        await store.event_append(
            pipeline_id, "WAVE_COMPLETED", {"wave": wave_num, "steps": wave}
        )

        # ---- Supervised: pausar después de cada ola ----
        if pipeline.mode == "supervised":
            await store.pipeline_update_status(
                pipeline_id, PipelineStatus.interrupted, next_idx, pipeline.context
            )
            await store.event_append(
                pipeline_id, "PIPELINE_INTERRUPTED",
                {"after_wave": wave_num, "next_index": next_idx,
                 "reason": "supervised — awaiting /resume"},
            )
            return

    # ---- Todas las olas terminaron ----
    await store.pipeline_update_status(
        pipeline_id, PipelineStatus.completed, len(pipeline.plan), pipeline.context
    )
    await store.event_append(pipeline_id, "PIPELINE_COMPLETED")
