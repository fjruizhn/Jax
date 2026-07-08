# ============================================================================
#  REEMPLAZOS PARA routes.py
#
#  Cambian DOS endpoints: resume_pipeline y approve_step.
#  El resto de routes.py NO cambia.
#
#  Motivo: ambos asumían "un solo step bloqueado en steps[current_idx]".
#  Con olas, puede haber VARIOS steps bloqueados a la vez (p.ej. dos hyde en
#  paralelo, o una ola supervised completa). Ahora desbloquean por estado,
#  no por índice.
# ============================================================================


# ---- REEMPLAZAR el cuerpo de resume_pipeline (desde "await store.event_append(
#      pipeline_id, "PIPELINE_RESUMED"..." hasta el return) por esto: ----

# @router.post("/pipeline/{pipeline_id}/resume")
# async def resume_pipeline(...):
#     ... (validaciones idénticas: policy, 404, status interrupted, kill switch) ...

    await store.event_append(pipeline_id, "PIPELINE_RESUMED", {"by": req.invoked_by})

    # Desbloquear TODOS los steps en estado blocked (una ola supervised pudo
    # dejar varios). El executor recalcula las olas desde los refs en context,
    # así que basta con poner los blocked en pending para que entren a su ola.
    steps = await store.steps_by_pipeline(pipeline_id)
    for s in steps:
        if s.status == StepStatus.blocked:
            s.status = StepStatus.pending
            await store.step_upsert(s)

    pipeline.plan = steps
    background.add_task(run_pipeline, pipeline)

    return {
        "pipeline_id": pipeline_id,
        "status": "resuming",
        "from_index": pipeline.current_step_index,
    }


# ---- REEMPLAZAR en approve_step el bloque que obtiene/valida current_step
#      (desde "steps = await store.steps_by_pipeline" hasta el background.add_task)
#      por esto: ----

# @router.post("/pipeline/{pipeline_id}/approve-step")
# async def approve_step(...):
#     ... (validaciones idénticas: policy, 404, status interrupted, kill switch) ...

    steps = await store.steps_by_pipeline(pipeline_id)

    # Buscar TODOS los steps en human gate (una ola puede tener varios hyde).
    gated = [s for s in steps if s.status == StepStatus.blocked_human_gate]
    if not gated:
        raise HTTPException(
            status_code=409, detail="No hay steps pendientes de aprobación (human gate)"
        )

    # En modo no-supervised, approve-step solo aplica a hyde. En supervised,
    # aplica a cualquier step de la ola pausada.
    if pipeline.mode != "supervised":
        non_hyde = [s for s in gated if s.facet != "hyde"]
        if non_hyde:
            raise HTTPException(
                status_code=422,
                detail="approve-step solo válido para hyde o pipelines en modo supervised",
            )

    approved_indices = []
    for current_step in gated:
        await store.event_append(
            pipeline_id, "STEP_APPROVED",
            {"step_index": current_step.step_index, "facet": current_step.facet,
             "by": req.invoked_by},
            current_step.step_id,
        )
        current_step.status = StepStatus.pending
        current_step.error  = None
        await store.step_upsert(current_step)
        if current_step.facet == "hyde":
            pipeline.context[f"hyde_approved_{current_step.step_id}"] = True
        approved_indices.append(current_step.step_index)

    # Persistir las marcas hyde_approved_* en context antes de reanudar.
    await store.pipeline_update_status(
        pipeline_id, pipeline.status, pipeline.current_step_index, pipeline.context
    )

    pipeline.plan = steps
    background.add_task(run_pipeline, pipeline)

    return {
        "pipeline_id":     pipeline_id,
        "status":          "resuming",
        "approved_steps":  approved_indices,
    }
