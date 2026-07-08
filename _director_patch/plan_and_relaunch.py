# ============================================================================
#  AJUSTE 1 — tools/jacobs_relaunch.py
#
#  El relanzador resetea de --from-step en adelante y llama run_pipeline.
#  Con el wave scheduler esto YA FUNCIONA sin cambios: run_pipeline recalcula
#  las olas desde los refs en context, y los steps reseteados (output_ref=None)
#  quedan fuera de `done`, así que entran a su ola correspondiente.
#
#  NO se requiere cambio funcional. Pero conviene un mensaje más honesto:
#  "desde el step N" ahora significa "desde la ola que contiene a N y los
#  steps que dependían de los reseteados". Cambio cosmético opcional:
# ============================================================================

# OPCIONAL — reemplazar la línea:
#     print(f"▶ Reanudando desde step {from_step}...")
# por:
#     print(f"▶ Reanudando desde step {from_step} (el director recalculará las olas)...")

# El resto del relanzador queda IGUAL. Ya valida que las deps de los steps a
# re-ejecutar tienen su ref — esa verificación sigue siendo correcta y útil.


# ============================================================================
#  AJUSTE 2 — jacobs/plan.py  (regla clean-room: auditor ≠ facets auditados)
#
#  Añadir al texto de _PLAN_SYSTEM_MODULAR, justo antes de la línea final
#  "Salida: SOLO el array JSON.", el siguiente párrafo:
# ============================================================================

_CLEANROOM_RULE = (
    "\nREGLA DE AUDITORÍA INDEPENDIENTE (clean-room): el step que audita, valida "
    "o critica el trabajo de otros steps DEBE usar un facet DISTINTO al de los "
    "steps que revisa. Un facet no se audita a sí mismo. Si los módulos backend "
    "y frontend fueron diseñados por 'ada' y 'kimi', su auditor debe ser 'thot' "
    "(u otro facet que no sea ada ni kimi). La revisión independiente es la "
    "garantía de calidad: quien produce no es quien aprueba.\n"
)

# Es decir, _PLAN_SYSTEM_MODULAR pasa de terminar en:
#     "...incluí un step que lo verifique. 'El que supone se equivoca.'\n\n"
#     "Salida: SOLO el array JSON."
# a:
#     "...incluí un step que lo verifique. 'El que supone se equivoca.'\n"
#     + _CLEANROOM_RULE +
#     "\nSalida: SOLO el array JSON."


# ============================================================================
#  AJUSTE 3 (opcional, defensa en profundidad) — validación post-plan en plan.py
#
#  La regla de arriba es una INSTRUCCIÓN al planificador (Ada). Para garantizar
#  que se cumple aunque Ada la ignore, se puede añadir una verificación mecánica
#  en _parse_plan_json o tras construir los steps: si un step de capability de
#  auditoría comparte facet con alguna de sus depends_on, loguear WARNING.
#  NO se auto-corrige (eso sería suponer la intención); solo se declara.
# ============================================================================

_AUDIT_CAPABILITIES = frozenset({
    "validate_consistency", "critique", "review", "audit",
})

def _check_cleanroom(steps: list) -> list[str]:
    """Devuelve lista de violaciones clean-room (auditor con facet de un dep).
    No modifica el plan — solo reporta. 'El que supone se equivoca': declarar,
    no asumir que está bien."""
    warnings = []
    by_index = {s.step_index: s for s in steps}
    for s in steps:
        if s.capability not in _AUDIT_CAPABILITIES:
            continue
        for dep in s.depends_on:
            dep_step = by_index.get(dep)
            if dep_step and dep_step.facet == s.facet:
                warnings.append(
                    f"step {s.step_index} ({s.facet}/{s.capability}) audita al "
                    f"step {dep} que es del MISMO facet '{s.facet}' — no es "
                    f"auditoría independiente"
                )
    return warnings

# Uso sugerido en PlanBuilder._from_spec o _from_objective, tras construir steps:
#     for w in _check_cleanroom(steps):
#         logger.warning("Jacobs clean-room: %s", w)
