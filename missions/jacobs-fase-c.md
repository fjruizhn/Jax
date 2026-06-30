faceta: hyde

# Jacobs Fase C — Contexto completo según `depends_on`

> Editar el Jacobs CANÓNICO `~/jax/jacobs/executor.py`. HYDE: backup, gate, rollback si falla.
> Hipatia inicia con `/using-superpowers` y `/ruflo`.

## Objetivo y causa raíz
`_build_context_input` (línea ~71) pasa a cada step un RESUMEN de 500 chars de TODOS los
steps anteriores (`summary = str(result_text)[:500]`), indiscriminado. Para el plan modular
esto está mal:
- **Truncado:** el step de invariantes necesita el TEXTO COMPLETO de los tipos comunes y
  módulos, no 500 chars — si no, inventa sobre lo que no vio.
- **Indiscriminado:** pasa outputs de steps que el actual NO necesita (ruido), e ignora la
  declaración `depends_on` que Ada ya generó.

**Fix:** cada step recibe el output COMPLETO de EXACTAMENTE los steps en su `depends_on`.
Si `depends_on` está vacío (steps triviales / sin dependencias), comportamiento de respaldo:
mantener el resumen corto de los anteriores (no romper el caso simple).

## Cambio — `~/jax/jacobs/executor.py`
Backup: `executor.py.backup-faseC-$(date +%Y%m%d-%H%M%S)`

### C1 — `_build_context_input`: cargar dependencias completas
Reemplazar el cuerpo del loop de contexto para que:
- Si `step.depends_on` NO está vacío → iterar SOLO esos índices, y cargar el output
  COMPLETO (sin truncar a 500). Usar un tope de seguridad alto y CONFIGURABLE (ver C3).
- Si `step.depends_on` está vacío → comportamiento actual (resumen 500 de anteriores),
  para no romper pipelines triviales.

Estructura propuesta (adaptar a las variables existentes):
```python
def _build_context_input(step: Step, pipeline: Pipeline) -> dict:
    """Construye el input enriquecido. Si el step declara depends_on, carga el
    output COMPLETO de esas dependencias; si no, resumen corto de los anteriores."""
    objective = pipeline.context.get("objective", "")
    previous_outputs: list[dict] = []

    deps = getattr(step, "depends_on", []) or []
    if deps:
        indices = [j for j in deps if 0 <= j < step.step_index]
        full = True
    else:
        indices = list(range(step.step_index))
        full = False

    for j in indices:
        ref = pipeline.context.get(f"step_{j}_ref", "")
        if not ref:
            continue
        facet_name = pipeline.plan[j].facet if j < len(pipeline.plan) else "unknown"
        try:
            data = _load_ref(ref)
            result_text = data.get("result") or data.get("text") or json.dumps(data)
            text = str(result_text)
            if full:
                content = text[:MAX_DEP_CONTEXT_CHARS]      # completo (con tope alto)
                truncated = len(text) > MAX_DEP_CONTEXT_CHARS
            else:
                content = text[:500]                         # resumen (caso simple)
                truncated = len(text) > 500
        except Exception:  # noqa: BLE001
            content = f"[ref: {ref}]"
            truncated = False
        previous_outputs.append({
            "step_index": j,
            "facet": facet_name,
            "summary": content,
            "truncated": truncated,
        })

    # Log de tamaño total del contexto ensamblado (para detectar steps que rozan la ventana)
    total_chars = sum(len(p["summary"]) for p in previous_outputs)
    logger.info(
        "Jacobs step %s deps=%s contexto=%d chars%s",
        step.step_index, deps, total_chars,
        " [ALGUNA DEP TRUNCADA]" if any(p.get("truncated") for p in previous_outputs) else "",
    )

    return {
        "objective": objective,
        "previous_outputs": previous_outputs,
        "prompt": step.input.get("prompt", ""),
    }
```
> Verificar que `logger` exista en executor.py (si no, `import logging; logger = logging.getLogger("jacobs.executor")`).

### C2 — `_enrich_prompt`: rotular dependencias y avisar truncado
Ajustar `_enrich_prompt` para que, cuando viene de depends_on, el encabezado diga
"Salida COMPLETA de la dependencia (step N)" en vez de "contexto de pasos anteriores",
y si `truncated` es True, agregar una nota visible al modelo:
```python
    prev = ctx_input.get("previous_outputs", [])
    if prev:
        parts.append("\nSalidas de las dependencias declaradas (usalas como fuente, no las reinventes):")
        for p in prev:
            nota = " [TRUNCADO — dependencia excede el tope]" if p.get("truncated") else ""
            parts.append(
                f"\n--- Dependencia: step {p['step_index']} ({p['facet']}){nota} ---\n{p['summary']}"
            )
```
> Mantener `_EVIDENCE_RULE` y el `objective` como están (van primero).

### C3 — Constante de tope configurable
Agregar cerca del tope del archivo:
```python
MAX_DEP_CONTEXT_CHARS = 60000   # ~15K tokens por dependencia. Tope de seguridad
                                # para no exceder la ventana al ensamblar muchas deps.
```
> 60000 chars ≈ 15K tokens por dependencia. Para un ensamble con 10 deps eso podría ser
> mucho — por eso el log de C1 avisa el total. Si en el examen el step de ensamble roza el
> límite, se ajusta. Declarar este valor como PROPUESTO.

## Gate (obligatorio)
```bash
sudo systemctl restart jax-las-manos && sleep 8 && systemctl is-active jax-las-manos

# Verificación estática: el código carga depends_on completo
grep -n "MAX_DEP_CONTEXT_CHARS\|depends_on\|full" ~/jax/jacobs/executor.py | head

# Prueba funcional MÍNIMA y BARATA: un pipeline de 2 steps donde el 2º depende del 1º,
# y el 2º debe ver el output COMPLETO del 1º (no 500 chars).
# Usar facetas locales/baratas para no gastar Ada en el gate.
curl -s --max-time 120 http://127.0.0.1:7777/jacobs/pipeline -X POST -H "Content-Type: application/json" \
  -d '{
    "name":"gate-faseC",
    "objective":"prueba de contexto completo",
    "invoked_by":"Fernando",
    "mode":"execute",
    "steps":[
      {"facet":"jax_local","capability":"reason","prompt":"Escribí exactamente este texto largo único: BLOQUE-ALFA seguido de 100 palabras inventadas distintas.","depends_on":[]},
      {"facet":"jax_local","capability":"reason","prompt":"Repetí TEXTUALMENTE el contenido completo que recibiste del step 0. Si solo viste un resumen cortado, decí TRUNCADO.","depends_on":[0]}
    ]
  }' | python3 -m json.tool | head -20

# Revisar en logs el tamaño de contexto que se pasó
sudo journalctl -u jax-las-manos -n 20 --no-pager | grep -i "contexto\|deps\|step" | tail -8
```
**Criterio de aceptación:**
- Servicio 'active'.
- El log muestra el tamaño de contexto pasado por step y el `deps=[...]`.
- El step 2 del gate puede repetir el contenido COMPLETO del step 0 (no dice TRUNCADO) →
  prueba que recibió el output completo de su dependencia, no 500 chars.
- Pipelines SIN depends_on siguen funcionando (resumen 500, caso simple intacto).

> Si el step 2 dice TRUNCADO o no ve el contenido del step 0 → el wiring de depends_on no
> está pasando el output completo. Reportar y revisar cómo se guarda `step_{j}_ref` en el context.

## Reporte final
Diff de executor.py (las 3 partes), resultado del gate (¿el step 2 vio el output completo?),
el log de tamaños de contexto, valor de MAX_DEP_CONTEXT_CHARS usado, incertidumbres,
rollback (*.backup-faseC-*).
