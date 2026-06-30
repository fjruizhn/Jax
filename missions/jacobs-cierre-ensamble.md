faceta: hyde

# Jacobs — Cierre de pipeline: reconciliación + ensamble MECÁNICO

> Editar el Jacobs CANÓNICO `~/jax/jacobs/`. HYDE: backup, gate, rollback si falla.
> Hipatia inicia con `/using-superpowers` y `/ruflo`.

## Contexto (del examen de admisión)
El examen generó 9 módulos coherentes (cada uno usa los tipos del step 0, imports verificados)
y Thot detectó errores de integración REALES (records_iter/depth_keys faltantes, G5 sin
implementar). PERO el step de ensamble falló: le pedía a Ada reescribir los 9 módulos juntos
(116K chars) → excedió la ventana. Causa raíz: el ensamble NO debe ser regenerativo.

**Cierre correcto = 2 pasos:**
- **Reconciliación (Ada):** toma los hallazgos del validador (Thot) + SOLO los módulos
  afectados, produce los PARCHES puntuales (no reescribe todo). Input acotado.
- **Ensamble (MECÁNICO, sin LLM):** el código concatena los módulos ya generados + aplica
  los parches + arma el manifest. No pasa por ningún modelo → no puede fallar por tamaño.

## Cambio 1 — `~/jax/jacobs/plan.py`: instruir el cierre correcto en el prompt
Backup: `plan.py.backup-cierre-$(date +%Y%m%d-%H%M%S)`

En `_PLAN_SYSTEM_MODULAR`, reemplazar los puntos 4 y 5 actuales por:
```
"4. El ANTEPENÚLTIMO step es 'validación de consistencia' (facet thot, capability "
"'validate_consistency'): revisa nombres huérfanos, tipos no definidos, referencias rotas. "
"Devuelve SOLO discrepancias con referencia al step y nombre.\n"
"5. El PENÚLTIMO step es 'reconciliación' (facet ada, capability 'reconcile'): recibe los "
"hallazgos del validador y los módulos afectados, y produce SOLO los PARCHES puntuales que "
"corrigen cada hallazgo (ej: agregar el método faltante a un módulo). NO reescribe los módulos "
"completos — solo los fragmentos a corregir, identificando módulo y ubicación.\n"
"6. El ÚLTIMO step es 'ensamble' (facet ada, capability 'assemble'): describe el manifest del "
"paquete (orden de módulos, versiones, índice). El ensamble FÍSICO de los módulos lo hace el "
"sistema mecánicamente; este step solo produce el manifest/índice, NO el documento completo.\n"
```
> Actualizar también la línea ~151 (la descripción corta) para mencionar reconciliación.
> Y en el ejemplo de _ada_plan (B2.2), ajustar para mostrar capability 'reconcile' y 'assemble'.

## Cambio 2 — `~/jax/jacobs/executor.py`: ensamble mecánico (NO va a Ada)
Backup: `executor.py.backup-cierre-$(date +%Y%m%d-%H%M%S)`

En el dispatch por faceta (donde está `if step.facet == "ada": return await _invoke_ada(...)`,
~línea 465), agregar ANTES de ese chequeo un caso especial por capability:
```python
    # Ensamble mecánico: NO pasa por ningún LLM. Concatena los módulos ya generados.
    if step.capability == "assemble":
        return _assemble_mechanical(step, pipeline)
```
> Necesita acceso a `pipeline` para leer los outputs de los steps previos. Verificar la
> firma de la función de dispatch — si no recibe `pipeline`, pasárselo (el executor ya lo
> tiene en el scope de run_pipeline). Reportar cómo se resolvió el acceso a pipeline.

Agregar la función `_assemble_mechanical`:
```python
def _assemble_mechanical(step: Step, pipeline: Pipeline) -> dict:
    """Ensamble MECÁNICO del paquete final. Sin LLM. Concatena los outputs de los
    módulos ya generados (steps de diseño), incluye el manifest que generó este step
    (si su prompt produjo uno) y los parches de reconciliación. No puede fallar por tamaño."""
    partes = []
    # Encabezado / manifest
    partes.append("# PAQUETE MODULAR ENSAMBLADO\n")
    partes.append(f"# Pipeline: {pipeline.pipeline_id}\n")
    partes.append(f"# Objetivo: {pipeline.context.get('objective','')}\n")
    partes.append(f"# Generado por Jacobs (ensamble mecánico) — {len(pipeline.plan)} steps\n\n")

    # Concatenar el output de cada step de DISEÑO en orden (excluir validación/reconcile/assemble)
    skip_caps = {"validate_consistency", "critique", "reconcile", "assemble"}
    patches_text = ""
    for j in range(step.step_index):
        prev = pipeline.plan[j]
        ref = pipeline.context.get(f"step_{j}_ref", "")
        if not ref:
            continue
        data = _load_ref(ref)
        result = data.get("result") or data.get("text") or ""
        if prev.capability == "reconcile":
            patches_text = str(result)   # guardar parches para anexar al final
            continue
        if prev.capability in skip_caps:
            continue
        partes.append(f"\n{'='*70}\n## MÓDULO (step {j}): {prev.capability}\n{'='*70}\n")
        partes.append(str(result))

    # Anexar los parches de reconciliación al final
    if patches_text:
        partes.append(f"\n{'='*70}\n## PARCHES DE RECONCILIACIÓN (correcciones del validador)\n{'='*70}\n")
        partes.append(patches_text)

    documento = "\n".join(partes)
    logger.info("Jacobs ensamble mecánico: %d chars de %d módulos", len(documento), step.step_index)
    return {
        "success": True,
        "facet": "ada",        # nominal; el trabajo fue mecánico
        "model": "mechanical_assembler",
        "result": documento,
    }
```
> Verificar que `_load_ref`, `logger`, `Step`, `Pipeline` estén importados/disponibles en el scope.

## Cambio 3 — Tope de contexto: la reconciliación NO necesita los 9 módulos completos
El step de reconciliación depende del validador (Thot) + los módulos afectados. Si depends_on
lo hace depender de TODOS, su contexto vuelve a ser enorme. Mitigación PROPUESTA: dejar que el
prompt de reconciliación pida depender SOLO del step de validación + los módulos que Thot
mencione. Esto se instruye en el prompt (Cambio 1) — Ada decide el depends_on del reconcile.
> Si el reconcile igual recibe demasiado contexto, el log de C1 lo avisará y se ajusta.

## Gate (obligatorio)
```bash
sudo systemctl restart jax-las-manos && sleep 8 && systemctl is-active jax-las-manos

# Verificar que el dispatch tiene el caso assemble mecánico:
grep -n "_assemble_mechanical\|capability == \"assemble\"\|capability == 'assemble'" ~/jax/jacobs/executor.py

# Prueba MÍNIMA del ensamble mecánico: pipeline corto con un step assemble al final.
# Steps de diseño baratos (jax_local) + un assemble que debe concatenarlos SIN llamar a Ada.
PIPE=$(curl -s --max-time 120 http://127.0.0.1:7777/jacobs/pipeline -X POST -H "Content-Type: application/json" \
  -d '{
    "name":"gate-ensamble",
    "objective":"prueba ensamble mecanico",
    "invoked_by":"Fernando",
    "mode":"autonomous",
    "steps":[
      {"facet":"jax_local","capability":"design","prompt":"Escribí: MODULO-A contenido alfa.","depends_on":[]},
      {"facet":"jax_local","capability":"design","prompt":"Escribí: MODULO-B contenido beta.","depends_on":[]},
      {"facet":"ada","capability":"assemble","prompt":"Manifest del paquete.","depends_on":[0,1]}
    ]
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['pipeline_id'])")
echo "pipeline: $PIPE"
sleep 40
curl -s http://127.0.0.1:7777/jacobs/pipeline/$PIPE | python3 -c "
import sys,json; d=json.load(sys.stdin)
for s in d.get('steps',[]):
    out = s.get('output_ref') or ''
    txt = out
    if out.startswith('inline:'):
        import json as j
        try: txt = j.loads(out[7:]).get('result', out)
        except: pass
    print('=== step', s['step_index'], s['capability'], s['status'], '===')
    print(txt[:400]); print()
"
sudo journalctl -u jax-las-manos -n 15 --no-pager | grep -i "ensamble\|mechanical\|assemble" | tail -5
```
**Criterio de aceptación:**
- Servicio 'active'.
- El step assemble (step 2) tiene `model: mechanical_assembler` y su output contiene
  TANTO "MODULO-A contenido alfa" COMO "MODULO-B contenido beta" → concatenó mecánicamente.
- El step assemble NO llamó a Ada (no consumió Z.ai) → el log "ensamble mecánico" aparece.
- El ensamble NO falla por tamaño aunque haya muchos módulos.

## Reporte final
Diffs de plan.py (prompt) y executor.py (caso assemble + _assemble_mechanical), cómo se
resolvió el acceso a `pipeline` en el dispatch, resultado del gate (¿el ensamble concatenó
A y B sin llamar a Ada?), incertidumbres, rollback (*.backup-cierre-*).
