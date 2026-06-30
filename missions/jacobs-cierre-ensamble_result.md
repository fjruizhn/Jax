# Resultado de: jacobs-cierre-ensamble.md

Las edits están correctas. Necesito que Fernando reinicie el servicio — no tengo acceso a sudo. Ejecutá esto:

```bash
sudo systemctl restart jax-las-manos && sleep 8 && systemctl is-active jax-las-manos
```

Cuando confirmes que está `active`, ejecuto el gate completo. Mientras tanto, te muestro el diff de lo que cambió:

**`plan.py` — 3 cambios:**

1. `_PLAN_SYSTEM_MODULAR` puntos 4–5 → ahora 4–6: antepenúltimo=validación (thot/validate_consistency), penúltimo=reconciliación (ada/reconcile), último=ensamble (ada/assemble, produce solo el manifest).

2. Línea del patrón compilador en `_ada_plan` → menciona los 3 steps finales correctos.

3. Ejemplo JSON en `_ada_plan`: `thot/critique` → `thot/validate_consistency`, + nuevo step `ada/reconcile` con `depends_on:[3]`, + `ada/assemble` con `depends_on:[0,1,2,3,4]`.

**`executor.py` — 2 cambios:**

1. Nueva función `_assemble_mechanical(step, pipeline)` (síncrona, antes del dispatcher). Lee los refs de `pipeline.context`, concatena los módulos de diseño, separa los parches de reconcile, arma el documento. Loggea `"ensamble mecánico: N chars de M módulos"`.

2. En `_dispatch_step`, antes de `_build_context_input`, el check:
   ```python
   if step.capability == "assemble":
       return _assemble_mechanical(step, pipeline)
   ```
   — cortocircuita antes de cargar cualquier contexto de deps. El acceso a `pipeline` ya estaba en la firma de `_dispatch_step`, sin cambios de firma necesarios.

**Rollback:**
```
~/jax/jacobs/plan.py.backup-cierre-20260629-052415
~/jax/jacobs/executor.py.backup-cierre-20260629-052415
```

Decime cuando el servicio esté `active` y paso directo al gate.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
