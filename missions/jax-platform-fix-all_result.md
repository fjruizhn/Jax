# Resultado de: jax-platform-fix-all.md

Todo listo, Fernando. Resumen de lo que hice:

---

## Qué se cambió y por qué

### TAREA 1 — Pipeline results

**Bug raíz**: el handler `pipeline_step_changed && completed` en el store no tenía deduplicación — si el evento llegaba dos veces (reconexión WS), se agregaban mensajes duplicados. El filtro `s.result` también era demasiado estricto.

**Fixes en `useJaxStore.js`**:
- `addMessage` ahora ignora mensajes con ID duplicado
- Set `_pipelineCompletedShown` previene que el mismo pipeline llene el chat dos veces
- Filtro cambiado a solo `status === 'completed'` — el conteo es correcto ahora
- El mensaje final dice "N de M steps, Xs totales"

### TAREA 2 — Modo autonomous

**Bug raíz**: bloqueado en 3 lugares (`policy.py`, `models.py`, `routes.py`). El executor YA ejecutaba todos los steps en secuencia para cualquier modo no-supervised — solo había que desbloquear la política.

**Verificado en vivo**: 3 steps `jax_local` completaron en 18 segundos sin intervención.

El PipelineModal ahora tiene `⚡ Autonomous` como opción y `jax_local` en la lista de facetas.

### TAREA 3 — Comando resultado en chat

**Bug raíz crítico**: `command.py` capturaba el stdout del proceso JAX (mensajes de consola) y lo sobreescribía sobre el `_result.md` que JAX había escrito internamente con la respuesta real. El WS enviaba console output en lugar de la respuesta de la IA.

**Fix**: se usa `DEVNULL` para stdout/stderr del proceso, y se lee el archivo que JAX escribió. El evento WS ahora incluye `result` (completo) + `result_preview` (500 chars).

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
