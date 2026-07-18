/using-superpowers
/ruflo

# Misión: _invoke_hyde real en Jacobs — portar el mecanismo ya probado del CLI viejo

## Contexto (leer completo antes de tocar nada)

Hyde ya funciona hace meses en el CLI de un solo shot (`jax/core/main.py` +
`jax/muscles/subprocess_muscle.py`): shellea al binario `claude` (Claude Code
CLI) en modo headless. Confirmado con evidencia real: `jax --task X --facet
hyde` ya produjo fixes de código reales (ej. el fix de streaming SSE de
junio, commit con diff verificado).

En el wave scheduler nuevo (`jacobs/executor.py`), Hyde quedó en placeholder
desde v0.2:
```python
if step.facet == "hyde":
    # Llegamos aquí solo si Fernando aprobó vía /approve-step.
    # En v0.2 devuelve placeholder — conexión real pendiente para v0.3.
    return {
        "success":  True,
        "facet":    "hyde",
        "result":   "[v0.2] Hyde aprobado por Fernando. Conexión a ejecución real pendiente para v0.3.",
        "approved": True,
    }
```

Esta misión es v0.3: portar el mecanismo de `subprocess_muscle.py` a
`jacobs/executor.py`, SIN reinventar nada — es el mismo `create_subprocess_exec`
que ya funciona, adaptado a la firma de Jacobs.

## 1. Reconocimiento (read-only, reportar antes de tocar nada)

```bash
cat /home/fruiz/jax/jax/muscles/subprocess_muscle.py
sed -n '600,650p' /home/fruiz/jax/jacobs/executor.py
grep -n "^async def _invoke_\|^_HTTP_FACETS\|^_MOTOR_FACETS\|^def _get_active_model\|facet_models" /home/fruiz/jax/jacobs/executor.py
grep -A15 "\[personalities.hyde\]" /home/fruiz/jax/config/config.toml
```

Reportar: ¿ya existe `_get_active_model` (de la misión `facet_models`, que
corrió en el repo `jax-platform`, NO en este repo `jax`)? Este repo (`jax`,
el wave scheduler) es distinto del repo `jax-platform` (el backend/frontend
de Axioma) — confirmar si comparten la misma MariaDB `jax_memory` o si hay
que conectar credenciales nuevas.

## 2. Implementar `_invoke_hyde` real

Reemplazar el placeholder por una función que:
- Lee el modelo activo de la faceta `hyde` desde la tabla `facet_models`
  (MariaDB `jax_memory`), con fallback a `"sonnet"` si la tabla no responde
  (fail-open, mismo patrón que `_CATALOG_CAPS`).
- Invoca `claude` vía `asyncio.create_subprocess_exec`, con los mismos flags
  que ya usa `subprocess_muscle.py`: `--model`, `--append-system-prompt`,
  `--print`, `--output-format text`, `--permission-mode acceptEdits`,
  `--allowedTools Write,Edit,Read,Bash`, `--add-dir /home/fruiz/jax/workspace`.
- Timeout con `proc.kill()` + `await proc.wait()` para cosechar el zombie,
  igual que el original — no dejar procesos huérfanos.
- Trunca el prompt a 32k chars, igual que `MAX_PROMPT_CHARS` del original.

## 3. Concurrencia — decisión de diseño a NO improvisar

Jacobs corre steps de una misma ola en paralelo vía `asyncio.gather`. El CLI
viejo es siempre secuencial (una invocación de `claude` a la vez), por lo que
nunca hubo dos procesos `claude` escribiendo en `/home/fruiz/jax/workspace`
al mismo tiempo. Si dos steps `hyde` caen en la misma ola paralela en Jacobs,
van a competir por el mismo `workspace_dir`.

Agregar un semáforo async (`asyncio.Semaphore(1)`, nivel de módulo, mismo
patrón que `GPU_SEMAPHORE` en `ollama_muscle.py`) específico para invocaciones
de Hyde, de forma que aunque el DAG programe dos steps `hyde` en la misma
ola, se ejecuten secuencialmente entre sí (sin bloquear a las demás facetas
de esa misma ola, que sí corren en paralelo). Documentar la razón en un
comentario, igual que el semáforo de GPU.

## 4. Explícitamente NO hacer en esta misión

- NO tocar `subprocess_muscle.py` ni el CLI viejo — queda intacto, es la
  fuente de verdad que estamos portando, no modificando.
- NO cambiar el gate de aprobación (`/approve-step`, `hyde_pending`, líneas
  ~797-818) — eso ya funciona y no es parte de este alcance.
- NO conectar `FacetClient.for_hyde()` en `las_manos/facet_client.py` (ese
  cliente es para el flujo LAS MANOS/Intent Envelope, un mecanismo distinto
  y separado del wave scheduler de Jacobs — no mezclar los dos sistemas).

## 5. Gate de prueba (obligatorio)

Crear una pipeline de prueba con un solo step `facet=hyde`, algo simple y
verificable en disco, ej: pedirle que escriba un archivo con contenido
específico en `/home/fruiz/jax/workspace/`, y confirmar:
1. El archivo existe con el contenido esperado (no confiar en el texto de
   respuesta — verificar con `cat`/`stat` el archivo real).
2. El campo `model` en el resultado del step refleja el modelo activo real
   leído de `facet_models` (probar cambiándolo a `opus` vía el dropdown de
   Axioma antes de correr el test, confirmar que Jacobs lo recogió).
3. Timeout: forzar un timeout corto (ej. 2s) en un prompt que tarde más,
   confirmar que el proceso se mata limpio (sin zombies — verificar con
   `ps aux | grep claude` que no quedó nada colgado).
4. Dos steps `hyde` en la misma ola (si el plan de prueba lo permite):
   confirmar que se ejecutan secuencialmente entre sí, no en paralelo
   destructivo sobre el mismo workspace.

## 6. Reporte final

Archivos tocados + diff, resultado de cada caso del gate, incertidumbres
declaradas, path de rollback.
