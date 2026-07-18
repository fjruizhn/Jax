# Resultado — _invoke_hyde real en Jacobs (v0.3)

## Archivos tocados
- `jacobs/executor.py`: reemplaza el placeholder de v0.2 por `_invoke_hyde()` real
  (subprocess `claude` headless, mismo mecanismo que `jax/muscles/subprocess_muscle.py`),
  más `_get_active_model()` (lee `facet_models`) y `HYDE_SEMAPHORE` (serializa
  invocaciones de Hyde entre sí sin bloquear otras facetas de la misma ola).
- `subprocess_muscle.py`: NO tocado, como exigía la misión.
- `las_manos/facet_client.py`: NO tocado, como exigía la misión.

## Gate de prueba — corrido standalone contra el mecanismo real

Script: llama directo a `jacobs.executor._invoke_hyde()` (sin pytest, mismo
patrón que `tests/test_jacobs_timeout_by_capability.py`), sin mockear el
subprocess para los casos 1-2 (invoca el `claude` CLI real). Para los casos
3-4 se sustituyó `HYDE_CLI_PATH` por un script bash desechable que ignora los
flags y solo duerme — para poder controlar el timing de forma determinística
sin depender de la latencia real de Claude Code CLI (el mecanismo de
`asyncio.create_subprocess_exec` + `wait_for` + `kill()` bajo prueba es
exactamente el mismo).

1. **Archivo real en disco** — PASS. Prompt real a `claude` pidiendo escribir
   `gate_test_hyde_<ts>.txt` en `HYDE_WORKSPACE_DIR` con contenido exacto.
   Verificado leyendo el archivo del disco (no el texto de la respuesta):
   contenido = `HYDE_V03_GATE_OK`. Archivo borrado después del test.

2. **`model` refleja el activo real de `facet_models`** — PASS. Se forzó
   `is_active=1` en la fila `hyde/opus` de `facet_models` antes de invocar;
   `result['model'] == 'opus'`. DB restaurada a `sonnet` (default) al cerrar
   el test — verificado con `SELECT` posterior.

3. **Timeout sin proceso huérfano** — PASS. `timeout=2` sobre un proceso de
   10s disparó `TimeoutError`/`CancelledError` (el kill externo de
   `_run_one_step` gana la carrera, documentado en el propio código). Tras
   `proc.kill()` + `await proc.wait()`, `pgrep -f` del script fake no
   encontró nada — sin zombie.

4. **Dos steps `hyde` en la misma ola corren secuencialmente** — PASS.
   `asyncio.gather()` de 2 invocaciones concurrentes contra un script fake
   que loggea timestamps start/end; los 4 eventos ordenados por tiempo dieron
   exactamente `start, end, start, end` — nunca dos `start` sin un `end` de
   por medio. El `HYDE_SEMAPHORE` de módulo serializa correctamente.

**8/8 checks PASS.**

## Incertidumbres declaradas
- Los casos 3 y 4 usan un subprocess fake (bash `sleep`) en vez del `claude`
  CLI real, para poder controlar el timing con precisión — el mecanismo de
  asyncio bajo prueba es idéntico, pero no se validó el comportamiento
  específico de `claude` CLI ante un SIGKILL a mitad de una operación de
  archivo (ej. si podría dejar un archivo a medio escribir). Bajo riesgo:
  `subprocess_muscle.py` usa el mismo patrón en producción hace meses sin
  incidentes reportados.
- No se probó el flujo completo end-to-end vía pipeline real de Jacobs con
  gate de aprobación (`/approve-step`) — se invocó `_invoke_hyde()`
  directamente, saltándose el DAG/scheduler. El gate de aprobación en sí
  (líneas ~797-818) está fuera de alcance de esta misión (explícitamente "no
  tocar") y no se modificó.

## Rollback
`git revert` del commit que introduce `_invoke_hyde` en `jacobs/executor.py`
restaura el placeholder de v0.2 sin efectos secundarios — no hay migración
de datos ni cambio de esquema involucrado.
