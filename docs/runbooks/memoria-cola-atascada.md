# Memoria: cola de extracción atascada y avisos de las unidades

Origen: auditoría escalón 3 del 2026-10-05. Este procedimiento no concede permiso de
producción; las acciones de abajo las ejecuta una persona autorizada.

## Qué significa que `jax-memory-worker` esté rojo

La unidad sale con error mientras haya jobs `QUARANTINED` o `UNKNOWN` en
`memory_extraction_jobs`: `pending()` los excluye, así que sin ese conteo una cola
«vacía» saldría verde con conversaciones que nunca se destilaron. Con
`OnFailure=aviso-fallo@%n.service` el fallo llega a Telegram (agrupado por incidente).

Causas de la cuarentena: límites o autoridad de la FUENTE (mensajes de más, caracteres
de más, origen sin permiso, fuente cambiada), commit inconsistente o intentos agotados.
Una salida mal formada del extractor NO cuarentena al primer intento: se reintenta con
backoff hasta `JAX_MEMORY_MAX_ATTEMPTS` (default 3) con `error_code=EXTRACTOR_OUTPUT_INVALID`.

## Resolver

1. Listar: `python -m jax.memory.worker --atascados` (conversación, estado, error, intentos).
2. Revisar la conversación y el `error_code` ANTES de actuar. `INCONSISTENT_*` o
   `frozen extraction source changed` no se arreglan re-encolando: el job volverá a
   cuarentena (el re-encolado conserva `frozen_output`/`input_digest`).
3. Re-encolar lo ya revisado:
   `python -m jax.memory.worker --reencolar <conversation_id> --motivo "<por qué>" [--actor <quién>]`.
   Solo acepta `QUARANTINED`; deja el job en `READY` con intentos 0 y escribe el evento
   `REQUEUE` (quién, por qué, estado y error anteriores) en `memory_extraction_job_events`.
4. Auditoría: `SELECT * FROM memory_extraction_job_events WHERE conversation_id=<id>`.
5. La unidad vuelve a verde en la corrida siguiente cuando no queda ningún `QUARANTINED`/`UNKNOWN`.

## Conversaciones abiertas hace más de 7 días

El worker las lista en el journal como WARNING (`JAX_MEMORY_OPEN_CONVERSATION_WARN_DAYS`).
No las cierra: una conversación con `ended_at NULL` no entra nunca a la cola. Cerrarla es
decisión de una persona, por la frontera autorizada de cierre.

## Embeddings

- El espacio de embeddings lleva el digest real del modelo (`/api/tags` de Ollama). Si Ollama
  no responde o no trae el modelo, `jax-memory-embedding` y `jax-memory-vector-health` fallan.
- Un digest nuevo abre un espacio nuevo y deja un WARNING («el digest del modelo cambió»);
  las revisiones se re-vectorizan en él, sin mezclarse con el espacio anterior. Mientras tanto
  vector-health queda en rojo.
- Una revisión cuyo embedding siempre falla se salta tras `JAX_MEMORY_EMBED_MAX_ATTEMPTS` (5)
  intentos, con ERROR en el journal; vector-health sigue en rojo por ella. Para reintentarla:
  `DELETE FROM embedding_generation_attempts WHERE revision_id='<id>'` (acción de una persona).
