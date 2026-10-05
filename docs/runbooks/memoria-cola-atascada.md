# Memoria: cola de extracción atascada y avisos de las unidades

Origen: auditoría escalón 3 del 2026-10-05 (y la de Jax#354). Este procedimiento no concede permiso de
producción; las acciones de abajo las ejecuta una persona autorizada.

## Códigos de salida (aviso-fallo distingue el incidente por `ExecMainStatus`)

| Unidad | Código | Significa |
|---|---|---|
| `jax-memory-worker` | 0 | sano |
| | 1 | fallo general: extracción, conexión, configuración inválida |
| | 2 | **solo** hay jobs atascados esperando a una persona (si además hubo fallos nuevos, sale 1) |
| `jax-memory-vector-health` | 0 | sano |
| | 1 | faltan vectores o fallo general (Ollama no responde, no trae el modelo o el digest) |
| | 3 | el tripwire pide volver a medir el recall del HNSW (`messages` creció 10x sobre las 1.607 filas medidas) |
| `jax-memory-embedding` | 1 | fallos de vectorización, o el digest del modelo cambió a mitad de la corrida |

## Qué significa que `jax-memory-worker` salga 2

Hay jobs en `memory_extraction_jobs` que esperan a una persona: `QUARANTINED`, o `UNKNOWN` que se
quedó (ver abajo). `pending()` los excluye, así que sin este conteo una cola «vacía» saldría verde con
conversaciones que nunca se destilaron. Con `OnFailure=aviso-fallo@%n.service` el fallo llega a Telegram.

Causas de la cuarentena: límites o autoridad de la FUENTE (mensajes de más, caracteres de más, origen
sin permiso, fuente cambiada), commit inconsistente o intentos agotados. Una salida mal formada del
extractor NO cuarentena al primer intento: se reintenta con backoff hasta `JAX_MEMORY_MAX_ATTEMPTS`
(default 3) con `error_code=EXTRACTOR_OUTPUT_INVALID`. Un límite de configuración inválido
(`JAX_MEMORY_MAX_*`, etc.) hace fallar la corrida (código 1) ANTES de reclamar o llamar al extractor y
no cuarentena nada.

### UNKNOWN: cuándo cuenta y por qué `--reencolar` lo rechaza

`UNKNOWN` significa «el commit se envió pero no hubo acuse». No lo resuelve una persona: lo resuelve el
siguiente reclamo, que lee los marcadores de commit (`pending()` lo incluye). Por eso solo cuenta como
atascado si superó `JAX_MEMORY_MAX_ATTEMPTS` intentos o lleva más de `JAX_MEMORY_LEASE_SECONDS` sin
cambiar. `--reencolar` lo rechaza: re-encolarlo descartaría esa resolución y podría duplicar memoria.
Si un `UNKNOWN` se queda, investigar los marcadores (`memory_extraction_results`, `memory_processed`).

## Resolver una cuarentena

1. Listar: `python -m jax.memory.worker --atascados` (conversación, estado, error, intentos).
2. Revisar la conversación y el `error_code` ANTES de actuar.
3. Re-encolar lo ya revisado:
   `python -m jax.memory.worker --reencolar <conversation_id> --motivo "<por qué>" [--actor <quién>]`.
   - Solo acepta `QUARANTINED`; deja el job en `READY` con intentos 0.
   - **Rechaza** los `INCONSISTENT_*` (los marcadores de commit se contradicen; re-encolar podría
     duplicar memoria: requieren reconciliación manual) y las conversaciones con
     `memory_processed=TRUE` (ya se destilaron).
   - No toca `frozen_output`/`input_digest`: una cuarentena por «fuente cambiada» volverá a caer.
   - Escribe el evento `REQUEUE` en `memory_extraction_job_events` con el **usuario real del proceso**
     (por uid) y, aparte, el `--actor` declarado en `details.declared_actor`.
   - **Quién es la persona.** El comando necesita las credenciales de `/etc/jax/.env`, así que se corre con
     `sudo` (como `jaxsvc` o `root`): el «actor real» del evento será `jaxsvc` o `root`, no tú. La persona
     se rastrea en el **registro de sudo** (`journalctl _COMM=sudo` o `/var/log/auth.log`, a la hora del
     evento `occurred_at`) y en el `--actor` declarado. Poner siempre un `--actor` verdadero.
4. Auditoría: `SELECT * FROM memory_extraction_job_events WHERE conversation_id=<id>`.
5. La unidad vuelve a verde en la corrida siguiente cuando no queda ningún atascado.

## Conversaciones abiertas hace más de 7 días

El worker las lista en el journal como WARNING (`JAX_MEMORY_OPEN_CONVERSATION_WARN_DAYS`). Se mide por
la **última actividad** (el mensaje más reciente, o `started_at` si no tiene mensajes): una conversación
vieja pero con mensajes recientes no está abandonada. No las cierra: una conversación con `ended_at NULL`
no entra nunca a la cola; cerrarla es decisión de una persona, por la frontera autorizada de cierre.

## Embeddings

- El espacio de embeddings lleva el digest real del modelo (`/api/tags` de Ollama). Si Ollama no
  responde o no trae el modelo, `jax-memory-embedding` y `jax-memory-vector-health` fallan.
- Un digest nuevo abre un espacio nuevo y deja un WARNING («el digest del modelo cambió»); las
  revisiones se re-vectorizan en él, sin mezclarse con el espacio anterior. Mientras tanto vector-health
  queda en rojo.
- **Riesgo residual del digest.** Se lee al inicio de la corrida y otra vez al final; si cambió a mitad,
  la corrida sale 1 y lo generado en ella no es confiable (puede haber vectores de dos modelos bajo un
  mismo digest). Aun así, un cambio que ocurra justo entre la última lectura y el final del proceso, o
  Ollama sirviendo pesos distintos con el mismo digest, no se detecta. Tras un fallo así: re-ejecutar
  la unidad con el modelo estable y revisar `embedding_generations` de esa ventana.
- Una revisión cuyo embedding siempre falla se salta tras `JAX_MEMORY_EMBED_MAX_ATTEMPTS` (5) intentos,
  con ERROR en el journal; vector-health sigue en rojo por ella. Para reintentarla:
  `DELETE FROM embedding_generation_attempts WHERE revision_id='<id>'` (acción de una persona).

## Recall del HNSW (código 3 de vector-health)

Hay que volver a medir el recall contra la búsqueda exacta (el mensaje del journal dice cómo) y,
con el resultado, subir `FILAS_AL_MEDIR_RECALL` en `jax/memory/recall_tripwire.py` en el mismo cambio
que registra la medición.

## Marcha atrás de las migraciones 007–011

Ver `jax/memory/b9_migrations/README.md`.
