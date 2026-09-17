# Hallazgos de la auditoría de jax + contrato de sub-pipelines — diseño

> Fecha: 2026-09-16 · Autor: Mr. Hyde · Decisiones: Fernando (chat, 2026-09-16).
> Origen: auditoría de `jax` en `bd95237` con la skill `auditando-sobre-ingenieria`,
> verificada por terceros (anexo `anexo-auditoria-2026-09-16/`). Gemelo del spec de
> jax-platform `2026-09-16-hallazgos-auditoria-design.md` (frentes A-D), del que se
> heredan las **reglas comunes** (TDD con rojo contra el código viejo, i18n, sin
> hardcoding, fail-closed, cachés con invalidación, las cuatro del rendimiento, barrera de
> DB de producción en tests, CI verificado rompiéndolo, mirror-sync de los dos repos en el
> mismo paso, deploy con 0 pipelines en vuelo, registro en la Biblioteca).
> **Regla de Fernando: se corrige todo hoy.**

## E. Limpieza, defectos y reglas de jax

### E.1 Aplicar (VÁLIDO, o PARCIAL con remedio ya corregido)

| id | Cambio | Nota del verificador |
|---|---|---|
| E-01 | Borrar `_director_patch/*.py` (4). Los `.md` se mueven a `docs/historia/director_patch/`. Las pruebas de `_NO_PARSEA` que exigían la existencia del archivo se reconstruyen con un `.py` roto en tmpdir (si no, el bucle pasa sin comprobar nada) | policy.yml:734 comentario; piso 170 < 205 |
| E-02 | Borrar `StepResult` | |
| E-03 | `VALID_FACETS` deja de ser lista fija: el planner valida contra la tabla `facet` (diseño C1.4 #6); se borran las dos copias. Junto con E-17 | la ficha proponía borrar la copia equivocada |
| E-04 | Borrar `TRAFFIC_CLASSES` (el `Literal` de envelope.py manda) | |
| E-05 | Borrar `MotorCatalog.enabled_motors()` | |
| E-06 | Borrar `voices/*.onnx*` (Piper sin uso, 121 MB; quedan en historia) | |
| E-07 | Borrar `scripts/cleanup.sh` **y** actualizar el docstring de `jax-platform/backend/jax_engine/owner_cleanup.py:5-11` que lo nombra; dejar escrito quién retiene `~/jax/missions` (el reaper de web-tasks de jax-platform, ajuste `web_task_retention_days` del frente C) | la ficha decía 0 menciones en jax-platform: FALSO |
| E-08 | Borrar `[motors.kimi]`/`[motors.ada]` de `las_manos/config.toml`; `closed_vocabulary.yaml` cita la tabla `motor` como fuente | |
| E-09 | Imports sin uso (lista de la ficha 9) | |
| E-10 | `las_manos/crypto_secrets.py` → symlink; `nota` de la familia; assert `is_symlink()` | |
| E-11 | `las_manos/credential_resolver.py` y `model_catalog.py` → symlinks con doble import en el canónico; actualizar docstrings (`db_connect_config.py:31-32`, `closed_vocabulary.yaml:72`, `nota` de la familia); declarar que en contexto `.:las_manos` el canónico usa los módulos bare | el scanner no deduplica por `resolve()`: inocuo, declararlo |
| E-12 | `aiofiles` → `asyncio.to_thread` (incluido el `mkdir`); quitar de requirements y del CI (comentario L557) | |
| E-13 | `MAX_STEPS_PER_PIPELINE` vive en `jacobs/models.py`; policy, routes y el validador lo importan; los mensajes leen la constante | importar desde policy crea un ciclo |
| E-14 | `text[:4000]` | |
| E-15 | Helper `_sin_autoetiqueta`; el prefijo sale de la config (`authority_origin`), no literal | |

### E.2 Defectos

| id | Defecto | Arreglo |
|---|---|---|
| E-16 | Cuerpos de error de proveedor recortados ANTES de redactar y sin la credencial conocida (`executor.py:386,448`, `base.py:264,306`, `ollama_muscle.py:119`, **`plan.py:655` a log sin redactar**, **`main.py:487` a archivo sin redactar**) | `recortar_redactado(texto, 200, [credencial])` en todos; test de política que prohíbe `[:N]` sobre cuerpos de respuesta de proveedor |
| E-17 | Faceta desconocida del plan LLM reemplazada por `jax_local` en silencio (`plan.py:799`) | Rechazo del plan (`PlanRejected`, 422, evento) validando contra la tabla `facet` |
| E-18 | Justificación falsa del `# fail-soft:` (`executor.py:1067`) | Reescribir: copia de cortesía visible en el admin; el canónico está en `output_ref` |
| E-19 | `requirements.txt` sin `cryptography` ni `pyyaml` | Fijarlas (`cryptography==49.0.0`, `pyyaml==6.0.3`, versiones de los venvs vivos) y que CI instale desde el archivo |
| E-20 | `hyde_requires_human_gate()` devuelve `True` fijo y nadie la llama | Borrarla (el gate real: `capability.requires_human_gate`); test que lo documenta |

### E.3 Reglas

| id | Regla | Arreglo |
|---|---|---|
| E-21 | URLs fijas: `LAS_MANOS_BASE`, `OLLAMA_URL` (executor, plan), embed de memoria, OpenAI por defecto | Variables de entorno con validación al arrancar (fail-closed) |
| E-22 | Rutas fijas: venv de Kokoro, `~/jax/repo/documents` (compartida con jax-platform `REPO_BASE`) | Variables de entorno; la de documentos es la misma en los dos repos |
| E-23 | `VALID_FACETS` fija | Ver E-03/E-17 |
| E-24 | `httpx.AsyncClient` nuevo por llamada (executor 287/383/445/577/598/656, base 260/301/412, plan ~749, memoria 485) | Cliente compartido por proceso con ciclo de vida (cierre al apagar), timeouts por llamada; medir antes/después |

### E.4 Pendientes con contrato vencido y otros

| id | Qué | Cómo |
|---|---|---|
| E-25 | Fallback de credenciales a env (B1.4): el timer `check-b14-exit-criterion` venció el 2026-08-26 y no dejó veredicto | **Medir hoy**: journal de los 3 servicios, últimos 7 días, `source=env_fallback`. Si 0 y hubo rotación → retirar el fallback en los dos repos (espejado) y el timer. Si aparece alguno → identificar el consumidor y reportar a Fernando antes de tocar |
| E-26 | 25 backups `*.backup-pre-*` sin trackear (el home no está en restic) | Borrar solo los que sean byte-idénticos a un blob de git (`git hash-object` contra `git rev-list --all --objects`); listar el resto a Fernando |
| — | Se conservan con motivo: `KeyProvider` (B1.3/R2), `usage_writer` duplicado a propósito, `review_audit` (criterio 6 de la Mesa) | |

## F. Contrato de sub-pipelines (Ada multiagente)

Decisión de Fernando: Ada está en el sistema por su capacidad multiagente; **no se quita**,
se construye el contrato completo hoy (Principio IX) y el emisor se diseña con él hoy mismo.

- **Emisión (servidor):** `jacobs` emite `subpipeline_token` para un pipeline padre en
  ejecución y un step de Ada: 32 bytes aleatorios (`secrets.token_urlsafe`), se guarda
  **solo el hash** (sha256) en `jacobs_subpipeline_tokens` (hash PK, parent_pipeline_id,
  parent_step, depth_hijo, emitido_at, vence_at, usado_at NULL, hijo_pipeline_id NULL;
  índice por parent_pipeline_id). TTL configurable por env, default acotado.
- **Validación en `validate_create`:** para `invoked_by="ada"`: el hash existe, no venció,
  `usado_at IS NULL`, el padre sigue activo; el consumo es atómico
  (`UPDATE … SET usado_at=NOW(), hijo_pipeline_id=? WHERE hash=? AND usado_at IS NULL`,
  una fila afectada o rechazo). **La profundidad sale de la fila del token**
  (`depth_hijo = profundidad_del_padre + 1`), nunca del cuerpo; `MAX_SUBPIPELINE_DEPTH`
  por env. El pipeline hijo guarda `parent_pipeline_id` y `depth`.
- **Auditoría:** eventos `SUBPIPELINE_TOKEN_EMITIDO`, `SUBPIPELINE_CREADO`,
  `SUBPIPELINE_RECHAZADO` (con motivo, sin el token) en `jacobs_events`.
- **Kill switch:** un hijo respeta el mismo interruptor (frente B).
- **Pruebas de ataque (peor caso):** token inventado, reusado, de otro padre, vencido,
  padre terminado, profundidad excedida, carrera de dos consumos simultáneos → todos
  rechazados; camino legítimo → aceptado, con evento. Arnés de test que hace de Ada.
- **Fuera de este frente:** el emisor (cómo decide Ada lanzar un sub-pipeline, qué puede
  delegar, límites de costo, human gate) → sesión de diseño con Fernando hoy.
