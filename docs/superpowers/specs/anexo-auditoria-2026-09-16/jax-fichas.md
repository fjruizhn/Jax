1. delete: `_director_patch/*.py` (un parche viejo que ya está aplicado: `executor_block.py`, `plan_and_relaunch.py`, `routes_block.py`, `test_jacobs_director.py`) → borrar los 4 .py y sacar la entrada `"_director_patch/routes_block.py"` de `_NO_PARSEA` en 4 tests   [_director_patch/executor_block.py:L1-L248, routes_block.py:L1-L96, plan_and_relaunch.py:L1-L83, test_jacobs_director.py:L1-L132]
   llamadores: `grep -n "def _compute_waves\|def _run_one_step" jacobs/executor.py` → 980 y 1019 (el bloque ya está dentro de executor). `grep -n -i cleanroom jacobs/plan.py` → `_CLEANROOM_RULE` en 159 y `_check_cleanroom` en 243 (el ajuste 2 también está aplicado). `git grep -l _director_patch` → policy.yml:734 (comentario), CONTEXT.md, policy/tests/test_no_fail_open_except.py:74, tests/test_aiomysql_connect_timeout_tripwire.py:83 y :425, tests/test_no_blocking_in_async.py:40, tests/test_payload_max_tokens_literal_tripwire.py:69. Nadie lo importa. `git -C ~/jax-platform grep _director_patch` → 0.
   porqué: `git log --oneline -- _director_patch` → a39e5df "wip(jacobs): _director_patch — Jacobs Director (executor/routes/plan blocks + E2E fase2)". Encabezado del archivo: "REEMPLAZO PARA executor.py… Sustituye la función run_pipeline()". CONTEXT.md:538 lo llama "parche WIP suelto, ni siquiera parte del paquete".
   espejo: `grep -n _director_patch scripts/check_mirror_sync.py` → 0. No es una familia espejada.
   equivalencia: los escaneos con `rglob("*.py")` ven 4 archivos menos. `PISO_ARCHIVOS_ESCANEADOS = 170` (policy/tests/test_no_fail_open_except.py:309) hay que revisarlo contra la cuenta real. Los .md del mismo directorio no entran (ver No tocar).
   riesgo: medio
   tests: tests/test_aiomysql_connect_timeout_tripwire.py:421-431 `test_el_archivo_declarado_en_NO_PARSEA_no_rompe_la_corrida` exige que el archivo exista. Lo mismo hacen tests/test_no_blocking_in_async.py:189-191 y policy/tests/test_no_fail_open_except.py:336-347. Esa cobertura (que un archivo declarado que no parsea no rompa la corrida) se reemplaza con un .py roto en un tmpdir y `_NO_PARSEA` parcheado. También hay que ajustar el "Piso exacto de tests CORRIDOS" de policy.yml si cambia la cuenta.

2. delete: `class StepResult` → borrar   [jacobs/models.py:L155-L160]
   llamadores: conteo por regex `\bStepResult\b` sobre todos los .py de jax (tests incluidos) → 1, la definición. `git grep -l StepResult -- tests '*_test.py' policy/tests` → vacío. Mismo conteo sobre jax-platform (.py/.js/.jsx) → 0.
   porqué: `git log -S"class StepResult"` → 1d616d8 (2026-06-19, "Jacobs v0.2"). Sin comentario ni decisión.
   espejo: `grep StepResult scripts/check_mirror_sync.py` → 0. jax-platform solo importa `jacobs.store` (policy.yml:643 de jax-platform).
   equivalencia: ninguna. Es un modelo Pydantic que nadie instancia ni serializa.
   riesgo: bajo
   tests: ninguno (grep de arriba vacío)

3. delete: `VALID_FACETS` en jacobs/models.py, copia muerta de la de plan.py → borrar   [jacobs/models.py:L47-L49]
   llamadores: `grep -rn VALID_FACETS --include=*.py` → la definición de models.py:47, plan.py:81 (su propia definición) y plan.py:799 (usa la de plan.py). Ningún `from jacobs.models import VALID_FACETS`. `git -C ~/jax-platform grep VALID_FACETS` → 0.
   porqué: `git log -S'VALID_FACETS = frozenset' -- jacobs/models.py` → 1d616d8. Sin razón escrita.
   espejo: no está en check_mirror_sync.py. Las dos copias viven en el mismo repo y ya pueden divergir.
   equivalencia: ninguna, nadie lee la de models.py.
   riesgo: bajo
   tests: ninguno (`git grep VALID_FACETS -- tests '*_test.py'` → 0)

4. delete: `TRAFFIC_CLASSES`, una tupla "documentada para referencia" que repite el `Literal` de `Envelope.traffic_class` → borrar y dejar el comentario apuntando a envelope.py:75   [las_manos/audit.py:L40-L49]
   llamadores: conteo `\bTRAFFIC_CLASSES\b` en jax → 1. En jax-platform → 0. El propio comentario dice "audit.py no los valida (lo hace el Envelope vía Pydantic)". envelope.py:75-78 tiene los 6 mismos valores.
   porqué: `git log -S"TRAFFIC_CLASSES"` → e3f0e66 (2026-06-16, "Thot Audit Watch"). Es documentación, no una decisión que proteja algo.
   espejo: ninguno. `grep traffic_class scripts/check_mirror_sync.py` → 0.
   equivalencia: ninguna en ejecución. La lista queda solo en el `Literal`, que es la que se aplica.
   riesgo: bajo
   tests: ninguno (`git grep TRAFFIC_CLASSES -- tests` → 0; tests/test_audit_traffic_class.py usa strings literales)

5. delete: `MotorCatalog.enabled_motors()` → borrar   [las_manos/motor_registry/catalog.py:L181-L182]
   llamadores: conteo de métodos `\benabled_motors\b` en jax → 1, en jax-platform → 0. `git grep enabled_motors -- tests '*_test.py'` → 0.
   porqué: `git log -S"def enabled_motors"` → 1d616d8 (2026-06-19). Sin comentario.
   espejo: jax-platform no lo tiene ni lo llama (grep → 0).
   equivalencia: ninguna.
   riesgo: bajo
   tests: ninguno

6. delete: `voices/es_MX-ald-medium.onnx`, `voices/es_MX-claude-high.onnx` y sus `.onnx.json` (voces Piper de unos 63 MB cada una, en git) → borrar del árbol   [voices/*]
   llamadores: `git grep -n "es_MX\|voices/\|\.onnx\|piper"` en .py/.toml/.sh → 0. La voz real es Kokoro (jax/voice/tts.py:35-36 `KOKORO_PYTHON`, `kokoro_worker.py`; config/config.toml usa `voice_id = "em_alex"`). `git -C ~/jax-platform grep "voices\|\.onnx\|piper"` → 0. `grep -rn "voices\|onnx" /etc/systemd/system/jax-*.service` → 0. `grep -in "piper\|onnx" CONTEXT.md DEUDA.md README.md` → 0.
   porqué: `git log -- voices` → 06bd1ea (2026-06-15, commit inicial). Sin decisión escrita.
   espejo: ninguno.
   equivalencia: ninguna en ejecución. Ojo: borrarlos de HEAD no achica la historia del repo.
   riesgo: bajo
   tests: ninguno (`git grep voices -- tests` → 0)

7. delete: `scripts/cleanup.sh` (borra `*.backup*` de más de 30 días en ~/jax y ~/jax-platform y misiones viejas) → borrar   [scripts/cleanup.sh:L1-L6]
   llamadores: `git grep "cleanup.sh"` en jax → 0, en jax-platform → 0. `crontab -l` → sin entradas. `grep -rln cleanup.sh /etc/systemd /etc/cron* ~/.config/systemd` → 0. `ls ~/jax/missions` → vacío.
   porqué: `git log -- scripts/cleanup.sh` → 1d616d8 (2026-06-19). Sin comentario.
   espejo: ninguno.
   equivalencia: nadie lo ejecuta hoy. Si alguien lo corriera a mano, borraría los `*.backup-pre-*` de jacobs/ y las_manos/, y el home no está en restic.
   riesgo: bajo
   tests: ninguno

8. delete: secciones `[motors.kimi]` y `[motors.ada]` de la config TOML, que producción no lee desde R4 (el catálogo sale de la DB) → borrar y actualizar la fuente citada en closed_vocabulary.yaml   [las_manos/config.toml:L143-L184]
   llamadores: `git grep -n "\"motors\"\|motors\]"` en .py → solo el `governance["motors"]` que arma store.py:568 desde la DB, y dicts armados a mano en tests. `MotorCatalog(` con dict aparece solo en tests (`git grep "MotorCatalog("` → _tool_authority_test, _worker_max_tokens_test, _worker_tool_loop_test, _policy_test). routes.py:48-54 lee de `_CONFIG` solo `server.kill_switch_path`. Su comentario dice que "[motors.*]/[capabilities.*] de config.toml ya no se leen".
   porqué: `git log -S"[motors.kimi]"` → 1d616d8. catalog.py:5-8 y routes.py:48 dejan escrito que dejó de leerse. El valor ya está viejo: `model = "glm-5.2"` para ada, y PR-K midió glm-5.3 en el binding (plan.py:66-70).
   espejo: sí. policy/vocabulary/closed_vocabulary.yaml:67-76 cita "Fuente: las_manos/config.toml, secciones [motors.*]" para `motors: kimi, ada`, y ese vocabulario lo consume jax-platform (backend/shadow_validation.py, governance_context.py). jax-platform migrations.py:663 lo cita solo como historia. El arreglo tiene que corregir la fuente declarada en el yaml en el mismo paso.
   equivalencia: `load_validation_context` lee de ese TOML solo `ops` (validator.py:93-100), no `motors`. tests/test_governance_validator.py:148-157 hashea el archivo real en tiempo de test, así que el hash cambia pero el test lo recalcula. Una afirmación ya guardada con el hash viejo del archivo dejaría de coincidir.
   riesgo: medio
   tests: ninguno cambia (`git grep config.toml -- tests '*_test.py'` → ningún test lee `[motors]` del TOML)

9. delete: imports sin uso → borrarlos   [jacobs/executor.py:L18 `Any`, L20 `resolve_credential_instrumented, CredentialUnavailableError` (la línea entera), L21 `FacetUnavailableError`; jax/muscles/base.py:L23 `os`, L29 `decrypt_secret`; jax/muscles/ollama_muscle.py:L34 `MuscleTimeoutError`; las_manos/audit.py:L21 `os`; las_manos/policy.py:L15 `re`; las_manos/server.py:L31 `uuid`]
   llamadores: script AST que compara nombres importados contra `ast.Name` y constantes string de cada archivo → la lista de arriba. `git grep` de `executor\.(resolve_credential_instrumented|CredentialUnavailableError|FacetUnavailableError)`, `base\.(os|decrypt_secret)`, `ollama_muscle\.MuscleTimeoutError`, `server\.uuid` → 0. Los tests parchean `worker.resolve_credential_instrumented` y `base.resolve_credential_instrumented`, no `executor.` ni `base.decrypt_secret`. `from jax.muscles.ollama_muscle import` solo trae `OllamaMuscle` (main.py:49 y tests).
   porqué: `git log -S"import uuid" -- las_manos/server.py` → 06bd1ea (commit inicial). Los demás son restos de refactors. Ninguno tiene comentario.
   espejo: ninguno de estos archivos es espejo (check_mirror_sync compara símbolos con nombre, no imports).
   equivalencia: el import de `credential_resolver` en executor.py tiene un efecto al importarse, pero `facet_resolver` (L21, que se queda) ya importa `credential_resolver` en jax/core/facet_resolver.py:22, así que el módulo se sigue cargando.
   riesgo: bajo
   tests: ninguno

10. shrink: `las_manos/crypto_secrets.py`, copia real byte a byte de jax/core/crypto_secrets.py → symlink `../jax/core/crypto_secrets.py`, el mismo patrón que facet_resolver, redaccion y cola_uso   [las_manos/crypto_secrets.py:L1-L78]
   llamadores: `diff jax/core/crypto_secrets.py las_manos/crypto_secrets.py` → sin salida (idénticos). Lo importan en forma bare las_manos/server.py:41 y las_manos/credential_resolver.py:17. `git grep crypto_secrets -- tests '*_test.py'` → 0.
   porqué: `git log --diff-filter=A -- las_manos/crypto_secrets.py` → 54c6f09 (2026-08-09, "descifrar API keys Fernet en memory-worker y las_manos"). El docstring de la familia en check_mirror_sync.py:191-194 dice "Tres archivos reales otra vez -- ninguno es symlink, medido el 2026-09-01": describe lo que había, no decide conservarlo. El archivo no importa nada de `jax.*`, así que no necesita doble import.
   espejo: familia `crypto_secrets` en scripts/check_mirror_sync.py:187-222 (CI policy.yml:385-394). Con symlink la comparación dentro de jax queda como no-op a propósito, igual que en las otras familias. La copia de jax-platform no se toca. Hay que actualizar la `nota` de la familia ("TRES archivos reales").
   equivalencia: el contenido es idéntico, así que no cambia nada en ejecución. Un clone fresco en CI conserva el symlink (ya pasa con 7 symlinks más).
   riesgo: bajo
   tests: ninguno cambia. Conviene un assert `is_symlink()` como el de tests/test_redaccion.py:26-31.

11. canon: `las_manos/credential_resolver.py` y `las_manos/model_catalog.py`, segundas copias reales dentro de jax (difieren solo en el import y, en model_catalog, en el texto de un comentario) → en el canónico de jax/core, doble import con `try: from crypto_secrets / from db_connect_config … except ImportError: from jax.core…` como facet_resolver.py:19-35, y reemplazar las copias por symlinks   [las_manos/credential_resolver.py:L1-L146, las_manos/model_catalog.py:L1-L129]
   llamadores: `diff jax/core/credential_resolver.py las_manos/credential_resolver.py` → solo L17-25 (imports). `diff jax/core/model_catalog.py las_manos/model_catalog.py` → L22-28 (imports) y el texto del comentario `# fail-soft:` de L122/128. Importan en forma bare: jacobs/executor.py:20 y :23, jacobs/plan.py:24, las_manos/motor_registry/worker.py:36, jax/core/facet_resolver.py:22. Tests: las_manos/_motor_usage_writer_test.py, _catalog_from_db_test.py, scripts/_check_mirror_sync_test.py:180 (solo el docstring). `git -C ~/jax-platform grep "from model_catalog"` → 0.
   porqué: `git log --diff-filter=A` → cb4d606 (credential_resolver, 2026-08-09) y 179387e (model_catalog, 2026-08-09). check_mirror_sync.py:228-235 deja anotado que la copia "NO es un symlink, es un TERCER ARCHIVO REAL… hasta hoy nadie la comparaba": es una constatación, no una decisión. model_catalog ni siquiera está en `FAMILIAS` (`grep model_catalog scripts/check_mirror_sync.py` → 0), así que su copia no la vigila nadie.
   espejo: sí. credential_resolver es familia de check_mirror_sync con copia en jax-platform backend/credential_resolver.py. Ese lado usa `from crypto_secrets import`, y el `ImportFrom` no se compara (check_mirror_sync.py:236-240), así que el doble import no genera drift. Hay que actualizar la `nota`. model_catalog no tiene espejo en jax-platform (grep → 0).
   equivalencia: con PYTHONPATH `.:las_manos` (CI), el orden "bare primero" hace que `jax.core.credential_resolver` y `credential_resolver` sigan siendo dos módulos con cachés distintos, como hoy. En el REPL (solo `.`) el bare falla y cae a `jax.core`, que es el comportamiento actual. `test_no_fail_open_except` deja de ver el sitio `except` de la copia como archivo aparte: se resuelve por `path.resolve()` (L85) y queda una sola marca `# fail-soft:`, la del canónico. Hay que elegir cuál de los dos textos queda.
   riesgo: medio
   tests: scripts/_check_mirror_sync_test.py:180 (docstring). Asserts `is_symlink()` nuevos como tests/test_contrato_dispatch_repl_ada.py:777. Los jobs de policy.yml con `PYTHONPATH=las_manos` (L322, L375) hay que confirmarlos en CI.

12. native: `aiofiles` usado en un solo lugar para escribir un .md → `await asyncio.to_thread(filepath.write_text, content, encoding="utf-8")`; sacar `aiofiles` de requirements.txt y del `pip install` de CI   [jacobs/executor.py:L1221, L1257-L1258; requirements.txt:L1; .github/workflows/policy.yml:L557-L561]
   llamadores: `git grep -nE "^\s*(import|from) aiofiles"` → solo executor.py:1221. `git grep aiofiles -- tests` → 0. tests/test_hipatia_fuentes.py:204-213 ejercita `_persist_step_to_repo` leyendo el archivo que queda en disco.
   porqué: `git log -S"import aiofiles"` → bfe9699 (2026-06-29). El comentario de CI (policy.yml:557-558) solo explica por qué se instala.
   espejo: ninguno. jax-platform tiene su propio requirements.
   equivalencia: el mismo archivo y el mismo encoding. `write_text` abre en modo "w" igual que `aiofiles.open(..., "w")`, y los errores de E/S suben igual (OSError). Además saca del loop el trabajo bloqueante. tests/test_no_blocking_in_async.py mira llamadas, no referencias pasadas a `to_thread` (L97).
   riesgo: bajo
   tests: ninguno cambia (test_hipatia_fuentes.py:204 sigue valiendo)

13. canon: el tope de pasos `20` repetido como literal → usar `MAX_STEPS_PER_PIPELINE` de jacobs/policy.py   [jacobs/routes.py:L100-L104; jacobs/models.py:L129-L130]
   llamadores: `git grep -n "max_steps > 20"` → routes.py:100 (`plan_only`) y models.py:129 (`PipelineCreateRequest`). La constante está en policy.py:16 y la usa `validate_create`. jacobs/_plan_timeout_ceiling_test.py menciona `max_steps`.
   porqué: `git log -S"max_steps > 20"` → 1d616d8. Sin comentario que justifique tenerlo repetido.
   espejo: ninguno.
   equivalencia: el mismo valor y los mismos códigos (422). Cambiar la constante movería los tres lugares a la vez, y hoy no lo hace. models.py no importa policy (policy importa models: policy.py:12), así que la constante tiene que vivir en models.py o en un módulo sin ciclo.
   riesgo: bajo
   tests: ninguno (`git grep "max_steps" -- tests` → 0; _plan_timeout_ceiling_test.py no verifica el 20)

14. shrink: `texto = text[:4000] if len(text) > 4000 else text` → `texto = text[:4000]`   [jax/memory/db.py:L484]
   llamadores: lo usa `get_embedding`. Tests: test_degradaciones_declaradas.py, test_memoria_no_traga_fallos_de_escritura.py, test_memory_embedding_config.py, test_memory_scope_denormalized.py, test_memory_vector_zero_io.py (`git grep -l get_embedding -- tests`).
   porqué: `git log -S"text[:4000] if len(text) > 4000"` → 06bd1ea (commit inicial). Sin comentario sobre la forma.
   espejo: ninguno.
   equivalencia: para `str` el slice ya devuelve el string entero si mide menos de 4000. Si `text` fuera `None` las dos formas fallan (`len(None)` y `None[:4000]`), y el `except` de L499 lo atrapa igual.
   riesgo: bajo
   tests: ninguno

15. shrink: el filtro `[l for l in texto.splitlines() if not l.strip().startswith("⚙️ *Origen")]` repetido en dos métodos → un helper del módulo `_sin_autoetiqueta(texto)`   [jax/muscles/base.py:L270-L273, L336-L339]
   llamadores: `git grep -n 'Origen")'` → base.py:272 y :338. `git grep Origen -- tests` → solo policy/tests/test_no_deploy_from_unmerged_branch.py, que no tiene relación.
   porqué: `git log -S'⚙️ *Origen'` no dejó una razón para duplicarlo, y los comentarios de las dos copias son iguales.
   espejo: ninguno.
   equivalencia: idéntica, las dos hacen `"\n".join(...).strip()`.
   riesgo: bajo
   tests: ninguno

## Defectos
- **Errores de proveedor guardados sin redactar.** `jacobs/executor.py:386` (`resp.text[:200]` en `_invoke_http_openai_compat`) y `:448` (Ollama) mandan el cuerpo del error a `RuntimeError`, y de ahí a `jacobs_steps.error` vía `_fail_step`. No pasa por `recortar_redactado`, que redaccion.py:161-167 declara "la única forma de recortar un texto de error de proveedor en jax" (el camino de Gemini, L292, sí la usa). Lo mismo en el REPL: `jax/muscles/base.py:264` (DeepSeek), `:305` (OpenAI, `body[:200]`) y `jax/muscles/ollama_muscle.py:119`. Además recorta antes de redactar, justo el orden que el módulo prohíbe. No verificado en vivo si cada proveedor devuelve la key en el cuerpo.
- **El token de sub-pipeline se valida solo por presencia.** `jacobs/policy.py:48`: `invoked_by == "ada" and not subpipeline_token` deja pasar cualquier string no vacío como autorización. `git grep subpipeline_token` → no se valida en ningún otro lado.
- **El límite de profundidad de sub-pipeline nunca actúa.** `jacobs/routes.py:146-152` llama `validate_create(...)` sin `subpipeline_depth`, así que siempre vale el default 0 y `MAX_SUBPIPELINE_DEPTH` (policy.py:69) no puede dispararse.
- **Una faceta desconocida se cambia en silencio.** `jacobs/plan.py:799-800`: una faceta que el LLM propone y no está en `VALID_FACETS` se reemplaza por `jax_local` sin log ni evento. La capability, en el mismo bucle (L815-819), sí deja un warning. Esto choca con P10.
- **La justificación del `# fail-soft:` es falsa.** `jacobs/executor.py:1067` dice que "nadie lee ese .md" de `~/jax/repo/documents`. Pero `jax-platform/backend/api/admin/repository.py:10-11` (`REPO_BASE = ~/jax/repo`, `ALLOWED_FOLDERS` con "documents") lo lista y lo sirve en `/api/admin/repo`. La marca de la auditoría fail-open se apoya en una premisa equivocada.
- **(jax-platform, visto de paso) La guarda de rutas se puede saltar.** `backend/api/admin/repository.py:14-19`: `_safe_path` usa `target.startswith(base)` sin separador, así que `documents/../../repo-x/…` resuelve a un hermano `~/jax/repo-x` y pasa la guarda. Solo superadmin, y hoy no existe ese hermano (no verificado en vivo).
- **Faltan dependencias en requirements.txt.** No trae `cryptography` (importado por jax/core/crypto_secrets.py:16 y las_manos/crypto_secrets.py:16) ni `pyyaml` (policy/governance/loaders.py:23, policy/tools/*.py). README.md:72 indica `pip install -r requirements.txt`, así que ese entorno no puede importar credential_resolver ni la gobernanza.

## No tocar sin decisión de Fernando
- **`jacobs/policy.py:96-98` `hyde_requires_human_gate()`**: devuelve `True` fijo y no tiene ningún llamador (conteo → 1). Parece un candado pero no protege nada; el gate real sale de `capability.requires_human_gate` en la DB (store.py:509-538). Es un chequeo de autoridad: borrarlo o conectarlo lo decide Fernando.
- **Fallback a variables de entorno (B1.4)**: `resolve_credential_instrumented` y `_PROVIDER_ENV_KEY_MAP` (jax/core/credential_resolver.py:29-39, 124-139) tienen un criterio de salida pendiente ("7 días sin source=env_fallback"). Queda fuera de alcance. Dato relacionado: el timer `/etc/systemd/system/check-b14-exit-criterion.timer` dice 2026-08-16 en la descripción y `OnCalendar=2026-08-26`, y `ops/b14-exit-criterion.log` no existe. No hay registro del veredicto (no verificado en vivo, no se leyó el journal). `ops/check-b14-exit-criterion.sh` sigue ahí por ese mismo criterio.
- **`KeyProvider`/`EnvKeyProvider`** (jax/core/crypto_secrets.py:19-37): una abstracción con una sola implementación, pero con decisión documentada ("el día que se mueva a un KMS/Vault", jax-platform/docs/fase1-credenciales-diseno.md B1.3) y replicada igual en jax-platform/backend/crypto_secrets.py:9-28.
- **`jacobs/usage_writer.py` vs `las_manos/motor_registry/usage_writer.py`**: casi duplicados, pero el módulo documenta la decisión "copia adaptada, no un import cruzado" (L7-13, "trade-off documentado desde Fase 1") y los dos difieren en comportamiento (reintentos, `status`/`job_id`).
- **`las_manos/facet_client.py:192-248` (`AuditFinding`, `AuditReview`, `review_audit`)**: solo lo usa tests/test_thot_connection.py. Es la demostración del criterio 6 de la Mesa del 15-jun-2026 (d6c5c77).
- **`_director_patch/CAPABILITIES_CONTRACT.md` y `E2E_FASE2_RESULTADO.md`**: son volcados de evidencia ("para decisión de corrección de raíz"), no código. Moverlos a docs/ o borrarlos lo decide Fernando.
- **Backups sin trackear** `jacobs/*.backup-pre-*` y `las_manos/**/*.backup-pre-*` (14 o más, de 2026-08-21 y 08-27; `git status --ignored`): no están en git y el home no está en restic. Borrarlos es irreversible.

## regla:
- **Sin hardcoding (URLs):** `jacobs/executor.py:44-45` (`LAS_MANOS_BASE`, `OLLAMA_URL`), `jacobs/plan.py:46`, `jax/memory/db.py:487` (`http://localhost:11434/api/embed`), `jax/muscles/ollama_muscle.py:62`, `jax/muscles/base.py:285` (`https://api.openai.com/v1/chat/completions` por defecto).
- **Sin hardcoding (rutas):** `jax/voice/tts.py:35` (`~/kokoro-test/.venv`), `jacobs/executor.py:1224` (`~/jax/repo/documents`, repetido en jax-platform repository.py:10), `jacobs/policy.py:14` (`/etc/jax/PAUSE` fijo, mientras LAS MANOS lo lee de `server.kill_switch_path` en config.toml): dos fuentes.
- **Fuente de verdad:** `jacobs/plan.py:81-83` `VALID_FACETS` es una lista fija de facetas cuando la verdad vive en las tablas `facet`/`facet_binding`. El router del REPL ya la trata solo como fallback de arranque (router.py:59-62); el planner la usa como vocabulario cerrado.
- **Cache (clientes HTTP compartidos):** se crea un `httpx.AsyncClient` nuevo por llamada en `jax/memory/db.py:485` (`get_embedding`, camino caliente de memoria), `jacobs/executor.py:285/383/445` y `jax/muscles/base.py:260/300`.

neto estimado: unas -1.330 líneas de código y config (más -994 líneas de JSON y unos 126 MB de binarios en voices/), -1 dependencia (aiofiles), si se aplicaran todas las fichas antes de la verificación de terceros