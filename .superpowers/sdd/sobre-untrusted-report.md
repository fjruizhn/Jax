# Sobre fuente no confiable — read_file envuelve, no ejecuta

Rama `feat/sobre-fuente-no-confiable`, worktree `/home/fruiz/worktrees/jax-sobre-untrusted`.

## Qué se hizo

`las_manos/motor_registry/tool_authority.py::_read_file` ahora envuelve el contenido
leído en:

```
<untrusted_source path="RUTA/RELATIVA" sha256="...">
…contenido…
</untrusted_source>
```

Antes de envolver, `_neutralize_injection_sentinels()` desactiva (espacio de ancho cero
U+200B insertado después del primer carácter, NUNCA se borra) cualquier token de control
de plantilla de chat que el archivo pudiera traer: `<|loquesea|>` (cualquier forma, no una
lista enumerada — issue graphify #3183, se le escapaban los de Llama 3), `<<SYS>>`/`<</SYS>>`,
`[INST]`/`[/INST]`/`[SYSTEM]`/`[/SYSTEM]`, una línea que sea sólo `### system:`/`## instruction:`,
y el propio `</untrusted_source>` (así un archivo no puede forjar un cierre temprano y sacar
instrucciones afuera). El sha256 se calcula sobre el contenido ORIGINAL, antes de neutralizar.

`las_manos/motor_registry/worker.py` contabiliza el presupuesto acumulado de lectura
(`MAX_TOTAL_READ_BYTES`) con el nuevo campo `bytes_read` (tamaño crudo, puesto por
`tool_authority._read_file` a propósito) en vez de `len(result["content"])` — mismo patrón
que `bytes_written` para `write_file`, con el mismo comentario explicando por qué los dos
números son distintos.

**Fuera de alcance, no tocado**: `_write_file` (su `content` es un mensaje de estado
nuestro, no texto de un tercero), el jail, los límites, la gobernanza, y
`las_manos/workers/file_worker.py` (worker SSH remoto, namespace de operaciones
completamente distinto — ver "Consumidores", abajo).

## Procedencia y atribución

Idea y regex tomadas de **graphify** (Apache License 2.0), `Graphify-Labs/graphify`,
`graphify/llm.py` (rama `v8`, commit `ec12e5e341580eacbd6f15859ecd0260ac85055f`,
2026-09-10), funciones `_neutralise_injection_sentinels()`/`_wrap_untrusted()`
(líneas ~550-600 de ese archivo). Copyright 2026 Safi Shamsi y los contribuyentes de
Graphify. Licencia completa en `las_manos/motor_registry/LICENSE-graphify` (copia íntegra
del `LICENSE` del repo origen, idéntica byte a byte al patrón ya usado en
`claude-skills/common/skills/LICENSE-finance-anthropic`), y referencia con commit/versión
en el comentario que antecede `_INJECTION_SENTINELS` en `tool_authority.py`.

## Consumidores del `content` de read_file — comando y resultado

```
grep -rn "authorize_and_execute_tool_call" --include="*.py" .
```
→ Solo tres sitios en todo el árbol (`las_manos/`, `jacobs/`, `scripts/`, `tests/`):
- `las_manos/motor_registry/worker.py:1008` — el ÚNICO caller de producción. Consume
  `result["content"]` en dos lugares: (a) `messages.append({"role": "tool", ...})` — lo que
  el modelo ve de vuelta (afectado A PROPÓSITO, es el objetivo del cambio); (b) el cálculo
  de `total_read_bytes` (corregido para usar `bytes_read`, no `content`).
- `las_manos/_tool_authority_test.py:115,202` — llamador directo de tests. Actualizado.
- `las_manos/_worker_tool_loop_test.py` (vía `worker.run()`) — llamador indirecto de
  tests. Actualizado.

```
grep -rln "read_file" --include="*.py" las_manos jacobs scripts tests policy
```
→ Aparece además en `las_manos/planner.py`, `las_manos/facet_client.py`,
`las_manos/envelope.py`, `las_manos/motor_registry/tools_catalog.py`,
`las_manos/workers/file_worker.py`, `jacobs/plan.py`, y varios `tests/test_governance_*`,
`tests/test_plan_*`, `tests/test_read_audit_log_unbound.py`. Verificado uno por uno: todos
usan el string `"read_file"` como **nombre de capability/operación en un catálogo**
(gobernanza, planificación SSH con snapshot y gate humano — `jacobs/plan.py` →
`las_manos/planner.py` → `las_manos/workers/file_worker.py`, un sistema de lectura/escritura
remota por SSH **completamente separado** del bucle de tool-calling de Motor Registry), NO
consumen el `content` devuelto por `tool_authority._read_file`. Ninguno se ve afectado. No
se tocaron.

`las_manos/server.py` menciona `tool_authority` sólo para importar `WORKSPACE_ROOT`; no
consume `content`.

## Suite completa — comando y resultado

Verificado en **Python 3.12 (Docker, `python:3.12`)**, usuario no-root (el root del
contenedor bypasea permisos de archivo y falseaba en verde
`test_archivo_sin_permisos_es_execution_error` — no es un efecto de este cambio, se
documenta y se descartó de la contabilidad).

Comando (el mismo de `.github/workflows/policy.yml`, job `tests-puros`, sin
`JAX_DB_HOST` — así corre el job real, según su propia nota):

```
PYTHONPATH=.:las_manos python -m pytest -q <los 163 archivos de la lista de ese job>
```

| Checkout | failed | passed | skipped | xfailed | subtests |
|---|---|---|---|---|---|
| master (`/home/fruiz/jax`, sin cambios), 2 corridas | 17 (estable, mismo set) | 2340 | 18 | 1 | 16 |
| esta rama, 2 corridas | 19 (estable, mismo set) | 2348 | 18 | 1 | 16 |

Total corrido+fallado: master `2357`, rama `2367` → delta exacto **+10**, igual al número
de tests nuevos (`grep -c "^    async def test_" las_manos/_tool_authority_test.py
las_manos/_worker_tool_loop_test.py`: `26→35` y `24→25`).

Ninguna de las diferencias en el CONJUNTO de fallos toca `tool_authority.py`, `worker.py`
ni sus dos archivos de test — confirmado con
`grep -i "tool_authority\|worker_tool_loop\|read_file" <log> | grep -i "FAIL\|ERROR"` → sin
resultados. Las diferencias son:
- **+4 en la rama** (`tests/test_interruptor_sin_rutas_fijas.py`): `git ls-files` devuelve
  exit 128 dentro del contenedor porque sólo monté el directorio del worktree, no el
  `.git` real del repo principal (`/home/fruiz/jax/.git/worktrees/jax-sobre-untrusted`) —
  limitación de MI arnés de verificación (git worktree), no del código. `actions/checkout@v4`
  en CI real entrega un clon plano, no un worktree, así que esto no ocurre ahí. Confirmado
  aislando el archivo y reproduciendo el mismo `CalledProcessError`.
- **+2 en master** (`test_env_se_lee_con_sudo.py`, `test_retiro_de_la_voz.py`): pasan
  solos en AMBOS checkouts (`pytest tests/test_env_se_lee_con_sudo.py
  tests/test_retiro_de_la_voz.py` → `6 passed`); sólo fallan dentro de la corrida completa
  de 163 archivos, y sólo bajo el contenedor de master, de forma estable en 2 corridas. No
  tocan `read_file`/`tool_authority`/`worker` — ruido de orden entre tests preexistente,
  no de este diff.

`policy/tests/test_archivos_de_test_wireados_en_ci.py` (el guardia de que todo archivo de
test esté enganchado a algún job): `7 passed`, sin cambios — no agregué archivos de test
nuevos, sólo tests a archivos ya wireados en las dos listas del job `tests-puros`.

## Piso de CI actualizado (mismo commit)

`.github/workflows/policy.yml`, job `tests-puros`: el bloque histórico `2317 -> 2334 ...`
**no se tocó**. Se agregó la entrada `2358 -> 2368` (nueva, al final) y se subió el `grep`
del piso de `"^2358 passed, 17 skipped"` a `"^2368 passed, 17 skipped"` — delta de +10
medido contra master con el comando exacto de ese paso, no contra el passed crudo local
(que difiere por los artefactos de contenedor de arriba, documentados en el propio
comentario). `"17 skipped"` no cambia: ninguno de los 10 tests nuevos toca DB.

## Tests existentes que se rompieron — y por qué (información, no estorbo)

Dos aserciones en tests VIEJOS comparaban el `content` crudo EXACTO del mensaje `role:tool`
— es la consecuencia directa y esperada de envolver, no una regresión:

1. `las_manos/_tool_authority_test.py::test_1_read_file_legitimo_ejecuta` — comparaba
   `r["content"] == "contenido legítimo\n"`. Actualizado para comparar contra el envoltorio
   exacto (tag + sha256 + contenido + tag de cierre).
2. `las_manos/_worker_tool_loop_test.py::test_8_9_historial_correcto_y_tool_call_id_correlacionado`
   (línea del historial que el modelo ve de vuelta) y
   `test_read_after_write_lee_lo_escrito_no_cache` — mismo patrón, actualizados a esperar
   el envoltorio (el primero con igualdad exacta del wrapper completo; el segundo con
   `in`/`startswith` más la verificación de `bytes_read`).

Ambos vistos en rojo contra el código VIEJO antes de escribir la implementación (ver
mutación M1 en la tabla de abajo, que es exactamente "revertir el envoltorio" — reproduce
el mismo rojo).

## Mutation testing — 6 mutaciones, cada una vista roja

Harness: copia de trabajo aislada (`rsync --exclude=.git`), UNA mutación por corrida sobre
`tool_authority.py`/`worker.py`, `python -m pytest -q las_manos/_tool_authority_test.py
las_manos/_worker_tool_loop_test.py` en Docker (`python:3.12`), copia resincronizada limpia
entre mutaciones. La fila `test_archivo_sin_permisos_es_execution_error` en las columnas de
abajo es el mismo artefacto de root-en-contenedor de arriba (aparece en las 6 corridas por
igual, no es señal de la mutación) y se excluye del conteo de "tests que la detectan".

| # | Mutación | Cómo se aplicó | Tests que la detectan (rojo) |
|---|---|---|---|
| M1 | Quitar el envoltorio (`content: content` en vez de `content: wrapped`) | `_read_file` devuelve el contenido crudo sin pasar por `_wrap_untrusted_source` | 11: los 9 nuevos de `_tool_authority_test.py` que dependen del wrapper (envuelve, bytes_read, sha256-original, pipe, llama3, corchetes, línea-system, cierre-forjado) + `test_1_read_file_legitimo_ejecuta` (10) + 2 en `_worker_tool_loop_test.py` (`test_8_9_historial...`, `test_read_after_write...`) |
| M2 | Dejar de neutralizar el cierre forjado | se saca `r"</?untrusted_source\b[^>]*>"` del regex `_INJECTION_SENTINELS` | 1: `test_read_file_neutraliza_cierre_forjado_no_escapa_el_bloque` |
| M3 | El presupuesto cuenta el tamaño ENVUELTO, no el crudo | en `worker.py` se vuelve a `len(result["content"].encode("utf-8"))` en vez de `result.get("bytes_read", 0)` | 1: `test_presupuesto_de_lectura_cuenta_bytes_crudos_no_el_envoltorio` |
| M4 | Dejar de neutralizar `<\|pipe\|>` | se saca `r"<\|[A-Za-z0-9_.\-]{1,64}\|>"` del regex | 2: `test_read_file_neutraliza_pipe_token_de_plantilla`, `test_read_file_neutraliza_tokens_llama3_no_enumerados` |
| M5 | No neutralizar NADA (`safe = content`) | `_wrap_untrusted_source` deja de llamar a `_neutralize_injection_sentinels` | 5: `test_read_file_neutraliza_cierre_forjado_no_escapa_el_bloque`, `..._corchetes_inst_y_system`, `..._linea_system_sola`, `..._pipe_token_de_plantilla`, `..._tokens_llama3_no_enumerados` |
| M6 | sha256 sobre el texto YA neutralizado, no el original | se calcula `sha` después de `safe = _neutralize_injection_sentinels(content)`, sobre `safe` | 1: `test_read_file_sha256_es_sobre_el_original_no_el_neutralizado` |

Las 6 mutaciones quedaron cada una con AL MENOS un test en rojo. Ninguna pasó
desapercibida.

## Verificación final (sin mutar, worktree real)

```
PYTHONPATH=.:las_manos python -m pytest -v las_manos/_tool_authority_test.py las_manos/_worker_tool_loop_test.py
```
→ **60 passed** (35 en `_tool_authority_test.py`, 25 en `_worker_tool_loop_test.py`).

## Archivos tocados

- `las_manos/motor_registry/tool_authority.py` — `_wrap_untrusted_source`,
  `_neutralize_injection_sentinels`, `_INJECTION_SENTINELS`; `_read_file` envuelve y
  devuelve `bytes_read`.
- `las_manos/motor_registry/worker.py` — contabiliza `bytes_read` en vez de `content`.
- `las_manos/motor_registry/LICENSE-graphify` — nueva, licencia Apache 2.0 íntegra.
- `las_manos/_tool_authority_test.py` — 9 tests nuevos + 1 actualizado (`test_1`).
- `las_manos/_worker_tool_loop_test.py` — 1 test nuevo + 2 actualizados.
- `.github/workflows/policy.yml` — piso del job `tests-puros` `2358 → 2368`, nueva entrada
  de historial (el bloque `2317 passed` no se tocó).

## Reservado a Fernando / fuera de alcance (reportado, no tocado)

- `las_manos/workers/file_worker.py` tiene su PROPIO `read_file` (lectura remota por SSH,
  con gate humano y snapshot) — mismo nombre de operación, sistema completamente distinto.
  No envuelve nada porque no pasa por Motor Registry / tool-calling; si algún día un
  extracto leído por ahí también termina en el contexto de un modelo sin pasar por
  `jacobs/plan.py`'s gate humano, esa ruta necesitaría su propio envoltorio — no se tocó
  por estar fuera del alcance del encargo.
