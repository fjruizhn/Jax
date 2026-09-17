# Frente E · jax: limpieza, defectos y reglas: plan de implementación

> **Para agentes ejecutores:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development`
> (recomendado) o `superpowers:executing-plans` para implementar tarea por tarea. Los pasos usan
> casillas (`- [ ]`) para seguimiento.

**Objetivo:** cerrar hoy E-01..E-26 del spec de hallazgos de jax. Hay que retirar el código muerto, arreglar los siete
defectos, cumplir las reglas de URLs, rutas y clientes HTTP, y resolver los dos pendientes con contrato vencido con una
medición que decide antes de tocar nada.

**Arquitectura:** cada hallazgo entra con un test que falla contra el código viejo, salvo que la tarea diga por qué eso no
se puede. Se suman tres piezas nuevas:
- `jax/core/config_entorno.py`: URL y ruta obligatorias del entorno, fail-closed.
- `jax/core/cliente_http_compartido.py`: un `httpx.AsyncClient` por event loop, con cierre al apagar.
- La validación de facetas contra la tabla `facet`, dentro de la misma consulta de gobernanza que ya hace el planner.

Las copias espejadas dentro de jax pasan a ser symlinks. Los cambios que tocan jax-platform van en la rama gemela, en el
mismo paso.

**Stack:** Python 3.12 en CI y 3.14 en los venvs de hall9000. Se usan pytest, unittest, httpx 0.28.1, aiomysql 0.3.2,
pydantic 2 y FastAPI 0.139. No hay dependencias nuevas: `cryptography` y `pyyaml` ya estaban en uso y sólo se fijan.

**Spec:** `docs/superpowers/specs/2026-09-16-hallazgos-auditoria-jax-design.md` §E. Las reglas comunes se heredan de
`jax-platform/docs/superpowers/specs/2026-09-16-hallazgos-auditoria-design.md` §0. La evidencia está en
`docs/superpowers/specs/anexo-auditoria-2026-09-16/`, y los veredictos del verificador pisan las fichas.

---

## Discrepancias con el spec

Cada una lleva su evidencia (medida el 2026-09-16 sobre jax `bd95237` y jax-platform `26c9cd5`, en solo lectura).

1. **E-07: hoy nadie retiene `~/jax/missions`, y el reaper de jax-platform tampoco lo hace.**
   `jax-platform/backend/jax_engine/owner_cleanup.py:28-45` (`reap_orphaned_command_owner_files`) sólo borra
   `web-task-*_owner.json`. No toca `web-task-*.md` ni `*_result.md`, que siguen escribiendo `api/command.py:33-34` y
   `jax/core/main.py:473`. El spec pide dejar escrito que retiene "el reaper de web-tasks (frente C)". Eso sólo es verdad
   si el frente C hace que el reaper borre también misión y resultado según `web_task_retention_days`. La Tarea 3 trae
   los dos textos exactos del docstring y la regla para elegir uno. Un docstring que afirma una retención inexistente es
   una memoria falsa.
2. **E-21: el spec deja afuera sitios de la misma clase, y la URL de Ollama tiene hoy tres fuentes.**
   - `jax/muscles/base.py:243` (DeepSeek) y `:378` (Gemini) tienen defaults literales iguales al de OpenAI (`:284`).
   - `jax/muscles/ollama_muscle.py:62` tiene `DEFAULT_OLLAMA_URL`.
   - `config/config.toml` trae `api_url` en `jax_local` (L35), `ada` (L284) y `kimi` (L334).
   - `jax-platform/backend/api/chat.py:705` (`_call_ollama`) lee `personalities.jax_local.api_url` del config.toml de jax.

   El plan hace lo siguiente:
   - La URL de Ollama sale de `JAX_OLLAMA_URL` en todos los consumidores: Jacobs, planner, memoria, REPL y el
     `_call_ollama` de jax-platform, con un cambio pareado.
   - Se quita `api_url` del config.toml.
   - Los defaults de proveedor (OpenAI, DeepSeek y Gemini) se **borran** y no se pasan a variables de entorno. La fuente
     de verdad es `provider.base_url` (el catálogo), y una variable sería una segunda fuente. Sin URL del catálogo, el
     músculo no despacha y lo dice.
   - Consecuencia declarada: con la DB caída, el REPL ya no despacha a esos proveedores. Hoy lo hacía con URL fija y
     credencial de fallback.
3. **E-24: la lista del spec tiene cuatro huecos.** Construyen un `httpx.AsyncClient` por llamada y no están en la lista:
   `jacobs/reaper.py:100`, `las_manos/motor_registry/worker.py:164` (el camino de kimi),
   `jax/core/grounding_sources.py:128` y `jax/muscles/ollama_muscle.py:114`. El plan cierra la clase con un tripwire que
   recorre todo el árbol de servicio, así que entran los cuatro.
4. **E-16: tres ajustes de forma.**
   - `jax/core/main.py:487` no es un cuerpo de proveedor recortado. Es el error de la tarea escrito entero en
     `<tarea>_result.md`. Se redacta con `redactar_secretos` sin recortar a 200: recortar el archivo de diagnóstico
     perdería la causa.
   - `jacobs/plan.py:655` y `jax/muscles/base.py:306` reciben `bytes` (`await resp.aread()`) y se decodifican antes de
     `recortar_redactado`.
   - `jax/muscles/ollama_muscle.py:127` (`str(data)[:200]`) es el JSON ya parseado de un 200 con forma inesperada del
     Ollama local, que no lleva credencial. Queda fuera del detector, y el detector lo declara.
5. **E-25: "el journal de los 3 servicios" no alcanza.**
   - Los consumidores de `resolve_credential_instrumented` corren en cuatro unidades: `jax-platform`, `jax-las-manos`,
     `jax-memory-worker` y `jax-memory-synthesis`.
   - Hay un quinto consumidor, el REPL. Corre en tmux, fuera de journald (`~/.local/bin/jax`), y su `env_fallback` no se
     puede medir con el journal.
   - El gate suma una consulta a la DB: los cinco proveedores de `_PROVIDER_ENV_KEY_MAP` tienen que tener credencial
     `active`.
   - **Contradicción entre los dos specs:** §A.5 del spec de jax-platform lista "fallback de credenciales B1.4" entre lo
     que se conserva. E-25 lo retira si el gate da limpio. El plan sigue a E-25, que es más específico y tiene medición,
     pero el resultado del gate se le reporta a Fernando nombrando A.5 antes de tocar jax-platform (Tarea 14, Paso 6).
6. **E-03/E-17: la tabla `facet` tiene `status ENUM('active','degraded','disabled')`**
   (`jax-platform/backend/db/migrations.py:290`), y `resolve_facet` sólo resuelve `status='active'`
   (`jax/core/facet_resolver.py:246`). El planner valida contra las facetas `active`, que es lo mismo que puede
   despachar. La semilla incluye `hyde` (`migrations.py:613`), pero el estado real en producción no está verificado: la
   Tarea 6 lo mide antes de desplegar.
7. **E-14 no puede tener un test rojo.** `text[:4000]` y el ternario son equivalentes para todo `str`, así que un test
   pasaría igual contra el código viejo, y un control que no falla no valida. Se aplica sin test y se dice.
8. **E-11:** el verificador corrigió la ficha: `test_no_fail_open_except` no deduplica por `resolve()`. Un symlink se
   escanea dos veces con la misma marca, lo que es inocuo, y se declara en la `nota` de la familia.
9. **E-19:** las versiones vivas son `cryptography-49.0.0` en `.venv` y en `las_manos/.venv`, y `pyyaml-6.0.3` sólo en
   `.venv` (listado de `site-packages`, 2026-09-16). En producción nadie hace `pip install` en el deploy, así que no hay
   efecto en vivo.
10. **E-15 cambia comportamiento a propósito.** El filtro sólo limpiaba la autoetiqueta de kimi (`⚙️`). Con la cabecera
    tomada de la etiqueta configurada, también se limpia la que imite DeepSeek (`🧠`).
11. **E-01:** la copia hermana de `test_no_fail_open_except.py` en jax-platform no tiene `_NO_PARSEA` (grep = 0), así que
    no hay nada que replicar allá.

## Solapamientos con los frentes en paralelo

Si hay conflictos, se rebasea sobre lo que ya esté mergeado y se **vuelven a medir los pisos de CI en el runner**.

| Archivo | Frente E (este plan) | Otro frente | Qué hacer |
|---|---|---|---|
| `jacobs/policy.py` | E-13 (importa la constante), E-20 (borra `hyde_requires_human_gate`) | **B** (`KILL_SWITCH_PATH` → `JAX_KILL_SWITCH_PATH`), **F** (`validate_create` con token y profundidad) | Merge textual. E-13 toca sólo el import y la línea de la constante |
| `jacobs/routes.py` | E-13 (`PlanRequest`, chequeo de `plan_only`) | **B** (`check_kill_switch`), **F** (`create_pipeline` con token) | Mismo bloque de `plan_only`: rebasear y correr `tests/test_max_steps_una_sola_constante.py` |
| `jacobs/models.py` | E-02, E-03 (borrar `VALID_FACETS`), E-13 (constante y defaults) | **F** (`PipelineCreateRequest`, `Pipeline.parent_pipeline_id/depth`) | Merge textual en la misma clase |
| `jacobs/store.py` | E-03 (`get_motor_governance` devuelve `facets`) | **F** (tabla `jacobs_subpipeline_tokens` en `init_tables`) | Funciones distintas |
| `las_manos/config.toml` | E-08 (borra `[motors.*]`) | **B** (borra `server.kill_switch_path`) | Regiones distintas |
| `las_manos/server.py` | E-09 (`uuid`), E-24 (hook de shutdown), E-25 condicional (comentario del logger) | **B** (lectura del kill switch) | Merge textual |
| `las_manos/motor_registry/worker.py` | E-24 (cliente compartido), E-25 condicional | **B** (kill switch del worker) | Merge textual |
| `jax/core/main.py` | E-16 (`_texto_de_error_de_tarea`), E-21 (`build_muscles`), E-24 (cierre) | **B** (kill switch del REPL) | Merge textual |
| `conftest.py` (raíz jax) | E-21/E-22 (fija `LAS_MANOS_URL`, `JAX_OLLAMA_URL`, `JAX_REPO_BASE_DIR`) | **B** (probablemente `JAX_KILL_SWITCH_PATH` para los tests) | Las dos asignaciones conviven |
| `.github/workflows/policy.yml` (jax) | Listas y pisos de `tests-puros` y `jacobs-gobernanza-db`, `pip install -r` | **B** y **F** (tests nuevos); **A** (piso de `mirror-sync` si A-22 agrega tests del checker) | **Nunca sumar a mano**: medir en el runner después del rebase |
| `scripts/check_mirror_sync.py` + `_check_mirror_sync_test.py` | E-10/E-11 (`nota` y comentarios), E-25 condicional (`compartidos`) | **A** (A-22: familia nueva de keywords de chat) | Familias distintas. El piso 14 del job lo mueve A |
| jax-platform `api/chat.py` | E-21 (`_call_ollama` lee `JAX_OLLAMA_URL`), E-25 condicional (renombre) | **A** (A-04 borra `_load_jax_env`, A-16 unifica el if/else de `_call_ollama`, A-51/A-53 códigos), **D** (imágenes por transporte, Ollama `images`) | **Mismo cuerpo de función que A-16 y D.** Coordinar el orden. Quien rebasee conserva `_url_de_ollama()` |
| jax-platform `api/image.py` | E-25 condicional | **A** (A-04, A-14, A-30) | Merge textual |
| jax-platform `api/admin/repository.py` | E-22: sólo verifica que `REPO_BASE` lea `JAX_REPO_BASE_DIR` | **A** (A-55 hace el cambio; también A-26, A-31, A-40) | A es dueño del archivo. Si A eligió otro nombre de variable, **este frente adopta el de A** (Tarea 9, Paso 1) |
| jax-platform `jax_engine/owner_cleanup.py` | E-07 (docstring) | **C** (`web_task_retention_days`) | El texto depende de lo que implemente C (Tarea 3) |
| jax-platform `.github/workflows/policy.yml` | E-21 (`JAX_OLLAMA_URL` en el `env` de los dos jobs backend) | **A/C/D** (pisos) | Agregar sólo la variable |
| `/etc/jax/.env` | `JAX_OLLAMA_URL`, `JAX_KOKORO_PYTHON`, `JAX_REPO_BASE_DIR` | **B** (`JAX_KILL_SWITCH_PATH`); **A** (A-54/A-55 variables) | Un backup por frente, con timestamp. Verificar con `grep -c` que ninguna variable quede duplicada |
| `DEUDA.md`, `CONTEXT.md` (jax) | Tarea 18 | **Todos** | Secciones propias. Rebasear el PR de docs al final |

---

## Restricciones globales

Heredadas del §0 del spec de jax-platform, con los valores exactos:

- **TDD:** test rojo contra el código viejo antes del arreglo, porque un control que no falla no valida. El rojo se ve y se
  anota en el commit.
- **i18n:** ningún texto visible literal. es/en en paridad. Este frente no toca frontend, y los textos nuevos son
  excepciones o logs de servidor, no UI.
- **Sin hardcoding:** la configuración va en `/etc/jax/.env` o en la DB.
- **Fail-closed.** Todo caché declara su invalidación en el mismo commit.
- **Las cuatro del rendimiento:**
  - índice verificado con EXPLAIN sobre la consulta real;
  - nada bloqueante dentro de `async def`;
  - prueba de carga con número registrado para todo camino de usuario modificado.
- **Tests con barrera de DB de producción:** todo test que abre conexión exige `JAX_DB_NAME == "jax_memory_test"` y, si
  no, `raise RuntimeError` al importar. Los tests que no abren conexión fijan `os.environ["JAX_DB_NAME"] =
  "jax_memory_test"` antes de importar jacobs.
- **CI:** todo test nuevo lo corre un job y entra en **las dos listas** del job (la de `pytest -v` y la del contador del
  piso). Los pisos se fijan con lo que cuenta el runner, **nunca con una resta a mano**. Se verifica rompiéndolo (rojo en
  el sha real, leído por API).
- **Mirror-sync:** si se toca un símbolo espejado (`scripts/check_mirror_sync.py`), el cambio va en los dos repos en el
  mismo paso.
- **Deploy:**
  - `sudo systemctl restart jax-las-manos.service` con 0 pipelines en vuelo.
  - Para jax-platform, `sudo systemctl restart jax-platform.service` con 0 pipelines en vuelo.
  - El REPL lo reinicia Fernando en tmux.
- **Registro en la Biblioteca** (`jax/DEUDA.md` y `jax/CONTEXT.md`) antes de cerrar.
- **Marca P10:** todo `except` amplio que siga de largo lleva `# fail-soft: <razón>` o `# fail-closed: <razón>` **en la
  línea del `except`**.
- **Nunca cambiar de rama en `/home/fruiz/jax` ni en `/home/fruiz/jax-platform`:** son los checkouts que sirven
  producción. Todo se hace en los worktrees `/home/fruiz/worktrees/jax-frente-e` y `/home/fruiz/worktrees/jax-platform-frente-e`.
- **Commits:** mensaje en castellano, terminado en la línea
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`. Los PR terminan con
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **Comando de tests local (jax).** `/home/fruiz/jax/.venv` tiene todas las dependencias (Python 3.14). Nunca con
  `/etc/jax/.env` cargado. Todos los "Run" de este plan usan este prefijo:
  ```bash
  cd /home/fruiz/worktrees/jax-frente-e && env -u JAX_DB_HOST -u JAX_DB_PORT -u JAX_DB_USER -u JAX_DB_PASSWORD \
    JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos /home/fruiz/jax/.venv/bin/python -m pytest
  ```
  En los pasos aparece abreviado como `$PYTEST_JAX`.
- **Comando de tests local (jax-platform):**
  `cd /home/fruiz/worktrees/jax-platform-frente-e/backend && JAX_REPO_PATH=/home/fruiz/worktrees/jax-frente-e .venv/bin/python -m pytest`
  (el venv es el de `/home/fruiz/jax-platform/backend/.venv`; si el worktree no lo tiene, usar la ruta absoluta
  `/home/fruiz/jax-platform/backend/.venv/bin/python`). Abreviado `$PYTEST_PLAT`.

---

### Task 0: Preparación: worktrees, línea base medida y gate de facetas

**Archivos:** ninguno del árbol. Todo lo que se mide va al scratchpad de la sesión (`$SCRATCH`).

**Interfaces:**
- Consume: nada.
- Produce: las ramas `fix/hallazgos-frente-e` en los dos repos, `$SCRATCH/linea-base.txt` y `$SCRATCH/facetas-produccion.txt`.

- [ ] **Paso 1: crear los worktrees desde el master remoto**

```bash
git -C /home/fruiz/jax fetch origin && git -C /home/fruiz/jax-platform fetch origin
git -C /home/fruiz/jax worktree add -b fix/hallazgos-frente-e /home/fruiz/worktrees/jax-frente-e origin/master
git -C /home/fruiz/jax-platform worktree add -b fix/hallazgos-frente-e /home/fruiz/worktrees/jax-platform-frente-e origin/master
git -C /home/fruiz/worktrees/jax-frente-e rev-parse --short HEAD; git -C /home/fruiz/worktrees/jax-platform-frente-e rev-parse --short HEAD
```

Esperado: `bd95237` y `26c9cd5`. Si alguno es otro (otro frente ya mergeó), anotarlo en `$SCRATCH/linea-base.txt` y
**volver a verificar contra ese sha las líneas que cita cada tarea antes de editarla**, porque un spec envejece.

- [ ] **Paso 2: línea base de los tests que este frente toca**

Run:
```bash
$PYTEST_JAX -q tests/test_aiomysql_connect_timeout_tripwire.py tests/test_no_blocking_in_async.py \
  tests/test_payload_max_tokens_literal_tripwire.py tests/test_hipatia_fuentes.py tests/test_contrato_dispatch_repl_ada.py \
  tests/test_gemini_key_en_cabecera.py tests/test_repl_fuentes.py tests/test_memory_embedding_config.py \
  jacobs/_plan_timeout_ceiling_test.py 2>&1 | tail -3 | tee -a "$SCRATCH/linea-base.txt"
cd /home/fruiz/worktrees/jax-frente-e && env JAX_PLATFORM_REPO_ROOT=/home/fruiz/worktrees/jax-platform-frente-e \
  /home/fruiz/jax/.venv/bin/python -m pytest -q policy/tests/test_no_fail_open_except.py 2>&1 | tail -2 | tee -a "$SCRATCH/linea-base.txt"
```

Esperado: todo en verde. Si algo ya está rojo en master, **parar y reportar**: no se construye sobre un rojo ajeno.

- [ ] **Paso 3: gate de facetas en producción (sólo lectura, alimenta E-17)**

Run:
```bash
set -a; source <(grep -E '^JAX_DB_(HOST|PORT|USER|PASSWORD|NAME)=' /etc/jax/.env); set +a
/home/fruiz/jax/.venv/bin/python - <<'PY' | tee "$SCRATCH/facetas-produccion.txt"
import asyncio, os, aiomysql
async def main():
    conn = await aiomysql.connect(host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.environ["JAX_DB_USER"], password=os.environ["JAX_DB_PASSWORD"], db=os.environ["JAX_DB_NAME"],
        connect_timeout=10)
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT `key`, status FROM facet ORDER BY `key`")
            for fila in await cur.fetchall():
                print(*fila)
            await cur.execute("EXPLAIN SELECT `key` FROM facet WHERE status = 'active'")
            print("EXPLAIN", await cur.fetchall())
    finally:
        conn.close()
asyncio.run(main())
PY
```

Regla de decisión:
- Si `hipatia`, `jekyll`, `thot`, `ada`, `kimi`, `hyde` y `jax_local` salen todas `active`, se sigue.
- Si alguna **no** está `active`, **parar la Tarea 6 y reportar a Fernando** qué faceta y en qué estado. Con E-17, un plan
  que la use pasará a 422 en vez de cambiarse en silencio, y eso tiene que saberse antes.

El EXPLAIN se anota tal cual. `facet` es un catálogo de 7 filas: un recorrido completo sin índice es aceptable y se
declara así en DEUDA (Tarea 18). Si el conteo crece más allá de un catálogo, la consulta necesita `idx_facet_status`.

---

### Task 1: E-01 · retirar `_director_patch/*.py` y reconstruir los controles de `_NO_PARSEA`

**Archivos:**
- Borrar: `_director_patch/executor_block.py`, `_director_patch/plan_and_relaunch.py`, `_director_patch/routes_block.py`, `_director_patch/test_jacobs_director.py`
- Mover: `_director_patch/CAPABILITIES_CONTRACT.md`, `_director_patch/E2E_FASE2_RESULTADO.md` → `docs/historia/director_patch/`
- Modificar: `policy/tests/test_no_fail_open_except.py:69-80` (tabla) y `:336-347` (test de la tabla)
- Modificar: `tests/test_aiomysql_connect_timeout_tripwire.py:82-92` (tabla) y `:421-429` (test)
- Modificar: `tests/test_no_blocking_in_async.py:39-44` (tabla) y `:186-192` (test)
- Modificar: `tests/test_payload_max_tokens_literal_tripwire.py:68-71` (tabla) y el final del archivo (tests nuevos)
- Modificar: `.github/workflows/policy.yml` (comentario de `tests-puros` que nombra `_director_patch/routes_block.py`)

**Interfaces:**
- Consume: nada.
- Produce: `_entradas_que_sobran(no_parsea: dict[str, str], raiz: Path) -> list[str]` en `policy/tests/test_no_fail_open_except.py`. Las cuatro tablas `_NO_PARSEA` quedan vacías (`{}`), con el mecanismo intacto.

- [ ] **Paso 1: borrar los `.py` y mover los `.md` a historia**

```bash
cd /home/fruiz/worktrees/jax-frente-e
mkdir -p docs/historia/director_patch
git mv _director_patch/CAPABILITIES_CONTRACT.md docs/historia/director_patch/CAPABILITIES_CONTRACT.md
git mv _director_patch/E2E_FASE2_RESULTADO.md docs/historia/director_patch/E2E_FASE2_RESULTADO.md
git rm _director_patch/executor_block.py _director_patch/plan_and_relaunch.py _director_patch/routes_block.py _director_patch/test_jacobs_director.py
rm -rf _director_patch
```

- [ ] **Paso 2: ver el rojo: los controles exigían que el archivo existiera**

Run:
```bash
$PYTEST_JAX -q tests/test_aiomysql_connect_timeout_tripwire.py tests/test_no_blocking_in_async.py 2>&1 | tail -5
cd /home/fruiz/worktrees/jax-frente-e && /home/fruiz/jax/.venv/bin/python -m pytest -q policy/tests/test_no_fail_open_except.py 2>&1 | tail -5
```

Esperado: FAIL en `test_el_archivo_declarado_en_NO_PARSEA_no_rompe_la_corrida`, en `test_un_no_modulo_declarado_no_ensucia`
y en `test_un_archivo_declarado_en_NO_PARSEA_sigue_sin_parsear`, los tres con "ya no existe".

- [ ] **Paso 3: vaciar las cuatro tablas**

En los cuatro archivos, reemplazar el bloque `_NO_PARSEA = { "_director_patch/routes_block.py": ( ... ), }` entero por:

```python
# Vacía desde el 2026-09-16 (E-01): el único no-módulo real del árbol,
# _director_patch/routes_block.py, se retiró. El mecanismo se conserva y se
# ejercita con un .py roto declarado a propósito dentro del test.
_NO_PARSEA: dict[str, str] = {}
```

En `tests/test_payload_max_tokens_literal_tripwire.py` la tabla es de strings directos (`"...": "fragmento..."`). Se
reemplaza con el mismo bloque.

- [ ] **Paso 4: reconstruir el control en `policy/tests/test_no_fail_open_except.py`**

Reemplazar la función `test_un_archivo_declarado_en_NO_PARSEA_sigue_sin_parsear` (L336-347) por:

```python
def _entradas_que_sobran(no_parsea: dict[str, str], raiz: Path) -> list[str]:
    """Entradas de _NO_PARSEA que ya no se justifican: el archivo no existe o
    ya parsea. Separado del test para ejercitarlo con archivos de mentira: con
    la tabla vacía (E-01, 2026-09-16) un bucle sobre la tabla real pasa sin
    comprobar nada."""
    sobran = []
    for rel in no_parsea:
        ruta = raiz / rel
        if not ruta.exists():
            sobran.append(f"{rel} ya no existe: retirar la entrada de _NO_PARSEA")
            continue
        try:
            ast.parse(ruta.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        sobran.append(f"{rel} ya parsea: retirar la entrada de _NO_PARSEA")
    return sobran


def test_ninguna_entrada_de_NO_PARSEA_sobra():
    assert _entradas_que_sobran(_NO_PARSEA, _THIS_REPO_ROOT) == []


def test_el_control_de_entradas_sobrantes_se_pone_rojo(tmp_path):
    (tmp_path / "roto.py").write_text("def f(:\n    pass\n")
    (tmp_path / "sano.py").write_text("x = 1\n")
    tabla = {"roto.py": "roto", "sano.py": "ya parsea", "fantasma.py": "no existe"}
    assert _entradas_que_sobran(tabla, tmp_path) == [
        "sano.py ya parsea: retirar la entrada de _NO_PARSEA",
        "fantasma.py ya no existe: retirar la entrada de _NO_PARSEA",
    ]


def test_un_archivo_roto_declarado_no_es_violacion(tmp_path, monkeypatch):
    roto = tmp_path / "roto.py"
    roto.write_text("def f(:\n    pass\n")
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOTS", [tmp_path])
    monkeypatch.setitem(_NO_PARSEA, "roto.py", "roto a propósito para este test")
    assert find_fail_open_excepts(files=[roto]) == []
```

- [ ] **Paso 5: reconstruir el control en `tests/test_aiomysql_connect_timeout_tripwire.py`**

Reemplazar el método `test_el_archivo_declarado_en_NO_PARSEA_no_rompe_la_corrida` (L421-429) por:

```python
    def test_un_archivo_roto_declarado_en_NO_PARSEA_no_rompe_la_corrida(self):
        """Control de la excepción explícita. Hasta el 2026-09-16 se probaba con
        _director_patch/routes_block.py; se retiró (E-01) y la tabla quedó
        vacía, así que el camino se ejercita con un .py roto declarado acá."""
        import tempfile
        from unittest.mock import patch
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=RAIZ, delete=True) as fh:
            fh.write("def f(:\n    pasa\n")
            fh.flush()
            ruta = Path(fh.name)
            rel = str(ruta.relative_to(RAIZ))
            with patch.dict(_NO_PARSEA, {rel: "roto a propósito para este test"}):
                self.assertEqual(_hallazgos_en(ruta), [])
            self.assertEqual(len(_hallazgos_en(ruta)), 1, "sin la declaración tiene que reportarse")
```

- [ ] **Paso 6: reconstruir el control en `tests/test_no_blocking_in_async.py`**

Reemplazar `test_un_no_modulo_declarado_no_ensucia` (L186-192) por:

```python
    def test_un_no_modulo_declarado_no_ensucia(self):
        """Control de la excepción: lo declarado no cuenta, y si algún día
        parsea, la entrada sobra. Con la tabla real vacía desde E-01
        (2026-09-16), la declaración se ejercita con un .py roto de mentira."""
        import tempfile
        from unittest.mock import patch
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=RAIZ, delete=True) as fh:
            fh.write("def f(:\n    pass\n")
            fh.flush()
            ruta = Path(fh.name)
            rel = ruta.relative_to(RAIZ).as_posix()
            with patch.dict(_NO_PARSEA, {rel: "roto a propósito para este test"}):
                self.assertEqual(_hallazgos_en(ruta), [])
        for rel in _NO_PARSEA:
            ruta = RAIZ / rel
            self.assertTrue(ruta.exists(), f"{rel} ya no existe: retirar de _NO_PARSEA")
            self.assertEqual(_hallazgos_en(ruta), [], f"{rel} esta declarado y no deberia dar hallazgo")
```

- [ ] **Paso 7: cubrir `_NO_PARSEA` en `tests/test_payload_max_tokens_literal_tripwire.py` (antes no tenía test)**

Agregar al final del archivo:

```python
class NoParseaTest(unittest.TestCase):
    """E-01 (2026-09-16): con la tabla vacía, las dos ramas del SyntaxError
    quedaban sin ejercitar. Se prueban con archivos de mentira."""

    def test_un_archivo_roto_no_declarado_se_reporta(self):
        hallazgos = _con_codigo("def f(:\n    pass\n")
        self.assertEqual(len(hallazgos), 1, hallazgos)
        self.assertIn("no parsea", hallazgos[0])

    def test_un_archivo_roto_declarado_no_se_reporta(self):
        from unittest.mock import patch
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=RAIZ, delete=True) as fh:
            fh.write("def f(:\n    pass\n")
            fh.flush()
            ruta = Path(fh.name)
            with patch.dict(_NO_PARSEA, {str(ruta.relative_to(RAIZ)): "roto a propósito"}):
                self.assertEqual(_hallazgos_en(ruta), [])
```

- [ ] **Paso 8: comentario de CI**

En `.github/workflows/policy.yml` (job `tests-puros`), reemplazar:
```
          #   un archivo roto no declarado se reporta; el único caso real del
          #   árbol, _director_patch/routes_block.py, se verifica declarado y sin
          #   romper la corrida). Las 9 formas nuevas, vistas en rojo contra el
```
por:
```
          #   un archivo roto no declarado se reporta; el único caso real del
          #   árbol entonces, _director_patch/routes_block.py, se verificaba
          #   declarado -- retirado el 2026-09-16 (E-01): el control se ejercita
          #   con un .py roto de mentira). Las 9 formas nuevas, vistas en rojo contra el
```

- [ ] **Paso 9: verde, y que el escaneo P10 siga por encima de su piso**

Run:
```bash
$PYTEST_JAX -q tests/test_aiomysql_connect_timeout_tripwire.py tests/test_no_blocking_in_async.py tests/test_payload_max_tokens_literal_tripwire.py 2>&1 | tail -2
cd /home/fruiz/worktrees/jax-frente-e && env JAX_PLATFORM_REPO_ROOT=/home/fruiz/worktrees/jax-platform-frente-e /home/fruiz/jax/.venv/bin/python -m pytest -v policy/tests/test_no_fail_open_except.py 2>&1 | tail -4
git -C /home/fruiz/worktrees/jax-frente-e grep -n "_director_patch" -- ':!docs' ':!CONTEXT.md'
```

Esperado:
- Los tres tripwires verdes, con 2 tests más que en la línea base, en `test_payload_max_tokens_literal_tripwire.py`.
- `test_no_fail_open_except.py` verde, con 2 tests más (se quita 1 y entran 3).
- El `git grep` sólo devuelve el comentario de `policy.yml` del Paso 8.
- `PISO_ARCHIVOS_ESCANEADOS = 170` sigue cumpliéndose: el escaneo baja de 209 a 205 archivos (medido por el verificador).

- [ ] **Paso 10: commit**

```bash
cd /home/fruiz/worktrees/jax-frente-e && git add -A _director_patch docs/historia policy/tests/test_no_fail_open_except.py \
  tests/test_aiomysql_connect_timeout_tripwire.py tests/test_no_blocking_in_async.py tests/test_payload_max_tokens_literal_tripwire.py \
  .github/workflows/policy.yml
git commit -m "limpieza(E-01): retirar _director_patch/*.py y ejercitar _NO_PARSEA con un .py roto de mentira" \
  -m "Los .md pasan a docs/historia/director_patch/. Visto en rojo: los tres controles exigian que el archivo existiera." \
  -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: E-02, E-04, E-05, E-06, E-09, E-20 · código muerto fuera

**Archivos:**
- Crear: `tests/test_frente_e_retiros.py`
- Modificar: `jacobs/models.py:155-160` (borrar `StepResult`)
- Modificar: `las_manos/audit.py:40-49` (borrar `TRAFFIC_CLASSES`), `:21` (`import os`)
- Modificar: `las_manos/motor_registry/catalog.py:181-182` (borrar `enabled_motors`)
- Borrar: `voices/es_MX-ald-medium.onnx`, `voices/es_MX-ald-medium.onnx.json`, `voices/es_MX-claude-high.onnx`, `voices/es_MX-claude-high.onnx.json`
- Modificar: `jacobs/executor.py:18,20,21`, `jax/muscles/base.py:23,29`, `jax/muscles/ollama_muscle.py:34`, `las_manos/policy.py:15`, `las_manos/server.py:31`
- Modificar: `jacobs/policy.py:96-98` (borrar `hyde_requires_human_gate`)
- Modificar: `.github/workflows/policy.yml` (las dos listas de `tests-puros`)

**Interfaces:**
- Consume: nada.
- Produce: `tests/test_frente_e_retiros.py` con `RAIZ` y `_importados(rel) -> set[str]`. Las Tareas 3 y 4 le agregan tests.

- [ ] **Paso 1: escribir los tests que fallan**

```python
# tests/test_frente_e_retiros.py
"""Frente E de la auditoría de jax (2026-09-16): lo que se RETIRA queda retirado.

Cada test falla contra el código anterior al retiro (visto en rojo antes de
borrar) y cita el id del hallazgo. Spec:
docs/superpowers/specs/2026-09-16-hallazgos-auditoria-jax-design.md §E.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import inspect
import os
import typing
from pathlib import Path

os.environ["JAX_DB_NAME"] = "jax_memory_test"  # barrera: nada acá abre conexión, y si algo se escapa no es producción

RAIZ = Path(__file__).resolve().parents[1]


def _importados(rel: str) -> set[str]:
    arbol = ast.parse((RAIZ / rel).read_text(encoding="utf-8"))
    nombres: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            nombres.update(a.asname or a.name.split(".")[0] for a in nodo.names)
        elif isinstance(nodo, ast.ImportFrom):
            nombres.update(a.asname or a.name for a in nodo.names)
    return nombres


def test_E02_stepresult_ya_no_existe():
    from jacobs import models
    assert not hasattr(models, "StepResult")


def test_E04_traffic_classes_ya_no_existe_y_el_literal_del_envelope_manda():
    import audit
    from envelope import Envelope
    assert not hasattr(audit, "TRAFFIC_CLASSES")
    valores = set(typing.get_args(Envelope.model_fields["traffic_class"].annotation))
    assert valores == {"test_structural", "test_semantic", "dry_run", "production", "adversarial_test", "unknown"}


def test_E05_enabled_motors_ya_no_existe():
    from motor_registry.catalog import MotorCatalog
    assert not hasattr(MotorCatalog, "enabled_motors")


def test_E06_las_voces_piper_ya_no_estan_en_el_arbol():
    assert sorted((RAIZ / "voices").glob("*.onnx*")) == []


_IMPORTS_RETIRADOS = {
    "jacobs/executor.py": {"Any", "resolve_credential_instrumented", "CredentialUnavailableError", "FacetUnavailableError"},
    "jax/muscles/base.py": {"os", "decrypt_secret"},
    "jax/muscles/ollama_muscle.py": {"MuscleTimeoutError"},
    "las_manos/audit.py": {"os"},
    "las_manos/policy.py": {"re"},
    "las_manos/server.py": {"uuid"},
}


def test_E09_imports_sin_uso_retirados():
    sobreviven = {rel: sorted(nombres & _importados(rel)) for rel, nombres in _IMPORTS_RETIRADOS.items()}
    assert {rel: n for rel, n in sobreviven.items() if n} == {}


def test_E20_el_gate_de_hyde_es_el_de_la_capability_y_no_una_funcion_que_devuelve_true():
    from jacobs import policy, store
    assert not hasattr(policy, "hyde_requires_human_gate")
    assert "requires_human_gate" in inspect.getsource(store.get_motor_governance)
```

- [ ] **Paso 2: ver el rojo**

Run: `$PYTEST_JAX -v tests/test_frente_e_retiros.py`
Esperado: 6 FAIL. E-20 falla por `hasattr`, no por la segunda aserción: la gobernanza ya lee `requires_human_gate`.

- [ ] **Paso 3: borrar**
  - `jacobs/models.py`: la clase `StepResult` completa (L155-160) y la línea en blanco sobrante.
  - `las_manos/audit.py`: L21 `import os`. Reemplazar el bloque L40-49 (comentario + tupla `TRAFFIC_CLASSES`) por:
    ```python
    # Valores válidos de traffic_class (Mesa, 16-jun-2026): el Literal de
    # Envelope.traffic_class (envelope.py) es la única lista y la que se aplica.
    # audit.py no los valida. La tupla que los repetía se retiró el 2026-09-16 (E-04).
    ```
  - `las_manos/motor_registry/catalog.py`: el método `enabled_motors` (L181-182) y su línea en blanco.
  - `git rm voices/es_MX-ald-medium.onnx voices/es_MX-ald-medium.onnx.json voices/es_MX-claude-high.onnx voices/es_MX-claude-high.onnx.json`
  - `jacobs/executor.py`: borrar L18 `from typing import Any` y L20 `from credential_resolver import resolve_credential_instrumented, CredentialUnavailableError`. En L21 dejar `from facet_resolver import resolve_facet, ResolvedFacet`.
  - `jax/muscles/base.py`: borrar L23 `import os` y L29 `from jax.core.crypto_secrets import decrypt_secret`.
  - `jax/muscles/ollama_muscle.py` L34: `from jax.muscles.base import DispatchConfigMuscleError, Muscle, MuscleInvocationError`.
  - `las_manos/policy.py`: borrar L15 `import re`. `las_manos/server.py`: borrar L31 `import uuid`.
  - `jacobs/policy.py`: borrar `def hyde_requires_human_gate()` completo (L96-98) y las líneas en blanco que quedan al final.

- [ ] **Paso 4: verde, y nada más roto**

Run:
```bash
$PYTEST_JAX -v tests/test_frente_e_retiros.py
$PYTEST_JAX -q tests/test_motor_catalog_mode.py tests/test_motor_catalog_sello.py tests/test_hipatia_fuentes.py tests/test_contrato_dispatch_repl_ada.py las_manos/_worker_tool_loop_test.py 2>&1 | tail -2
cd /home/fruiz/worktrees/jax-frente-e && /home/fruiz/jax/.venv/bin/python -m py_compile jacobs/executor.py jax/muscles/base.py jax/muscles/ollama_muscle.py las_manos/audit.py las_manos/policy.py las_manos/server.py
```
Esperado: 6 PASS, el resto verde y `py_compile` sin salida.

- [ ] **Paso 5: registrar el archivo en CI (las dos listas de `tests-puros`)**

En `.github/workflows/policy.yml`, en la lista de `python -m pytest -v` (después de `tests/test_respaldo_de_uso_aislado.py`), agregar
`          tests/test_frente_e_retiros.py`. En la lista del contador (después de `tests/test_respaldo_de_uso_aislado.py \`),
agregar `            tests/test_frente_e_retiros.py \`. El número del piso **no se toca acá**: lo fija la Tarea 16 con el runner.

- [ ] **Paso 6: commit**

```bash
cd /home/fruiz/worktrees/jax-frente-e && git add -A tests/test_frente_e_retiros.py jacobs/models.py las_manos/audit.py \
  las_manos/motor_registry/catalog.py voices jacobs/executor.py jax/muscles/base.py jax/muscles/ollama_muscle.py \
  las_manos/policy.py las_manos/server.py jacobs/policy.py .github/workflows/policy.yml
git commit -m "limpieza(E-02,E-04,E-05,E-06,E-09,E-20): StepResult, TRAFFIC_CLASSES, enabled_motors, voces Piper, imports sin uso y hyde_requires_human_gate" \
  -m "El gate real de Hyde es capability.requires_human_gate (store.get_motor_governance). Seis tests vistos en rojo." \
  -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: E-07 · retirar `scripts/cleanup.sh` y decir la verdad sobre la retención (jax + jax-platform)

**Archivos:**
- Borrar: `scripts/cleanup.sh` (jax)
- Modificar: `tests/test_frente_e_retiros.py` (se agrega 1 test)
- Modificar: `jax-platform/backend/jax_engine/owner_cleanup.py:1-18` (docstring del módulo)

**Interfaces:**
- Consume: `RAIZ` de `tests/test_frente_e_retiros.py` (Tarea 2).
- Produce: nada.

- [ ] **Paso 1: test que falla**

Agregar a `tests/test_frente_e_retiros.py`:

```python
def test_E07_cleanup_sh_retirado():
    """No tenía scheduler (crontab y systemd sin menciones, 2026-09-16) y, si
    alguien lo corría a mano, borraba *.backup* de un home que no está en restic."""
    assert not (RAIZ / "scripts" / "cleanup.sh").exists()
```

Run: `$PYTEST_JAX -v tests/test_frente_e_retiros.py::test_E07_cleanup_sh_retirado`. Esperado: FAIL.

- [ ] **Paso 2: borrar**

```bash
cd /home/fruiz/worktrees/jax-frente-e && git rm scripts/cleanup.sh
```

Run: `$PYTEST_JAX -v tests/test_frente_e_retiros.py`. Esperado: 7 PASS.

- [ ] **Paso 3: decidir el texto del docstring según lo que hace el frente C**

Run:
```bash
git -C /home/fruiz/jax-platform fetch origin
for ref in origin/master $(git -C /home/fruiz/jax-platform branch -r --list 'origin/fix/hallazgos-frente-c*'); do
  echo "== $ref"; git -C /home/fruiz/jax-platform grep -n "web_task_retention_days\|_result.md\|unlink" "$ref" -- backend/jax_engine/owner_cleanup.py
done
```

Regla:
- **Texto A**, si en algún ref el reaper borra `web-task-{id}.md` **y** `web-task-{id}_result.md` cuando superan
  `web_task_retention_days`.
- **Texto B**, en cualquier otro caso. Además, reportar a Fernando en una línea: "las misiones y resultados de web-tasks
  crecen sin techo; C no los retiene".

- [ ] **Paso 4: reemplazar el párrafo "Commands:" del docstring (jax-platform)**

En `/home/fruiz/worktrees/jax-platform-frente-e/backend/jax_engine/owner_cleanup.py`, reemplazar desde
`Commands: reaped as soon as BOTH ...` hasta `... ever cleaned.` (L5-11) por el texto elegido.

Texto A:
```
Commands: an owner file is reaped as soon as BOTH its mission and result
files are gone, or once it is older than COMMAND_OWNER_MAX_AGE_SECONDS.
Retention of the mission and result files themselves (~/jax/missions) is this
reaper's job too, driven by the `web_task_retention_days` setting. The old
external script (~/jax/scripts/cleanup.sh) never had a scheduler and was
retired on 2026-09-16 (jax, audit finding E-07).
```

Texto B:
```
Commands: an owner file is reaped as soon as BOTH its mission and result
files are gone, or once it is older than COMMAND_OWNER_MAX_AGE_SECONDS.
NOTHING retains the mission and result files themselves (~/jax/missions):
the old external script (~/jax/scripts/cleanup.sh) never had a scheduler and
was retired on 2026-09-16 (jax, audit finding E-07). Reported to Fernando the
same day; until a retention exists, those .md files grow without a ceiling.
```

Run:
```bash
git -C /home/fruiz/worktrees/jax-platform-frente-e grep -n "cleanup.sh" -- backend
cd /home/fruiz/worktrees/jax-platform-frente-e/backend && /home/fruiz/jax-platform/backend/.venv/bin/python -m py_compile jax_engine/owner_cleanup.py
```
Esperado: una sola línea, la del texto nuevo, que nombra el script retirado. `py_compile` sin salida.

- [ ] **Paso 5: commits (uno por repo)**

```bash
cd /home/fruiz/worktrees/jax-frente-e && git add -A scripts/cleanup.sh tests/test_frente_e_retiros.py
git commit -m "limpieza(E-07): retirar scripts/cleanup.sh (sin scheduler)" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
cd /home/fruiz/worktrees/jax-platform-frente-e && git add backend/jax_engine/owner_cleanup.py
git commit -m "docs(E-07): owner_cleanup ya no nombra un script que no corre y dice quien retiene ~/jax/missions" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: E-08 · `[motors.*]` fuera de `las_manos/config.toml`; el vocabulario cita la tabla `motor`

**Archivos:**
- Modificar: `las_manos/config.toml:127-178` (encabezado MOTOR REGISTRY + `[motors.kimi]` + `[motors.ada]`)
- Modificar: `policy/vocabulary/closed_vocabulary.yaml:67-76` (bloque `motors:`)
- Modificar: `tests/test_frente_e_retiros.py` (se agregan 2 tests)

**Interfaces:**
- Consume: `RAIZ` (Tarea 2).
- Produce: nada.

- [ ] **Paso 1: tests que fallan**

Agregar a `tests/test_frente_e_retiros.py`:

```python
def test_E08_config_toml_de_las_manos_ya_no_trae_motores():
    import tomllib
    with open(RAIZ / "las_manos" / "config.toml", "rb") as fh:
        assert "motors" not in tomllib.load(fh)


def test_E08_el_vocabulario_cita_la_tabla_motor_como_fuente():
    texto = (RAIZ / "policy" / "vocabulary" / "closed_vocabulary.yaml").read_text(encoding="utf-8")
    bloque = texto.split("\nmotors:\n", 1)[1].split("\nconfig_paths:\n", 1)[0]
    assert "tabla `motor`" in bloque
    assert "config.toml" not in bloque
```

Run: `$PYTEST_JAX -v tests/test_frente_e_retiros.py -k E08`. Esperado: 2 FAIL.

- [ ] **Paso 2: borrar las secciones**

En `las_manos/config.toml`, borrar desde la línea `# ============================================================` que abre
`#  MOTOR REGISTRY — motores de IA especializados` hasta `max_tokens = 8000` de `[motors.ada]`, inclusive. En su lugar va:

```toml
# ------------------------------------------------------------
#  MOTORES -- viven en la DB (R4). Fuente única: tabla `motor` de
#  jax_memory, leída por MotorCatalog.from_db(). Las secciones
#  [motors.kimi]/[motors.ada] que estaban acá dejaron de leerse con R4
#  y se retiraron el 2026-09-16 (E-08); su historia queda en git.
# ------------------------------------------------------------
```

- [ ] **Paso 3: corregir la fuente declarada**

En `policy/vocabulary/closed_vocabulary.yaml`, reemplazar el comentario bajo `motors:` (las 7 líneas que empiezan con
`# Fuente: las_manos/config.toml, secciones [motors.*].`) por:

```yaml
  # Fuente: tabla `motor` de jax_memory (MotorCatalog.from_db(), R4) --
  # la fila de cada motor y su modelo (jax/core/model_catalog.py, symlink
  # en las_manos/). Lista parcial: solo los motores que este vocabulario
  # necesita nombrar. Hasta 2026-09-16 citaba las secciones [motors.*] de
  # las_manos/config.toml, que ya no se leían (E-08).
```

- [ ] **Paso 4: verde, gobernanza incluida**

Run:
```bash
$PYTEST_JAX -v tests/test_frente_e_retiros.py
cd /home/fruiz/worktrees/jax-frente-e && env -u JAX_DB_HOST JAX_DB_NAME=jax_memory_test /home/fruiz/jax/.venv/bin/python -m pytest -q tests/test_governance_claims.py tests/test_governance_loaders.py tests/test_governance_validator.py tests/test_governance_vocab_sweep.py tests/test_governance_renderer.py tests/test_governance_grounding.py 2>&1 | tail -2
```
Esperado: 9 PASS, y gobernanza `90 passed`, el mismo piso de su job.

- [ ] **Paso 5: commit**

```bash
cd /home/fruiz/worktrees/jax-frente-e && git add las_manos/config.toml policy/vocabulary/closed_vocabulary.yaml tests/test_frente_e_retiros.py
git commit -m "limpieza(E-08): [motors.*] fuera de config.toml; closed_vocabulary cita la tabla motor" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: E-13 · `MAX_STEPS_PER_PIPELINE` es una sola constante

**Archivos:**
- Crear: `tests/test_max_steps_una_sola_constante.py`
- Modificar: `jacobs/models.py` (constante nueva y defaults de `Pipeline.max_steps` y `PipelineCreateRequest.max_steps`; validador L129-130)
- Modificar: `jacobs/policy.py:12,16` (import y borrar la definición)
- Modificar: `jacobs/routes.py:21-29,93,101-105` (import, default de `PlanRequest`, chequeo)
- Modificar: `jacobs/plan.py:22,437` (import y default de `build`)
- Modificar: `.github/workflows/policy.yml` (las dos listas de `tests-puros`)

**Interfaces:**
- Consume: nada.
- Produce: `jacobs.models.MAX_STEPS_PER_PIPELINE: int = 20`, importada por `jacobs.policy`, `jacobs.routes` y `jacobs.plan`. `jacobs.policy.MAX_STEPS_PER_PIPELINE` sigue existiendo como nombre importado.

- [ ] **Paso 1: tests que fallan**

```python
# tests/test_max_steps_una_sola_constante.py
"""E-13 (2026-09-16): el tope de steps por pipeline es UNA constante.

Estaba repetido como literal 20 en jacobs/models.py (request y validador),
jacobs/routes.py (plan_only) y jacobs/policy.py. Cambiar la constante no movía
los otros dos. Vive en models.py porque policy.py ya importa models: al revés
sería un import circular (verificador, 2026-09-16).
"""
from __future__ import annotations

import ast
import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

os.environ["JAX_DB_NAME"] = "jax_memory_test"

RAIZ = Path(__file__).resolve().parents[1]
_ARCHIVOS = ("jacobs/models.py", "jacobs/routes.py", "jacobs/policy.py", "jacobs/plan.py")


def _nombre(nodo: ast.AST) -> str | None:
    if isinstance(nodo, ast.Name):
        return nodo.id
    if isinstance(nodo, ast.Attribute):
        return nodo.attr
    return None


def _literales_de_max_steps(fuente: str) -> list[int]:
    lineas = []
    for nodo in ast.walk(ast.parse(fuente)):
        if isinstance(nodo, ast.Compare):
            lados = [nodo.left, *nodo.comparators]
            if any(_nombre(l) == "max_steps" for l in lados) and any(
                    isinstance(l, ast.Constant) and isinstance(l.value, int) and l.value > 1 for l in lados):
                lineas.append(nodo.lineno)
        elif isinstance(nodo, ast.AnnAssign) and _nombre(nodo.target) == "max_steps" \
                and isinstance(nodo.value, ast.Constant):
            lineas.append(nodo.lineno)
        elif isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = nodo.args.args[len(nodo.args.args) - len(nodo.args.defaults):]
            for arg, default in zip(args, nodo.args.defaults):
                if arg.arg == "max_steps" and isinstance(default, ast.Constant):
                    lineas.append(default.lineno)
    return lineas


def test_la_constante_vive_en_models_y_policy_la_importa():
    from jacobs import models, policy
    assert models.MAX_STEPS_PER_PIPELINE == 20
    arbol = ast.parse((RAIZ / "jacobs" / "policy.py").read_text(encoding="utf-8"))
    definiciones = [n for n in arbol.body if isinstance(n, ast.Assign)
                    and any(_nombre(t) == "MAX_STEPS_PER_PIPELINE" for t in n.targets)]
    assert definiciones == [], "policy.py vuelve a definir su propio tope"
    assert policy.MAX_STEPS_PER_PIPELINE is models.MAX_STEPS_PER_PIPELINE


def test_ningun_literal_de_tope_para_max_steps():
    hallazgos = {rel: _literales_de_max_steps((RAIZ / rel).read_text(encoding="utf-8")) for rel in _ARCHIVOS}
    assert {rel: l for rel, l in hallazgos.items() if l} == {}


def test_el_request_lee_la_constante(monkeypatch):
    from jacobs import models
    monkeypatch.setattr(models, "MAX_STEPS_PER_PIPELINE", 5, raising=False)
    with pytest.raises(ValueError) as e:
        models.PipelineCreateRequest(name="n", objective="o", invoked_by="plataforma", mode="dry_run", max_steps=6)
    assert "entre 1 y 5" in str(e.value)


def test_plan_only_lee_la_constante(monkeypatch):
    from fastapi import HTTPException
    from jacobs import routes
    monkeypatch.setattr(routes, "MAX_STEPS_PER_PIPELINE", 5, raising=False)
    monkeypatch.setattr(routes, "check_kill_switch", lambda: False)
    monkeypatch.setattr(routes, "_build_plan_or_reject", AsyncMock(side_effect=AssertionError("no debía planificar")))
    req = routes.PlanRequest(name="n", objective="o", invoked_by="plataforma", mode="dry_run", max_steps=6)
    with pytest.raises(HTTPException) as e:
        asyncio.run(routes.plan_only(req))
    assert e.value.status_code == 422
    assert "(5)" in e.value.detail
```

Nota para quien implementa: si el frente B ya cambió la firma de `check_kill_switch` en `routes`, el
`monkeypatch.setattr(routes, "check_kill_switch", ...)` se adapta al nombre nuevo. Es el único punto de contacto.

- [ ] **Paso 2: ver el rojo**

Run: `$PYTEST_JAX -v tests/test_max_steps_una_sola_constante.py`
Esperado: 4 FAIL. Por qué falla cada uno:
- `AttributeError` en models.
- literales en models L101/115/129, routes L93/101 y plan L437.
- el validador con 20.
- `AssertionError("no debía planificar")` en vez de `HTTPException`.

- [ ] **Paso 3: implementar**

`jacobs/models.py`, después de `VALID_MODES = frozenset({...})`:
```python
# Tope duro de steps por pipeline (E-13, 2026-09-16). Vive acá y no en
# policy.py porque policy importa models: al revés sería un import circular.
# policy.py, routes.py, plan.py y el validador de abajo lo importan de acá.
MAX_STEPS_PER_PIPELINE = 20
```
En `Pipeline` y en `PipelineCreateRequest`: `max_steps: int = MAX_STEPS_PER_PIPELINE`, respetando la alineación de cada
clase. Validador:
```python
        if self.max_steps < 1 or self.max_steps > MAX_STEPS_PER_PIPELINE:
            raise ValueError(
                f"max_steps debe estar entre 1 y {MAX_STEPS_PER_PIPELINE} (límite duro v0.1)"
            )
```
Dentro del validador, la constante se busca por nombre de módulo en tiempo de ejecución. Por eso el test la puede parchear.

`jacobs/policy.py`: `from jacobs.models import INVOKER_PLATAFORMA, MAX_STEPS_PER_PIPELINE, VALID_INVOKERS`, y borrar la
línea `MAX_STEPS_PER_PIPELINE  = 20`.

`jacobs/routes.py`: agregar `MAX_STEPS_PER_PIPELINE,` al `from jacobs.models import (...)`. `PlanRequest.max_steps: int = MAX_STEPS_PER_PIPELINE`, y en `plan_only`:
```python
    if req.max_steps > MAX_STEPS_PER_PIPELINE:
        raise HTTPException(
            status_code=422,
            detail=f"max_steps={req.max_steps} excede límite duro ({MAX_STEPS_PER_PIPELINE})",
        )
```

`jacobs/plan.py` L22: `from jacobs.models import MAX_STEPS_PER_PIPELINE, MOTOR_FACETS, Step`. En `build`: `max_steps: int = MAX_STEPS_PER_PIPELINE,`.

- [ ] **Paso 4: verde**

Run:
```bash
$PYTEST_JAX -v tests/test_max_steps_una_sola_constante.py
$PYTEST_JAX -q tests/test_jacobs_invoked_by_rol.py jacobs/_plan_timeout_ceiling_test.py 2>&1 | tail -2
```
Esperado: 4 PASS, y el resto verde (`_plan_timeout_ceiling_test.py` sigue en 16).

- [ ] **Paso 5: CI y commit**

Agregar `tests/test_max_steps_una_sola_constante.py` a las dos listas de `tests-puros`, igual que en la Tarea 2, Paso 5.

```bash
cd /home/fruiz/worktrees/jax-frente-e && git add jacobs/models.py jacobs/policy.py jacobs/routes.py jacobs/plan.py tests/test_max_steps_una_sola_constante.py .github/workflows/policy.yml
git commit -m "fix(E-13): MAX_STEPS_PER_PIPELINE una sola constante en jacobs/models.py" -m "4 tests vistos en rojo." -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: E-03, E-17, E-23 · las facetas del planner salen de la tabla `facet`; una desconocida rechaza el plan

**Archivos:**
- Crear: `tests/test_plan_facetas_de_la_tabla.py`
- Crear: `tests/test_facetas_de_gobernanza_db.py`
- Modificar: `jacobs/store.py:488-568` (`get_motor_governance` devuelve `facets`)
- Modificar: `jacobs/plan.py:81-94` (borrar `VALID_FACETS` y ajustar el comentario siguiente), `build()` (`:433-463`), `_from_spec` (`:489`), `_parse_plan_json` (`:797-800`)
- Modificar: `jacobs/models.py:47-49` (borrar `VALID_FACETS`)
- Modificar: `jacobs/executor.py:420` (docstring de `_invoke_ollama`)
- Modificar: `jacobs/_plan_timeout_ceiling_test.py:48-56` (la gobernanza fabricada suma `facets`)
- Modificar: `.github/workflows/policy.yml` (listas de `tests-puros` y de `jacobs-gobernanza-db`)

**Interfaces:**
- Consume: el gate de la Tarea 0, Paso 3 (todas las facetas `active`).
- Produce:
  - `store.get_motor_governance() -> {"capabilities": ..., "motors": ..., "facets": frozenset[str]}`.
  - `plan._check_facets(steps: list[Step], facetas_activas: frozenset[str]) -> list[PlanViolation]`.
  - El motivo del rechazo contiene "tabla `facet`".

- [ ] **Paso 1: tests puros que fallan**

```python
# tests/test_plan_facetas_de_la_tabla.py
"""E-03 / E-17 / E-23 (2026-09-16): las facetas del planner salen de la tabla
`facet`, y una que no está activa RECHAZA el plan.

Antes jacobs/plan.py:799 reemplazaba una faceta desconocida del LLM por
jax_local sin log ni evento, contra una lista fija (VALID_FACETS, duplicada en
models.py). El plan corría con una faceta que nadie pidió. Ahora el rechazo es
PlanRejected -> 422 + PLAN_REJECTED en jacobs_events, por los dos caminos
(spec y LLM), porque la validación vive en build().

La gobernanza se parchea: se prueba la POLÍTICA. La lectura real de la tabla
la prueba tests/test_facetas_de_gobernanza_db.py contra MariaDB.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock

import pytest

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from jacobs import models, routes  # noqa: E402
from jacobs import plan as plan_mod  # noqa: E402

GOBERNANZA = {
    "capabilities": {
        "research": {"allowed_motors": [], "max_execution_minutes": 5},
        "analysis": {"allowed_motors": ["kimi"], "max_execution_minutes": 5},
    },
    "motors": {"kimi": True, "jax_local": True},
    "facets": frozenset({"hipatia", "jekyll", "kimi", "jax_local"}),
}


@pytest.fixture(autouse=True)
def gobernanza(monkeypatch):
    from jacobs import store
    monkeypatch.setattr(store, "get_motor_governance", AsyncMock(return_value=GOBERNANZA))


def _build(steps_spec=None, llm=None):
    async def correr():
        b = plan_mod.PlanBuilder()
        if llm is not None:
            b._llm_plan = llm
            b._ada_plan = llm
        return await b.build(pipeline_id="p-facetas", objective="algo trivial", max_steps=3, steps_spec=steps_spec)
    return asyncio.run(correr())


def test_valid_facets_ya_no_existe_en_ningun_modulo():
    assert not hasattr(plan_mod, "VALID_FACETS")
    assert not hasattr(models, "VALID_FACETS")


def test_parse_plan_json_no_cambia_una_faceta_desconocida():
    specs = asyncio.run(plan_mod.PlanBuilder._parse_plan_json(
        '[{"facet": "inventada", "capability": "research", "prompt": "x"}]', 3))
    assert specs[0]["facet"] == "inventada"


def test_build_rechaza_una_faceta_del_spec_que_no_esta_en_la_tabla():
    with pytest.raises(plan_mod.PlanRejected) as e:
        _build(steps_spec=[{"facet": "inventada", "capability": "research", "prompt": "x"}])
    [violacion] = e.value.violations
    assert violacion.facet == "inventada"
    assert "tabla `facet`" in violacion.reason


def test_build_rechaza_la_faceta_inventada_por_el_llm():
    async def llm(objective, max_steps, capability_hint):
        return await plan_mod.PlanBuilder._parse_plan_json(
            '[{"facet": "inventada", "capability": "research", "prompt": "x"}]', max_steps)
    with pytest.raises(plan_mod.PlanRejected) as e:
        _build(llm=llm)
    assert [v.facet for v in e.value.violations] == ["inventada"]


def test_build_acepta_las_facetas_activas_de_la_tabla():
    steps = _build(steps_spec=[
        {"facet": "hipatia", "capability": "research", "prompt": "x"},
        {"facet": "jekyll", "capability": "research", "prompt": "y", "depends_on": [0]},
    ])
    assert [s.facet for s in steps] == ["hipatia", "jekyll"]


def test_un_spec_sin_faceta_se_rechaza_en_vez_de_caer_a_jax_local():
    with pytest.raises(plan_mod.PlanRejected) as e:
        _build(steps_spec=[{"capability": "research", "prompt": "x"}])
    assert e.value.violations[0].facet == ""


def test_el_rechazo_sale_como_422_con_evento_PLAN_REJECTED(monkeypatch):
    from fastapi import HTTPException
    from jacobs import store
    evento = AsyncMock()
    monkeypatch.setattr(store, "event_append", evento)
    with pytest.raises(HTTPException) as e:
        asyncio.run(routes._build_plan_or_reject(
            "p-422", "o", 3, [{"facet": "inventada", "capability": "research", "prompt": "x"}]))
    assert e.value.status_code == 422
    pipeline_id, tipo, payload = evento.await_args.args
    assert (pipeline_id, tipo) == ("p-422", "PLAN_REJECTED")
    assert payload["violations"][0]["facet"] == "inventada"
```

- [ ] **Paso 2: ver el rojo**

Run: `$PYTEST_JAX -v tests/test_plan_facetas_de_la_tabla.py`
Esperado: 6 FAIL y 1 PASS. El PASS es `test_build_acepta_las_facetas_activas_de_la_tabla`, que es el control.
- `test_un_spec_sin_faceta...` falla porque la violación viene con `facet == "jax_local"` (rechazo por `capability_motor`, no por la faceta).
- El test de 422 falla porque el plan se acepta.

- [ ] **Paso 3: test de DB que falla (corre en `jacobs-gobernanza-db`)**

```python
# tests/test_facetas_de_gobernanza_db.py
"""E-03 (2026-09-16): get_motor_governance() trae las facetas ACTIVAS de la
tabla `facet`, la misma foto que capabilities y motors (una sola consulta por
build). Contra MariaDB real: el esquema lo crean las migraciones de
jax-platform en el job jacobs-gobernanza-db.

`facet` es un catálogo (7 filas en producción, 2026-09-16): la consulta por
status sin índice es un recorrido de 7 filas, declarado en DEUDA.md.
"""
from __future__ import annotations

import os
import unittest

_db = os.environ.get("JAX_DB_NAME", "")
if _db != "jax_memory_test":
    raise RuntimeError(f"JAX_DB_NAME={_db!r}: este test escribe una fila de `facet`; solo corre contra jax_memory_test.")

from jacobs import store  # noqa: E402

_CLAVE = "zz_test_facet_deshabilitada"


class FacetasDeGobernanzaDBTest(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM facet WHERE `key`=%s", (_CLAVE,))
            await conn.commit()
        finally:
            conn.close()

    async def test_las_facetas_son_las_activas_de_la_tabla(self):
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await cur.execute("SELECT `key` FROM facet WHERE status = 'active'")
                esperadas = frozenset(k for (k,) in await cur.fetchall())
        finally:
            conn.close()
        gobernanza = await store.get_motor_governance()
        self.assertEqual(gobernanza["facets"], esperadas)
        self.assertIn("hipatia", gobernanza["facets"])

    async def test_una_faceta_deshabilitada_no_entra(self):
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO facet (`key`, display_name, transport, status) VALUES (%s, 'test', 'ollama', 'disabled')",
                    (_CLAVE,))
            await conn.commit()
        finally:
            conn.close()
        gobernanza = await store.get_motor_governance()
        self.assertNotIn(_CLAVE, gobernanza["facets"])
```

Run local:
```bash
cd /home/fruiz/worktrees/jax-frente-e && set -a && source <(grep -E '^JAX_DB_(HOST|PORT|USER|PASSWORD)=' /etc/jax/.env) && set +a && \
  JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos /home/fruiz/jax/.venv/bin/python -m pytest -v tests/test_facetas_de_gobernanza_db.py
```
Esperado: 2 FAIL con `KeyError: 'facets'`. Si falla con `1146 Table 'jax_memory_test.facet' doesn't exist`, **no crear la
tabla a mano en la instancia de producción**: anotarlo, y el rojo y el verde se leen en el job de CI (Tarea 16).

- [ ] **Paso 4: implementar**

`jacobs/store.py`, dentro de `get_motor_governance`, después del bucle de `capability_motor` y antes del `finally`:
```python
            # E-03 (2026-09-16): el vocabulario de facetas del planner es la
            # tabla `facet`, no una lista fija. Mismas reglas que resolve_facet
            # (status='active'): lo que el planner acepta es lo que se puede
            # despachar. Catálogo de 7 filas: sin índice, declarado en DEUDA.md.
            await cur.execute("SELECT `key` FROM facet WHERE status = 'active'")
            facets = frozenset(key for (key,) in await cur.fetchall())
```
Return: `return {"capabilities": capabilities, "motors": motors, "facets": facets}`. En el docstring, agregar a
"Devuelve:" la línea `"facets": frozenset de facet.key con status='active'}` y "4 SELECTs" donde decía "3 SELECTs".

`jacobs/plan.py`:
- Borrar el bloque `VALID_FACETS = frozenset({...})` (L81-83).
- En el comentario siguiente, reemplazar `# Espejo de VALID_FACETS para capabilities (FASE A §3.3). Vocabulario del` por
  `# Capabilities (FASE A §3.3), igual que las facetas desde E-03: vocabulario del`.
- Agregar después de `_check_cleanroom`:
```python
def _check_facets(steps: list, facetas_activas: frozenset) -> list[PlanViolation]:
    """E-17 (2026-09-16): una faceta que no está activa en la tabla `facet`
    rechaza el plan. Antes `_parse_plan_json` la reemplazaba por jax_local en
    silencio y el plan corría con una faceta que nadie pidió. Corre en build(),
    así que vale para los dos caminos (spec y LLM) y para el plan de respaldo."""
    return [
        PlanViolation(
            s.step_index, s.facet, s.motor, s.capability,
            f"faceta '{s.facet}' no existe o no está activa en la tabla `facet`",
        )
        for s in steps
        if s.facet not in facetas_activas
    ]
```
- En `build()`, justo antes de `cleanroom_violations = _check_cleanroom(steps)`:
```python
        facet_violations = _check_facets(steps, governance["facets"])
        if facet_violations:
            raise PlanRejected(facet_violations)
```
- En `_from_spec`: `facet=spec.get("facet", ""),`. Una spec sin faceta la rechaza el gate; ya no cae a jax_local.
- En `_parse_plan_json`, reemplazar
```python
            facet = item.get("facet", "")
            if facet not in VALID_FACETS:
                facet = "jax_local"
```
por
```python
            # E-17: la faceta viaja TAL CUAL. build() la valida contra la tabla
            # `facet` y rechaza el plan si no está activa: antes se cambiaba por
            # jax_local sin rastro.
            facet = str(item.get("facet", ""))[:50]
```
  y en el comentario de capability, `(espejo de la mecánica facet→jax_local de arriba)` pasa a `(las facetas, en cambio, se rechazan en build())`.

`jacobs/models.py`: borrar `VALID_FACETS = frozenset({...})` (L47-49).

`jacobs/executor.py:420`: `f.base_url + "/api/chat"). jax_local SI esta en VALID_FACETS (plan.py),` pasa a
`f.base_url + "/api/chat"). jax_local es una faceta activa de la tabla facet (E-03),`.

`jacobs/_plan_timeout_ceiling_test.py`: en `GOBERNANZA` agregar `"facets": frozenset({"hipatia", "jekyll", "kimi", "jax_local"}),`.

- [ ] **Paso 5: verde**

Run:
```bash
$PYTEST_JAX -v tests/test_plan_facetas_de_la_tabla.py
$PYTEST_JAX -q jacobs/_plan_timeout_ceiling_test.py tests/test_contrato_dispatch_repl_ada.py tests/test_jacobs_invoked_by_rol.py tests/test_jacobs_timeout_by_capability.py 2>&1 | tail -2
git -C /home/fruiz/worktrees/jax-frente-e grep -n "VALID_FACETS" -- '*.py'
```
Luego el comando DB del Paso 3. Esperado:
- 7 PASS, y `_plan_timeout_ceiling_test.py` sigue en `16 passed`.
- El grep no devuelve nada.
- DB: 2 PASS, o el 1146 anotado.

- [ ] **Paso 6: CI y commit**

- `tests/test_plan_facetas_de_la_tabla.py` va a las dos listas de `tests-puros`.
- `tests/test_facetas_de_gobernanza_db.py` va a las dos listas de `jacobs-gobernanza-db`: después de `jacobs/_direct_usage_test.py`, en el `pytest -v` y en el contador (`jacobs/_direct_usage_test.py \` seguido de `tests/test_facetas_de_gobernanza_db.py \`).

```bash
cd /home/fruiz/worktrees/jax-frente-e && git add jacobs/store.py jacobs/plan.py jacobs/models.py jacobs/executor.py jacobs/_plan_timeout_ceiling_test.py \
  tests/test_plan_facetas_de_la_tabla.py tests/test_facetas_de_gobernanza_db.py .github/workflows/policy.yml
git commit -m "fix(E-03,E-17,E-23): facetas del planner desde la tabla facet; una faceta desconocida rechaza el plan (422 + PLAN_REJECTED)" \
  -m "Antes plan.py:799 la cambiaba por jax_local en silencio. 6 tests puros y 2 de DB vistos en rojo." \
  -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: E-10, E-11 · `crypto_secrets`, `credential_resolver` y `model_catalog` de `las_manos/` pasan a symlinks

**Archivos:**
- Crear: `tests/test_espejos_symlink_frente_e.py`
- Reemplazar por symlink: `las_manos/crypto_secrets.py`, `las_manos/credential_resolver.py`, `las_manos/model_catalog.py`
- Modificar: `jax/core/credential_resolver.py:1-18` (docstring y doble import), `jax/core/model_catalog.py:1-22` (docstring y doble import)
- Modificar: `scripts/check_mirror_sync.py:187-235` (comentarios y `nota` de dos familias)
- Modificar: `scripts/_check_mirror_sync_test.py:180-182` (docstring)
- Modificar: `jax/core/db_connect_config.py:31-32` (docstring)
- Modificar: `.github/workflows/policy.yml:346-349` (comentario) y las dos listas de `tests-puros`
- Modificar (jax-platform): `backend/credential_resolver.py:4` (docstring)

**Interfaces:**
- Consume: nada.
- Produce: canónicos en `jax/core/`, que importan bare primero y caen a `jax.core.*`.

- [ ] **Paso 1: tests que fallan**

```python
# tests/test_espejos_symlink_frente_e.py
"""E-10 / E-11 (2026-09-16): dentro de jax, cada módulo espejado es UN archivo.

las_manos/crypto_secrets.py, credential_resolver.py y model_catalog.py eran
copias reales de jax/core (crypto byte a byte; las otras dos distintas solo en
los imports), así que podían driftear DENTRO del repo, y model_catalog ni
siquiera estaba en FAMILIAS. Pasan a symlinks, como facet_resolver y redaccion.

El canónico importa BARE primero (LAS MANOS y Jacobs: WorkingDirectory o
PYTHONPATH con las_manos/, donde jax.core NO es importable) y cae a jax.core
(REPL y workers). En un contexto `.:las_manos` (CI) gana el bare:
jax.core.credential_resolver usa los módulos bare. Inocuo: ningún test parcha
jax.core.crypto_secrets.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("archivo", ["crypto_secrets.py", "credential_resolver.py", "model_catalog.py"])
def test_la_copia_de_las_manos_es_symlink_al_canonico(archivo):
    link = RAIZ / "las_manos" / archivo
    assert link.is_symlink(), f"las_manos/{archivo} debe ser symlink"
    assert link.resolve() == (RAIZ / "jax" / "core" / archivo).resolve()


@pytest.mark.parametrize("archivo, modulos", [
    ("credential_resolver.py", {"crypto_secrets", "db_connect_config"}),
    ("model_catalog.py", {"db_connect_config"}),
])
def test_el_canonico_importa_bare_primero_y_cae_a_jax_core(archivo, modulos):
    arbol = ast.parse((RAIZ / "jax" / "core" / archivo).read_text(encoding="utf-8"))
    vistos = set()
    for nodo in arbol.body:
        if not isinstance(nodo, ast.Try):
            continue
        bare = {n.module for n in nodo.body if isinstance(n, ast.ImportFrom)}
        for handler in nodo.handlers:
            if isinstance(handler.type, ast.Name) and handler.type.id == "ImportError":
                calificados = {n.module for n in handler.body if isinstance(n, ast.ImportFrom)}
                vistos |= {m for m in bare if f"jax.core.{m}" in calificados}
    assert modulos <= vistos


def _python(codigo: str, pythonpath: Path, cwd: Path) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pythonpath)
    r = subprocess.run([sys.executable, "-c", codigo], cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_en_el_repl_el_canonico_usa_jax_core():
    salida = _python(
        "import jax.core.credential_resolver as c, jax.core.model_catalog as m;"
        "print(c.decrypt_secret.__module__, c.db_connect_timeout_seconds.__module__, m.db_connect_timeout_seconds.__module__)",
        RAIZ, RAIZ)
    assert salida.split() == ["jax.core.crypto_secrets", "jax.core.db_connect_config", "jax.core.db_connect_config"]


def test_en_las_manos_el_bare_resuelve_al_archivo_canonico():
    salida = _python(
        "from pathlib import Path; import credential_resolver as c, model_catalog as m, crypto_secrets as s;"
        "print(Path(c.__file__).resolve(), Path(m.__file__).resolve(), Path(s.__file__).resolve(), c.decrypt_secret.__module__)",
        RAIZ / "las_manos", RAIZ / "las_manos")
    rutas = salida.split()
    assert rutas[:3] == [str((RAIZ / "jax" / "core" / n).resolve())
                         for n in ("credential_resolver.py", "model_catalog.py", "crypto_secrets.py")]
    assert rutas[3] == "crypto_secrets"


def test_las_notas_de_las_familias_dicen_symlink_y_no_tres_archivos():
    spec = importlib.util.spec_from_file_location("check_mirror_sync_e11", RAIZ / "scripts" / "check_mirror_sync.py")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    familias = {f.nombre: f for f in modulo.FAMILIAS}
    for nombre in ("crypto_secrets", "credential_resolver"):
        assert "TRES archivos reales" not in familias[nombre].nota
        assert "symlink" in familias[nombre].nota
```

Run: `$PYTEST_JAX -v tests/test_espejos_symlink_frente_e.py`
Esperado: 8 tests, 7 FAIL y 1 PASS.
- Fallan los 3 de symlink, los 2 de AST, el de las_manos (`__file__` sin resolver a jax/core) y el de las notas.
- Pasa el del REPL: es control, porque hoy el canónico ya usa jax.core.

- [ ] **Paso 2: doble import en los canónicos**

`jax/core/credential_resolver.py`, reemplazar el docstring del módulo y las líneas `from jax.core.crypto_secrets import decrypt_secret` / `from jax.core.db_connect_config import db_connect_timeout_seconds` por:

```python
"""
Resolver de credenciales de proveedor — Fase 1 (DB como fuente de verdad).

Dos archivos reales: este (jax/core, canónico dentro de jax) y la copia de
jax-platform (backend/credential_resolver.py). las_manos/credential_resolver.py
es SYMLINK a este desde 2026-09-16 (E-11); hasta entonces era una tercera copia
que podía driftear dentro de jax. La familia la vigila scripts/check_mirror_sync.py.

Diseño completo: jax-platform/docs/fase1-credenciales-diseno.md (B1.2/B1.4).
Resuelve R3 de la auditoria (rotar una key no la propagaba sin restart).
"""
import logging
import os
import time

import aiomysql

try:
    # LAS MANOS y Jacobs: este archivo corre como las_manos/credential_resolver.py
    # (symlink) con WorkingDirectory o PYTHONPATH en las_manos/, donde jax.core
    # NO es importable; crypto_secrets también es symlink ahí. En contextos
    # `.:las_manos` (CI) gana este camino también para jax.core.credential_resolver.
    from crypto_secrets import decrypt_secret
except ImportError:
    # REPL y workers de jax.memory (PYTHONPATH=. desde la raíz del repo).
    from jax.core.crypto_secrets import decrypt_secret

try:
    from db_connect_config import db_connect_timeout_seconds
except ImportError:
    from jax.core.db_connect_config import db_connect_timeout_seconds
```

`jax/core/model_catalog.py`:
- En el docstring, reemplazar `Espejo\nminimo en jax-platform, jax/core, las_manos, mismo patron que` por `Espejo\nminimo en jax-platform y jax/core (las_manos/model_catalog.py es symlink a este desde\n2026-09-16, E-11), mismo patron que`.
- Reemplazar `from jax.core.db_connect_config import db_connect_timeout_seconds` por el bloque `try/except ImportError` de `db_connect_config` de arriba.
- El comentario `# fail-soft:` se conserva tal como está en el canónico.

- [ ] **Paso 3: symlinks**

```bash
cd /home/fruiz/worktrees/jax-frente-e
for f in crypto_secrets credential_resolver model_catalog; do
  git rm -q "las_manos/$f.py" && ln -s "../jax/core/$f.py" "las_manos/$f.py" && git add "las_manos/$f.py"
done
ls -la las_manos | grep -E "crypto_secrets|credential_resolver|model_catalog"
```
Esperado: tres líneas `-> ../jax/core/<nombre>.py`.

- [ ] **Paso 4: notas y docstrings que decían "tres archivos reales"**

`scripts/check_mirror_sync.py`, familia `crypto_secrets`:
- El comentario de `espejos` (`# Tres archivos reales otra vez -- ninguno es symlink, medido el 2026-09-01. ...`) pasa a
```python
            # las_manos/crypto_secrets.py es SYMLINK a jax/core desde el
            # 2026-09-16 (E-10): comparar ahí es un no-op, a propósito, como
            # facet_resolver. Hasta ese día eran tres archivos reales.
```
- Nota:
```python
        nota="Dos archivos reales (jax/core y jax-platform) + symlink en las_manos/ "
             "desde 2026-09-16 (E-10). jax-platform tiene ademas encrypt_secret y "
             "decrypt_db_secret, excluidos por diseno (es el lado que cifra).",
```

Familia `credential_resolver`:
- El comentario `# OJO -- aca la forma NO es la de facet_resolver ...` hasta `... nadie la comparaba con nada.` pasa a
```python
            # las_manos/credential_resolver.py es SYMLINK a jax/core desde el
            # 2026-09-16 (E-11). Hasta ese día era un TERCER archivo real que
            # podía driftear dentro de jax. El test de no-fail-open escanea el
            # symlink por las dos rutas (no deduplica por resolve()): inocuo,
            # las dos muestran la misma marca.
```
- Nota:
```python
        nota="Dos archivos reales (jax/core y jax-platform) + symlink en las_manos/ "
             "desde 2026-09-16 (E-11). El canónico de jax importa bare primero y cae "
             "a jax.core; los ImportFrom no se comparan.",
```

`scripts/_check_mirror_sync_test.py`, docstring de `test_compara_TODOS_los_espejos_no_solo_el_primero`:
```python
    """credential_resolver tuvo TRES archivos reales hasta el 2026-09-16 (E-11).
    Un comparador que se quedara en el primer espejo dejaria al tercero sin
    vigilancia -- que es la situacion que habia -- y cualquier familia con
    varios espejos reales vuelve a tener ese riesgo."""
```

`jax/core/db_connect_config.py`: en el docstring, reemplazar `\`las_manos/credential_resolver.py\`,\n\`las_manos/model_catalog.py\`,` por `\`jax/core/credential_resolver.py\` y\n\`jax/core/model_catalog.py\` (symlinkeados en \`las_manos/\` desde 2026-09-16),`. Respetar el salto de línea real del archivo.

`.github/workflows/policy.yml`, comentario del job `mirror-sync`: reemplazar desde `# credential_resolver tiene TRES archivos reales, no dos:` hasta `# hasta hoy nadie la comparaba con nada.` por:
```
  # credential_resolver tuvo TRES archivos reales hasta el 2026-09-16: la copia
  # de las_manos/ no era symlink. Desde E-11 lo es (igual crypto_secrets, E-10,
  # y model_catalog), asi que dentro de jax queda un solo archivo por modulo.
```

jax-platform, `backend/credential_resolver.py:4`: `Espejo minimo en los 3 codebases (jax-platform, jax/core, las_manos), mismo` pasa a
`Espejo minimo en dos archivos reales (este y jax/core en jax; las_manos/ lo ve por symlink desde 2026-09-16), mismo`.

Comprobación, que se corre ya:
`git -C /home/fruiz/worktrees/jax-frente-e grep -n "las_manos/model_catalog.py" -- policy/vocabulary/closed_vocabulary.yaml`.
Esperado: nada, porque la Tarea 4 ya reescribió ese comentario.

- [ ] **Paso 5: verde, y el checker de espejos contra jax-platform**

Run:
```bash
$PYTEST_JAX -v tests/test_espejos_symlink_frente_e.py
$PYTEST_JAX -q tests/test_cola_uso_escritores.py las_manos/_worker_max_tokens_test.py las_manos/_worker_tool_loop_test.py tests/test_contrato_dispatch_repl_ada.py 2>&1 | tail -2
cd /home/fruiz/worktrees/jax-frente-e && JAX_PLATFORM_REPO_ROOT=/home/fruiz/worktrees/jax-platform-frente-e /home/fruiz/jax/.venv/bin/python scripts/check_mirror_sync.py; echo "exit=$?"
cd /home/fruiz/worktrees/jax-frente-e && /home/fruiz/jax/.venv/bin/python -m pytest -q scripts/_check_mirror_sync_test.py 2>&1 | tail -1
cd /home/fruiz/worktrees/jax-frente-e && PYTHONPATH=las_manos /home/fruiz/jax/.venv/bin/python -m pytest -q las_manos/_facet_resolver_seal_test.py 2>&1 | tail -1
```
Esperado:
- 8 PASS y el resto verde.
- `exit=0`.
- `14 passed` en `_check_mirror_sync_test.py`.
- `13 passed` en `_facet_resolver_seal_test.py`.

- [ ] **Paso 6: CI y commits**

`tests/test_espejos_symlink_frente_e.py` va a las dos listas de `tests-puros`.

```bash
cd /home/fruiz/worktrees/jax-frente-e && git add -A las_manos/crypto_secrets.py las_manos/credential_resolver.py las_manos/model_catalog.py \
  jax/core/credential_resolver.py jax/core/model_catalog.py jax/core/db_connect_config.py scripts/check_mirror_sync.py \
  scripts/_check_mirror_sync_test.py tests/test_espejos_symlink_frente_e.py .github/workflows/policy.yml
git commit -m "refactor(E-10,E-11): crypto_secrets, credential_resolver y model_catalog de las_manos/ son symlinks al canonico de jax/core" \
  -m "El canonico importa bare primero y cae a jax.core. 7 tests vistos en rojo; 1 control." \
  -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
cd /home/fruiz/worktrees/jax-platform-frente-e && git add backend/credential_resolver.py
git commit -m "docs(E-11): credential_resolver es espejo de dos archivos reales, no tres" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: E-21 · URLs de servicio desde el entorno, validadas al arrancar; sin defaults de proveedor

**Archivos:**
- Crear: `jax/core/config_entorno.py` y el symlink `las_manos/config_entorno.py -> ../jax/core/config_entorno.py`
- Crear: `tests/test_config_entorno.py`
- Modificar: `conftest.py` (raíz jax: valores de test para `LAS_MANOS_URL` y `JAX_OLLAMA_URL`)
- Modificar: `jacobs/executor.py:44-45`, `jacobs/plan.py:46`
- Modificar: `jax/memory/db.py:476-497` (`get_embedding`)
- Modificar: `jax/muscles/ollama_muscle.py:62-84` (sin `DEFAULT_OLLAMA_URL`, `api_url` obligatorio)
- Modificar: `jax/muscles/base.py:243,284,378` (sin defaults de proveedor)
- Modificar: `jax/voice/tts.py:35,124` (`JAX_KOKORO_PYTHON` perezoso)
- Modificar: `jax/core/main.py:69-108` (`build_muscles`) y el arranque de `main()` (validación temprana)
- Modificar: `config/config.toml:35,284,334` (quitar `api_url`)
- Modificar: `tests/test_contrato_dispatch_repl_ada.py:55-60,711-722`, `tests/test_gemini_key_en_cabecera.py:179-183`, `tests/test_repl_fuentes.py:73-77`
- Modificar (jax-platform): `backend/api/chat.py:704-705` (`_call_ollama`), `backend/tests/test_chat_http_pooling.py:26-42`, `backend/tests/test_chat_usage_capture.py:76`, `.github/workflows/policy.yml` (`env` de `backend-tests-con-db` y `backend-tests-no-db`)
- Modificar: `.github/workflows/policy.yml` de jax (las dos listas de `tests-puros`)

**Interfaces:**
- Consume: nada.
- Produce:
  - `config_entorno.EntornoInvalido(RuntimeError)`.
  - `config_entorno.url_requerida(nombre: str) -> str`, sin barra final.
  - `config_entorno.ruta_absoluta_requerida(nombre: str) -> Path`.
  - Variables: `LAS_MANOS_URL` (ya existe en `/etc/jax/.env`), `JAX_OLLAMA_URL` (base, sin `/api/...`) y `JAX_KOKORO_PYTHON`.
  - `jacobs.executor.LAS_MANOS_BASE` y `.OLLAMA_URL`, `jacobs.plan.OLLAMA_URL`, `jax.voice.tts._python_de_kokoro() -> Path`.
  - `HttpMuscle._url_del_catalogo() -> str`.

- [ ] **Paso 1: tests que fallan**

```python
# tests/test_config_entorno.py
"""E-21 (2026-09-16): las URLs de servicio salen del entorno (/etc/jax/.env),
validadas al arrancar; las de proveedor salen del catálogo, sin default.

Antes: LAS_MANOS_BASE y OLLAMA_URL fijos en jacobs/executor.py y plan.py, el
embed de la memoria a http://localhost:11434 fijo, DEFAULT_OLLAMA_URL en el
músculo local, api_url en config.toml, y URLs de OpenAI/DeepSeek/Gemini como
default en jax/muscles/base.py (con la DB caída se despachaba a una URL que
nadie eligió).
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

os.environ["JAX_DB_NAME"] = "jax_memory_test"

RAIZ = Path(__file__).resolve().parents[1]


def test_url_que_falta_es_error_que_nombra_la_variable(monkeypatch):
    from jax.core.config_entorno import EntornoInvalido, url_requerida
    monkeypatch.delenv("JAX_URL_DE_PRUEBA", raising=False)
    with pytest.raises(EntornoInvalido) as e:
        url_requerida("JAX_URL_DE_PRUEBA")
    assert "JAX_URL_DE_PRUEBA" in str(e.value) and "/etc/jax/.env" in str(e.value)


def test_url_sin_esquema_o_sin_host_es_error(monkeypatch):
    from jax.core.config_entorno import EntornoInvalido, url_requerida
    for valor in ("localhost:11434", "ftp://x", "http://"):
        monkeypatch.setenv("JAX_URL_DE_PRUEBA", valor)
        with pytest.raises(EntornoInvalido):
            url_requerida("JAX_URL_DE_PRUEBA")


def test_url_valida_vuelve_sin_barra_final(monkeypatch):
    from jax.core.config_entorno import url_requerida
    monkeypatch.setenv("JAX_URL_DE_PRUEBA", " https://servicio.test:8443/ ")
    assert url_requerida("JAX_URL_DE_PRUEBA") == "https://servicio.test:8443"


def test_ruta_relativa_es_error(monkeypatch):
    from jax.core.config_entorno import EntornoInvalido, ruta_absoluta_requerida
    monkeypatch.setenv("JAX_RUTA_DE_PRUEBA", "jax/repo")
    with pytest.raises(EntornoInvalido):
        ruta_absoluta_requerida("JAX_RUTA_DE_PRUEBA")


def test_ruta_absoluta_valida(monkeypatch):
    from jax.core.config_entorno import ruta_absoluta_requerida
    monkeypatch.setenv("JAX_RUTA_DE_PRUEBA", "/srv/algo")
    assert ruta_absoluta_requerida("JAX_RUTA_DE_PRUEBA") == Path("/srv/algo")


def _proceso_limpio(codigo: str, **entorno) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{RAIZ}{os.pathsep}{RAIZ / 'las_manos'}"
    for clave, valor in entorno.items():
        if valor is None:
            env.pop(clave, None)
        else:
            env[clave] = valor
    return subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ, env=env,
                          capture_output=True, text=True, timeout=120)


def test_jacobs_toma_las_urls_del_entorno():
    r = _proceso_limpio(
        "from jacobs import executor, plan; print(executor.LAS_MANOS_BASE, executor.OLLAMA_URL, plan.OLLAMA_URL)",
        LAS_MANOS_URL="http://las-manos.test:7777/", JAX_OLLAMA_URL="http://ollama.test:11434")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["http://las-manos.test:7777", "http://ollama.test:11434/api/chat",
                                "http://ollama.test:11434/api/chat"]


def test_jacobs_no_arranca_sin_LAS_MANOS_URL():
    r = _proceso_limpio("from jacobs import executor", LAS_MANOS_URL=None)
    assert r.returncode != 0
    assert "LAS_MANOS_URL" in r.stderr


def test_la_memoria_sin_JAX_OLLAMA_URL_falla_visible(monkeypatch):
    from jax.core.config_entorno import EntornoInvalido
    from jax.memory.db import MemoryDB
    monkeypatch.delenv("JAX_OLLAMA_URL", raising=False)
    with pytest.raises(EntornoInvalido):
        asyncio.run(MemoryDB().get_embedding("hola"))


def test_la_memoria_pide_el_embedding_a_JAX_OLLAMA_URL(monkeypatch):
    from jax.memory import db as dbmod
    monkeypatch.setenv("JAX_OLLAMA_URL", "http://ollama.test:11434/")
    urls = []

    async def handle(transport_self, request):
        urls.append(str(request.url))
        return httpx.Response(200, json={"embeddings": [[0.0] * dbmod.EMBED.dim]})

    async def correr():
        memoria = dbmod.MemoryDB()
        with patch("httpx.AsyncHTTPTransport.handle_async_request", handle):
            vector = await memoria.get_embedding("hola")
        await memoria.close()
        return vector

    assert len(asyncio.run(correr())) == dbmod.EMBED.dim
    assert urls == ["http://ollama.test:11434/api/embed"]


def test_el_musculo_local_no_tiene_url_por_defecto_y_el_repl_la_toma_del_entorno(monkeypatch):
    from jax.core.main import build_muscles
    from jax.muscles.ollama_muscle import OllamaMuscle
    assert inspect.signature(OllamaMuscle.__init__).parameters["api_url"].default is inspect.Parameter.empty
    monkeypatch.setenv("JAX_OLLAMA_URL", "http://ollama.test:11434")
    cfg = {"jax": {"timeout_seconds": 10}, "personalities": {"jax_local": {
        "type": "ollama", "provider": "ollama", "model_default": "q", "models_allowed": ["q"], "system_prompt": "s"}}}
    assert build_muscles(cfg)["jax_local"].api_url == "http://ollama.test:11434/api/chat"


@pytest.mark.parametrize("proveedor, metodo", [
    ("deepseek", "_call_deepseek"), ("openai", "_call_openai"), ("gemini", "_call_gemini"),
])
def test_sin_url_del_catalogo_el_musculo_http_no_despacha(proveedor, metodo):
    from jax.muscles import base
    musculo = base.HttpMuscle(name="f", provider=proveedor, model_default="x", models_allowed=["x"],
                              system_prompt="s", timeout=10)
    red = AsyncMock(side_effect=AssertionError("salió a la red sin URL del catálogo"))

    async def correr():
        with patch("httpx.AsyncHTTPTransport.handle_async_request", red), \
             patch.object(base.HttpMuscle, "_resolve_api_key", AsyncMock(return_value="k")), \
             patch.object(base.HttpMuscle, "_limite_de_salida", AsyncMock(return_value={})):
            await getattr(musculo, metodo)("hola", "x")

    with pytest.raises(base.MuscleInvocationError) as e:
        asyncio.run(correr())
    assert "catálogo" in str(e.value)


def test_la_voz_toma_el_python_de_kokoro_del_entorno(monkeypatch):
    from jax.core.config_entorno import EntornoInvalido
    from jax.voice import tts
    assert not hasattr(tts, "KOKORO_PYTHON")
    monkeypatch.delenv("JAX_KOKORO_PYTHON", raising=False)
    with pytest.raises(EntornoInvalido):
        tts._python_de_kokoro()
    monkeypatch.setenv("JAX_KOKORO_PYTHON", "/opt/voz/bin/python")
    assert tts._python_de_kokoro() == Path("/opt/voz/bin/python")


_SIN_URLS_LITERALES = ("jacobs/executor.py", "jacobs/plan.py", "jax/memory/db.py",
                       "jax/muscles/base.py", "jax/muscles/ollama_muscle.py", "jax/voice/tts.py")


def _urls_literales(fuente: str) -> list[int]:
    arbol = ast.parse(fuente)
    docstrings = {id(n.value) for n in ast.walk(arbol)
                  if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
    return [n.lineno for n in ast.walk(arbol)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
            and re.search(r"\bhttps?://", n.value)]


def test_ningun_modulo_de_servicio_tiene_una_url_escrita():
    hallazgos = {rel: _urls_literales((RAIZ / rel).read_text(encoding="utf-8")) for rel in _SIN_URLS_LITERALES}
    assert {rel: l for rel, l in hallazgos.items() if l} == {}


def test_config_toml_de_jax_no_trae_urls():
    with open(RAIZ / "config" / "config.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    assert [clave for clave, p in cfg["personalities"].items() if "api_url" in p] == []
```

Run: `$PYTEST_JAX -v tests/test_config_entorno.py`
Esperado: 16 tests, todos FAIL. Los cinco primeros por `ModuleNotFoundError: jax.core.config_entorno`; el resto por las URLs fijas y los defaults.

- [ ] **Paso 2: el módulo y su symlink**

```python
# jax/core/config_entorno.py
"""Configuración de servicio que sale del entorno (/etc/jax/.env), validada.

E-21 / E-22 (2026-09-16): URLs y rutas de servicio (LAS MANOS, Ollama, el venv
de la voz, el directorio de documentos) vivían escritas en el código. Salen de
variables de entorno. Sin la variable, o con un valor que no sirve, el proceso
NO arranca con un default silencioso: levanta EntornoInvalido nombrando la
variable. Un servicio apuntado a un host equivocado no se ve hasta que falla
delante de alguien; uno que no arranca se ve en el journal.

Un solo archivo real. las_manos/config_entorno.py es symlink (Jacobs y LAS
MANOS lo importan bare); el REPL, los workers y jax-platform (vía
jax.memory.db) lo importan como jax.core.config_entorno.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit


class EntornoInvalido(RuntimeError):
    """Falta una variable de entorno de servicio o su valor no sirve."""


def _valor(nombre: str) -> str:
    valor = os.environ.get(nombre, "").strip()
    if not valor:
        raise EntornoInvalido(
            f"{nombre} no está seteada: agregala a /etc/jax/.env (sin default silencioso)."
        )
    return valor


def url_requerida(nombre: str) -> str:
    """URL http(s) con host, sin barra final."""
    valor = _valor(nombre)
    partes = urlsplit(valor)
    if partes.scheme not in ("http", "https") or not partes.hostname:
        raise EntornoInvalido(f"{nombre}={valor!r} no es una URL http(s) con host.")
    return valor.rstrip("/")


def ruta_absoluta_requerida(nombre: str) -> Path:
    valor = _valor(nombre)
    ruta = Path(valor)
    if not ruta.is_absolute():
        raise EntornoInvalido(f"{nombre}={valor!r} tiene que ser una ruta absoluta.")
    return ruta
```

```bash
cd /home/fruiz/worktrees/jax-frente-e && ln -s ../jax/core/config_entorno.py las_manos/config_entorno.py && git add jax/core/config_entorno.py las_manos/config_entorno.py
```

- [ ] **Paso 3: el conftest de la raíz fija valores de test**

En `conftest.py`, después de `os.environ["JAX_USAGE_SPOOL_DIR"] = tempfile.mkdtemp(prefix="jax-test-respaldo-uso-")`:

```python
#: E-21 (2026-09-16): jacobs/executor.py y plan.py leen LAS_MANOS_URL y
#: JAX_OLLAMA_URL al importarse y NO arrancan sin ellas (fail-closed). Se fijan
#: acá, antes de cualquier import, con los mismos valores que producción: los
#: tests no salen a la red (cada uno parchea el transporte), así que el valor
#: solo tiene que ser una URL válida y estable para las aserciones de ruta.
os.environ["LAS_MANOS_URL"] = "http://127.0.0.1:7777"
os.environ["JAX_OLLAMA_URL"] = "http://localhost:11434"
```

- [ ] **Paso 4: consumidores**

`jacobs/executor.py`:
- Agregar el import `from config_entorno import url_requerida` junto a `from redaccion import ...`.
- L44-45:
```python
# E-21 (2026-09-16): del entorno, validadas al importar. Sin ellas LAS MANOS no
# arranca (EntornoInvalido en el journal) en vez de apuntar a un host fijo.
LAS_MANOS_BASE = url_requerida("LAS_MANOS_URL")
OLLAMA_URL     = url_requerida("JAX_OLLAMA_URL") + "/api/chat"
```

`jacobs/plan.py`: import `from config_entorno import url_requerida`. L46: `OLLAMA_URL = url_requerida("JAX_OLLAMA_URL") + "/api/chat"  # E-21: del entorno, validada al importar`.

`jax/memory/db.py`: import `from jax.core.config_entorno import url_requerida`. En `get_embedding`, la URL va **antes** del `try`. Si falta configuración no es "Ollama falló", y no se convierte en `None`:
```python
        # E-21: la URL sale del entorno y se lee FUERA del try. Una variable que
        # falta es un error de configuración visible, no un embedding que
        # "falló" y deja la fila en ceros sin que nadie lo sepa.
        url = url_requerida("JAX_OLLAMA_URL") + "/api/embed"
        try:
            # Tope de contexto de los modelos; truncamos para evitar 500.
            texto = text[:4000] if len(text) > 4000 else text
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    url,
                    json={"model": EMBED.model, "input": texto},
                )
```
El resto del cuerpo queda igual. E-14 y E-24 lo tocan después.

`jax/muscles/ollama_muscle.py`: borrar `DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"` y la línea en blanco. Firma:
```python
    def __init__(
        self,
        name: str,
        model_default: str,
        models_allowed: list[str],
        system_prompt: str,
        timeout: float,
        *,
        api_url: str,
        authority_origin: str = "",
    ) -> None:
```
En el docstring del módulo, `API HTTP local en localhost:11434/api/chat` pasa a `API HTTP local en $JAX_OLLAMA_URL/api/chat (E-21)`.

`jax/muscles/base.py`, en `HttpMuscle` (después de `_limite_de_salida`):
```python
    def _url_del_catalogo(self) -> str:
        """E-21 (2026-09-16): la URL del proveedor sale SOLO del catálogo
        (provider.base_url, puesta en api_url por registro_facetas al arrancar).
        Antes había URLs de OpenAI/DeepSeek/Gemini como default: con la DB
        caída se despachaba a una URL que nadie eligió."""
        if not self.api_url:
            raise MuscleInvocationError(
                f"[{self.name}] sin URL del proveedor: sale del catálogo (provider.base_url) "
                f"al arrancar y el catálogo no la dio; no se despacha a una URL fija."
            )
        return self.api_url
```
- `_call_deepseek`: `url = self._url_del_catalogo()`, y se borran las dos líneas de comentario de PR-K sobre el default.
- `_call_openai`: `url = self._url_del_catalogo()`.
- `_call_gemini`: `base = self._url_del_catalogo()`. En el comentario de PR-K ronda 2 se quita la frase `El default solo queda para el arranque sin DB (config.toml completo).`

`jax/voice/tts.py`: borrar `KOKORO_PYTHON = Path.home() / "kokoro-test" / ".venv" / "bin" / "python"` y agregar:
```python
from jax.core.config_entorno import ruta_absoluta_requerida


def _python_de_kokoro() -> Path:
    """E-21 (2026-09-16): el python del venv de Kokoro sale de
    JAX_KOKORO_PYTHON (/etc/jax/.env). Antes era ~/kokoro-test/.venv fijo."""
    return ruta_absoluta_requerida("JAX_KOKORO_PYTHON")
```
En `_ensure_worker`: `str(_python_de_kokoro()), str(WORKER_SCRIPT),`.

`jax/core/main.py`:
- Import `from jax.core.config_entorno import url_requerida`.
- En `build_muscles`, rama `ollama`: `p["system_prompt"], timeout, api_url=url_requerida("JAX_OLLAMA_URL") + "/api/chat",`.
- En `main()`, como primera instrucción después del docstring o de la carga de config, validar lo que el REPL necesita, para fallar al arrancar y no en el primer turno:
```python
    # E-21: sin estas variables el REPL no arranca; fallar acá y no en el
    # primer turno o la primera frase hablada.
    url_requerida("JAX_OLLAMA_URL")
    from jax.voice.tts import _python_de_kokoro
    _python_de_kokoro()
```

`config/config.toml`: borrar las tres líneas `api_url = ...` (L35 `jax_local`, L284 `ada`, L334 `kimi`).

- [ ] **Paso 5: tests existentes que construían músculos sin URL**

- `tests/test_contrato_dispatch_repl_ada.py::_muscle` (L56-60): agregar `api_url="https://api.proveedor.example/v1/chat/completions",` a `base.HttpMuscle(...)`.
- `tests/test_contrato_dispatch_repl_ada.py::test_repl_ollama_usa_el_endpoint_nativo_del_toml_y_num_predict` (L711-722): renombrar a `test_repl_ollama_usa_el_endpoint_nativo_del_entorno_y_num_predict`. Quitar `"api_url": nativo` del dict de la personalidad y reemplazar `nativo = "http://localhost:11434/api/chat"` por:
  ```python
        nativo = "http://localhost:11434/api/chat"  # JAX_OLLAMA_URL del conftest + /api/chat
  ```
- `tests/test_gemini_key_en_cabecera.py::ReplGeminiCabeceraTest._muscle`: agregar `api_url="https://generativelanguage.googleapis.com/v1beta",`.
- `tests/test_repl_fuentes.py:73`: agregar `api_url="https://generativelanguage.googleapis.com/v1beta",`.

Run: `git -C /home/fruiz/worktrees/jax-frente-e grep -n "OllamaMuscle(\|HttpMuscle(" -- tests las_manos jacobs ':!tests/_probe*'`.
Toda construcción tiene que pasar `api_url=`. Los `_probe_*` no corren en CI: se ajustan igual si fallan al importar.

- [ ] **Paso 6: cambio pareado en jax-platform (`_call_ollama`)**

`/home/fruiz/worktrees/jax-platform-frente-e/backend/api/chat.py`, arriba de `async def _call_ollama`:
```python
def _url_de_ollama() -> str:
    """E-21 (2026-09-16): el host de Ollama sale de JAX_OLLAMA_URL (/etc/jax/.env),
    la misma variable que usan Jacobs, el REPL y la memoria de jax. Antes se leía
    de personalities.jax_local.api_url del config.toml de jax, que ya no la trae.
    Sin la variable: error explícito, no un default a localhost."""
    valor = os.environ.get("JAX_OLLAMA_URL", "").strip().rstrip("/")
    if not valor:
        raise RuntimeError("JAX_OLLAMA_URL no está seteada: agregala a /etc/jax/.env.")
    return valor
```
En `_call_ollama`: `url = f"{_url_de_ollama()}/api/chat"`. El parámetro `config` queda: la firma la usan los llamadores y los tests.

`backend/tests/test_chat_http_pooling.py::test_call_ollama_uses_the_shared_client`: agregar el parámetro `monkeypatch`, `monkeypatch.setenv("JAX_OLLAMA_URL", "http://127.0.0.1:11434")` al principio, y pasar
`{"personalities": {"jax_local": {}}}` como config. La aserción `url == "http://127.0.0.1:11434/api/chat"` queda.

`backend/tests/test_chat_usage_capture.py:76`: reemplazar `{"personalities": {"jax_local": {"api_url": "http://localhost:11434/api/chat"}}}` por `{"personalities": {"jax_local": {}}}` y agregar `monkeypatch.setenv("JAX_OLLAMA_URL", "http://localhost:11434")` en ese test. Si el test no recibe `monkeypatch`, se agrega el parámetro.

`jax-platform/.github/workflows/policy.yml`: en el bloque `env:` de `backend-tests-con-db` (después de `JAX_REPO_PATH: /tmp/jax`) y en el de `backend-tests-no-db` (después de `JAX_REPO_PATH: /tmp/jax`), agregar `      JAX_OLLAMA_URL: http://127.0.0.1:11434`.

Run:
```bash
$PYTEST_PLAT -q tests/test_chat_http_pooling.py tests/test_chat_usage_capture.py tests/test_kimi_chat_transport.py tests/test_facet_model_wiring.py tests/test_facet_health_outcomes.py 2>&1 | tail -3
```
Rojo primero: sin el cambio de `chat.py`, `test_call_ollama_uses_the_shared_client` falla con `KeyError: 'api_url'`. Esperado después: verde.

- [ ] **Paso 7: verde en jax**

Run:
```bash
$PYTEST_JAX -v tests/test_config_entorno.py
$PYTEST_JAX -q tests/test_contrato_dispatch_repl_ada.py tests/test_gemini_key_en_cabecera.py tests/test_repl_fuentes.py \
  tests/test_memory_embedding_config.py tests/test_repl_modelos_permitidos.py tests/test_hipatia_fuentes.py \
  tests/test_motor_contexto_y_salida.py tests/test_chunking.py 2>&1 | tail -2
```
Esperado: 16 PASS y el resto verde. Si `test_ningun_modulo_de_servicio_tiene_una_url_escrita` marca una constante que no es
una URL de despacho, **reportarla al controller**: no se agrega una excepción sin revisión.

- [ ] **Paso 8: CI y commits**

`tests/test_config_entorno.py` va a las dos listas de `tests-puros`.

```bash
cd /home/fruiz/worktrees/jax-frente-e && git add jax/core/config_entorno.py las_manos/config_entorno.py conftest.py jacobs/executor.py jacobs/plan.py \
  jax/memory/db.py jax/muscles/ollama_muscle.py jax/muscles/base.py jax/voice/tts.py jax/core/main.py config/config.toml \
  tests/test_config_entorno.py tests/test_contrato_dispatch_repl_ada.py tests/test_gemini_key_en_cabecera.py tests/test_repl_fuentes.py .github/workflows/policy.yml
git commit -m "fix(E-21): URLs de servicio desde el entorno (LAS_MANOS_URL, JAX_OLLAMA_URL, JAX_KOKORO_PYTHON) con validacion fail-closed; sin defaults de proveedor" \
  -m "La URL de proveedor sale solo del catalogo. config.toml sin api_url. 16 tests vistos en rojo." \
  -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
cd /home/fruiz/worktrees/jax-platform-frente-e && git add backend/api/chat.py backend/tests/test_chat_http_pooling.py backend/tests/test_chat_usage_capture.py .github/workflows/policy.yml
git commit -m "fix(E-21): _call_ollama toma el host de JAX_OLLAMA_URL, no del config.toml de jax" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: E-12, E-22, E-18 · documentos del repo: ruta del entorno, escritura fuera del loop, justificación verdadera

**Archivos:**
- Crear: `tests/test_documentos_del_repo.py`
- Modificar: `conftest.py` (`JAX_REPO_BASE_DIR` a un tmpdir)
- Modificar: `jacobs/executor.py` (constante `REPO_DOCUMENTS_DIR`, `_persist_step_to_repo` en L1212-1260 y la marca de L1067)
- Modificar: `tests/test_hipatia_fuentes.py:204-213`
- Modificar: `.github/workflows/policy.yml` (las dos listas de `tests-puros`)

**Interfaces:**
- Consume: `config_entorno.ruta_absoluta_requerida` (Tarea 8).
- Produce: `jacobs.executor.REPO_DOCUMENTS_DIR: Path` (= `$JAX_REPO_BASE_DIR/documents`). La variable `JAX_REPO_BASE_DIR` es la misma que usa `REPO_BASE` en jax-platform.

- [ ] **Paso 1: confirmar el nombre de la variable con el frente A (A-55)**

Run:
```bash
git -C /home/fruiz/jax-platform fetch origin
for ref in origin/master $(git -C /home/fruiz/jax-platform branch -r --list 'origin/fix/hallazgos-frente-a*'); do
  echo "== $ref"; git -C /home/fruiz/jax-platform grep -n "REPO_BASE" "$ref" -- backend/api/admin/repository.py
done
```

Regla:
- Si algún ref ya lee `REPO_BASE` de una variable de entorno con **otro** nombre, usar ese nombre en esta tarea y en la
  Tarea 17: reemplazar `JAX_REPO_BASE_DIR` en todo lo que sigue.
- Si ninguno lo hace todavía, el nombre es `JAX_REPO_BASE_DIR`, y se avisa al controller del frente A en una línea para
  que A-55 use el mismo nombre.

- [ ] **Paso 2: tests que fallan**

```python
# tests/test_documentos_del_repo.py
"""E-12 / E-22 / E-18 (2026-09-16): la copia .md de cada step.

- E-22: el directorio sale de JAX_REPO_BASE_DIR (/etc/jax/.env), la MISMA
  variable con la que jax-platform lista y sirve esos documentos
  (api/admin/repository.py, REPO_BASE). Antes: ~/jax/repo/documents fijo en
  los dos repos.
- E-12: la escritura corre en asyncio.to_thread (mkdir incluido), sin
  aiofiles, una dependencia usada en un solo lugar.
- E-18: la marca fail-soft decía "nadie lee ese .md". Es falso: el admin de
  jax-platform lo lista.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

RAIZ = Path(__file__).resolve().parents[1]


def test_el_documento_se_escribe_fuera_del_loop_en_el_directorio_configurado(tmp_path):
    from jacobs import executor
    hilos = []
    real = asyncio.to_thread

    async def espia(fn, *args, **kwargs):
        hilos.append(fn)
        return await real(fn, *args, **kwargs)

    async def correr():
        with patch.object(executor, "REPO_DOCUMENTS_DIR", tmp_path / "documents"), \
             patch.object(executor.asyncio, "to_thread", espia):
            await executor._persist_step_to_repo(
                pipeline_id="12345678-x", pipeline_name="t", step_index=1, facet="jekyll",
                capability="analysis", raw_output={"success": True, "result": "cuerpo", "model": "m"})

    asyncio.run(correr())
    assert hilos, "la escritura corrió dentro del event loop"
    assert (tmp_path / "documents" / "12345678_01_jekyll.md").read_text(encoding="utf-8").endswith("cuerpo")


def test_la_marca_fail_soft_de_la_copia_dice_la_verdad():
    fuente = (RAIZ / "jacobs" / "executor.py").read_text(encoding="utf-8")
    [linea] = [l for l in fuente.splitlines() if "except Exception as _persist_err" in l]
    assert "# fail-soft:" in linea
    assert "nadie lee" not in linea
    assert "/api/admin/repo" in linea and "output_ref" in linea


def test_los_tests_no_escriben_en_el_repo_de_produccion():
    base = Path(os.environ["JAX_REPO_BASE_DIR"])
    assert base.is_absolute() and str(base).startswith(tempfile.gettempdir())
    from jacobs import executor
    assert executor.REPO_DOCUMENTS_DIR == base / "documents"
```

Run: `$PYTEST_JAX -v tests/test_documentos_del_repo.py`
Esperado: 3 FAIL. Primero `AttributeError: REPO_DOCUMENTS_DIR`; después, la marca con "nadie lee"; y `KeyError: 'JAX_REPO_BASE_DIR'`.

- [ ] **Paso 3: implementar**

`conftest.py`, después de las variables de la Tarea 8:
```python
#: E-22 (2026-09-16): el .md de cortesía de cada step se escribe en
#: $JAX_REPO_BASE_DIR/documents. En producción es /home/fruiz/jax/repo, que el
#: admin de jax-platform lista. Un test que ejercite _persist_step_to_repo no
#: puede dejar ahí un documento de mentira: mismo criterio que el respaldo de uso.
os.environ["JAX_REPO_BASE_DIR"] = tempfile.mkdtemp(prefix="jax-test-repo-")
```

`jacobs/executor.py`:
- Import `from config_entorno import ruta_absoluta_requerida, url_requerida`.
- Después de `OLLAMA_URL = ...`:
```python
# E-22 (2026-09-16): `documents/` dentro de JAX_REPO_BASE_DIR, la MISMA variable
# con la que jax-platform (api/admin/repository.py, REPO_BASE) lista y sirve
# estos .md. Validada al importar: sin ella LAS MANOS no arranca.
REPO_DOCUMENTS_DIR = ruta_absoluta_requerida("JAX_REPO_BASE_DIR") / "documents"
```
- `_persist_step_to_repo` completo:
```python
async def _persist_step_to_repo(
    pipeline_id: str,
    pipeline_name: str,
    step_index: int,
    facet: str,
    capability: str,
    raw_output: dict,
) -> None:
    """Guarda el output de un step como .md en REPO_DOCUMENTS_DIR: copia de
    cortesía que el admin de jax-platform lista en /api/admin/repo. La escritura
    (mkdir incluido) corre en un hilo: nada de disco dentro del event loop (E-12)."""
    filename = f"{pipeline_id[:8]}_{step_index:02d}_{facet}.md"
    directorio = REPO_DOCUMENTS_DIR
    filepath = directorio / filename

    result_text = raw_output.get("result", "")
    sources     = raw_output.get("sources", [])
    model       = raw_output.get("model", "desconocido")
    success     = raw_output.get("success", False)

    lines = [
        f"# {pipeline_name}",
        f"",
        f"| Campo | Valor |",
        f"|-------|-------|",
        f"| Pipeline | `{pipeline_id}` |",
        f"| Step | {step_index + 1} |",
        f"| Faceta | {facet} |",
        f"| Capability | {capability} |",
        f"| Modelo | {model} |",
        f"| Estado | {'✓ completado' if success else '✗ fallido'} |",
        f"",
        f"## Respuesta",
        f"",
        result_text,
    ]

    if sources:
        lines += ["", "## Fuentes", "", render_sources_block(sources)]

    content = "\n".join(lines)

    def _escribir() -> None:
        directorio.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")

    await asyncio.to_thread(_escribir)

    logger.info("Step output persistido: %s", filename)
```
- La marca de L1067 queda en una sola línea:
```python
        except Exception as _persist_err:  # noqa: BLE001  # fail-soft: es la copia .md de cortesía en REPO_DOCUMENTS_DIR que el admin de jax-platform lista (/api/admin/repo) -- el output canónico ya quedó en output_ref y en store.step_upsert antes de este try; si la copia falla queda el warning y el step sigue completado
```

`tests/test_hipatia_fuentes.py::test_el_md_del_repo_muestra_url_final_y_cita`:
```python
    async def test_el_md_del_repo_muestra_url_final_y_cita(self):
        with tempfile.TemporaryDirectory() as base, \
                patch.object(executor, "REPO_DOCUMENTS_DIR", Path(base) / "documents"):
            await executor._persist_step_to_repo(
                pipeline_id="abcdef12-x", pipeline_name="t", step_index=0, facet="hipatia",
                capability="research",
                raw_output={"success": True, "result": "texto", "model": "m", "sources": [
                    {**_src(quotes=[_Q2]), "final_url": "https://www.postgresql.org/about/news/", "resolved": True}]},
            )
            md = (Path(base) / "documents" / "abcdef12_00_hipatia.md").read_text(encoding="utf-8")
        assert "https://www.postgresql.org/about/news/" in md and _Q2 in md, md
```

- [ ] **Paso 4: verde**

Run:
```bash
$PYTEST_JAX -v tests/test_documentos_del_repo.py
$PYTEST_JAX -q tests/test_hipatia_fuentes.py tests/test_no_blocking_in_async.py tests/test_config_entorno.py 2>&1 | tail -2
git -C /home/fruiz/worktrees/jax-frente-e grep -n "aiofiles\|expanduser(\"~/jax/repo" -- jacobs jax las_manos
cd /home/fruiz/worktrees/jax-frente-e && env JAX_PLATFORM_REPO_ROOT=/home/fruiz/worktrees/jax-platform-frente-e /home/fruiz/jax/.venv/bin/python -m pytest -q policy/tests/test_no_fail_open_except.py 2>&1 | tail -1
```
Esperado: 3 PASS, el resto verde, el grep vacío y P10 verde.

- [ ] **Paso 5: CI y commit**

`tests/test_documentos_del_repo.py` va a las dos listas de `tests-puros`.

```bash
cd /home/fruiz/worktrees/jax-frente-e && git add conftest.py jacobs/executor.py tests/test_documentos_del_repo.py tests/test_hipatia_fuentes.py .github/workflows/policy.yml
git commit -m "fix(E-12,E-22,E-18): documentos en JAX_REPO_BASE_DIR, escritura en asyncio.to_thread sin aiofiles, marca fail-soft verdadera" \
  -m "3 tests vistos en rojo." -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: E-19 · `requirements.txt` completo y CI que instala desde él

**Archivos:**
- Crear: `tests/test_requirements_completos.py`
- Modificar: `requirements.txt` (quitar `aiofiles`, fijar `cryptography==49.0.0` y `PyYAML==6.0.3`)
- Modificar: `.github/workflows/policy.yml`: `pip install` de `facet-health-unit`, `facet-health-io`, `memory-vector-zero-io`, `facet-resolver-seal`, `plan-timeout-ceiling`, `mirror-sync` (paso de `aiomysql`), `governance` y `tests-puros`; y las dos listas de `tests-puros`

**Interfaces:**
- Consume: la Tarea 9 (ningún `import aiofiles` en el árbol).
- Produce: nada.

- [ ] **Paso 1: confirmar las versiones vivas (sólo lectura)**

Run:
```bash
ls /home/fruiz/jax/.venv/lib/python*/site-packages | grep -iE "^(cryptography|pyyaml)-.*dist-info"
ls /home/fruiz/jax/las_manos/.venv/lib/python*/site-packages | grep -iE "^(cryptography|pyyaml)-.*dist-info"
```
Esperado: `cryptography-49.0.0.dist-info` y `pyyaml-6.0.3.dist-info` (este sólo en `.venv`). Si otra versión aparece, se fija la
que diga el listado y se corrige esta tarea.

- [ ] **Paso 2: tests que fallan**

```python
# tests/test_requirements_completos.py
"""E-19 (2026-09-16): requirements.txt es la fuente ÚNICA de las dependencias
de servicio, y CI instala desde ahí.

Faltaban cryptography (jax/core/crypto_secrets.py) y pyyaml
(policy/governance/loaders.py): con `pip install -r requirements.txt`, como dice
README.md, ni credential_resolver ni la gobernanza importaban. CI no lo veía
porque cada job instalaba su propia lista a mano.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
_ARBOLES = ("jax", "jacobs", "las_manos", "policy")
_EXCLUIDOS = {".git", "__pycache__", ".venv", "venv", "node_modules", "tests", ".pytest_cache"}

# import de nivel superior -> distribución fijada en requirements.txt
_DISTRIBUCION = {
    "aiomysql": "aiomysql", "cryptography": "cryptography", "fastapi": "fastapi",
    "httpx": "httpx", "pydantic": "pydantic", "uvicorn": "uvicorn", "yaml": "pyyaml",
}

# Terceros que NO van en requirements.txt, cada uno con su motivo.
_FUERA_DE_REQUIREMENTS = {
    "faster_whisper": "jax/voice: venv propio de la voz (JAX_KOKORO_PYTHON), no el de servicio",
    "kokoro": "jax/voice: venv propio de la voz (JAX_KOKORO_PYTHON), no el de servicio",
    "numpy": "jax/voice: venv propio de la voz (JAX_KOKORO_PYTHON), no el de servicio",
    "soundfile": "jax/voice: venv propio de la voz (JAX_KOKORO_PYTHON), no el de servicio",
    "sentence_transformers": "reranker opcional de jax/memory/db.py::_get_reranker, import perezoso con fail-soft declarado",
}


def _es_test(p: Path) -> bool:
    return p.name.startswith("test_") or p.name.endswith("_test.py")


def _recorrer(base: Path):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUIDOS]
        for nombre in filenames:
            yield Path(dirpath) / nombre


def _modulos_locales() -> set[str]:
    locales = set()
    for dirpath, dirnames, filenames in os.walk(RAIZ):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUIDOS]
        locales.update(dirnames)
        locales.update(Path(f).stem for f in filenames if f.endswith(".py"))
    return locales


def _terceros_en(fuente: str, locales: set[str]) -> set[str]:
    tops = set()
    for nodo in ast.walk(ast.parse(fuente)):
        if isinstance(nodo, ast.Import):
            tops.update(a.name.split(".")[0] for a in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.level == 0 and nodo.module:
            tops.add(nodo.module.split(".")[0])
    return {t for t in tops if t not in sys.stdlib_module_names and t not in locales}


def _archivos_de_servicio():
    for arbol in _ARBOLES:
        for p in _recorrer(RAIZ / arbol):
            if p.suffix == ".py" and not _es_test(p):
                yield p


def _requirements() -> dict[str, str]:
    fijadas = {}
    for linea in (RAIZ / "requirements.txt").read_text(encoding="utf-8").splitlines():
        linea = linea.split("#", 1)[0].strip()
        if linea:
            nombre, _, version = linea.partition("==")
            fijadas[nombre.strip().lower().replace("_", "-")] = version.strip()
    return fijadas


def test_todo_import_de_terceros_esta_declarado():
    locales = _modulos_locales()
    sin_declarar: dict[str, list[str]] = {}
    for p in _archivos_de_servicio():
        for tercero in _terceros_en(p.read_text(encoding="utf-8", errors="replace"), locales):
            if tercero not in _DISTRIBUCION and tercero not in _FUERA_DE_REQUIREMENTS:
                sin_declarar.setdefault(tercero, []).append(str(p.relative_to(RAIZ)))
    assert sin_declarar == {}


def test_cada_distribucion_de_servicio_esta_fijada_con_version_exacta():
    fijadas = _requirements()
    assert sorted(d for d in set(_DISTRIBUCION.values()) if not fijadas.get(d)) == []


def test_aiofiles_ya_no_se_usa_ni_se_instala():
    assert "aiofiles" not in _requirements()
    locales = _modulos_locales()
    assert [str(p.relative_to(RAIZ)) for p in _archivos_de_servicio()
            if "aiofiles" in _terceros_en(p.read_text(encoding="utf-8", errors="replace"), locales)] == []


def test_el_ci_instala_desde_requirements():
    lineas = (RAIZ / ".github" / "workflows" / "policy.yml").read_text(encoding="utf-8").splitlines()
    inicio = lineas.index("  tests-puros:")
    fin = next(i for i in range(inicio + 1, len(lineas))
               if lineas[i].startswith("  ") and not lineas[i].startswith("   ") and lineas[i].rstrip().endswith(":"))
    assert any("pip install -r requirements.txt" in l for l in lineas[inicio:fin])
    assert [l for l in lineas if "pip install" in l and "aiofiles" in l] == []


def test_el_detector_ve_un_import_de_terceros():
    fuente = "import yaml\nimport os\nfrom jacobs import store\nfrom paquete_raro.sub import x\n"
    assert _terceros_en(fuente, {"jacobs"}) == {"yaml", "paquete_raro"}
```

Run: `$PYTEST_JAX -v tests/test_requirements_completos.py`
Esperado: 3 FAIL (cryptography y pyyaml sin fijar; aiofiles en requirements; CI sin `-r`) y 2 PASS. Los PASS son
`test_todo_import...`, que ya es verdad, y el control del detector.

- [ ] **Paso 3: `requirements.txt`**

Borrar la línea `aiofiles==25.1.0`. Agregar `cryptography==49.0.0` después de `click==8.4.2` y `PyYAML==6.0.3` después de `PyMySQL==1.2.0`.

- [ ] **Paso 4: CI instala desde el archivo**

En `.github/workflows/policy.yml`:
- `tests-puros`: reemplazar las 4 líneas de comentario (`# aiofiles: ...` hasta `# motor_registry.routes (pin de requirements.txt).`) y el `pip install` por:
  ```
      # Dependencias de servicio desde requirements.txt (E-19, 2026-09-16): el
      # archivo es la fuente única; tests/test_requirements_completos.py falla si
      # un import de terceros no está fijado ahí o si este job deja de usarlo.
      - run: pip install -r requirements.txt pytest pytest-asyncio
  ```
- `facet-health-unit`: `- run: pip install -r requirements.txt pytest`
- `facet-health-io`: `- run: pip install -r requirements.txt pytest`
- `memory-vector-zero-io`: `- run: pip install -r requirements.txt pytest`
- `facet-resolver-seal`: `- run: pip install -r requirements.txt pytest`
- `plan-timeout-ceiling`: `- run: pip install -r requirements.txt pytest`
- `governance`: `- run: pip install -r requirements.txt pytest`
- `mirror-sync`: `- run: pip install aiomysql` pasa a `- run: pip install -r requirements.txt`

Los comentarios que explican por qué cada job necesitaba cada paquete quedan: siguen siendo verdad. En cada uno se agrega la línea
`# Desde E-19 (2026-09-16) todo sale de requirements.txt.`

- [ ] **Paso 5: verde, y probar el archivo en un venv limpio**

Run:
```bash
$PYTEST_JAX -v tests/test_requirements_completos.py
python3 -m venv "$SCRATCH/venv-e19" && "$SCRATCH/venv-e19/bin/pip" install -q -r /home/fruiz/worktrees/jax-frente-e/requirements.txt pytest pytest-asyncio && \
  cd /home/fruiz/worktrees/jax-frente-e && env -u JAX_DB_HOST JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos "$SCRATCH/venv-e19/bin/python" -m pytest -q \
  tests/test_requirements_completos.py tests/test_espejos_symlink_frente_e.py tests/test_hipatia_fuentes.py tests/test_governance_loaders.py 2>&1 | tail -2
```
Esperado: 5 PASS en el venv de siempre, y verde en el venv limpio (3.14). Que también instale en 3.12 lo confirma el runner
en la Tarea 16.

- [ ] **Paso 6: CI y commit**

`tests/test_requirements_completos.py` va a las dos listas de `tests-puros`.

```bash
cd /home/fruiz/worktrees/jax-frente-e && git add requirements.txt .github/workflows/policy.yml tests/test_requirements_completos.py
git commit -m "fix(E-19): requirements.txt con cryptography y pyyaml fijados, sin aiofiles; CI instala desde el archivo" \
  -m "3 tests vistos en rojo; 2 controles." -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: E-14, E-15 · recorte simple del embedding y `_sin_autoetiqueta` con la cabecera de la config

**Archivos:**
- Crear: `tests/test_sin_autoetiqueta.py`
- Modificar: `jax/memory/db.py` (`texto = text[:4000]`)
- Modificar: `jax/muscles/base.py` (helper del módulo; `_call_deepseek` y `_call_openai`)
- Modificar: `.github/workflows/policy.yml` (las dos listas de `tests-puros`)

**Interfaces:**
- Consume: `HttpMuscle._url_del_catalogo` (Tarea 8).
- Produce: `jax.muscles.base._sin_autoetiqueta(texto: str, etiqueta: str) -> str`.

- [ ] **Paso 1: tests que fallan (E-15)**

```python
# tests/test_sin_autoetiqueta.py
"""E-15 (2026-09-16): el filtro de autoetiquetas del modelo usa la cabecera de
la etiqueta CONFIGURADA (config.toml, authority_origin), no un literal.

Antes, las dos copias del filtro buscaban "⚙️ *Origen" (la etiqueta de kimi)
escrito en el código: si DeepSeek imitaba SU etiqueta (🧠), la línea falsa
llegaba al usuario pegada a la verdadera que agrega el sistema.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from jax.muscles import base  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
KIMI = "⚙️ *Origen de autoridad: Kimi K2.7 Code (Moonshot AI). Sin verificación externa.*"
PROPIO = "🧠 *Origen de autoridad: conocimiento propio del modelo (no verificado en web).*"


def test_quita_la_autoetiqueta_de_kimi():
    assert base._sin_autoetiqueta(f"hola\n{KIMI}\nchau", KIMI) == "hola\nchau"


def test_quita_la_autoetiqueta_imitada_de_otra_faceta():
    assert base._sin_autoetiqueta("respuesta\n  🧠 *Origen de autoridad: inventada*", PROPIO) == "respuesta"


def test_sin_etiqueta_configurada_no_quita_nada():
    assert base._sin_autoetiqueta(f"  a\n{KIMI}  ", "") == f"a\n{KIMI}"


def test_el_prefijo_de_kimi_ya_no_esta_escrito_en_el_codigo():
    assert "⚙️ *Origen" not in (RAIZ / "jax" / "muscles" / "base.py").read_text(encoding="utf-8")


class DeepseekLimpiaSuPropiaEtiquetaTest(unittest.IsolatedAsyncioTestCase):
    async def test_la_respuesta_sale_sin_la_etiqueta_imitada(self):
        musculo = base.HttpMuscle(
            name="jekyll", provider="deepseek", model_default="d", models_allowed=["d"], system_prompt="s",
            timeout=10, authority_origin=PROPIO, api_url="https://api.deepseek.example/chat/completions")
        cuerpo = {"model": "d", "choices": [{"message": {"content": f"respuesta\n{PROPIO}"}}]}

        async def handle(transport_self, request):
            return httpx.Response(200, json=cuerpo)

        with patch("httpx.AsyncHTTPTransport.handle_async_request", handle), \
             patch.object(base.HttpMuscle, "_resolve_api_key", AsyncMock(return_value="k")), \
             patch.object(base.HttpMuscle, "_limite_de_salida", AsyncMock(return_value={})), \
             patch.object(base, "record_resolved_version_safe", AsyncMock()):
            salida = await musculo._call_deepseek("hola", "d")
        self.assertEqual(salida, "respuesta")
```

Run: `$PYTEST_JAX -v tests/test_sin_autoetiqueta.py`. Esperado: 5 FAIL, por `AttributeError` y por el literal presente.

- [ ] **Paso 2: implementar**

`jax/muscles/base.py`, después de `_PROVIDER_ID_MAP`:
```python
def _sin_autoetiqueta(texto: str, etiqueta: str) -> str:
    """Quita las líneas en que el MODELO imita el origen de autoridad que el
    SISTEMA agrega después (_append_authority, Decisión 3). La cabecera sale de
    la etiqueta configurada (config.toml, `authority_origin`): lo que está
    antes del primer ':' -- p. ej. "⚙️ *Origen de autoridad". Antes era un
    literal de kimi en dos copias y las demás facetas no se limpiaban (E-15)."""
    cabecera = etiqueta.split(":", 1)[0].strip() if etiqueta else ""
    if not cabecera:
        return texto.strip()
    lineas = [l for l in texto.splitlines() if not l.strip().startswith(cabecera)]
    return "\n".join(lineas).strip()
```

`_call_deepseek`: borrar las líneas `# Kimi K2.7 incluye ...`, `# Limpiar auto-etiquetas ...` y `lineas = [l for l in texto.splitlines() ...]` dentro del `async with`. El final queda:
```python
        await record_resolved_version_safe(self.name, data.get("model"))
        # Kimi K2.7 trae reasoning_content aparte: no se usa. La autoetiqueta
        # que el modelo imite se quita con la cabecera configurada (E-15).
        return _sin_autoetiqueta(texto, self.authority_origin)
```
`_call_openai`: reemplazar las 4 líneas finales (comentarios, `lineas = [...]` y `return "\n".join(lineas).strip()`) por
`return _sin_autoetiqueta(texto, self.authority_origin)`, con el mismo comentario de dos líneas.

- [ ] **Paso 3: E-14 (sin test: no hay rojo posible, Discrepancia 7)**

`jax/memory/db.py`, en `get_embedding`: `texto = text[:4000] if len(text) > 4000 else text` pasa a `texto = text[:4000]`.

- [ ] **Paso 4: verde**

Run:
```bash
$PYTEST_JAX -v tests/test_sin_autoetiqueta.py
$PYTEST_JAX -q tests/test_contrato_dispatch_repl_ada.py tests/test_repl_fuentes.py tests/test_memory_embedding_config.py tests/test_degradaciones_declaradas.py tests/test_memoria_no_traga_fallos_de_escritura.py 2>&1 | tail -2
```
Esperado: 5 PASS y el resto verde.

- [ ] **Paso 5: CI y commit**

`tests/test_sin_autoetiqueta.py` va a las dos listas de `tests-puros`.

```bash
cd /home/fruiz/worktrees/jax-frente-e && git add jax/muscles/base.py jax/memory/db.py tests/test_sin_autoetiqueta.py .github/workflows/policy.yml
git commit -m "fix(E-14,E-15): _sin_autoetiqueta con la cabecera de authority_origin; recorte simple del embedding" \
  -m "E-15: 5 tests vistos en rojo. E-14 es equivalente y no admite rojo (declarado)." -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: E-16 · cuerpos de error de proveedor redactados antes de recortar, con la credencial conocida

**Archivos:**
- Crear: `tests/test_errores_de_proveedor_redactados.py`
- Modificar: `jacobs/executor.py:386,448`, `jacobs/plan.py:653-657`
- Modificar: `jax/muscles/base.py` (`_call_deepseek`, `_call_openai`), `jax/muscles/ollama_muscle.py:116-120`
- Modificar: `jax/core/main.py` (helper `_texto_de_error_de_tarea` y L486-487)
- Modificar: `.github/workflows/policy.yml` (las dos listas de `tests-puros`)

**Interfaces:**
- Consume: `redaccion.recortar_redactado(texto, limite, secretos)`, `redaccion.redactar_secretos(texto, secretos=())`, `HttpMuscle._url_del_catalogo` (Tarea 8).
- Produce: `jax.core.main._texto_de_error_de_tarea(e: BaseException) -> str`.

- [ ] **Paso 1: tests que fallan**

```python
# tests/test_errores_de_proveedor_redactados.py
"""E-16 (2026-09-16): el cuerpo de un error de proveedor se REDACTA (con la
credencial conocida) ANTES de recortarlo, en todos los caminos.

jax/core/redaccion.py:161 lo declara: recortar_redactado es "la única forma de
recortar un texto de error de proveedor en jax". Los caminos de Gemini lo
cumplían; estos no: executor (openai-compat y Ollama), REPL (DeepSeek, OpenAI,
Ollama) y el cerebro Ada del planner, que además lo mandaba a logger.error.
Recortar primero parte un secreto en el carácter 200 y el pedazo ya no lo
reconoce ninguna regla. run_task escribía el error de la tarea a disco sin
redactar. Todas las keys son FALSAS.
"""
from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

os.environ["JAX_DB_NAME"] = "jax_memory_test"

RAIZ = Path(__file__).resolve().parents[1]
# Sin forma reconocible: SOLO se tapa si se pasa como secreto conocido.
SECRETO = "zq7Wn2-sin-forma-reconocible-88Hk"
CUERPO_CON_SECRETO = "p" * 192 + SECRETO + " fin"
# Con forma reconocible (Authorization): el token cruza el corte de 200.
CUERPO_CON_BEARER = "p" * 170 + "Authorization: Bearer tok-FAKE-e16-0123456789 fin"


def _cuerpos_recortados(fuente: str) -> list[int]:
    """Líneas con `X.text[:N]` o `body[:N]` (body = await resp.aread())."""
    lineas = set()
    for funcion in ast.walk(ast.parse(fuente)):
        if not isinstance(funcion, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        leidos = {
            t.id for n in ast.walk(funcion)
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Await) and isinstance(n.value.value, ast.Call)
            and isinstance(n.value.value.func, ast.Attribute) and n.value.value.func.attr == "aread"
            for t in n.targets if isinstance(t, ast.Name)
        }
        for n in ast.walk(funcion):
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice):
                v = n.value
                if (isinstance(v, ast.Attribute) and v.attr == "text") or (isinstance(v, ast.Name) and v.id in leidos):
                    lineas.add(n.lineno)
    return sorted(lineas)


def _archivos_de_servicio():
    for arbol in ("jacobs", "jax", "las_manos"):
        for dirpath, dirnames, filenames in os.walk(RAIZ / arbol):
            dirnames[:] = [d for d in dirnames if d not in {".venv", "__pycache__", "tests"}]
            for nombre in filenames:
                if nombre.endswith(".py") and not nombre.startswith("test_") and not nombre.endswith("_test.py"):
                    yield Path(dirpath) / nombre


def test_ningun_cuerpo_de_proveedor_se_recorta_sin_redactar():
    hallazgos = {}
    for p in _archivos_de_servicio():
        lineas = _cuerpos_recortados(p.read_text(encoding="utf-8", errors="replace"))
        if lineas:
            hallazgos[str(p.relative_to(RAIZ))] = lineas
    assert hallazgos == {}, "usar recortar_redactado(texto, N, [credencial]) (jax/core/redaccion.py)"


def test_el_detector_ve_resp_text_recortado():
    assert _cuerpos_recortados("def f(resp):\n    return resp.text[:200]\n") == [2]


def test_el_detector_ve_un_cuerpo_leido_con_aread_y_recortado():
    fuente = "async def f(resp):\n    body = await resp.aread()\n    raise E(body[:200])\n"
    assert _cuerpos_recortados(fuente) == [3]


def test_el_detector_acepta_recortar_redactado():
    assert _cuerpos_recortados("def f(resp, k):\n    return recortar_redactado(resp.text, 200, [k])\n") == []


def _responder(status: int, texto: str):
    async def handle(transport_self, request):
        return httpx.Response(status, text=texto)
    return patch("httpx.AsyncHTTPTransport.handle_async_request", handle)


def _faceta(key="jekyll"):
    from facet_resolver import ResolvedFacet
    return ResolvedFacet(key=key, provider_id="deepseek", base_url="https://api.proveedor.example/v1", model="m",
                         credential=SECRETO, transport="http_openai_compat", persona=None, params=None)


class JacobsTest(unittest.IsolatedAsyncioTestCase):
    async def test_openai_compat_redacta_la_credencial_antes_de_recortar(self):
        from jacobs import executor
        with _responder(400, CUERPO_CON_SECRETO), patch.object(executor, "limite_de_salida", AsyncMock(return_value={})):
            with self.assertRaises(RuntimeError) as ctx:
                await executor._invoke_http_openai_compat(_faceta(), "hola", 10)
        mensaje = str(ctx.exception)
        self.assertIn("HTTP 400", mensaje)
        self.assertNotIn(SECRETO[:8], mensaje)
        self.assertIn("p" * 192 + "***", mensaje)

    async def test_ollama_redacta_antes_de_recortar(self):
        from jacobs import executor
        faceta = _faceta(key="jax_local")
        with _responder(500, CUERPO_CON_BEARER), patch.object(executor, "limite_de_salida", AsyncMock(return_value={})):
            with self.assertRaises(RuntimeError) as ctx:
                await executor._invoke_ollama(faceta, "hola", 10)
        self.assertNotIn("tok-FAKE", str(ctx.exception))

    async def test_ada_redacta_el_cuerpo_en_el_motivo_y_en_el_log(self):
        from jacobs import plan
        with _responder(400, CUERPO_CON_SECRETO), \
             patch.object(plan, "resolve_facet", AsyncMock(return_value=_faceta(key="ada"))), \
             patch.object(plan, "limite_de_salida", AsyncMock(return_value={"max_tokens": 10})), \
             self.assertLogs("jacobs.plan", level="ERROR") as logs:
            with self.assertRaises(plan.CerebroNoDisponible) as ctx:
                await plan.PlanBuilder()._ada_plan("objetivo", 3)
        self.assertIn("Ada HTTP 400", str(ctx.exception))
        self.assertNotIn(SECRETO[:8], str(ctx.exception))
        self.assertNotIn(SECRETO[:8], "\n".join(logs.output))


class ReplTest(unittest.IsolatedAsyncioTestCase):
    def _musculo(self, proveedor):
        from jax.muscles import base
        return base.HttpMuscle(name="jekyll", provider=proveedor, model_default="d", models_allowed=["d"],
                               system_prompt="s", timeout=10, api_url="https://api.proveedor.example/v1/chat/completions")

    async def _error(self, proveedor, metodo):
        from jax.muscles import base
        with _responder(400, CUERPO_CON_SECRETO), \
             patch.object(base.HttpMuscle, "_resolve_api_key", AsyncMock(return_value=SECRETO)), \
             patch.object(base.HttpMuscle, "_limite_de_salida", AsyncMock(return_value={})):
            with self.assertRaises(base.MuscleInvocationError) as ctx:
                await getattr(self._musculo(proveedor), metodo)("hola", "d")
        return str(ctx.exception)

    async def test_deepseek_redacta_la_credencial_conocida(self):
        mensaje = await self._error("deepseek", "_call_deepseek")
        self.assertIn("DeepSeek HTTP 400", mensaje)
        self.assertNotIn(SECRETO[:8], mensaje)

    async def test_openai_en_streaming_redacta_la_credencial_conocida(self):
        mensaje = await self._error("openai", "_call_openai")
        self.assertIn("OpenAI HTTP 400", mensaje)
        self.assertNotIn(SECRETO[:8], mensaje)

    async def test_ollama_local_redacta_antes_de_recortar(self):
        from jax.muscles.base import MuscleInvocationError
        from jax.muscles.ollama_muscle import OllamaMuscle
        musculo = OllamaMuscle("jax_local", "q", ["q"], "s", 10, api_url="http://ollama.example/api/chat")
        with _responder(500, CUERPO_CON_BEARER), \
             patch("jax.muscles.ollama_muscle.limite_de_salida", AsyncMock(return_value={"options": {"num_predict": 5}})):
            with self.assertRaises(MuscleInvocationError) as ctx:
                await musculo._call("hola", "q")
        self.assertNotIn("tok-FAKE", str(ctx.exception))


def test_el_error_de_una_tarea_se_escribe_redactado():
    from jax.core import main
    texto = main._texto_de_error_de_tarea(RuntimeError("fallo con api_key=sk-FAKE-tarea-0123456789"))
    assert "sk-FAKE-tarea" not in texto
    assert "api_key=***" in texto


def test_run_task_escribe_el_error_con_el_texto_redactado():
    arbol = ast.parse((RAIZ / "jax" / "core" / "main.py").read_text(encoding="utf-8"))
    run_task = next(n for n in ast.walk(arbol) if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_task")
    asignaciones = [n for n in ast.walk(run_task) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "error_msg" for t in n.targets)]
    assert asignaciones, "run_task ya no arma error_msg"
    for a in asignaciones:
        assert isinstance(a.value, ast.Call) and getattr(a.value.func, "id", None) == "_texto_de_error_de_tarea"
```

Run: `$PYTEST_JAX -v tests/test_errores_de_proveedor_redactados.py`
Esperado: 12 tests, 9 FAIL y 3 PASS. Los PASS son los controles del detector.
- El detector sobre el árbol falla con 6 sitios: executor 386 y 448, plan 655, base 264 y 306, ollama_muscle 119.
- Fallan los 6 de comportamiento y los 2 de `main`.

- [ ] **Paso 2: implementar**

`jacobs/executor.py`:
- L386: `raise RuntimeError(f"[{f.key}] HTTP {resp.status_code}: {recortar_redactado(resp.text, 200, [f.credential])}")`
- L448: `raise RuntimeError(f"Ollama HTTP {resp.status_code}: {recortar_redactado(resp.text, 200)}")`

`jacobs/plan.py`: import `from redaccion import recortar_redactado` (junto a los imports bare). L653-655:
```python
                        body = await resp.aread()
                        # E-16: redactar (con la credencial de Ada) ANTES de recortar;
                        # este motivo va a logger.error y a jacobs_events.
                        cuerpo = recortar_redactado(body.decode("utf-8", errors="replace"), 200, [f.credential])
                        motivo = f"Ada HTTP {resp.status_code}: {cuerpo}"
```

`jax/muscles/base.py`:
- En `_call_deepseek`: `api_key = await self._resolve_api_key()` y `headers = {"Authorization": f"Bearer {api_key}"}`. El raise:
```python
                raise MuscleInvocationError(
                    f"[{self.name}] DeepSeek HTTP {resp.status_code}: {recortar_redactado(resp.text, 200, [api_key])}"
                )
```
- En `_call_openai`: `api_key = await self._resolve_api_key()` y `"Authorization": f"Bearer {api_key}",`. El raise:
```python
                    body = await resp.aread()
                    cuerpo = recortar_redactado(body.decode("utf-8", errors="replace"), 200, [api_key])
                    raise MuscleInvocationError(
                        f"[{self.name}] OpenAI HTTP {resp.status_code}: {cuerpo}"
                    )
```

`jax/muscles/ollama_muscle.py`: import `from jax.core.redaccion import recortar_redactado`. El raise de L116-120:
```python
                    raise MuscleInvocationError(
                        f"[{self.name}] Ollama HTTP {resp.status_code}: "
                        f"{recortar_redactado(resp.text, 200)}"
                    )
```
`str(data)[:200]` (L127) queda, con este comentario encima: `# JSON ya parseado de un 200 del Ollama local (sin credencial): fuera de E-16, declarado.`

`jax/core/main.py`, después de `humanizar_error`:
```python
def _texto_de_error_de_tarea(e: BaseException) -> str:
    """El error de run_task se ESCRIBE en <tarea>_result.md: se redacta antes de
    tocar disco (E-16, 2026-09-16). Entero, sin recortar: es el diagnóstico de
    la tarea. humanizar_error ya redactaba lo que se imprime; el archivo no."""
    from jax.core.redaccion import redactar_secretos
    return redactar_secretos(str(e) or repr(e) or "error sin detalle")
```
En `run_task` (L487): `error_msg = _texto_de_error_de_tarea(e)`.

- [ ] **Paso 3: verde**

Run:
```bash
$PYTEST_JAX -v tests/test_errores_de_proveedor_redactados.py
$PYTEST_JAX -q tests/test_gemini_key_en_cabecera.py tests/test_contrato_dispatch_repl_ada.py tests/test_redaccion.py tests/test_degradaciones_declaradas.py 2>&1 | tail -2
```
Esperado: 12 PASS y el resto verde. `test_ada_http_no_200_es_error_y_queda_como_motivo_del_evento` sigue viendo "Ada HTTP 400".

- [ ] **Paso 4: CI y commit**

`tests/test_errores_de_proveedor_redactados.py` va a las dos listas de `tests-puros`.

```bash
cd /home/fruiz/worktrees/jax-frente-e && git add jacobs/executor.py jacobs/plan.py jax/muscles/base.py jax/muscles/ollama_muscle.py jax/core/main.py tests/test_errores_de_proveedor_redactados.py .github/workflows/policy.yml
git commit -m "fix(E-16): errores de proveedor redactados con la credencial conocida antes de recortar (executor, Ada, REPL) y error de tarea redactado a disco" \
  -m "Tripwire de clase sobre resp.text[:N] / body[:N]. 9 tests vistos en rojo; 3 controles del detector." \
  -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: E-24 · cliente HTTP compartido por proceso, con ciclo de vida, medido antes y después

**Archivos:**
- Crear: `scripts/medir_cliente_http_compartido.py`
- Crear: `jax/core/cliente_http_compartido.py` y el symlink `las_manos/cliente_http_compartido.py`
- Crear: `tests/test_cliente_http_compartido.py`
- Modificar: `jacobs/executor.py` (6 sitios), `jacobs/plan.py` (2 sitios), `jacobs/reaper.py:100`, `las_manos/motor_registry/worker.py:164`
- Modificar: `jax/core/grounding_sources.py:121-133`, `jax/muscles/base.py` (3 sitios), `jax/muscles/ollama_muscle.py:114`, `jax/memory/db.py` (`__init__`, `close`, `get_embedding`)
- Modificar: `las_manos/server.py` (hook de shutdown), `jax/core/main.py` (dos `finally`), `jax/memory/worker.py:398-399`, `jax/memory/synthesis_worker.py:191-192`, `jax/memory/embedding_worker.py:199`
- Modificar: `.github/workflows/policy.yml` (las dos listas de `tests-puros`)

**Interfaces:**
- Consume: la Tarea 8 (URLs del entorno) y la Tarea 12 (cuerpos redactados).
- Produce:
  - `cliente_http_compartido.crear_cliente_http() -> httpx.AsyncClient`, el único constructor del árbol de servicio.
  - `obtener_cliente_http() -> httpx.AsyncClient`, uno por event loop.
  - `async cerrar_cliente_http() -> None`.
  - `MemoryDB._http`: cliente propio, cerrado en `MemoryDB.close()`.

- [ ] **Paso 1: script de medición (se usa antes y después)**

```python
#!/usr/bin/env python3
"""E-24 (2026-09-16): medición antes/después del cliente HTTP compartido.

  embedding  el camino caliente REAL: MemoryDB.get_embedding contra el Ollama
             de hall9000, con el código del checkout que esté en PYTHONPATH
             (master para "antes", la rama para "después").
  aislado    el costo de abrir un cliente por llamada sin Ollama en el medio:
             un servidor HTTP/1.1 local con keep-alive que responde al
             instante, `por-llamada` contra `compartido`.

Solo lee: no escribe en la DB ni en disco. Percentiles con el mismo método que
el arnés de carga (scripts/load_test.py::percentil).

Uso:
  PYTHONPATH=<checkout> python scripts/medir_cliente_http_compartido.py embedding -c 1 -n 200
  python scripts/medir_cliente_http_compartido.py aislado --variante por-llamada -c 10 -n 2000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_test import percentil  # noqa: E402


async def medir(llamada, concurrencia: int, peticiones: int) -> dict:
    semaforo = asyncio.Semaphore(concurrencia)
    latencias: list[float] = []
    errores = 0

    async def una() -> None:
        nonlocal errores
        async with semaforo:
            t0 = time.perf_counter()
            try:
                ok = await llamada()
            except Exception:  # fail-soft: en una medición el error se CUENTA en `errores`, que sale en el resultado; no se esconde ni corta la corrida
                errores += 1
                return
            if ok:
                latencias.append(time.perf_counter() - t0)
            else:
                errores += 1

    inicio = time.perf_counter()
    await asyncio.gather(*(una() for _ in range(peticiones)))
    segundos = time.perf_counter() - inicio

    def ms(valor):
        return None if valor is None else round(valor * 1000, 2)

    return {
        "concurrencia": concurrencia, "peticiones": peticiones, "errores": errores,
        "rps": round(peticiones / segundos, 1),
        "p50_ms": ms(percentil(latencias, 50)), "p95_ms": ms(percentil(latencias, 95)),
        "p99_ms": ms(percentil(latencias, 99)),
    }


async def modo_embedding(concurrencia: int, peticiones: int) -> dict:
    from jax.memory.db import MemoryDB
    memoria = MemoryDB()
    texto = "medición E-24: cliente HTTP compartido en la memoria de JAX"

    async def llamada() -> bool:
        return await memoria.get_embedding(texto) is not None

    try:
        return await medir(llamada, concurrencia, peticiones)
    finally:
        await memoria.close()


class _Rapido(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        cuerpo = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def log_message(self, *args):
        return


async def modo_aislado(variante: str, concurrencia: int, peticiones: int) -> dict:
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), _Rapido)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{servidor.server_address[1]}/"
    compartido = httpx.AsyncClient() if variante == "compartido" else None

    async def llamada() -> bool:
        if compartido is not None:
            respuesta = await compartido.post(url, json={}, timeout=10)
        else:
            async with httpx.AsyncClient(timeout=10) as propio:
                respuesta = await propio.post(url, json={})
        return respuesta.status_code == 200

    try:
        return await medir(llamada, concurrencia, peticiones)
    finally:
        if compartido is not None:
            await compartido.aclose()
        servidor.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser(description="E-24: medición del cliente HTTP compartido")
    ap.add_argument("modo", choices=("embedding", "aislado"))
    ap.add_argument("--variante", choices=("por-llamada", "compartido"), default="compartido")
    ap.add_argument("-c", "--concurrencia", type=int, default=1)
    ap.add_argument("-n", "--peticiones", type=int, default=200)
    args = ap.parse_args()
    if args.modo == "embedding":
        resultado = asyncio.run(modo_embedding(args.concurrencia, args.peticiones))
    else:
        resultado = asyncio.run(modo_aislado(args.variante, args.concurrencia, args.peticiones))
    resultado["modo"] = args.modo if args.modo == "embedding" else f"aislado/{args.variante}"
    print(json.dumps(resultado, ensure_ascii=False))
    return 0 if resultado["errores"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Paso 2: medir ANTES, sobre el código de master**

Con 0 pipelines en vuelo: el embedding usa la GPU de producción, y una corrida concurrente ensucia los dos números.

```bash
git -C /home/fruiz/jax worktree add --detach /home/fruiz/worktrees/jax-medicion-antes bd95237
set -a; source <(grep -E '^JAX_MEMORY_EMBED_' /etc/jax/.env); set +a
cd /home/fruiz/worktrees/jax-medicion-antes
for args in "-c 1 -n 200" "-c 10 -n 500"; do
  PYTHONPATH=/home/fruiz/worktrees/jax-medicion-antes /home/fruiz/jax/.venv/bin/python /home/fruiz/worktrees/jax-frente-e/scripts/medir_cliente_http_compartido.py embedding $args
done | tee "$SCRATCH/e24-antes.jsonl"
for v in por-llamada compartido; do for args in "-c 1 -n 1000" "-c 10 -n 2000"; do
  /home/fruiz/jax/.venv/bin/python /home/fruiz/worktrees/jax-frente-e/scripts/medir_cliente_http_compartido.py aislado --variante $v $args
done; done | tee "$SCRATCH/e24-aislado.jsonl"
```
Esperado: 6 líneas JSON con `"errores": 0`. Si hay errores, **no seguir**: primero se entiende por qué.

- [ ] **Paso 3: tests que fallan**

```python
# tests/test_cliente_http_compartido.py
"""E-24 (2026-09-16): un httpx.AsyncClient por PROCESO (por event loop), no uno
por llamada. Política 2 de LAS CUATRO DEL RENDIMIENTO: los clientes HTTP se
comparten. Un cliente por llamada abre un socket (y un handshake TLS contra
los proveedores) por pedido: 14 sitios en Jacobs, LAS MANOS, el REPL y la memoria.

Uno por loop y no uno global: un AsyncClient queda atado al loop donde abrió
sus conexiones. Los servicios tienen un loop; los tests, uno por asyncio.run.
Ciclo de vida: cerrar_cliente_http() al apagar (LAS MANOS en shutdown, el REPL
y los workers en su finally); la memoria cierra el suyo en MemoryDB.close().
"""
from __future__ import annotations

import ast
import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import httpx

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from jax.core import cliente_http_compartido as chc  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
_PERMITIDO = (RAIZ / "jax" / "core" / "cliente_http_compartido.py").resolve()


def _construcciones(fuente: str) -> list[int]:
    lineas = []
    for n in ast.walk(ast.parse(fuente)):
        if isinstance(n, ast.Call):
            f = n.func
            nombre = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if nombre in ("AsyncClient", "Client"):
                lineas.append(n.lineno)
    return lineas


def test_ningun_modulo_de_servicio_construye_su_propio_cliente():
    hallazgos = []
    for arbol in ("jacobs", "jax", "las_manos"):
        for dirpath, dirnames, filenames in os.walk(RAIZ / arbol):
            dirnames[:] = [d for d in dirnames if d not in {".venv", "__pycache__", "tests"}]
            for nombre in filenames:
                p = Path(dirpath) / nombre
                if (p.suffix != ".py" or nombre.startswith("test_") or nombre.endswith("_test.py")
                        or p.is_symlink() or p.resolve() == _PERMITIDO):
                    continue
                hallazgos += [f"{p.relative_to(RAIZ)}:{l}" for l in _construcciones(p.read_text(encoding="utf-8"))]
    assert hallazgos == [], "usar obtener_cliente_http() (jax/core/cliente_http_compartido.py)"


def test_el_detector_ve_un_cliente_propio():
    assert _construcciones("import httpx\nasync def f():\n    async with httpx.AsyncClient() as c:\n        pass\n") == [3]


def test_dentro_de_un_loop_es_siempre_el_mismo_cliente():
    async def correr():
        a, b = chc.obtener_cliente_http(), chc.obtener_cliente_http()
        await chc.cerrar_cliente_http()
        return a, b
    a, b = asyncio.run(correr())
    assert a is b


def test_otro_loop_recibe_otro_cliente():
    async def uno():
        return chc.obtener_cliente_http()
    assert asyncio.run(uno()) is not asyncio.run(uno())


def test_cerrar_lo_cierra_y_el_siguiente_pedido_trae_uno_nuevo():
    async def correr():
        a = chc.obtener_cliente_http()
        await chc.cerrar_cliente_http()
        b = chc.obtener_cliente_http()
        await chc.cerrar_cliente_http()
        return a, b
    a, b = asyncio.run(correr())
    assert a.is_closed and a is not b


def test_cada_llamada_lleva_su_timeout():
    from jacobs import executor
    vistos = []

    async def handle(transport_self, request):
        vistos.append(request.extensions["timeout"])
        return httpx.Response(200, json={})

    async def correr():
        with patch("httpx.AsyncHTTPTransport.handle_async_request", handle):
            await executor._cancel_motor_job("job-x")

    asyncio.run(correr())
    assert vistos == [{"connect": 5, "read": 5, "write": 5, "pool": 5}]


def test_jacobs_reusa_el_cliente_entre_llamadas():
    from jacobs import executor
    clientes = []

    async def espia(client_self, request, **kwargs):
        clientes.append(client_self)
        return httpx.Response(200, json={}, request=request)

    async def correr():
        with patch.object(httpx.AsyncClient, "send", espia):
            await executor._cancel_motor_job("a")
            await executor._cancel_motor_job("b")

    asyncio.run(correr())
    assert len(clientes) == 2 and clientes[0] is clientes[1]


def test_la_memoria_reusa_su_cliente_y_lo_cierra(monkeypatch):
    from jax.memory import db as dbmod
    monkeypatch.setenv("JAX_OLLAMA_URL", "http://ollama.test:11434")
    clientes = []

    async def espia(client_self, request, **kwargs):
        clientes.append(client_self)
        return httpx.Response(200, json={"embeddings": [[0.1] * dbmod.EMBED.dim]}, request=request)

    async def correr():
        memoria = dbmod.MemoryDB()
        with patch.object(httpx.AsyncClient, "send", espia):
            await memoria.get_embedding("uno")
            await memoria.get_embedding("dos")
        await memoria.close()

    asyncio.run(correr())
    assert len(clientes) == 2 and clientes[0] is clientes[1] and clientes[0].is_closed


def test_las_manos_cierra_el_cliente_al_apagar():
    arbol = ast.parse((RAIZ / "las_manos" / "server.py").read_text(encoding="utf-8"))
    cierres = [
        f for f in ast.walk(arbol) if isinstance(f, ast.AsyncFunctionDef)
        and any(isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "on_event" and d.args
                and getattr(d.args[0], "value", None) == "shutdown" for d in f.decorator_list)
    ]
    assert any("cerrar_cliente_http" in ast.unparse(f) for f in cierres)


def test_el_repl_y_los_workers_cierran_el_cliente_al_salir():
    for rel in ("jax/core/main.py", "jax/memory/worker.py", "jax/memory/synthesis_worker.py", "jax/memory/embedding_worker.py"):
        assert "await cerrar_cliente_http()" in (RAIZ / rel).read_text(encoding="utf-8"), rel
```

Run: `$PYTEST_JAX -v tests/test_cliente_http_compartido.py`
Esperado: error de colección por `ModuleNotFoundError: jax.core.cliente_http_compartido`, o sea 10 en rojo.

- [ ] **Paso 4: el módulo y su symlink**

```python
# jax/core/cliente_http_compartido.py
"""Cliente HTTP compartido por proceso (E-24, 2026-09-16).

Política 2 de LAS CUATRO DEL RENDIMIENTO: los clientes HTTP se comparten, no se
rehacen por request. Un httpx.AsyncClient por llamada abre un socket nuevo (y un
handshake TLS contra los proveedores) en cada pedido.

Uno por EVENT LOOP, no uno global: un AsyncClient queda atado al loop donde abrió
sus conexiones. Los servicios (uvicorn, REPL, workers) tienen un solo loop; los
tests traen uno por asyncio.run, y reusar el cliente de un loop cerrado revienta
con "Event loop is closed". Se guarda el último par (loop, cliente): si el loop
cambió o el cliente se cerró, se crea otro.

Invalidación / ciclo de vida: `cerrar_cliente_http()` al apagar (LAS MANOS en
shutdown; REPL y workers en su finally). El timeout lo pone CADA llamada; el del
cliente queda en el default de httpx (5 s), así que una llamada que lo olvide cae
en 5 s y no en "sin límite".

Un solo archivo real: las_manos/cliente_http_compartido.py es symlink.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio

import httpx

_loop: asyncio.AbstractEventLoop | None = None
_cliente: httpx.AsyncClient | None = None


def crear_cliente_http() -> httpx.AsyncClient:
    """ÚNICO lugar del árbol de servicio que construye un AsyncClient (lo vigila
    tests/test_cliente_http_compartido.py). MemoryDB lo usa para el suyo propio."""
    return httpx.AsyncClient()


def obtener_cliente_http() -> httpx.AsyncClient:
    global _loop, _cliente
    loop = asyncio.get_running_loop()
    if _cliente is None or _cliente.is_closed or _loop is not loop:
        _cliente = crear_cliente_http()
        _loop = loop
    return _cliente


async def cerrar_cliente_http() -> None:
    global _loop, _cliente
    if _cliente is not None and _loop is asyncio.get_running_loop():
        await _cliente.aclose()
    _cliente = None
    _loop = None
```

```bash
cd /home/fruiz/worktrees/jax-frente-e && ln -s ../jax/core/cliente_http_compartido.py las_manos/cliente_http_compartido.py && git add jax/core/cliente_http_compartido.py las_manos/cliente_http_compartido.py
```

- [ ] **Paso 5: migrar los sitios**

Import por contexto:
- `jacobs/executor.py`, `jacobs/plan.py`, `jacobs/reaper.py` y `las_manos/motor_registry/worker.py`: `from cliente_http_compartido import obtener_cliente_http`.
- `jax/muscles/base.py` y `jax/muscles/ollama_muscle.py`: `from jax.core.cliente_http_compartido import obtener_cliente_http`.
- `jax/core/grounding_sources.py`: `try: from cliente_http_compartido import obtener_cliente_http` / `except ImportError: from jax.core.cliente_http_compartido import obtener_cliente_http`, con el comentario de doble camino como en `facet_resolver.py`.
- `jax/memory/db.py`: `from jax.core.cliente_http_compartido import crear_cliente_http`.

Cada `async with httpx.AsyncClient(timeout=T) as client:` que envuelve **una** llamada pasa a una llamada directa con `timeout=T`. El cuerpo se desindenta un nivel:

- `executor._invoke_http_gemini._call`: `resp = await obtener_cliente_http().post(url, headers=headers, json=payload, timeout=timeout)`
- `executor._invoke_http_openai_compat`: `resp = await obtener_cliente_http().post(url, headers=headers, json=payload, timeout=timeout)`
- `executor._invoke_ollama`: `resp = await obtener_cliente_http().post(OLLAMA_URL, json=payload, timeout=timeout)`
- `executor._invoke_motor` (dispatch): `resp = await obtener_cliente_http().post(f"{LAS_MANOS_BASE}/motor/dispatch", json=payload, timeout=30)`
- `executor._invoke_motor` (poll): `resp = await obtener_cliente_http().get(f"{LAS_MANOS_BASE}/motor/job/{job_id}", timeout=15)`
- `executor._cancel_motor_job`: `resp = await obtener_cliente_http().post(f"{LAS_MANOS_BASE}/motor/job/{job_id}/cancel", timeout=5)`
- `plan._ada_plan`: `async with obtener_cliente_http().stream("POST", url, headers=headers, json=payload, timeout=ADA_TIMEOUT) as resp:`
- `plan._llm_plan`: `resp = await obtener_cliente_http().post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)`
- `reaper.send_telegram_alert`: `resp = await obtener_cliente_http().post(f"https://api.telegram.org/bot{token}/sendMessage", data={"chat_id": chat_id, "text": message}, timeout=10.0)`. Esta URL de la API de Telegram es de un proveedor externo, no configuración de servicio. Queda, y `reaper.py` no entra en el test de URLs literales de la Tarea 8.
- `worker._call_http_openai_compat`: `response = await obtener_cliente_http().post(f"{api_url}/chat/completions", json=payload, headers=headers, timeout=timeout)`
- `base._call_deepseek`: `resp = await obtener_cliente_http().post(url, headers=headers, json=payload, timeout=self.timeout)`
- `base._call_openai`: `async with obtener_cliente_http().stream("POST", url, headers=headers, json=payload, timeout=self.timeout) as resp:`
- `base._call_gemini._request`: `resp = await obtener_cliente_http().post(url, headers=headers, json=payload, timeout=self.timeout)`
- `ollama_muscle._call`: `resp = await obtener_cliente_http().post(self.api_url, json=payload, timeout=self.timeout)`, siempre dentro de `async with GPU_SEMAPHORE:`.

`grounding_sources`:
```python
async def resolve_redirects(sources: list[dict], client: httpx.AsyncClient | None = None) -> list[dict]:
    """Agrega `final_url` y `resolved` a cada fuente, todas en paralelo, y
    fusiona las que resultan ser el mismo documento (ver _merge_by_final_url).
    `client` es para tests (MockTransport); en producción, el cliente
    compartido del proceso (E-24)."""
    if not sources:
        return sources
    cliente = client if client is not None else obtener_cliente_http()
    await asyncio.gather(*(_resolve_one(cliente, s) for s in sources))
    _merge_by_final_url(sources)
    return sources
```
En `_resolve_one`, pasar `timeout=RESOLVE_TIMEOUT_SECONDS` a `client.head(...)` y a `client.stream("GET", ...)`.

`jax/memory/db.py`:
- En `MemoryDB.__init__`: `self._http: Optional[httpx.AsyncClient] = None  # E-24: cliente propio, cerrado en close()`.
- En `close()`, antes de `if self.pool:`:
```python
        if self._http is not None:
            await self._http.aclose()
            self._http = None
```
- En `get_embedding`, dentro del `try`, reemplazar el `async with httpx.AsyncClient(timeout=10.0) as client:` por:
```python
            if self._http is None or self._http.is_closed:
                self._http = crear_cliente_http()
            resp = await self._http.post(
                url,
                json={"model": EMBED.model, "input": texto},
                timeout=10.0,
            )
```
  y desindentar el resto.

Cierres:
- `las_manos/server.py`, después de `_jacobs_init`:
```python
@app.on_event("shutdown")
async def _cerrar_cliente_http() -> None:
    """E-24: el cliente HTTP compartido del proceso se cierra al apagar."""
    from cliente_http_compartido import cerrar_cliente_http
    await cerrar_cliente_http()
```
- `jax/core/main.py`: import `from jax.core.cliente_http_compartido import cerrar_cliente_http`. En el `finally` de `run_task`, después de `await voice.shutdown()`, agregar `await cerrar_cliente_http()`; en el `finally` de `main()`, después de `await db.close()`, lo mismo.
- `jax/memory/worker.py` y `jax/memory/synthesis_worker.py`: import igual, y `await cerrar_cliente_http()` después de `await db.close()` en su `finally`.
- `jax/memory/embedding_worker.py`: lo mismo después de `await db.close()` en `main()`.

Imports de `httpx` que quedan sin uso: listarlos y borrar los que salgan.
```bash
cd /home/fruiz/worktrees/jax-frente-e && /home/fruiz/jax/.venv/bin/python - <<'PY'
import ast, pathlib
for rel in ["jacobs/executor.py","jacobs/plan.py","jacobs/reaper.py","las_manos/motor_registry/worker.py","jax/core/grounding_sources.py","jax/muscles/base.py","jax/muscles/ollama_muscle.py","jax/memory/db.py"]:
    t = ast.parse(pathlib.Path(rel).read_text())
    usado = any(isinstance(n, ast.Name) and n.id == "httpx" for n in ast.walk(t))
    print(rel, "httpx usado" if usado else "httpx SIN USO: borrar el import")
PY
```

- [ ] **Paso 6: verde, y todo lo que parchea httpx**

Run:
```bash
$PYTEST_JAX -v tests/test_cliente_http_compartido.py
$PYTEST_JAX -q tests/test_hipatia_fuentes.py tests/test_fuentes_dedup.py tests/test_repl_fuentes.py tests/test_contrato_dispatch_repl_ada.py \
  tests/test_motor_contrato_dispatch.py tests/test_motor_job_cancel_and_length.py tests/test_motor_contexto_y_salida.py \
  tests/test_invoke_motor_rechazo.py tests/test_invoke_motor_cancel_on_timeout.py tests/test_gemini_key_en_cabecera.py \
  tests/test_memory_embedding_config.py tests/test_errores_de_proveedor_redactados.py tests/test_config_entorno.py \
  tests/test_no_blocking_in_async.py las_manos/_worker_max_tokens_test.py las_manos/_worker_tool_loop_test.py 2>&1 | tail -2
cd /home/fruiz/worktrees/jax-frente-e && env JAX_PLATFORM_REPO_ROOT=/home/fruiz/worktrees/jax-platform-frente-e /home/fruiz/jax/.venv/bin/python -m pytest -q policy/tests/test_no_fail_open_except.py 2>&1 | tail -1
```
Esperado: 10 PASS, el resto verde y P10 verde. Si un falso de un test existente falla por el `timeout=` nuevo, se le agrega
`**kw` a su firma. Medido el 2026-09-16: todos los `fake_post`/`fake_get`/`fake_stream` del árbol ya aceptan `**kw`.

- [ ] **Paso 7: medir DESPUÉS**

```bash
set -a; source <(grep -E '^JAX_MEMORY_EMBED_' /etc/jax/.env); set +a
cd /home/fruiz/worktrees/jax-frente-e
for args in "-c 1 -n 200" "-c 10 -n 500"; do
  JAX_OLLAMA_URL=http://localhost:11434 PYTHONPATH=/home/fruiz/worktrees/jax-frente-e /home/fruiz/jax/.venv/bin/python scripts/medir_cliente_http_compartido.py embedding $args
done | tee "$SCRATCH/e24-despues.jsonl"
git -C /home/fruiz/jax worktree remove /home/fruiz/worktrees/jax-medicion-antes
```

Regla:
- Los números se anotan tal cual en la Tarea 18.
- Si en `embedding` el p95 **después** es peor que **antes** en más de un 10 % a igual concurrencia, **parar y reportar**
  con los dos JSON.
- Si queda igual dentro del 10 %, se registra como "sin ganancia medible en este camino: Ollama local serializa y domina
  la latencia". La justificación del cambio queda con la medición `aislado` y con la regla.
- No se inventa una mejora.

- [ ] **Paso 8: CI y commit**

`tests/test_cliente_http_compartido.py` va a las dos listas de `tests-puros`.

```bash
cd /home/fruiz/worktrees/jax-frente-e && git add -A jax/core/cliente_http_compartido.py las_manos/cliente_http_compartido.py scripts/medir_cliente_http_compartido.py \
  tests/test_cliente_http_compartido.py jacobs/executor.py jacobs/plan.py jacobs/reaper.py las_manos/motor_registry/worker.py \
  jax/core/grounding_sources.py jax/muscles/base.py jax/muscles/ollama_muscle.py jax/memory/db.py las_manos/server.py jax/core/main.py \
  jax/memory/worker.py jax/memory/synthesis_worker.py jax/memory/embedding_worker.py .github/workflows/policy.yml
git commit -m "perf(E-24): cliente HTTP compartido por proceso con cierre al apagar; timeouts por llamada; tripwire de clase" \
  -m "Antes/despues en el embedding (c=1 y c=10) y aislado: ver DEUDA.md. 8 tests vistos en rojo; 2 controles." \
  -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: E-25 · criterio de salida B1.4: medir hoy, retirar solo si el gate da limpio

**Archivos, sólo si el gate da A (Paso 6):**
- jax:
  - `jax/core/credential_resolver.py`, borrando `_PROVIDER_ENV_KEY_MAP` y `resolve_credential_instrumented`.
  - Renombrar los llamadores: `jax/core/facet_resolver.py`, `jax/muscles/base.py`, `las_manos/motor_registry/worker.py`, `las_manos/server.py` (comentario del logger), `scripts/check_mirror_sync.py` (`compartidos`).
  - Los tests que lo parchean: `las_manos/_worker_max_tokens_test.py`, `las_manos/_worker_tool_loop_test.py`, `tests/test_contrato_dispatch_repl_ada.py`, `tests/test_motor_contexto_y_salida.py`, `tests/test_motor_contrato_dispatch.py`, `tests/test_motor_job_cancel_and_length.py`.
  - Crear `tests/test_credenciales_sin_fallback_env.py`.
  - Borrar `ops/check-b14-exit-criterion.sh`.
- jax-platform:
  - `backend/credential_resolver.py`, `backend/facet_resolver.py`, `backend/api/chat.py`, `backend/api/image.py`, `backend/main.py`, `backend/model_catalog.py`, `backend/db/migrations.py` (comentario).
  - Los tests que lo nombran (`git grep`).
  - Crear `backend/tests/test_credenciales_sin_fallback_env.py`.
- Sistema: `/etc/systemd/system/check-b14-exit-criterion.{timer,service}`.

**Interfaces:**
- Consume: nada.
- Produce: en jax, `$SCRATCH/e25-veredicto.txt`. Si el gate da A, `resolve_credential(provider_id) -> str` es la única entrada en los dos repos.

- [ ] **Paso 1: cobertura del journal (sólo lectura)**

```bash
for u in jax-platform jax-las-manos jax-memory-worker jax-memory-synthesis; do
  printf "%s primera=" "$u"; sudo journalctl -u "$u.service" --since "7 days ago" --no-pager -o short-iso | sed -n 2p | cut -c1-25
done | tee "$SCRATCH/e25-cobertura.txt"
date -Iseconds -d "7 days ago" | tee -a "$SCRATCH/e25-cobertura.txt"
```
Una unidad cubre la ventana si su primera línea está a menos de 1 h del instante de hace 7 días, o si la unidad no tiene
líneas porque no corrió. El caso de jax-las-manos y jax-platform, que corren siempre, se decide a mano: sin líneas tempranas
**no cubren**.

- [ ] **Paso 2: conteos**

```bash
sudo journalctl -u jax-platform.service -u jax-las-manos.service -u jax-memory-worker.service -u jax-memory-synthesis.service \
  --since "7 days ago" --no-pager -o short-iso > "$SCRATCH/e25-journal.txt"
{
  echo "credential_resolution=$(grep -c 'credential_resolution' "$SCRATCH/e25-journal.txt")"
  echo "source_db=$(grep 'credential_resolution' "$SCRATCH/e25-journal.txt" | grep -c 'source=db')"
  echo "env_fallback=$(grep 'credential_resolution' "$SCRATCH/e25-journal.txt" | grep -c 'source=env_fallback')"
  grep 'source=env_fallback' "$SCRATCH/e25-journal.txt" | grep -oP 'provider=\K[a-z_]+' | sort | uniq -c
  grep 'source=env_fallback' "$SCRATCH/e25-journal.txt" | awk '{print $1, $3}' | sort | uniq -c | tail -20
} | tee "$SCRATCH/e25-conteos.txt"
```

- [ ] **Paso 3: rotaciones y credenciales activas (DB de producción, sólo SELECT)**

```bash
set -a; source <(grep -E '^JAX_DB_(HOST|PORT|USER|PASSWORD|NAME)=' /etc/jax/.env); set +a
/home/fruiz/jax/.venv/bin/python - <<'PY' | tee "$SCRATCH/e25-db.txt"
import asyncio, os, aiomysql
async def main():
    conn = await aiomysql.connect(host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.environ["JAX_DB_USER"], password=os.environ["JAX_DB_PASSWORD"], db=os.environ["JAX_DB_NAME"],
        connect_timeout=10)
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT provider_id, state, activated_at FROM credential "
                              "WHERE activated_at >= NOW() - INTERVAL 7 DAY ORDER BY activated_at")
            print("ROTACIONES_7D", await cur.fetchall())
            await cur.execute("SELECT provider_id, COUNT(*) FROM credential WHERE state='active' GROUP BY provider_id")
            print("ACTIVAS", await cur.fetchall())
    finally:
        conn.close()
asyncio.run(main())
PY
systemctl list-timers --all --no-pager | grep -i b14 | tee -a "$SCRATCH/e25-db.txt"
```

- [ ] **Paso 4: veredicto (regla de decisión)**

Escribir en `$SCRATCH/e25-veredicto.txt` una sola de estas tres líneas, con los números.

- **A: RETIRAR.** Tienen que cumplirse las cinco condiciones:
  1. jax-platform y jax-las-manos cubren la ventana (Paso 1);
  2. `source_db > 0`, porque el instrumento registra: un control que no ve nada no valida;
  3. `env_fallback == 0`;
  4. `ROTACIONES_7D` tiene al menos una fila;
  5. `ACTIVAS` incluye `openai`, `deepseek`, `gemini`, `moonshot` y `zhipu`. Esto cubre al REPL, que no está en el journal (Discrepancia 5).
- **B: CONSUMIDOR EN FALLBACK.** `env_fallback > 0`. Se reportan a Fernando el o los proveedores, la unidad y los timestamps del Paso 2. **No se toca nada.**
- **C: CRITERIO NO MEDIBLE O NO CUMPLIDO.** Cualquier otro caso: sin cobertura, `source_db == 0`, sin rotación o falta una credencial activa. Se reporta a Fernando con los números. **No se toca nada.**

- [ ] **Paso 5: si el veredicto es B o C, la tarea termina acá**

Se reporta a Fernando: veredicto, los tres archivos de evidencia y el estado del timer vencido. El retiro del timer y del
script también queda a su decisión, porque son el instrumento del criterio.

- [ ] **Paso 6: si el veredicto es A, confirmar con Fernando en una línea**

Mensaje:

"E-25: 7 días con N lecturas `source=db` y 0 `env_fallback`, rotación en `<proveedor>`, `<fecha>`, y los 5 proveedores con
credencial activa. El spec de jax-platform (§A.5) lista el fallback B1.4 entre lo que se conserva; el de jax (E-25) lo
retira con este resultado. ¿Retiro en los dos repos?"

Con **sí**, se sigue al Paso 7. Con cualquier otra respuesta, la tarea termina y queda registrada.

- [ ] **Paso 7 (solo A + sí): tests que fallan, en los dos repos**

```python
# tests/test_credenciales_sin_fallback_env.py  (jax)
"""E-25 (2026-09-16): retiro del fallback de credenciales a variables de
entorno (B1.4). Criterio de salida medido el <fecha>: <N> source=db, 0
env_fallback en 7 días, rotación de <proveedor>, 5 proveedores con credencial
activa. Veredicto y evidencia en DEUDA.md."""
from __future__ import annotations

import ast
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def _servicio():
    for arbol in ("jacobs", "jax", "las_manos"):
        for dirpath, dirnames, filenames in os.walk(RAIZ / arbol):
            dirnames[:] = [d for d in dirnames if d not in {".venv", "__pycache__", "tests"}]
            for nombre in filenames:
                p = Path(dirpath) / nombre
                if nombre.endswith(".py") and not nombre.endswith("_test.py") and not p.is_symlink():
                    yield p


def test_el_resolver_ya_no_tiene_fallback_a_entorno():
    from jax.core import credential_resolver as cr
    assert not hasattr(cr, "resolve_credential_instrumented")
    assert not hasattr(cr, "_PROVIDER_ENV_KEY_MAP")


def test_nadie_llama_al_resolver_instrumentado():
    hallazgos = [str(p.relative_to(RAIZ)) for p in _servicio()
                 if any(isinstance(n, ast.Name) and n.id == "resolve_credential_instrumented"
                        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))))]
    assert hallazgos == []


def test_ningun_modulo_registra_env_fallback():
    assert [str(p.relative_to(RAIZ)) for p in _servicio() if "env_fallback" in p.read_text(encoding="utf-8")] == []
```

```python
# backend/tests/test_credenciales_sin_fallback_env.py  (jax-platform)
"""E-25 (2026-09-16): espejo del retiro de jax. Ver jax/DEUDA.md."""
import ast
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def test_el_resolver_ya_no_tiene_fallback_a_entorno():
    import credential_resolver as cr
    assert not hasattr(cr, "resolve_credential_instrumented")
    assert not hasattr(cr, "_PROVIDER_ENV_KEY_MAP")


def test_nadie_llama_al_resolver_instrumentado():
    hallazgos = []
    for p in BACKEND.rglob("*.py"):
        if ".venv" in p.parts or "tests" in p.parts:
            continue
        arbol = ast.parse(p.read_text(encoding="utf-8"))
        if any(isinstance(n, ast.Name) and n.id == "resolve_credential_instrumented" for n in ast.walk(arbol)):
            hallazgos.append(str(p.relative_to(BACKEND)))
    assert hallazgos == []
```

Run: `$PYTEST_JAX -v tests/test_credenciales_sin_fallback_env.py` y `$PYTEST_PLAT -v tests/test_credenciales_sin_fallback_env.py`. Esperado: 3 FAIL y 2 FAIL.

- [ ] **Paso 8 (solo A + sí): retirar primero la definición y después renombrar a los llamadores**

En `jax/core/credential_resolver.py` **y** en `jax-platform/backend/credential_resolver.py`:
- Borrar el bloque `_PROVIDER_ENV_KEY_MAP = {...}`, su comentario y la función `resolve_credential_instrumented` completa.
- En el docstring de `resolve_credential`, agregar: `Única entrada desde 2026-09-16 (E-25): el fallback a variables de entorno (B1.4) se retiró con su criterio de salida medido.`

Recién después:
```bash
cd /home/fruiz/worktrees/jax-frente-e && git grep -l "resolve_credential_instrumented" | xargs sed -i 's/resolve_credential_instrumented/resolve_credential/g'
cd /home/fruiz/worktrees/jax-platform-frente-e && git grep -l "resolve_credential_instrumented" | xargs sed -i 's/resolve_credential_instrumented/resolve_credential/g'
git -C /home/fruiz/worktrees/jax-frente-e grep -n "env_fallback\|_PROVIDER_ENV_KEY_MAP"; git -C /home/fruiz/worktrees/jax-platform-frente-e grep -n "env_fallback\|_PROVIDER_ENV_KEY_MAP" -- backend
```
- Cada línea del último grep se reescribe a mano: comentarios (`las_manos/server.py:234-238`, `db/migrations.py`) y docstrings. Tienen que decir que el logger `credential_resolver` sigue en INFO por `serving_stale` y `FAIL_CLOSED`.
- `scripts/check_mirror_sync.py`: quitar `"_PROVIDER_ENV_KEY_MAP"` de `compartidos` y el párrafo de comentario que lo explica. Quitar también el `resolve_credential` duplicado que deja el sed.
- Revisar que ningún import quedó con `resolve_credential, resolve_credential` duplicado:
  `git grep -n "resolve_credential, resolve_credential"` en los dos worktrees, y corregirlo.

- [ ] **Paso 9 (solo A + sí): verde en los dos repos y el checker de espejos**

Run:
```bash
$PYTEST_JAX -q tests/test_credenciales_sin_fallback_env.py las_manos/_worker_max_tokens_test.py las_manos/_worker_tool_loop_test.py \
  tests/test_contrato_dispatch_repl_ada.py tests/test_motor_contexto_y_salida.py tests/test_motor_contrato_dispatch.py tests/test_motor_job_cancel_and_length.py 2>&1 | tail -2
cd /home/fruiz/worktrees/jax-frente-e && JAX_PLATFORM_REPO_ROOT=/home/fruiz/worktrees/jax-platform-frente-e /home/fruiz/jax/.venv/bin/python scripts/check_mirror_sync.py; echo "exit=$?"
$PYTEST_PLAT -q 2>&1 | tail -3
```
Esperado: verde, `exit=0`, y la suite de jax-platform sin fallos nuevos contra su línea base.

- [ ] **Paso 10 (solo A + sí): timer y script fuera, con copia verificada**

```bash
mkdir -p /home/fruiz/backups/check-b14-units-20260916
sudo cp -a /etc/systemd/system/check-b14-exit-criterion.timer /etc/systemd/system/check-b14-exit-criterion.service /home/fruiz/backups/check-b14-units-20260916/
cmp /etc/systemd/system/check-b14-exit-criterion.timer /home/fruiz/backups/check-b14-units-20260916/check-b14-exit-criterion.timer && \
cmp /etc/systemd/system/check-b14-exit-criterion.service /home/fruiz/backups/check-b14-units-20260916/check-b14-exit-criterion.service && echo COPIA_OK
```
Sólo con `COPIA_OK`:
```bash
sudo systemctl disable --now check-b14-exit-criterion.timer
sudo rm /etc/systemd/system/check-b14-exit-criterion.timer /etc/systemd/system/check-b14-exit-criterion.service
sudo systemctl daemon-reload && systemctl list-timers --all --no-pager | grep -ci b14
cd /home/fruiz/worktrees/jax-frente-e && git rm ops/check-b14-exit-criterion.sh
```
Esperado: el último conteo da `0`. El contenido de las dos unidades se copia también en la entrada de DEUDA de la Tarea 18:
`/home/fruiz/backups` no está en restic.

- [ ] **Paso 11 (solo A + sí): CI y commits**

`tests/test_credenciales_sin_fallback_env.py` va a las dos listas de `tests-puros`.

```bash
cd /home/fruiz/worktrees/jax-frente-e && git add -A && git commit -m "fix(E-25): retirar el fallback de credenciales a entorno (B1.4) con su criterio de salida medido" \
  -m "<N> source=db, 0 env_fallback en 7 dias, rotacion <proveedor>, 5 credenciales activas. Confirmado por Fernando." \
  -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
cd /home/fruiz/worktrees/jax-platform-frente-e && git add -A && git commit -m "fix(E-25): espejo del retiro del fallback de credenciales a entorno" \
  -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: E-26 · los 25 backups sin trackear: borrar solo los idénticos a un blob de git

**Archivos:** ningún archivo trackeado. Se trabaja sobre `/home/fruiz/jax` (checkout de producción): sólo lectura, más
`rm` de archivos **ignorados** después del gate.

**Interfaces:**
- Consume: nada.
- Produce: `$SCRATCH/e26-veredicto.txt`, con una línea `IDENTICO|UNICO <sha> <ruta>` por archivo.

- [ ] **Paso 1: inventario y veredicto por hash (sólo lectura)**

```bash
cd /home/fruiz/jax
git status --short --ignored | awk '/^!! / && /\.backup-pre-/ {print substr($0, 4)}' > "$SCRATCH/e26-backups.txt"
wc -l < "$SCRATCH/e26-backups.txt"
git rev-list --all --objects | awk '{print $1}' | sort -u > "$SCRATCH/e26-objetos.txt"
while IFS= read -r f; do
  h=$(git hash-object -- "$f")
  if grep -qx "$h" "$SCRATCH/e26-objetos.txt"; then
    c=$(git log --all --format=%h -1 --find-object="$h")
    echo "IDENTICO $h $f commit=$c"
  else
    echo "UNICO $h $f"
  fi
done < "$SCRATCH/e26-backups.txt" | tee "$SCRATCH/e26-veredicto.txt"
grep -c '^IDENTICO' "$SCRATCH/e26-veredicto.txt"; grep -c '^UNICO' "$SCRATCH/e26-veredicto.txt"
```
Esperado: 25 archivos en el inventario (2026-09-16: 14 en `jacobs/` y 11 en `las_manos/`). Si el conteo es otro, se anota y
se sigue con el que dio.

- [ ] **Paso 2: regla de decisión**

- `IDENTICO`: el contenido es un blob alcanzable en la historia de git, así que tiene copia verificada. Se puede borrar.
- `UNICO`: no se borra. Se lista a Fernando con, para cada uno, `wc -l` y la diferencia contra el archivo vivo:
  ```bash
  while read -r _ h f; do base="${f%%.backup-pre-*}"; echo "== $f"; diff <(git show "HEAD:$base" 2>/dev/null) "$f" | head -20; done < <(grep '^UNICO' "$SCRATCH/e26-veredicto.txt")
  ```

- [ ] **Paso 3: borrar solo los IDÉNTICOS, re-verificando cada uno justo antes**

```bash
cd /home/fruiz/jax
grep '^IDENTICO' "$SCRATCH/e26-veredicto.txt" | while read -r _ h f _; do
  if [ "$(git hash-object -- "$f")" = "$h" ] && git cat-file -p "$h" | cmp -s - "$f"; then
    rm -- "$f" && echo "BORRADO $f"
  else
    echo "SALTADO (cambió o no coincide) $f"
  fi
done | tee "$SCRATCH/e26-borrados.txt"
git -C /home/fruiz/jax status --short --ignored | grep -c '\.backup-pre-'
```
Esperado: una línea `BORRADO` por cada `IDENTICO`, ningún `SALTADO`, y el conteo final igual al número de `UNICO`.

- [ ] **Paso 4: reporte a Fernando**

Mensaje: cuántos se borraron, con commit de origen por archivo (del Paso 1), y la lista de los `UNICO` con sus diferencias.
La decisión sobre esos es de Fernando. Todo queda en la Tarea 18.

---

### Task 16: pisos de CI medidos en el runner y canario

**Archivos:**
- Modificar: `.github/workflows/policy.yml` (líneas `grep -qE` y comentarios de los pisos de `tests-puros` y `jacobs-gobernanza-db`)
- Modificar, temporal: `tests/test_cliente_http_compartido.py` y `tests/test_facetas_de_gobernanza_db.py`, rotos a propósito y revertidos

**Interfaces:**
- Consume: las Tareas 1 a 14.
- Produce: pisos exactos medidos y la evidencia del rojo en el sha real.

- [ ] **Paso 1: suma esperada (solo para comparar, NUNCA para fijar el piso)**

`tests-puros`, partiendo de `494 passed, 1 skipped`:

| Tarea | Tests |
|---|---|
| T1 | +2 |
| T2, T3, T4 | +9 |
| T5 | +4 |
| T6 | +7 |
| T7 | +8 |
| T8 | +16 |
| T9 | +3 |
| T10 | +5 |
| T11 | +5 |
| T12 | +12 |
| T13 | +10 |
| **Total** | **+81 → `575 passed, 1 skipped`** |

- Con E-25 retirado: +3 → `578 passed, 1 skipped`.
- `jacobs-gobernanza-db`: 27 + 2 = `29 passed`.
- `governance`: `90`. `plan-timeout-ceiling`: `16`. `mirror-sync`: `14` y `15`. `facet-resolver-seal`: `13`. Ninguno cambia.
- `no-fail-open-except` no tiene piso: su corrida tiene que mostrar 2 tests más que en la línea base.

- [ ] **Paso 2: push y conteo del runner**

```bash
cd /home/fruiz/worktrees/jax-frente-e && git push -u origin fix/hallazgos-frente-e
cd /home/fruiz/worktrees/jax-platform-frente-e && git push -u origin fix/hallazgos-frente-e
RUN=$(gh run list -R fjruizhn/Jax --branch fix/hallazgos-frente-e --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN" -R fjruizhn/Jax --exit-status; gh run view "$RUN" -R fjruizhn/Jax --log > "$SCRATCH/ci-jax.log"
grep -E "^tests-puros.*[0-9]+ passed|^jacobs-gobernanza-db.*[0-9]+ passed|^no-fail-open-except.*[0-9]+ passed|PISO ROTO" "$SCRATCH/ci-jax.log" | tail -20
```
Regla:
- Si el runner cuenta lo mismo que el Paso 1, se fija ese número.
- Si cuenta **otro**, **no se ajusta**: primero se encuentra la diferencia (un test que se saltó, un archivo que falta en
  una lista, otro frente que mergeó y rebaseó) y se explica.
- Un `2 skipped` en `tests-puros` es un defecto que se arregla, no un piso nuevo.

- [ ] **Paso 3: fijar los pisos con el número del runner**

En `tests-puros`:
```
          # 494 -> <N> el 2026-09-16, MEDIDO por el runner (frente E de la auditoria de jax):
          #   +2 E-01 (_NO_PARSEA ejercitado con un .py roto de mentira), +9 retiros
          #   (E-02/04/05/06/07/08/09/20), +4 E-13, +7 E-03/E-17, +8 E-10/E-11, +16 E-21,
          #   +3 E-12/E-22/E-18, +5 E-19, +5 E-15, +12 E-16, +10 E-24<, +3 E-25>. Todos
          #   vistos en rojo antes del arreglo salvo los controles declarados en cada
          #   archivo. E-14 no admite rojo (equivalente) y no tiene test.
          grep -qE "^<N> passed, 1 skipped" /tmp/out || {
            echo "PISO ROTO: se esperaban <N> tests CORRIDOS y exactamente 1 skipped."; exit 1; }
```
En `jacobs-gobernanza-db`:
```
          # 27 -> 29 el 2026-09-16 (E-03): tests/test_facetas_de_gobernanza_db.py (+2) --
          #   get_motor_governance trae las facetas active de la tabla facet, y una
          #   disabled no entra. MEDIDO por el runner.
          grep -qE "^29 passed" /tmp/out || {
            echo "PISO ROTO: se esperaban 29 tests CORRIDOS."; exit 1; }
```

```bash
cd /home/fruiz/worktrees/jax-frente-e && git add .github/workflows/policy.yml && git commit -m "ci: pisos de tests-puros y jacobs-gobernanza-db medidos en el runner (frente E)" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" && git push
RUN=$(gh run list -R fjruizhn/Jax --branch fix/hallazgos-frente-e --limit 1 --json databaseId --jq '.[0].databaseId'); gh run watch "$RUN" -R fjruizhn/Jax --exit-status
```
Esperado: todos los jobs en `success`.

- [ ] **Paso 4: canario: rojo en el sha real**

En `tests/test_cliente_http_compartido.py::test_dentro_de_un_loop_es_siempre_el_mismo_cliente` cambiar `assert a is b` por
`assert a is not b  # CANARIO: rojo a propósito`. En
`tests/test_facetas_de_gobernanza_db.py::test_las_facetas_son_las_activas_de_la_tabla` cambiar
`self.assertIn("hipatia", ...)` por `self.assertNotIn("hipatia", gobernanza["facets"])  # CANARIO`.

```bash
cd /home/fruiz/worktrees/jax-frente-e && git commit -am "test(canario): rojo a proposito para verificar que CI corre los tests nuevos" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" && git push
SHA=$(git rev-parse HEAD); RUN=$(gh run list -R fjruizhn/Jax --branch fix/hallazgos-frente-e --limit 1 --json databaseId,headSha --jq ".[] | select(.headSha==\"$SHA\") | .databaseId")
gh run watch "$RUN" -R fjruizhn/Jax || true
gh api "repos/fjruizhn/Jax/commits/$SHA/check-runs" --jq '.check_runs[] | select(.name=="tests-puros" or .name=="jacobs-gobernanza-db") | "\(.name) \(.conclusion)"'
```
Esperado: `tests-puros failure` y `jacobs-gobernanza-db failure`, sobre ese `$SHA`.

- [ ] **Paso 5: revertir el canario y volver a verde**

```bash
cd /home/fruiz/worktrees/jax-frente-e && git revert --no-edit HEAD && git push
SHA=$(git rev-parse HEAD); RUN=$(gh run list -R fjruizhn/Jax --branch fix/hallazgos-frente-e --limit 1 --json databaseId,headSha --jq ".[] | select(.headSha==\"$SHA\") | .databaseId")
gh run watch "$RUN" -R fjruizhn/Jax --exit-status
gh api "repos/fjruizhn/Jax/commits/$SHA/check-runs" --jq '.check_runs[] | "\(.name) \(.conclusion)"' | sort
```
Esperado: todas `success`. Anotar los dos `$SHA` (canario y revert) para la Tarea 18.

- [ ] **Paso 6: CI de jax-platform**

```bash
RUN=$(gh run list -R fjruizhn/jax-platform --branch fix/hallazgos-frente-e --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN" -R fjruizhn/jax-platform --exit-status
```
Esperado: verde. Si la Tarea 14 agregó `backend/tests/test_credenciales_sin_fallback_env.py`, repetir allá el canario de
los Pasos 4 y 5 sobre ese archivo, con `gh api repos/fjruizhn/jax-platform/commits/$SHA/check-runs`.

---

### Task 17: PRs, orden de merge, deploy con 0 pipelines en vuelo y verificación en vivo

**Archivos:** `/etc/jax/.env` (con backup). Checkouts de producción `/home/fruiz/jax` y `/home/fruiz/jax-platform`, sólo `pull --ff-only` sobre master.

**Interfaces:**
- Consume: las Tareas 1 a 16 en verde.
- Produce: producción en los merges del frente E, con evidencia en vivo.

- [ ] **Paso 1: PRs**

```bash
cd /home/fruiz/worktrees/jax-platform-frente-e && gh pr create -R fjruizhn/jax-platform --base master --head fix/hallazgos-frente-e \
  --title "Frente E (jax-platform): docstrings E-07/E-11, _call_ollama desde JAX_OLLAMA_URL (E-21)<, retiro B1.4 (E-25)>" --body-file "$SCRATCH/pr-plat.md"
cd /home/fruiz/worktrees/jax-frente-e && gh pr create -R fjruizhn/Jax --base master --head fix/hallazgos-frente-e \
  --title "Frente E: limpieza, defectos y reglas de jax (E-01..E-26)" --body-file "$SCRATCH/pr-jax.md"
```
Cada body tiene la tabla de ids con el test que lo cubre, las Discrepancias, los números de E-24 y los veredictos de E-25 y
E-26. Termina con `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

- [ ] **Paso 2: GO de Fernando**

Se pide GO para merge y deploy con un resumen de 5 líneas: qué cambia en vivo (variables nuevas en `.env`, config.toml sin
URLs, planes con faceta desconocida pasan a 422) y el orden del Paso 4. Sin GO no se sigue.

- [ ] **Paso 3: variables en `/etc/jax/.env`, con backup verificado**

Los valores se verifican antes de escribirlos:
```bash
test -x /home/fruiz/kokoro-test/.venv/bin/python && echo KOKORO_OK
test -d /home/fruiz/jax/repo/documents && echo DOCS_OK
grep -E '^LAS_MANOS_URL=' /etc/jax/.env
grep -cE '^(JAX_OLLAMA_URL|JAX_KOKORO_PYTHON|JAX_REPO_BASE_DIR)=' /etc/jax/.env
```
Esperado: `KOKORO_OK`, `DOCS_OK`, `LAS_MANOS_URL=http://127.0.0.1:7777` y `0`. Si `LAS_MANOS_URL` tiene otro valor, parar y
preguntar. Si el conteo no es 0, otro frente ya las agregó: comparar valores y no duplicar.

Como script, porque la terminal parte los pegados largos:
```bash
cat > "$SCRATCH/e-env.sh" <<'SH'
#!/bin/bash
set -euo pipefail
destino=/etc/jax/.env
copia="/etc/jax/.env.backup-pre-frente-e-$(date +%Y%m%d-%H%M%S)"
cp -a "$destino" "$copia"
cmp "$destino" "$copia"
cat >> "$destino" <<'EOF'
# Frente E (2026-09-16): URLs y rutas de servicio fuera del código (E-21, E-22)
JAX_OLLAMA_URL=http://localhost:11434
JAX_KOKORO_PYTHON=/home/fruiz/kokoro-test/.venv/bin/python
JAX_REPO_BASE_DIR=/home/fruiz/jax/repo
EOF
grep -cE '^(JAX_OLLAMA_URL|JAX_KOKORO_PYTHON|JAX_REPO_BASE_DIR)=' "$destino"
echo "backup: $copia"
SH
sudo bash "$SCRATCH/e-env.sh"
```
Esperado: `3` y la ruta del backup.

- [ ] **Paso 4: orden de merge y deploy**

Primero jax-platform: su `_call_ollama` ya no lee `api_url` del config.toml, y la memoria de la plataforma lee `JAX_OLLAMA_URL`.
Recién después jax, que quita `api_url` del TOML. Invertido, el chat con JAX Local se rompe entre un paso y otro. Si E-25 se
retiró, mirror-sync de jax clona el master de jax-platform, y el orden es el mismo.

4a. Pipelines en vuelo:
```bash
set -a; source <(grep -E '^JAX_DB_(HOST|PORT|USER|PASSWORD|NAME)=' /etc/jax/.env); set +a
cd /home/fruiz/jax && PYTHONPATH=/home/fruiz/jax:/home/fruiz/jax/las_manos /home/fruiz/jax/.venv/bin/python -c \
  "import asyncio; from jacobs import store; print('en_vuelo', asyncio.run(store.pipeline_count_active()))"
systemctl is-active jax-memory-worker.service jax-memory-synthesis.service
```
Esperado: `en_vuelo 0` e `inactive inactive`. Si no, esperar y repetir. **No se reinicia con pipelines en vuelo.**

4b. jax-platform:
```bash
gh pr merge <n-plat> -R fjruizhn/jax-platform --merge
git -C /home/fruiz/jax-platform pull --ff-only && git -C /home/fruiz/jax-platform log --oneline -1
sudo systemctl restart jax-platform.service && sleep 5 && systemctl is-active jax-platform.service
sudo journalctl -u jax-platform.service --since "-2 min" --no-pager | grep -iE "Traceback|RuntimeError|JAX_OLLAMA_URL" || echo SIN_ERRORES
```

4c. jax (repetir 4a justo antes):
```bash
gh pr merge <n-jax> -R fjruizhn/Jax --merge
git -C /home/fruiz/jax pull --ff-only && git -C /home/fruiz/jax log --oneline -1
ls -la /home/fruiz/jax/las_manos | grep -E "crypto_secrets|credential_resolver|model_catalog|config_entorno|cliente_http_compartido"
sudo systemctl restart jax-las-manos.service && sleep 8 && systemctl is-active jax-las-manos.service
sudo journalctl -u jax-las-manos.service --since "-2 min" --no-pager | grep -iE "Traceback|EntornoInvalido|ImportError" || echo SIN_ERRORES
```
Esperado: los cinco symlinks, `active` y `SIN_ERRORES`.

Rollback si `jax-las-manos` no queda `active`:
1. `git -C /home/fruiz/jax reset --hard <sha-previo-anotado>` y `sudo systemctl restart jax-las-manos.service`.
2. Reportar a Fernando.

El `.env` nuevo no molesta al código viejo, así que no se revierte.

- [ ] **Paso 5: verificación en vivo**

5a. Las variables que lee el proceso vivo:
```bash
sudo cat "/proc/$(systemctl show -p MainPID --value jax-las-manos.service)/environ" | tr '\0' '\n' | grep -E '^(LAS_MANOS_URL|JAX_OLLAMA_URL|JAX_REPO_BASE_DIR)='
```

5b. E-17 en vivo (`/jacobs/plan` no persiste ni ejecuta):
```bash
curl -s -o "$SCRATCH/e17.json" -w "%{http_code}\n" -X POST http://127.0.0.1:7777/jacobs/plan -H 'Content-Type: application/json' \
  -d '{"name":"verif-e17","objective":"verificacion E-17","invoked_by":"plataforma","mode":"dry_run","steps":[{"facet":"inventada","capability":"research","prompt":"x"}]}'
cat "$SCRATCH/e17.json"
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:7777/jacobs/plan -H 'Content-Type: application/json' \
  -d '{"name":"verif-e17-ok","objective":"verificacion E-17","invoked_by":"plataforma","mode":"dry_run","steps":[{"facet":"hipatia","capability":"research","prompt":"x"}]}'
```
Esperado: `422` con "tabla `facet`" en el detalle, y `200` en el segundo.

5c. E-22, E-12 y E-24 en un pipeline real mínimo, sólo con la GPU local. Primero una capability permitida para `jax_local`:
```bash
PYTHONPATH=/home/fruiz/jax:/home/fruiz/jax/las_manos /home/fruiz/jax/.venv/bin/python -c \
  "import asyncio; from jacobs import store; g=asyncio.run(store.get_motor_governance()); print(sorted(k for k,v in g['capabilities'].items() if 'jax_local' in v['allowed_motors']))"
```
Con una de esas, `<cap>`:
```bash
curl -s -X POST http://127.0.0.1:7777/jacobs/pipeline -H 'Content-Type: application/json' \
  -d '{"name":"verif-frente-e","objective":"Responde en una linea: que es un symlink","invoked_by":"plataforma","mode":"autonomous","max_steps":1,"steps":[{"facet":"jax_local","capability":"<cap>","prompt":"Responde en una linea: que es un symlink"}]}' | tee "$SCRATCH/e-pipeline.json"
```
Esperar a que termine con `curl -s http://127.0.0.1:7777/jacobs/pipeline/<id>` y `status` completed. Después:
```bash
ls -la --time-style=full-iso /home/fruiz/jax/repo/documents | tail -3
sudo journalctl -u jax-las-manos.service --since "-10 min" --no-pager | grep -iE "No se pudo persistir|Traceback" || echo SIN_ERRORES
```
Esperado: un `.md` nuevo con el prefijo del `pipeline_id` y `SIN_ERRORES`.

5d. Memoria:
```bash
sudo systemctl start jax-memory-worker.service
sudo journalctl -u jax-memory-worker.service --since "-5 min" --no-pager | grep -iE "EntornoInvalido|Traceback|get_embedding fallo" || echo SIN_ERRORES
systemctl show -p Result --value jax-memory-worker.service
```
Esperado: `SIN_ERRORES` y `success`.

5e. Con Fernando:
- En la Mesa web, un mensaje a JAX Local (E-21 en `_call_ollama` de jax-platform) y uno a Jekyll.
- En tmux, reiniciar el REPL (`jax`), mandar un mensaje a `jax_local` y otro a `jekyll`. Tiene que arrancar sin
  `EntornoInvalido`.
- Si Fernando usa voz, una frase hablada (`JAX_KOKORO_PYTHON`).

Se anota la respuesta de Fernando.

5f. Carga del camino modificado (política 4). Se vuelve a medir el embedding con el código ya desplegado:
```bash
set -a; source <(grep -E '^JAX_MEMORY_EMBED_|^JAX_OLLAMA_URL=' /etc/jax/.env); set +a
cd /home/fruiz/jax && PYTHONPATH=/home/fruiz/jax /home/fruiz/jax/.venv/bin/python scripts/medir_cliente_http_compartido.py embedding -c 10 -n 500 | tee "$SCRATCH/e24-produccion.jsonl"
```
Esperado: `"errores": 0` y un p95 dentro del 10 % del "después" de la Tarea 13.

- [ ] **Paso 6: limpieza**

```bash
git -C /home/fruiz/jax worktree remove /home/fruiz/worktrees/jax-frente-e
git -C /home/fruiz/jax-platform worktree remove /home/fruiz/worktrees/jax-platform-frente-e
git -C /home/fruiz/jax branch -d fix/hallazgos-frente-e; git -C /home/fruiz/jax-platform branch -d fix/hallazgos-frente-e
rm -rf "$SCRATCH/venv-e19"
```
Los `$SCRATCH/e2*` y `e-*.json` se borran recién después de copiar sus números a la Biblioteca (Tarea 18).

---

### Task 18: Biblioteca: `jax/DEUDA.md` y `jax/CONTEXT.md`

**Archivos:**
- Modificar: `DEUDA.md` (sección nueva arriba de `## Cerrado — tanda A`)
- Modificar: `CONTEXT.md` (§4, §6 y §9, y la línea de última actualización)
- Si la Tarea 3 eligió el Texto B: agregar en `DEUDA.md` → `## Bloquea trabajo` el pendiente de retención de `~/jax/missions`, con fecha y dueño.

**Interfaces:**
- Consume: toda la evidencia de `$SCRATCH` (Tareas 0, 13, 14, 15, 16 y 17).
- Produce: el registro.

- [ ] **Paso 1: rama de docs**

```bash
git -C /home/fruiz/jax fetch origin && git -C /home/fruiz/jax worktree add -b docs/frente-e-biblioteca /home/fruiz/worktrees/jax-frente-e-docs origin/master
```

- [ ] **Paso 2: DEUDA.md**

Sección `## Cerrado — frente E de la auditoría de jax: limpieza, defectos y reglas (2026-09-16)`, con este contenido:
- **Qué:** una línea por id (E-01..E-26), con commit y test que lo cubre. E-14 "sin test, equivalente".
- **Por qué:** el spec y el anexo, con las rutas.
- **Discrepancias con el spec:** las 11 de este plan, marcadas como resueltas y con cómo.
- **Números:**
  - pisos de CI (`tests-puros` 494 → N, `jacobs-gobernanza-db` 27 → 29);
  - sha del canario rojo y sha del revert verde;
  - E-24 antes, después y en producción (los JSON de `$SCRATCH/e24-*.jsonl`);
  - EXPLAIN de la consulta de facetas (Tarea 0) y la declaración "catálogo de 7 filas, sin índice".
- **E-25:** veredicto (A, B o C) con los conteos, la cobertura y la consulta de DB. Si fue A: la respuesta de Fernando, y el
  contenido literal de las dos unidades systemd borradas, porque `/home/fruiz/backups` no está en restic.
- **E-26:** borrados (archivo → commit de origen), los `UNICO` listados y lo que decidió Fernando.
- **Deploy:**
  - hora del restart de cada servicio y `en_vuelo 0`;
  - backup de `.env` (ruta);
  - verificación en vivo 5a-5f con sus resultados;
  - lo que dijo Fernando en 5e.
- **Lecciones:**
  1. Un `_NO_PARSEA` vacío deja un bucle que pasa sin comprobar nada: el control se ejercita con un archivo de mentira.
  2. Un default de URL "para el arranque sin DB" es un despacho a un destino que nadie eligió.
  3. Un filtro con el literal de una sola faceta no limpia a las otras.
  4. El número de un piso se lee en el runner.

- [ ] **Paso 3: CONTEXT.md**

- §4 (arquitectura): `jax/core/config_entorno.py` (variables de servicio fail-closed) y `jax/core/cliente_http_compartido.py`
  (un cliente por loop, cierre al apagar). En `las_manos/`, `crypto_secrets`, `credential_resolver`, `model_catalog`,
  `config_entorno` y `cliente_http_compartido` son symlinks. El planner valida facetas contra la tabla `facet`.
- §6 (voz): `JAX_KOKORO_PYTHON`.
- §9 (hitos): entrada del 2026-09-16 con enlace a la sección de DEUDA.
- Primera línea: "Última actualización: 16 de septiembre de 2026 (frente E …)".

- [ ] **Paso 4: commit, PR y merge**

```bash
cd /home/fruiz/worktrees/jax-frente-e-docs && git add DEUDA.md CONTEXT.md && git commit -m "docs(biblioteca): frente E de la auditoria de jax cerrado (2026-09-16)" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" && git push -u origin docs/frente-e-biblioteca
gh pr create -R fjruizhn/Jax --base master --head docs/frente-e-biblioteca --title "Biblioteca: frente E de la auditoria de jax" --body-file "$SCRATCH/pr-docs.md"
```
Con CI verde, se mergea y se hace `git -C /home/fruiz/jax pull --ff-only`. Después se borran el worktree y la rama, y los
archivos de `$SCRATCH` cuyos números ya están en DEUDA.

---

## Auto-revisión de este plan

**Cobertura del spec:**

| id | Tarea |
|---|---|
| E-01 | T1 |
| E-02 | T2 |
| E-03 | T6 |
| E-04, E-05, E-06 | T2 |
| E-07 | T3 |
| E-08 | T4 |
| E-09 | T2 |
| E-10, E-11 | T7 |
| E-12 | T9 |
| E-13 | T5 |
| E-14, E-15 | T11 |
| E-16 | T12 |
| E-17 | T6 |
| E-18 | T9 |
| E-19 | T10 |
| E-20 | T2 |
| E-21 | T8 |
| E-22 | T9 |
| E-23 | T6 |
| E-24 | T13 |
| E-25 | T14 |
| E-26 | T15 |

- "Se conservan con motivo" (`KeyProvider`, `usage_writer`, `review_audit`): no se tocan y se nombran en DEUDA.
- Reglas comunes: TDD en cada tarea, barrera de DB (T6 y los `os.environ["JAX_DB_NAME"]`), CI con canario (T16),
  mirror-sync (T7 y T14), deploy con 0 en vuelo (T17), Biblioteca (T18), carga en el camino modificado (T13 y T17).
- i18n y dark/light no aplican: este frente no toca frontend.

**Placeholders:** los `<N>`, `<fecha>`, `<proveedor>`, `<cap>`, `<n-plat>` y `<n-jax>` son valores que sólo existen al
ejecutar (conteo del runner, resultado del gate, número de PR). Cada uno tiene el comando exacto que lo produce en el mismo
paso o en el anterior.

**Consistencia de nombres:**
- `url_requerida`, `ruta_absoluta_requerida`, `EntornoInvalido`: T8, usados en T9.
- `obtener_cliente_http`, `crear_cliente_http`, `cerrar_cliente_http`: T13.
- `REPO_DOCUMENTS_DIR`: T9, usado en `test_hipatia_fuentes`.
- `_check_facets` y `governance["facets"]`: T6, con el fake de `_plan_timeout_ceiling_test` actualizado.
- `MAX_STEPS_PER_PIPELINE` en `jacobs.models`: T5.
- `_sin_autoetiqueta`: T11.
- `_texto_de_error_de_tarea`: T12.
- `_url_del_catalogo`: T8, usado en T11 y T12 (los músculos de los tests pasan `api_url`).
- `JAX_OLLAMA_URL` es la base sin `/api/...` en jax y en jax-platform.
