# Proyectos E2a — subir documentos al proyecto · Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que un CONTRIBUTOR (o más) suba por lotes documentos a un proyecto desde Chat o Pipeline, los vea procesarse en la pestaña Documentos y pueda ocultarlos; con LAS MANOS trabajando por `project_uuid` y LACTOVI migrado como primer proyecto con documentos.

**Architecture:** jax-platform recibe el lote, lo escribe por streaming a una carpeta de **entrada** del proyecto (`proyectos/<uuid>/entrada/<lote>/`) y registra cada archivo en `project_documents`. Un despachador de fondo en jax-platform reparte las filas pendientes en trabajos de LAS MANOS (máximo 50 rutas por trabajo, 4 trabajos a la vez) y sincroniza su estado; la ingesta de LAS MANOS copia el original a `fuente/` y deja el extracto en `procesado/<sha256>/`, como hoy. La autorización vive solo en jax-platform (B9); LAS MANOS solo comprueba que el proyecto exista y esté ACTIVE.

**Tech Stack:** Python 3.12, FastAPI, aiomysql, MariaDB 12.3 (127.0.0.1:3308), pytest/unittest (base `jax_test`); React 19, Zustand, axios, vitest, Tailwind con tokens.

**Spec:** `docs/superpowers/specs/2026-10-02-proyectos-e2-adenda-design.md` (Jax#323, `6a7215f`), que continúa `docs/superpowers/specs/2026-09-22-proyectos-y-selector-design.md`.

## Decisiones que este plan ya incorpora

| Decisión | Quién / cuándo |
|---|---|
| Ocultar un documento **entra** en E2a: un CONTRIBUTOR lo oculta y lo restaura; borrarlo de verdad queda para Destruir | Fernando, 2026-10-03 («de acuerdo», §8.1 de la adenda) |
| El Ejecutor (E2b-3) queda **fuera** de E2. En E2a el selector de proyecto y el botón de documentos van en **Chat y Pipeline**, no en Ejecutor | Fernando, 2026-10-03 (§8.2) |
| **Corrección a la adenda §3.3** (hecho del código, no decisión nueva): la plataforma NO escribe en `fuente/`. `procesamiento/ingesta.py::ingerir` copia el original a `fuente/` ella misma (`_asegurar_en_fuente`, con nombre libre y deduplicado por contenido); si la plataforma escribiera ahí, cada archivo quedaría dos veces. La plataforma escribe en `proyectos/<uuid>/entrada/<lote>/` y borra esa copia cuando el trabajo termina | Hyde, 2026-10-03, leyendo `procesamiento/ingesta.py:345-389,628-640` |
| **Despachador de fondo** en jax-platform: LAS MANOS acepta como máximo `JAX_PROCESAMIENTO_MAX_RUTAS` = 50 rutas por trabajo y 4 trabajos concurrentes (429 si no hay cupo). Un lote de 250 son 5 trabajos y no siempre entran a la vez | Hyde, 2026-10-03, `las_manos/procesamiento_routes.py:160,628-660` |

## Global Constraints

- Papeles: lector = `VIEWER`, editor = `CONTRIBUTOR`, dueño = `OWNER`. Subir, ocultar y restaurar: **CONTRIBUTOR o más**. Listar: **VIEWER o más**.
- 404 si el proyecto no existe, está oculto o no eres miembro (sin revelar cuál); 403 si eres miembro y te falta el papel — mismo contrato que E1 (`backend/api/proyectos.py::_http`).
- Solo se sube a un proyecto **ACTIVE**. Archivado → 409 `proyecto_no_activo`.
- Topes (Fernando, 2026-09-25): **100 MB por archivo; 250 archivos o 1 GB por lote.** Viven en `axioma_config` (claves de la Tarea 5), nunca en el código.
- Tipos aceptados: los que `procesamiento/` sabe extraer — `.pdf .xlsx .xls .docx .png .jpg .jpeg .tif .tiff .csv .txt .md`. Lo demás se ignora y el resumen lo dice. La lista vive en un solo módulo de la plataforma (Tarea 5) y la comparte el frontend vía `GET /api/proyectos/documentos/limites`.
- `project_documents` es la única fuente del estado de un documento; el frontend nunca habla con LAS MANOS.
- i18n en `frontend/src/i18n/es.js` y `en.js` (raíz `proyectos.documentos`); colores solo con tokens; claro y oscuro; controles ≥ 24 px; ningún `confirm(`, `alert(` ni `prompt(` (tampoco desnudos): toda confirmación en `components/Dialogo.jsx`.
- MariaDB en **3308**. Las pruebas usan `jax_test` (`~/.config/jax/test-db.env`); **nunca** `/etc/jax/.env` ni `jax_memory`. Las pruebas de disco usan `tmp_path`, nunca `~/jax-workspace`.
- Subagentes: sin `sudo`, sin `git push`, sin `gh`; commits con `git commit -F <archivo>`. Nunca `docker system prune` ni nada que pode Docker.
- Rama base: `master` en los dos repos. Los pisos de CI se fijan con el número **del runner**, no con el local.
- Producción (Parte C) solo con el GO de Fernando o su ventana abierta.

## Review Focus

1. **Lote que pasa el tope a mitad de la subida** (el navegador dice 900 MB pero el multipart trae 1,2 GB, o un archivo dice 90 MB y trae 140) → se corta al cruzar el tope con 413 y código estable, no queda ningún archivo parcial en `entrada/` ni fila en `project_documents` del archivo cortado; los archivos ya completos del lote sí quedan. Prueba en la Tarea 6 (`test_tope_real_no_el_declarado`).
2. **El mismo PDF subido dos veces (o subido mientras otro lote con él sigue procesándose)** → la segunda vez se informa `duplicado` y no se crea fila ni trabajo; si el existente está oculto, `duplicado_oculto`. Dos subidas concurrentes del mismo contenido terminan con **una** fila (la restricción única decide, no un SELECT previo). Prueba en la Tarea 6 (`test_duplicado_concurrente_una_sola_fila`).
3. **LAS MANOS caída o sin cupo cuando llega el lote** → la subida responde 202 igual, las filas quedan `en_cola`, y el despachador las manda cuando LAS MANOS vuelve; nada se pierde ni queda `procesando` para siempre. Prueba en la Tarea 7 (`test_429_deja_en_cola_y_reintenta`, `test_las_manos_caida_no_pierde_filas`).
4. **jax-platform se reinicia con trabajos en vuelo** → al volver, el despachador consulta esos `job_id` y sigue; un `job_id` que LAS MANOS ya no conoce (404) pasa la fila a `error` con causa `trabajo_perdido`, no la deja colgada. Prueba en la Tarea 7 (`test_trabajo_perdido_pasa_a_error`).
5. **Nombre de archivo hostil** (`../../etc/passwd`, `con\x00nul.pdf`, 600 caracteres, solo emojis, `Informe.PDF` y `informe.pdf` en el mismo lote) → el nombre en disco es seguro y único, el original se conserva solo como texto en `nombre_original` (recortado a 1024), y nada escribe fuera de `entrada/<lote>/`. Prueba en la Tarea 6 (`test_nombres_hostiles`).

---

## Parte A — jax (rama `feat/proyectos-e2a-jax`, worktree `~/worktrees/jax-proyectos-e2a`)

### Task 1: Tabla `project_documents` (migración 006a)

**Files:**
- Modify: `jax/memory/project_authority_migrations.py` (paso 006a en `_apply_project_lifecycle_migration`, constantes arriba)
- Modify: `jax_memory_schema.sql` (la misma tabla, después de `projects`, para que el esquema de referencia no mienta)
- Test: `tests/test_project_documents_schema_mariadb.py` (nuevo; los ayudantes `asincrono`, `_conn_params`, `_ensure_schema`, `_sql`, `_crear_tenant`, `_crear_usuario`, `_crear_proyecto_activo` se copian del encabezado de `tests/test_project_authority_mariadb.py` o se importan de él, lo que haga hoy el repo)

**Interfaces:**
- Produces: tabla `project_documents` (la consumen las Tareas 4, 5, 6 y 7) con exactamente estas columnas y nombres de índice.

- [ ] **Step 1: Escribir las pruebas que fallan**

```python
@asincrono
async def test_006a_crea_project_documents_con_indices_y_fks():
    await _ensure_schema()
    cols = {r[0]: r[1] for r in await _sql(
        "SELECT COLUMN_NAME, COLUMN_TYPE FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='project_documents'")}
    assert cols["project_id"] == "int(11)"
    assert cols["sha256"] == "char(64)"
    assert cols["subido_por"] == "int(11)"
    assert cols["estado"].startswith("enum('en_cola','pendiente','procesando','listo','parcial','error','sin_extractor','cancelado')")
    idx = {r[0] for r in await _sql(
        "SELECT INDEX_NAME FROM information_schema.STATISTICS "
        "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='project_documents'")}
    assert {"uq_project_documents_sha", "idx_project_documents_lista", "idx_project_documents_despacho"} <= idx
    fks = {r[0] for r in await _sql(
        "SELECT CONSTRAINT_NAME FROM information_schema.REFERENTIAL_CONSTRAINTS "
        "WHERE CONSTRAINT_SCHEMA=DATABASE() AND TABLE_NAME='project_documents'")}
    assert fks == {"fk_project_documents_project", "fk_project_documents_subido_por", "fk_project_documents_oculto_por"}


@asincrono
async def test_006a_es_idempotente():
    await _ensure_schema()
    await _ensure_schema()   # segunda pasada: sin error, sin cambios


@asincrono
async def test_006a_mismo_contenido_dos_veces_en_un_proyecto_choca():
    await _ensure_schema()
    t = await _crear_tenant("d1"); u = await _crear_usuario(t); p = await _crear_proyecto_activo(t, name="D")
    ins = ("INSERT INTO project_documents (project_id, sha256, nombre_original, bytes, tipo, subido_por) "
           "VALUES (%s, %s, 'a.pdf', 10, 'pdf', %s)")
    await _sql(ins, (p, "a" * 64, u))
    with pytest.raises(Exception, match="1062"):
        await _sql(ins, (p, "a" * 64, u))
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `pytest tests/test_project_documents_schema_mariadb.py -v`
Expected: FAIL (`KeyError: 'project_id'`: la tabla no existe).

- [ ] **Step 3: Implementar 006a**

En `jax/memory/project_authority_migrations.py`, junto a las demás constantes:

```python
#: E2a (2026-10-03, adenda E2 §3.1): un documento subido a un proyecto. Vive acá,
#: aunque la escribe jax-platform, por la misma razón que 005h/005i: este hook es el
#: que jax-platform corre en cada arranque (`db/migrations.py:124-142`), y la lee
#: también LAS MANOS en E2b. CREATE ... IF NOT EXISTS: re-correrlo no hace nada.
_DDL_PROJECT_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS project_documents (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  project_id INT(11) NOT NULL,
  sha256 CHAR(64) NOT NULL,
  nombre_original VARCHAR(1024) NOT NULL,
  ruta_entrada VARCHAR(1024) NULL,
  carpeta_procesado VARCHAR(255) NULL,
  bytes BIGINT UNSIGNED NOT NULL,
  tipo VARCHAR(16) NOT NULL,
  estado ENUM('en_cola','pendiente','procesando','listo','parcial','error','sin_extractor','cancelado')
    NOT NULL DEFAULT 'en_cola',
  error VARCHAR(1000) NULL,
  job_id VARCHAR(64) NULL,
  subido_por INT(11) NOT NULL,
  oculto_at DATETIME(6) NULL,
  oculto_por INT(11) NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  UNIQUE KEY uq_project_documents_sha (project_id, sha256),
  KEY idx_project_documents_lista (project_id, oculto_at, id),
  KEY idx_project_documents_despacho (estado, job_id, id),
  CONSTRAINT fk_project_documents_project FOREIGN KEY (project_id) REFERENCES projects (id),
  CONSTRAINT fk_project_documents_subido_por FOREIGN KEY (subido_por) REFERENCES jax_users (user_id),
  CONSTRAINT fk_project_documents_oculto_por FOREIGN KEY (oculto_por) REFERENCES jax_users (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""
```

Al final de `_apply_project_lifecycle_migration`:

```python
    # 006a (E2a): documentos del proyecto, ver _DDL_PROJECT_DOCUMENTS.
    await cursor.execute(_DDL_PROJECT_DOCUMENTS)
```

En `revert_project_lifecycle_migration`, **antes** de los pasos de 005 (es la tabla más nueva y referencia a `projects`): `DROP TABLE IF EXISTS project_documents` solo si está vacía; si tiene filas, falla cerrado con el mismo patrón de conteo que usa hoy esa función.

Copiar el mismo `CREATE TABLE` (sin `IF NOT EXISTS`, con el estilo de comillas invertidas del archivo) a `jax_memory_schema.sql`, después de `projects`.

- [ ] **Step 4: Correr y ver que pasan**

Run: `pytest tests/test_project_documents_schema_mariadb.py tests/test_project_authority_mariadb.py -v`
Expected: PASS, y las pruebas de autoridad siguen verdes.

- [ ] **Step 5: Commit**

```bash
git add jax/memory/project_authority_migrations.py jax_memory_schema.sql tests/test_project_documents_schema_mariadb.py
git commit -F /tmp/msg-t1.txt   # "feat(proyectos): E2a 006a tabla project_documents"
```

---

### Task 2: LAS MANOS trabaja por `project_uuid`

**Files:**
- Modify: `las_manos/procesamiento_routes.py` (modelo, `_trabajo_de`, validación, docstring de cabecera)
- Create: `las_manos/proyecto_activo.py` (una sola consulta, inyectable)
- Modify: `scripts/procesar_archivos.py` (`--project-uuid` en lugar del `proyecto` posicional)
- Test: `las_manos/_procesamiento_routes_test.py` (casos nuevos), `tests/test_proyecto_activo_mariadb.py` (nuevo)

**Interfaces:**
- Produces:
  - `POST /procesamiento/trabajos` con cuerpo `{"project_uuid": str, "rutas": list[str], "usuario": str}` (`extra="forbid"`: un `proyecto` viejo da 422).
  - 422 `{"detail": {"code": "project_uuid_invalido"}}` si no es un UUID canónico de 36 caracteres; 422 `{"detail": {"code": "proyecto_no_activo"}}` si no existe o su `jax_project_scope.status` no es `ACTIVE`.
  - Carpeta del trabajo: `WORKSPACE_ROOT / "proyectos" / <project_uuid>`.
  - `las_manos/proyecto_activo.py`: `async def estado_del_proyecto(project_uuid: str) -> str | None` — `'ACTIVE'|'ARCHIVED'|'HIDDEN'|'DISABLED'` o `None` si no existe.
- Consumes: `jacobs.store.conexion()` (pool que LAS MANOS ya tiene, `jacobs/store.py:678`).

- [ ] **Step 1: Pruebas que fallan** (en `_procesamiento_routes_test.py`, mismo estilo `unittest` + `TestClient` del archivo; parchear `rutas_mod.proyecto_activo.estado_del_proyecto` con `AsyncMock`)

```python
class ProjectUuidTest(unittest.TestCase):
    UUID = "0192f1d2-7c3a-7b4e-9a10-3f5e2d1c0b9a"

    def _post(self, cuerpo, estado="ACTIVE"):
        with patch.object(rutas_mod.proyecto_activo, "estado_del_proyecto", AsyncMock(return_value=estado)), \
             patch.object(rutas_mod, "_ejecutar_trabajo", AsyncMock()):
            return self.client.post("/procesamiento/trabajos", json=cuerpo, headers=self.cabeceras_plataforma)

    def test_proyecto_viejo_por_nombre_da_422(self):
        r = self._post({"proyecto": "lacteos", "rutas": ["a.pdf"], "usuario": "u1"})
        self.assertEqual(r.status_code, 422)

    def test_uuid_no_canonico_da_422_con_codigo(self):
        r = self._post({"project_uuid": "../../etc", "rutas": ["a.pdf"], "usuario": "u1"})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["detail"]["code"], "project_uuid_invalido")

    def test_proyecto_archivado_da_422_y_no_toma_cupo(self):
        r = self._post({"project_uuid": self.UUID, "rutas": ["a.pdf"], "usuario": "u1"}, estado="ARCHIVED")
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["detail"]["code"], "proyecto_no_activo")
        self.assertFalse(rutas_mod._SEMAFORO_TRABAJOS.locked())

    def test_proyecto_inexistente_da_422(self):
        r = self._post({"project_uuid": self.UUID, "rutas": ["a.pdf"], "usuario": "u1"}, estado=None)
        self.assertEqual(r.json()["detail"]["code"], "proyecto_no_activo")

    def test_carpeta_del_trabajo_es_el_uuid(self):
        self.assertEqual(rutas_mod._trabajo_de(self.UUID),
                         tool_authority.WORKSPACE_ROOT / "proyectos" / self.UUID)
```

(`setUp` arma `self.client` y `self.cabeceras_plataforma` con el mismo patrón que ya usan las clases del archivo para la identidad de plataforma.)

En `tests/test_proyecto_activo_mariadb.py`, contra `jax_test`: proyecto con scope `ACTIVE` → `'ACTIVE'`; con scope `ARCHIVED` → `'ARCHIVED'`; uuid inexistente → `None`.

- [ ] **Step 2: Correr y ver que fallan**

Run: `PYTHONPATH=.:las_manos python -m pytest -v las_manos/_procesamiento_routes_test.py -k ProjectUuid` y `pytest tests/test_proyecto_activo_mariadb.py -v`
Expected: FAIL (`proyecto_activo` no existe; el modelo todavía pide `proyecto`).

- [ ] **Step 3: Implementar**

`las_manos/proyecto_activo.py`:

```python
"""¿Existe el proyecto y está ACTIVE? (E2a, adenda E2 §3.2). LAS MANOS NO verifica
membresía: solo la credencial de plataforma llama a /procesamiento y jax-platform
ya verificó el papel. Esto solo impide trabajar sobre un proyecto archivado,
oculto o inexistente. Usa `idx` único de projects.project_uuid y la PK de scope."""
from __future__ import annotations

from jacobs import store as jacobs_store

_CONSULTA = (
    "SELECT s.status FROM projects p JOIN jax_project_scope s ON s.project_id = p.id "
    "WHERE p.project_uuid = %s"
)


async def estado_del_proyecto(project_uuid: str) -> str | None:
    async with jacobs_store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_CONSULTA, (project_uuid,))
            fila = await cur.fetchone()
    if fila is None:
        return None
    valor = fila["status"] if isinstance(fila, dict) else fila[0]
    return str(valor).upper()
```

(Si `jacobs_store.conexion()` entrega otro tipo de cursor, adaptar la lectura de `fila`; la prueba de MariaDB lo fija.)

En `procesamiento_routes.py`:

```python
import proyecto_activo

_UUID_CANONICO = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class TrabajoRequest(BaseModel):
    project_uuid: str = Field(min_length=36, max_length=36)
    rutas: list[str]
    usuario: str = Field(min_length=1, max_length=_MAX_USUARIO_LEN)
    model_config = ConfigDict(extra="forbid")


def _trabajo_de(project_uuid: str) -> Path:
    # El uuid ya pasó _UUID_CANONICO: no hace falta slug, y renombrar el
    # proyecto no mueve archivos (spec §5).
    return tool_authority.WORKSPACE_ROOT / "proyectos" / project_uuid
```

En `crear_trabajo`, en lugar del bloque de `proyecto` (MINOR-6 y el chequeo UTF-8 de `proyecto` dejan de hacer falta: el uuid es ASCII por la regex), y **antes** de mirar el semáforo:

```python
    if not _UUID_CANONICO.fullmatch(req.project_uuid):
        raise HTTPException(status_code=422, detail={"code": "project_uuid_invalido"})
    if await proyecto_activo.estado_del_proyecto(req.project_uuid) != "ACTIVE":
        raise HTTPException(status_code=422, detail={"code": "proyecto_no_activo"})
    proyecto = req.project_uuid
```

El resto del handler sigue igual (`_STORE.update(job_id, proyecto=proyecto)` guarda el uuid). Borrar `_MAX_PROYECTO_LEN` y actualizar la línea `POST /procesamiento/trabajos {proyecto, ...}` del docstring de cabecera a `{project_uuid, rutas[], usuario}`. Las pruebas viejas del archivo que mandan `proyecto` se actualizan a `project_uuid` con el estado parcheado a `ACTIVE`; ninguna se borra.

`scripts/procesar_archivos.py`: el posicional `proyecto` pasa a `--project-uuid` (obligatorio, misma regex) y la carpeta a `proyectos/<uuid>`; el guion sigue sin hablar con la base (es manual, lo corre `fruiz`): el operador es responsable de que el uuid sea de un proyecto real, y el `--help` lo dice.

- [ ] **Step 4: Correr y ver que pasan**

Run: `PYTHONPATH=.:las_manos python -m pytest -v las_manos/_procesamiento_routes_test.py` y `pytest tests/test_proyecto_activo_mariadb.py -v`
Expected: PASS (los 56 de antes más los nuevos).

- [ ] **Step 5: Commit**

```bash
git add las_manos/procesamiento_routes.py las_manos/proyecto_activo.py scripts/procesar_archivos.py las_manos/_procesamiento_routes_test.py tests/test_proyecto_activo_mariadb.py
git commit -F /tmp/msg-t2.txt   # "feat(procesamiento): E2a trabajos por project_uuid, 422 si el proyecto no está ACTIVE"
```

---

### Task 3: Permisos de `proyectos/` (retomar Jax#276)

Jax#276 (rama `ops/permisos-proyectos`, `c136399`, del 2026-09-25) ya trae `ops/permisos_proyectos.py` (`--verificar` / `--aplicar`, respaldo con `getfacl`), su prueba y el runbook `docs/runbooks/workspace-proyectos.md`. Está abierto y sin integrar.

**Files:** los del PR: `ops/permisos_proyectos.py`, `tests/test_permisos_proyectos.py`, `docs/runbooks/workspace-proyectos.md`, `las_manos/motor_registry/tool_authority.py`, `las_manos/_tool_authority_test.py`, `.github/workflows/policy.yml`.

**Interfaces:**
- Produces: tras `--aplicar` en el host, `proyectos/` y lo nuevo debajo con grupo `jaxsvc`, setgid y ACL por defecto `g:jaxsvc:rwx`: `jaxsvc` (jax-platform y LAS MANOS) puede crear `proyectos/<uuid>/entrada/...` y `fuente/`, y `fruiz` sigue pudiendo correr el guion a mano.

- [ ] **Step 1:** Traer la rama a este worktree: `git merge --no-ff origin/ops/permisos-proyectos` sobre `feat/proyectos-e2a-jax`, resolviendo los conflictos contra el `master` de hoy (en `tool_authority.py` y en `policy.yml` hubo cambios desde el 25-sep: conservar los dos lados, nunca descartar el de `master`).
- [ ] **Step 2:** Leer el diff completo de #276 contra `master` y confirmar que **no** cambia ningún comportamiento de `tool_authority.py` más allá de lo que su descripción dice; si cambia algo más, anotarlo para el auditor de la Tarea 4.
- [ ] **Step 3:** Agregar a `tests/test_permisos_proyectos.py` un caso portable: en un `tmp_path` con setgid y ACL por defecto aplicadas por el propio script (modo de prueba que ya trae), un subdirectorio `entrada/lote1/` creado después **hereda** el grupo y la ACL. Run: `pytest tests/test_permisos_proyectos.py -v` → PASS (la parte "real como jaxsvc" sigue marcada como solo-host y fallando contra el estado de hoy, como exige el spec §5).
- [ ] **Step 4:** Ajustar el piso de `policy.yml` con el número que dé el runner.
- [ ] **Step 5:** Commit (`"ops(proyectos): E2a integra los permisos de proyectos/ (Jax#276) y cubre entrada/"`). Al abrir el PR de la Parte A, cerrar #276 con un comentario que apunte a ese PR.

---

### Task 4: Guion de LACTOVI (`scripts/proyectos_e2a_lactovi.py`) + cierre de la Parte A

**Files:**
- Create: `scripts/proyectos_e2a_lactovi.py`
- Create: `docs/runbooks/proyectos-e2a-produccion.md`
- Test: `tests/test_proyectos_e2a_lactovi_mariadb.py`

**Interfaces:**
- Consumes: `ProjectAuthorityAdmin.create_project` (`jax/memory/project_authority.py:475`), tabla `project_documents` (Tarea 1), `procesamiento.ficha.Ficha.desde_json` y `procesamiento.ficha.sha256_de`.
- Produces: CLI
  `python -m scripts.proyectos_e2a_lactovi --workspace <ruta> --carpeta lacteos-victoria --nombre "Lácteos Victoria" --dueno-user-id <id> --tenant-id 1 [--aplicar] [--revertir <mapa.json>]`
  Por defecto es **ensayo** (no escribe nada y dice lo que haría). Con `--aplicar`:
  1. Calcula `sha256` de cada archivo de `fuente/` (antes).
  2. Crea el proyecto con `create_project` (dueño = `--dueno-user-id`); obtiene `project_uuid`.
  3. Escribe el mapa de reversión `<workspace>/proyectos/.e2a-lactovi-<fecha>.json` (uuid, id, ruta vieja, ruta nueva, sha256 de antes) **antes** de mover.
  4. `os.rename(proyectos/<carpeta>, proyectos/<uuid>)` (mismo sistema de archivos; si `rename` falla, aborta sin borrar nada).
  5. Recalcula los `sha256` (después) y exige igualdad exacta; si no, revierte el `rename` y sale con código 3.
  6. Por cada `procesado/<sha>/ficha.json`: una fila en `project_documents` con `sha256`, `nombre_original` (el nombre en `fuente/` cuyo sha coincide), `bytes`, `tipo` (extensión sin punto, minúsculas), `estado` = `ficha.estado` mapeado (`ok`→`listo`, `parcial`→`parcial`, `error`→`error`, `sin_extractor`→`sin_extractor`), `carpeta_procesado` = `proyectos/<uuid>/procesado/<sha>`, `subido_por` = dueño, `job_id` NULL. Archivos de `fuente/` sin ficha → fila `en_cola` (el despachador los procesará). `INSERT IGNORE` por la restricción única.
  7. Imprime conteos: archivos en `fuente/`, fichas, filas insertadas, y la ruta del mapa.
  `--revertir <mapa>`: borra las filas de `project_documents` de ese proyecto, deshace el `rename` verificando `sha256`, y deja el proyecto **archivado** con `set_project_lifecycle` (no lo borra: Destruir no existe).

- [ ] **Step 1: Pruebas que fallan**, contra `jax_test` y un `tmp_path` armado con 3 archivos en `fuente/` (uno con ficha `ok`, uno con ficha `parcial`, uno sin ficha) y un `.claude-flow/` suelto:
  - `test_ensayo_no_escribe_nada` (ni base ni disco cambian).
  - `test_aplicar_mueve_registra_y_conserva_sha` (carpeta nueva = uuid del proyecto creado; 3 filas con estados `listo`, `parcial`, `en_cola`; sha iguales; mapa escrito).
  - `test_aplicar_dos_veces_no_duplica` (segunda corrida: la carpeta vieja ya no está → sale con código 2 y mensaje claro, sin crear otro proyecto).
  - `test_sha_distinto_revierte_el_rename` (parchear el recálculo para que dé distinto → la carpeta vuelve a su nombre, código 3).
  - `test_revertir_deja_todo_como_antes_y_proyecto_archivado`.
- [ ] **Step 2:** Correr → FAIL (el guion no existe).
- [ ] **Step 3:** Implementar el guion siguiendo el patrón de `scripts/proyectos_e1_migrar.py` (banderas explícitas, se niega a correr contra una base que no sea la indicada, mapa de reversión antes de confirmar).
- [ ] **Step 4:** Correr → PASS.
- [ ] **Step 5:** Escribir `docs/runbooks/proyectos-e2a-produccion.md` con el orden de la Parte C, los comandos exactos y la verificación de cada paso; commit.

**Cierre de la Parte A:** auditoría de escalón 3 (`arquitecto-adversarial`) sobre la rama completa, con el diff de #276 incluido; PR a `master` de jax; se integra con CI verde y sin BLOCK/MAJOR abiertos. La Parte B usa `master` de jax en su CI (el hook de migraciones se carga por ruta), así que **A se integra antes de abrir el PR de B**.

---

## Parte B — jax-platform (rama `feat/proyectos-e2a`, worktree `~/worktrees/jax-platform-proyectos-e2a`)

### Task 5: Configuración, tipos y repositorio de documentos

**Files:**
- Modify: `backend/ajustes.py` (cuatro claves nuevas en `CLAVES` y `DEFINICIONES`)
- Modify: `backend/db/migrations.py` (siembra `INSERT IGNORE`, mismo patrón que `_jacobs_tope_devoluciones_v1`, llamada junto a ella)
- Create: `backend/proyectos_documentos/__init__.py`, `backend/proyectos_documentos/tipos.py`, `backend/proyectos_documentos/repositorio.py`
- Test: `backend/tests/test_proyectos_documentos_repositorio.py`, `backend/tests/test_ajustes.py` (casos nuevos)

**Interfaces:**
- Produces:
  - Claves de `axioma_config` y su valor inicial: `proyectos.documentos.max_bytes_archivo` = `104857600` (rango 1 MiB..2 GiB), `proyectos.documentos.max_archivos_lote` = `250` (1..1000), `proyectos.documentos.max_bytes_lote` = `1073741824` (1 MiB..10 GiB), `proyectos.documentos.rutas_por_trabajo` = `50` (1..50; el techo es el de LAS MANOS).
  - `tipos.py`: `EXTENSIONES_ACEPTADAS: frozenset[str]` = `{"pdf","xlsx","xls","docx","png","jpg","jpeg","tif","tiff","csv","txt","md"}`; `def tipo_de(nombre: str) -> str | None` (extensión en minúsculas si está aceptada, si no `None`).
  - `repositorio.py` (todas reciben un pool aiomysql de la plataforma):
    - `async def insertar(pool, *, project_id: int, sha256: str, nombre_original: str, ruta_entrada: str, bytes_: int, tipo: str, subido_por: int) -> int | None` — id de la fila nueva, o `None` si chocó con `uq_project_documents_sha`.
    - `async def existente_por_sha(pool, *, project_id: int, sha256: str) -> dict | None` — `{"id", "oculto": bool}`.
    - `async def listar(pool, *, project_id: int, ocultos: bool, antes_de: int | None, limite: int) -> list[dict]` — orden `id DESC`; cada dict `{"id","nombre","bytes","tipo","estado","error","subido_por_email","creado","oculto"}`.
    - `async def ocultar(pool, *, project_id: int, documento_id: int, user_id: int) -> bool` / `async def restaurar(pool, *, project_id: int, documento_id: int) -> bool` — `False` si el documento no es de ese proyecto.
    - `async def tomar_en_cola(pool, *, limite: int) -> list[dict]` — filas `en_cola` de proyectos ACTIVE, orden `id`; cada dict `{"id", "project_id", "project_uuid", "ruta_entrada", "subido_por_email"}` (el email va como `usuario` a LAS MANOS).
    - `async def marcar_despachadas(pool, *, ids: list[int], job_id: str) -> None` → `pendiente` + `job_id`.
    - `async def trabajos_abiertos(pool) -> list[str]` — `job_id` distintos con filas `pendiente`/`procesando`.
    - `async def aplicar_resultado(pool, *, job_id: str, ruta_entrada: str, estado: str, carpeta_procesado: str | None, error: str | None) -> None`.
    - `async def marcar_job_perdido(pool, *, job_id: str) -> None` → `error`, `error='trabajo_perdido'`.

- [ ] **Step 1: Pruebas que fallan** — en `test_ajustes.py`: cada clave nueva se lee con su valor inicial tras la siembra; un valor fuera de rango da `AjusteIlegible`. En `test_proyectos_documentos_repositorio.py` (fixture de base de `tests/identidades.py` / `base_de_test.py`, igual que `test_proyectos_api.py`):

```python
async def test_insertar_duplicado_devuelve_none(pool, proyecto_activo, usuario):
    a = await repo.insertar(pool, project_id=proyecto_activo, sha256="b" * 64, nombre_original="x.pdf",
                            ruta_entrada="entrada/l1/x.pdf", bytes_=5, tipo="pdf", subido_por=usuario)
    b = await repo.insertar(pool, project_id=proyecto_activo, sha256="b" * 64, nombre_original="y.pdf",
                            ruta_entrada="entrada/l2/y.pdf", bytes_=5, tipo="pdf", subido_por=usuario)
    assert a is not None and b is None


async def test_listar_excluye_ocultos_y_pagina(pool, proyecto_activo, usuario):
    ids = [await repo.insertar(pool, project_id=proyecto_activo, sha256=f"{i:064x}", nombre_original=f"{i}.pdf",
                               ruta_entrada=f"entrada/l/{i}.pdf", bytes_=1, tipo="pdf", subido_por=usuario)
           for i in range(3)]
    assert await repo.ocultar(pool, project_id=proyecto_activo, documento_id=ids[1], user_id=usuario)
    visibles = await repo.listar(pool, project_id=proyecto_activo, ocultos=False, antes_de=None, limite=50)
    assert [d["id"] for d in visibles] == [ids[2], ids[0]]
    ocultos = await repo.listar(pool, project_id=proyecto_activo, ocultos=True, antes_de=None, limite=50)
    assert [d["id"] for d in ocultos] == [ids[1]]


async def test_ocultar_documento_de_otro_proyecto_es_false(pool, proyecto_activo, otro_proyecto, usuario):
    d = await repo.insertar(pool, project_id=otro_proyecto, sha256="c" * 64, nombre_original="z.pdf",
                            ruta_entrada="entrada/l/z.pdf", bytes_=1, tipo="pdf", subido_por=usuario)
    assert await repo.ocultar(pool, project_id=proyecto_activo, documento_id=d, user_id=usuario) is False


async def test_tomar_en_cola_ignora_proyectos_archivados(pool, proyecto_activo, proyecto_archivado, usuario):
    a = await repo.insertar(pool, project_id=proyecto_activo, sha256="d" * 64, nombre_original="a.pdf",
                            ruta_entrada="entrada/l/a.pdf", bytes_=1, tipo="pdf", subido_por=usuario)
    await repo.insertar(pool, project_id=proyecto_archivado, sha256="e" * 64, nombre_original="b.pdf",
                        ruta_entrada="entrada/l/b.pdf", bytes_=1, tipo="pdf", subido_por=usuario)
    filas = await repo.tomar_en_cola(pool, limite=100)
    assert [f["id"] for f in filas] == [a]
    assert filas[0]["subido_por_email"] and filas[0]["project_uuid"]
```

(Los fixtures `proyecto_activo`, `otro_proyecto`, `proyecto_archivado` y `usuario` se arman con los ayudantes de `tests/identidades.py` que usa `test_proyectos_api.py`; si no existen con esos nombres, se crean en el `conftest` local del archivo.)

- [ ] **Step 2:** Correr → FAIL.
- [ ] **Step 3:** Implementar. Todo el SQL en `repositorio.py`, parametrizado. `listar` usa `idx_project_documents_lista` (`WHERE project_id=%s AND oculto_at IS NULL AND id < %s ORDER BY id DESC LIMIT %s`, con `IS NOT NULL` para la vista de ocultos). `tomar_en_cola` hace `JOIN projects` + `JOIN jax_project_scope` con `status='ACTIVE'` y usa `idx_project_documents_despacho`.
- [ ] **Step 4:** Correr → PASS. `EXPLAIN` de `listar` y `tomar_en_cola` sobre `jax_test` con 1 000 filas sembradas: ambos con `key` = el índice esperado, sin `Using filesort`. Pegar la salida en el mensaje del commit.
- [ ] **Step 5:** Commit (`"feat(proyectos): E2a configuración, tipos y repositorio de documentos"`).

---

### Task 6: API de documentos (subir, listar, ocultar, restaurar, límites)

**Files:**
- Create: `backend/proyectos_documentos/almacen.py` (escritura en `entrada/`), `backend/api/proyectos_documentos.py` (router)
- Modify: `backend/main.py` (registrar el router junto a `proyectos_router`)
- Test: `backend/tests/test_proyectos_documentos_api.py`

**Interfaces:**
- Consumes: `repositorio` y `tipos` (Tarea 5); `ajustes.valor`; la verificación de papel de `backend/api/proyectos.py` (`get_current_user`, `get_pool()`, la consulta de proyecto visible + papel mínimo que ya usa ese router; traducción de errores con `_http`); `adjuntos.cuota.exigir_disco_libre`; `JAX_WORKSPACE_DIR` de `/etc/jax/.env`.
- Produces:
  - `GET /api/proyectos/documentos/limites` → `{"max_bytes_archivo", "max_archivos_lote", "max_bytes_lote", "extensiones": [...]}` (cualquier usuario con sesión).
  - `POST /api/proyectos/{id}/documentos` (multipart, campo `archivos` repetido; CONTRIBUTOR; proyecto ACTIVE) → **202** `{"lote": str, "aceptados": [{"id","nombre"}], "ignorados": [{"nombre","motivo"}]}` con `motivo` ∈ `tipo_no_admitido | demasiado_grande | duplicado | duplicado_oculto | nombre_invalido`. 413 `lote_demasiado_grande` si se cruza el tope de archivos o bytes del lote (lo ya escrito y registrado se conserva). 409 `proyecto_no_activo`. 507 `sin_espacio`.
  - `GET /api/proyectos/{id}/documentos?vista=visibles|ocultos&antes_de=&limite=` (VIEWER; `ocultos` exige CONTRIBUTOR) → `{"documentos": [...], "siguiente": int | null}`.
  - `POST /api/proyectos/{id}/documentos/{doc}/ocultar` y `/restaurar` (CONTRIBUTOR) → 204; 404 si el documento no es del proyecto.
  - `almacen.py`: `def carpeta_entrada(workspace: Path, project_uuid: str, lote: str) -> Path`; `def nombre_seguro(original: str, usados: set[str]) -> str`; `async def escribir_streaming(upload, destino: Path, tope: int) -> tuple[int, str]` → `(bytes, sha256)`, lanza `DemasiadoGrande` al cruzar `tope` y borra el parcial.

- [ ] **Step 1: Pruebas que fallan** (cliente de prueba y cabeceras de `tests/identidades.py`, workspace en `tmp_path` vía `monkeypatch.setenv("JAX_WORKSPACE_DIR", ...)`, topes bajados por ajuste para no generar GBs):

```python
def test_contributor_sube_y_queda_en_cola(cli, proyecto, contributor, tmp_path):
    r = cli.post(f"/api/proyectos/{proyecto.id}/documentos", headers=cabeceras(contributor),
                 files=[("archivos", ("a.pdf", b"%PDF-1.4 a", "application/pdf")),
                        ("archivos", ("b.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 202
    cuerpo = r.json()
    assert [a["nombre"] for a in cuerpo["aceptados"]] == ["a.pdf"]
    assert cuerpo["ignorados"] == [{"nombre": "b.exe", "motivo": "tipo_no_admitido"}]
    escritos = list((tmp_path / "proyectos" / proyecto.uuid / "entrada").rglob("*.pdf"))
    assert len(escritos) == 1


def test_viewer_no_sube_403_y_no_escribe(cli, proyecto, viewer, tmp_path): ...
def test_no_miembro_404_igual_que_inexistente(cli, proyecto, ajeno): ...
def test_proyecto_archivado_409(cli, proyecto_archivado, contributor): ...


def test_tope_real_no_el_declarado(cli, proyecto, contributor, tmp_path, ajuste):
    ajuste("proyectos.documentos.max_bytes_archivo", "1048576")          # 1 MiB
    grande = b"%PDF" + b"0" * (1048576 + 10)
    r = cli.post(f"/api/proyectos/{proyecto.id}/documentos", headers=cabeceras(contributor),
                 files=[("archivos", ("ok.pdf", b"%PDF ok", "application/pdf")),
                        ("archivos", ("grande.pdf", grande, "application/pdf"))])
    assert r.status_code == 202
    assert {"nombre": "grande.pdf", "motivo": "demasiado_grande"} in r.json()["ignorados"]
    assert [p.name for p in (tmp_path / "proyectos").rglob("*.pdf")] == ["ok.pdf"]   # sin parcial


def test_lote_que_cruza_el_tope_de_archivos_413_conserva_lo_anterior(cli, proyecto, contributor, ajuste): ...


def test_duplicado_y_duplicado_oculto(cli, proyecto, contributor): ...


async def test_duplicado_concurrente_una_sola_fila(cli_async, proyecto, contributor):
    # dos POST simultáneos con el mismo contenido -> una fila, el otro ignorado como duplicado
    ...


def test_nombres_hostiles(cli, proyecto, contributor, tmp_path):
    nombres = ["../../etc/passwd.pdf", "con\x00nul.pdf", "x" * 600 + ".pdf", "🙂🙂.pdf",
               "Informe.PDF", "informe.pdf"]
    files = [("archivos", (n, f"%PDF {i}".encode(), "application/pdf")) for i, n in enumerate(nombres)]
    r = cli.post(f"/api/proyectos/{proyecto.id}/documentos", headers=cabeceras(contributor), files=files)
    raiz = (tmp_path / "proyectos" / proyecto.uuid / "entrada").resolve()
    for p in (tmp_path / "proyectos").rglob("*"):
        assert p.resolve().is_relative_to(raiz) or p.is_dir()
    en_disco = [p.name for p in raiz.rglob("*") if p.is_file()]
    assert len(en_disco) == len(set(n.lower() for n in en_disco))       # únicos sin distinguir mayúsculas


def test_ocultar_restaurar_y_vista_ocultos(cli, proyecto, contributor, viewer): ...
def test_limites_publica_los_ajustes_y_las_extensiones(cli, contributor): ...
```

(Los `...` se escriben completos en el mismo estilo; cada uno con su aserción de código HTTP, cuerpo y estado de disco/base.)

- [ ] **Step 2:** Correr → FAIL.
- [ ] **Step 3: Implementar.**
  - Orden por archivo: tipo (`tipos.tipo_de`) → nombre seguro → escritura streaming con tope por archivo (calcula `sha256` mientras escribe, en `asyncio.to_thread` por bloques de 1 MiB, igual que `adjuntos/almacen.copiar_subida`) → `repositorio.insertar` → si devuelve `None`, borrar el archivo escrito y `existente_por_sha` decide `duplicado` / `duplicado_oculto`.
  - Antes del primer archivo: `exigir_disco_libre(carpeta, max_bytes_lote)`. Contadores del lote (archivos y bytes reales, no los declarados): al cruzar un tope, 413 con lo ya aceptado intacto.
  - `nombre_seguro`: NFC, quita separadores de ruta, `\x00` y controles, recorta a 200 caracteres conservando la extensión, y si ya está usado (comparación sin distinguir mayúsculas) agrega ` (2)`, ` (3)`… Nunca vacío (`documento.<ext>`).
  - `lote` = `secrets.token_urlsafe(12)`; `ruta_entrada` relativa a `JAX_WORKSPACE_DIR` (`proyectos/<uuid>/entrada/<lote>/<nombre>`), que es la ruta que LAS MANOS recibe.
  - Esta tarea **no** llama a LAS MANOS: las filas quedan `en_cola`. El aviso inmediato al despachador lo agrega la Tarea 7.
  - Proyecto, membresía y papel: reutilizar exactamente la verificación que usa `backend/api/proyectos.py` para sus endpoints de lectura y de miembros; ningún SQL de papeles nuevo en este router.
- [ ] **Step 4:** Correr → PASS, y `test_proyectos_api.py` sigue verde.
- [ ] **Step 5:** Commit (`"feat(proyectos): E2a API de documentos: subida por lotes, lista, ocultar y restaurar"`).

---

### Task 7: Despachador y sincronizador de fondo

**Files:**
- Create: `backend/proyectos_documentos/despachador.py`
- Modify: `backend/main.py` (`asyncio.create_task(despachador.start_despachador())` junto a `start_limpieza_de_adjuntos`, línea ~189)
- Modify: `backend/api/proyectos_documentos.py` (al terminar un lote con al menos un aceptado, `despachador.despachar_ahora()`, sin esperar)
- Test: `backend/tests/test_proyectos_documentos_despachador.py`

**Interfaces:**
- Consumes: `repositorio` (Tarea 5); `http_client.get_http_client()`, `jax_engine.state.LAS_MANOS_URL`, `credencial_las_manos.encabezados_las_manos()`; `ajustes.valor("proyectos.documentos.rutas_por_trabajo")`.
- Produces:
  - `async def ciclo(pool) -> None` — una vuelta: (1) `GET_LOCK('proyectos_documentos_despacho', 0)`; si no lo obtiene, vuelve; (2) sincroniza cada `job_id` abierto; (3) despacha filas `en_cola` agrupadas por proyecto en trozos de `rutas_por_trabajo`; (4) `RELEASE_LOCK`.
  - `def despachar_ahora() -> None` — programa un `ciclo` inmediato (sin esperar).
  - `async def start_despachador()` — bucle de fondo: `ciclo` cada `INTERVALO_SEGUNDOS = 10`; nunca muere por un fallo (mismo patrón y comentario que `start_limpieza_de_adjuntos`).
  - Mapeo de estados de LAS MANOS → `project_documents.estado`: `ok`→`listo`, `parcial`→`parcial`, `error`/`rechazado`→`error` (con `error`), `sin_extractor`→`sin_extractor`, `cancelado`→`cancelado`; trabajo `running` sin resultado del archivo → `procesando`.
  - Cuando una fila llega a estado terminal, borra su archivo de `entrada/` (y la carpeta del lote si quedó vacía). El original ya está en `fuente/`.

- [ ] **Step 1: Pruebas que fallan** (LAS MANOS falsa con `httpx.MockTransport`, base de test real):
  - `test_despacha_en_trozos_de_rutas_por_trabajo` (120 filas `en_cola`, `rutas_por_trabajo=50` → 3 POST con 50/50/20 rutas, cuerpos con `project_uuid` y `usuario` = email del que subió).
  - `test_429_deja_en_cola_y_reintenta` (primer POST 429 → filas siguen `en_cola`; siguiente `ciclo` con 202 → `pendiente` con `job_id`).
  - `test_las_manos_caida_no_pierde_filas` (`httpx.ConnectError` → `en_cola`, sin excepción hacia afuera).
  - `test_sincroniza_resultados_y_borra_entrada` (GET devuelve `completed` con un `ok` y un `error` → `listo` con `carpeta_procesado` y `error` con su causa; los dos archivos de `entrada/` ya no están).
  - `test_trabajo_perdido_pasa_a_error` (GET 404 → `error`, `trabajo_perdido`).
  - `test_dos_ciclos_simultaneos_no_despachan_dos_veces` (dos `ciclo` concurrentes → un solo POST por trozo, gracias a `GET_LOCK`).
  - `test_proyecto_archivado_con_filas_en_cola_no_se_despacha` (las filas quedan `en_cola`; al reactivar el proyecto, salen).
  - `test_subida_avisa_al_despachador` (un POST de la API con un aceptado llama a `despachar_ahora` una vez; con cero aceptados, ninguna).
- [ ] **Step 2:** Correr → FAIL.
- [ ] **Step 3:** Implementar. `GET_LOCK` en una conexión dedicada del pool que se mantiene durante todo el `ciclo`. Timeouts HTTP: 10 s. Un 422 `proyecto_no_activo` de LAS MANOS deja las filas `en_cola` (el proyecto se archivó entre medio). Cualquier otro 4xx pasa las filas del trozo a `error` con el código recibido.
- [ ] **Step 4:** Correr → PASS.
- [ ] **Step 5:** Commit (`"feat(proyectos): E2a despachador de fondo: reparte en trabajos de LAS MANOS y sincroniza el estado"`).

---

### Task 8: Cliente y textos (frontend)

**Files:**
- Modify: `frontend/src/api/proyectos.js` (funciones nuevas), `frontend/src/api/proyectos.test.js`
- Modify: `frontend/src/i18n/es.js`, `frontend/src/i18n/en.js` (raíz `proyectos.documentos`)

**Interfaces:**
- Produces:
  - `limitesDeDocumentos()` → `GET /proyectos/documentos/limites`.
  - `subirDocumentos(id, archivos: File[], { onProgreso })` → `POST /proyectos/{id}/documentos` con `FormData` (campo `archivos` repetido; para carpetas, el nombre es `file.webkitRelativePath || file.name`) y `onUploadProgress` de axios → `onProgreso(0..1)`.
  - `listarDocumentos(id, { vista = 'visibles', antesDe = null, limite = 50 })`, `ocultarDocumento(id, doc)`, `restaurarDocumento(id, doc)`.
  - Textos `proyectos.documentos.*`: `pestana`, `agregar`, `vacio`, `estados.{en_cola,pendiente,procesando,listo,parcial,error,sin_extractor,cancelado}`, `motivos.{tipo_no_admitido,demasiado_grande,duplicado,duplicado_oculto,nombre_invalido}`, `resumen.{titulo,archivos,peso,tipos,ignorados,confirmar,cancelar}`, `ocultar`, `restaurar`, `verOcultos`, `subiendo`, `elegirProyecto.{titulo,texto,crear}`, `sinPermiso`, `errores.{lote_demasiado_grande,proyecto_no_activo,sin_espacio}`, `boton` (aria-label del botón de documentos).
- [ ] **Step 1:** Pruebas en `proyectos.test.js` (axios mockeado, mismo estilo que las de E1): URL, método, `FormData` con dos archivos y `webkitRelativePath`, y que `onProgreso` recibe la fracción. Una prueba que compara las claves de `proyectos.documentos` en `es.js` y `en.js` (mismo conjunto).
- [ ] **Step 2:** Correr `npx vitest run src/api/proyectos.test.js` → FAIL.
- [ ] **Step 3:** Implementar y escribir los textos en los dos idiomas.
- [ ] **Step 4:** Correr → PASS.
- [ ] **Step 5:** Commit.

---

### Task 9: El Selector (resumen, confirmación y subida)

**Files:**
- Create: `frontend/src/components/proyectos/SelectorDeDocumentos.jsx`, `frontend/src/components/proyectos/resumenDeLote.js`
- Test: `frontend/src/components/proyectos/SelectorDeDocumentos.test.jsx`, `frontend/src/components/proyectos/resumenDeLote.test.js`

**Interfaces:**
- Consumes: Tarea 8; `components/Dialogo.jsx` (`idTitulo, titulo, onCerrar, cerrable, className, children`).
- Produces:
  - `resumirLote(files: File[], limites) -> { aceptados: File[], ignorados: [{nombre, motivo}], totalBytes, porTipo: {[ext]: n}, excedeLote: 'archivos'|'bytes'|null }` (pura; aplica tipo, tope por archivo, duplicados por nombre+tamaño dentro del lote y los dos topes de lote).
  - `<SelectorDeDocumentos proyectoId onTerminado(resultado) onCerrar />` — dos `<input type="file">` ocultos (uno `multiple`, otro `webkitdirectory`), botones «Archivos» y «Carpeta»; al elegir, abre `Dialogo` con el resumen (cuántos, cuánto pesan, de qué tipos, cuáles se ignoran y por qué); «Subir» deshabilitado si `excedeLote` o `aceptados` vacío; al confirmar, sube con barra de avance y muestra el resultado del servidor (aceptados e ignorados con sus motivos traducidos).
- [ ] **Step 1:** Pruebas: `resumirLote` (tipo no admitido, demasiado grande, 251 archivos → `excedeLote='archivos'`, 1 GB + 1 byte → `'bytes'`, duplicado dentro del lote); el componente: abre el resumen con los números correctos, Escape cierra sin subir, «Subir» llama a `subirDocumentos` una sola vez aunque se haga doble clic, la barra refleja `onProgreso`, y un 413 del servidor se muestra con su texto. En el archivo de prueba, una aserción de que el código del componente no contiene `confirm(`, `alert(` ni `prompt(`.
- [ ] **Step 2:** `npx vitest run src/components/proyectos/` → FAIL.
- [ ] **Step 3:** Implementar con tokens de color, foco inicial en «Subir» y retorno del foco al botón que abrió el diálogo.
- [ ] **Step 4:** → PASS.
- [ ] **Step 5:** Commit.

---

### Task 10: Pestaña Documentos

**Files:**
- Create: `frontend/src/components/proyectos/Documentos.jsx`
- Modify: `frontend/src/pages/ProyectoDetalle.jsx` (pestaña `documentos` primera para todos los miembros; orden `documentos, miembros, ajustes`)
- Test: `frontend/src/components/proyectos/Documentos.test.jsx`, `frontend/src/pages/ProyectoDetalle.test.jsx` (casos nuevos)

**Interfaces:**
- Consumes: Tareas 8 y 9; `proyecto.papel` y `esAdmin` como ya los usa `ProyectoDetalle.jsx`.
- Produces: lista paginada (nombre, tamaño legible, quién lo subió, fecha, estado con su texto y color de token; `error` muestra la causa); «Agregar documentos» (solo CONTRIBUTOR+, abre el Selector); «Ocultar» por fila y conmutador «Ver ocultos» con «Restaurar» (solo CONTRIBUTOR+). Mientras haya filas en `en_cola`/`pendiente`/`procesando`, vuelve a pedir la lista cada 5 s; con todas terminales, deja de pedir. Al desmontar, cancela el temporizador.
- [ ] **Step 1:** Pruebas: VIEWER ve la lista sin «Agregar» ni «Ocultar»; CONTRIBUTOR ve ambos; el sondeo se detiene cuando todo es terminal (temporizadores falsos de vitest); ocultar saca la fila y aparece en «Ver ocultos»; la navegación por teclado de pestañas de `ProyectoDetalle` sigue funcionando con tres pestañas.
- [ ] **Step 2:** → FAIL. **Step 3:** Implementar. **Step 4:** → PASS. **Step 5:** Commit.

---

### Task 11: Selector de proyecto y botón de documentos en Chat y Pipeline

**Files:**
- Modify: `frontend/src/components/BottomBar/BottomBar.jsx` (hoy `SelectorDeProyecto` solo con `mode === 'chat'`, líneas ~73 y ~158)
- Create: `frontend/src/components/BottomBar/BotonDocumentos.jsx`
- Test: `frontend/src/components/BottomBar/BottomBar.proyecto.test.jsx` (casos nuevos), `frontend/src/components/BottomBar/BotonDocumentos.test.jsx`

**Interfaces:**
- Consumes: `useJaxStore` (`proyectoActivo`, `setProyectoActivo`), `SelectorDeProyecto`, Tarea 9.
- Produces: el selector de proyecto se ve en `chat` y `pipeline` (no en `ejecutor` ni en `imagen`), con el mismo tamaño y estilo de E1.1. `BotonDocumentos` junto al selector en esos dos modos:
  - con proyecto elegido y papel CONTRIBUTOR+ → abre `SelectorDeDocumentos` para ese proyecto;
  - con proyecto elegido y papel VIEWER → deshabilitado con `title`/`aria-label` `sinPermiso`;
  - en «Personal» → abre un `Dialogo` `elegirProyecto` con la lista de proyectos activos (reusa `SelectorDeProyecto`) y el enlace «Crear proyecto» (abre `CrearProyectoModal`); al elegir, sigue al Selector.
  - El botón de adjuntar del chat (`AttachButton`) **no cambia**: los adjuntos de un mensaje siguen siendo cosa del chat hasta E3.
- [ ] **Step 1:** Pruebas: selector visible en `chat` y `pipeline`, ausente en `ejecutor` e `imagen`; los tres casos del botón; cambiar de modo conserva el proyecto elegido; la prueba de tamaño (`politica/tamano44Proyectos.test.jsx`) cubre el botón nuevo con la regla de 24 px de E1.1.
- [ ] **Step 2:** → FAIL. **Step 3:** Implementar. **Step 4:** → PASS. **Step 5:** Commit.

---

### Task 12: Las Cuatro, pisos y cierre de la Parte B

**Files:**
- Create: `loadtest/proyectos_e2a.py`, `loadtest/test_proyectos_e2a.py` (mismo patrón que `loadtest/proyectos_e1.py`)
- Modify: `.github/workflows/policy.yml` (pisos medidos en el runner)

- [ ] **Step 1: Carga, peor caso** (adenda §6.4), contra una instancia de prueba en hall9000 con base de prueba y workspace en un directorio temporal, nunca producción: un lote del tamaño de LACTOVI (**120 archivos, 295 MB**, generados con los tamaños reales de `fuente/` de LACTOVI, sin copiar su contenido) mientras **20 usuarios** chatean; medir p95 del turno de chat con y sin la subida, y el tiempo hasta que la última fila deja `en_cola`. Además, 10 subidas simultáneas de lotes chicos a proyectos distintos: 0 errores 5xx.
- [ ] **Step 2:** `EXPLAIN` de las consultas de la Tarea 5 sobre la base de carga (10 000 filas en `project_documents`): sin `Using filesort` ni `Using temporary`.
- [ ] **Step 3:** Revisión visual en claro y oscuro de la pestaña, el Selector y el botón (capturas en `docs/`), y barrido de `confirm(`/`alert(`/`prompt(` desnudos en `frontend/src`.
- [ ] **Step 4:** Pisos de `policy.yml` con el número del runner; commit con los números de carga en el mensaje y en `docs/historia/2026-10-03-proyectos-e2a.md` del repo jax (va en el PR de la Parte C).
- [ ] **Step 5:** Auditoría de escalón 3 sobre la rama completa; PR a `master`; se integra con CI verde y sin BLOCK/MAJOR abiertos.

---

## Parte C — producción (sesión principal, con GO de Fernando)

### Task 13: Despliegue coordinado

Sigue `docs/runbooks/proyectos-e2a-produccion.md` (Tarea 4). Cada paso con su verificación antes del siguiente:

1. Respaldo de `jax_memory` con restauración probada (`~/respaldos-despliegue/2026-10-0X-e2a/`).
2. `ops/permisos_proyectos.py --verificar`, luego `--aplicar`, luego la prueba real como `jaxsvc` en verde.
3. jax a producción (LAS MANOS con `project_uuid`); reiniciar `jax-las-manos`; un `POST /procesamiento/trabajos` con un uuid inexistente da 422 `proyecto_no_activo`.
4. jax-platform a producción; al arrancar, 006a crea `project_documents` y se siembran las cuatro claves (verificar con `SELECT`); publicar el sitio.
5. LACTOVI: `scripts/proyectos_e2a_lactovi.py` en ensayo, revisar la salida, luego `--aplicar`; conteos y `sha256` iguales; la pestaña Documentos de LACTOVI muestra sus documentos.
6. **Restauración probada de `proyectos/` con la ruta nueva**: esperar el snapshot de restic posterior al traslado (o lanzar uno), restaurar `proyectos/<uuid-lactovi>/` a una ruta aparte y comparar `sha256` contra las fichas. **Sin esto, la subida no se anuncia como disponible.**
7. Prueba de humo: Fernando sube 3 documentos a LACTOVI desde el Chat y los ve llegar a `listo`.
8. Biblioteca: `docs/historia/2026-10-03-proyectos-e2a.md` con lo hecho, números de carga, `du` de `proyectos/` antes y después (costo de R2), y lo que queda para E2b.
