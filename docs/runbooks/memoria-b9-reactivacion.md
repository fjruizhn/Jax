# Reactivación observada de memoria B9

Este procedimiento no concede permiso de producción. Requiere autorización del
responsable y un SHA integrado con revisión independiente; los snapshots no
sustituyen la autoridad y el estado vigente consultables mediante `jaxctl`.

Lectura de la base de producción (solo `SELECT`, `SHOW` y `EXPLAIN`): MariaDB en
`127.0.0.1:3308` (la 3306 está muerta). La contraseña va en un archivo de opciones
`600` que se borra al salir, nunca como `-p...` en la línea de comandos:

```bash
( set -euo pipefail; set -a; . <(sudo -n cat /etc/jax/.env); set +a
  T=$(mktemp -d); chmod 700 "$T"; trap 'rm -rf "$T"' EXIT
  printf '[client]\nhost=127.0.0.1\nport=3308\nuser=%s\npassword="%s"\ndatabase=%s\n' \
    "$JAX_DB_USER" "$JAX_DB_PASSWORD" "$JAX_DB_NAME" > "$T/c.cnf"; chmod 600 "$T/c.cnf"
  mariadb --defaults-extra-file="$T/c.cnf" -t < /ruta/a/la-consulta.sql )
```

(El archivo `/etc/jax/.env` es `root:jaxsvc 640`: se lee con `sudo -n cat`, como exige
`tests/test_env_se_lee_con_sudo.py`; el subshell evita que las variables queden en tu sesión.)

## 1. Propiedad de archivos, timers y SHA

Coordinar la propiedad de archivos (`/srv/jax-prod/jax` es de `jaxsvc` y debe estar en
`master` y limpio: el freno `jax-checkout-de-produccion-sano.sh` lo exige) y registrar
el SHA desplegado: `sudo git -c safe.directory=/srv/jax-prod/jax -C /srv/jax-prod/jax rev-parse HEAD`.
Conservar los horarios y drop-ins instalados (copia fechada de `/etc/systemd/system/jax-*`).

**Los timers pueden estar habilitados.** El 2026-10-05 lo estaban `jax-memory-embedding`,
`jax-memory-vector-health`, `jax-memory-lifecycle` y `jax-revisar-indice-vectorial`; los de
`jax-memory-worker` y `jax-memory-synthesis` no. Es una VERDAD OPERACIONAL que caduca:
se mide el día que se hace, no se copia de aquí.

1. **Registrar el estado**, a un archivo fechado fuera de `/` y de `~`:
   `systemctl list-unit-files 'jax-memory-*.timer' jax-revisar-indice-vectorial.timer --no-legend`
   (columna `enabled`/`disabled`) y `systemctl list-timers 'jax-memory-*' jax-revisar-indice-vectorial.timer --all`
   (próximos y últimos disparos).
2. **Detener los habilitados de modo que un reinicio del servidor no los rearme**:
   `sudo systemctl disable --now <timer>` para cada uno que el registro marque `enabled`
   (`stop` a secas los deja habilitados y volverían al arrancar).
3. **Esperar a que no haya corridas en curso**: `systemctl is-active jax-memory-worker.service jax-memory-embedding.service jax-memory-synthesis.service jax-memory-lifecycle.service jax-memory-vector-health.service jax-revisar-indice-vectorial.service`
   no debe mostrar `activating` ni `active`. No se corta una corrida a medias (un worker a
   mitad de un commit queda `UNKNOWN`; ver `docs/runbooks/memoria-cola-atascada.md`).
4. **Comprobar que están detenidos**: `systemctl list-timers --all` ya no los muestra con un próximo disparo.

El paso 8 devuelve cada timer a **lo que decía el registro**, no a lo que parezca razonable.

## 2. Respaldo con restauración

Respaldar íntegramente la base, restaurarla en una instancia sin red y comprobar
tablas y conteos. Conservar dump, manifiesto y planes con permisos privados.

## 3. Base fresca de pruebas y driver de regresión

`tests/memory_b9_regression_driver.py` ejercita el worker, la API, los permisos, los
triggers, los locks y los commits contra una MariaDB real; solo el proveedor (el extractor)
está simulado. Corre como archivo Python, nunca con pytest, y exige una base **fresca**.

**Fixtures** (los construye `tests/memory_b9_provision.py`; el CI hace lo mismo en el job
`memory-b9-regression`, sobre un segundo servicio MariaDB, así el driver no vuelve a quedar viejo):

- base nueva `jax_memory_test_memb9_<sufijo>`, usuario `jax_test` (con permisos solo sobre ella);
- `jax_memory_schema.sql` **sin** sus dos primeras sentencias (`CREATE DATABASE IF NOT EXISTS jax_memory`
  y `USE jax_memory`): cargarlo crudo con el cliente apuntaría a la base de **producción**. El
  constructor las quita y falla cerrado si el archivo cambia;
- `jax_tenants` y `jax_users` mínimas (con la forma de producción; las crea jax-platform, no este repo, y
  la migración 003 tiene una FK a `jax_users`);
- migraciones `jax/memory/b9_migrations/` 001 → 004, luego **005**, que solo existe en Python
  (`apply_project_authority_migration`), luego 006 → 013;
- filas: `jax_tenants` 1 y 2; `jax_users` 1 (tenant 1) y 2 (tenant 2), activos y con rol `admin` (la
  adopción legacy exige un administrador real), y 4 (tenant 1, activo, rol `operator`): el miembro sin
  privilegio al que se le deniega la importación y que no debe ver lo privado del usuario 1.

```bash
# MariaDB desechable, solo en loopback y en un puerto libre distinto de 3306/3308; el archivo de entorno es 600
docker run -d --name b9-driver -p 127.0.0.1:3399:3306 --env-file /ruta/privada/b9.env mariadb:12.3.3
# b9.env: MARIADB_ROOT_PASSWORD, MARIADB_DATABASE=jax_memory_test_memb9_<sufijo>, MARIADB_USER=jax_test, MARIADB_PASSWORD
export JAX_DB_HOST=127.0.0.1 JAX_DB_PORT=3399 JAX_DB_USER=jax_test JAX_DB_NAME=jax_memory_test_memb9_<sufijo> PYTHONPATH=.
export JAX_DB_PASSWORD=...        # de b9.env, sin escribirla en la línea de comandos
python tests/memory_b9_provision.py                 # PROVISION_OK jax_memory_test_memb9_<sufijo>
python tests/memory_b9_regression_driver.py         # JSON con "status": "PASS" y 19 casos
docker rm -f -v b9-driver                           # -v: sin él el datadir queda huérfano
```

Fallos de guarda: `test_database_guard` (usuario o nombre de base equivocados), `fresh_database_required`
(la base ya tiene tablas: el constructor nunca borra; se usa otra) y `test_configuration_missing`.
Medido el 2026-10-05 contra MariaDB 12.3.3 desechable: 19 casos PASS. Complementar con las
regresiones unitarias y las pruebas SQL de atomicidad.

## 4. Migración de jobs (006 y las que siguen): se aplica una vez, y primero se mira si ya está

El arranque de los workers no debe ejecutar migraciones ni modificar ownership legacy.
**006 no es idempotente** (`CREATE TRIGGER` y `ADD CONSTRAINT` sin `IF NOT EXISTS`): repetirla entera
falla o duplica. Antes de aplicar nada, se comprueba con estas consultas de solo lectura:

```sql
-- 006/007/009: seis tablas (jobs, resultados, síntesis, items de síntesis, eventos de jobs, intentos de embedding)
SELECT COUNT(*) AS n FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME IN
 ('memory_extraction_jobs','memory_extraction_results','memory_synthesis_jobs','memory_synthesis_job_items',
  'memory_extraction_job_events','embedding_generation_attempts');                                   -- esperado: 6
-- 006: tenant_id NOT NULL en memory_revisions, su trigger y su FK
SELECT IS_NULLABLE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='memory_revisions' AND COLUMN_NAME='tenant_id';   -- NO
SELECT COUNT(*) FROM information_schema.TRIGGERS WHERE TRIGGER_SCHEMA=DATABASE() AND TRIGGER_NAME='memory_revision_tenant_compat';                   -- 1
SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS WHERE CONSTRAINT_SCHEMA=DATABASE() AND TABLE_NAME='memory_revisions' AND CONSTRAINT_NAME='fk_memory_revision_tenant';  -- 1
-- 004: tenant_id NOT NULL en los bindings legacy
SELECT COUNT(*) FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='memory_legacy_bindings' AND COLUMN_NAME='tenant_id' AND IS_NULLABLE='NO';  -- 1
-- 005 (la aplica jax-platform al arrancar): constraint v2, tabla de creación idempotente y 3 columnas de procedencia
SELECT CONSTRAINT_NAME FROM information_schema.CHECK_CONSTRAINTS WHERE CONSTRAINT_SCHEMA=DATABASE() AND TABLE_NAME='jax_project_scope';            -- chk_jax_project_scope_status_v2
SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='jax_project_creation_request';                          -- 1
SELECT COUNT(*) FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='jax_project_membership' AND COLUMN_NAME IN ('grant_origin','pre_admin_role','pre_admin_status');  -- 3
-- índices que deben ESTAR (006, 008, 010, 011, 012, 004): 8 pares tabla/índice
SELECT TABLE_NAME, INDEX_NAME FROM information_schema.STATISTICS WHERE TABLE_SCHEMA=DATABASE() AND INDEX_NAME IN
 ('uq_memory_object_tenant','ix_memory_revision_private_feed','ix_memory_revision_shared_feed','ix_memory_provenance_revision_order',
  'uq_embedding_generation_revision_space','idx_messages_conversation_turn','idx_conversations_open','uq_memory_legacy_binding_tenant')
 GROUP BY TABLE_NAME, INDEX_NAME;                                                                   -- 8 filas
-- índices que deben FALTAR (010, 013, 004): ninguna fila
SELECT TABLE_NAME, INDEX_NAME FROM information_schema.STATISTICS WHERE TABLE_SCHEMA=DATABASE() AND
 ((TABLE_NAME='embedding_generations' AND INDEX_NAME='idx_embedding_generation_revision')
  OR (TABLE_NAME='messages' AND INDEX_NAME='idx_conversation')
  OR (TABLE_NAME='memory_objects' AND INDEX_NAME='uq_memory_legacy_binding')) GROUP BY TABLE_NAME, INDEX_NAME;      -- 0 filas
```

Medido el 2026-10-05 sobre producción: **todas** dan lo esperado, es decir, 001-013 ya están aplicadas.
En ese caso el paso es «verificado, nada que aplicar». Si alguna da otra cosa, se inspeccionan columnas,
triggers, constraints e índices y se completan **exclusivamente los pasos faltantes**, uno por uno, con la
ventana y el respaldo del paso 2 (011, 012 y 013 bloquean escrituras 1-3 s: ver
`jax/memory/b9_migrations/README.md`). Las migraciones 007-013 son repetibles (`IF [NOT] EXISTS`); 006 no.

## 5. Plan de adopción legacy: verificar si ya está aplicado y que repetirlo crea 0 eventos

Generar y revisar el plan privado de adopción (`build_plan` de `jax/memory/legacy_adoption.py`, archivo
`0600`). Aplicarlo mediante la API y un administrador real con `ops/memory/adopt_legacy.py`
(`--env-file`, `--plan`, `--backup-manifest`, `--actor-user-id`, `--expected-database`,
`--verify-owner-id`, `--verify-other-user-id`).

**Comprobar si ya está adoptado** (solo lectura). Cada fila adoptada deja un binding y exactamente
**dos eventos** (`IMPORT_LEGACY` y `RE_SCOPE`):

```sql
SELECT event_kind, COUNT(*) AS n FROM memory_events GROUP BY event_kind ORDER BY 1;
SELECT legacy_source_type, COUNT(*) AS n FROM memory_legacy_bindings GROUP BY legacy_source_type;
SELECT COUNT(*) AS importaciones_sin_binding FROM memory_events e
 WHERE e.event_kind='IMPORT_LEGACY' AND NOT EXISTS (SELECT 1 FROM memory_legacy_bindings b WHERE b.memory_id=e.memory_id);   -- 0
```

Esperado: `IMPORT_LEGACY` = `RE_SCOPE` = total de `memory_legacy_bindings`, y 0 importaciones sin binding.
El 2026-10-05 en producción: 136 / 136 / 136 (96 facts, 22 decisions, 18 action_items), 0 sin binding.
Las filas legacy que el plan **cuarentena** (sin dueño, vencidas, no elegibles) tampoco tienen binding y no son
un error: contar «filas legacy sin binding» no distingue adoptables de cuarentenadas; esa lista la da el plan.

**Repetirlo no puede crear eventos.** `ops/memory/adopt_legacy.py` aplica el plan, lo vuelve a aplicar y
compara `SELECT COUNT(*) FROM memory_events` antes y después de la repetición; si difiere, falla cerrado con
`adoption_repeat_created_events`. En su JSON de salida: `repeat_events_added` tiene que ser `0`; sobre una
base **ya adoptada**, también `events_added` es `0`. Se confirma con las cuentas de `memory_events` y
`memory_legacy_bindings` iguales antes y después, y un segundo `SELECT` independiente del anterior.
Verificar lectura del dueño, aislamiento entre usuarios (`administrator_reader_counts`) y reconciliación.

## 6. Unidades a mano antes de habilitar timers

Ejecutar unidades manualmente antes de habilitar timers. Verificar estado de salida, registros durables y
duración; un lote con fallos debe terminar no cero aunque haya completado conversaciones sanas. Observar
embedding, lifecycle, síntesis y salud vectorial, incluyendo los casos sin trabajo elegible.

## 7. Conversación real

Con una sesión autenticada, comprobar un recuerdo legacy y una conversación nueva. Cerrar únicamente la
conversación controlada mediante su frontera autorizada; observar extracción y lectura B9, sin marcar
procesado a mano.

## 8. Timers y cierre

Habilitar timers sólo tras resultados satisfactorios. Los que el registro del paso 1 marcaba `enabled` vuelven con
`sudo systemctl enable --now <timer>`; los que estaban `disabled` (worker y synthesis el 2026-10-05) se quedan así
hasta que el responsable decida lo contrario. Registrar próximos disparos (`systemctl list-timers --all`,
comparados con los del registro), identidad del código, evidencia y límites de lo realmente probado.

Ante fallo, detener la unidad afectada y conservar jobs, datos y evidencia. No borrar resultados ni reiniciar
estados para obtener una salida verde. La recuperación debe respetar el resultado congelado y los marcadores de
commit. La marcha atrás del código y del esquema está en `docs/runbooks/deployment-rollback.md`.

La migración 006 incorpora DDL de tenant, FK, trigger e índices. Tras una interrupción no debe ejecutarse
completa a ciegas: inspeccionar columnas, triggers, constraints e índices (las consultas del paso 4) y completar
exclusivamente los pasos faltantes. La validación de readiness detecta una instalación incompleta sin repararla.
Síntesis excluye scopes con user_id NULL y exige fuentes B9 verificadas; un ciclo sin fuentes elegibles no
demuestra síntesis de memoria de proyecto.

## `jax.faro` en este despliegue: qué unidad lo corre y por qué falta `mcp`

Medido el 2026-10-05, en solo lectura:

- **Ninguna unidad de systemd corre `jax.faro`.** No hay referencias a `faro` en `config/systemd/`, en
  `/etc/systemd/system`, `/usr/lib/systemd/system`, `/run/systemd/system`, `/etc/cron*`, `/etc/sudoers.d` ni en
  `~/.config/systemd`; `systemctl list-unit-files` no tiene ninguna unidad con ese nombre. El `main` de
  `jax/faro/servicio.py` solo **verifica el arranque** (no sirve): el bucle del servicio y su unidad son el paso 0.10
  del plan de la fase 0 (`docs/superpowers/plans/2026-10-02-faro-fase-0.md`), todavía no hecho.
- **Nada de lo que corre en producción lo importa**: ni `jax-platform`, ni LAS MANOS, ni `jacobs`, ni `jaxctl`, ni las
  unidades `jax-memory-*`. `jax.faro` llega al servidor solo como archivos del checkout.
- **Los dos usos reales no necesitan `mcp`**: `projects/las-voces/authority/faro_readonly_mcp.py` (el MCP de solo
  lectura que lanza Qwen con `python3`) implementa el protocolo a mano e importa solo `jax.faro.config` y
  `jax.faro.paquete`, que son biblioteca estándar; y `ops/las-voces/faro_paquete_permanente.py` (como root, con
  `/usr/bin/python3 -I`) importa los mismos dos módulos.
- **Lo que sí importa `mcp`** es `jax/faro/puerto.py`, `jax/faro/transporte.py` y, por ellos, `jax/faro/servicio.py`
  (y las pruebas `tests/test_faro_*.py`). `mcp` no está ni en el venv de producción (`/srv/jax-prod/jax/.venv`) ni en
  `/usr/bin/python3`; está fijado con hashes en `requirements-faro.txt` (`mcp==2.2.0`), un archivo aparte a
  propósito, que instalan los jobs de CI de Faro con `--require-hashes`.
- **Conclusión para este despliegue:** el diff de `jax/faro` (memoria.buscar, `herramientas/memoria.py`, `puerto.py`,
  `servicio.py`, `transporte.py`) es código que **no se ejecuta** en producción todavía; que el venv no tenga `mcp` no
  rompe nada hoy. **Pendiente de otra persona (no se instala nada aquí):** cuando se haga el paso 0.10, la unidad del
  Faro necesita un intérprete con `requirements-faro.txt` instalado (`pip install --require-hashes`); usar el venv de
  producción sin esa instalación haría fallar `import jax.faro.servicio` con `ModuleNotFoundError: No module named 'mcp'`.
