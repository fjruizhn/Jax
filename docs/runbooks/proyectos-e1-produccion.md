# Proyectos E1 en producción
## Purpose
Llevar a `jax_memory` de producción: HAMURABI como proyecto 1 con alcance, el proyecto archivado «Evaluación grounding SP3 · 2026-09-03» que recibe los `project_id` huérfanos, y las llaves foráneas `project_id -> projects(id)` de las cinco tablas de contenido.
## Scope
`conversations`, `messages`, `facts`, `decisions`, `action_items` en `jax_memory` (MariaDB `127.0.0.1:3308`). Guiones: `scripts/proyectos_e1_migrar.py` y `scripts/proyectos_e1_fks.py`. No toca jax-platform salvo la verificación por efecto (paso 13).
## Preconditions
- Ventana abierta o GO de Fernando para producción (`bin/ventana estado` desde la sesión; si salió cerrada, parar).
- Ensayo hecho sobre una copia restaurada (paso 7). Lo hace la sesión principal con GO; sin sus números no se aplica nada.
- **El actor es siempre `--actor-user-id 1`.** El digest de idempotencia de `create_project` incluye al usuario: con otro actor, la segunda corrida da `IdempotencyKeyConflict`.
- **Dónde y con qué se corre (exacto).** Directorio `/srv/jax-prod/jax`, intérprete `/srv/jax-prod/jax/.venv/bin/python` (`.venv` es un enlace a `/opt/jax/venv`, Python 3.14.4 con `aiomysql`; verificado el 2026-10-02 leyendo el directorio sin `sudo`). Los guiones nunca abren `/etc/jax/.env`: el entorno `JAX_DB_*` lo carga quien opera con el mismo procedimiento de `docs/runbooks/despliegue.md §0` de jax-platform (`sudo bash -c 'set -a; . /etc/jax/.env; set +a; …'`), de modo que **la contraseña nunca va en la línea de comandos**. Atajo para este runbook (una función de la shell, pegar una vez):

  ```bash
  e1() {   # uso: e1 scripts/<guion>.py <argumentos>
    sudo bash -c 'set -euo pipefail; set -a; . /etc/jax/.env; set +a
      cd /srv/jax-prod/jax; export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:las_manos
      exec .venv/bin/python "$@"' _ "$@"
  }
  ```
  `/etc/jax/.env` define `JAX_DB_HOST`, `JAX_DB_PORT` (3308; la 3306 está muerta), `JAX_DB_USER`, `JAX_DB_PASSWORD` y `JAX_DB_NAME` (verificado el 2026-10-02 por la sesión principal, solo los nombres). Aun así los guiones de este runbook llevan `--database jax_memory` explícito; `revisar_indice_vectorial.py` toma la base de `JAX_DB_NAME` e **imprime su nombre en la primera línea**: debe decir `jax_memory:`.
- Las consultas SQL de este documento se corren con el archivo de opciones `600` que crea el §0 de ese mismo runbook (nunca `-p"$JAX_DB_PASSWORD"`).
## Authority impact
Escribe en producción: crea alcance y proyectos, reescribe `project_id` y crea FKs (DDL). Nada de esto se ejecuta sin ventana o GO. La migración de datos es idempotente y reversible con `--revertir`; el bootstrap de HAMURABI no se revierte (el alcance queda).
## Safe procedure
### 1. Ventana o GO
`bin/ventana estado` abierta, o GO de Fernando. Repetir `bin/ventana estado` antes de cada paso que escribe.
### 2. Respaldo con restauración probada (antes de todo)
**El respaldo es el §0 de `docs/runbooks/despliegue.md` de jax-platform** (léase con `git -C /home/fruiz/jax-platform show origin/master:docs/runbooks/despliegue.md`): volcado de `jax_memory` con `mariadb-dump --single-transaction --routines --triggers`, `ESTADO-ANTES.txt` con los SHAs desplegados, y **restauración probada** a una base descartable quitando los `DEFINER`, comparando conteos contra producción. Para esta migración el bucle de comparación del §0 se corre con estas tablas (en lugar de las que trae): `conversations messages facts decisions action_items projects jax_users`. Cada par tiene que dar el mismo número; si alguno difiere, no se sigue. Un volcado sin restauración probada no es respaldo (Principio VI). La carpeta `$D` del §0 es donde se guardan también el mapa de reversión y las salidas de los pasos 8 a 12; **la carpeta de resultados de otro despliegue no es el respaldo de este**.
### 3. Desplegar jax (T1-T3)
Orden obligado: **jax primero**, luego jax-platform. Con GO, con el procedimiento de `docs/runbooks/despliegue.md` de jax-platform (sección jax).
### 4. Desplegar jax-platform, UNA sola vez
La migración 005 (`apply_project_authority_migration`) no la corre jax: la corre **jax-platform al arrancar**. `backend/main.py:159` llama a `run_migrations()` y esta llama a `_apply_jax_project_authority_migration` (`backend/db/migrations.py`, `origin/master`), que carga `jax/memory/project_authority_migrations.py` desde `JAX_REPO_PATH`. O sea que lee el código de jax ya desplegado; no hay otro comando que la aplique. (Verificado leyendo `origin/master` de jax-platform; la bajada es `scripts/b9_revertir_005.py` de jax.)

El despliegue de jax-platform **se coordina con la otra sesión que lo tiene pendiente**: los PR #170-#174 de jax-platform no están desplegados y se despliega **una sola vez**, con todo junto, no uno por sesión. Los comandos son los de `docs/runbooks/despliegue.md` de jax-platform (con GO). No se avanza al paso 5 sin el despliegue hecho.
### 5. Verificar el índice de la migración 005e
En producción, ya arrancada jax-platform:

```sql
SHOW INDEX FROM jax_project_membership WHERE Key_name = 'idx_jax_project_membership_user_list';
```

Tiene que devolver filas. Si no, la migración no corrió: mirar el log de arranque de jax-platform (falla de `run_migrations`) y parar; no se aplica a mano.
### 6. Comprobar al actor y a los administradores (D2), solo lectura
```sql
SELECT user_id, tenant_id, role, status FROM jax_users WHERE user_id = 1;
SELECT id, name FROM projects WHERE id = 1;
SELECT user_id, email, role FROM jax_users
 WHERE tenant_id = 1 AND status = 'active' AND LOWER(role) IN ('admin','superadmin','super_admin') ORDER BY user_id;
```
- La primera tiene que devolver **una** fila: `tenant_id = 1`, `status = 'active'` y `role` `superadmin` (o `super_admin`). Si no existe, está inactivo, es de otro tenant o no es superadmin: **parar** (el bootstrap falla cerrado con `ProjectRoleInsufficient`/`TargetUserNotEligible` y no hay que forzarlo).
- La segunda tiene que devolver la fila de HAMURABI; si no, parar.
- La tercera es la lista de **quienes quedarán OWNER de HAMURABI** (D2: todo administrador activo del tenant 1 recibe OWNER). Anotarla en `$D` y que Fernando la apruebe: si hay alguien que no debe ser dueño, se corrige su rol **antes**, no después.
### 7. Ensayo sobre una copia restaurada (lo hace la sesión principal, con GO)
Va **después del paso 5**: `proyectos_e1_migrar.py` lee `jax_project_creation_request` y `jax_project_scope`, que crea la migración 005; una copia anterior a ella falla.
1. Restaurar el volcado del paso 2 con el procedimiento de restauración del §0 (con el `sed` que quita los `DEFINER`) a **`jax_memory_test_e1ensayo`** — el nombre pasa `es_base_de_test`, así que el barredor y las guardas de pruebas lo reconocen como desechable; **sin** el `DROP DATABASE` final del §0 (se borra en el punto 5 de este paso).
2. En la copia: `e1 scripts/proyectos_e1_migrar.py --verificar --actor-user-id 1 --database jax_memory_test_e1ensayo`, luego `--aplicar --actor-user-id 1 --salida-reversion <ruta nueva> --database jax_memory_test_e1ensayo`, y para probar la vuelta `--revertir <esa ruta> --database jax_memory_test_e1ensayo` (los huérfanos tienen que volver a sus `project_id` originales) y de nuevo `--aplicar` con otra ruta.
3. En la copia: `e1 scripts/proyectos_e1_fks.py --ensayar --database jax_memory_test_e1ensayo`. Anotar por tabla `algoritmo`, `segundos` y `hnsw_intacto`.
4. Solo van a producción (en `--tablas`) las tablas con `"algoritmo": "INPLACE"`, `"aplicada": true` y, si tienen índice vectorial (`messages`, `facts`), `"hnsw_intacto": true`. Las demás se registran en `DEUDA.md` con el motivo y la validación queda en la aplicación.
5. Borrar la copia (`DROP DATABASE jax_memory_test_e1ensayo`, solo esa).

> **Límite del ensayo, declarado.** Una copia restaurada **reconstruye** el índice vectorial al cargar los datos: su HNSW es nuevo y limpio. **No representa la historia del de producción**, que fue creciendo y recibiendo borrados en cascada (la causa medida del índice que miente, `docs/runbooks/indice-vectorial-envenenado.md`). Por eso un `hnsw_intacto: true` en el ensayo no prueba nada sobre producción: la medición que cuenta es la de producción, en los pasos 8, 10 y 12.
### 8. Medir el índice vectorial en producción, ANTES de migrar (solo lectura)
```bash
e1 scripts/revisar_indice_vectorial.py | tee $D/hnsw-1-antes-de-migrar.txt
```
Solo mira (no repara salvo `--reparar-de-verdad`, que aquí **no** se usa). Salida `0` = sanos; `1` = alguno miente o la caché está al límite; `2` = no pudo revisar. La primera línea debe ser `jax_memory:`. Anotar el resultado. Si algún índice miente, **parar** y seguir `docs/runbooks/indice-vectorial-envenenado.md`: migrar y poner FKs sobre un índice ya roto impide saber después qué lo rompió.
### 9. Medir y migrar los datos
```bash
e1 scripts/proyectos_e1_migrar.py --verificar --actor-user-id 1 --database jax_memory
```
Anotar `huerfanos_ids`, `filas_por_tabla` y `fuera_de_alcance`. **Si `fuera_de_alcance` trae algo** (ids fuera de `RESERVED_PROJECT_ID_RANGE` o filas de un usuario/tenant que no es el 1), `--aplicar` no escribirá nada y saldrá con 4: revisar esos casos a mano antes de seguir. Luego:
```bash
e1 scripts/proyectos_e1_migrar.py --aplicar --actor-user-id 1 --database jax_memory \
  --salida-reversion $D/mapa-reversion.json --confirmo-produccion
```
(`$D` lo expande tu shell antes de `sudo`; tiene que ser una ruta absoluta, p. ej. `D=~/respaldos-despliegue/AAAA-MM-DD-proyectos-e1` del §0.)
- `--salida-reversion` es obligatorio con `--aplicar` y la ruta tiene que **no existir** (si existe, aborta antes de tocar la base). Escribe un mapa JSON 0600: `{"evaluacion_project_id": N, "filas": [{"tabla", "id", "project_id_anterior"}]}`. Guardarlo fuera del repo y con el respaldo. **No se cambia su dueño**: lo crea quien corre la migración (con la función `e1`, root), queda 0600 y **`--revertir` lo rechaza (salida 2) si no es del usuario que lo corre o tiene permisos más abiertos que 0600**; por eso `--revertir` se corre también con `e1`, y el mapa se lee con `sudo` si hace falta. Un mapa que otro pudo escribir decide qué `UPDATE` corre la reversión.
- Comparar: `despues.huerfanos_ids` vacío y `filas_movidas` igual a `antes.filas_por_tabla`.
- Códigos de salida:
  - `0` hecho; repetirlo es seguro (idempotente).
  - `1` quedan huérfanos tras el commit (aparecieron después de medir), conteos que no coinciden (hizo rollback) o la ruta ya existía: **volver a correr con una ruta NUEVA**. El mapa anterior, si se escribió, se conserva.
  - `2` argumentos o guarda: falta la base, `--salida-reversion`, `--actor-user-id` o `--confirmo-produccion` sobre `jax_memory`. No se tocó nada.
  - `3` el commit quedó incierto: pudo aplicarse. El mapa quedó en `<ruta>.incierto` (o sigue en la ruta original si el mensaje dice que no se pudo renombrar). Correr `--verificar`: si no hay huérfanos, se aplicó (el mapa sirve para revertir); si siguen, no se aplicó y se vuelve a correr con ruta nueva. Decidir con esa medición, sin adivinar.
  - `4` huérfanos fuera de alcance (id fuera del rango reservado, o fila de otro tenant): **no se escribió nada**. No reintentar: revisar los ids/filas que imprime.
  - `5` error no previsto: **no reintentar sin revisar.** Correr `--verificar` y mirar el estado; no es «volver a correr».
  - `6` solo con `--revertir`: el mapa no cuadra con la base; rollback, nada cambió.
### 10. Verificación independiente tras migrar, y segunda medición del índice
Con la consulta propia (no con la salida del guion), por cada tabla y con `<evaluación>` = `evaluacion_project_id` del JSON:
```sql
SELECT 'conversations' AS t, COUNT(*) FROM conversations WHERE project_id = <evaluación>
UNION ALL SELECT 'messages',     COUNT(*) FROM messages     WHERE project_id = <evaluación>
UNION ALL SELECT 'facts',        COUNT(*) FROM facts        WHERE project_id = <evaluación>
UNION ALL SELECT 'decisions',    COUNT(*) FROM decisions    WHERE project_id = <evaluación>
UNION ALL SELECT 'action_items', COUNT(*) FROM action_items WHERE project_id = <evaluación>;
```
Cada conteo tiene que ser igual a `filas_movidas` de su tabla (y a las filas del mapa de esa tabla). Si no, parar. Después, entre migración y FKs:
```bash
e1 scripts/revisar_indice_vectorial.py | tee $D/hnsw-2-tras-migrar.txt
```
Comparar con el de antes; si algo pasó a mentir, **parar** (el `UPDATE` de `project_id` no toca el vector, pero se mide, no se supone).
### 11. FKs (ensayo y después `--aplicar --tablas`)
La lista de `--tablas` se arma **solo** con las tablas que en el ensayo dieron `"aplicada": true` y, si tienen índice vectorial, también `"hnsw_intacto": true`:
```bash
e1 scripts/proyectos_e1_fks.py --aplicar --confirmo-produccion --database jax_memory --tablas <t1,t2,...>
```
Sobre `jax_memory`, `--aplicar` sin `--tablas` sale con 2; un nombre fuera de las cinco, también. Una línea JSON por tabla: `tabla`, `filas`, `algoritmo`, `segundos`, `hnsw_intacto`, `aplicada` (y `ya_existia`, `error` o `motivo` si corresponde). **Repetirlo revalida, no da por bueno lo anterior:** con la FK ya creada no se vuelve a crear, pero se recuentan los huérfanos (si hay, la tabla sale con error y el código no es 0) y se vuelve a correr el detector HNSW. **Ante cualquier error del guion de FKs, volver a correr:** revalida todo y es seguro repetirlo.
- El `ALTER` usa `LOCK=SHARED` y espera el lock de metadatos 5 s (`lock_wait_timeout`): si hay una transacción abierta sobre la tabla sale `aplicada: false`, `motivo: lock_timeout`, y sigue con la siguiente. Reintentar cuando esa transacción termine; un ALTER encolado sin límite bloquearía el chat.
- El guion corre cada `ALTER` con `foreign_key_checks=0` **solo en esa sesión** (MariaDB no admite `INPLACE` de otro modo), y cuenta huérfanos antes y después de cada uno. Si aparece uno durante el ALTER, la tabla sale `aplicada: false` con el error: corregir los huérfanos o quitar esa FK (paso 14).
- Una tabla con huérfanos aborta sola; las demás siguen.
- `"algoritmo": "COPY"` = MariaDB rechazó INPLACE: no se aplicó; va a `DEUDA.md`.
- `hnsw_intacto` se calcula en **toda tabla con índice vectorial** (hoy `messages` y `facts`; `null` = la tabla no tiene índice VECTOR). `false` = el índice vectorial miente: seguir `docs/runbooks/indice-vectorial-envenenado.md` (`scripts/revisar_indice_vectorial.py`, que solo mira salvo `--reparar-de-verdad`).
- Salida `1`: alguna tabla sin aplicar o con hnsw roto; `2`: argumentos o guarda.
### 12. Tercera medición del índice vectorial, DESPUÉS de las FKs
```bash
e1 scripts/revisar_indice_vectorial.py | tee $D/hnsw-3-tras-fks.txt
```
Comparar con las dos anteriores y anotar las tres en la Biblioteca del proyecto con fecha. Un `ADD FOREIGN KEY` reconstruye la tabla o la toca de lado: ahí es donde un índice podía cambiar de estado, y por eso se mide también aquí.
### 13. Verificación por efecto
`/chat` con `project_id=1`: como Fernando da **200**; como **otro usuario que NO sea admin** (sin membresía en HAMURABI; un admin del tenant 1 ya es OWNER por D2 y vería 200 legítimamente) da **403**. Un 200 para ese usuario es un fallo cerrado roto: parar y escalar.
## Verification
Pasos 6 a 13 con sus salidas anotadas en `$D`. Además, las FKs puestas:
```sql
SELECT TABLE_NAME, CONSTRAINT_NAME FROM information_schema.KEY_COLUMN_USAGE
WHERE TABLE_SCHEMA = 'jax_memory' AND COLUMN_NAME = 'project_id' AND REFERENCED_TABLE_NAME = 'projects';
```
## Fail-closed condition
Respaldo sin restauración probada, ventana cerrada, usuario 1 que no cumple el paso 6, `SHOW INDEX` vacío tras el despliegue de jax-platform, ensayo sin números, un índice vectorial que miente en cualquiera de las tres mediciones, un código de salida no previsto (`5`), `fuera_de_alcance` no vacío, o un 200 donde se espera 403: parar y reportar.
## Recovery / escalation
### 14. Revertir cada paso
- **FKs:** una por tabla, solo si hace falta, y **con el nombre real de la FK**, no suponiendo `fk_<tabla>_project`: tomarlo de la consulta de `## Verification` (columna `CONSTRAINT_NAME`). En UNA sesión de `mariadb` (el `SET SESSION` solo vale para esa conexión):
  ```sql
  SET SESSION lock_wait_timeout=5;
  ALTER TABLE <tabla> DROP FOREIGN KEY <nombre real de la consulta>, ALGORITHM=INPLACE, LOCK=NONE;
  ```
  Si da `lock wait timeout` (hay una transacción abierta sobre la tabla), reintentar luego: un `ALTER` sin límite de espera encolaría detrás de esa transacción y bloquearía a todo el que llegue después, incluido el chat.
- **Huérfanos:** con las FKs ya quitadas (si estaban puestas, las filas volverían a apuntar a ids que no existen en `projects` y el `UPDATE` fallaría), con el mapa del paso 9 (o `<ruta>.incierto`):
  ```bash
  e1 scripts/proyectos_e1_migrar.py --revertir <mapa.json> --database jax_memory --confirmo-produccion
  ```
  El mapa **no se da por bueno**: antes de cualquier `UPDATE`, dentro de la transacción, se comprueba que `evaluacion_project_id` es el proyecto creado con la llave de la evaluación, y que cada `project_id_anterior` está en `RESERVED_PROJECT_ID_RANGE` y **no** existe en `projects`; si algo falla, salida `6` sin tocar nada. Luego corre un `UPDATE <tabla> SET project_id=<anterior> WHERE id=<id> AND project_id=<evaluación>` por entrada y **solo confirma si la suma de filas afectadas es igual a la del mapa** (si no: rollback, salida `6`, y el mensaje lista las entradas `(tabla, id)` que no cuadraron). Códigos de `--revertir`:
  - `0` revertido; repetirlo da `6` (ya no hay filas con la evaluación).
  - `2` argumentos o guarda: falta `--confirmo-produccion`, el mapa es ilegible o inválido (tabla fuera de las cinco), **no es del usuario que corre o tiene permisos más abiertos que 0600**. No se tocó nada.
  - `3` el commit quedó incierto: pudo aplicarse. **No repetir a ciegas:** correr `--verificar` y contar con SQL las filas con `project_id = <evaluación>` por tabla (paso 10). Si son 0 en todas, se revirtió; si siguen las del mapa, no; decidir con esa cuenta antes de volver a correr.
  - `5` error no previsto (p. ej. una FK sigue puesta): no reintentar sin revisar.
  - `6` el mapa no cuadra con la base: rollback, nada cambió.

  Después repetir la consulta del paso 10 con `<evaluación>`: tiene que dar 0 en cada tabla. **Antes de volver a poner FKs, medir el índice vectorial** (`e1 scripts/revisar_indice_vectorial.py | tee $D/hnsw-tras-revertir.txt`, solo lectura); si miente, parar y seguir `docs/runbooks/indice-vectorial-envenenado.md`.
- **Bootstrap de HAMURABI y proyecto de evaluación:** no se revierten; el alcance de HAMURABI queda.
- **Todo:** restaurar el respaldo del paso 2 es el último recurso y es decisión de Fernando.
## Prohibited actions
`migrate:fresh` o equivalentes; `DELETE` de filas para «limpiar» huérfanos (se reasignan); correr `--aplicar` sin el respaldo restaurado; otro actor que no sea `--actor-user-id 1`; pegar credenciales en este documento o en la línea de comandos; reutilizar una ruta de `--salida-reversion`; `--reparar-de-verdad` dentro de este runbook; borrar una FK suponiendo su nombre.
