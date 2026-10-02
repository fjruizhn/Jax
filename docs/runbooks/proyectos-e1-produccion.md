# Proyectos E1 en producción
## Purpose
Llevar a `jax_memory` de producción: HAMURABI como proyecto 1 con alcance, el proyecto archivado «Evaluación grounding SP3 · 2026-09-03» que recibe los `project_id` huérfanos, y las llaves foráneas `project_id -> projects(id)` de las cinco tablas de contenido.
## Scope
`conversations`, `messages`, `facts`, `decisions`, `action_items` en `jax_memory` (MariaDB `127.0.0.1:3308`). Guiones: `scripts/proyectos_e1_migrar.py` y `scripts/proyectos_e1_fks.py`. No toca jax-platform salvo la verificación por efecto (paso 6).
## Preconditions
- Ventana abierta o GO de Fernando para producción (`bin/ventana estado` desde la sesión; si salió cerrada, parar).
- Ensayo hecho sobre una copia (sección «Ensayo previo»). Lo hace la sesión principal con GO; sin sus números no se aplica nada.
- El entorno `JAX_DB_HOST/PORT/USER/PASSWORD/NAME` lo carga quien opera, en su shell; los guiones nunca abren `/etc/jax/.env` y este documento no lleva secretos. `JAX_DB_PORT` es 3308 (la 3306 está muerta).
- **El actor es siempre `--actor-user-id 1`.** El digest de idempotencia de `create_project` incluye al usuario: con otro actor, la segunda corrida da `IdempotencyKeyConflict`.
## Authority impact
Escribe en producción: crea alcance y proyectos, reescribe `project_id` y crea FKs (DDL). Nada de esto se ejecuta sin ventana o GO. La migración de datos es idempotente; el bootstrap de HAMURABI no se revierte (el alcance queda).
## Ensayo previo (lo hace la sesión principal, con GO)
1. Volcado de `jax_memory` con el procedimiento de respaldo de este runbook (paso 2) y restauración a `jax_memory_e1_ensayo` en el mismo MariaDB.
2. En la copia: `python scripts/proyectos_e1_migrar.py --verificar --actor-user-id 1 --database jax_memory_e1_ensayo`, luego `--aplicar --actor-user-id 1 --salida-reversion <ruta nueva> --database jax_memory_e1_ensayo`.
3. En la copia: `python scripts/proyectos_e1_fks.py --ensayar --database jax_memory_e1_ensayo`. Anotar por tabla `algoritmo`, `segundos` y, en `messages`, `hnsw_intacto`.
4. Solo van a producción las tablas con `"algoritmo": "INPLACE"`, `"aplicada": true` y (messages) `"hnsw_intacto": true`. Las demás se registran en `DEUDA.md` con el motivo y la validación queda en la aplicación.
5. Borrar la copia al terminar (`DROP DATABASE jax_memory_e1_ensayo`, solo esa).
## Safe procedure
### 1. Ventana o GO
`bin/ventana estado` abierta, o GO de Fernando. Repetir `bin/ventana estado` antes de cada paso que escribe.
### 2. Respaldo con restauración probada (antes de todo)
Mismo procedimiento que `~/respaldos-despliegue/2026-10-02-suscripcion-fase1`: volcado de `jax_memory` comprimido, `ESTADO-ANTES.txt` con los SHAs desplegados y conteos, y **restauración probada** a una base de prueba comparando conteos (`messages`, `jax_users`, etc.) contra producción. Un volcado sin restauración probada no es respaldo (Principio VI). No se sigue sin el conteo igual.
### 3. Desplegar jax con T1-T3 y comprobar el índice de la migración 005e
La migración 005 de B9 **no** corre al arrancar el servicio ni al importar: se aplica con `apply_project_authority_migration` (`jax/memory/project_authority_migrations.py`, sin copia `.sql`; ver `jax/memory/b9_migrations/README.md`). Antes de desplegar la plataforma, en producción:

```sql
SHOW INDEX FROM jax_project_membership WHERE Key_name = 'idx_jax_project_membership_user_list';
```

Tiene que devolver filas. Si no devuelve nada, la migración de proyectos no está aplicada: **se aplica primero** (con GO, con el respaldo del paso 2 ya hecho) y se repite el `SHOW INDEX`. Su bajada es `scripts/b9_revertir_005.py` (simulacro por omisión). Después se despliega jax con T1-T3.
### 4. Medir y migrar los datos
```bash
python scripts/proyectos_e1_migrar.py --verificar --actor-user-id 1
```
Anotar `huerfanos_ids` y `filas_por_tabla`. Luego:
```bash
python scripts/proyectos_e1_migrar.py --aplicar --actor-user-id 1 \
  --salida-reversion <ruta nueva> --confirmo-produccion
```
- `--salida-reversion` es obligatorio con `--aplicar` y la ruta tiene que **no existir** (si existe, aborta antes de tocar la base). Escribe un mapa JSON 0600: `{"evaluacion_project_id": N, "filas": [{"tabla", "id", "project_id_anterior"}]}`. Guardarlo fuera del repo y con el respaldo.
- Comparar: `despues.huerfanos_ids` vacío y `filas_movidas` igual a `antes.filas_por_tabla`.
- Códigos de salida:
  - `0` hecho; repetirlo es seguro (idempotente).
  - `1` quedan huérfanos tras el commit (aparecieron después de medir), conteos que no coinciden (hizo rollback) o la ruta ya existía: **volver a correr con una ruta NUEVA**. El mapa anterior, si se escribió, se conserva.
  - `2` argumentos o guarda: falta la base, `--salida-reversion`, o `--confirmo-produccion` sobre `jax_memory`. No se tocó nada.
  - `3` el commit quedó incierto: pudo aplicarse. El mapa quedó en `<ruta>.incierto`. Correr `--verificar`: si no hay huérfanos, se aplicó (el mapa sirve para revertir); si siguen, no se aplicó y se vuelve a correr con ruta nueva. Decidir con esa medición, sin adivinar.
### 5. FKs
```bash
python scripts/proyectos_e1_fks.py --aplicar --confirmo-produccion
```
Una línea JSON por tabla: `tabla`, `filas`, `algoritmo`, `segundos`, `hnsw_intacto`, `aplicada` (y `ya_existia`, `error` o `motivo` si corresponde). Repetirlo es seguro: una FK existente no se toca.
- El guion corre cada `ALTER` con `foreign_key_checks=0` **solo en esa sesión** (MariaDB no admite `INPLACE` de otro modo), y cuenta huérfanos antes y después de cada uno. Si aparece uno durante el ALTER, la tabla sale `aplicada: false` con el error: corregir los huérfanos o `DROP FOREIGN KEY` (paso 7).
- Una tabla con huérfanos aborta sola; las demás siguen.
- `"algoritmo": "COPY"` = MariaDB rechazó INPLACE: no se aplicó; va a `DEUDA.md`.
- En `messages`, `hnsw_intacto: false` = el índice vectorial miente. Seguir `docs/runbooks/indice-vectorial-envenenado.md` (`scripts/revisar_indice_vectorial.py`, que solo mira salvo `--reparar-de-verdad`).
- Salida `1`: alguna tabla sin aplicar o con hnsw roto; `2`: argumentos o guarda.
### 6. Verificación por efecto
`/chat` con `project_id=1`: como Fernando da **200**; como otro usuario (sin membresía) da **403**. Un 200 para el otro usuario es un fallo cerrado roto: parar y escalar.
## Verification
Pasos 4 a 6 con sus salidas anotadas. Además:
```sql
SELECT TABLE_NAME, CONSTRAINT_NAME FROM information_schema.KEY_COLUMN_USAGE
WHERE TABLE_SCHEMA = 'jax_memory' AND COLUMN_NAME = 'project_id' AND REFERENCED_TABLE_NAME = 'projects';
```
## Fail-closed condition
Respaldo sin restauración probada, ventana cerrada, `SHOW INDEX` vacío sin migración aplicada, ensayo sin números, un código de salida no previsto, o un 200 donde se espera 403: parar y reportar.
## Recovery / escalation
### 7. Revertir cada paso
- **FKs:** una por tabla, solo si hace falta: `ALTER TABLE <tabla> DROP FOREIGN KEY fk_<tabla>_project;`
- **Huérfanos:** con el mapa del paso 4 (o `<ruta>.incierto`), un `UPDATE` por fila según `tabla`, `id` y `project_id_anterior`:
  ```sql
  UPDATE `<tabla>` SET project_id = <project_id_anterior> WHERE id = <id>;
  ```
  Si hay FKs aplicadas, primero se quitan (las filas vuelven a apuntar a ids que no existen en `projects`). Se generan las sentencias desde el JSON con un script revisado y se corren dentro de una transacción; comprobar el conteo de filas afectadas contra el mapa antes del `COMMIT`.
- **Bootstrap de HAMURABI y proyecto de evaluación:** no se revierten; el alcance de HAMURABI queda.
- **Todo:** restaurar el respaldo del paso 2 es el último recurso y es decisión de Fernando.
## Prohibited actions
`migrate:fresh` o equivalentes; `DELETE` de filas para «limpiar» huérfanos (se reasignan); correr `--aplicar` sin el respaldo restaurado; otro actor que no sea `--actor-user-id 1`; pegar credenciales en este documento o en la línea de comandos; reutilizar una ruta de `--salida-reversion`.
