# Proyectos E2a en producción
## Purpose
Llevar a producción la subida de documentos a un proyecto (E2a): permisos de `proyectos/`, LAS MANOS por `project_uuid`, la tabla `project_documents` (migración 006a), la plataforma con la pestaña Documentos y la conversión de la carpeta suelta de LACTOVI en un proyecto real. Orden de la Parte C del plan (`docs/superpowers/plans/2026-10-03-proyectos-e2a-plan.md`, Tarea 13).
## Scope
`/home/fruiz/jax-workspace/proyectos/` (permisos y la carpeta `lacteos-victoria` -> `<uuid>`), `jax_memory` (MariaDB `127.0.0.1:3308`: tabla `project_documents`, el proyecto de LACTOVI), los servicios `jax-las-manos` y `jax-platform`. Guiones: `ops/permisos_proyectos.py`, `scripts/proyectos_e2a_lactovi.py`.
## Preconditions
- **Ventana de Fernando abierta o su GO** (`bin/ventana estado` desde la sesión; si sale cerrada, parar). Repetirlo antes de cada paso que escribe (pasos 2 a 5).
- **Parte A (jax) integrada en `master` antes de abrir el PR de la Parte B**: el hook de migraciones de jax-platform carga `jax/memory/project_authority_migrations.py` desde `JAX_REPO_PATH`, o sea que lee el código de jax ya desplegado.
- **Actor de LACTOVI**: `--dueno-user-id` es el usuario que queda OWNER del proyecto. Confirmarlo antes (solo lectura): `SELECT user_id, tenant_id, role, status FROM jax_users WHERE user_id = <id>;` tiene que dar **una** fila, `status = 'active'`, `tenant_id = 1`. (En E1 el actor fue el 1; con otro, el digest de `create_project` cambia, pero esta llave de idempotencia es nueva.) Los demás administradores activos del tenant 1 reciben OWNER por la regla D2, igual que en E1: la lista sale de `SELECT user_id, email FROM jax_users WHERE tenant_id = 1 AND status = 'active' AND LOWER(role) IN ('admin','superadmin','super_admin');`.
- **Regla de sudoers para `--aplicar` de permisos.** El repo NO fija una regla de sudoers (ni `workspace-proyectos.md` ni el guion la traen) y `/etc/sudoers.d` no la tiene escrita por el repo. Antes del paso 2 se verifica con `sudo -l` (como `fruiz`) que están permitidas, sin contraseña, estas órdenes exactas (las que ejecuta el guion; `sudo -n` falla cerrado si falta alguna):
  - `/usr/bin/python3 -I /usr/local/sbin/jax-permisos-proyectos --nucleo-privilegiado`
  - `/usr/bin/python3 -I /usr/local/sbin/jax-permisos-proyectos --nucleo-respaldo`
  - `/usr/bin/python3 -I /usr/local/sbin/jax-permisos-proyectos --nucleo-deshacer`
  - `grep ^JAX_WORKSPACE_DIR= /etc/jax/.env` y `true` (comprobaciones previas del guion).
  Si falta alguna, **parar y pedirle a Fernando la regla**: no se amplía sudoers desde el runbook ni desde una sesión.
- **Dónde y con qué se corre.** Igual que `proyectos-e1-produccion.md`: directorio `/srv/jax-prod/jax`, intérprete `/srv/jax-prod/jax/.venv/bin/python`; el guion nunca abre `/etc/jax/.env`, el entorno `JAX_DB_*` lo carga quien opera y **la contraseña nunca va en la línea de comandos**. Función de la shell (pegar una vez):

  ```bash
  e2a() {   # uso: e2a -m scripts.proyectos_e2a_lactovi <argumentos>
    ( set -euo pipefail; set -a; . <(sudo -n cat /etc/jax/.env); set +a
      cd /srv/jax-prod/jax; export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:las_manos
      exec .venv/bin/python "$@" )
  }
  D=~/respaldos-despliegue/2026-10-0X-e2a; mkdir -p "$D"     # AAAA-MM-DD real del despliegue
  ```
  Las consultas SQL se corren con el archivo de opciones `600` del §0 de `docs/runbooks/despliegue.md` de jax-platform (nunca `-p"$JAX_DB_PASSWORD"`).
## Authority impact
Escribe en producción: cambia dueño/ACL de `proyectos/`, despliega código, crea una tabla (006a), un proyecto con su alcance y membresías, y **renombra una carpeta de datos reales**. Nada de esto sin ventana o GO. El guion de LACTOVI es reversible (`--revertir`) mientras `fuente/` no cambie; el proyecto creado no se destruye (queda archivado).
## Safe procedure
### 1. Respaldo con restauración probada (antes de todo)
Volcado de `jax_memory` (`mariadb-dump --single-transaction --routines --triggers`) a `$D`, más `ESTADO-ANTES.txt` con los SHAs desplegados y `du -sh /home/fruiz/jax-workspace/proyectos` (el «antes» de la Biblioteca, costo de R2). **Restaurar a una base descartable** (`jax_memory_test_e2aensayo`, quitando los `DEFINER`) y comparar conteos contra producción con el §0 de `despliegue.md` de jax-platform; si un par difiere, no se sigue. Un volcado sin restauración probada no es respaldo (Principio VI). Verificación: los conteos de `projects jax_users conversations messages` coinciden; `DROP DATABASE jax_memory_test_e2aensayo` al terminar (solo esa).
### 2. Permisos de `proyectos/`
```bash
python3 ops/permisos_proyectos.py --verificar      # solo lectura; anotar la salida
sudo -l                                            # confirmar la regla de sudoers (Preconditions)
sudo install -o root -g root -m 0755 ops/permisos_proyectos.py /usr/local/sbin/jax-permisos-proyectos
python3 ops/permisos_proyectos.py --aplicar
python3 ops/permisos_proyectos.py --verificar      # tiene que dar 0
```
Hacerlo desde un checkout cuyo HEAD tenga el guion commiteado (el núcleo se compara contra `git show HEAD:ops/permisos_proyectos.py`). **Requisito de `workspace-proyectos.md`:** el `fchmod(0o660)` de `tool_authority.py` tiene que estar ya en `/srv/jax-prod` (paso 3 de esta secuencia lo despliega; si no está, `--aplicar` NO se corre: ver «Aplicación en producción: BLOQUEADA» en ese runbook). Verificación: `--verificar` sale `0`; luego la **prueba real como `jaxsvc`** de ese runbook (paso 6: escritura cruzada `jaxsvc`/`fruiz` en un subdirectorio propio) en verde. Reversión: `python3 ops/permisos_proyectos.py --deshacer`.
> **Orden.** Si el despliegue de jax (paso 3) es el que trae el `fchmod`, se hace el paso 3 antes del `--aplicar` de este paso. Los demás pasos de este runbook no dependen de ello.
### 3. jax a producción (LAS MANOS con `project_uuid`)
Con el procedimiento de `docs/runbooks/despliegue.md` de jax-platform (sección jax). Reiniciar `jax-las-manos`. Verificación: un `POST /procesamiento/trabajos` con un `project_uuid` inexistente responde **422** con `proyecto_no_activo`; `systemctl is-active jax-las-manos` da `active`; **el arreglo de la ingesta tiene que estar desplegado** (`grep -c _origen_ya_en_fuente /srv/jax-prod/jax/procesamiento/ingesta.py` da `1` o más): sin él, procesar un archivo de LACTOVI que ya vive en `fuente/sub/` lo **copiaría** a la raíz de `fuente/` (duplica datos del cliente, cambia los sha y cierra la reversión), y el paso 5 no se corre; `/proc/<pid>/cwd` del servicio apunta a `/srv/jax-prod/jax` (un servicio sirve desde un checkout: se verifica, no se supone).
### 4. jax-platform a producción
Una sola vez, con los comandos de `despliegue.md`. Al arrancar, `run_migrations()` aplica la 006a y siembra las cuatro claves de `axioma_config` de la Tarea 5. Verificación (consulta propia):
```sql
SHOW CREATE TABLE project_documents;                    -- existe, con uq_project_documents_sha y las 3 FKs
SELECT config_key, config_value FROM axioma_config WHERE config_key LIKE 'proyectos.documentos.%';   -- las 4 claves
```
Si la tabla no está, mirar el log de arranque (falla de `run_migrations`) y parar: no se crea a mano. Publicar el sitio (frontend) según `despliegue.md` y comprobar que sirve el bundle nuevo.
### 5. LACTOVI: de carpeta suelta a proyecto
Rutas reales: workspace `/home/fruiz/jax-workspace`, carpeta `lacteos-victoria`. Antes: `find .../proyectos/lacteos-victoria/fuente -type f | wc -l` (en el ensayo del 2026-10-03: 120) y `ls .../procesado | wc -l` (88), y se anota. **Mismo sistema de archivos**: `os.rename` entre `proyectos/lacteos-victoria` y `proyectos/<uuid>` es dentro del mismo directorio.
```bash
# 5a. Ensayo (no escribe nada: solo lee disco y base)
e2a -m scripts.proyectos_e2a_lactovi --workspace /home/fruiz/jax-workspace --carpeta lacteos-victoria \
  --nombre "Lácteos Victoria" --dueno-user-id <ID> --tenant-id 1 --database jax_memory | tee $D/lactovi-ensayo.json
```
Revisar: `archivos_fuente` (120), `fichas` (88), `filas_a_insertar` (una por sha256 distinto: ≤ 120; la diferencia son `duplicados_sha`), `estados` (cuántas filas `listo`/`parcial`/`error`/`sin_extractor`/`en_cola`), `ignorado` (debe listar `.claude-flow`, que viaja con el rename y no se registra), `ignorados` (symlinks y no-regulares de **cualquier** nivel de `fuente/`: no se hashean ni se registran; cada uno se revisa a mano), `fichas_sin_archivo` (idealmente vacío), `mapa_previsto`. Si algo no es lo esperado, parar.
### 5a2. Respaldo de la carpeta ANTES de moverla (sin esto no se aplica)
El respaldo del paso 1 es de la base. La carpeta `lacteos-victoria` (datos del cliente) necesita el suyo, con restauración probada, antes del `rename`. Se usa el mismo mecanismo que `backup-hall9000.sh` (Step 5b: `restic backup --host hall9000 --one-file-system` contra `/etc/restic/local.env` y `/etc/restic/r2.env`, **como `fruiz`, nunca como root**: dejaría packs de root que el prune de `fruiz` no puede borrar), con una etiqueta propia. Un restic largo **no va en primer plano**: se lanza con `nohup` y se espera su `rc`.
```bash
ORIG=/home/fruiz/jax-workspace/proyectos/lacteos-victoria
for ENV in /etc/restic/local.env /etc/restic/r2.env; do
  [ -f "$ENV" ] || { echo "FALTA $ENV: no se aplica"; continue; }
  N=$(basename "$ENV" .env)
  nohup bash -c "set -euo pipefail; source '$ENV'; restic backup --tag e2a-lactovi-pre --host hall9000 --one-file-system '$ORIG'" \
    > "$D/restic-pre-$N.log" 2>&1 ; echo "$N rc=$?" | tee -a "$D/restic-pre.rc"
done
```
Los dos repos son obligatorios si existen los dos `.env`. Cada `rc` tiene que ser `0` (con `3` hubo snapshot pero algún archivo no se leyó: no se sigue hasta saber cuál). Después, **por cada repo**, restaurar a una ruta aparte y comparar `sha256` contra el original:
```bash
for ENV in /etc/restic/local.env /etc/restic/r2.env; do
  [ -f "$ENV" ] || continue; N=$(basename "$ENV" .env); T="$D/restauracion-pre-$N"; rm -rf "$T"
  ( set -euo pipefail; source "$ENV"
    ID=$(restic snapshots --json --tag e2a-lactovi-pre --host hall9000 --latest 1 | python3 -c 'import json,sys; print(json.load(sys.stdin)[-1]["short_id"])')
    echo "$N snapshot $ID"
    restic restore "$ID" --target "$T" --include "$ORIG" )
  R="$T$ORIG"
  diff <(cd "$ORIG" && find . -type f -print0 | sort -z | xargs -0 sha256sum) \
       <(cd "$R"    && find . -type f -print0 | sort -z | xargs -0 sha256sum) && echo "$N: sha256 IGUALES"
  diff <(cd "$ORIG" && find . -type l -printf '%p -> %l\n' | sort) <(cd "$R" && find . -type l -printf '%p -> %l\n' | sort) && echo "$N: symlinks IGUALES"
done
```
**Verificación:** los dos `diff` salen vacíos (`sha256 IGUALES` y `symlinks IGUALES`) para cada repo. Cualquier diferencia: **no se corre 5b**; se escala a Fernando. Borrar las rutas `restauracion-pre-*` al terminar (solo esas). El hash de `fuente/` de «antes» que pide la verificación de 5b se anota aquí.
```bash
# 5b. Aplicar (con ventana abierta: bin/ventana estado)
e2a -m scripts.proyectos_e2a_lactovi --workspace /home/fruiz/jax-workspace --carpeta lacteos-victoria \
  --nombre "Lácteos Victoria" --dueno-user-id <ID> --tenant-id 1 --database jax_memory \
  --aplicar --confirmo-produccion | tee $D/lactovi-aplicar.json
cp /home/fruiz/jax-workspace/proyectos/.e2a-lactovi-*.json $D/     # el mapa de reversión, a salvo
```
Los archivos sin ficha de un tipo que se extrae (`pdf xlsx xls docx png jpg jpeg tif tiff csv txt md`) entran como `en_cola` con `ruta_entrada = proyectos/<uuid>/fuente/<ruta>`: el despachador de la plataforma los manda a procesar y la ingesta, al ver que el archivo **ya está dentro de `fuente/`**, lo procesa en el lugar sin copiarlo (arreglo `_origen_ya_en_fuente`, paso 3). Los de otro tipo entran como `sin_extractor`, sin `ruta_entrada`. **Comprobación tras el despacho:** `find /home/fruiz/jax-workspace/proyectos/<uuid>/fuente -type f | wc -l` sigue siendo el de antes (120): si crece, hay copias y se detiene el despachador.
Códigos de salida: `0` hecho; `1` el mapa o el destino ya existen, el proyecto no está ACTIVO, o falló el registro (se deshizo el rename, el proyecto queda creado y ACTIVO; se revisa antes de reintentar); `2` argumentos, guarda de producción o carpeta que no existe (p. ej. **ya se movió**: no crea otro proyecto); `3` los `sha256` no cuadran tras mover (se deshizo el rename); `4` el commit de las filas tiene desenlace desconocido (**el disco no se tocó**: verificar las filas y, si faltan, `--completar`); `5` error no previsto: **no reintentar sin revisar**; `6` (`--revertir`/`--completar`) el mapa no cuadra con la base; `7` (`--revertir`) hay trabajos `pendiente`/`procesando`.

**Qué hacer en cada corte** (el disco y la base se miran antes de cualquier reintento):

| Dónde se cortó | Estado que queda | Qué hacer |
|---|---|---|
| Tras `create_project`, antes del mapa | proyecto ACTIVO, carpeta sin mover, sin mapa | Volver a correr `--aplicar` (misma llave: reutiliza el proyecto) |
| Mapa escrito, `rename` falló o no ocurrió | mapa existe, carpeta en su sitio | Renombrar el mapa (`mv .e2a-lactovi-<fecha>.json mapa-corte.json`; nunca se pisa) y volver a `--aplicar` |
| Tras el `rename`, antes de las filas (corte del proceso) | carpeta en `<uuid>`, sin filas | `--completar <mapa>` (verifica `sha256` contra el mapa y registra lo que falte) |
| Commit de las filas incierto (código 4) | carpeta en `<uuid>`, filas quizá | Contar filas (`SELECT COUNT(*)`); si faltan, `--completar` (idempotente) |
| Fallo al registrar con error previo al commit (código 1) | el guion devolvió la carpeta | Renombrar el mapa y volver a `--aplicar` |
| `--revertir` cortado antes del commit | nada cambió o se deshizo solo | Repetir `--revertir` |
| `--revertir` cortado en el archivado | carpeta devuelta, sin filas, proyecto ACTIVO | Repetir `--revertir`: detecta la carpeta ya devuelta y solo archiva |

Tras un `--revertir` el proyecto queda ARCHIVADO: para volver a aplicar hay que restaurarlo (`RESTORE_PROJECT`, ARCHIVED → ACTIVE) desde la plataforma y renombrar el mapa viejo; `--aplicar` se niega con ese mensaje si no.
**Verificación independiente** (no con la salida del guion):
```sql
SELECT estado, COUNT(*) FROM project_documents WHERE project_id = <project_id> GROUP BY estado;
SELECT COUNT(*) FROM project_documents WHERE project_id = <project_id>;      -- = filas_insertadas
SELECT project_uuid FROM projects WHERE id = <project_id>;                   -- = el nombre de la carpeta nueva
```
```bash
ls -d /home/fruiz/jax-workspace/proyectos/<uuid>        # existe
ls -d /home/fruiz/jax-workspace/proyectos/lacteos-victoria 2>&1   # ya no existe
cd /home/fruiz/jax-workspace/proyectos/<uuid>/fuente && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
```
Y el último hash se compara con el mismo cálculo hecho en el paso de «antes» sobre la carpeta vieja (anotarlo antes de 5b). La pestaña Documentos de LACTOVI en la plataforma tiene que listar los documentos.
**Reversión** (solo antes de que la plataforma reciba subidas a LACTOVI; si no, los `sha256` no cuadran y sale `3` sin tocar nada). **Primero se detiene el despachador** (vive en jax-platform): con trabajos en vuelo, la ingesta recrearía `proyectos/<uuid>/fuente/` después de devolver la carpeta y la dejaría partida. El guion se niega (código 7) si alguna fila está `pendiente` o `procesando`, pero eso no sustituye detener el servicio:
```bash
sudo systemctl stop jax-platform
systemctl is-active jax-platform          # tiene que decir inactive
# consulta propia: tiene que dar 0
SELECT COUNT(*) FROM project_documents WHERE project_id = <project_id> AND estado IN ('pendiente','procesando');
e2a -m scripts.proyectos_e2a_lactovi --revertir $D/.e2a-lactovi-<fecha>.json --database jax_memory --confirmo-produccion
```
Borra las filas de `project_documents`, devuelve la carpeta a `lacteos-victoria` verificando `sha256`, y deja el proyecto **ARCHIVADO** (no se borra: Destruir no existe). Verificar: la carpeta vieja existe y `<uuid>` no; 0 filas; alcance ARCHIVED. Es reintentable. Después, `sudo systemctl start jax-platform`.
### 6. Restauración probada de `proyectos/` con la ruta nueva
**Sin esto, la subida no se anuncia como disponible.** Esperar el snapshot de restic posterior al traslado (o lanzar uno con el procedimiento de respaldo vigente; un restic largo nunca en primer plano), restaurar solo `proyectos/<uuid>/` a una ruta aparte (`restic restore <snapshot> --target /tmp/e2a-restauracion --include /home/fruiz/jax-workspace/proyectos/<uuid>`), y comparar el `sha256` de cada archivo de `fuente/` contra el `sha256` de las fichas/`project_documents`. Cualquier diferencia o archivo ausente: la subida NO se anuncia y se escala a Fernando. Borrar la ruta de ensayo al terminar.
### 7. Prueba de humo
Fernando sube 3 documentos a LACTOVI desde el Chat y los ve pasar a `listo`. Verificación: `SELECT nombre_original, estado FROM project_documents WHERE project_id = <project_id> ORDER BY id DESC LIMIT 3;` y cada archivo bajo `proyectos/<uuid>/entrada/` (o donde fije la plataforma) con dueño `jaxsvc:fruiz`.
### 8. Biblioteca
Escribir `docs/historia/2026-10-03-proyectos-e2a.md` (fecha, qué se hizo, por qué, lecciones, pendientes, alternativas descartadas, quién decidió) con los números de carga de la Tarea 5, `du -sh proyectos/` antes y después (costo de R2) y lo que queda para E2b.
## Verification
Cada paso lleva la suya arriba. Cierre: `ops/permisos_proyectos.py --verificar` en 0; `422 proyecto_no_activo` con un uuid inexistente; `project_documents` con el conteo del guion; `sha256` de `fuente/` iguales antes y después; restauración de restic comparada; prueba de humo de Fernando.
## Fail-closed condition
Cualquier verificación que no dé lo esperado detiene la secuencia: no se pasa al paso siguiente. El guion de LACTOVI se niega a correr contra `jax_memory` sin `--confirmo-produccion`, no pisa mapas ni destinos existentes y, si los `sha256` no cuadran o el registro falla, devuelve la carpeta a su nombre.
## Recovery / escalation
Paso 2: `--deshacer`. Paso 5: `--revertir` con el mapa guardado en `$D`. Pasos 3 y 4: `docs/runbooks/deployment-rollback.md` y el respaldo del paso 1; la migración 006a baja con `scripts/b9_revertir_005.py` solo si `project_documents` está vacía (con filas falla cerrado). Si el estado queda irreconocible, parar y escalar a Fernando antes de tocar a mano.
## Prohibited actions
`migrate:fresh` ni nada equivalente; borrar `proyectos/<uuid>` o `lacteos-victoria` a mano; editar `project_documents` a mano; correr el guion de LACTOVI dos veces «para ver»; tocar `/etc/sudoers.d` desde una sesión; anunciar la subida disponible sin el paso 6.
