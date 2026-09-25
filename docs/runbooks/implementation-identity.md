# Implementation identity
## Purpose
Recover B7 deployment identity availability.
## Scope
Implementation manifest/identity.
## Preconditions
Verified build/deployment source.
## Authority impact
Evidence integrity, not policy authority.
## Safe procedure
Deploy verified identity and manifest together.

**Exact procedure (jax#260, revisado en la ronda de PR#264 tras leer
`_configure_b7_trusted_runtime` completo y todo lo que llama).**
`las_manos/server.py::_configure_b7_trusted_runtime` corre **de forma
síncrona en el evento `startup` de FastAPI** (`@app.on_event("startup")` →
`_jacobs_init()`), sin try/except alrededor. No es un solo archivo: son
**nueve precondiciones encadenadas**, y cualquiera de las nueve ausente
tira abajo el arranque completo antes de que `/health` responda una sola
vez. No hay modo degradado/parcial: fail-closed, según
`docs/operations/trusted-files.md`.

### 1. Variables de entorno
`JAX_DEPLOYMENT_ID`, `JAX_DB_HOST`, `JAX_DB_PORT` tienen que estar no
vacías (`/etc/jax/.env`, cargado por `EnvironmentFile=` en el unit de
systemd). Si falta cualquiera, `_configure_b7_trusted_runtime` corta de
entrada con `RuntimeError("B7 trusted composition requires deployment and
MariaDB configuration")`. Esto no aparecía documentado en ningún lado del
repo antes de esta ronda. Verificado en vivo en hall9000 (2026-09-22): las
tres ya están puestas (`JAX_DEPLOYMENT_ID=hall9000-prod`,
`JAX_DB_HOST=127.0.0.1`, `JAX_DB_PORT=3308`) — este paso es para el próximo
host, no una alarma sobre hall9000 hoy.

### 2. Esquema `jax_evidence` instalado
11 tablas y sus 22 triggers de inmutabilidad (verificado con `grep -c
"^CREATE TABLE" / "^CREATE TRIGGER"` contra
`policy/enforcement_evidence/migrations/001_enforcement_evidence.sql` —
corrige un conteo anterior de "9 tablas + 18 triggers" que no coincidía con
el archivo real). `MariaDBEvidenceStore` los usa directo durante el
arranque (`__record_identity` inserta en `implementation_identities`,
`get_evidence_blob` lee de `evidence_blobs`); si el esquema no existe,
el fallo es un error crudo de MariaDB ("table doesn't exist"), no uno con
mensaje claro.

### 3. Esquema `jax_execution`, y que pase `inspect_database_control`
No basta con que las tablas existan: `DatabaseControlInspector.inspect_one_decision_one_execution()`
—el último paso de `_configure_b7_trusted_runtime`— llama
`inspect_database_control()`
(`policy/enforcement_evidence/database_evidence.py:23-35`), que exige
**todo esto a la vez** y tira `DatabaseObservationMismatchError` si falta
cualquiera:
- las tablas `execution_records`, `execution_events`,
  `execution_authorization_consumptions`;
- un índice único sobre `execution_records.decision_id`;
- un índice único sobre `execution_authorization_consumptions`;
- los 4 triggers `UPDATE`/`DELETE` sobre `execution_records` y
  `execution_events` (de los 10 que crea la migración en total, sobre 5
  tablas — éstos 4 son los únicos que el arranque verifica).

Todo esto lo crea `policy/execution_control/migrations/001_governed_execution.sql`
(7 tablas, 10 triggers).

### Desplegar este PR en `/srv/jax-prod/jax` -- sin esto no hay paso 6 ni migraciones idempotentes
**BLOCK-A, ronda 5 de revisión -- corrige una afirmación falsa de la
ronda 4.** La ronda 4 decía que `/srv/jax-prod/jax` en `2794cf3` "YA es
posterior al merge que agrega `scripts/generar_manifiesto_identidad.py`,
así que ese checkout SÍ tiene el script hoy". **Eso es falso, verificado
contra el árbol real, no contra lo que decía la ronda anterior**: el
generador lo agrega `cd764f5`
("feat(jax#260): generador del manifiesto de identidad de implementación"),
que vive SÓLO en esta rama sin mergear (`feat/manifiesto-identidad`) --
`git merge-base --is-ancestor cd764f5 2794cf3` confirma que NO es
ancestro, y `git -C /srv/jax-prod/jax show HEAD:scripts/generar_manifiesto_identidad.py`
da "path does not exist". El error de la ronda 4 fue medir el COMMIT
correctamente (`2794cf3`, con fecha) pero SUPONER sin verificarlo que ese
commit ya incluía el merge de este PR -- exactamente lo que el Principio
I prohíbe: medir dos veces, no una vez y adivinar la segunda.

Esto no es sólo el script ausente: las migraciones instaladas en
`/srv/jax-prod/jax` en `2794cf3` tampoco tienen la idempotencia de la
ronda 1 de revisión de este PR -- medido con `grep -c "CREATE TRIGGER IF
NOT EXISTS"` contra el archivo real en ese commit: **0**, contra **22**
en esta rama. Aplicar las migraciones del paso 4/5 tal como están hoy en
disco corta en el primer `CREATE TRIGGER` con "already exists" ante
cualquier re-run a medio aplicar (ver la nota de idempotencia más abajo).

**Por eso este PR tiene que estar mergeado y desplegado en
`/srv/jax-prod/jax` ANTES de tocar el paso 4/5 de abajo** -- el
procedimiento normal de despliegue de este repo, corrido como `jaxsvc`
por el mismo motivo de `dubious ownership` que el paso 6:

```bash
sudo -u jaxsvc git -C /srv/jax-prod/jax fetch origin
sudo -u jaxsvc git -C /srv/jax-prod/jax reset --hard origin/master
git -c safe.directory=/srv/jax-prod/jax -C /srv/jax-prod/jax log -1 --oneline   # confirmar el commit real
test -x /srv/jax-prod/jax/scripts/generar_manifiesto_identidad.py && echo "OK -- el generador ya está en este checkout"
```

`origin/master` asume que este PR ya está mergeado a `master` en el
remoto -- si todavía no lo está, este comando despliega lo último que SÍ
está mergeado, NO este PR. Mergear es una decisión de Fernando, no algo
que este runbook dispare por sí solo.

### 4 y 5. Aplicar las dos migraciones y otorgar privilegios -- una sola sesión, con un usuario ADMIN
**Crear los esquemas (`CREATE SCHEMA`/`TABLE`/`TRIGGER`) no es un
privilegio de aplicación, aunque el usuario de aplicación termine con
`ALL PRIVILEGES` sobre sus DOS esquemas propios (MAJOR-1, ver el `GRANT`
de abajo).** `$JAX_DB_USER` de `/etc/jax/.env` sigue acotado a
`jax_evidence`/`jax_execution` -- nunca a otro esquema, y nunca es la
cuenta que aplica las migraciones. Todo lo de abajo (crear los esquemas)
usa una cuenta ADMINISTRATIVA aparte, provista por quien lo ejecuta (nunca
la misma que `$JAX_DB_USER`, y nunca escrita en este archivo).

**Los pasos 4 y 5 son UNA sola sesión de shell, de punta a punta.** La
ronda anterior los separaba y el archivo de credenciales (`$CNF`) se
borraba al final del paso 4 -- dejando al paso 5 con una instrucción de
"reconstruir `$CNF` si ya se borró" sin el comando, Y con la contraseña ya
destruida por el propio `unset`. No hay forma de reconstruirlo sin
volver a pedir la contraseña, así que la solución real es no partirlo:
`$CNF` vive hasta el final de este bloque combinado, se usa en los dos
pasos, y se borra una sola vez al final.

**Por qué `cd` primero:** las rutas de las migraciones son relativas al
repo (`policy/enforcement_evidence/migrations/...`); sin pararse en
`/srv/jax-prod/jax`, `mysql < policy/...` no encuentra el archivo desde
cualquier otro directorio.

```bash
cd /srv/jax-prod/jax

set -a; . <(sudo -n cat /etc/jax/.env); set +a   # sólo para HOST/PORT

read -srp "Usuario admin de MariaDB: " JAX_DB_ADMIN_USER; echo
read -srp "Password de $JAX_DB_ADMIN_USER: " JAX_DB_ADMIN_PASSWORD; echo

CNF=$(mktemp); trap 'rm -f "$CNF"' EXIT; chmod 600 "$CNF"
cat > "$CNF" <<EOF
[client]
host=$JAX_DB_HOST
port=$JAX_DB_PORT
user=$JAX_DB_ADMIN_USER
password=$JAX_DB_ADMIN_PASSWORD
EOF
unset JAX_DB_ADMIN_PASSWORD
```

**`SHOW GRANTS` con AVISO explícito, todavía no un corte real de shell
(ronda 4 de revisión — corrige una afirmación falsa de la ronda
anterior).** La ronda anterior decía que esto "CORTA el script"; medido
de verdad con un pty real: un comando que falla bajo `set -e` en una
shell interactiva SÍ la cierra por completo, exactamente igual que
`exit` — que es lo que este bloque explícitamente decidió evitar. Sin
`set -e` activo (que es el caso acá), `false` no corta absolutamente
nada por sí sola: sólo dejar `$? != 0` e imprimir el mensaje. Si el
operador sigue pegando los bloques de abajo en la misma terminal, van a
intentar correr igual — MariaDB los va a rechazar por falta de
privilegio recién ahí, no antes. Este chequeo es un AVISO temprano con
mensaje claro, no una compuerta de shell: **si ves el `NO-GO` de abajo,
frená vos, no sigas pegando.**

**MINOR-4 (ronda 4 de revisión): el chequeo anterior no podía fallar de
verdad.** `grep -qi '\bCREATE\b'` matchea el substring `CREATE` DENTRO de
`CREATE VIEW` (un privilegio real y mucho más débil), y no mira sobre QUÉ
objeto está el privilegio — una cuenta con `CREATE VIEW, TRIGGER` en
`jax_evidence` pasaba como si tuviera `CREATE TABLE` global. Confirmado
en rojo contra el código viejo, con `GRANT CREATE VIEW, SELECT, TRIGGER
ON \`jax_evidence\`.*` como único grant no-`ALL PRIVILEGES`: el chequeo
viejo decía "OK (CREATE + TRIGGER)". El chequeo nuevo parsea cada línea
`GRANT ...`, separa privilegios por coma con match EXACTO (no substring),
y sólo cuenta privilegios otorgados sobre `*.*` o sobre alguno de los dos
esquemas de este PR:

```bash
GRANTS=$(mysql --defaults-extra-file="$CNF" -N -e "SHOW GRANTS FOR CURRENT_USER();")

cuenta_ok=0
while IFS= read -r linea; do
  case "$linea" in
    GRANT\ *) ;;
    *) continue ;;
  esac
  objetivo=$(echo "$linea" | sed -E 's/^GRANT .* ON ([^ ]+) TO .*$/\1/I')
  case "$objetivo" in
    '*.*'|'`jax_evidence`.*'|jax_evidence.\*|'`jax_execution`.*'|jax_execution.\*) ;;
    *) continue ;;   # privilegio sobre otro esquema: no cuenta para este gate
  esac
  privilegios=$(echo "$linea" | sed -E 's/^GRANT (.*) ON .*$/\1/I')
  if echo "$privilegios" | grep -qi "ALL PRIVILEGES"; then cuenta_ok=1; break; fi
  tiene_create=0; tiene_trigger=0
  while IFS= read -r priv; do
    priv=$(echo "$priv" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')
    case "$priv" in
      [Cc][Rr][Ee][Aa][Tt][Ee]) tiene_create=1 ;;
      [Tt][Rr][Ii][Gg][Gg][Ee][Rr]) tiene_trigger=1 ;;
    esac
  done < <(echo "$privilegios" | tr ',' '\n')
  if [ "$tiene_create" = "1" ] && [ "$tiene_trigger" = "1" ]; then cuenta_ok=1; break; fi
done <<< "$GRANTS"

if [ "$cuenta_ok" = "1" ]; then
  echo "cuenta admin OK (CREATE+TRIGGER o ALL PRIVILEGES, con alcance real)."
else
  echo "NO-GO: la cuenta admin no muestra CREATE+TRIGGER (ni ALL PRIVILEGES) con" >&2
  echo "alcance sobre *.*, jax_evidence o jax_execution:" >&2
  echo "$GRANTS" >&2
  echo "-- PARAR ACÁ. No seguir pegando los bloques de abajo con esta cuenta." >&2
  false   # deja $? != 0 a propósito -- NO usa exit/return, y TAMPOCO se activa
          # set -e alrededor: medido de verdad con un pty real, `set -e` + un
          # comando que falla SÍ cierra una shell interactiva entera, igual que
          # `exit` -- exactamente lo que este bloque evita. Esto es un AVISO con
          # mensaje claro, no un corte de shell; MariaDB rechaza el DDL sin este
          # privilegio de todos modos, esa es la barrera real. Si ves el NO-GO de
          # arriba, frená vos: no sigas pegando los bloques de abajo.
fi
```

**Backup antes -- de verdad fail-closed, no silencioso, y por esquema.**
La versión original tenía `2>/dev/null || echo "(esquemas aún no
existen...)"`, que se traga CUALQUIER error (credenciales, permisos,
disco) e inventa una causa que puede ser falsa. Además dumpeaba
`--databases jax_evidence jax_execution` siempre JUNTOS: en el estado a
medio aplicar que `IF NOT EXISTS` existe justamente para tolerar (uno de
los dos esquemas ya está, el otro no), eso muere con "Unknown database" y
**no deja backup de ninguno de los dos** -- ni siquiera del que sí existe
y sí tiene datos reales que proteger. Cada esquema se respalda por
separado, sólo si existe; `set -e` sólo dentro del subshell, para que un
`mysqldump` que falla de verdad corte ACÁ sin apagar el resto de la
sesión si el operador sigue pegando comandos después:

```bash
(
  set -euo pipefail
  mkdir -p ~/backups
  for esquema in jax_evidence jax_execution; do
    existe=$(mysql --defaults-extra-file="$CNF" -N -e "SHOW DATABASES LIKE '$esquema'" | wc -l)
    if [ "$existe" = "0" ]; then
      echo "$esquema: no existe todavía -- nada que respaldar."
      continue
    fi
    B=~/backups/${esquema}_pre_260_$(date +%F-%H%M).sql
    ( umask 077
      mysqldump --defaults-extra-file="$CNF" --single-transaction --skip-lock-tables --no-tablespaces \
        --databases "$esquema" > "$B" )
    echo "$esquema: backup en $B ($(stat -c%s "$B") bytes)"
  done
)
```

**`DELIMITER` es una directiva del cliente `mysql`, no SQL** — el
`mysql` CLI interactivo o alimentado por stdin la entiende nativamente:

```bash
mysql --defaults-extra-file="$CNF" < policy/enforcement_evidence/migrations/001_enforcement_evidence.sql
mysql --defaults-extra-file="$CNF" < policy/execution_control/migrations/001_governed_execution.sql
```

Si el camino de aplicación NO pasa por el cliente `mysql` (por ejemplo, un
runner en Python contra `pymysql`, que no entiende `DELIMITER`), hay que
partir el archivo primero — la receta exacta, ya EJERCITADA de verdad
contra una MariaDB real en el job `governed-execution-mariadb` de CI, es
`tests/policy/test_execution_mariadb_integration.py::_apply_migration` /
`_apply_evidence_migration`: `sql.split("DELIMITER //", 1)` para separar
tablas de triggers, y dentro de cada mitad, `split(";")` / `split("//")`
por sentencia.

Las dos migraciones usan `CREATE TABLE IF NOT EXISTS` (ya eran
idempotentes) y, desde la ronda 1 de revisión, también
`CREATE TRIGGER IF NOT EXISTS` (soportado en MariaDB desde 10.1) — antes,
un re-run a medio aplicar moría en el primer trigger con "already exists",
que se lee como "ya estaba migrado" mientras los triggers de inmutabilidad
que faltan simplemente nunca se crean. Con `IF NOT EXISTS`, volver a
correr cualquiera de las dos migraciones completas siempre termina con
todo presente, se haya cortado donde se haya cortado antes --
**ejercitado de verdad** en
`test_migraciones_son_idempotentes_aplicadas_dos_veces`
(`tests/policy/test_execution_mariadb_integration.py`), que aplica las dos
migraciones completas DOS VECES seguidas (sin el atajo de
`_apply_migration`, que no vuelve a ejecutar nada si la tabla ya existe) y
confirma que la segunda pasada no falla.

**Otorgar `ALL PRIVILEGES` al usuario de aplicación sobre LOS DOS
esquemas, con la MISMA sesión y el MISMO `$CNF` de arriba (MAJOR-1, ronda
4 de revisión — corrige un `GRANT` que tumbaba el primer despacho
gobernado real).** La ronda anterior daba `INSERT, SELECT` en
`jax_evidence` y sólo `SELECT` en `jax_execution`. Dos problemas reales,
verificados contra el código, no supuestos:
- `policy/execution_control/storage.py` hace `INSERT` en `jax_execution`
  en varios caminos (`insert_authorization`, `create_execution` — dos
  `INSERT`, incluido el de `execution_authorization_consumptions` —,
  `append_event`, `consume_approval`, `save_dry_run`; líneas 82, 108, 109,
  112, 178, 187, 200). Con sólo `SELECT`, LAS MANOS arranca bien — el
  fallo no aparece hasta el primer despacho gobernado real, que muere con
  `1142 (42000): INSERT command denied`.
- `information_schema.triggers` sólo lista los triggers para los que
  quien consulta tiene el privilegio `TRIGGER` sobre ese esquema. Con
  `SELECT` solamente, la consulta de triggers del paso 3
  (`inspect_database_control()`) puede devolver CERO filas aun con el
  esquema perfectamente instalado, y el arranque falla con
  `DatabaseObservationMismatchError` sin que falte nada de verdad.

**Decisión del controlador:** `jax_evidence` y `jax_execution` son
esquemas dedicados de esta aplicación, no compartidos con ningún otro
sistema — `ALL PRIVILEGES` sobre los dos, en vez de ir ampliando una
lista de privilegios mínimos cada vez que aparece un camino de escritura
nuevo:

```bash
mysql --defaults-extra-file="$CNF" <<EOF
GRANT ALL PRIVILEGES ON jax_evidence.* TO '<JAX_DB_USER>'@'<host>';
GRANT ALL PRIVILEGES ON jax_execution.* TO '<JAX_DB_USER>'@'<host>';
FLUSH PRIVILEGES;
EOF
```
(sustituir `<JAX_DB_USER>` por el usuario real de `JAX_DB_USER` en
`/etc/jax/.env` — no se cita a mano acá porque es una credencial. **MINOR-C
(ronda 5 de revisión): `<host>` también hay que sustituirlo, y NO por
`%`.** "Sustituir por el usuario real" sin decir esto último deja
`'jaxappuser'@'%'` → `'jax_user'@'%'` -- una cuenta que no existe: la
cuenta real de este entorno tiene el host ACOTADO, no `%`. MariaDB
matchea `GRANT`/autenticación por el par (usuario, host) EXACTO; un
`GRANT` a `'jax_user'@'%'` cuando la cuenta real es
`'jax_user'@'172.30.5.%'` crea una fila nueva sin tocar la real, y el
usuario de aplicación real sigue sin los privilegios -- exactamente el
mismo síntoma que MAJOR-1 pero por la razón contraria. Sacar el host
EXACTO con `SELECT CURRENT_USER();` corrido por la propia aplicación (o,
desde la cuenta admin, `SELECT user, host FROM mysql.user WHERE user =
'<JAX_DB_USER>'`), no adivinarlo.)

**Recién ahora se borra `$CNF`**, al final de los dos pasos:

```bash
rm -f "$CNF"; trap - EXIT
```

**Antes de generar, un aviso de secuencia (MINOR de la ronda 2; hecho
corregido en la ronda 4 y otra vez en la ronda 5 -- las dos veces por
confiar en un número sin volver a medirlo en el momento correcto).** La
ronda 1 citaba `e09c3b3`; la ronda 4 lo corrigió a `2794cf3` (medido en
vivo el 2026-09-22 ~22:51 UTC) pero de ahí SUPUSO, sin verificarlo, que
ese commit "ya era posterior al merge que agrega
`scripts/generar_manifiesto_identidad.py`" -- **falso**: ese script lo
agrega `cd764f5`, que en `2794cf3` todavía NO está mergeado (ver la
sección nueva "Desplegar este PR en `/srv/jax-prod/jax`", antes del paso
4/5, con la verificación real y el comando de despliegue). **Cualquier
commit citado acá es una VERDAD OPERACIONAL, caduca por diseño -- volver
a medir con `git -C /srv/jax-prod/jax rev-parse HEAD` en el momento,
nunca reusar un número de este archivo, y nunca dar por hecho qué
contiene un commit sin comprobarlo (`git merge-base --is-ancestor`, o
`git show <commit>:<ruta>` sobre el archivo puntual).** Además, el mismo
día hubo DOS estados distintos a la vez -- el checkout en disco más
nuevo que el proceso `jax-las-manos` VIVO -- y es justo el escenario que
la PRECONDICIÓN del paso 9 y el gate `INSUFFICIENT_EVIDENCE` existen
para atrapar.

El script importa `_V1_REQUIRED_SOURCE_PATHS` desde SU PROPIO checkout
(el de donde se invoca), pero hashea los archivos de `--repo-root` — si
se lo corre desde un worktree nuevo (`--repo-root /srv/jax-prod/jax`)
contra un prod desactualizado, produce un manifiesto con la LISTA de
archivos requerida por el código nuevo pero los BYTES del código viejo
(cuando ni siquiera fallan por ausencia, dos versiones que no se
corresponden). **Orden correcto, siempre:** 1) desplegar el código nuevo
en `/srv/jax-prod/jax` primero (la sección nueva antes del paso 4/5, con
la verificación de que el script YA está ahí -- no darlo por supuesto);
2) recién ahí, invocar `/srv/jax-prod/jax/scripts/generar_manifiesto_identidad.py`
-- el script DE ESE checkout, no el de un worktree de desarrollo -- con
`--repo-root /srv/jax-prod/jax`.

### 5.5. Sembrar las definiciones de control -- sin esto, el paso 9 NO PUEDE dar GO

**Agregado por jax#266, tras ejecutar este runbook completo por primera vez
el 2026-09-22 y chocarse con esto en el último paso.** Aplicar la migración
del paso 4 deja `jax_evidence.control_definitions` **en CERO filas**: la
migración crea la tabla, nada la llena. En producción **nadie la sembraba** --
el único escritor era `MariaDBEvidenceStore.__persist_control_definition`,
privado, y sólo lo llamaba la fixture `_b8_runtime_identity_fixture` de
`tests/policy/test_execution_mariadb_integration.py`.

Con la tabla vacía, `readonly_status_snapshot` (`mariadb_store.py`, función
`capture()`) exige una fila EXACTA para la definición empaquetada, no la
encuentra, y levanta `EvidenceArtifactIntegrityError("snapshot definition
mismatch")`. El `except Exception` de `jaxctl/runtime.py::control_status` lo
convierte en `{"classification":"UNAVAILABLE"}` con exit 2 **sin la clave
`verdict`** -- o sea, el paso 9 de abajo, que sólo acepta `SUPPORTED`, era
INALCANZABLE en cualquier despliegue nuevo, y su triage mandaba a revisar el
usuario que corre `jaxctl`, que no era el problema.

Se corre en CADA despliegue, igual que el manifiesto: es idempotente (el
store inserta sólo si falta, y compara si ya está), y un control nuevo en el
catálogo necesita su fila antes de que alguien lo consulte. Mismo patrón que
los pasos 7 y 9 para las variables -- del proceso vivo, nunca sourceando
`/etc/jax/.env`:

```bash
sudo -u jaxsvc bash -c '
  cd /srv/jax-prod/jax &&
  MAINPID=$(systemctl show -p MainPID --value jax-las-manos) &&
  while IFS= read -r -d "" e; do
    case "$e" in JAX_DB_HOST=*|JAX_DB_PORT=*|JAX_DB_USER=*|JAX_DB_PASSWORD=*|JAX_DB_NAME=*) export "$e" ;;
  esac; done < "/proc/$MAINPID/environ" &&
  PYTHONPATH=/srv/jax-prod/jax /srv/jax-prod/jax/las_manos/.venv/bin/python3 \
    scripts/sembrar_definiciones_de_control.py'
```

(`JAX_DB_NAME` va en el `case` aunque el script tenga `"jax_memory"` de
default -- el mismo default que `jaxctl/runtime.py`. Heredarlo del proceso
vivo es lo correcto: si alguna vez difiere, el fallo tiene que ser de
conexión y no una escritura silenciosa en la base equivocada.)

Imprime una línea por control APENAS lo siembra (`sembrado` / `ya estaba`) y
un resumen al final. La salida es en vivo a propósito: cada `persist`
commitea su propia conexión, así que un fallo a mitad deja las anteriores
escritas y el operador tiene que ver cuáles.

**Si la base ya tiene OTRA definición para el mismo control** (deriva entre el
código desplegado y la base), el error real es de MariaDB -- medido el
2026-09-23 contra una base con una fila vieja puesta a mano:

    pymysql.err.IntegrityError: (1062, "Duplicate entry
    'CTL.B6.GOVERNED_DISPATCH-1' for key 'control_id'")

**y NO `EvidenceArtifactIntegrityError("definition collision")`**, como decía
la primera versión de este paso: el `SELECT ... FOR UPDATE` busca por
`control_definition_hash`, así que una definición distinta tiene otro hash,
no matchea, y el `INSERT` choca contra `UNIQUE(control_id, control_version)`.
La rama de "definition collision" sólo es alcanzable con el MISMO hash y
distinto payload, o sea una colisión de SHA-256. En cualquiera de los dos
casos, **no se arregla pisando la fila** (la tabla es append-only): hay que
resolver la deriva entre el código y la base.

Si el primer despliegue todavía no tiene un `jax-las-manos` vivo de dónde leer
el entorno, vale la misma excepción del paso 7 (`sudo -n cat /etc/jax/.env`
una vez, exportando a mano).

### 6. Crear el directorio de destino, y generar el manifiesto directo ahí
`/etc/jax` es `root:root 755` y `/etc/jax/build/` **no existe** la primera
vez -- root lo crea una sola vez, con el dueño ya correcto, para que
`jaxsvc` pueda escribir ahí directamente de acá en adelante:

```bash
sudo install -d -o jaxsvc -g jaxsvc -m 750 /etc/jax/build
```

Sin este paso, `escribir_atomico()` intenta `mkdir(parents=True)` sobre
`/etc/jax/build/` corriendo como `jaxsvc`, que no puede crear nada bajo
`/etc/jax` (root:root 755) → `PermissionError`. Desde la ronda 2 de
revisión el script atrapa esto y lo reporta como `ERROR: ...` con exit 1
(antes salía como traceback crudo) -- pero la solución sigue siendo crear
el directorio, no leer el traceback.

Con el directorio ya en manos de `jaxsvc`, generar escribe DIRECTO en la
ruta final -- no hace falta un paso aparte de "instalar" ni ningún
`chown`:

```bash
sudo -u jaxsvc /srv/jax-prod/jax/scripts/generar_manifiesto_identidad.py \
    --repo-root /srv/jax-prod/jax --production \
    --manifest-output /etc/jax/build/implementation-identity.manifest.json
```

(`--manifest-output` explícito por legibilidad: el default de
`--production` sin esa opción es
`implementation-identity.json.manifest.json`, técnicamente correcto pero
confuso de citar a mano en el paso siguiente -- ver MAYOR-6 de la ronda 2,
abajo.)

**Nota sobre las dos escrituras (ronda 3 de revisión):** este comando
escribe el manifest y la identidad por separado -- no son atómicas ENTRE
SÍ (si el proceso muere a mitad de camino, una queda y la otra no). El
orden (manifest primero, identidad después) es el seguro a propósito, y
está documentado en el propio script (`scripts/generar_manifiesto_identidad.py`,
comentario junto a las dos llamadas a `escribir_atomico`): un manifest sin
identidad todavía es inofensivo, mientras que el orden inverso dejaría una
identidad apuntando a un manifest que el paso 7 no podría leer.

`--production` fija la identidad en
`/etc/jax/build/implementation-identity.json` — el script ya no tiene un
default silencioso ahí (ronda 1 de revisión: un `--output` con default
apuntando a la ruta real de despliegue hacía que cualquier corrida de
prueba sin `--output` explícito intentara escribir ahí por accidente).

**`sudo -u jaxsvc` es obligatorio, no cosmético.** `/srv/jax-prod/jax` es
`jaxsvc:jaxsvc`; corrido como cualquier otro usuario (`fruiz` incluido),
`git` sale con `fatal: detected dubious ownership in repository at
'/srv/jax-prod/jax'` — código de salida 128, y el script lo refleja tal
cual (`RuntimeError`, ver Fail-closed condition). La alternativa, si por lo
que sea `sudo -u jaxsvc` no está disponible en el momento, es declarar la
excepción para ESE usuario invocador:
`git config --global --add safe.directory /srv/jax-prod/jax` — pero eso es
una excepción de confianza global de git para ese usuario, no algo para
dejar puesto sin pensarlo; `sudo -u jaxsvc` no la necesita porque jaxsvc ya
es el dueño.

El script se niega igual (nunca produce un `CLEAN` falso) si `git status`
advierte por **stderr** con código de salida 0 — el mismo síntoma que
"dubious ownership" pero sin el 128: un `Permission denied` puntual sobre
algún archivo del checkout. Antes de la ronda 1 de revisión, `arbol_sucio()`
sólo leía stdout y ese caso pasaba desapercibido.

**Ruta alternativa, sólo para inspeccionar sin tocar `/etc/jax/`** (por
ejemplo, para mirar el JSON antes de decidir si instalarlo): generar a un
`--output` de scratch, y recién si se ve bien, copiarlo a mano --
`install` necesita origen y destino DISTINTOS, así que esto NUNCA se hace
con `--production` apuntando al mismo lugar dos veces:

```bash
sudo -u jaxsvc /srv/jax-prod/jax/scripts/generar_manifiesto_identidad.py \
    --repo-root /srv/jax-prod/jax \
    --output /tmp/dry-run-identity.json \
    --manifest-output /tmp/dry-run-identity.manifest.json
# revisar /tmp/dry-run-identity.json a mano; recién entonces:
sudo install -o jaxsvc -g jaxsvc -m 600 /tmp/dry-run-identity.json /etc/jax/build/implementation-identity.json
sudo install -o jaxsvc -g jaxsvc -m 600 /tmp/dry-run-identity.manifest.json /etc/jax/build/implementation-identity.manifest.json
sudo rm -f /tmp/dry-run-identity.json /tmp/dry-run-identity.manifest.json
```

**Nunca instalar un artefacto `DIRTY`** (`source_state: "DIRTY"` en el
JSON) en `/etc/jax/build/` — ver la advertencia del paso 9.

### 7. Instalar el blob del build manifest en la base de evidencia
La identidad sólo lleva `build_manifest_blob_hash` — un *puntero*.
`verify_build_manifest` lo resuelve con
`store.get_evidence_blob(...)`, y en producción `store` es
`MariaDBEvidenceStore`, leyendo `jax_evidence.evidence_blobs`. Insertar los
bytes generados en el paso 6 (`put_evidence_blob` es idempotente,
direccionado por contenido).

**Esto NO es un script de Python cualquiera.** `policy.enforcement_evidence`
sólo es importable con el mismo intérprete/`PYTHONPATH` que usa el
servicio real -- el mismo venv que systemd invoca
(`/srv/jax-prod/jax/las_manos/.venv/bin/python3`), parado en la raíz del
repo, con `PYTHONPATH=/srv/jax-prod/jax` (el mismo que fija el drop-in
`z-pythonpath.conf`).

**El arranque de systemd está versionado en el repo** (decisión de
Fernando, 2026-09-25, opción A -- **el repo describe lo que corre, y
producción no se toca**, decisión reafirmada el mismo día al cerrar el
hallazgo de abajo): los cuatro drop-ins de cada servicio
(`checkout-de-produccion.conf`, `cuenta-de-servicio.conf`, y para
`jax-las-manos` además `z-pythonpath.conf`) que hasta entonces sólo vivían
en `/etc` de hall9000 viven ahora en `config/systemd/jax-las-manos.service.d/`
(y, para los otros tres servicios de B9/el proxy del Ejecutor, en
`config/systemd/*.service.d/` y `ops/ejecutor/jax-ejecutor-proxy.service.d/`),
junto con las CUATRO unidades base (`jax-las-manos.service`,
`jax-memory-worker.service`, `jax-memory-synthesis.service`,
`jax-ejecutor-proxy.service`) y el guion de sanidad
`ops/sbin/jax-checkout-de-produccion-sano.sh` -- 14 archivos en total, ver
`ops/manifiesto-arranque-instalado.tsv`. Las cuatro unidades base son el
fragmento CRUDO tal como está instalado en `/etc` (`User=fruiz`,
`WorkingDirectory=/home/fruiz/jax...`, sin `Environment=HOME=`) -- el
`User=jaxsvc`/`WorkingDirectory=/srv/jax-prod/jax`/`HOME` propio llegan
SIEMPRE por los drop-ins, nunca horneados en la base. Ningún consumidor
del repo depende ya de que la base sola describa el resultado final
(ver el punto de la configuración EFECTIVA, abajo).
- **Instalación**: a mano, con GO de Fernando --
  `sudo cp <archivo del repo> <ruta de /etc o /usr/local/sbin correspondiente
  (ver ops/manifiesto-arranque-instalado.tsv)>` seguido de
  `sudo systemctl daemon-reload` y, si corresponde, un reinicio del
  servicio afectado. Esto NO se automatiza: cada copia es una decisión de
  desplegar arranque nuevo a un servicio de producción.
- **Verificación**: `ops/verificar-arranque-instalado.sh` compara, archivo
  por archivo según `ops/manifiesto-arranque-instalado.tsv`, lo que hay en
  el repo contra lo instalado (sale 0 si coincide, y imprime cada
  diferencia si no). `tests/test_arranque_instalado.py` ejercita la forma
  del repo siempre (también en CI, sin necesitar el host de producción) y,
  sólo en el host de producción, corre el guion de verdad -- cableado al
  job `arranque-instalado-versionado` de `.github/workflows/policy.yml`
  (piso medido: 8 passed, 1 skipped fuera del host de producción).
- **Los instaladores YA instalan los drop-ins, derivados del manifiesto**:
  `ops/instalar-dropins-de-servicio.sh <unidad>.service.d <REPO> [DESTDIR]`
  lee `ops/manifiesto-arranque-instalado.tsv`, filtra las filas de esa
  unidad y copia cada una (`sudo install -o root -g root -m 0644`) --
  nunca una lista de archivos aparte. `config/systemd/install-memory-scope.sh`
  y `ops/ejecutor/instalar_registro_y_cerco.sh` lo invocan justo antes de
  su propio `daemon-reload`, para `jax-memory-worker.service.d` y
  `jax-ejecutor-proxy.service.d` respectivamente. `DESTDIR` (vacío por
  defecto, comportamiento de siempre) existe sólo para poder probar la
  instalación de las unidades systemd con una raíz temporal, sin tocar el
  `/etc` real -- el resto de esos dos guiones (nftables, `/etc/jax/.env`,
  `systemctl restart` de servicios reales) NO respeta `DESTDIR` y sigue
  tocando el sistema real siempre; por eso no se prueban de punta a punta,
  sólo la instalación de unidades (probada con éxito contra un `DESTDIR`
  temporal, ver la Biblioteca del cierre de este hallazgo, 2026-09-25).
- **La configuración EFECTIVA, no sólo la base, es lo que se audita**:
  `tests/test_ejecutor_cuenta_de_servicio.py::_configuracion_efectiva`
  fusiona la unidad base con sus drop-ins en el mismo orden que systemd
  (base primero, después cada `*.conf` de `<unidad>.d/` en orden
  alfabético -- el mismo criterio por el que `z-pythonpath.conf` en
  `jax-las-manos.service.d/` se aplica último a propósito) y exige
  `User=jaxsvc`/`HOME=/var/lib/jaxsvc` en el resultado fusionado, no en el
  archivo base solo. Antes (cuando la base ya declaraba `jaxsvc` a mano)
  mirar sólo la base alcanzaba; ahora que la base es el fragmento crudo,
  sólo la EFECTIVA prueba algo real -- verificado con un control negativo:
  quitar `cuenta-de-servicio.conf` hace fallar el test con el `User`/`HOME`
  reales (`fruiz`/ninguno).

**De dónde salen `JAX_DB_HOST`/`PORT`/`USER`/`PASSWORD` (ronda 3 de
revisión).** Este paso NO carga `/etc/jax/.env` directo -- ni con un punto
ni con `source` pegados a la ruta, y ni siquiera pasando por
`sudo -n cat`. El control mecánico `tests/test_env_se_lee_con_sudo.py`
escanea TODO el árbol (código, scripts, runbooks) buscando exactamente esa
forma -- un punto o un `source` seguidos directo de la ruta del archivo --
y la marca fail-closed si aparece en una instrucción; su propio docstring
explica por qué: esa forma de cargar el archivo tiene que salir en rojo
en este control, no en medio de un despliegue. La ronda anterior de este
runbook caía justo en eso (dos apariciones, en lo que hoy son los pasos 7
y 9) y puso CI en rojo. La decisión: en vez de agregar una excepción al
control o volver a cargar el archivo así, las variables salen del entorno
del PROCESO VIVO del servicio -- el mismo patrón que ya usa esta casa para
inspeccionar procesos ajenos sin tocar el archivo de secretos.
`/proc/<pid>/environ` de un proceso `jaxsvc` es legible por `jaxsvc`
mismo sin ningún privilegio extra (es dueño de su propio proceso) --
como todo este bloque corre `sudo -u jaxsvc`, no hace falta ni siquiera
`sudo -n cat` acá adentro. **Los valores nunca se imprimen** en ningún
paso:

```bash
cat > /tmp/instalar_blob_manifiesto.py <<'PY'
import os
import pymysql
from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore

def connect():
    return pymysql.connect(host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.environ["JAX_DB_USER"], password=os.environ["JAX_DB_PASSWORD"],
        database=os.environ.get("JAX_DB_NAME", "jax_memory"), charset="utf8mb4", autocommit=False)

data = open("/etc/jax/build/implementation-identity.manifest.json", "rb").read()
MariaDBEvidenceStore(connect).put_evidence_blob(data)
print("blob instalado")
PY

sudo -u jaxsvc bash -c '
  cd /srv/jax-prod/jax &&
  MAINPID=$(systemctl show -p MainPID --value jax-las-manos) &&
  while IFS= read -r -d "" entrada; do
    case "$entrada" in
      JAX_DB_HOST=*|JAX_DB_PORT=*|JAX_DB_USER=*|JAX_DB_PASSWORD=*) export "$entrada" ;;
    esac
  done < "/proc/$MAINPID/environ" &&
  PYTHONPATH=/srv/jax-prod/jax /srv/jax-prod/jax/las_manos/.venv/bin/python3 /tmp/instalar_blob_manifiesto.py
'
rm -f /tmp/instalar_blob_manifiesto.py
```

Si `jax-las-manos` no está corriendo todavía (primer despliegue, antes de
que exista un `MainPID`), este paso no tiene de dónde leer -- en ese caso
sí hace falta `sudo -n cat /etc/jax/.env` una única vez, EXACTAMENTE como
en los pasos 4/5 (el control lo permite: la forma que prohíbe es sourcear
DIRECTO, no leer con `sudo -n cat` y exportar a mano).

### 8. El símlink `/srv/jax`
**El desajuste `/srv/jax` vs `/srv/jax-prod/jax`.**
`_DEPLOYMENT_REPOSITORY_ROOT` en
`policy/enforcement_evidence/implementation_identity.py` es la constante
fija `"/srv/jax"` — y jax#260 la dejó así **a propósito**: "These are
deployment constants, not an API... accepting a path/root here would let a
caller redefine the bytes whose integrity is being claimed." Un request
nunca puede redirigir la verificación a otra raíz. Pero producción vive en
`/srv/jax-prod/jax` — verificado en vivo en hall9000:
`systemctl status jax-las-manos` muestra
`WorkingDirectory=/srv/jax-prod/jax/las_manos`, del drop-in
`checkout-de-produccion.conf` (2026-09-20) -- versionado en
`config/systemd/jax-las-manos.service.d/checkout-de-produccion.conf`
(y su equivalente para cada uno de los otros tres servicios) desde
2026-09-25, ver el punto anterior. La resolución es un
**símlink**, creado por root (`/srv/` es root-owned; `/srv/jax-prod/jax` es
`jaxsvc:jaxsvc`), como paso de despliegue explícito, no un cambio de
código:

```bash
sudo ln -sfn /srv/jax-prod/jax /srv/jax
readlink /srv/jax   # verificación: tiene que imprimir /srv/jax-prod/jax
```

Parametrizar `_DEPLOYMENT_REPOSITORY_ROOT` en vez de esto reabriría
exactamente la pregunta de autoridad que jax#260 cerró a propósito. El
símlink deja la constante intacta y la indirección del filesystem
explícita y auditable.

## Verification
Run identity/build-manifest verification.

### 9. Confirmar el arranque, y que el claim sea `SUPPORTED` -- nada menos que eso

**Sobre `set +e` en este paso (ronda 3 de revisión, razón corregida en la
ronda 4 -- la justificación original citaba un problema que la propia
ronda 3 ya había cerrado).** La ronda 3 decía que el `set -euo pipefail`
del paso 4/5 "persiste" en la misma terminal y por eso corta este paso
solo. **Eso es falso tal como quedó escrito ese mismo bloque**: el
`set -euo pipefail` de ahí vive DENTRO de un subshell entre paréntesis
(`( set -euo pipefail; ... )`, ver el backup del paso 4/5) desde la
propia ronda 3 -- las opciones de shell de un subshell no se propagan al
padre, así que no hay ningún `set -e` heredado esperando acá (ver también
el MAJOR-4 de la ronda 4, en el paso 4/5: ni ESE `set -e` ni el `false`
de la validación de grants cortan la terminal del operador). La razón
real para el `set +e` explícito de abajo es más simple y sigue siendo
válida por sí sola: este paso existe justamente para LEER códigos de
salida != 0 como DATOS, no como fallas -- `systemctl is-active` si el
servicio no llegó a levantar (exit 3), `curl` si `/health` no contesta
(exit 7), o `SALIDA=$(...)` si `jaxctl` devuelve `UNAVAILABLE` (exit 2).
Sin `set +e` explícito acá, cualquier `set -e` que el operador tenga
activo en su PROPIA sesión (su `.bashrc`, o haber pegado a mano algún
`set -e` de otra parte) cortaría este paso antes de que se vea el
`NO-GO` de abajo -- ese es el riesgo real, no una fuga del paso 4/5. Cada
bloque de este paso empieza con `set +e` explícito para no depender de
que nadie se acuerde ni de qué trae el operador puesto:

```bash
set +e
```

**MAJOR-3 (ronda 4 de revisión): PRECONDICIÓN antes de reiniciar --
producción es una puerta de un solo sentido.** Hoy LAS MANOS sirve
tráfico desde un proceso YA VIVO con el código VIEJO (medido en vivo,
2026-09-22: el proceso corre desde las 14:38 CST, mientras que
`/srv/jax-prod/jax` en disco está en un commit de las 22:38 -- ver la
nota de VERDAD OPERACIONAL antes del paso 6). Reiniciar aplica el código
nuevo de una sola vez: si el manifiesto no está bien instalado, el blob
no está en la base, o algún esquema/grant está incompleto,
`_configure_b7_trusted_runtime` tira el arranque completo y LAS MANOS
queda ABAJO -- no hay modo degradado (ver el punto 3 más arriba). **No
reiniciar sin correr esto primero**, con las funciones REALES de
producción -- no una reimplementación propia del chequeo, para no confiar
en una segunda versión del mismo criterio que se pueda desincronizar de
la real:

```bash
cd /srv/jax-prod/jax

ESPERADAS_TABLAS_EVIDENCE=$(grep -c '^CREATE TABLE' policy/enforcement_evidence/migrations/001_enforcement_evidence.sql)
ESPERADOS_TRIGGERS_EVIDENCE=$(grep -c '^CREATE TRIGGER' policy/enforcement_evidence/migrations/001_enforcement_evidence.sql)
ESPERADAS_TABLAS_EXECUTION=$(grep -c '^CREATE TABLE' policy/execution_control/migrations/001_governed_execution.sql)
ESPERADOS_TRIGGERS_EXECUTION=$(grep -c '^CREATE TRIGGER' policy/execution_control/migrations/001_governed_execution.sql)

cat > /tmp/precondicion_reinicio.py <<PY
# Conteos calculados AHORA MISMO contra las migraciones reales de este
# checkout -- nunca hardcodeados a mano, para no quedar desincronizados si
# el archivo de migración cambia en un commit futuro.
ESPERADAS_TABLAS_EVIDENCE = $ESPERADAS_TABLAS_EVIDENCE
ESPERADOS_TRIGGERS_EVIDENCE = $ESPERADOS_TRIGGERS_EVIDENCE
ESPERADAS_TABLAS_EXECUTION = $ESPERADAS_TABLAS_EXECUTION
ESPERADOS_TRIGGERS_EXECUTION = $ESPERADOS_TRIGGERS_EXECUTION

import os, sys
from datetime import datetime, timezone

import pymysql
from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
from policy.enforcement_evidence.implementation_identity import TrustedImplementationIdentityProvider
from policy.enforcement_evidence.database_evidence import inspect_database_control


def connect():
    return pymysql.connect(
        host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.environ["JAX_DB_USER"], password=os.environ["JAX_DB_PASSWORD"],
        database=os.environ.get("JAX_DB_NAME", "jax_memory"),
        charset="utf8mb4", autocommit=False, connect_timeout=5)


fallas = []

# 1) manifiesto instalado y legible por jaxsvc, blob en la base, y verifica
#    contra los bytes REALES de /srv/jax -- la MISMA llamada que hace el
#    arranque real (TrustedImplementationIdentityProvider.load()), de sólo
#    lectura: no escribe nada.
try:
    TrustedImplementationIdentityProvider(MariaDBEvidenceStore(connect)).load()
    print("OK  manifiesto+identidad instalados, legibles por jaxsvc, blob en la base, y verifican contra /srv/jax")
except Exception as exc:
    fallas.append(f"manifiesto/identidad: {exc!r}")

# 2) esquema jax_execution: la MISMA inspección que corre el arranque real
#    (también de sólo lectura) más la completitud de la migración entera
#    (el arranque sólo mira una parte; acá se mira todo).
try:
    inspect_database_control(
        connect, "CTL.B6.ONE_DECISION_ONE_EXECUTION", 1,
        deployment_id=os.environ["JAX_DEPLOYMENT_ID"],
        observed_at_utc=datetime.now(timezone.utc))
    print("OK  jax_execution pasa la misma inspección que corre el arranque real")
except Exception as exc:
    fallas.append(f"jax_execution (inspección de arranque): {exc!r}")

con = connect()
try:
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='jax_evidence'")
    n_tablas_evidence = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM information_schema.triggers WHERE trigger_schema='jax_evidence'")
    n_triggers_evidence = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='jax_execution'")
    n_tablas_execution = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM information_schema.triggers WHERE trigger_schema='jax_execution'")
    n_triggers_execution = cur.fetchone()[0]
finally:
    con.close()

# jax_evidence no tiene inspector de arranque propio (el arranque sólo la
# USA, no la inspecciona primero) -- esto llena ese hueco antes de apostar
# un reinicio.
if n_tablas_evidence != ESPERADAS_TABLAS_EVIDENCE or n_triggers_evidence != ESPERADOS_TRIGGERS_EVIDENCE:
    fallas.append(f"esquema jax_evidence incompleto: {n_tablas_evidence} tablas (esperadas {ESPERADAS_TABLAS_EVIDENCE}), {n_triggers_evidence} triggers (esperados {ESPERADOS_TRIGGERS_EVIDENCE})")
else:
    print(f"OK  esquema jax_evidence completo ({n_tablas_evidence} tablas, {n_triggers_evidence} triggers)")

if n_tablas_execution != ESPERADAS_TABLAS_EXECUTION or n_triggers_execution != ESPERADOS_TRIGGERS_EXECUTION:
    fallas.append(f"esquema jax_execution incompleto: {n_tablas_execution} tablas (esperadas {ESPERADAS_TABLAS_EXECUTION}), {n_triggers_execution} triggers (esperados {ESPERADOS_TRIGGERS_EXECUTION})")
else:
    print(f"OK  esquema jax_execution completo ({n_tablas_execution} tablas, {n_triggers_execution} triggers)")

# 3) grants del usuario de aplicación -- el mismo usuario que va a usar el
#    proceso NUEVO, consultado con sus propias credenciales (MAJOR-1).
#    MAJOR-A (ronda 5 de revisión): el chequeo de la ronda 4 era
#    `"ALL PRIVILEGES" in grants and "jax_evidence" in grants and
#    "jax_execution" in grants` -- un substring SIN esquema. El usuario de
#    aplicación YA puede tener "ALL PRIVILEGES" sobre OTRO esquema (p.ej.
#    jax_memory) y una línea cualquiera que sólo mencione "jax_evidence"
#    (aunque sea con SELECT nomás) alcanza para dar el substring por
#    satisfecho -- exactamente el escenario de MAJOR-1 (GRANT SELECT en
#    vez de ALL PRIVILEGES) pasando GO y muriendo en el primer INSERT real.
#    Se reusa acá el mismo parser con alcance real que ya se escribió para
#    el chequeo de la cuenta admin en el paso 4/5 (MINOR-4), portado a
#    Python en vez de bash.
import re as _re


def _grants_con_alcance_ok(texto_grants):
    objetivos_validos = {"*.*", "`jax_evidence`.*", "jax_evidence.*",
                          "`jax_execution`.*", "jax_execution.*"}
    for linea in texto_grants.splitlines():
        linea = linea.strip()
        if not linea.upper().startswith("GRANT "):
            continue
        m = _re.match(r"^GRANT (.+) ON (\S+) TO .*$", linea, _re.IGNORECASE)
        if not m:
            continue
        privilegios, objetivo = m.group(1), m.group(2)
        if objetivo not in objetivos_validos:
            continue   # privilegio sobre otro esquema: no cuenta para este gate
        privs = [p.strip().upper() for p in privilegios.split(",")]
        if "ALL PRIVILEGES" in privs:
            return True
    return False


con = connect()
try:
    cur = con.cursor()
    cur.execute("SHOW GRANTS FOR CURRENT_USER()")
    grants = "\n".join(row[0] for row in cur.fetchall())
finally:
    con.close()
if _grants_con_alcance_ok(grants):
    print("OK  grants del usuario de aplicación (ALL PRIVILEGES sobre jax_evidence y jax_execution, con alcance real)")
else:
    fallas.append("grants del usuario de aplicación insuficientes o sin alcance sobre jax_evidence/jax_execution (ver MAJOR-1, paso 4/5):\n" + grants)

if fallas:
    print("\nNO-GO -- no reiniciar. Fallas:")
    for f in fallas:
        print(" -", f)
    sys.exit(1)

print("\nGO -- las precondiciones pasan. Recién ahora es seguro reiniciar.")
PY

sudo -u jaxsvc bash -c '
  cd /srv/jax-prod/jax &&
  MAINPID=$(systemctl show -p MainPID --value jax-las-manos) &&
  while IFS= read -r -d "" entrada; do
    case "$entrada" in
      JAX_DB_HOST=*|JAX_DB_PORT=*|JAX_DB_USER=*|JAX_DB_PASSWORD=*|JAX_DEPLOYMENT_ID=*) export "$entrada" ;;
    esac
  done < "/proc/$MAINPID/environ" &&
  PYTHONPATH=/srv/jax-prod/jax /srv/jax-prod/jax/las_manos/.venv/bin/python3 /tmp/precondicion_reinicio.py
'
PRECOND=$?
rm -f /tmp/precondicion_reinicio.py

if [ "$PRECOND" = "0" ]; then
  sudo systemctl restart jax-las-manos
else
  echo "NO-GO: no reiniciar -- resolver los fallos de arriba primero." >&2
fi
```

Esto lee del proceso VIEJO todavía vivo (el que está por reiniciarse) --
funciona porque `JAX_DB_*`/`JAX_DEPLOYMENT_ID` no cambian entre el código
viejo y el nuevo, sólo cambian los ARCHIVOS que se están verificando. Si
`jax-las-manos` no está corriendo todavía (primer despliegue absoluto,
sin ningún proceso vivo de dónde leer), usar `sudo -n cat /etc/jax/.env`
una vez, igual que la excepción ya documentada en el paso 7.

**MINOR-3 (ronda 4 de revisión): al `systemctl restart` de arriba le
faltaba `sudo`.** A diferencia de todos los demás comandos privilegiados
de este runbook, sin `sudo` esto pide una contraseña interactiva y falla
sin tty si se pega en un pipe o un script no interactivo -- ya corregido
arriba (`sudo systemctl restart jax-las-manos`, condicionado a que la
precondición haya dado `PRECOND=0`).

```bash
systemctl is-active jax-las-manos    # esperado: active (código != 0 es información, no un corte)
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7777/health   # esperado: 200
```

**Esto NO alcanza.** `verify_build_manifest`/`verify_build_manifest_bytes`
NUNCA leen `source_state` — sólo comparan hashes de archivo. Un manifiesto
`DIRTY` instalado por error arranca LAS MANOS igual de bien que uno
`CLEAN`, y los dos comandos de arriba dan verde igual. Lo único que nota
la diferencia es `status_engine.derive_assertion`. Por eso el paso final
es consultar un claim de verdad, con la herramienta de sólo-lectura que ya
existe (`jaxctl`, `docs/operations/operational-manual.md`).

**`jaxctl` tiene que correr como `jaxsvc` -- si no, el gate lee verde
cuando en realidad falló.** El JSON de identidad es `600` propiedad de
`jaxsvc`; si se consulta como el operador (`fruiz`, o quien sea que esté
corriendo este runbook), `jaxctl` NO puede leer el archivo, y
`jaxctl/runtime.py::control_status` tiene un `except Exception` amplio que
convierte CUALQUIER falla en `UnavailableSource` →
`{"classification":"UNAVAILABLE",...}`, **sin la clave `"verdict"` en
absoluto**, exit 2. Mismo patrón que
`test_b8_jaxctl_control_unavailable_is_zero_write`
(`tests/policy/test_execution_mariadb_integration.py`) ya prueba contra
una MariaDB real: identidad no legible → `status == 2` y
`"status":"UNAVAILABLE"` en la salida.

Mismo motivo que el paso 7 para NO sourcear `/etc/jax/.env`: las
variables de DB salen del entorno del proceso vivo de `jax-las-manos`, y
como todo esto corre `sudo -u jaxsvc` (dueño de ese proceso), no hace
falta ningún `sudo` extra para leerlas. Nunca se imprimen.

```bash
cat > /tmp/verificar_claim.sh <<'EOF'
#!/bin/bash
cd /srv/jax-prod/jax
MAINPID=$(systemctl show -p MainPID --value jax-las-manos)
while IFS= read -r -d "" entrada; do
  case "$entrada" in
    JAX_DB_HOST=*|JAX_DB_PORT=*|JAX_DB_USER=*|JAX_DB_PASSWORD=*) export "$entrada" ;;
  esac
done < "/proc/$MAINPID/environ"
export PYTHONPATH=/srv/jax-prod/jax
exec /srv/jax-prod/jax/las_manos/.venv/bin/python3 -m jaxctl control \
    CTL.B6.GOVERNED_DISPATCH --version 1 --claim WRITTEN \
    --scope '{"environment":"SANDBOX_RUNTIME"}' --subjects '[]' --json
EOF
chmod +x /tmp/verificar_claim.sh

set +e   # SALIDA=$(...) con jaxctl devolviendo != 0 (UNAVAILABLE es exit 2) es
         # justo lo que este paso necesita LEER, no algo que tiene que cortar la sesión
SALIDA=$(sudo -u jaxsvc /tmp/verificar_claim.sh); EXIT=$?
rm -f /tmp/verificar_claim.sh
echo "$SALIDA"
```

**El chequeo es un ALLOW-LIST de un solo caso, no un deny-list de dos.**
La ronda anterior sólo declaraba NO-GO ante `UNAVAILABLE` o `STALE`
explícitos -- cualquier otra cosa (exit 0, sin esas dos palabras) caía en
un "revisar a mano" que en la práctica se lee como verde. Eso deja pasar
el escenario real más peligroso: código desplegado en
`/srv/jax-prod/jax` SIN reiniciar el servicio todavía -- el proceso sigue
vivo, `/health` sigue en 200, pero los hashes de las fuentes en disco ya
no coinciden con los que el manifiesto instalado describe. Eso hace
`_written=False` en `status_engine.py`, y con `--claim WRITTEN` el
verdict es `INSUFFICIENT_EVIDENCE` -- exit 0, sin `STALE`, sin
`UNAVAILABLE`, indistinguible de un `SUPPORTED` para un deny-list. Es
EXACTAMENTE el drift que este manifiesto existe para atrapar. La regla
correcta es al revés: sólo `SUPPORTED` es GO, TODO lo demás es NO-GO,
nombrando el verdict que encontró:

```bash
VERDICT=$(echo "$SALIDA" | grep -o '"verdict":"[A-Z_]*"' | head -1)

if [ "$EXIT" = "0" ] && [ "$VERDICT" = '"verdict":"SUPPORTED"' ]; then
  echo "GO: $VERDICT"
else
  echo "NO-GO: exit=$EXIT ${VERDICT:-(sin campo verdict -- ver classification/detail arriba)}"
fi
```

**Corrección sobre `"verdict":"UNVERIFIABLE"` (ronda 3 de revisión --
la glosa de la ronda 2 estaba mal).** La ronda anterior decía que este
verdict era "inalcanzable en la práctica" vía `jaxctl`. Eso es cierto para
UN solo camino que lo produce (`status_engine.py:17`,
`is_trusted_implementation_identity`), porque `readonly_status_snapshot`
sella la identidad como confiable o directamente lanza una excepción --
nunca deja pasar una identidad "no confiable" hasta `derive_assertion`.
Pero hay un SEGUNDO camino, independiente, que sí es alcanzable con una
identidad perfectamente legible: `status_engine.py:24` --
`if any(x.outcome is ObservationOutcome.ERROR for x in matched): return
AssertionVerdict.UNVERIFIABLE`. Si existe una observación real, para este
control/identidad/scope, con `outcome=ERROR`, el verdict es
`UNVERIFIABLE` aunque la identidad esté perfecta. La afirmación de la
ronda 2 era falsa por generalizar de un solo camino. No importa para el
gate de arriba (el allow-list ya trata cualquier verdict que no sea
`SUPPORTED` como NO-GO, `UNVERIFIABLE` incluido) -- importaba para no
dejar escrita una afirmación incorrecta sobre el código.

If either check fails, read `journalctl -u jax-las-manos -n 50` before
retrying anything.

**Triage por excepción, del paso donde puede aparecer cada una:**
- `RuntimeError("B7 trusted composition requires deployment and MariaDB
  configuration")` → paso 1, variable de entorno vacía.
- Error crudo de MariaDB ("table ... doesn't exist") → paso 2, esquema
  `jax_evidence` no aplicado.
- `DatabaseObservationMismatchError` → normalmente paso 3, `jax_execution`
  incompleto (tabla, índice único o trigger ausente) — leer el mensaje,
  dice cuál de los cuatro. **Pero también puede ser un falso positivo de
  privilegios (MAJOR-1, ronda 4 de revisión):** `information_schema.triggers`
  sólo devuelve los triggers para los que `JAX_DB_USER` tiene el privilegio
  `TRIGGER` — con el esquema perfectamente instalado pero sin
  `ALL PRIVILEGES` (paso 4/5), la consulta puede volver vacía y este mismo
  error aparece sin que falte nada de verdad. Confirmar con
  `SHOW GRANTS FOR '<JAX_DB_USER>'@'<host>';` (host EXACTO, no `%` --
  MINOR-C, ver el `GRANT` del paso 4/5) antes de tocar el esquema.
- `NO-GO: la cuenta admin no muestra CREATE+TRIGGER...` → paso 4/5, la
  cuenta usada para migrar no es admin -- no usar `$JAX_DB_USER` acá.
  **Esto es un AVISO, no un corte de shell** (ronda 4 de revisión — la
  afirmación anterior de que "corta" el script era falsa): deja `$?` != 0
  y el mensaje en pantalla, pero si el operador sigue pegando los bloques
  de abajo en la misma terminal, van a intentar correr igual -- MariaDB
  los va a rechazar por falta de privilegio recién ahí, no antes. Frenar
  acá es responsabilidad de quien pega los bloques, no del script (ver la
  nota "SHOW GRANTS con AVISO explícito" más arriba, con la medición real
  de por qué no se usa `set -e` para esto).
- el subshell del backup sale con `$? != 0` (por su propio `set -e`
  interno) sin haber terminado los backups → paso 4/5, falló el
  `mysqldump` de verdad (credenciales/permisos/disco). **Esto tampoco
  corta la sesión** (ronda 4 de revisión — misma corrección que la entrada
  de arriba): el `set -e` de ese bloque vive SÓLO dentro del subshell
  entre paréntesis, no se propaga a la terminal que lo invocó. Revisar
  `echo $?` inmediatamente después de ese bloque y resolver el fallo real
  ANTES de pegar las migraciones de abajo -- no saltarse el backup.
- Error de MariaDB por permisos (`1142` / `command denied`, en un
  `INSERT` sobre `jax_execution` o en cualquier otra operación) → paso 5,
  falta el `GRANT ALL PRIVILEGES` sobre el esquema que reclama el error
  (ronda 4 de revisión — la ronda anterior sólo otorgaba `SELECT` en
  `jax_execution`, y `policy/execution_control/storage.py` sí hace
  `INSERT` ahí; ver MAJOR-1).
- `ERROR: ...` (ya no traceback crudo, desde la ronda 2 de revisión) sobre
  `/etc/jax/build/` al GENERAR (paso 6) → falta el paso `sudo install -d
  -o jaxsvc -g jaxsvc -m 750 /etc/jax/build`.
- `git ... fatal: detected dubious ownership in repository at
  '/srv/jax-prod/jax'` → paso 6, no se corrió como `jaxsvc` (o falta
  `safe.directory`).
- `ModuleNotFoundError: No module named 'policy'` en el paso 7 → no se usó
  el intérprete del venv del servicio, o falta `PYTHONPATH=/srv/jax-prod/jax`,
  o no se hizo `cd /srv/jax-prod/jax` primero.
- `EvidenceBlobMissingError` → paso 7, el blob nunca se insertó en
  `jax_evidence.evidence_blobs` (o el símlink `/srv/jax` del paso 8 apunta
  a otro lado y el hash no coincide con lo que hay en disco).
- `UntrustedImplementationIdentityError` ("build manifest inválido o
  drift") → el código cambió después de generar el manifiesto del paso 6:
  regenerar (ver Fail-closed condition, abajo).
- El paso 9 imprime `NO-GO: exit=... (sin campo verdict...)` con
  `"detail":"Block 7 authoritative status unavailable"` → **primero mirar si
  se corrió el paso 5.5**: con `control_definitions` vacía el síntoma es
  EXACTAMENTE éste, y correr el sembrador lo resuelve sin tocar nada más
  (medido en vivo el 2026-09-22, que es de donde salió ese paso). Recién si
  el paso 5.5 ya se corrió, seguir con lo de abajo.
- El paso 9 imprime `NO-GO: exit=... (sin campo verdict...)` → casi
  siempre `"classification":"UNAVAILABLE"` en `$SALIDA`: `jaxctl` no pudo
  leer o verificar la identidad -- lo primero a revisar es si corrió como
  `jaxsvc`; si ya corrió como `jaxsvc`, leer `"detail"` en la salida
  completa (`echo "$SALIDA"`, ya impreso arriba).
- El paso 9 imprime `NO-GO: exit=0 "verdict":"STALE"` → se instaló (o
  quedó) un manifiesto `DIRTY`; regenerar sin `--allow-dirty` e instalar
  de nuevo (pasos 6-7).
- El paso 9 imprime `NO-GO: exit=0 "verdict":"INSUFFICIENT_EVIDENCE"` →
  el escenario real que el allow-list existe para atrapar: código nuevo
  en `/srv/jax-prod/jax` pero el manifiesto instalado describe hashes que
  ya no coinciden (deploy sin regenerar, o regenerado contra el checkout
  equivocado -- ver el aviso de secuencia antes del paso 6). Regenerar e
  instalar de nuevo (pasos 6-7), NO reiniciar y esperar que se arregle
  solo.
- El paso 9 imprime `NO-GO: exit=0 "verdict":"UNVERIFIABLE"` → existe una
  observación real con `outcome=ERROR` para este control/identidad/scope
  (`status_engine.py:24`) -- no es un problema del manifiesto en sí,
  revisar qué observación falló antes de reintentar nada de este runbook.
- El paso 9 imprime `NO-GO: exit=0 "verdict":"FAILED"` (MINOR-5, ronda 4
  de revisión -- faltaba en esta lista) → `status_engine.py:23`, existe
  una observación real, para este control/identidad/scope, con
  `outcome=FAILED`. A diferencia de `UNVERIFIABLE` (`outcome=ERROR`, la
  observación en sí falló al evaluarse), acá el control SÍ se evaluó y el
  resultado fue negativo -- revisar qué observación dio `FAILED` antes de
  reintentar nada de este runbook; tampoco es un problema del manifiesto.
## Fail-closed condition
Drift or missing identity: stop.

**El manifiesto se regenera en CADA despliegue — nunca se reutiliza entre
commits.** `build_manifest_blob_hash` ata la identidad a los bytes fuente
exactos de `_V1_REQUIRED_SOURCE_PATHS` al momento de generarlo; el próximo
commit cambia al menos uno de esos hashes (ese es el propósito completo
del manifiesto: `verify_build_manifest_bytes` recalcula el SHA-256 de cada
archivo listado contra lo que hay en disco y falla ante cualquier
discrepancia). Desplegar código nuevo con el `implementation-identity.json`
de ayer no sirve una identidad vieja-pero-funcional en silencio: falla
cerrado igual que un archivo ausente, porque los hashes ya no coinciden.
Regenerar (pasos 6-7 de arriba) es el arreglo — no hay actualización
parcial/incremental.
## Recovery / escalation

**Rollback real, con comandos -- no "escalar" en abstracto (MAJOR-3,
ronda 4 de revisión).** El paso 9 es una puerta de un solo sentido: un
reinicio que sale mal deja LAS MANOS abajo hasta que se resuelva. Si la
PRECONDICIÓN de antes del reinicio no se corrió, o si después de
reiniciar el `NO-GO` no se resuelve rápido, el camino de vuelta es un
`git reset --hard` al último commit que NO exige el manifiesto, más un
reinicio.

**MAJOR-B (ronda 5 de revisión): el radio de impacto de este rollback,
medido, no supuesto.** `a444a9e..2794cf3` son **180 commits**, 21 de
ellos merges de PR -- incluidos #256, #257, #258, #259, **#260 (este
mismo trabajo)**, #261, #262, #263 y #265. Un `git reset --hard` a
`a444a9e` no vuelve atrás sólo el manifiesto: vuelve atrás TODO lo que
esos 21 PRs cambiaron. Y `/srv/jax-prod/jax` no es sólo LAS MANOS --
medido con `systemctl show -p WorkingDirectory` contra las unidades
reales de este host, **seis servicios systemd** corren código de ESTE
checkout: `jax-las-manos`, `jax-ejecutor-proxy`,
`jax-limpiar-bases-de-test`, `jax-memory-synthesis`, `jax-memory-worker`,
`jax-revisar-indice-vectorial` (los últimos cuatro además tienen su
propio `.timer` que los dispara periódicamente). **El paso 2 de abajo
sólo reinicia `jax-las-manos`** -- el `reset --hard` del código en disco
afecta a los seis, pero los otros cinco siguen corriendo con el código
VIEJO en memoria hasta que también se reinicien (o hasta que su próximo
disparo de `.timer` los relance ya contra el código rebajado). Antes de
decidir un rollback, confirmar con Fernando si alguno de los otros cinco
importa para el incidente en curso -- normalmente no, para un rollback de
emergencia acotado a LAS MANOS, pero es una decisión, no un hecho
automático.

**Cómo volver a avanzar después de un rollback.** Un `reset --hard` deja
el checkout DETRÁS del remoto -- no alcanza con "esperar al próximo
deploy": el próximo despliegue normal (`git fetch` + `reset --hard
origin/master`, el mismo paso que este runbook agrega antes del paso 4/5)
VUELVE A TRAER el commit con el manifiesto exigido, y sin completar los
pasos 4 a 8 de este runbook antes de ese redeploy, se repite el mismo
apagón. El camino de vuelta es: resolver la causa del `NO-GO` (ver el
triage del paso 9), volver a desplegar (`git fetch` + `reset --hard
origin/master`), completar los pasos 4 a 8 contra ese checkout, y recién
ahí reintentar el paso 9 -- nunca reintentar el reinicio sin haber
corrido la PRECONDICIÓN de nuevo.

**1) Encontrar el commit al que volver -- como `jaxsvc`, no como el
operador (BLOCK-B, ronda 5 de revisión -- corrige un comando que fallaba
tal como estaba escrito).** El manifiesto lo exige
`_configure_b7_trusted_runtime` (`las_manos/server.py`, jax#260). Esto se
busca en el momento, con el repo real -- no se copia un hash citado en
este archivo, porque puede haber más commits después que la reintroduzcan
de otra forma. **La ronda 4 escribía esto con `cd /srv/jax-prod/jax` y
`git log` directo, como el operador -- confirmado en rojo:** sale `fatal:
detected dubious ownership`, `INTRODUCTORIO` queda vacío, y el paso 2 de
abajo termina siendo `reset --hard "^"` -- justo en el peor momento para
que un comando falle así. Mismo motivo que el paso 6: correr como
`jaxsvc`, dueño del checkout:

```bash
INTRODUCTORIO=$(sudo -u jaxsvc git -C /srv/jax-prod/jax log -S"_configure_b7_trusted_runtime" --oneline --reverse -- las_manos/server.py | head -1 | cut -d' ' -f1)
OBJETIVO_ROLLBACK="${INTRODUCTORIO}^"
# verificación: tiene que dar 0 -- ese commit todavía no tiene la función
sudo -u jaxsvc git -C /srv/jax-prod/jax show "$OBJETIVO_ROLLBACK":las_manos/server.py | grep -c _configure_b7_trusted_runtime
```

Medido en vivo el 2026-09-22, ya como `jaxsvc`: `INTRODUCTORIO=55a9ad1`
("fix: close Block 7 evidence trust boundaries"),
`OBJETIVO_ROLLBACK=a444a9e` ("fix: preserve MariaDB integration migration
isolation", 2026-09-21) — 0 coincidencias, confirmado. **Este hash
también es una VERDAD OPERACIONAL de hoy: volver a correr el comando de
arriba antes de confiar en un valor citado acá.**

**2) Rollback -- mismo usuario dueño del checkout que el paso 6 (mismo
motivo: `dubious ownership` si se corre como cualquier otro usuario), y
`sudo` en el restart igual que el MINOR-3 de arriba. Sólo reinicia
`jax-las-manos` -- ver el radio de impacto de arriba antes de asumir que
esto alcanza:**

```bash
sudo -u jaxsvc git -C /srv/jax-prod/jax reset --hard "$OBJETIVO_ROLLBACK"
sudo systemctl restart jax-las-manos
systemctl is-active jax-las-manos   # esperado: active
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7777/health   # esperado: 200
```

**3) Si el rollback tampoco levanta**, el problema no es el manifiesto —
es otra cosa (algo recién introducido por el mismo deploy, o
infraestructura). Recién ahí corresponde `journalctl -u jax-las-manos -n
100` y escalar a Fernando con esa evidencia -- no antes, y no en lugar de
intentar el rollback de arriba.
## Prohibited actions
Do not hand-edit CLEAN identity claims.

Do not hand-write `implementation-identity.json` o su blob de manifiesto
-- siempre a través de `scripts/generar_manifiesto_identidad.py`, para que
los hashes salgan calculados por el mismo algoritmo con el que
`verify_build_manifest_bytes` los revisa, nunca adivinados o copiados de
un despliegue anterior.

No instalar un artefacto con `source_state: "DIRTY"` en
`/etc/jax/build/` -- ver la advertencia del paso 9.
