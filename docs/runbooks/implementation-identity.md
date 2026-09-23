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

### 4. Aplicar las dos migraciones si faltan, con backup antes -- con un usuario ADMIN
**`CREATE SCHEMA`/`TABLE`/`TRIGGER` no son privilegios de aplicación.**
`$JAX_DB_USER` de `/etc/jax/.env` es (debe ser) un usuario de mínimo
privilegio -- el que el paso 5 le otorga `INSERT`/`SELECT` a propósito,
nunca DDL. Este paso usa una cuenta ADMINISTRATIVA aparte, provista por
quien lo ejecuta (nunca la misma que `$JAX_DB_USER`, y nunca escrita en
este archivo). Un archivo de *defaults* con permisos `600` evita que la
contraseña quede visible en `ps` para cualquier usuario del host
(`-p$PASSWORD` en la línea de comandos sí queda visible así):

```bash
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

# Chequeo previo: si esta cuenta no tiene DDL, mejor enterarse ACÁ que a
# mitad de una migración a medio aplicar.
mysql --defaults-extra-file="$CNF" -e "SHOW GRANTS FOR CURRENT_USER();"
```

**Backup antes -- de verdad fail-closed, no silencioso.** La versión
anterior de este paso tenía `2>/dev/null || echo "(esquemas aún no
existen...)"`, que se traga CUALQUIER error (credenciales, permisos,
disco) e inventa una causa que puede ser falsa -- en un segundo intento,
con `jax_evidence` ya poblado, un dump que falla de verdad deja un archivo
vacío y un mensaje tranquilizador, y el paso siguiente aplica DDL sin
backup real. La distinción correcta es "el esquema no existe" (se
verifica ANTES, con `SHOW DATABASES`, no se infiere del resultado del
dump) contra "cualquier otro error" (que corta, con `set -e`):

```bash
set -euo pipefail

evidence_existe=$(mysql --defaults-extra-file="$CNF" -N -e "SHOW DATABASES LIKE 'jax_evidence'" | wc -l)
execution_existe=$(mysql --defaults-extra-file="$CNF" -N -e "SHOW DATABASES LIKE 'jax_execution'" | wc -l)

if [ "$evidence_existe" = "0" ] && [ "$execution_existe" = "0" ]; then
  echo "los dos esquemas son nuevos -- nada que respaldar todavía."
else
  B=~/backups/jax_evidence_execution_pre_260_$(date +%F-%H%M).sql
  ( umask 077
    mysqldump --defaults-extra-file="$CNF" --single-transaction --skip-lock-tables --no-tablespaces \
      --databases jax_evidence jax_execution > "$B" )
  # Sin `set -e` esto no cortaría solo: CON `set -e`, un mysqldump que
  # falla de verdad (no "no existe la base", sino permisos/disco/red)
  # termina el script ACÁ, antes de tocar DDL.
  echo "backup en $B ($(stat -c%s "$B") bytes)"
fi
```

**`DELIMITER` es una directiva del cliente `mysql`, no SQL** — el
`mysql` CLI interactivo o alimentado por stdin la entiende nativamente:

```bash
mysql --defaults-extra-file="$CNF" < policy/enforcement_evidence/migrations/001_enforcement_evidence.sql
mysql --defaults-extra-file="$CNF" < policy/execution_control/migrations/001_governed_execution.sql
rm -f "$CNF"; trap - EXIT
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
idempotentes) y, desde esta ronda, también `CREATE TRIGGER IF NOT EXISTS`
(soportado en MariaDB desde 10.1) — antes, un re-run a medio aplicar moría
en el primer trigger con "already exists", que se lee como "ya estaba
migrado" mientras los triggers de inmutabilidad que faltan simplemente
nunca se crean. Con `IF NOT EXISTS`, volver a correr cualquiera de las dos
migraciones completas siempre termina con todo presente, se haya cortado
donde se haya cortado antes -- **ejercitado de verdad** en
`test_migraciones_son_idempotentes_aplicadas_dos_veces`
(`tests/policy/test_execution_mariadb_integration.py`), que aplica las dos
migraciones completas DOS VECES seguidas (sin el atajo de
`_apply_migration`, que no vuelve a ejecutar nada si la tabla ya existe) y
confirma que la segunda pasada no falla.

### 5. El usuario de la DB del servicio necesita `INSERT` en `jax_evidence.*`
El arranque escribe la identidad (`__record_identity`) y los artefactos de
evidencia del propio `DatabaseControlInspector` — sin `INSERT`, la primera
escritura del arranque falla. Otorgar con la MISMA cuenta admin del paso 4
(reconstruir `$CNF` si ya se borró):

```sql
GRANT INSERT, SELECT ON jax_evidence.* TO 'jaxappuser'@'%';
GRANT SELECT ON jax_execution.* TO 'jaxappuser'@'%';
FLUSH PRIVILEGES;
```
(sustituir `'jaxappuser'@'%'` por el usuario real de `JAX_DB_USER` en
`/etc/jax/.env` — no se cita a mano acá porque es una credencial; aplicar
con `mysql --defaults-extra-file="$CNF"`, nunca `-p` en la línea de
comandos).

**Antes de generar, un aviso de secuencia (MINOR de la ronda 2 de
revisión).** `/srv/jax-prod/jax` hoy corre `e09c3b3` — un commit ANTERIOR
al que agrega `scripts/generar_manifiesto_identidad.py`, así que ese
checkout ni siquiera tiene el script todavía. El script importa
`_V1_REQUIRED_SOURCE_PATHS` desde SU PROPIO checkout (el de donde se
invoca), pero hashea los archivos de `--repo-root` — si se lo corre desde
un worktree nuevo (`--repo-root /srv/jax-prod/jax`) contra un prod
desactualizado, produce un manifiesto con la LISTA de archivos requerida
por el código nuevo pero los BYTES del código viejo (cuando ni siquiera
fallan por ausencia, dos versiones que no se corresponden). **Orden
correcto, siempre:** 1) desplegar el código nuevo en `/srv/jax-prod/jax`
primero (el paso normal de deploy de este repo); 2) recién ahí, invocar
`/srv/jax-prod/jax/scripts/generar_manifiesto_identidad.py` -- el script
DE ESE checkout, no el de un worktree de desarrollo -- con
`--repo-root /srv/jax-prod/jax`.

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
`z-pythonpath.conf`). Y `/etc/jax/.env` tiene valores que pueden llevar
espacios o comillas -- `env $(cat archivo | xargs)` los rompe; `. archivo`
en un subshell no:

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
  set -a; . /etc/jax/.env; set +a &&
  PYTHONPATH=/srv/jax-prod/jax /srv/jax-prod/jax/las_manos/.venv/bin/python3 /tmp/instalar_blob_manifiesto.py
'
rm -f /tmp/instalar_blob_manifiesto.py
```

(`jaxsvc` puede leer `/etc/jax/.env` directamente -- es `root:jaxsvc 640` --
no hace falta `sudo -n cat` acá como en los pasos 4/5, que corren como un
usuario distinto.)

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
`checkout-de-produccion.conf` (2026-09-20). La resolución es un
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

### 9. Confirmar el arranque, y que el claim no haya quedado STALE ni UNAVAILABLE
```bash
systemctl restart jax-las-manos      # o esperar al próximo deploy
systemctl is-active jax-las-manos    # esperado: active
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7777/health   # esperado: 200
```

**Esto NO alcanza.** `verify_build_manifest`/`verify_build_manifest_bytes`
NUNCA leen `source_state` — sólo comparan hashes de archivo. Un manifiesto
`DIRTY` instalado por error arranca LAS MANOS igual de bien que uno
`CLEAN`, y los dos comandos de arriba dan verde igual. Lo único que nota
la diferencia es `status_engine.derive_assertion`, que hace `STALE`
cualquier claim B7 cuya identidad no sea `CLEAN` — en silencio, sin que el
arranque ni el health-check lo digan. Por eso el paso final es consultar
un claim de verdad, con la herramienta de sólo-lectura que ya existe
(`jaxctl`, `docs/operations/operational-manual.md`).

**`jaxctl` tiene que correr como `jaxsvc` -- si no, el gate lee verde
cuando en realidad falló.** El JSON de identidad es `600` propiedad de
`jaxsvc`; si se consulta como el operador (`fruiz`, o quien sea que esté
corriendo este runbook), `jaxctl` NO puede leer el archivo, y
`jaxctl/runtime.py::control_status` tiene un `except Exception` amplio que
convierte CUALQUIER falla en `UnavailableSource` →
`{"classification":"UNAVAILABLE",...}`, **sin la clave `"verdict"` en
absoluto**, exit 2. Si el chequeo sólo busca los strings `"STALE"` o
`"UNVERIFIABLE"` en la salida, NINGUNO de los dos aparece -- y un operador
apurado lee eso como GO. Mismo patrón que `test_b8_jaxctl_control_unavailable_is_zero_write`
(`tests/policy/test_execution_mariadb_integration.py`) ya prueba contra
una MariaDB real: identidad no legible → `status == 2` y
`"status":"UNAVAILABLE"` en la salida, sin ninguna mención a `STALE`.

```bash
cat > /tmp/verificar_claim.sh <<'EOF'
#!/bin/bash
set -a; . /etc/jax/.env; set +a
cd /srv/jax-prod/jax
export PYTHONPATH=/srv/jax-prod/jax
exec /srv/jax-prod/jax/las_manos/.venv/bin/python3 -m jaxctl control \
    CTL.B6.GOVERNED_DISPATCH --version 1 --claim WRITTEN \
    --scope '{"environment":"SANDBOX_RUNTIME"}' --subjects '[]' --json
EOF
chmod +x /tmp/verificar_claim.sh

SALIDA=$(sudo -u jaxsvc /tmp/verificar_claim.sh); EXIT=$?
rm -f /tmp/verificar_claim.sh
echo "$SALIDA"
```

**El chequeo tiene que ser explícito sobre los TRES resultados posibles**,
no sólo buscar la palabra "STALE":

```bash
if [ "$EXIT" != "0" ]; then
  echo "NO-GO: jaxctl salió con código $EXIT"
elif echo "$SALIDA" | grep -q '"classification":"UNAVAILABLE"'; then
  echo "NO-GO: UNAVAILABLE -- jaxctl no pudo leer o verificar la identidad (ver 'detail' arriba). ¿Corrió como jaxsvc?"
elif echo "$SALIDA" | grep -q '"verdict":"STALE"'; then
  echo "NO-GO: STALE -- el manifiesto instalado no es CLEAN, o describe un commit distinto al que corre"
else
  echo "sin UNAVAILABLE ni STALE -- revisar 'verdict' igual a mano antes de dar GO"
fi
```

**Corrección sobre `"verdict":"UNVERIFIABLE"` (ronda 2 de revisión):** ese
verdict es, en la práctica, INALCANZABLE a través de `jaxctl` -- no porque
no exista en el código (`derive_assertion` sí lo puede devolver), sino
porque el camino real de `jaxctl` (`query_control_status` →
`TrustedImplementationIdentityProvider` → `MariaDBEvidenceStore.readonly_status_snapshot`)
**lanza una excepción antes** de llegar a construir un `ControlStatusView`
en cualquier escenario donde la identidad no sea de fiar (archivo
inaccesible, hash que no matchea, fila ausente en la DB) -- y esa
excepción es exactamente lo que produce `UNAVAILABLE` de arriba. Lo que
hay que vigilar es `classification == "UNAVAILABLE"`, no el string
`"UNVERIFIABLE"`.

If either check fails, read `journalctl -u jax-las-manos -n 50` before
retrying anything.

**Triage por excepción, del paso donde puede aparecer cada una:**
- `RuntimeError("B7 trusted composition requires deployment and MariaDB
  configuration")` → paso 1, variable de entorno vacía.
- Error crudo de MariaDB ("table ... doesn't exist") → paso 2, esquema
  `jax_evidence` no aplicado.
- `DatabaseObservationMismatchError` → paso 3, `jax_execution` incompleto
  (tabla, índice único o trigger ausente) — leer el mensaje, dice cuál de
  los cuatro.
- `SHOW GRANTS FOR CURRENT_USER()` sin `CREATE`/`TRIGGER`/`ALTER` → paso 4,
  la cuenta usada para migrar no es admin -- no usar `$JAX_DB_USER` acá.
- `mysqldump` corta el script (con `set -e`) sin instalar nada → paso 4,
  falló el backup por algo real (credenciales/permisos/disco) -- resolver
  ESO antes de reintentar, no saltarse el backup.
- Error de MariaDB por permisos al insertar (`1142` / `command denied`) →
  paso 5, falta el `GRANT INSERT`.
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
- `"classification":"UNAVAILABLE"` en el paso 9 → `jaxctl` no pudo leer o
  verificar la identidad -- lo primero a revisar es si corrió como
  `jaxsvc`; si ya corrió como `jaxsvc`, leer `"detail"` en la salida.
- `"verdict":"STALE"` con todo lo anterior en verde → paso 9, se instaló
  (o quedó) un manifiesto `DIRTY`; regenerar sin `--allow-dirty` e
  instalar de nuevo (pasos 6-7).
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
Redeploy verified build or escalate.
## Prohibited actions
Do not hand-edit CLEAN identity claims.

Do not hand-write `implementation-identity.json` o su blob de manifiesto
-- siempre a través de `scripts/generar_manifiesto_identidad.py`, para que
los hashes salgan calculados por el mismo algoritmo con el que
`verify_build_manifest_bytes` los revisa, nunca adivinados o copiados de
un despliegue anterior.

No instalar un artefacto con `source_state: "DIRTY"` en
`/etc/jax/build/` -- ver la advertencia del paso 9.
