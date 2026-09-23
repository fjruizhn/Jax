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

### 4 y 5. Aplicar las dos migraciones y otorgar `INSERT` -- una sola sesión, con un usuario ADMIN
**`CREATE SCHEMA`/`TABLE`/`TRIGGER` no son privilegios de aplicación.**
`$JAX_DB_USER` de `/etc/jax/.env` es (debe ser) un usuario de mínimo
privilegio -- el que este mismo paso le otorga `INSERT`/`SELECT` al final,
nunca DDL. Todo lo de abajo usa una cuenta ADMINISTRATIVA aparte, provista
por quien lo ejecuta (nunca la misma que `$JAX_DB_USER`, y nunca escrita
en este archivo).

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

**`SHOW GRANTS` con gate de verdad, no sólo un print.** La ronda anterior
imprimía los grants y confiaba en que el operador los leyera -- esto CORTA
el script si la cuenta no alcanza, ANTES de tocar DDL:

```bash
GRANTS=$(mysql --defaults-extra-file="$CNF" -N -e "SHOW GRANTS FOR CURRENT_USER();")
if echo "$GRANTS" | grep -qi "ALL PRIVILEGES"; then
  echo "cuenta admin OK (ALL PRIVILEGES)."
elif echo "$GRANTS" | grep -qi '\bCREATE\b' && echo "$GRANTS" | grep -qi '\bTRIGGER\b'; then
  echo "cuenta admin OK (CREATE + TRIGGER)."
else
  echo "NO-GO: la cuenta admin no muestra CREATE+TRIGGER (ni ALL PRIVILEGES):" >&2
  echo "$GRANTS" >&2
  echo "-- PARAR ACÁ. No seguir pegando los bloques de abajo con esta cuenta." >&2
  false   # deja $? != 0 a propósito: no se usa exit/return -- este bloque se
          # pega interactivo, y `exit` cerraría la terminal entera en vez de
          # sólo frenar este chequeo. MariaDB igual rechaza el DDL sin este
          # privilegio -- esto es un fail-fast con mensaje claro, no la
          # única barrera real.
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

**Otorgar `INSERT`/`SELECT` al usuario de aplicación, con la MISMA sesión
y el MISMO `$CNF` de arriba** (el arranque escribe la identidad vía
`__record_identity` y los artefactos de `DatabaseControlInspector` — sin
`INSERT`, la primera escritura del arranque falla):

```bash
mysql --defaults-extra-file="$CNF" <<EOF
GRANT INSERT, SELECT ON jax_evidence.* TO 'jaxappuser'@'%';
GRANT SELECT ON jax_execution.* TO 'jaxappuser'@'%';
FLUSH PRIVILEGES;
EOF
```
(sustituir `'jaxappuser'@'%'` por el usuario real de `JAX_DB_USER` en
`/etc/jax/.env` — no se cita a mano acá porque es una credencial.)

**Recién ahora se borra `$CNF`**, al final de los dos pasos:

```bash
rm -f "$CNF"; trap - EXIT
```

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

### 9. Confirmar el arranque, y que el claim sea `SUPPORTED` -- nada menos que eso

**Sobre `set -e` y bloques compartiendo una sola terminal (ronda 3 de
revisión).** Si un operador va pegando los bloques de este runbook, uno
tras otro, EN LA MISMA terminal, el `set -euo pipefail` del paso 4/5
persiste ahí -- no se apaga solo al terminar ese bloque de código, porque
`set -e` es una opción de la shell, no del bloque de markdown. Sin
neutralizarlo, este paso 9 (que existe justamente para leer códigos de
salida != 0 como DATOS, no como fallas) se corta solo: en
`systemctl is-active` si el servicio no llegó a levantar (exit 3), en
`curl` si `/health` no contesta (exit 7), o en `SALIDA=$(...)` si `jaxctl`
devuelve `UNAVAILABLE` (exit 2) -- el operador nunca llega a ver el
`NO-GO: UNAVAILABLE` de abajo, la terminal simplemente termina ahí. Cada
bloque de este paso empieza con `set +e` explícito para no depender de que
nadie se acuerde:

```bash
set +e
systemctl restart jax-las-manos      # o esperar al próximo deploy
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
- `DatabaseObservationMismatchError` → paso 3, `jax_execution` incompleto
  (tabla, índice único o trigger ausente) — leer el mensaje, dice cuál de
  los cuatro.
- `NO-GO: la cuenta admin no muestra CREATE+TRIGGER...` → paso 4/5, la
  cuenta usada para migrar no es admin -- no usar `$JAX_DB_USER` acá; esto
  CORTA el script antes de tocar DDL, no es sólo un aviso.
- el subshell del backup corta (con `set -e`, sin instalar nada) → paso
  4/5, falló el `mysqldump` de verdad (credenciales/permisos/disco) --
  resolver ESO antes de reintentar, no saltarse el backup.
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
