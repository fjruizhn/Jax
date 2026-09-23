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

### 4. Aplicar las dos migraciones si faltan, con backup antes
```bash
set -a; . <(sudo -n cat /etc/jax/.env); set +a

# Backup primero (regla del carpintero) -- aunque los esquemas sean nuevos,
# nunca se asume que están vacíos sin mirar.
B=~/backups/jax_evidence_execution_pre_260_$(date +%F-%H%M).sql
mysql -h$JAX_DB_HOST -P$JAX_DB_PORT -u$JAX_DB_USER -p$JAX_DB_PASSWORD \
  -e "SHOW DATABASES LIKE 'jax_evidence'; SHOW DATABASES LIKE 'jax_execution';"
( umask 077; mysqldump --single-transaction --skip-lock-tables --no-tablespaces \
    --databases jax_evidence jax_execution \
    -h$JAX_DB_HOST -P$JAX_DB_PORT -u$JAX_DB_USER -p$JAX_DB_PASSWORD \
    > "$B" 2>/dev/null || echo "(esquemas aún no existen -- backup vacío, esperado la primera vez)" )
```

**`DELIMITER` es una directiva del cliente `mysql`, no SQL** — el
`mysql` CLI interactivo o alimentado por stdin la entiende nativamente:

```bash
mysql -h$JAX_DB_HOST -P$JAX_DB_PORT -u$JAX_DB_USER -p$JAX_DB_PASSWORD < policy/enforcement_evidence/migrations/001_enforcement_evidence.sql
mysql -h$JAX_DB_HOST -P$JAX_DB_PORT -u$JAX_DB_USER -p$JAX_DB_PASSWORD < policy/execution_control/migrations/001_governed_execution.sql
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
donde se haya cortado antes.

### 5. El usuario de la DB del servicio necesita `INSERT` en `jax_evidence.*`
El arranque escribe la identidad (`__record_identity`) y los artefactos de
evidencia del propio `DatabaseControlInspector` — sin `INSERT`, la primera
escritura del arranque falla. Verificar/otorgar (con un usuario con
privilegio de administración, no con el de la aplicación):

```sql
GRANT INSERT, SELECT ON jax_evidence.* TO 'jaxappuser'@'%';
GRANT SELECT ON jax_execution.* TO 'jaxappuser'@'%';
FLUSH PRIVILEGES;
```
(sustituir `'jaxappuser'@'%'` por el usuario real de `JAX_DB_USER` en
`/etc/jax/.env` — no se cita a mano acá porque es una credencial).

### 6. Generar el manifiesto sobre el commit exacto desplegado
```bash
sudo -u jaxsvc python3 scripts/generar_manifiesto_identidad.py \
    --repo-root /srv/jax-prod/jax --production
```

`--production` fija las rutas de salida a
`/etc/jax/build/implementation-identity.json` y su `.manifest.json` — el
script ya no tiene un default silencioso ahí (ronda PR#264: un `--output`
con default apuntando a la ruta real de despliegue hacía que cualquier
corrida de prueba sin `--output` explícito intentara escribir ahí por
accidente).

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
algún archivo del checkout. Antes de esta ronda, `arbol_sucio()` sólo leía
stdout y ese caso pasaba desapercibido.

### 7. Instalar la identidad — el `chown` es OBLIGATORIO
`/etc/jax` es `root:root 755` → el JSON generado en el paso 6 (si se copia
a mano, o si `sudo -u jaxsvc` no tenía permiso de escritura en
`/etc/jax/build/` y hubo que generar como root y copiar) siempre termina
propiedad de quien lo escribió. El proceso de LAS MANOS corre como
`jaxsvc` (drop-in `cuenta-de-servicio.conf`). Sin `chown jaxsvc:jaxsvc`,
el arranque muere con `PermissionError` al intentar `open()` un archivo
`600` que no le pertenece:

```bash
sudo install -o jaxsvc -g jaxsvc -m 600 /ruta/generada/implementation-identity.json /etc/jax/build/implementation-identity.json
sudo install -o jaxsvc -g jaxsvc -m 600 /ruta/generada/implementation-identity.json.manifest.json /etc/jax/build/implementation-identity.manifest.json
```

Si el paso 6 ya corrió como `jaxsvc` con `/etc/jax/build/` escribible por
ese usuario, este `chown` es un no-op — pero se ejecuta SIEMPRE, no
condicionalmente: no hay forma barata de saber desde este runbook si el
paso anterior ya dejó el dueño correcto, y `chown` sobre el dueño correcto
no tiene efecto secundario.

**Nunca instalar un artefacto `DIRTY`** (`source_state: "DIRTY"` en el
JSON) en esta ruta — ver la advertencia del paso 9.

### 8. Instalar el blob del build manifest, y el símlink `/srv/jax`
La identidad sólo lleva `build_manifest_blob_hash` — un *puntero*.
`verify_build_manifest` lo resuelve con
`store.get_evidence_blob(...)`, y en producción `store` es
`MariaDBEvidenceStore`, leyendo `jax_evidence.evidence_blobs`. Insertar los
bytes generados en el paso 6 (`put_evidence_blob` es idempotente,
direccionado por contenido):

```bash
sudo -u jaxsvc env $(sudo -n cat /etc/jax/.env | xargs) python3 - <<'PY'
import os
from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
import pymysql
def connect():
    return pymysql.connect(host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.environ["JAX_DB_USER"], password=os.environ["JAX_DB_PASSWORD"],
        database=os.environ.get("JAX_DB_NAME", "jax_memory"), charset="utf8mb4", autocommit=False)
data = open("/etc/jax/build/implementation-identity.manifest.json", "rb").read()
MariaDBEvidenceStore(connect).put_evidence_blob(data)
PY
```

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

### 9. Confirmar el arranque, y que el claim no haya quedado STALE
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
(`jaxctl`, `docs/operations/operational-manual.md`):

```bash
set -a; . <(sudo -n cat /etc/jax/.env); set +a
python3 -m jaxctl control CTL.B6.GOVERNED_DISPATCH --version 1 --claim WRITTEN \
    --scope '{"environment":"SANDBOX_RUNTIME"}' --subjects '[]' --json
```

`"verdict":"STALE"` en la salida = el manifiesto instalado no es `CLEAN`,
o describe un commit distinto al que corre. `"verdict":"UNVERIFIABLE"` =
la identidad ni siquiera pasó `is_trusted_implementation_identity` (no
llegó a instalarse por el camino confiable). Ninguno de los dos es GO,
aunque `/health` haya dado 200.

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
- Error de MariaDB por permisos al insertar (`1142` / `command denied`) →
  paso 5, falta el `GRANT INSERT`.
- `git ... fatal: detected dubious ownership in repository at
  '/srv/jax-prod/jax'` → paso 6, no se corrió como `jaxsvc` (o falta
  `safe.directory`).
- `FileNotFoundError` sobre `/etc/jax/build/implementation-identity.json`
  → el paso 7 no se hizo, o se hizo en una ruta distinta a la fija.
- `PermissionError` al abrir ese mismo archivo → paso 7, falta el `chown`
  (o se hizo con el dueño equivocado).
- `EvidenceBlobMissingError` → paso 8, el blob nunca se insertó en
  `jax_evidence.evidence_blobs` (o el símlink `/srv/jax` apunta a otro
  lado y el hash no coincide con lo que hay en disco).
- `UntrustedImplementationIdentityError` ("build manifest inválido o
  drift") → el código cambió después de generar el manifiesto del paso 6:
  regenerar (ver Fail-closed condition, abajo).
- `"verdict":"STALE"` con todo lo anterior en verde → paso 9, se instaló
  (o quedó) un manifiesto `DIRTY`; regenerar sin `--allow-dirty` e
  instalar de nuevo (pasos 6-8).
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
Regenerar (pasos 6-8 de arriba) es el arreglo — no hay actualización
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
