# Rutas de producción de jax — cortar del checkout de trabajo

**Decisión de diseño (Hyde, 2026-09-25):** producción no depende del checkout
de trabajo de un agente. Este documento corta las tres claves de
`/etc/jax/.env` que hoy apuntan a `/home/fruiz/jax/...` (el checkout que un
agente tiene abierto, hoy en otra rama) hacia rutas versionadas o de datos
propias de producción, sin cambiar comportamiento observable.

**Revisión 2 (2026-09-25), tras auditoría de escalón 3 sobre PR jax#277
(APROBADO CON CAMBIOS):** esta versión cierra 5 hallazgos MAYORES sobre la
revisión 1 -- ventana de `.env` mundialmente legible durante la edición
(MAJOR-1), reversión que podía perder eventos de auditoría y archivos
nuevos (MAJOR-2), verificador que fallaba abierto en 6 casos (MAJOR-3, ver
`ops/rutas_de_produccion_verificador.py` y su suite de tests -- corregido
en el propio código, no aquí), documentos que quedaban legibles por
cualquier usuario local tras el rsync (MAJOR-4), y falta de un chequeo de
"nada en vuelo" antes de detener los servicios (MAJOR-5) -- más los
MINOR de horarios reales, aserciones de no-vacío, retirar el paso que
ensuciaba la auditoría forense, y declarar explícitamente lo que queda
fuera de alcance.

**Rama:** `ops/rutas-de-produccion` (jax). **Requisito previo:** esta rama
tiene que estar mergeada a `master` y desplegada en `/srv/jax-prod/jax` antes
de ejecutar este runbook -- el paso final usa
`/srv/jax-prod/jax/ops/rutas-de-produccion.sh --verificar`, que no existe en
producción hasta que el PR se integre. **Solo Fernando da el GO** para
ejecutar esto contra producción (jerarquía de autoridad); este documento no
lo sustituye.

## Qué cambia y qué no

| Clave | Hoy | Pasa a | Consumidores reales (verificado 2026-09-25) |
|---|---|---|---|
| `JAX_CONFIG_PATH` | `/home/fruiz/jax/config/config.toml` | `/srv/jax-prod/jax/config/config.toml` | `jax-platform/backend/api/chat.py` (`CONFIG_PATH = ruta_absoluta_requerida(...)`) |
| `JAX_AUDIT_LOG_PATH` | `/home/fruiz/jax/las_manos/logs/audit.jsonl` | `/var/log/jax/las_manos/audit.jsonl` | `las_manos/server.py` (escribe, `AuditLog`), `jax-platform/backend/api/audit.py` (lee) |
| `JAX_REPO_BASE` | `/home/fruiz/jax/repo` | `/srv/jax-data/repo` | `jacobs/executor.py` dentro de LAS MANOS (escribe `documents/`), `jax-platform/backend/api/admin/repository.py` (lee, `require_superadmin`) |

**JAX_MISSIONS_DIR y JAX_BIN NO se tocan en este runbook.** Tienen consumidor
real (`jax-platform/backend/api/command.py`: `MISSIONS_DIR` y
`JAX_BIN = ruta_absoluta_requerida("JAX_BIN")`, invocado como `$JAX_BIN
--task <mission_file>`), pero el lanzador de `JAX_BIN`
(`/home/fruiz/.local/bin/jax`) hace `cd $HOME/jax` y usa su propio `.venv`:
moverlo decide DESDE QUÉ CHECKOUT corre cada misión (el código que se
ejecuta), una decisión de arquitectura mayor que la de este cambio (rutas de
config/auditoría/documentos) y que Fernando no tomó todavía.
`ops/rutas-de-produccion.sh --verificar` las excluye explícitamente
(`EXCEPCIONES_FASE_A` en `ops/rutas_de_produccion_verificador.py`), con este
mismo motivo escrito. Quedan **PENDIENTES** (ver el informe de esta tarea).

**Servicios a tocar: `jax-las-manos` y `jax-platform` únicamente.**
Verificado con grep sobre `/srv/jax-prod/jax` y `/srv/jax-prod/jax-platform`
completos (no sólo `*.py`: también `*.sh`, `*.toml`, `*.service`): ningún
otro de los cinco servicios con `EnvironmentFile=/etc/jax/.env`
(`jax-ejecutor-proxy`, `jax-memory-worker`, `jax-memory-synthesis`) lee
`JAX_CONFIG_PATH`, `JAX_AUDIT_LOG_PATH` ni `JAX_REPO_BASE` en ningún punto de
su código. Tienen la variable en su entorno (la carga `EnvironmentFile`
entera), pero no la usan: no hace falta reiniciarlos para que esto surta
efecto, y no se tocan en este runbook.

## Explícitamente FUERA DE ALCANCE de este runbook

Declarado, no resuelto -- para que nadie lo confunda con "no se vio":

- **`JAX_MISSIONS_DIR` / `JAX_BIN`** -- ver arriba. Pendiente, decisión de
  Fernando.
- **Los 4 defaults silenciosos de `JAX_WORKSPACE_DIR`** (verificado
  2026-09-25, `grep -rn 'JAX_WORKSPACE_DIR' --include=*.py | grep getenv`):
  `procesamiento/extractores/ocr.py:76`, `jacobs/executor.py:123`,
  `las_manos/motor_registry/tool_authority.py:64` y
  `jax/muscles/subprocess_muscle.py:55` -- los cuatro caen a
  `/home/fruiz/jax-workspace` si falta la variable, en vez de fallar visible
  (`EntornoInvalido`). `JAX_WORKSPACE_DIR` SÍ está puesta hoy en
  `/etc/jax/.env` (se queda ahí, es una decisión de diseño, ver tabla de
  arriba), así que ninguno de los cuatro cae al default en producción hoy
  -- pero el default sigue vivo en el código, sin test que lo cierre. Fuera
  de alcance de esta tarea (que es sobre `JAX_CONFIG_PATH`/
  `JAX_AUDIT_LOG_PATH`/`JAX_REPO_BASE`, no sobre `JAX_WORKSPACE_DIR`).
- **`/var/log/jax` sin logrotate.** Este runbook crea
  `/var/log/jax/las_manos/audit.jsonl` y lo deja creciendo sin límite
  (`AuditLog` sólo hace `open(..., "a")`, nunca rota ni trunca). El archivo
  de trabajo viejo tampoco tenía logrotate -- no es una regresión, pero
  tampoco se resuelve acá. Pendiente aparte.
- **`fruiz` pierde lectura directa (sin `sudo`) del audit log nuevo.**
  Verificado 2026-09-25: `fruiz` NO pertenece al grupo `jaxsvc` (`id fruiz`
  no lo lista), y `/var/log/jax/las_manos` queda `jaxsvc:jaxsvc 0750` -- ni
  siquiera el grupo ayuda porque el grupo es el mismo dueño. Hoy, con el
  archivo en el checkout de trabajo (`fruiz:jaxsvc 0664`, directorio
  `fruiz:jaxsvc 02775`), `fruiz` lee sin `sudo`. Después de este corte,
  hace falta `sudo` para leer el audit log -- documentado, no arreglado
  (arreglarlo es decidir si `fruiz` entra al grupo `jaxsvc` o si se relaja
  el modo, y esa es una decisión de superficie de acceso que no le
  corresponde a este runbook tomar sola).

## Horas a evitar

Horarios REALES medidos el 2026-09-25 (`sudo systemctl cat <unidad>.timer` +
`systemctl list-timers`), no aproximados:

| Timer | Disparo | Mecanismo |
|---|---|---|
| `jax-memory-worker.timer` | cada ~20 min, observado en :10/:30/:50 (con segundos que derivan) | `OnUnitActiveSec=20min` -- relativo a la última corrida, NO horario fijo; el minuto exacto se corre con el tiempo |
| `jax-memory-synthesis.timer` | 04:30:00 en punto, todos los días | `OnCalendar=*-*-* 04:30:00`, sin `RandomizedDelaySec` |
| `jax-limpiar-bases-de-test.timer` | ~00:00-00:15 (observado 00:02:28) | `OnCalendar=daily` + `RandomizedDelaySec=15m` |
| `jax-revisar-indice-vectorial.timer` | lunes ~06:30-06:50 (observado 06:41:35) | `OnCalendar=Mon *-*-* 06:30:00` + `RandomizedDelaySec=20m` |

Ninguno de los cuatro lee las tres claves que este cambio mueve, así que no
hay una razón de CORRECTITUD para evitarlos -- es prudencia: no solapar un
reinicio de servicios con un timer disparando en la misma ventana.

**Ventana recomendada, cada hora:** minuto **12 a 28** de cada bloque de 20
(es decir `HH:12`–`HH:28`, `HH:32`–`HH:48`, `HH:52`–`HH:08` del siguiente
bloque) -- dos minutos de margen a cada lado de los disparos observados de
`jax-memory-worker` en :10/:30/:50. Evitar además **04:25–04:35** (la
sintesis, horario fijo, sin margen aleatorio) y, si el corte cae un lunes,
**06:25–06:55**. Elegir además una ventana de tráfico bajo para
`jax-platform` (servicio web con usuarios reales).

## Paso 0 — antes de la ventana (servicios siguiendo corriendo)

Prepara el destino, ancla el timestamp de esta corrida para que sobreviva a
un corte de sesión SSH, y adelanta la copia grande.

```bash
set -euo pipefail

# El TS se persiste en un archivo -- si la sesión de quien ejecuta se corta
# a mitad de camino, la reversión (más abajo) lo puede releer en vez de
# adivinar cuál backup es el bueno.
date +%Y%m%d-%H%M%S > ~/rutas-prod.TS
TS="$(cat ~/rutas-prod.TS)"
echo "TS=$TS"

# 1. El código de JAX_CONFIG_PATH ya está en /srv/jax-prod/jax/config/config.toml
#    -- verificado 2026-09-25, diff vacío contra el checkout de trabajo. No
#    hace falta copiar nada para esta clave.
diff <(sudo cat /srv/jax-prod/jax/config/config.toml) <(cat /home/fruiz/jax/config/config.toml) \
  && echo "IDENTICOS -- nada que copiar para JAX_CONFIG_PATH"

# 2. /srv/jax-data/repo: primera pasada de rsync, SIN --delete (no borra nada
#    del lado destino si ya existiera algo), preservando ACL y xattrs. Los
#    permisos "other" del origen (ver MAJOR-4 más abajo) se cierran DESPUÉS
#    del rsync FINAL (Paso 4), no acá -- esta pasada es sólo para achicar el
#    delta de la ventana corta.
sudo mkdir -p /srv/jax-data/repo
sudo rsync -aHAX --numeric-ids /home/fruiz/jax/repo/ /srv/jax-data/repo/

# 3. /var/log/jax/las_manos: se crea ahora (vacío); la copia FINAL del audit
#    log va en la ventana, para no perder eventos escritos entre esta copia
#    y el corte.
sudo mkdir -p /var/log/jax/las_manos
sudo chown jaxsvc:jaxsvc /var/log/jax/las_manos
sudo chmod 0750 /var/log/jax/las_manos
```

**N0 NO se captura acá.** Con los servicios todavía corriendo, cualquier
tráfico real puede seguir escribiendo en el audit log de trabajo entre este
paso y el Paso 3 (detener) -- si N0 se anclara ahora, esas líneas
"de en medio" quedarían contadas de más y R2 (reversión) las duplicaría al
anexarlas de vuelta. N0 se captura recién en el Paso 5, con los servicios
YA detenidos (Paso 3) y el archivo de trabajo por lo tanto quieto -- ver ahí.

## Paso 1 — respaldo de `/etc/jax/.env`

```bash
sudo cp -p /etc/jax/.env "/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}"
sudo chown root:root "/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}"
sudo chmod 600 "/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}"
ls -la "/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}"
```

Verificación por efecto: el `ls -la` muestra `-rw-------  1 root root` y el
mismo tamaño que `/etc/jax/.env` (`sudo stat -c%s /etc/jax/.env`).

## Ventana corta empieza aquí

### Paso 2 — drenar ANTES de detener (MAJOR-5)

**No se detiene nada con trabajo en vuelo.** Dos chequeos, los dos tienen
que dar limpio:

```bash
set -euo pipefail

# (a) Ningún pipeline pendiente/corriendo/interrumpido -- lectura SELECT
#     pura contra la MariaDB real de producción, puerto 3308 (el 3306 está
#     muerto). Valores de status verificados en el código
#     (jacobs/continuar.py, jacobs/cupo.py, jacobs/_cupo_io_test.py):
#     'pending', 'running', 'interrupted' son literales reales, no
#     inventados. Se sourcea el .env con `sudo cat` vía sustitución de
#     proceso (fruiz no pertenece al grupo jaxsvc y no puede leerlo directo)
#     y mysql corre COMO fruiz, sin sudo -- sudo sólo hacía falta para leer
#     el archivo, no para hablar con la base.
set -a; . <(sudo cat /etc/jax/.env); set +a
EN_VUELO="$(mysql -h 127.0.0.1 -P 3308 -u"$JAX_DB_USER" -p"$JAX_DB_PASSWORD" -N "$JAX_DB_NAME" <<'SQL'
SELECT COUNT(*) FROM jacobs_pipelines WHERE status IN ('pending','running','interrupted');
SQL
)"
echo "pipelines en vuelo: $EN_VUELO"
[ "$EN_VUELO" -eq 0 ] || { echo "ABORTAR: hay $EN_VUELO pipeline(s) en vuelo -- no se detiene nada"; exit 1; }

# (b) Ningún proceso de misión corriendo bajo la cuenta de servicio --
#     command.py lanza literalmente "$JAX_BIN --task <mission_file>".
PROCESOS="$(pgrep -u jaxsvc -f -- '--task' || true)"
[ -z "$PROCESOS" ] || { echo "ABORTAR: hay procesos --task corriendo: $PROCESOS"; exit 1; }

echo "drenado OK: nada en vuelo, se puede detener"
```

Si cualquiera de los dos aborta: **no seguir**. Esperar a que drene solo, o
investigar por qué no drena, antes de reintentar este paso.

### Paso 3 — detener los dos servicios

```bash
sudo systemctl stop jax-las-manos jax-platform
systemctl is-active jax-las-manos jax-platform   # esperar "inactive" en las dos
```

### Paso 4 — rsync final (delta) de `JAX_REPO_BASE` + verificación sha256 + cerrar el acceso `other` (MAJOR-4)

```bash
set -euo pipefail

sudo rsync -aHAX --numeric-ids --delete /home/fruiz/jax/repo/ /srv/jax-data/repo/

# Verificación por efecto, no por confianza en rsync: sha256 de CADA archivo
# de los dos lados tiene que coincidir. Se exige al menos 1 archivo
# comparado -- un árbol vacío no es un "todo coincide", es "no se comparó
# nada" (Principio I: no dar verde por vacío).
ORIGEN_SHA="$(cd /home/fruiz/jax/repo && sudo find . -type f -exec sha256sum {} \; | sort)"
DESTINO_SHA="$(cd /srv/jax-data/repo && sudo find . -type f -exec sha256sum {} \; | sort)"
N_ARCHIVOS="$(printf '%s\n' "$ORIGEN_SHA" | grep -c .)"
[ "$N_ARCHIVOS" -gt 0 ] || { echo "ABORTAR: 0 archivos comparados -- el arbol esta vacio o el find fallo en silencio"; exit 1; }
diff <(printf '%s\n' "$ORIGEN_SHA") <(printf '%s\n' "$DESTINO_SHA") \
  && echo "REPO_BASE: sha256 identico en los $N_ARCHIVOS archivos"

# MAJOR-4: /home/fruiz es 750 (fruiz:fruiz), así que los archivos de origen
# con "other::r--" nunca fueron alcanzables por otro usuario -- la puerta de
# atrás estaba cerrada por el PADRE, no por el propio archivo. /srv/jax-data
# es 775 (fruiz:jaxsvc): CUALQUIER usuario local del servidor puede
# atravesarlo. Sin este paso, el rsync (que preserva permisos del origen tal
# cual) deja todos los documentos legibles por cualquiera con shell en
# hall9000. Se cierra "other" explícitamente, recursivo, con ACL default
# para que los archivos NUEVOS que jacobs/executor.py escriba después
# también nazcan sin acceso de "other" (los archivos ya tenían ACL --
# verificado con getfacl -- así que el bit "other" clásico no basta solo).
sudo chmod -R o-rwx /srv/jax-data/repo
sudo setfacl -R -m o::--- -m d:o::--- /srv/jax-data/repo

# Verificación por efecto: CERO archivos con "other" con cualquier permiso.
SOBRANTES="$(sudo find /srv/jax-data/repo \( -perm -o=r -o -perm -o=w -o -perm -o=x \) 2>/dev/null | wc -l)"
[ "$SOBRANTES" -eq 0 ] || { echo "ABORTAR: $SOBRANTES archivo(s) siguen con permiso 'other'"; exit 1; }
echo "MAJOR-4 cerrado: 0 archivos con acceso 'other' bajo /srv/jax-data/repo"
```

Si el `diff` de sha256 muestra algo, o `SOBRANTES` no da 0: **no seguir** --
revisar antes de continuar.

### Paso 5 — copiar el audit log a `/var/log/jax/las_manos/audit.jsonl`

```bash
set -euo pipefail

# N0 se ancla ACÁ, no en el Paso 0: los servicios ya están detenidos (Paso
# 3), así que el archivo de trabajo está quieto -- nada puede escribirle una
# línea más entre este `wc -l` y el `cp -p` de abajo. Si N0 se hubiera
# anclado antes de detener los servicios, cualquier tráfico real entre ese
# momento y el corte habría quedado contado de más, y R2 (reversión) habría
# duplicado esas líneas al anexarlas de vuelta.
wc -l < /home/fruiz/jax/las_manos/logs/audit.jsonl | tee ~/rutas-prod.N0

sudo cp -p /home/fruiz/jax/las_manos/logs/audit.jsonl /var/log/jax/las_manos/audit.jsonl
sudo chown jaxsvc:jaxsvc /var/log/jax/las_manos/audit.jsonl
sudo chmod 0640 /var/log/jax/las_manos/audit.jsonl

sha256_origen="$(sudo sha256sum /home/fruiz/jax/las_manos/logs/audit.jsonl | awk '{print $1}')"
sha256_destino="$(sudo sha256sum /var/log/jax/las_manos/audit.jsonl | awk '{print $1}')"
[ "$sha256_origen" = "$sha256_destino" ] && echo "AUDIT_LOG: sha256 identico" \
  || { echo "AUDIT_LOG: sha256 NO COINCIDE -- no seguir"; exit 1; }
```

El original en `/home/fruiz/jax/las_manos/logs/audit.jsonl` **queda intacto**
por ahora (se congela recién en el Paso 9, una vez confirmado el corte) --
esto es una copia, no una migración destructiva.

### Paso 6 — editar `/etc/jax/.env` de forma atómica (MAJOR-1)

Cambia únicamente las tres líneas, preservando byte a byte todo lo demás
(credenciales, hosts, el resto de las 90+ claves) y el dueño/modo del
archivo. **El archivo temporal nace con permisos restrictivos desde el
primer byte** (`umask 077` dentro del mismo `sh -c` que lo crea, ANTES del
`awk` -- no un `chmod` después, que dejaría una ventana real, aunque
corta, con el `.env` completo -- credenciales incluidas -- en un archivo
`root:root 644`, legible por cualquiera):

```bash
set -euo pipefail

sudo sh -c '
  umask 077
  awk -v c="/srv/jax-prod/jax/config/config.toml" \
      -v a="/var/log/jax/las_manos/audit.jsonl" \
      -v r="/srv/jax-data/repo" "
    /^JAX_CONFIG_PATH=/    { print \"JAX_CONFIG_PATH=\" c;    next }
    /^JAX_AUDIT_LOG_PATH=/ { print \"JAX_AUDIT_LOG_PATH=\" a; next }
    /^JAX_REPO_BASE=/      { print \"JAX_REPO_BASE=\" r;      next }
    { print }
  " /etc/jax/.env > /etc/jax/.env.tmp
'
sudo chown root:jaxsvc /etc/jax/.env.tmp
sudo chmod 640 /etc/jax/.env.tmp
stat -c "%U:%G %a" /etc/jax/.env.tmp   # confirmar root:jaxsvc 640 ANTES de mover, no después

# Verificación MECÁNICA antes de mover -- no "mirar y parecer bien": el
# diff en formato unificado (-U0, sin contexto) tiene que mostrar
# EXACTAMENTE 3 pares de líneas cambiadas (una "-" y una "+" por cada
# clave, 6 líneas de contenido en total) más las 2 líneas de cabecera
# "---"/"+++" que agrega -U0 -- si aparece una sola línea de más o de
# menos, algo tocó una credencial o un host, y NO se sigue.
DIFF_TMP="$(diff -U0 <(sudo cat /etc/jax/.env) <(sudo cat /etc/jax/.env.tmp) || true)"
echo "$DIFF_TMP"
N_LINEAS_DE_CAMBIO="$(printf '%s\n' "$DIFF_TMP" | grep -cE '^[+-][^+-]')"
[ "$N_LINEAS_DE_CAMBIO" -eq 6 ] || {
  echo "ABORTAR: el diff tiene $N_LINEAS_DE_CAMBIO líneas de cambio, se esperaban exactamente 6 (3 claves x 2 líneas -/+). Revisar antes de mover -- NO se mueve el .tmp."
  sudo rm -f /etc/jax/.env.tmp
  exit 1
}
echo "diff OK: exactamente 3 claves cambiadas, nada más"

sudo mv /etc/jax/.env.tmp /etc/jax/.env   # rename atómico, mismo filesystem
sudo stat -c "%U:%G %a" /etc/jax/.env     # confirmar que sigue root:jaxsvc 640
```

El propio `diff` de arriba nunca imprime un valor que no sea una de estas
tres rutas (son las únicas líneas que cambian, y la aserción de 6 líneas lo
garantiza mecánicamente); no hace falta redactar nada.

### Paso 7 — arrancar los dos servicios

```bash
sudo systemctl start jax-las-manos jax-platform
systemctl is-active jax-las-manos jax-platform   # esperar "active" en las dos
```

### Paso 8 — verificación por efecto

**8.1 — Salud HTTP:**

```bash
curl -s -o /dev/null -w 'LAS MANOS /health -> %{http_code}\n'      http://127.0.0.1:7777/health
curl -s -o /dev/null -w 'jax-platform /api/health -> %{http_code}\n' http://127.0.0.1:8080/api/health
```

Las dos tienen que dar `200`. **`/health` de LAS MANOS ya prueba escritura
del audit log** (`comprobar_audit(audit.log_path)` en `las_manos/salud.py`)
-- por eso este runbook YA NO incluye un paso separado que dispare un
`ENVELOPE_REJECTED` sintético contra `/jacobs/preflight` (estaba en la
revisión 1 de este documento): ensuciaba la auditoría FORENSE con eventos
de prueba que no son un incidente real, y `/health` cubre la misma
pregunta ("¿se puede escribir en la ruta nueva?") sin escribir un evento de
mentira en un log de seguridad.

> **Nota para quien audite el log más adelante:** SÍ existe un evento
> `ENVELOPE_REJECTED` sintético, de una prueba manual de esta tarea (no de
> este runbook), con `@timestamp` **2026-09-25T10:49:56.746832+00:00**,
> `reason: "campos faltantes/mal-tipados: ['invoked_by', 'steps']"` --
> generado a propósito contra `/jacobs/preflight` con cuerpo `{}` para
> confirmar que el mecanismo de auditoría funciona, antes de escribir este
> runbook. No es un incidente. Vive en la copia de trabajo
> (`/home/fruiz/jax/las_manos/logs/audit.jsonl`), que este corte todavía no
> toca en el Paso 5 (sólo copia, no borra) -- por eso aparece también en
> `/var/log/jax/las_manos/audit.jsonl` tras la copia.

**8.2 — `/proc/<PID>/environ` con las rutas nuevas** (son rutas, no
secretos -- se imprimen sin problema):

```bash
PID_MANOS="$(systemctl show -p MainPID --value jax-las-manos)"
PID_PLATFORM="$(systemctl show -p MainPID --value jax-platform)"
for pid in "$PID_MANOS" "$PID_PLATFORM"; do
  echo "--- PID $pid ---"
  sudo cat "/proc/$pid/environ" | tr '\0' '\n' \
    | grep -E '^JAX_(CONFIG_PATH|AUDIT_LOG_PATH|REPO_BASE)='
done
```

Las dos salidas tienen que mostrar `/srv/jax-prod/jax/config/config.toml`,
`/var/log/jax/las_manos/audit.jsonl` y `/srv/jax-data/repo` -- **nunca**
`/home/fruiz/jax/...`.

**8.3 — Por efecto: un documento del repo se sirve.**

`GET /api/admin/repo/file` exige `require_superadmin` (JWT de sesión, no la
credencial de servicio) -- usar un token de una sesión de superadmin ya
autenticada (la misma que abre el panel de administración; este runbook no
genera ni imprime tokens ni contraseñas):

```bash
curl -s -o /dev/null -w 'GET /api/admin/repo -> %{http_code}\n' \
  -H "Authorization: Bearer $JWT_SUPERADMIN" \
  http://127.0.0.1:8080/api/admin/repo

# Con un archivo conocido que ya existía antes de la migración, p.ej.:
curl -s -o /dev/null -w 'GET /api/admin/repo/file -> %{http_code}\n' \
  -H "Authorization: Bearer $JWT_SUPERADMIN" \
  "http://127.0.0.1:8080/api/admin/repo/file?path=documents/04e02b09_00_hipatia.md"
```

Las dos tienen que dar `200`. Alternativa equivalente sin variable de
entorno: abrir el panel de administración de documentos en el navegador con
una sesión de Fernando ya iniciada y confirmar que el mismo archivo se lista
y se abre.

**8.4 — El guion de verificación de esta misma tarea:**

```bash
/srv/jax-prod/jax/ops/rutas-de-produccion.sh --verificar
echo "RC=$?"
```

Tiene que dar `RC=0`. El guion NO menciona `JAX_MISSIONS_DIR` ni `JAX_BIN`
en su salida (las excluye explícitamente, ver la tabla de arriba) -- si
`--verificar` las menciona, algo cambió (¿se retiró la excepción del
módulo? ¿se decidió su destino sin actualizar este runbook?) y hay que
revisar antes de dar el corte por cerrado.

### Paso 9 — congelar las copias viejas del audit log

**Sólo después de que el Paso 8 completo dio verde.** Las dos copias que ya
no son la fuente de verdad quedan como registro histórico, protegidas
contra una escritura accidental futura (un script viejo que todavía
apuntara a la ruta vieja, un `cd $HOME/jax` de costumbre):

```bash
sudo chattr +i /home/fruiz/jax/las_manos/logs/audit.jsonl
sudo chattr +i /srv/jax-prod/jax/las_manos/logs/audit.jsonl   # la copia de despliegue, si existe
lsattr /home/fruiz/jax/las_manos/logs/audit.jsonl /srv/jax-prod/jax/las_manos/logs/audit.jsonl 2>/dev/null
```

Si `chattr +i` no está disponible en el filesystem (algunos `overlayfs`/red
no lo soportan), usar `chmod a-w` sobre las dos rutas como alternativa más
débil (bloquea escritura normal, no bloquea `root` con `chattr` de por
medio) y anotarlo en el registro de esta tarea.

## Ventana corta termina aquí

## Reversión

Si cualquier verificación del Paso 8 falla, o algo se ve mal antes de
llegar al Paso 9 (que es irreversible: `chattr +i` no se deshace solo con
un restore de `.env`):

```bash
set -euo pipefail
TS="$(cat ~/rutas-prod.TS)"          # el mismo TS del Paso 0, releído del archivo
N0="$(cat ~/rutas-prod.N0)"
BACKUP="/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}"
[ -f "$BACKUP" ] || { echo "ABORTAR REVERSION: no existe $BACKUP -- no hay de donde restaurar"; exit 1; }
```

**R1 — detener:**

```bash
sudo systemctl stop jax-las-manos jax-platform
```

**R2 — anexar al audit log VIEJO lo que se haya escrito en el NUEVO
mientras estuvo activo, sin perder ni duplicar eventos:**

```bash
# Todo lo que el audit log NUEVO tiene después de la línea N0 (que es
# exactamente lo que el original de trabajo NO tenía cuando se copió en el
# Paso 5) se anexa al original -- así el registro completo queda en la ruta
# vieja, que vuelve a ser la fuente de verdad.
sudo tail -n +"$((N0 + 1))" /var/log/jax/las_manos/audit.jsonl \
  | sudo tee -a /home/fruiz/jax/las_manos/logs/audit.jsonl > /dev/null

# Verificación por efecto: el archivo viejo, desde la línea N0+1 en
# adelante, tiene que ser byte a byte lo mismo que el nuevo completo.
cmp <(sudo tail -n +"$((N0 + 1))" /home/fruiz/jax/las_manos/logs/audit.jsonl) \
    <(sudo cat /var/log/jax/las_manos/audit.jsonl) \
  && echo "R2 OK: el audit log viejo tiene todo lo que escribió el nuevo"
```

**R3 — rsync de `documents/` de vuelta, SIN `--delete`** (cualquier archivo
NUEVO que se haya creado en `/srv/jax-data/repo` mientras estuvo activo se
suma al origen; nada se borra del origen aunque falte en el destino nuevo):

```bash
sudo rsync -aHAX --numeric-ids /srv/jax-data/repo/ /home/fruiz/jax/repo/
```

**R4 — restaurar `.env`, con la misma disciplina de permisos del Paso 6:**

```bash
sudo sh -c "umask 077; cp -p '$BACKUP' /etc/jax/.env.restaurado"
sudo chown root:jaxsvc /etc/jax/.env.restaurado
sudo chmod 640 /etc/jax/.env.restaurado
sudo mv /etc/jax/.env.restaurado /etc/jax/.env
sudo stat -c "%U:%G %a" /etc/jax/.env
```

**R5 — arrancar y verificar:**

```bash
sudo systemctl start jax-las-manos jax-platform
curl -s -o /dev/null -w 'LAS MANOS /health -> %{http_code}\n' http://127.0.0.1:7777/health
curl -s -o /dev/null -w 'jax-platform /api/health -> %{http_code}\n' http://127.0.0.1:8080/api/health

PID_MANOS="$(systemctl show -p MainPID --value jax-las-manos)"
PID_PLATFORM="$(systemctl show -p MainPID --value jax-platform)"
for pid in "$PID_MANOS" "$PID_PLATFORM"; do
  sudo cat "/proc/$pid/environ" | tr '\0' '\n' \
    | grep -E '^JAX_(CONFIG_PATH|AUDIT_LOG_PATH|REPO_BASE)='
done
```

Las rutas tienen que volver a mostrar `/home/fruiz/jax/...` -- confirma que
la reversión surtió efecto de verdad, no sólo que el `.env` cambió.

No hace falta deshacer las copias de `/srv/jax-data/repo` ni
`/var/log/jax/las_manos/audit.jsonl` más allá de R2/R3: con el `.env`
restaurado los servicios vuelven a leer de `/home/fruiz/jax/...` exactamente
como antes, y las copias nuevas (ya sincronizadas de vuelta) quedan servidas
para un segundo intento sin repetir el rsync completo.
