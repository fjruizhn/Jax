# Rutas de producción de jax — cortar del checkout de trabajo

**Decisión de diseño (Hyde, 2026-09-25):** producción no depende del checkout
de trabajo de un agente. Este documento corta las tres claves de
`/etc/jax/.env` que hoy apuntan a `/home/fruiz/jax/...` (el checkout que un
agente tiene abierto, hoy en otra rama) hacia rutas versionadas o de datos
propias de producción, sin cambiar comportamiento observable.

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
config/auditoría/documentos) y que Fernando no tomó todavía. `ops/
rutas-de-produccion.sh --verificar` las excluye explícitamente, con este
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

## Horas a evitar

**No ejecutar entre 04:25 y 04:35** (hora del servidor, CST): `jax-
memory-worker.timer` dispara a las 04:30:43 y `jax-memory-synthesis.timer` a
las 04:30:00 (medido con el listado de timers del sistema, 2026-09-25).
Ninguno de los dos lee las claves que este cambio mueve, pero conviene no
solapar un reinicio de servicios con un timer disparando en la misma
ventana. Elegir además una ventana de tráfico bajo para `jax-platform`
(servicio web con usuarios reales).

## Paso 0 — antes de la ventana (servicios siguiendo corriendo)

Prepara el destino y adelanta la copia grande para que el corte real (con
servicios parados) sea lo más corto posible.

```bash
# 1. El código de JAX_CONFIG_PATH ya está en /srv/jax-prod/jax/config/config.toml
#    -- verificado 2026-09-25, diff vacío contra el checkout de trabajo. No
#    hace falta copiar nada para esta clave.
diff <(sudo cat /srv/jax-prod/jax/config/config.toml) <(cat /home/fruiz/jax/config/config.toml) \
  && echo "IDENTICOS -- nada que copiar para JAX_CONFIG_PATH"

# 2. /srv/jax-data/repo: primera pasada de rsync, SIN --delete (no borra nada
#    del lado destino si ya existiera algo), preservando ACL y xattrs.
sudo mkdir -p /srv/jax-data/repo
sudo rsync -aHAX --numeric-ids /home/fruiz/jax/repo/ /srv/jax-data/repo/

# 3. /var/log/jax/las_manos: se crea ahora (vacío); la copia FINAL del audit
#    log va en la ventana, para no perder eventos escritos entre esta copia
#    y el corte.
sudo mkdir -p /var/log/jax/las_manos
sudo chown jaxsvc:jaxsvc /var/log/jax/las_manos
sudo chmod 0750 /var/log/jax/las_manos
```

## Paso 1 — respaldo de `/etc/jax/.env`

```bash
TS="$(date +%Y%m%d-%H%M%S)"
sudo cp -p /etc/jax/.env "/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}"
sudo chown root:root "/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}"
sudo chmod 600 "/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}"
ls -la "/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}"
```

Verificación por efecto: el `ls -la` muestra `-rw-------  1 root root` y el
mismo tamaño que `/etc/jax/.env` (`sudo stat -c%s /etc/jax/.env`).

## Ventana corta empieza aquí

### Paso 2 — detener los dos servicios

```bash
sudo systemctl stop jax-las-manos jax-platform
systemctl is-active jax-las-manos jax-platform   # esperar "inactive" en las dos
```

### Paso 3 — rsync final (delta) de `JAX_REPO_BASE` + verificación sha256

```bash
sudo rsync -aHAX --numeric-ids --delete /home/fruiz/jax/repo/ /srv/jax-data/repo/

# Verificación por efecto, no por confianza en rsync: sha256 de CADA archivo
# de los dos lados tiene que coincidir.
diff \
  <(cd /home/fruiz/jax/repo && sudo find . -type f -exec sha256sum {} \; | sort) \
  <(cd /srv/jax-data/repo && sudo find . -type f -exec sha256sum {} \; | sort) \
  && echo "REPO_BASE: sha256 identico en todos los archivos"
```

Si el `diff` muestra algo: **no seguir** -- revisar qué archivo difiere antes
de continuar (¿se escribió algo en el origen durante el rsync final, con los
servicios ya parados? no debería, confirmar).

### Paso 4 — copiar el audit log a `/var/log/jax/las_manos/audit.jsonl`

```bash
sudo cp -p /home/fruiz/jax/las_manos/logs/audit.jsonl /var/log/jax/las_manos/audit.jsonl
sudo chown jaxsvc:jaxsvc /var/log/jax/las_manos/audit.jsonl
sudo chmod 0640 /var/log/jax/las_manos/audit.jsonl

sha256_origen="$(sudo sha256sum /home/fruiz/jax/las_manos/logs/audit.jsonl | awk '{print $1}')"
sha256_destino="$(sudo sha256sum /var/log/jax/las_manos/audit.jsonl | awk '{print $1}')"
[ "$sha256_origen" = "$sha256_destino" ] && echo "AUDIT_LOG: sha256 identico" \
  || { echo "AUDIT_LOG: sha256 NO COINCIDE -- no seguir"; exit 1; }
```

El original en `/home/fruiz/jax/las_manos/logs/audit.jsonl` **queda intacto**
como registro -- esto es una copia, no una migración destructiva.

### Paso 5 — editar `/etc/jax/.env` de forma atómica

Cambia únicamente las tres líneas, preservando byte a byte todo lo demás
(credenciales, hosts, el resto de las 90+ claves) y el dueño/modo del
archivo.

```bash
sudo awk -v c="/srv/jax-prod/jax/config/config.toml" \
         -v a="/var/log/jax/las_manos/audit.jsonl" \
         -v r="/srv/jax-data/repo" '
  /^JAX_CONFIG_PATH=/    { print "JAX_CONFIG_PATH=" c;    next }
  /^JAX_AUDIT_LOG_PATH=/ { print "JAX_AUDIT_LOG_PATH=" a; next }
  /^JAX_REPO_BASE=/      { print "JAX_REPO_BASE=" r;      next }
  { print }
' /etc/jax/.env | sudo tee /etc/jax/.env.tmp > /dev/null

sudo chown root:jaxsvc /etc/jax/.env.tmp
sudo chmod 640 /etc/jax/.env.tmp

# Verificación por efecto ANTES de mover: el diff tiene que mostrar
# EXACTAMENTE 3 líneas cambiadas, ninguna otra -- si aparece cualquier otra
# diferencia (una credencial, un host), no seguir. El propio diff nunca
# imprime un valor que no sea una de estas tres rutas (son las únicas
# líneas que cambian), así que no hace falta redactar nada.
diff <(sudo cat /etc/jax/.env) <(sudo cat /etc/jax/.env.tmp)

sudo mv /etc/jax/.env.tmp /etc/jax/.env   # rename atómico, mismo filesystem
sudo stat -c "%U:%G %a" /etc/jax/.env     # confirmar que sigue root:jaxsvc 640
```

### Paso 6 — arrancar los dos servicios

```bash
sudo systemctl start jax-las-manos jax-platform
systemctl is-active jax-las-manos jax-platform   # esperar "active" en las dos
```

### Paso 7 — verificación por efecto

**7.1 — Salud HTTP:**

```bash
curl -s -o /dev/null -w 'LAS MANOS /health -> %{http_code}\n'      http://127.0.0.1:7777/health
curl -s -o /dev/null -w 'jax-platform /api/health -> %{http_code}\n' http://127.0.0.1:8080/api/health
```

Las dos tienen que dar `200`.

**7.2 — `/proc/<PID>/environ` con las rutas nuevas** (son rutas, no
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

**7.3 — Por efecto: la auditoría escribe en la ruta nueva.**

Genera un evento real sin tocar ni imprimir ninguna credencial: la
credencial se lee de `/etc/jax/.env` con `sudo` y se pasa a `curl` dentro de
la MISMA línea, nunca queda en una variable que sobreviva ni se hace
`echo` de ella. `POST /jacobs/preflight` con el cuerpo vacío `{}` es
estructuralmente inválido (`PreflightRequest` exige `invoked_by` y `steps`);
la identidad `plataforma` SÍ puede alcanzar esa ruta (coincide con
`POST /jacobs/.+` en `las_manos/auth_servicio.py`, a diferencia de `/plan` y
`/audit/tail`, que hoy no tienen ningún llamador permitido -- ver
`policy/tests/test_archivos_de_test_wireados_en_ci.py`, EXCEPCIONES). El
rechazo estructural pasa por el `exception_handler(RequestValidationError)`
global de `server.py`, que llama `audit.log_envelope_rejected(...)` --
verificado en vivo el 2026-09-25 (evidencia en el informe de esta tarea):
`HTTP 422`, `{"detail":"ENVELOPE_REJECTED (estructural)", ...}`, y la línea
apareció de inmediato en el audit log con el timestamp de la prueba.

```bash
sudo bash -c '
  set -a; . /etc/jax/.env; set +a
  curl -s -o /dev/null -w "POST /jacobs/preflight -> %{http_code}\n" \
    -X POST http://127.0.0.1:7777/jacobs/preflight \
    -H "X-Jax-Credencial-Servicio: $JAX_LAS_MANOS_CREDENCIAL_PLATAFORMA" \
    -H "Content-Type: application/json" \
    -d "{}"
'
```

Debe dar `422`. Luego confirmar que el evento quedó en la ruta NUEVA (no en
la vieja, que ya no se toca):

```bash
sudo tail -n 3 /var/log/jax/las_manos/audit.jsonl
```

La última línea tiene que ser `"event": "ENVELOPE_REJECTED"` con un
`@timestamp` de hace segundos.

**7.4 — Por efecto: un documento del repo se sirve.**

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

**7.5 — El guion de verificación de esta misma tarea:**

```bash
/srv/jax-prod/jax/ops/rutas-de-produccion.sh --verificar
echo "RC=$?"
```

Tiene que dar `RC=0`. El guion NO menciona `JAX_MISSIONS_DIR` ni `JAX_BIN`
en su salida (las excluye explícitamente, ver la tabla de arriba) -- si
`--verificar` las menciona, algo cambió (¿se retiró la excepción del
guion? ¿se decidió su destino sin actualizar este runbook?) y hay que
revisar antes de dar el corte por cerrado.

## Ventana corta termina aquí

## Reversión

Si cualquier verificación del Paso 7 falla:

```bash
sudo systemctl stop jax-las-manos jax-platform
sudo cp -p "/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}" /etc/jax/.env
sudo chown root:jaxsvc /etc/jax/.env
sudo chmod 640 /etc/jax/.env
sudo systemctl start jax-las-manos jax-platform
curl -s -o /dev/null -w 'LAS MANOS /health -> %{http_code}\n' http://127.0.0.1:7777/health
curl -s -o /dev/null -w 'jax-platform /api/health -> %{http_code}\n' http://127.0.0.1:8080/api/health
```

No hace falta deshacer las copias de `/srv/jax-data/repo` ni
`/var/log/jax/las_manos/audit.jsonl`: son copias nuevas, no tocaron los
originales en `/home/fruiz/jax/...`, y con el `.env` restaurado los
servicios vuelven a leer de ahí exactamente como antes. Quedan servidas para
un segundo intento sin repetir el rsync completo.

## Pendiente (fuera de este runbook)

`JAX_MISSIONS_DIR` y `JAX_BIN` siguen apuntando a
`/home/fruiz/jax/missions` y `/home/fruiz/.local/bin/jax`. Tienen consumidor
real (`jax-platform/backend/api/command.py`) pero moverlos es una decisión
de arquitectura distinta y mayor (qué checkout ejecuta el código de cada
misión, no sólo dónde vive un archivo de config/log/documentos) que
Fernando no tomó todavía -- ver el informe de esta tarea para la propuesta.
