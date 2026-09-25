# Rutas de producción de jax — cortar del checkout de trabajo

**Decisión de diseño (Hyde, 2026-09-25):** producción no depende del checkout
de trabajo de un agente. Este documento corta las tres claves de
`/etc/jax/.env` que hoy apuntan a `/home/fruiz/jax/...` (el checkout que un
agente tiene abierto, hoy en otra rama) hacia rutas versionadas o de datos
propias de producción, sin cambiar comportamiento observable.

**Revisión 3 (2026-09-25), ronda 2 de auditoría de escalón 3 sobre PR
jax#277 (APROBADO CON CAMBIOS):** cierra 3 hallazgos MAYORES sobre la
revisión 2 -- el parser seguía fallando abierto ante líneas que systemd SÍ
aplica (MAJOR-A, corregido en `ops/rutas_de_produccion_verificador.py`, no
en este documento), la verificación de la reversión R2 nunca daba igual y
podía duplicar eventos (MAJOR-B), y el prerrequisito de desplegar esta rama
en `/srv/jax-prod/jax` desplegaría B9 sin sus tablas (MAJOR-C) -- más los
MINOR de aborts reales, variables releídas de archivo (no de memoria de
shell), reversión adaptativa según hasta dónde llegó el corte, sin secretos
en `argv` ni en el entorno de la shell interactiva, bloques en subshells
para no cerrar la sesión SSH de quien los corre, y sin `chattr +i` sobre un
archivo dentro de un checkout que `git clean` necesita poder tocar.

**Rama:** `ops/rutas-de-produccion` (jax). **Solo Fernando da el GO** para
ejecutar esto contra producción (jerarquía de autoridad); este documento no
lo sustituye.

## EJECUTADO 2026-09-25 06:12-06:13 (Hyde)

**El corte ya se hizo.** Pasos 2-9 de este runbook, commit `c42896b`
(código en `/etc/jax/.env`, no un despliegue -- ver MAJOR-C: se cortó
contra `bd10297`, ya desplegado). Resultado, evidencia literal:

- `jax-las-manos` y `jax-platform`: `active` los dos.
- `/health` (LAS MANOS) y `/api/health` (jax-platform): `200`/`200`.
- `/proc/<MainPID>/environ` de los DOS servicios: las tres rutas nuevas
  presentes (`/srv/jax-prod/jax/config/config.toml`,
  `/var/log/jax/las_manos/audit.jsonl`, `/srv/jax-data/repo`).
- `JAX_REPO_BASE`: 64 archivos, sha256 idéntico origen/destino.
- `JAX_AUDIT_LOG_PATH`: 1501 líneas, sha256 idéntico.
- `chattr +i` puesto en la copia vieja de trabajo (Paso 9).
- 0 errores en el journal de los dos servicios tras el arranque.

**Ronda 3 de la auditoría de escalón 3 (ejecutada DESPUÉS del corte, sobre
el resultado real): APROBADO, 6 MINOR.** Uno de ellos, encontrado
ejecutando el Paso 8.4 de verdad contra esta producción ya cortada, era un
**DEFECTO real en `ops/rutas_de_produccion_verificador.py`**, no en el
corte en sí: `jaxsvc_puede_leer`/`jaxsvc_puede_escribir` armaban `sudo -n -u
jaxsvc test -r -- <ruta>` -- `test` (a diferencia de `realpath`) NO soporta
`--` como fin de opciones; con 3 argumentos da SIEMPRE `returncode=2` (error
de uso), nunca una respuesta real. Fase B reportaba "jaxsvc no puede
LEER/ESCRIBIR" para las tres claves en alcance INCONDICIONALMENTE, aunque
el corte estuviera perfecto -- un falso negativo del verificador, no un
problema de producción. Confirmado en vivo:

```
$ sudo -u jaxsvc test -r -- /srv/jax-prod/jax/config/config.toml; echo $?
2
$ sudo -u jaxsvc test -r /srv/jax-prod/jax/config/config.toml; echo $?
0
```

Corregido (quitar `--`, exigir ruta absoluta como defensa en su lugar) y
re-verificado contra la producción YA CORTADA:

```
$ ./ops/rutas-de-produccion.sh --verificar
rutas-de-produccion: todas las rutas de producción en alcance están fuera
de /home, jaxsvc tiene el acceso que necesita, y el entorno VIVO de 2
servicio(s) coincide (7 claves de ruta revisadas)
$ echo $?
0
```

Los otros 5 MINOR de la ronda 3 (Fase C no falla ante 0 servicios
revisados, guardia previa en R2, marca de Paso 4 completo para R3, motivo
corregido de Paso 9, JWT fuera de `argv` en 8.3, `-c safe.directory` y
`sudo cmp` en el propio texto del runbook) están aplicados en las
secciones correspondientes de este documento, más abajo -- el documento
completo queda actualizado para la PRÓXIMA vez que haga falta (reversión,
u otro corte de esta naturaleza), no sólo como bitácora de lo ya hecho.

**Ronda 4, MAJOR-1(a):** la marca `~/rutas-prod.paso4-completo` (que R3 de
la Reversión exige, ver más abajo) no existía en el host real, porque se
agregó a este runbook DESPUÉS del corte -- Paso 4 nunca la escribió
retroactivamente. **Hyde la creó a mano en hall9000** con el texto
`20260925-061300 retroactiva: Paso 4 verificado 64/64 sha256 (Hyde, creada
2026-09-25 06:30)` -- documentado acá para que quede constancia de que esa
marca no salió de una ejecución real del Paso 4, sino de una verificación
posterior (el sha256 de 64/64 archivos, ya reportado arriba) traducida a la
misma marca que el runbook espera.

## Pendiente de esta ronda — reparación de ACLs (MINOR-3, MINOR-4)

**PROPUESTO, NO EJECUTADO por esta sesión** -- para que lo corra Fernando o
quien tenga el GO.

### Causa raíz (determinada, ver el informe de esta ronda para la evidencia completa)

El corte perdió las ACL con nombre (`user:jaxsvc:...`, 67-72 entradas según
cómo se cuenten) que `documents/` y su contenido tenían en el checkout de
trabajo. **No es el `rsync -aHAX` de Paso 4 en sí, ni el `setfacl` de
MAJOR-4** (los dos preservan ACL correctamente, probado en aislamiento,
incluso contra un XFS real como `/srv/jax-data`) -- es el **diseño de DOS
pasadas** que este runbook tenía hasta la ronda 3 (Paso 0 "de adelanto" +
Paso 4 "final"): cuando la segunda pasada de `rsync -A` encuentra un
archivo que ya dejó igual la primera, lo salta -- y ese salto BORRA la
entrada de ACL con nombre en vez de dejarla o reaplicarla. Reproducido de
forma determinística, con `--itemize-changes` mostrando la bandera de
cambio de ACL/xattr exactamente en los archivos salteados. `--checksum` en
la segunda pasada NO lo evita. La ronda 4 ya sacó la pasada "de adelanto"
del Paso 0 (ver ahí) para que esto no vuelva a pasar en el PRÓXIMO corte --
esta sección es sólo sobre reparar el que YA ocurrió.

### Reparación propuesta

```bash
# 1. Restaurar las ACLs perdidas: --ignore-times fuerza a rsync a
#    reexaminar CADA archivo de verdad (no confiar en "ya está igual") --
#    es lo que, probado, SÍ reaplica el ACL correctamente. Sin --delete,
#    para no tocar nada escrito en /srv/jax-data/repo desde el corte.
sudo rsync -aHAX --numeric-ids --ignore-times /home/fruiz/jax/repo/ /srv/jax-data/repo/

# 2. Este rsync acaba de traer de vuelta el "other::r--" original --
#    reaplicar el endurecimiento de MAJOR-4 (Paso 4) que eso deshizo:
sudo chmod -R o-rwx /srv/jax-data/repo
sudo setfacl -R -m o::--- -m d:o::--- /srv/jax-data/repo
SOBRANTES="$(sudo find /srv/jax-data/repo \( -perm -o=r -o -perm -o=w -o -perm -o=x \) 2>/dev/null | wc -l)"
[ "$SOBRANTES" -eq 0 ] && echo "MAJOR-4 OK" || echo "ATENCION: $SOBRANTES con acceso 'other'"

# 3. Verificación "jaxsvc lee todo lo que leía antes":
sudo find /srv/jax-data/repo -type f | while IFS= read -r f; do
  sudo -u jaxsvc test -r "$f" || echo "NO LEGIBLE: $f"
done
echo "verificación completa -- sin líneas 'NO LEGIBLE' arriba, jaxsvc lee todo"
```

### MINOR-4 — dos preguntas de diseño, con evidencia, sin ejecutar nada

**¿`/srv/jax-data/repo` pasa a dueño `jaxsvc`?** Evidencia del código (grep
sobre `jacobs/executor.py` y `jax-platform/backend/api/admin/repository.py`):
el ÚNICO consumidor que escribe ahí es `jacobs/executor.py`
(`_persist_step_to_repo`, dentro de LAS MANOS), y el único que lee es
`jax-platform` vía `admin/repository.py` -- **los dos corren como
`jaxsvc`** (HECHOS de la tarea original). `fruiz` no es un consumidor en
tiempo de ejecución, sólo un operador humano ocasional. Hoy el modelo
mezcla dos mecanismos: permisos de grupo (`fruiz:jaxsvc 775`, que alcanza
para directorios) y ACL con nombre (necesarias porque ARCHIVOS
individuales tienen grupo `fruiz`, no `jaxsvc`, heredado de cuando se
crearon en el checkout de trabajo) -- la mezcla es exactamente lo frágil
que este incidente mostró. Pasar a dueño (o al menos grupo uniforme)
`jaxsvc`, con `setgid` en los directorios para que lo nuevo herede el
grupo, sacaría la dependencia de ACL para el caso de uso real. Es una
decisión de superficie de acceso (¿pierde `fruiz` algo hoy?) que no le
corresponde a este runbook tomar sola.

**¿`.claude-flow/` se purga del repo de producción?** Evidencia: `grep -rn
"claude-flow"` sobre TODO el código de aplicación (`las_manos/`, `jacobs/`,
`jax/`, `jax-platform`) da CERO resultados fuera de dos listas de exclusión
de un escáner de tests (`policy/tests/test_archivos_de_test_wireados_en_ci.py`
y `test_auditor_c5_punto_unico.py`, que ignoran ESE nombre de carpeta al
recorrer el árbol -- no lo usan como dato). Ningún camino de LAS MANOS ni de
jax-platform lee o escribe `.claude-flow/`: es un directorio de estado de
una herramienta de agentes (el paquete npm `claude-flow`) que quedó
mezclado ahí porque algún agente trabajó DENTRO de `documents/` en algún
momento, no un dato de producción. Candidato razonable a purgar, pero es
contenido (posible memoria/histórico de sesiones de agentes) que alguien
podría querer conservar -- decisión de Fernando, no de esta sesión.

## Prerrequisito -- y la decisión de NO desplegar (MAJOR-C)

**Este runbook NO exige mergear ni desplegar esta rama en
`/srv/jax-prod/jax` antes de cortar.** Las revisiones 1 y 2 sí lo exigían;
la ronda 2 de la auditoría encontró por qué eso era peligroso:
`jax/memory/worker.py` (línea ~479, rama `phase1/shared-memory-integration`
ya en `master`) construye el escritor de B9
(`build_persistent_extraction_writer`) SIN CONDICIÓN -- y las tablas que B9
necesita (`memory_objects`, `jax_project_scope`) **no existen** en la
MariaDB real de producción (puerto 3308, verificado con `SHOW TABLES LIKE`
el 2026-09-25: cero filas para las dos). Desplegar el código de esta rama
en `/srv/jax-prod/jax` HOY desplegaría B9 sin sus tablas -- un problema que
no tiene nada que ver con las rutas de producción, y que no le corresponde
a este runbook resolver ni esperar.

**Decisión (Hyde, con la evidencia de la auditoría):** el corte se hace
contra el código YA DESPLEGADO (commit `bd10297`, verificado con
`sudo git -c safe.directory=/srv/jax-prod/jax -C /srv/jax-prod/jax log -1`
-- el `-c safe.directory=...` puesto así, inline, no persiste ningún config
global; sin él, git rechaza el checkout con "dubious ownership" porque
corre como root vía sudo sobre un directorio de otro dueño, verificado
2026-09-25). Ese `server.py` resuelve
`AUDIT_LOG_PATH` con `os.getenv("JAX_AUDIT_LOG_PATH", SERVER_CFG["audit_log"])`
-- todavía sin el fail-closed de esta rama, pero eso no hace falta para el
corte: con la variable PUESTA (que es exactamente lo que este runbook
hace), `os.getenv` la toma sin caer a ningún default. El despliegue del
código nuevo de esta rama (el fail-closed de `server.py`, el propio guion
`ops/rutas-de-produccion.sh`, el job de CI) queda **APARTE**, tras un GO
propio, detrás de que B9 tenga sus tablas.

**Por eso el Paso 8.4 de este runbook corre `ops/rutas-de-produccion.sh
--verificar` DESDE EL WORKTREE de esta rama, no desde `/srv/jax-prod/jax`**
-- ese guion no existe todavía en el árbol desplegado, y no tiene por qué:
sólo LEE `/etc/jax/.env` y el estado real del sistema, no necesita estar
"instalado" en ningún lado en particular.

> ⚠️ **Advertencia explícita para quien despliegue esta rama más adelante:**
> avanzar `/srv/jax-prod/jax` al HEAD de esta rama (o de `master` una vez
> mergeada) despliega B9 sin sus tablas. Ese despliegue espera su propio
> GO, detrás de las migraciones de B9 -- no lo dispares como efecto
> colateral de "ya que estoy actualizando el checkout para las rutas".

## Qué cambia y qué no

| Clave | Hoy | Pasa a | Permitidos (por clave, ver `ops/rutas_de_produccion_verificador.py`) |
|---|---|---|---|
| `JAX_CONFIG_PATH` | `/home/fruiz/jax/config/config.toml` | `/srv/jax-prod/jax/config/config.toml` | sólo bajo `/srv/jax-prod/jax/config/` |
| `JAX_AUDIT_LOG_PATH` | `/home/fruiz/jax/las_manos/logs/audit.jsonl` | `/var/log/jax/las_manos/audit.jsonl` | sólo bajo `/var/log/jax/las_manos/` |
| `JAX_REPO_BASE` | `/home/fruiz/jax/repo` | `/srv/jax-data/repo` | sólo exactamente `/srv/jax-data/repo` |

Consumidores reales (verificado 2026-09-25): `JAX_CONFIG_PATH` ->
`jax-platform/backend/api/chat.py`; `JAX_AUDIT_LOG_PATH` ->
`las_manos/server.py` (escribe), `jax-platform/backend/api/audit.py` (lee);
`JAX_REPO_BASE` -> `jacobs/executor.py` dentro de LAS MANOS (escribe
`documents/`), `jax-platform/backend/api/admin/repository.py` (lee,
`require_superadmin`).

**JAX_MISSIONS_DIR y JAX_BIN NO se tocan en este runbook.** Tienen
consumidor real (`jax-platform/backend/api/command.py`), pero el lanzador
de `JAX_BIN` (`/home/fruiz/.local/bin/jax`) hace `cd $HOME/jax` y usa su
propio `.venv`: moverlo decide DESDE QUÉ CHECKOUT corre cada misión, una
decisión de arquitectura mayor que Fernando no tomó todavía.
`ops/rutas-de-produccion.sh --verificar` las excluye explícitamente
(`EXCEPCIONES_FASE_A`), con el motivo escrito.

**Servicios a tocar: `jax-las-manos` y `jax-platform` únicamente.**
Verificado con grep completo sobre `/srv/jax-prod/jax` y
`/srv/jax-prod/jax-platform`: ningún otro servicio con
`EnvironmentFile=/etc/jax/.env` (`jax-ejecutor-proxy`, `jax-memory-worker`,
`jax-memory-synthesis`) lee estas tres claves.

## Explícitamente FUERA DE ALCANCE de este runbook

- **`JAX_MISSIONS_DIR` / `JAX_BIN`** -- ver arriba. Pendiente, decisión de
  Fernando.
- **Los 4 defaults silenciosos de `JAX_WORKSPACE_DIR`** (verificado
  2026-09-25): `procesamiento/extractores/ocr.py:76`, `jacobs/executor.py:123`,
  `las_manos/motor_registry/tool_authority.py:64`,
  `jax/muscles/subprocess_muscle.py:55`. `JAX_WORKSPACE_DIR` SÍ está puesta
  hoy en `/etc/jax/.env` (se queda ahí, decisión de diseño), así que ninguno
  cae al default en producción hoy -- pero el default sigue vivo en el
  código. Fuera de alcance de esta tarea.
- **`/var/log/jax` sin logrotate.** `AuditLog` sólo hace `open(..., "a")`,
  nunca rota ni trunca. Pendiente aparte.
- **`fruiz` pierde lectura directa (sin `sudo`) del audit log nuevo.**
  `fruiz` NO pertenece al grupo `jaxsvc` (verificado con `id`), y
  `/var/log/jax/las_manos` queda `jaxsvc:jaxsvc 0750`. Documentado, no
  arreglado.
- **El despliegue del código de esta rama a `/srv/jax-prod/jax`.** Ver
  MAJOR-C arriba -- espera su propio GO, detrás de las tablas de B9.

## Horas a evitar

| Timer | Disparo | Mecanismo |
|---|---|---|
| `jax-memory-worker.timer` | cada ~20 min, observado en :10/:30/:50 | `OnUnitActiveSec=20min` -- relativo, no horario fijo |
| `jax-memory-synthesis.timer` | 04:30:00 en punto, todos los días | `OnCalendar=*-*-* 04:30:00`, sin margen aleatorio |
| `jax-limpiar-bases-de-test.timer` | ~00:00-00:15 | `OnCalendar=daily` + `RandomizedDelaySec=15m` |
| `jax-revisar-indice-vectorial.timer` | lunes ~06:30-06:50 | `OnCalendar=Mon *-*-* 06:30:00` + `RandomizedDelaySec=20m` |

Ninguno lee las tres claves que este cambio mueve -- es prudencia, no
correctitud. **Ventana recomendada, cada hora:** minuto **12 a 28** de cada
bloque de 20 (`HH:12`–`HH:28`, `HH:32`–`HH:48`, `HH:52`–`HH:08` del
siguiente bloque). Evitar además **04:25–04:35** y, si el corte cae un
lunes, **06:25–06:55**. Ventana de tráfico bajo para `jax-platform`.

## Convención de los bloques de este runbook

**Todo bloque que pueda abortar corre en un SUBSHELL** -- `( set -euo
pipefail; ...; )` -- nunca en la shell interactiva pelada. Un `exit 1`
dentro de `( ... )` termina SÓLO el subshell; el mismo `exit 1` sin los
paréntesis, pegado directo en una sesión SSH interactiva, cierra la sesión
entera y saca a quien lo estaba corriendo. Copiar y pegar cada bloque
COMPLETO, paréntesis incluidos.

**Nada de secretos en `argv` ni sourceados en la shell interactiva.** Donde
hace falta la credencial de la base (`JAX_DB_USER`/`JAX_DB_PASSWORD`), se arma
un `--defaults-extra-file` temporal de MySQL, modo 600, borrado con `trap`
al salir del subshell -- nunca `-p"$CONTRASEÑA"` en la línea de comandos
(visible en `ps aux` a cualquiera con shell en la máquina) ni `source
/etc/jax/.env` completo en la shell de quien ejecuta (eso expondría TODAS
las credenciales del archivo al entorno de esa sesión, no sólo las dos que
hacen falta).

**Las variables que cruzan de un bloque a otro se persisten en archivos**
(`~/rutas-prod.TS`, `~/rutas-prod.N0`), nunca en variables de shell: cada
bloque puede correr en una sesión SSH distinta.

## Paso 0 — antes de la ventana (servicios siguiendo corriendo)

```bash
(
set -euo pipefail

date +%Y%m%d-%H%M%S > ~/rutas-prod.TS
TS="$(cat ~/rutas-prod.TS)"
echo "TS=$TS"

if diff <(sudo cat /srv/jax-prod/jax/config/config.toml) <(cat /home/fruiz/jax/config/config.toml) > /tmp/diff-config-$$; then
  echo "IDENTICOS -- nada que copiar para JAX_CONFIG_PATH"
  rm -f /tmp/diff-config-$$
else
  echo "ABORTAR: /srv/jax-prod/jax/config/config.toml difiere del checkout de trabajo -- revisar antes de seguir"
  cat /tmp/diff-config-$$
  rm -f /tmp/diff-config-$$
  exit 1
fi

sudo mkdir -p /srv/jax-data/repo

# NO se hace una copia "de adelanto" del repo acá (ronda 4, MAJOR-3): las
# revisiones 1-3 de este documento SÍ la hacían -- una pasada de rsync -A
# temprana (este Paso) y otra final con --delete (Paso 4), pensadas para
# achicar la ventana. Causa raíz real, reproducida (ver el informe de esta
# ronda): cuando la SEGUNDA pasada de `rsync -aHAX` encuentra un archivo
# que YA está igual (mismo tamaño/mtime que dejó la primera pasada), rsync
# lo salta -- y ese salto NO reaplica el ACL con nombre (`user:jaxsvc:...`),
# lo BORRA. Confirmado con `--itemize-changes` (el archivo sin cambios sale
# marcado con la bandera de "cambio de xattr/ACL" y pierde la entrada con
# nombre; un archivo nuevo, transferido de verdad, la conserva) y
# reproducido tanto en ext4→ext4 como en un XFS real (loopback, el mismo
# tipo de filesystem que /srv/jax-data) -- no es un problema de los flags
# usados, es la interacción entre dos invocaciones separadas de rsync -A
# sobre el mismo destino. `--checksum` en la segunda pasada NO lo evita
# (probado). La única transferencia del repo es la del Paso 4 -- un solo
# rsync -A, nunca uno "de adelanto" seguido de otro "final" sobre el mismo
# árbol.

sudo mkdir -p /var/log/jax/las_manos
sudo chown jaxsvc:jaxsvc /var/log/jax/las_manos
sudo chmod 0750 /var/log/jax/las_manos
)
```

**N0 NO se captura acá.** Con los servicios todavía corriendo, tráfico real
puede seguir escribiendo en el audit log de trabajo entre este paso y el
Paso 3 (detener) -- si N0 se anclara ahora, esas líneas "de en medio"
quedarían contadas de más y R2 (reversión) las duplicaría. N0 se captura en
el Paso 5, con los servicios YA detenidos.

## Paso 1 — respaldo de `/etc/jax/.env`

```bash
(
set -euo pipefail

# Relee TS de archivo -- este bloque puede correr en una sesión SSH nueva,
# sin la variable del Paso 0 en memoria.
TS="$(cat ~/rutas-prod.TS)"

sudo cp -p /etc/jax/.env "/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}"
sudo chown root:root "/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}"
sudo chmod 600 "/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}"

# cmp byte a byte (ronda 3, MINOR): comparar tamaños es más débil -- dos
# archivos distintos pueden coincidir en tamaño por casualidad. cmp -s
# confirma que son el MISMO contenido, no sólo el mismo largo.
MODO_BACKUP="$(sudo stat -c%a "/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}")"
if sudo cmp -s /etc/jax/.env "/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}" && [ "$MODO_BACKUP" = "600" ]; then
  echo "Backup OK: idéntico byte a byte, modo 600"
else
  echo "ABORTAR: backup no coincide byte a byte con el original, o el modo no es 600 (modo: $MODO_BACKUP)"
  exit 1
fi
)
```

## Ventana corta empieza aquí

### Paso 2 — drenar ANTES de detener (MAJOR-5)

```bash
(
set -euo pipefail

# El .env se lee con sudo (fruiz no está en el grupo jaxsvc), pero SOLO se
# extraen las 3 claves de conexión a un archivo de config de MySQL temporal,
# modo 600, borrado al salir -- nunca `source` del .env completo en esta
# shell, nunca la contraseña como argumento de mysql.
CNF="$(mktemp)"
trap 'rm -f "$CNF"' EXIT
chmod 600 "$CNF"
{
  echo "[client]"
  echo "host=127.0.0.1"
  echo "port=3308"
  sudo awk -F= '/^JAX_DB_USER=/{print "user=" $2} /^JAX_DB_PASSWORD=/{print "password=" $2}' /etc/jax/.env
} > "$CNF"
JAX_DB_NAME="$(sudo awk -F= '/^JAX_DB_NAME=/{print $2}' /etc/jax/.env)"

# (a) Ningún pipeline pendiente/corriendo/interrumpido. Literales de status
#     verificados en el código (jacobs/continuar.py, jacobs/cupo.py,
#     jacobs/_cupo_io_test.py): 'pending', 'running', 'interrupted' son
#     reales, no inventados.
EN_VUELO="$(mysql --defaults-extra-file="$CNF" -N "$JAX_DB_NAME" <<'SQL'
SELECT COUNT(*) FROM jacobs_pipelines WHERE status IN ('pending','running','interrupted');
SQL
)"
echo "pipelines en vuelo: $EN_VUELO"
if [ "$EN_VUELO" -ne 0 ]; then
  echo "ABORTAR: hay $EN_VUELO pipeline(s) en vuelo -- no se detiene nada"
  exit 1
fi

# (b) Ningún proceso de misión corriendo -- command.py lanza literalmente
#     "$JAX_BIN --task <mission_file>".
PROCESOS="$(pgrep -u jaxsvc -f -- '--task' || true)"
if [ -n "$PROCESOS" ]; then
  echo "ABORTAR: hay procesos --task corriendo: $PROCESOS"
  exit 1
fi

echo "drenado OK: nada en vuelo, se puede detener"
)
```

Si aborta: **no seguir**. Esperar a que drene, o investigar por qué no
drena, antes de reintentar.

### Paso 3 — detener los dos servicios

```bash
sudo systemctl stop jax-las-manos jax-platform
systemctl is-active jax-las-manos jax-platform   # esperar "inactive" en las dos
```

### Paso 4 — rsync final (delta) de `JAX_REPO_BASE` + verificación sha256 + cerrar el acceso `other` (MAJOR-4)

```bash
(
set -euo pipefail

sudo rsync -aHAX --numeric-ids --delete /home/fruiz/jax/repo/ /srv/jax-data/repo/

ORIGEN_SHA="$(cd /home/fruiz/jax/repo && sudo find . -type f -exec sha256sum {} \; | sort)"
DESTINO_SHA="$(cd /srv/jax-data/repo && sudo find . -type f -exec sha256sum {} \; | sort)"
N_ARCHIVOS="$(printf '%s\n' "$ORIGEN_SHA" | grep -c .)"
if [ "$N_ARCHIVOS" -eq 0 ]; then
  echo "ABORTAR: 0 archivos comparados -- el arbol esta vacio o el find fallo en silencio"
  exit 1
fi
if [ "$ORIGEN_SHA" = "$DESTINO_SHA" ]; then
  echo "REPO_BASE: sha256 identico en los $N_ARCHIVOS archivos"
else
  echo "ABORTAR: sha256 NO COINCIDE entre origen y destino"
  diff <(printf '%s\n' "$ORIGEN_SHA") <(printf '%s\n' "$DESTINO_SHA") || true
  exit 1
fi

# MAJOR-4: /home/fruiz es 750 (bloqueaba "other"); /srv/jax-data es 775
# (world-traversable). El rsync preserva permisos del origen tal cual --
# sin esto, los documentos quedan legibles por cualquier usuario local.
sudo chmod -R o-rwx /srv/jax-data/repo
sudo setfacl -R -m o::--- -m d:o::--- /srv/jax-data/repo

SOBRANTES="$(sudo find /srv/jax-data/repo \( -perm -o=r -o -perm -o=w -o -perm -o=x \) 2>/dev/null | wc -l)"
if [ "$SOBRANTES" -eq 0 ]; then
  echo "MAJOR-4 cerrado: 0 archivos con acceso 'other' bajo /srv/jax-data/repo"
else
  echo "ABORTAR: $SOBRANTES archivo(s) siguen con permiso 'other'"
  exit 1
fi

# Marca de "Paso 4 terminó de verdad" (ronda 3, MINOR-3): R3 (reversión) la
# comprueba antes de sincronizar de vuelta -- sin esto, R3 sólo podía
# adivinar si el rsync final había corrido mirando si /srv/jax-data/repo
# existía, y ese directorio también lo crea el Paso 0 (con mkdir, aunque ya
# no con una copia de adelanto -- ver ronda 4, MAJOR-3).
date +%Y%m%d-%H%M%S > ~/rutas-prod.paso4-completo
)
```

### Paso 5 — copiar el audit log a `/var/log/jax/las_manos/audit.jsonl`

```bash
(
set -euo pipefail

# N0 se ancla ACÁ: los servicios ya están detenidos (Paso 3), el archivo de
# trabajo está quieto. Anclarlo antes (Paso 0) habría contado de más
# cualquier tráfico real entre ese momento y el corte.
wc -l < /home/fruiz/jax/las_manos/logs/audit.jsonl | tee ~/rutas-prod.N0

sudo cp -p /home/fruiz/jax/las_manos/logs/audit.jsonl /var/log/jax/las_manos/audit.jsonl
sudo chown jaxsvc:jaxsvc /var/log/jax/las_manos/audit.jsonl
sudo chmod 0640 /var/log/jax/las_manos/audit.jsonl

SHA_ORIGEN="$(sudo sha256sum /home/fruiz/jax/las_manos/logs/audit.jsonl | awk '{print $1}')"
SHA_DESTINO="$(sudo sha256sum /var/log/jax/las_manos/audit.jsonl | awk '{print $1}')"
if [ "$SHA_ORIGEN" = "$SHA_DESTINO" ]; then
  echo "AUDIT_LOG: sha256 identico"
else
  echo "ABORTAR: sha256 NO COINCIDE -- no seguir"
  exit 1
fi
)
```

El original en `/home/fruiz/jax/las_manos/logs/audit.jsonl` **queda intacto**
por ahora (se congela recién en el Paso 9) -- esto es una copia, no una
migración destructiva.

### Paso 6 — editar `/etc/jax/.env` de forma atómica (MAJOR-1)

```bash
(
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
MODO_TMP="$(sudo stat -c "%U:%G %a" /etc/jax/.env.tmp)"
if [ "$MODO_TMP" != "root:jaxsvc 640" ]; then
  echo "ABORTAR: /etc/jax/.env.tmp no quedo root:jaxsvc 640 (quedo: $MODO_TMP)"
  sudo rm -f /etc/jax/.env.tmp
  exit 1
fi

DIFF_TMP="$(diff -U0 <(sudo cat /etc/jax/.env) <(sudo cat /etc/jax/.env.tmp) || true)"
echo "$DIFF_TMP"
N_LINEAS_DE_CAMBIO="$(printf '%s\n' "$DIFF_TMP" | grep -cE '^[+-][^+-]')"
if [ "$N_LINEAS_DE_CAMBIO" -ne 6 ]; then
  echo "ABORTAR: el diff tiene $N_LINEAS_DE_CAMBIO lineas de cambio, se esperaban exactamente 6 (3 claves x 2 lineas -/+)."
  sudo rm -f /etc/jax/.env.tmp
  exit 1
fi
echo "diff OK: exactamente 3 claves cambiadas, nada mas"

sudo mv /etc/jax/.env.tmp /etc/jax/.env
MODO_FINAL="$(sudo stat -c "%U:%G %a" /etc/jax/.env)"
if [ "$MODO_FINAL" != "root:jaxsvc 640" ]; then
  echo "ABORTAR: /etc/jax/.env NO quedo root:jaxsvc 640 tras el mv (quedo: $MODO_FINAL) -- restaurar del backup YA"
  exit 1
fi
echo "/etc/jax/.env: $MODO_FINAL"
)
```

El archivo temporal nace con `umask 077` (600 desde el primer byte, nunca
una ventana con el `.env` completo -- credenciales incluidas -- en
`root:root 644` legible por cualquiera).

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

Las dos tienen que dar `200`. `/health` de LAS MANOS ya prueba escritura del
audit log (`comprobar_audit(audit.log_path)`, `las_manos/salud.py`) -- por
eso este runbook no dispara un `ENVELOPE_REJECTED` sintético contra
`/jacobs/preflight`: ensuciaría la auditoría forense con eventos de prueba.

> **Nota para quien audite el log más adelante:** SÍ existe un evento
> `ENVELOPE_REJECTED` sintético, de una prueba manual de esta tarea (no de
> este runbook), `@timestamp` **2026-09-25T10:49:56.746832+00:00**,
> `reason: "campos faltantes/mal-tipados: ['invoked_by', 'steps']"`. No es
> un incidente.

**8.2 — `/proc/<PID>/environ` con las rutas nuevas** (rutas, no secretos --
se imprimen sin problema):

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

`GET /api/admin/repo/file` exige `require_superadmin` (JWT de sesión) --
usar un token de una sesión de superadmin ya autenticada, puesto en
`$JWT_SUPERADMIN` (variable de shell, no argumento de proceso). El token
NUNCA va en la línea de comandos de `curl` (ronda 3, MINOR-5: `-H
"Authorization: Bearer $JWT..."` deja el token visible en `ps aux` a
cualquiera con shell en la máquina, igual que `-p"$PW"` de mysql) -- va en
un archivo de config de `curl` temporal, modo 600, borrado al salir:

```bash
(
set -euo pipefail
CURLCFG="$(mktemp)"
trap 'rm -f "$CURLCFG"' EXIT
chmod 600 "$CURLCFG"
printf 'header = "Authorization: Bearer %s"\n' "$JWT_SUPERADMIN" > "$CURLCFG"

curl -s -K "$CURLCFG" -o /dev/null -w 'GET /api/admin/repo -> %{http_code}\n' \
  http://127.0.0.1:8080/api/admin/repo

curl -s -K "$CURLCFG" -o /dev/null -w 'GET /api/admin/repo/file -> %{http_code}\n' \
  "http://127.0.0.1:8080/api/admin/repo/file?path=documents/04e02b09_00_hipatia.md"
)
```

Las dos tienen que dar `200`. Alternativa: abrir el panel de administración
de documentos en el navegador con una sesión de Fernando ya iniciada.

**8.4 — El guion de verificación, corrido DESDE EL WORKTREE (MAJOR-C -- NO
desde `/srv/jax-prod/jax`, que no lo tiene):**

```bash
cd ~/worktrees/jax-rutas-prod   # o donde esté el checkout de esta rama en el SHA auditado
git log -1 --oneline            # confirmar que HEAD es el commit auditado antes de confiar en la salida
./ops/rutas-de-produccion.sh --verificar
echo "RC=$?"
```

Tiene que dar `RC=0`. Incluye Fase C (verdad efectiva contra
`/proc/<MainPID>/environ` de los dos servicios) -- si 8.2 ya dio bien, esto
es la misma pregunta confirmada por el propio guion, no un paso nuevo que
pueda dar un resultado distinto sin que algo esté realmente mal.

### Paso 9 — congelar las copias viejas del audit log

**Sólo después de que el Paso 8 completo dio verde.**

```bash
sudo chattr +i /home/fruiz/jax/las_manos/logs/audit.jsonl
lsattr /home/fruiz/jax/las_manos/logs/audit.jsonl
```

**La copia de trabajo** (`/home/fruiz/jax/las_manos/logs/audit.jsonl`) se
congela con `chattr +i`. Corrección (ronda 3, MINOR-4): la revisión
anterior de este documento decía que este archivo "no vive dentro de
ningún árbol que `git clean` necesite poder tocar" -- **eso es falso**.
`/home/fruiz/jax` SÍ es un checkout git (el de trabajo de un agente), y
`las_manos/logs/` SÍ está adentro; lo que pasa es que `.gitignore:26`
(`*/logs/` -- no `logs/` a secas, que es la línea 25; corregido ronda 4,
MINOR-5) ya lo excluye de un `git clean -fd` NORMAL (sin `-x`), así que
`chattr +i` no agrega protección contra ESE caso -- sólo contra un `git
clean -fdx` (que sí incluye archivos ignorados) o contra una escritura
accidental de algún script viejo. Es una protección real, sólo que el
motivo original estaba mal explicado.

**La copia del checkout DE DESPLIEGUE**
(`/srv/jax-prod/jax/las_manos/logs/audit.jsonl`, si existe, ya
desactualizada desde 2026-09-20) **NO se marca `+i`**: ese archivo SÍ vive
dentro de un árbol donde un futuro `git clean -fdx` (o un redeploy que
regenere el checkout) necesita poder tocarlo -- un inmutable ahí rompería
esa operación de mantenimiento sin ninguna ganancia real (ya es una copia
vieja, no la fuente de verdad de nada). En su lugar, sólo lectura:

```bash
sudo test -f /srv/jax-prod/jax/las_manos/logs/audit.jsonl && \
  sudo chmod a-w /srv/jax-prod/jax/las_manos/logs/audit.jsonl
```

Si `chattr +i` no está disponible en el filesystem de la copia de trabajo,
usar `chmod a-w` ahí también y anotarlo en el registro de esta tarea.

## Ventana corta termina aquí

## Reversión

**Adaptativa (ronda 2, MINOR): cada paso comprueba si el correspondiente
paso de ida llegó a correr, y se salta solo si no.** Si el corte falló
temprano (antes del Paso 5), no hay `~/rutas-prod.N0` -- R2 no aplica. Si
falló antes del Paso 4, no hay nada nuevo en `/srv/jax-data/repo` que
sincronizar de vuelta -- R3 no aplica.

**Toda la secuencia de abajo (preámbulo + R1-R5) se probó de punta a punta
en `/tmp`, con `sudo unshare --mount`, reproduciendo el estado real
posterior al corte** (incluido el `chattr +i` del Paso 9 -- eso sólo se
puede probar con root de verdad, no con un mock) -- ver la evidencia en el
informe de esta ronda.

```bash
(
set -euo pipefail
TS="$(cat ~/rutas-prod.TS)"
BACKUP="/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}"
if [ ! -f "$BACKUP" ]; then
  echo "ABORTAR REVERSION: no existe $BACKUP -- no hay de donde restaurar"
  exit 1
fi
echo "BACKUP=$BACKUP (TS=$TS)"

# MAJOR-1(b), ronda 4: el Paso 9 dejó el audit log viejo INMUTABLE
# (`chattr +i`) a propósito. R2 (más abajo) necesita escribirle (`tee -a`)
# -- sin esto, `tee -a` falla con EPERM y la reversión aborta CON LOS
# SERVICIOS YA DETENIDOS (si esto corriera después de R1). Por eso va acá,
# en el preámbulo, ANTES de tocar ningún servicio: si quitar +i fallara,
# mejor enterarse ahora que a mitad de una ventana de corte.
OLD_AUDIT=/home/fruiz/jax/las_manos/logs/audit.jsonl
if lsattr "$OLD_AUDIT" 2>/dev/null | grep -q '^....i'; then
  sudo chattr -i "$OLD_AUDIT"
  if lsattr "$OLD_AUDIT" 2>/dev/null | grep -q '^....i'; then
    echo "ABORTAR: no se pudo quitar +i de $OLD_AUDIT"
    exit 1
  fi
  echo "OK: +i retirado de $OLD_AUDIT"
else
  echo "OK: $OLD_AUDIT ya no tiene +i (o el filesystem no lo soporta)"
fi
)
```

**R1 — detener:**

```bash
sudo systemctl stop jax-las-manos jax-platform
```

**R2 — anexar al audit log VIEJO lo nuevo, idempotente, con abort real
(MAJOR-B):**

```bash
(
set -euo pipefail
OLD=/home/fruiz/jax/las_manos/logs/audit.jsonl
NEW=/var/log/jax/las_manos/audit.jsonl

if [ ! -f ~/rutas-prod.N0 ]; then
  echo "R2: no existe ~/rutas-prod.N0 -- el Paso 5 nunca corrio, nada que anexar"
elif [ ! -f "$NEW" ]; then
  echo "R2: $NEW no existe -- el Paso 5 nunca corrio, nada que anexar"
else
  N0="$(cat ~/rutas-prod.N0)"
  if sudo cmp -s "$OLD" "$NEW"; then
    # Idempotente: si esto es una reversion REPETIDA (o nunca hubo
    # escrituras nuevas desde el corte), OLD y NEW ya coinciden -- anexar
    # de nuevo duplicaria las lineas. Antes esta comparacion comparaba
    # tail -n +N0+1 del VIEJO contra el NUEVO COMPLETO -- dos slices de
    # largo distinto que nunca daban igual si N0>0, y bajo `cmd && echo`
    # con set -e NO abortaba aunque el cmp fallara siempre (MAJOR-B).
    echo "R2: ya coinciden byte a byte -- nada que anexar"
  else
    # Guardia PREVIA (ronda 3, MINOR-2): antes de anexar, confirmar que el
    # viejo, TAL COMO ESTÁ HOY, son exactamente las primeras N0 líneas del
    # nuevo -- si el viejo cambió por cualquier otro motivo desde que se
    # congeló (Paso 5), anexar a ciegas mezclaría dos historias que no
    # encajan. `head -n N0 NEW | cmp - OLD` compara ESA porción del nuevo
    # contra el viejo completo sin escribir nada todavía.
    if ! sudo sh -c "head -n '$N0' '$NEW' | cmp -s - '$OLD'"; then
      echo "ABORTAR: el audit log viejo no coincide con las primeras $N0 líneas del nuevo -- algo lo cambió desde el Paso 5, no se anexa a ciegas"
      exit 1
    fi
    sudo tail -n +"$((N0 + 1))" "$NEW" | sudo tee -a "$OLD" > /dev/null
    if sudo cmp -s "$OLD" "$NEW"; then
      echo "R2 OK: el audit log viejo coincide byte a byte con el nuevo"
    else
      echo "ABORTAR: tras anexar, el audit log viejo AUN no coincide con el nuevo -- revisar a mano, no seguir"
      exit 1
    fi
  fi
fi
)
```

**R3 — rsync de `documents/` de vuelta, SIN `--delete`, adaptativo (ronda 3,
MINOR-3):**

```bash
(
set -euo pipefail
# La marca de ~/rutas-prod.paso4-completo (NO sólo "existe /srv/jax-data/repo",
# que también crea el Paso 0 con un mkdir vacío) confirma que el ÚNICO
# rsync del repo (Paso 4, ronda 4: ya no hay una pasada de adelanto en el
# Paso 0) de verdad corrió -- si el corte falló antes del Paso 4,
# /srv/jax-data/repo existe pero vacío o con datos parciales, y
# sincronizarlos de vuelta sería un error.
if [ -f ~/rutas-prod.paso4-completo ]; then
  # --ignore-existing (ronda 4, MAJOR-3): NUNCA toca un archivo que ya
  # existe en el destino (/home/fruiz/jax/repo), aunque el contenido sea
  # igual -- sólo trae lo genuinamente NUEVO. Reemplaza a -u (--update):
  # -u todavía deja que rsync "reexamine" archivos con igual contenido, y
  # eso es EXACTAMENTE lo que le borró el ACL con nombre a los documentos
  # originales en la primera versión de este runbook (ver la nota del
  # Paso 0) -- probado que --ignore-existing NO tiene ese problema: dejé
  # ACL rotas del lado del origen a propósito y el destino, que ya tenía
  # el archivo, salió intacto.
  sudo rsync -aHAX --ignore-existing --numeric-ids /srv/jax-data/repo/ /home/fruiz/jax/repo/
  echo "R3 OK: sincronizado de vuelta (sólo lo nuevo; nada preexistente se tocó)"
else
  echo "R3: no existe ~/rutas-prod.paso4-completo -- el Paso 4 nunca terminó, nada confiable que sincronizar"
fi
)
```

**R4 — restaurar `.env`, con la misma disciplina de permisos del Paso 6:**

```bash
(
set -euo pipefail
TS="$(cat ~/rutas-prod.TS)"
BACKUP="/etc/jax/.env.backup-pre-rutas-de-produccion-${TS}"
[ -f "$BACKUP" ] || { echo "ABORTAR: no existe $BACKUP"; exit 1; }

sudo sh -c "umask 077; cp -p '$BACKUP' /etc/jax/.env.restaurado"
sudo chown root:jaxsvc /etc/jax/.env.restaurado
sudo chmod 640 /etc/jax/.env.restaurado
sudo mv /etc/jax/.env.restaurado /etc/jax/.env
MODO_FINAL="$(sudo stat -c "%U:%G %a" /etc/jax/.env)"
if [ "$MODO_FINAL" != "root:jaxsvc 640" ]; then
  echo "ABORTAR: /etc/jax/.env no quedo root:jaxsvc 640 tras restaurar (quedo: $MODO_FINAL)"
  exit 1
fi
echo "/etc/jax/.env restaurado: $MODO_FINAL"
)
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
la reversión surtió efecto de verdad.

Si el corte nunca llegó al Paso 9 (congelar), no hace falta descongelar
nada. Si por error se congeló ANTES de decidir revertir, `sudo chattr -i
/home/fruiz/jax/las_manos/logs/audit.jsonl` antes de R2 (el `tee -a`
fallaría contra un archivo inmutable).
