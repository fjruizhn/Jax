# Proyectos E2a en producción
## Purpose
Llevar a producción la subida de documentos a un proyecto (E2a): permisos de `proyectos/`, LAS MANOS por `project_uuid`, la tabla `project_documents` (migración 006a), la plataforma con la pestaña Documentos y la conversión de la carpeta suelta de LACTOVI en un proyecto real. Orden de la Parte C del plan (`docs/superpowers/plans/2026-10-03-proyectos-e2a-plan.md`, Tarea 13).
## Scope
`/srv/jax-data/jax-workspace/proyectos/` (permisos y la carpeta `lacteos-victoria` -> `<uuid>`), `jax_memory` (MariaDB `127.0.0.1:3308`: tabla `project_documents`, el proyecto de LACTOVI), los servicios `jax-las-manos` y `jax-platform`. Guiones: `ops/permisos_proyectos.py`, `scripts/proyectos_e2a_lactovi.py`.
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
  **La regla que quedó instalada** (Fernando, 2026-10-03; `/etc/sudoers.d/jax-permisos-proyectos`, `440 root:root`, validada con `visudo -cf`):
  ```
  fruiz ALL=(root) NOPASSWD: /usr/bin/python3 -I /usr/local/sbin/jax-permisos-proyectos --nucleo-privilegiado, /usr/bin/python3 -I /usr/local/sbin/jax-permisos-proyectos --nucleo-respaldo, /usr/bin/python3 -I /usr/local/sbin/jax-permisos-proyectos --nucleo-deshacer, /usr/bin/grep \^JAX_WORKSPACE_DIR= /etc/jax/.env, /usr/bin/true
  ```
  **El `^` va escapado (`\^`).** En sudo 1.9.17, un argumento que empieza con `^` se interpreta como expresión regular, y sin escapar `visudo` lo rechaza con «unterminated regular expression». Con `\^` coincide con el `^` literal que pasa el guion y rechaza otros argumentos (comprobado con una regla temporal). `sudo -l` lo muestra como `\^JAX_WORKSPACE_DIR\=`.
  El núcleo lo instala root desde un checkout de jax master: `sudo install -o root -g root -m 0755 ops/permisos_proyectos.py /usr/local/sbin/jax-permisos-proyectos`. Se verifica con `sha256sum /usr/local/sbin/jax-permisos-proyectos`, que tiene que dar lo mismo que `git show origin/master:ops/permisos_proyectos.py | sha256sum`.
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
Volcado de `jax_memory` (`mariadb-dump --single-transaction --routines --triggers`) a `$D`, más `ESTADO-ANTES.txt` con los SHAs desplegados y `du -sh /srv/jax-data/jax-workspace/proyectos` (el «antes» de la Biblioteca, costo de R2). **Restaurar a una base descartable** (`jax_memory_test_e2aensayo`, quitando los `DEFINER`) y comparar conteos contra producción con el §0 de `despliegue.md` de jax-platform; si un par difiere, no se sigue. Un volcado sin restauración probada no es respaldo (Principio VI). Verificación: los conteos de `projects jax_users conversations messages` coinciden; `DROP DATABASE jax_memory_test_e2aensayo` al terminar (solo esa).
### 2. Permisos de `proyectos/`
```bash
python3 ops/permisos_proyectos.py --verificar      # solo lectura; anotar la salida
sudo -l                                            # confirmar la regla de sudoers (Preconditions)
sudo install -o root -g root -m 0755 ops/permisos_proyectos.py /usr/local/sbin/jax-permisos-proyectos
# --aplicar y el --verificar posterior van en el bloque de abajo, con las unidades de jaxsvc detenidas
```
**`--aplicar` y `--deshacer` se corren SOLO con TODAS las unidades de `jaxsvc` detenidas, en UN bloque, y se restauran al salir.** El guion falla cerrado, sin mutar nada, si hay cualquier proceso o hilo con el uid de `jaxsvc` (lo lee de `/proc`; si no puede leer el estado de un pid listado, también falla; `--verificar` no lo exige). Por qué, en dos líneas: un proceso `jaxsvc` vivo puede renombrar carpetas mientras root recorre el árbol, y todas las carreras de renombre (symlinks, hardlinks, intercambio de nombres, ocultas que cambian de proyecto) parten de eso; sin procesos `jaxsvc`, nadie con permiso de renombrar corre en paralelo (queda `fruiz`, dueño, y root, que es confiable por premisa). Las unidades solo se **detienen**: no se enmascaran, porque viven en `/etc/systemd/system` y un `mask --runtime` no las tapa; con los timers parados no se disparan. El bloque, en este orden: (a) premisas (sin archivos setuid/setgid de `jaxsvc` y sin crontab de `jaxsvc`; el `find` corre con `LC_ALL=C` y tolera SOLO un error `Permission denied` cuyo path sea exactamente un punto de montaje `fuse*` de `findmnt` —p. ej. el sshfs de hall9000, que da EACCES hasta a root—: cualquier otro error, o un `find` que falla sin mensaje, corta); (b) la lista de unidades y timers de `jaxsvc` (`list-units` y `list-timers` con `'jax*'`, filtradas por `User=`), con **cada consulta capturada: si una falla, o la lista sale vacía, o falta una de la lista mínima esperada, se corta**; el estado textual (`active`, `inactive`, `failed`, `activating`) de cada una, guardado ANTES de detener nada; (c) el `trap` de restauración; (d) `stop` de cada una; (e) comprobar por su salida textual que quedaron detenidas, `ps -u jaxsvc` vacío y `/proc/*/status` y `/proc/*/task/*/status` sin los cuatro uid; (f) el guion; (g) `--verificar`; (h) el trap **arranca en orden inverso las que estaban `active` o `activating` (una unidad que estaba arrancando se restaura igual), exige que vuelvan a `active` **y estables** (siguen `active` durante `ESTABLE` segundos, por defecto 5, y `NRestarts` —`systemctl show -p NRestarts --value`— no sube en ese intervalo: una unidad en bucle de reinicios se ve `active` o `activating` un instante y no cuenta) y, si alguna no vuelve, dice cuáles y cómo arrancarla a mano y sale con código distinto de 0 aunque el guion haya ido bien** (conserva el código del fallo original si lo hubo). **Desde justo antes del primer `stop` el bloque ignora INT, TERM y HUP** (los procesos hijos, el guion incluido, heredan el ignorar) y antes de detener nada imprime el aviso con la orden para arrancar a mano. Por qué, en dos líneas: cortar a mitad de la aplicación no es seguro (el árbol queda a medias y el trap tendría que restaurar mientras otra señal lo interrumpe), y `--aplicar` es idempotente, así que lo correcto es dejar que termine y repetirlo si hace falta. Si de verdad hay que cortarlo, `kill -9` y después `sudo systemctl start <las unidades activas>` (el aviso las lista).

**Bloque de `--aplicar`:**
```bash
# BLOQUE-APLICAR
set -euo pipefail
PERMISOS="${PERMISOS:-python3 ops/permisos_proyectos.py}"   # (las pruebas lo sustituyen)
RAIZ="${RAIZ:-/srv/jax-data/jax-workspace}"
PROC="${PROC:-/proc}"
# Lista mínima que TIENE que aparecer (hoy en hall9000): si alguna falta, el descubrimiento está incompleto y se corta.
ESPERADAS="${ESPERADAS:-jax-las-manos.service jax-platform.service jax-ariadna-pm.service jax-ejecutor-proxy.service jax-catalogo-modelos.service jax-limpiar-bases-de-test.service jax-catalogo-modelos.timer jax-limpiar-bases-de-test.timer}"
REINTENTOS="${REINTENTOS:-5}"; ESPERA="${ESPERA:-1}"
# ESTABLE: segundos que una unidad restaurada tiene que seguir 'active', sin que NRestarts suba, para contar como vuelta.
ESTABLE="${ESTABLE:-5}"
falla() { echo "NO CUMPLE: $*" >&2; exit 1; }
case "$ESTABLE" in ''|*[!0-9]*) falla "ESTABLE tiene que ser un entero de segundos: '$ESTABLE'" ;; esac

# (a) PREMISAS: si no se cumplen, el bloque falla antes de tocar nada.
#     Un montaje FUSE (en hall9000, sshfs) da EACCES incluso a root y `find` sale 1. Se toleran SOLO los errores
#     «Permission denied» cuyo path es EXACTAMENTE un punto de montaje fuse* de `findmnt`; cualquier otro error, o un
#     fallo sin mensaje, corta. `find` corre con LC_ALL=C (vía `env`, porque `sudo` puede limpiar el entorno): su mensaje
#     es `find: '/ruta': Permission denied` con comillas ASCII en cualquier idioma, y así se parsea.
#     (`findmnt -r` escapa los espacios del path como \x20: se decodifican para compararlos con lo que imprime find.)
FUSE="$(findmnt -rn -o TARGET,FSTYPE | awk '$2 ~ /^fuse/ {print $1}' | while IFS= read -r m; do printf '%b\n' "$m"; done)" \
  || falla "no se pudo listar los montajes (findmnt)"
ERRF="$(mktemp)" || falla "no se pudo crear un archivo temporal"
SETUID="$(sudo env LC_ALL=C find / "$RAIZ" -xdev \( -path /proc -o -path /sys \) -prune -o -type f -user jaxsvc -perm /6000 -print 2>"$ERRF")" || {
  [ -s "$ERRF" ] || { rm -f "$ERRF"; falla "no se pudo buscar archivos setuid/setgid de jaxsvc: find falló sin mensaje"; }
  while IFS= read -r linea; do
    p="${linea#"find: '"}"; p="${p%"': Permission denied"}"
    if [ "$p" = "$linea" ] || ! printf '%s\n' "$FUSE" | grep -qxF -- "$p"; then
      rm -f "$ERRF"; falla "no se pudo buscar archivos setuid/setgid de jaxsvc: $linea"
    fi
  done < "$ERRF"; }
rm -f "$ERRF"
[ -z "$SETUID" ] || falla "hay archivos setuid/setgid de jaxsvc (podrían volver a darle uid): $SETUID"
CRON="$(sudo crontab -u jaxsvc -l 2>&1 || true)"
case "$CRON" in *"no crontab"*) ;; *) falla "jaxsvc tiene crontab (podría arrancar procesos): $CRON" ;; esac

# (b) LISTA de unidades y timers de jaxsvc. NINGUNA consulta puede fallar en silencio: cada salida se captura y, si la
#     consulta falla, se corta. Un timer, socket o path cuenta por la unidad que dispara (Triggers=).
UNIDADES="$(sudo systemctl list-units --all --plain --no-legend 'jax*')" || falla "falló systemctl list-units"
TEMPORIZADORES="$(sudo systemctl list-timers --all --plain --no-legend 'jax*')" || falla "falló systemctl list-timers"
CANDIDATAS="$({ printf '%s\n' "$UNIDADES" | awk 'NF {print $1}'; \
  printf '%s\n' "$TEMPORIZADORES" | awk '{for (i = 1; i <= NF; i++) if ($i ~ /\.timer$/) print $i}'; } | sort -u)" \
  || falla "no se pudo armar la lista de unidades"
[ -n "$CANDIDATAS" ] || falla "la lista de unidades 'jax*' está vacía (en hall9000 hay al menos las de ESPERADAS)"
TIMERS=(); ACTIVADORES=(); SERVICIOS=()
while read -r u; do
  de_jaxsvc=no
  usuario="$(sudo systemctl show -p User --value "$u")" || falla "falló systemctl show -p User $u"
  [ "$usuario" = jaxsvc ] && de_jaxsvc=si
  if [ "$de_jaxsvc" = no ]; then
    disparadas="$(sudo systemctl show -p Triggers --value "$u")" || falla "falló systemctl show -p Triggers $u"
    for t in $disparadas; do
      usuario_t="$(sudo systemctl show -p User --value "$t")" || falla "falló systemctl show -p User $t"
      [ "$usuario_t" = jaxsvc ] && de_jaxsvc=si
    done
  fi
  [ "$de_jaxsvc" = si ] || continue
  case "$u" in *.timer) TIMERS+=("$u") ;; *.service) SERVICIOS+=("$u") ;; *) ACTIVADORES+=("$u") ;; esac
done <<< "$CANDIDATAS"
LISTA=("${TIMERS[@]}" "${ACTIVADORES[@]}" "${SERVICIOS[@]}")   # los timers se paran primero
[ "${#LISTA[@]}" -gt 0 ] || falla "ninguna unidad de jaxsvc en la lista: el descubrimiento falló"
for esperada in $ESPERADAS; do
  encontrada=no
  for u in "${LISTA[@]}"; do [ "$u" = "$esperada" ] && encontrada=si; done
  [ "$encontrada" = si ] || falla "falta la unidad esperada de jaxsvc $esperada: el descubrimiento está incompleto"
done

# ESTADO de CADA unidad ANTES de detener nada, por su salida textual (no solo por el código de salida).
ESTADOS=()
for u in "${LISTA[@]}"; do
  estado="$(sudo systemctl is-active "$u" || true)"
  case "$estado" in active|inactive|failed|activating) ;; *) falla "estado inesperado de $u: '$estado'" ;; esac
  ESTADOS+=("$estado")
done

# (c) TRAP de restauración (EXIT): arranca en orden INVERSO solo las que estaban 'active' o 'activating' y exige que
#     vuelvan a 'active' (con unos pocos reintentos) Y ESTABLES: siguen 'active' durante ESTABLE segundos y NRestarts
#     (propiedad de los .service; un .timer no la tiene) no sube en ese intervalo. Una unidad en bucle de reinicios se
#     ve 'active' o 'activating' un instante: eso no cuenta. Si alguna no vuelve, lo dice, explica cómo arrancarla y sale
#     con código distinto de 0 AUNQUE la aplicación haya ido bien. Conserva el código del fallo original si lo hubo.
restaurar() {
  local rc=$? i u intento t n fallidas=() NR0=() MALA=() MOTIVO=()      # rc del fallo real (o 0)
  trap - EXIT
  set +e
  for ((i = ${#LISTA[@]} - 1; i >= 0; i--)); do
    [ "${ESTADOS[i]}" = active ] || [ "${ESTADOS[i]}" = activating ] || continue
    sudo systemctl start "${LISTA[i]}" || echo "AVISO: systemctl start ${LISTA[i]} falló" >&2
  done
  # 1) cada una tiene que llegar a 'active' (reintentos); entonces se anota su NRestarts de partida
  for ((i = ${#LISTA[@]} - 1; i >= 0; i--)); do
    [ "${ESTADOS[i]}" = active ] || [ "${ESTADOS[i]}" = activating ] || continue
    u="${LISTA[i]}"
    for ((intento = 0; intento < REINTENTOS; intento++)); do
      [ "$(sudo systemctl is-active "$u")" = active ] && break
      sleep "$ESPERA"
    done
    if [ "$(sudo systemctl is-active "$u")" != active ]; then MALA[i]=1; MOTIVO[i]="no llegó a 'active'"; continue; fi
    n="$(sudo systemctl show -p NRestarts --value "$u")" || { MALA[i]=1; MOTIVO[i]="no se pudo leer NRestarts"; continue; }
    case "$n" in
      *[!0-9]*) MALA[i]=1; MOTIVO[i]="NRestarts ilegible: '$n'"; continue ;;
      '') case "$u" in *.service) MALA[i]=1; MOTIVO[i]="NRestarts vacío"; continue ;; esac ;;
    esac
    NR0[i]="$n"
  done
  # 2) ventana de estabilidad: ESTABLE segundos, todas a la vez; una que deja de estar 'active' queda marcada
  for ((t = 0; t < ESTABLE; t++)); do
    sleep 1
    for ((i = ${#LISTA[@]} - 1; i >= 0; i--)); do
      [ "${ESTADOS[i]}" = active ] || [ "${ESTADOS[i]}" = activating ] || continue
      [ -z "${MALA[i]:-}" ] || continue
      [ "$(sudo systemctl is-active "${LISTA[i]}")" = active ] || { MALA[i]=1; MOTIVO[i]="dejó de estar 'active' en la ventana de ${ESTABLE} s"; }
    done
  done
  # 3) NRestarts no tiene que haber subido
  for ((i = ${#LISTA[@]} - 1; i >= 0; i--)); do
    [ "${ESTADOS[i]}" = active ] || [ "${ESTADOS[i]}" = activating ] || continue
    [ -z "${MALA[i]:-}" ] || continue
    [ -n "${NR0[i]:-}" ] || continue
    n="$(sudo systemctl show -p NRestarts --value "${LISTA[i]}")" || n=""
    [ "$n" = "${NR0[i]}" ] || { MALA[i]=1; MOTIVO[i]="NRestarts pasó de ${NR0[i]} a '${n}' (bucle de reinicios)"; }
  done
  for ((i = ${#LISTA[@]} - 1; i >= 0; i--)); do
    [ -z "${MALA[i]:-}" ] || fallidas+=("${LISTA[i]}")
  done
  if [ "${#fallidas[@]}" -gt 0 ]; then
    echo "ERROR: no volvieron a 'active' estable (${ESTABLE} s, sin subir NRestarts): ${fallidas[*]}" >&2
    for ((i = ${#LISTA[@]} - 1; i >= 0; i--)); do
      [ -z "${MALA[i]:-}" ] || echo "  ${LISTA[i]}: ${MOTIVO[i]}" >&2
    done
    for u in "${fallidas[@]}"; do echo "  arrancarla a mano: sudo systemctl start $u" >&2; done
    [ "$rc" -ne 0 ] || rc=1
  fi
  exit "$rc"
}
trap restaurar EXIT

# VENTANA SIN SEÑALES: desde justo antes del primer stop (con el trap EXIT ya armado) INT, TERM y HUP se IGNORAN hasta
# el final, incluidos el guion y la restauración (los procesos hijos heredan el ignorar). Así no hay ventanas entre
# manejadores: el rc es el del fallo real o 0.
A_ARRANCAR=""
for ((i = ${#LISTA[@]} - 1; i >= 0; i--)); do
  if [ "${ESTADOS[i]}" = active ] || [ "${ESTADOS[i]}" = activating ]; then A_ARRANCAR="$A_ARRANCAR ${LISTA[i]}"; fi
done
echo "AVISO: durante la ventana el bloque no se interrumpe con Ctrl-C; si hace falta cortarlo, kill -9 y después: sudo systemctl start${A_ARRANCAR}" >&2
trap '' INT TERM HUP

# (d) PARAR cada timer y unidad (sin mask: las unidades viven en /etc/systemd/system y un mask --runtime no las tapa;
#     con los timers parados no se disparan, y arrancarlos solo puede hacerlo root, que es confiable por premisa).
for u in "${LISTA[@]}"; do sudo systemctl stop "$u" || falla "no se pudo detener $u"; done

# (e) COMPROBAR: ninguna activa (por su salida textual), `ps -u jaxsvc` vacío y /proc sin los cuatro uid de jaxsvc.
for u in "${LISTA[@]}"; do
  estado="$(sudo systemctl is-active "$u" || true)"
  case "$estado" in inactive|failed) ;; *) falla "$u no quedó detenida (estado: '$estado')" ;; esac
done
PS_SALIDA="$(ps -u jaxsvc -o pid= 2>&1)" || { rc_ps=$?; [ "$rc_ps" -eq 1 ] || falla "ps -u jaxsvc falló (rc $rc_ps)"; }
[ -z "$PS_SALIDA" ] || falla "quedan procesos de jaxsvc (ps -u jaxsvc): $PS_SALIDA"
command -v awk >/dev/null || falla "no hay awk para recorrer $PROC"
UID_JAXSVC="$(id -u jaxsvc)" || falla "no existe la cuenta jaxsvc"
shopt -s nullglob
ARCHIVOS_PROC=("$PROC"/[0-9]*/status "$PROC"/[0-9]*/task/*/status)
[ "${#ARCHIVOS_PROC[@]}" -gt 0 ] || falla "no se pudo leer $PROC: ningún status de proceso"
# (un proceso que termina mientras se lee puede hacer que awk avise de un archivo que ya no está: se tolera)
HALLADOS="$(awk -v u="$UID_JAXSVC" '/^Uid:/ { for (i = 2; i <= 5; i++) if ($i == u) print FILENAME }' "${ARCHIVOS_PROC[@]}" 2>/dev/null | sort -u || true)"
[ -z "$HALLADOS" ] || falla "/proc muestra procesos o hilos de jaxsvc (alguno de los cuatro uid): $HALLADOS"

# (f) APLICAR
$PERMISOS --aplicar
# (g) VERIFICAR
$PERMISOS --verificar
# (h) al salir, el trap restaura (start en orden inverso, verificado: 'active' estable y NRestarts sin subir).
echo "OK: --aplicar terminó con jaxsvc detenido; el trap restaura las unidades al salir"
```

**Bloque de `--deshacer`** (el mismo, sin `--verificar`: tras `--deshacer` el árbol ya no es el aplicado):
```bash
# BLOQUE-DESHACER
set -euo pipefail
PERMISOS="${PERMISOS:-python3 ops/permisos_proyectos.py}"   # (las pruebas lo sustituyen)
RAIZ="${RAIZ:-/srv/jax-data/jax-workspace}"
PROC="${PROC:-/proc}"
# Lista mínima que TIENE que aparecer (hoy en hall9000): si alguna falta, el descubrimiento está incompleto y se corta.
ESPERADAS="${ESPERADAS:-jax-las-manos.service jax-platform.service jax-ariadna-pm.service jax-ejecutor-proxy.service jax-catalogo-modelos.service jax-limpiar-bases-de-test.service jax-catalogo-modelos.timer jax-limpiar-bases-de-test.timer}"
REINTENTOS="${REINTENTOS:-5}"; ESPERA="${ESPERA:-1}"
# ESTABLE: segundos que una unidad restaurada tiene que seguir 'active', sin que NRestarts suba, para contar como vuelta.
ESTABLE="${ESTABLE:-5}"
falla() { echo "NO CUMPLE: $*" >&2; exit 1; }
case "$ESTABLE" in ''|*[!0-9]*) falla "ESTABLE tiene que ser un entero de segundos: '$ESTABLE'" ;; esac

# (a) PREMISAS: si no se cumplen, el bloque falla antes de tocar nada.
#     Un montaje FUSE (en hall9000, sshfs) da EACCES incluso a root y `find` sale 1. Se toleran SOLO los errores
#     «Permission denied» cuyo path es EXACTAMENTE un punto de montaje fuse* de `findmnt`; cualquier otro error, o un
#     fallo sin mensaje, corta. `find` corre con LC_ALL=C (vía `env`, porque `sudo` puede limpiar el entorno): su mensaje
#     es `find: '/ruta': Permission denied` con comillas ASCII en cualquier idioma, y así se parsea.
#     (`findmnt -r` escapa los espacios del path como \x20: se decodifican para compararlos con lo que imprime find.)
FUSE="$(findmnt -rn -o TARGET,FSTYPE | awk '$2 ~ /^fuse/ {print $1}' | while IFS= read -r m; do printf '%b\n' "$m"; done)" \
  || falla "no se pudo listar los montajes (findmnt)"
ERRF="$(mktemp)" || falla "no se pudo crear un archivo temporal"
SETUID="$(sudo env LC_ALL=C find / "$RAIZ" -xdev \( -path /proc -o -path /sys \) -prune -o -type f -user jaxsvc -perm /6000 -print 2>"$ERRF")" || {
  [ -s "$ERRF" ] || { rm -f "$ERRF"; falla "no se pudo buscar archivos setuid/setgid de jaxsvc: find falló sin mensaje"; }
  while IFS= read -r linea; do
    p="${linea#"find: '"}"; p="${p%"': Permission denied"}"
    if [ "$p" = "$linea" ] || ! printf '%s\n' "$FUSE" | grep -qxF -- "$p"; then
      rm -f "$ERRF"; falla "no se pudo buscar archivos setuid/setgid de jaxsvc: $linea"
    fi
  done < "$ERRF"; }
rm -f "$ERRF"
[ -z "$SETUID" ] || falla "hay archivos setuid/setgid de jaxsvc (podrían volver a darle uid): $SETUID"
CRON="$(sudo crontab -u jaxsvc -l 2>&1 || true)"
case "$CRON" in *"no crontab"*) ;; *) falla "jaxsvc tiene crontab (podría arrancar procesos): $CRON" ;; esac

# (b) LISTA de unidades y timers de jaxsvc. NINGUNA consulta puede fallar en silencio: cada salida se captura y, si la
#     consulta falla, se corta. Un timer, socket o path cuenta por la unidad que dispara (Triggers=).
UNIDADES="$(sudo systemctl list-units --all --plain --no-legend 'jax*')" || falla "falló systemctl list-units"
TEMPORIZADORES="$(sudo systemctl list-timers --all --plain --no-legend 'jax*')" || falla "falló systemctl list-timers"
CANDIDATAS="$({ printf '%s\n' "$UNIDADES" | awk 'NF {print $1}'; \
  printf '%s\n' "$TEMPORIZADORES" | awk '{for (i = 1; i <= NF; i++) if ($i ~ /\.timer$/) print $i}'; } | sort -u)" \
  || falla "no se pudo armar la lista de unidades"
[ -n "$CANDIDATAS" ] || falla "la lista de unidades 'jax*' está vacía (en hall9000 hay al menos las de ESPERADAS)"
TIMERS=(); ACTIVADORES=(); SERVICIOS=()
while read -r u; do
  de_jaxsvc=no
  usuario="$(sudo systemctl show -p User --value "$u")" || falla "falló systemctl show -p User $u"
  [ "$usuario" = jaxsvc ] && de_jaxsvc=si
  if [ "$de_jaxsvc" = no ]; then
    disparadas="$(sudo systemctl show -p Triggers --value "$u")" || falla "falló systemctl show -p Triggers $u"
    for t in $disparadas; do
      usuario_t="$(sudo systemctl show -p User --value "$t")" || falla "falló systemctl show -p User $t"
      [ "$usuario_t" = jaxsvc ] && de_jaxsvc=si
    done
  fi
  [ "$de_jaxsvc" = si ] || continue
  case "$u" in *.timer) TIMERS+=("$u") ;; *.service) SERVICIOS+=("$u") ;; *) ACTIVADORES+=("$u") ;; esac
done <<< "$CANDIDATAS"
LISTA=("${TIMERS[@]}" "${ACTIVADORES[@]}" "${SERVICIOS[@]}")   # los timers se paran primero
[ "${#LISTA[@]}" -gt 0 ] || falla "ninguna unidad de jaxsvc en la lista: el descubrimiento falló"
for esperada in $ESPERADAS; do
  encontrada=no
  for u in "${LISTA[@]}"; do [ "$u" = "$esperada" ] && encontrada=si; done
  [ "$encontrada" = si ] || falla "falta la unidad esperada de jaxsvc $esperada: el descubrimiento está incompleto"
done

# ESTADO de CADA unidad ANTES de detener nada, por su salida textual (no solo por el código de salida).
ESTADOS=()
for u in "${LISTA[@]}"; do
  estado="$(sudo systemctl is-active "$u" || true)"
  case "$estado" in active|inactive|failed|activating) ;; *) falla "estado inesperado de $u: '$estado'" ;; esac
  ESTADOS+=("$estado")
done

# (c) TRAP de restauración (EXIT): arranca en orden INVERSO solo las que estaban 'active' o 'activating' y exige que
#     vuelvan a 'active' (con unos pocos reintentos) Y ESTABLES: siguen 'active' durante ESTABLE segundos y NRestarts
#     (propiedad de los .service; un .timer no la tiene) no sube en ese intervalo. Una unidad en bucle de reinicios se
#     ve 'active' o 'activating' un instante: eso no cuenta. Si alguna no vuelve, lo dice, explica cómo arrancarla y sale
#     con código distinto de 0 AUNQUE la aplicación haya ido bien. Conserva el código del fallo original si lo hubo.
restaurar() {
  local rc=$? i u intento t n fallidas=() NR0=() MALA=() MOTIVO=()      # rc del fallo real (o 0)
  trap - EXIT
  set +e
  for ((i = ${#LISTA[@]} - 1; i >= 0; i--)); do
    [ "${ESTADOS[i]}" = active ] || [ "${ESTADOS[i]}" = activating ] || continue
    sudo systemctl start "${LISTA[i]}" || echo "AVISO: systemctl start ${LISTA[i]} falló" >&2
  done
  # 1) cada una tiene que llegar a 'active' (reintentos); entonces se anota su NRestarts de partida
  for ((i = ${#LISTA[@]} - 1; i >= 0; i--)); do
    [ "${ESTADOS[i]}" = active ] || [ "${ESTADOS[i]}" = activating ] || continue
    u="${LISTA[i]}"
    for ((intento = 0; intento < REINTENTOS; intento++)); do
      [ "$(sudo systemctl is-active "$u")" = active ] && break
      sleep "$ESPERA"
    done
    if [ "$(sudo systemctl is-active "$u")" != active ]; then MALA[i]=1; MOTIVO[i]="no llegó a 'active'"; continue; fi
    n="$(sudo systemctl show -p NRestarts --value "$u")" || { MALA[i]=1; MOTIVO[i]="no se pudo leer NRestarts"; continue; }
    case "$n" in
      *[!0-9]*) MALA[i]=1; MOTIVO[i]="NRestarts ilegible: '$n'"; continue ;;
      '') case "$u" in *.service) MALA[i]=1; MOTIVO[i]="NRestarts vacío"; continue ;; esac ;;
    esac
    NR0[i]="$n"
  done
  # 2) ventana de estabilidad: ESTABLE segundos, todas a la vez; una que deja de estar 'active' queda marcada
  for ((t = 0; t < ESTABLE; t++)); do
    sleep 1
    for ((i = ${#LISTA[@]} - 1; i >= 0; i--)); do
      [ "${ESTADOS[i]}" = active ] || [ "${ESTADOS[i]}" = activating ] || continue
      [ -z "${MALA[i]:-}" ] || continue
      [ "$(sudo systemctl is-active "${LISTA[i]}")" = active ] || { MALA[i]=1; MOTIVO[i]="dejó de estar 'active' en la ventana de ${ESTABLE} s"; }
    done
  done
  # 3) NRestarts no tiene que haber subido
  for ((i = ${#LISTA[@]} - 1; i >= 0; i--)); do
    [ "${ESTADOS[i]}" = active ] || [ "${ESTADOS[i]}" = activating ] || continue
    [ -z "${MALA[i]:-}" ] || continue
    [ -n "${NR0[i]:-}" ] || continue
    n="$(sudo systemctl show -p NRestarts --value "${LISTA[i]}")" || n=""
    [ "$n" = "${NR0[i]}" ] || { MALA[i]=1; MOTIVO[i]="NRestarts pasó de ${NR0[i]} a '${n}' (bucle de reinicios)"; }
  done
  for ((i = ${#LISTA[@]} - 1; i >= 0; i--)); do
    [ -z "${MALA[i]:-}" ] || fallidas+=("${LISTA[i]}")
  done
  if [ "${#fallidas[@]}" -gt 0 ]; then
    echo "ERROR: no volvieron a 'active' estable (${ESTABLE} s, sin subir NRestarts): ${fallidas[*]}" >&2
    for ((i = ${#LISTA[@]} - 1; i >= 0; i--)); do
      [ -z "${MALA[i]:-}" ] || echo "  ${LISTA[i]}: ${MOTIVO[i]}" >&2
    done
    for u in "${fallidas[@]}"; do echo "  arrancarla a mano: sudo systemctl start $u" >&2; done
    [ "$rc" -ne 0 ] || rc=1
  fi
  exit "$rc"
}
trap restaurar EXIT

# VENTANA SIN SEÑALES: desde justo antes del primer stop (con el trap EXIT ya armado) INT, TERM y HUP se IGNORAN hasta
# el final, incluidos el guion y la restauración (los procesos hijos heredan el ignorar). Así no hay ventanas entre
# manejadores: el rc es el del fallo real o 0.
A_ARRANCAR=""
for ((i = ${#LISTA[@]} - 1; i >= 0; i--)); do
  if [ "${ESTADOS[i]}" = active ] || [ "${ESTADOS[i]}" = activating ]; then A_ARRANCAR="$A_ARRANCAR ${LISTA[i]}"; fi
done
echo "AVISO: durante la ventana el bloque no se interrumpe con Ctrl-C; si hace falta cortarlo, kill -9 y después: sudo systemctl start${A_ARRANCAR}" >&2
trap '' INT TERM HUP

# (d) PARAR cada timer y unidad (sin mask: las unidades viven en /etc/systemd/system y un mask --runtime no las tapa;
#     con los timers parados no se disparan, y arrancarlos solo puede hacerlo root, que es confiable por premisa).
for u in "${LISTA[@]}"; do sudo systemctl stop "$u" || falla "no se pudo detener $u"; done

# (e) COMPROBAR: ninguna activa (por su salida textual), `ps -u jaxsvc` vacío y /proc sin los cuatro uid de jaxsvc.
for u in "${LISTA[@]}"; do
  estado="$(sudo systemctl is-active "$u" || true)"
  case "$estado" in inactive|failed) ;; *) falla "$u no quedó detenida (estado: '$estado')" ;; esac
done
PS_SALIDA="$(ps -u jaxsvc -o pid= 2>&1)" || { rc_ps=$?; [ "$rc_ps" -eq 1 ] || falla "ps -u jaxsvc falló (rc $rc_ps)"; }
[ -z "$PS_SALIDA" ] || falla "quedan procesos de jaxsvc (ps -u jaxsvc): $PS_SALIDA"
command -v awk >/dev/null || falla "no hay awk para recorrer $PROC"
UID_JAXSVC="$(id -u jaxsvc)" || falla "no existe la cuenta jaxsvc"
shopt -s nullglob
ARCHIVOS_PROC=("$PROC"/[0-9]*/status "$PROC"/[0-9]*/task/*/status)
[ "${#ARCHIVOS_PROC[@]}" -gt 0 ] || falla "no se pudo leer $PROC: ningún status de proceso"
# (un proceso que termina mientras se lee puede hacer que awk avise de un archivo que ya no está: se tolera)
HALLADOS="$(awk -v u="$UID_JAXSVC" '/^Uid:/ { for (i = 2; i <= 5; i++) if ($i == u) print FILENAME }' "${ARCHIVOS_PROC[@]}" 2>/dev/null | sort -u || true)"
[ -z "$HALLADOS" ] || falla "/proc muestra procesos o hilos de jaxsvc (alguno de los cuatro uid): $HALLADOS"

# (f) DESHACER
$PERMISOS --deshacer
# (g) no hay --verificar tras --deshacer: el árbol ya no es el aplicado (dueño fruiz:fruiz, sin la ACL de fruiz)
# (h) al salir, el trap restaura (start en orden inverso, verificado: 'active' estable y NRestarts sin subir).
echo "OK: --deshacer terminó con jaxsvc detenido; el trap restaura las unidades al salir"
```

Después del bloque de `--aplicar`, la verificación independiente (la de abajo). Si el guion dice «hay procesos de jaxsvc vivos (pids …)», se detienen esas unidades y se repite; un proceso que aparece durante la mutación se anota y el comando sale con 1.
Hacerlo desde un checkout cuyo HEAD tenga el guion commiteado (el núcleo se compara contra `git show HEAD:ops/permisos_proyectos.py`). **Requisito de `workspace-proyectos.md`:** el `fchmod(0o660)` de `tool_authority.py` tiene que estar ya en `/srv/jax-prod` (paso 3 de esta secuencia lo despliega; si no está, `--aplicar` NO se corre: ver «Aplicación en producción: BLOQUEADA» en ese runbook). Verificación: `--verificar` sale `0`; luego la **prueba real como `jaxsvc`** de ese runbook (paso 6: escritura cruzada `jaxsvc`/`fruiz` en un subdirectorio propio) en verde. Reversión: `python3 ops/permisos_proyectos.py --deshacer` (no reabre a otros; ver más abajo).
**Sin acceso para «otros» (spec madre §5: 2770 en directorios, 0660 en archivos).** `--aplicar` deja `other::---` en el modo y en la ACL de acceso y por defecto de todo `proyectos/`, y quita los bits de otros a la **raíz del workspace** (el padre de `proyectos/`, hoy `/srv/jax-data/jax-workspace`, `fruiz:jaxsvc` 770) sin cambiar su dueño ni su grupo. `--verificar` cuenta como NO CUMPLE: cualquier bit de otros en `proyectos/`, debajo y en esa raíz; que `jaxsvc` o `fruiz` no puedan atravesar la raíz; y **cualquier entrada ACL nombrada (usuario o grupo) que no sea de `jaxsvc` ni de `fruiz`**, en la raíz o en el árbol.
**La raíz del workspace es `fruiz:jaxsvc` 770** (el setgid solo si ya lo tiene): `--aplicar` la deja en `0770` sin cambiar dueño ni grupo, `--verificar` exige `770 fruiz:jaxsvc`, y si el dueño o el grupo no coinciden `--aplicar` **falla cerrado antes de mutar** (este guion no cambia dueños de la raíz). Un `setfacl -m` manual sobre la raíz recalcula la máscara y la deja en `0750`: `--verificar` lo marca. **Un archivo gobernado es exactamente `0660`**: `--verificar` marca cualquier bit de ejecución (modo o ACL efectiva). Las rutas con caracteres de control en el nombre se muestran **escapadas** (`\n`) y para ellas no se imprime ninguna orden copiable: se renombra la carpeta a mano. Si el núcleo se interrumpe o falla a medio mutar (incluido Ctrl-C), imprime `a_medio_aplicar`, la última ruta y la instrucción de volver a correrlo (es idempotente).
**`--aplicar` falla cerrado, sin cambiar nada**, en dos casos, antes de tocar un solo objeto (pasada de solo lectura): (1) `jaxsvc` o `fruiz` no podrían atravesar la raíz sin el bit de otros. Hoy `fruiz` entra a la raíz **como dueño** y `jaxsvc` **por el grupo** `jaxsvc` (ninguno está en el grupo del otro): una raíz `fruiz:fruiz 0755`, que es lo que deja un `mkdir` con umask 022, dejaría fuera a `jaxsvc`, y `--deshacer` restaura el modo desde el respaldo pero no arregla un dueño o grupo equivocado. (2) Hay una entrada ACL nombrada ajena: `--aplicar` **no la borra** (ni la amplía con la máscara): la lista y una persona decide si se quita (`setfacl -x u:<nombre> <ruta>`) o si es legítima. En ambos casos el mensaje dice qué objeto y qué cuenta; se corrige y se repite.
**Límite conocido (archivos 0600):** un archivo que su dueño crea con modo `0600` es una decisión de ese proceso, y la máscara heredada de la ACL por defecto lo respeta: el otro lado (`jaxsvc` o `fruiz`) no lo puede abrir, tanto después de `--aplicar` como después de `--deshacer`. No es un fallo del guion. Para corregir un caso se vuelve a correr `--aplicar` (deja `0660` con las dos cuentas en la ACL) o se hace `chmod 0660` sobre ese archivo. Las herramientas que escriben en `proyectos/` deben crear con `0660` (el escritor de LAS MANOS ya hace `fchmod 0660`).
**Efecto a saber antes de aplicar:** los archivos pasan a `0660` y pierden cualquier bit de ejecución que tuvieran (medido por la auditoría: 290 de 292 archivos lo tenían, todos por la máscara `rwx` que dejaba el guion viejo y ninguno de dueño).
**Verificación tras aplicar** (los dos puntos se hacen cumplir en el bloque; si algo no da, imprime `NO CUMPLE` y sale con código distinto de 0, y no se sigue): (a) `--verificar` da 0 (exige, entre otras cosas, la raíz `770 fruiz:jaxsvc`); (b) como root, `getfacl -R -p <raíz>/proyectos` mostrado solo para las líneas `other::` y `default:other::` tiene que dar **únicamente** `other::---` y `default:other::---`.
```bash
bash <<'VERIFICACION'
set -euo pipefail
RAIZ="${RAIZ:-/srv/jax-data/jax-workspace}"   # por defecto la JAX_WORKSPACE_DIR real; las pruebas la sustituyen
python3 ops/permisos_proyectos.py --verificar "$RAIZ" || { echo "NO CUMPLE: --verificar no dio 0" >&2; exit 1; }
OTROS=$(sudo getfacl -R -p "$RAIZ/proyectos" | grep -E '^(default:)?other::' | sort -u)
ESPERADO=$(printf 'default:other::---\nother::---')
if [ "$OTROS" != "$ESPERADO" ]; then
  echo "NO CUMPLE: other en proyectos/ no es solo ---:" >&2; echo "$OTROS" >&2; exit 1
fi
echo "OK: --verificar en 0 (incluida la raiz 770 fruiz:jaxsvc) y other cerrado en todo proyectos/"
VERIFICACION
```
Nota: **una lectura como `nobody` no se usa como prueba**. Mientras la raíz esté en 770, `nobody` no la atraviesa y esa lectura falla aunque `proyectos/` siguiera abierto: una prueba que no puede fallar no valida nada. Darle a `nobody` una entrada temporal para que sí pueda fallar abre los documentos de los clientes mientras dura la prueba, así que tampoco se hace: la verificación (b) mide directamente lo que importa, los bits de otros. (Y `test -r` no sirve como prueba de permisos en este host: el `test` de uutils mira solo los bits del modo, no las ACL.) **Carpetas ocultas** (`proyectos/<proyecto>/.<nombre>/`, estado de herramientas): **este guion NO las toca nunca**, ni a ellas ni a su contenido (ningún `chmod`, `chown` ni `setfacl` como root: un inode que `jaxsvc` pueda enlazar desde fuera sería una carrera). Solo las **mira**: si alguna tiene un bit de otros (modo, `other::` de acceso o, en directorios, por defecto) o un archivo con más de un enlace duro, `--verificar` la marca NO CUMPLE y **`--aplicar` y `--deshacer` fallan cerrado antes de mutar nada**, con cada ruta y la orden de corrección. **La corrección es manual: la ejecuta una persona**, con la oculta a la vista:
```bash
set -euo pipefail
OCULTA="${OCULTA:?asignar OCULTA con la ruta tal como la imprimió el guion (ya citada), sin retipearla}"
sudo chmod -R o-rwx -- "$OCULTA"
sudo find "$OCULTA" -type d -exec setfacl -d -m o::--- {} +     # solo si el aviso habla de la ACL por defecto
```
**No se retipea la ruta ni se pega a mano**: el nombre de una oculta lo controla quien la crea y puede llevar comillas, `$(...)` o `;`. El guion imprime la orden con la ruta ya citada con `shlex.quote`: se **asigna a la variable `OCULTA` con esa misma forma de citar** (`OCULTA='...'` tal cual salió, o con `printf %q`) y las órdenes usan siempre `"$OCULTA"` entre comillas dobles y `--` antes de la ruta. Si la orden impresa por el guion tiene `'` o `$` en la ruta, leerla entera antes de ejecutar nada, como root.
Un hardlink no se corrige con `chmod` (es el mismo inode que otra ruta, quizá fuera de `proyectos/`): se averigua quién lo enlaza (`sudo find / -xdev -samefile <archivo>`) y se quita el enlace de la oculta. Una oculta limpia (sin bits de otros ni hardlinks) no detiene a nada y no se modifica: la mutación compara por inode cada entrada abierta con la enumerada y no toca ningún inode que las pasadas previas vieron dentro de una oculta (aunque alguien intercambie nombres a mitad de la corrida; se anota «la entrada cambió durante el recorrido» y el comando sale con 1), y al final relee las ocultas y marca NO CUMPLE si su dueño, grupo o ACL cambiaron. El bloque de arriba las incluye porque `getfacl -R` las recorre.
**Después de cualquier traslado del workspace**, volver a medir la cadena completa con `namei -l <archivo de proyectos/>`: cada tramo, desde `/`, tiene que ser lo que se espera (sin bit de otros en la raíz del workspace ni abajo, y `fruiz` y `jaxsvc` con paso). El 2026-10-03 el traslado a `/srv/jax-data` quitó la barrera que daba `/home/fruiz` (750) y nadie lo notó.
**`--deshacer` devuelve el dueño a `fruiz:fruiz`, conserva a `jaxsvc` y NUNCA reabre a otros.** Deja `proyectos/` con dueño y grupo `fruiz:fruiz` (lo de antes de E2a), pero **conserva** `u:jaxsvc:rwx` (`rw-` en archivos) con su máscara, en la ACL de acceso y en la por defecto, y `other::---`: modo `0770` en directorios y `0660` en archivos, y `default:other::---` en los directorios (lo que se cree después nace cerrado). Así LAS MANOS (`jaxsvc` no es del grupo `fruiz`) sigue operando y nadie más entra; se quita solo la entrada nombrada de `fruiz`, que ya es dueño y grupo. De la raíz restaura solo dueño y grupo del modo que guardó el respaldo forense más reciente de confianza (`/var/backups/jax-permisos/`, que registra modo y ACL de la raíz); si ese modo tenía bits de otros **no los restaura y lo dice**. **Antes de mutar nada, `--deshacer` calcula si `jaxsvc` y `fruiz` atravesarían la raíz** con el modo que va a restaurar y el dueño y grupo actuales; si no, **falla cerrado sin tocar nada** y lo explica (p. ej. la raíz pasó a `fruiz:fruiz` sin ACL de `jaxsvc`: se corrige la raíz y se repite). Al terminar verifica de nuevo y solo imprime `OK` si los dos la atraviesan; si no, sale con código 1. Si no hay respaldo válido lo dice («Raíz del workspace: NO restaurada») y no la toca. Las carpetas ocultas no las toca (ver más arriba): si tienen bits de otros, falla cerrado antes de mutar. Un `--deshacer` es una reversión de propiedad, nunca una reapertura: si de verdad hiciera falta que otros lean `proyectos/`, es una decisión de Fernando y se hace a mano, no con este guion.
**Hay que reinstalar el núcleo privilegiado.** Este cambio modifica `ops/permisos_proyectos.py`, que es el núcleo (`--nucleo-privilegiado` corre ese mismo archivo). El instalado en `/usr/local/sbin/jax-permisos-proyectos` queda desfasado respecto de HEAD: `--verificar` lo avisa y `--aplicar` se niega hasta reinstalar. Lo hace root, desde un checkout con el cambio ya integrado en `master`, con la orden de arriba (`sudo install -o root -g root -m 0755 ops/permisos_proyectos.py /usr/local/sbin/jax-permisos-proyectos`) y se comprueba con `sha256sum`. La regla de sudoers no cambia (mismas tres órdenes).
> **Orden.** Si el despliegue de jax (paso 3) es el que trae el `fchmod`, se hace el paso 3 antes del `--aplicar` de este paso. Los demás pasos de este runbook no dependen de ello.
### 3. jax a producción (LAS MANOS con `project_uuid`)
Con el procedimiento de `docs/runbooks/despliegue.md` de jax-platform (sección jax). Reiniciar `jax-las-manos`. Verificación: un `POST /procesamiento/trabajos` con un `project_uuid` inexistente responde **422** con `proyecto_no_activo`; `systemctl is-active jax-las-manos` da `active`; **el arreglo de la ingesta tiene que estar desplegado** (`grep -c _origen_ya_en_fuente /srv/jax-prod/jax/procesamiento/ingesta.py` da `1` o más): sin él, procesar un archivo de LACTOVI que ya vive en `fuente/sub/` lo **copiaría** a la raíz de `fuente/` (duplica datos del cliente, cambia los sha y cierra la reversión), y el paso 5 no se corre; `/proc/<pid>/cwd` del servicio apunta a `/srv/jax-prod/jax` (un servicio sirve desde un checkout: se verifica, no se supone).
### 4. jax-platform a producción
Una sola vez, con los comandos de `despliegue.md`. Al arrancar, `run_migrations()` aplica la 006a y siembra las cuatro claves de `axioma_config` de la Tarea 5. Verificación (consulta propia):
```sql
SHOW CREATE TABLE project_documents;                    -- existe, con uq_project_documents_sha y las 3 FKs
SELECT config_key, config_value FROM axioma_config WHERE config_key LIKE 'proyectos.documentos.%';   -- las 4 claves
```
Si la tabla no está, mirar el log de arranque (falla de `run_migrations`) y parar: no se crea a mano.
**Precondiciones de la subida (solo verificar; cambiar nginx es de la Parte C, con GO):** `POST /api/proyectos/*/documentos` recibe lotes de hasta `proyectos.documentos.max_bytes_lote` (1 GiB por defecto).
```bash
sudo nginx -T 2>/dev/null | grep -n "client_max_body_size"    # en el server/location que sirve /api: >= max_bytes_lote + margen (p. ej. 1100m); si falta, nginx corta en 1m con 413
```
```sql
SELECT config_value FROM axioma_config WHERE config_key = 'proyectos.documentos.max_bytes_lote';   -- el tope a cubrir
```
```bash
systemctl show jax-platform -p Environment -p PrivateTmp      # buscar TMPDIR; sin TMPDIR, Starlette usa /tmp (con PrivateTmp=yes, el /tmp propio del servicio)
df -h /tmp                                                    # o el TMPDIR hallado: espacio libre >= max_bytes_lote
```
Starlette vuelca el multipart a disco **antes** de que la plataforma cuente bytes, así que un lote grande ocupa ese TMPDIR aunque luego se rechace con 413. Si `client_max_body_size` no cubre el tope o el TMPDIR no tiene espacio, no se abre la subida a usuarios: se anota para la Parte C. Publicar el sitio (frontend) según `despliegue.md` y comprobar que sirve el bundle nuevo.
### 5. LACTOVI: de carpeta suelta a proyecto
Rutas reales: workspace `/srv/jax-data/jax-workspace`, carpeta `lacteos-victoria`. Antes: `find .../proyectos/lacteos-victoria/fuente -type f | wc -l` (en el ensayo del 2026-10-03: 120) y `ls .../procesado | wc -l` (88) (conteos del ensayo medidos sobre la ruta anterior al traslado del 2026-10-03, `/home/fruiz/jax-workspace`), y se anota. **Mismo sistema de archivos**: `os.rename` entre `proyectos/lacteos-victoria` y `proyectos/<uuid>` es dentro del mismo directorio.
```bash
# 5a. Ensayo (no escribe nada: solo lee disco y base)
e2a -m scripts.proyectos_e2a_lactovi --workspace /srv/jax-data/jax-workspace --carpeta lacteos-victoria \
  --nombre "Lácteos Victoria" --dueno-user-id <ID> --tenant-id 1 --database jax_memory | tee $D/lactovi-ensayo.json
```
Revisar: `archivos_fuente` (120), `fichas` (88), `filas_a_insertar` (una por sha256 distinto: ≤ 120; la diferencia son `duplicados_sha`), `estados` (cuántas filas `listo`/`parcial`/`error`/`sin_extractor`/`en_cola`), `ignorado` (debe listar la carpeta oculta de estado de herramientas, que viaja con el rename y no se registra), `ignorados` (symlinks y no-regulares de **cualquier** nivel de `fuente/`: no se hashean ni se registran; cada uno se revisa a mano), `fichas_sin_archivo` (idealmente vacío), `mapa_previsto`. Si algo no es lo esperado, parar.
### 5a2. Respaldo de la carpeta ANTES de moverla (sin esto no se aplica)
El respaldo del paso 1 es de la base. La carpeta `lacteos-victoria` (datos del cliente) necesita el suyo, con restauración probada, antes del `rename`. Se usa el mismo mecanismo que `backup-hall9000.sh` (Step 5b: `restic backup --host hall9000 --one-file-system` contra `/etc/restic/local.env` y `/etc/restic/r2.env`, **como `fruiz`, nunca como root**: dejaría packs de root que el prune de `fruiz` no puede borrar), con una etiqueta propia. Un restic largo **no va en primer plano**: se lanza con `nohup` y se espera su `rc`.
```bash
ORIG=/srv/jax-data/jax-workspace/proyectos/lacteos-victoria
for ENV in /etc/restic/local.env /etc/restic/r2.env; do
  [ -f "$ENV" ] || { echo "FALTA $ENV: no se aplica"; continue; }
  N=$(basename "$ENV" .env)
  rm -f "$D/restic-pre-$N.rc"
  # EN SEGUNDO PLANO (&): la shell vuelve al instante; el log y el codigo de salida quedan en $D.
  nohup bash -c "set -uo pipefail; source '$ENV'; restic backup --tag e2a-lactovi-pre --host hall9000 --one-file-system '$ORIG'; echo \$? > '$D/restic-pre-$N.rc'" \
    > "$D/restic-pre-$N.log" 2>&1 &
  echo $! > "$D/restic-pre-$N.pid"; echo "$N lanzado, pid $!"
done
# Esperar el fin de los dos: cada uno escribe su .rc al terminar. Con TOPE de tiempo (TOPE_S, 2 h por defecto)
# y comprobando que el proceso sigue vivo: si muere sin dejar .rc, se sale con error claro en vez de colgarse.
esperar_restics() {
local TOPE_S=${TOPE_S:-7200} T0=$SECONDS PENDIENTE ENV N
while :; do
  PENDIENTE=0
  for ENV in /etc/restic/local.env /etc/restic/r2.env; do
    [ -f "$ENV" ] || continue; N=$(basename "$ENV" .env)
    [ -f "$D/restic-pre-$N.rc" ] && continue
    if kill -0 "$(cat "$D/restic-pre-$N.pid" 2>/dev/null)" 2>/dev/null; then PENDIENTE=1
    # Carrera: el proceso pudo terminar y escribir su .rc ENTRE el test de arriba y este kill -0. Se vuelve a mirar el .rc antes de declararlo muerto.
    elif [ -f "$D/restic-pre-$N.rc" ]; then :
    else echo "ERROR: el respaldo $N murio SIN dejar .rc: mirar $D/restic-pre-$N.log; NO se sigue" >&2; return 1; fi
  done
  [ "$PENDIENTE" = 0 ] && break
  [ $((SECONDS-T0)) -ge "$TOPE_S" ] && { echo "ERROR: pasaron ${TOPE_S}s sin que terminen los respaldos: mirar tail -f $D/restic-pre-*.log; NO se sigue" >&2; return 1; }
  sleep 20
done
}
# Lo que sigue va DENTRO del if: si la espera falla, el resumen no corre y el bloque pegado termina ahi.
if esperar_restics; then
  tail -n 3 "$D"/restic-pre-*.log      # el resumen de cada snapshot
  for f in "$D"/restic-pre-*.rc; do echo "$f -> $(cat "$f")"; done
else
  echo 'ESPERA FALLIDA: NO SE SIGUE. No pegues nada de lo que viene despues de este bloque hasta revisar los logs.' >&2
fi
```
**Si salio `ESPERA FALLIDA`, el procedimiento se detiene aqui** (no se restaura ni se pasa al paso siguiente): un `|| echo` solo no corta lo que se pegue despues, por eso el resumen va dentro del `if`.
Los dos repos son obligatorios si existen los dos `.env`. Cada `.rc` tiene que ser `0` (con `3` hubo snapshot pero algún archivo no se leyó: no se sigue hasta saber cuál). El bucle sale con error si un proceso muere sin `.rc` o si pasa `TOPE_S` (se puede subir exportándolo antes); con cualquiera de los dos **no se sigue**: mirar `tail -f "$D"/restic-pre-*.log` desde otra terminal. Después, **por cada repo**, restaurar a una ruta aparte y comparar `sha256` contra el original:
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
e2a -m scripts.proyectos_e2a_lactovi --workspace /srv/jax-data/jax-workspace --carpeta lacteos-victoria \
  --nombre "Lácteos Victoria" --dueno-user-id <ID> --tenant-id 1 --database jax_memory \
  --aplicar --confirmo-produccion | tee $D/lactovi-aplicar.json
mv /srv/jax-data/jax-workspace/proyectos/.e2a-lactovi-*.json $D/     # el mapa de reversión, a salvo y FUERA de proyectos/
```
Los archivos sin ficha que el extractor real acepta (`procesamiento.compuerta.tiene_extractor`: por extensión —`pdf xlsx xlsm docx png jpg jpeg tif tiff bmp webp`— o por contenido, un PDF o imagen con otro nombre) entran como `en_cola` con `ruta_entrada = proyectos/<uuid>/fuente/<ruta>`: el despachador de la plataforma los manda a procesar y la ingesta, al ver que el archivo **ya está dentro de `fuente/`**, lo procesa en el lugar sin copiarlo (arreglo `_origen_ya_en_fuente`, paso 3). Los de otro tipo entran como `sin_extractor`, sin `ruta_entrada`. El mapa se **mueve** (`mv`, no `cp`) a `$D`: una copia dejaría un `.e2a-lactovi-*.json` dentro de `proyectos/`, que `ops/permisos_proyectos.py --verificar` no reconoce y hace fallar; `--revertir` y `--completar` usan el de `$D`. **Comprobación tras el despacho:** `find /srv/jax-data/jax-workspace/proyectos/<uuid>/fuente -type f | wc -l` **tiene que ser igual al de antes (120), ni más ni menos**: más es que hay copias (se detiene el despachador); menos es que alguien borró un original (se detiene todo y se escala a Fernando: la regla es que nada borra bajo `fuente/`).
Códigos de salida: `0` hecho; `1` el mapa o el destino ya existen, el proyecto no está ACTIVO, o falló el registro (se deshizo el rename, el proyecto queda creado y ACTIVO; se revisa antes de reintentar); `2` argumentos, guarda de producción o carpeta que no existe (p. ej. **ya se movió**: no crea otro proyecto); `3` los `sha256` no cuadran tras mover (se deshizo el rename); `4` el commit de las filas tiene desenlace desconocido (**el disco no se tocó**: verificar las filas y, si faltan, `--completar`); `5` error no previsto: **no reintentar sin revisar**; `6` (`--revertir`/`--completar`) el mapa no cuadra con la base; `7` (`--revertir`) hay trabajos `pendiente`/`procesando`.

**Qué hacer en cada corte** (el disco y la base se miran antes de cualquier reintento):

| Dónde se cortó | Estado que queda | Qué hacer |
|---|---|---|
| Tras `create_project`, antes del mapa | proyecto ACTIVO, carpeta sin mover, sin mapa | Volver a correr `--aplicar` (misma llave: reutiliza el proyecto) |
| Mapa escrito, `rename` falló o no ocurrió | mapa existe, carpeta en su sitio | Mover el mapa FUERA de `proyectos/`, igual que en el caso de éxito (`mv /srv/jax-data/jax-workspace/proyectos/.e2a-lactovi-*.json $D/`; el guion no pisa uno existente y uno dentro de `proyectos/` hace fallar `--verificar`) y volver a `--aplicar` |
| Tras el `rename`, antes de las filas (corte del proceso) | carpeta en `<uuid>`, sin filas | `--completar <mapa>` (verifica `sha256` contra el mapa y registra lo que falte) |
| Commit de las filas incierto (código 4) | carpeta en `<uuid>`, filas quizá | Contar filas (`SELECT COUNT(*)`); si faltan, `--completar` (idempotente) |
| Fallo al registrar con error previo al commit (código 1) | el guion devolvió la carpeta | Mover el mapa a `$D` (fuera de `proyectos/`) y volver a `--aplicar` |
| `--revertir` cortado antes del commit | nada cambió o se deshizo solo | Repetir `--revertir` |
| `--revertir` cortado en el archivado | carpeta devuelta, sin filas, proyecto ACTIVO | Repetir `--revertir`: detecta la carpeta ya devuelta y solo archiva |

Tras un `--revertir` el proyecto queda ARCHIVADO: para volver a aplicar hay que restaurarlo (`RESTORE_PROJECT`, ARCHIVED → ACTIVE) desde la plataforma; el mapa viejo ya está en `$D` (fuera de `proyectos/`), y si quedara uno en `proyectos/`, renombrarlo; `--aplicar` se niega con ese mensaje si no.
**Verificación independiente** (no con la salida del guion):
```sql
SELECT estado, COUNT(*) FROM project_documents WHERE project_id = <project_id> GROUP BY estado;
SELECT COUNT(*) FROM project_documents WHERE project_id = <project_id>;      -- = filas_insertadas
SELECT project_uuid FROM projects WHERE id = <project_id>;                   -- = el nombre de la carpeta nueva
```
```bash
ls -d /srv/jax-data/jax-workspace/proyectos/<uuid>        # existe
ls -d /srv/jax-data/jax-workspace/proyectos/lacteos-victoria 2>&1   # ya no existe
cd /srv/jax-data/jax-workspace/proyectos/<uuid>/fuente && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
```
Y el último hash se compara con el mismo cálculo hecho en el paso de «antes» sobre la carpeta vieja (anotarlo antes de 5b). La pestaña Documentos de LACTOVI en la plataforma tiene que listar los documentos.
**Reversión** (solo antes de que la plataforma reciba subidas a LACTOVI; si no, los `sha256` no cuadran y sale `3` sin tocar nada). **Hay que detener los DOS servicios, en este orden, y por esta razón:** con un trabajo en vuelo, la ingesta recrea `proyectos/<uuid>/fuente/` (`mkdir(exist_ok=True)`) después de que el guion devolvió la carpeta, y la deja partida. *jax-platform* es quien **despacha** (pasa filas a `pendiente` y manda trabajos); *jax-las-manos* es quien **ejecuta**: un trabajo ya entregado a LAS MANOS sigue escribiendo aunque la plataforma esté parada y aunque la fila ya no diga `procesando`. Detener solo uno deja un escritor vivo. El guion se niega (código 7) si alguna fila está `pendiente` o `procesando`, pero esa comprobación no ve lo que LAS MANOS ya tiene en la mano: no sustituye detener los servicios.
`CNF` es el archivo de opciones `600` del §0 de `despliegue.md` (nunca `-p"$JAX_DB_PASSWORD"`); `PID` es el `project_id` de `$D/lactovi-aplicar.json`.
```bash
Q="SELECT COUNT(*) FROM project_documents WHERE project_id = $PID AND estado IN ('en_cola','pendiente','procesando')"
# (1) Con la plataforma VIVA, esperar a que el proyecto tenga 0 filas en vuelo (sin pasos nuevos de subida).
until [ "$(mariadb --defaults-extra-file="$CNF" -N -e "$Q")" = "0" ]; do sleep 10; done
# (2) Detener la plataforma (el despachador).
sudo systemctl stop jax-platform
systemctl is-active jax-platform                  # tiene que decir: inactive
# (3) Detener LAS MANOS (ningun trabajo ya entregado puede quedar escribiendo).
sudo systemctl stop jax-las-manos
systemctl is-active jax-las-manos                 # tiene que decir: inactive
# (4) Volver a contar, ya con los dos parados: tiene que dar 0.
mariadb --defaults-extra-file="$CNF" -N -e "$Q"
# (5) Revertir.
e2a -m scripts.proyectos_e2a_lactovi --revertir $D/.e2a-lactovi-<fecha>.json --database jax_memory --confirmo-produccion
# (6) Arrancar los dos servicios y verificar.
sudo systemctl start jax-las-manos jax-platform
systemctl is-active jax-las-manos jax-platform    # active, active
```
Borra las filas de `project_documents`, devuelve la carpeta a `lacteos-victoria` verificando `sha256`, y deja el proyecto **ARCHIVADO** (no se borra: Destruir no existe). Verificar: la carpeta vieja existe y `<uuid>` no; 0 filas; alcance ARCHIVED. Es reintentable (con los servicios aún parados). **Verificación del paso 6:** `systemctl is-active jax-las-manos jax-platform` da `active` las dos veces; el chequeo de salud vigente de `despliegue.md` responde, y `ls -d /srv/jax-data/jax-workspace/proyectos/lacteos-victoria/fuente` existe sin que haya aparecido `proyectos/<uuid>/`. Si el guion salió con 7, no se tocó nada: arrancar los servicios, esperar y repetir desde (1).
### 6. Restauración probada de `proyectos/` con la ruta nueva
**Sin esto, la subida no se anuncia como disponible.** Esperar el snapshot de restic posterior al traslado (o lanzar uno con el procedimiento de respaldo vigente; un restic largo nunca en primer plano), restaurar solo `proyectos/<uuid>/` a una ruta aparte (`restic restore <snapshot> --target /tmp/e2a-restauracion --include /srv/jax-data/jax-workspace/proyectos/<uuid>`), y comparar el `sha256` de cada archivo de `fuente/` contra el `sha256` de las fichas/`project_documents`. Cualquier diferencia o archivo ausente: la subida NO se anuncia y se escala a Fernando. Borrar la ruta de ensayo al terminar.
### 7. Prueba de humo
Fernando sube 3 documentos a LACTOVI desde el Chat y los ve pasar a `listo`. Verificación: `SELECT nombre_original, estado FROM project_documents WHERE project_id = <project_id> ORDER BY id DESC LIMIT 3;`. La ingesta copia cada original a **`proyectos/<uuid>/fuente/`** y la plataforma borra el de `entrada/` al terminar: comprobar que los 3 archivos están en `fuente/` con dueño `jaxsvc` y grupo `fruiz` (`stat -c '%U:%G %n' proyectos/<uuid>/fuente/*`), que el conteo de `fuente/` subió exactamente en 3 (de 120 a 123) y que la carpeta `proyectos/<uuid>/entrada/<lote>` **ya no existe**.
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

## Riesgos aceptados y notas
*(Ronda final de la Parte A, 2026-10-03. Lo de aquí no se arregló a propósito; está escrito para que quien venga no lo redescubra.)*
- La migración 006a usa `CREATE TABLE IF NOT EXISTS`: si la tabla ya existe con otra forma, no la toca. Un cambio futuro de `project_documents` tiene que ser una **006b con `ALTER`**, no una edición de la 006a.
- Carrera conteo→`DROP` en la reversión de 006a (`scripts/b9_revertir_005.py`): entre contar las filas y borrar la tabla puede entrar una fila. La reversión es **manual y con la plataforma parada**.
- Un proyecto **archivado a mitad de un trabajo de LAS MANOS**: el trabajo termina de escribir (aceptado); lo que ya está entregado no se frena.
- `_UUID_CANONICO` está duplicada en `scripts/` (que no es paquete y no puede importar de `las_manos/`): si cambia en uno, se cambia en el otro.
- El `fchmod 0660` de `write_file` aplica a **todo** el workspace. Fuera de `proyectos/` el grupo del archivo es `jaxsvc` (sin miembros extra), así que no abre lectura a nadie. Dentro de `proyectos/`, con setgid el grupo es `fruiz` y el 0660 da lectura y escritura a `fruiz`: es lo buscado, no una apertura. Sería un riesgo si apareciera un directorio con setgid y otro grupo.
- En la prueba de herencia de `tests/test_permisos_proyectos.py`, la escritura de `fruiz` no prueba nada: `fruiz` es dueña del `tmp_path`.
- `procesamiento/compuerta.py` repite la comprobación `tipo == "ole2"` (cosmético).
- **Para que Fernando lo vea (decisión suya):** el modelo de propiedad (`jaxsvc` dueño del árbol) sigue la spec madre §5, y por eso un proceso corriendo como `jaxsvc` podría reescribir la ACL de `proyectos/`. Se repara con root. Si se quiere cerrar, el dueño tendría que ser otro usuario.
