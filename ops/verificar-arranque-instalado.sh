#!/usr/bin/env bash
# ops/verificar-arranque-instalado.sh [RAIZ_PRUEBA]
#
# Compara la configuración de arranque de systemd que vive en el repo
# contra la que está instalada y CARGADA en esta máquina. Sólo lectura: no
# escribe en /etc, no reinicia nada, no hace daemon-reload. Sale 0 ó no
# según lo que hay, nunca cambia nada.
#
# RONDA 4 -- simplificado de raíz (no otro parche) después de 3 rondas de
# auditoría sobre este mismo guion:
#
#   1. Corre ENTERO como root (`sudo -n`, re-exec al principio). Así
#      desaparece de raíz el caso "un directorio que fruiz no puede
#      listar" (ronda 3, MAJOR-2, resuelto ahí sólo para el chequeo de
#      árbol limpio de los instaladores -- acá se elimina la causa, no el
#      síntoma). Sin `sudo` disponible, o si `sudo -n` no alcanza: FALLA
#      CERRADO (rc=2), nunca sigue como usuario sin privilegios fingiendo
#      que pudo revisar todo (medido: `systemd-delta` sin root ve 15
#      líneas contra 48 con root en esta misma máquina -- sin root, este
#      guion mentiría en verde por lo que no pudo ver).
#   2. Disco: la verdad la da `systemd-delta`, no una enumeración a mano
#      de directorios -- lee las 12 rutas de `systemd-analyze unit-paths`
#      (incluidas system.control, transient, generator.early) Y la
#      jerarquía de prefijos con guion, todo resuelto por systemd mismo.
#      Más `systemctl show -p FragmentPath` para el fragmento BASE activo
#      -- un fragmento servido desde system.control/transient en vez de
#      /etc/systemd/system/<u> tiene que dar DIFERENCIA, y `systemd-delta`
#      no resuelve por sí solo CUÁL fragmento está activo, sólo que hay
#      varios con el mismo nombre.
#   3. Cargado: `systemctl show -p FragmentPath -p DropInPaths
#      -p NeedDaemonReload`, y se EXIGE `NeedDaemonReload=no` -- la señal
#      real y estructurada de "lo cargado puede no reflejar el disco", en
#      vez de rascar avisos de texto libre por stderr.
#   4. MAJOR-A (ronda 4, RECHAZO de la ronda 3): NINGUNA función asigna
#      `fallo=1` (variable del script principal) DESDE DENTRO de una
#      sustitución de comando `$(...)` -- esa asignación vive en un
#      SUBSHELL y se pierde en cuanto termina, aunque el mensaje ya se
#      haya impreso por stderr. Las funciones ahora sólo hacen dos cosas:
#      imprimen su LISTA por stdout, y señalan su ESTADO por código de
#      salida (nunca tocan `fallo` ellas mismas) -- el LLAMADOR, fuera de
#      cualquier subshell, es quien pone `fallo=1` mirando ese código
#      (`if ! x="$(funcion ...)"; then fallo=1; fi`).
#
# El manifiesto (repo -> instalado) vive en ops/manifiesto-arranque-instalado.tsv,
# para que este guion y tests/test_arranque_instalado.py lean EXACTAMENTE
# la misma lista.
#
# Salida: 0 si TODO lo de abajo se cumple. Distinto de 0 (2 si no hay
# root disponible; 1 en cualquier otro fallo), con CADA diferencia
# impresa, si no -- y sigue revisando el resto (una unidad que
# `systemctl`/`systemd-delta` no puedan resolver no aborta la corrida).
#   1. cada archivo del manifiesto existe en el repo y en lo instalado, y
#      son byte a byte idénticos (el guion .sh además con el bit
#      ejecutable en los dos lados);
#   2. lo instalado NUNCA es un symlink, ni el archivo ni su directorio
#      padre;
#   3. propietario root:root en archivos y directorios, modo 644 en
#      archivos (755 en el .sh) y 755 en los directorios que los
#      contienen;
#   4. NINGÚN archivo de más participa del arranque de cada unidad del
#      manifiesto (disco vía `systemd-delta` + fragmento vía `systemctl
#      show`), y lo que systemd tiene CARGADO coincide (`systemctl show`)
#      con `NeedDaemonReload=no`.
set -euo pipefail
export LC_ALL=C

RAIZ_PRUEBA="${1:-}"

# --- 1) Corre ENTERO como root, sólo lectura -----------------------------
if [ "$(id -u)" -ne 0 ]; then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "verificar-arranque-instalado: no hay sudo disponible -- no se puede correr como root. Fallo cerrado." >&2
    exit 2
  fi
  if ! sudo -n true 2>/dev/null; then
    echo "verificar-arranque-instalado: sudo -n no alcanza (¿pide contraseña?) -- no se puede correr como root. Fallo cerrado." >&2
    exit 2
  fi
  exec sudo -n "$0" "$RAIZ_PRUEBA"
fi

REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
MANIFIESTO="$REPO/ops/manifiesto-arranque-instalado.tsv"

[ -f "$MANIFIESTO" ] || {
  echo "verificar-arranque-instalado: no se encontró el manifiesto: $MANIFIESTO" >&2
  exit 1
}

fallo=0
revisados=0

verificar_propietario_y_modo() {
  # $1 = ruta real a stat-ear (con RAIZ_PRUEBA si aplica), $2 = ruta a
  # mostrar en los mensajes (siempre la de producción), $3 = "1" si es el
  # .sh (755), si no 644. Llamada DIRECTA (nunca dentro de $(...)): puede
  # tocar `fallo` ella misma sin violar MAJOR-A.
  local real="$1" mostrar="$2" es_sh="$3" propietario grupo modo esperado
  read -r propietario grupo modo < <(stat -c '%U %G %a' "$real" 2>/dev/null) || {
    echo "NO SE PUDO LEER propietario/modo: $mostrar" >&2
    fallo=1
    return
  }
  if [ "$propietario" != root ] || [ "$grupo" != root ]; then
    echo "PROPIETARIO INCORRECTO: $mostrar es $propietario:$grupo -- se esperaba root:root" >&2
    fallo=1
  fi
  esperado=644
  [ "$es_sh" = 1 ] && esperado=755
  if [ "$modo" != "$esperado" ]; then
    echo "MODO INCORRECTO: $mostrar es $modo -- se esperaba $esperado" >&2
    fallo=1
  fi
}

declare -A DIRS_PADRE_REVISADOS

verificar_directorio_padre() {
  # Llamada DIRECTA (nunca dentro de $(...)): igual criterio que arriba.
  local archivo_real="$1" archivo_mostrar="$2" dir_real dir_mostrar propietario grupo modo
  dir_real="$(dirname -- "$archivo_real")"
  dir_mostrar="$(dirname -- "$archivo_mostrar")"
  [ -n "${DIRS_PADRE_REVISADOS[$dir_mostrar]:-}" ] && return
  DIRS_PADRE_REVISADOS[$dir_mostrar]=1

  if [ -L "$dir_real" ]; then
    echo "SYMLINK el directorio padre (se esperaba un directorio real): $dir_mostrar -> $(readlink -- "$dir_real")" >&2
    fallo=1
    return
  fi
  read -r propietario grupo modo < <(stat -c '%U %G %a' "$dir_real" 2>/dev/null) || {
    echo "NO SE PUDO LEER propietario/modo del directorio: $dir_mostrar" >&2
    fallo=1
    return
  }
  if [ "$propietario" != root ] || [ "$grupo" != root ]; then
    echo "PROPIETARIO INCORRECTO del directorio: $dir_mostrar es $propietario:$grupo -- se esperaba root:root" >&2
    fallo=1
  fi
  if [ "$modo" != 755 ]; then
    echo "MODO INCORRECTO del directorio: $dir_mostrar es $modo -- se esperaba 755" >&2
    fallo=1
  fi
}

while IFS=$'\t' read -r repo_rel instalada || [ -n "$repo_rel" ]; do
  [ -z "$repo_rel" ] && continue
  repo_abs="$REPO/$repo_rel"
  instalada_real="$RAIZ_PRUEBA$instalada"
  revisados=$((revisados + 1))

  if [ ! -f "$repo_abs" ]; then
    echo "FALTA en el repo: $repo_abs" >&2
    fallo=1
    continue
  fi

  verificar_directorio_padre "$instalada_real" "$instalada"

  if [ -L "$instalada_real" ]; then
    echo "SYMLINK instalado (se esperaba archivo regular): $instalada -> $(readlink "$instalada_real")" >&2
    fallo=1
    continue
  fi

  if [ ! -f "$instalada_real" ]; then
    echo "FALTA instalado: $instalada (repo: $repo_abs)" >&2
    fallo=1
    continue
  fi

  if ! cmp -s "$repo_abs" "$instalada_real"; then
    detalle="$(cmp "$repo_abs" "$instalada_real" 2>&1 || true)"
    echo "DIFIERE: $repo_abs != $instalada -- $detalle" >&2
    fallo=1
  fi

  case "$repo_rel" in
    *.sh)
      if [ ! -x "$repo_abs" ]; then
        echo "SIN BIT EJECUTABLE en el repo: $repo_abs" >&2
        fallo=1
      fi
      verificar_propietario_y_modo "$instalada_real" "$instalada" 1
      ;;
    *)
      verificar_propietario_y_modo "$instalada_real" "$instalada" 0
      ;;
  esac
done < "$MANIFIESTO"

if [ "$revisados" -eq 0 ]; then
  echo "verificar-arranque-instalado: el manifiesto está vacío -- $MANIFIESTO" >&2
  exit 1
fi

# --- 4) Ningún archivo de más participa del arranque ---------------------

UNIT_PATHS_SYSTEMD=(
  "/etc/systemd/system.control"
  "/run/systemd/system.control"
  "/run/systemd/transient"
  "/etc/systemd/system"
  "/etc/systemd/system.attached"
  "/run/systemd/system"
  "/run/systemd/system.attached"
  "/run/systemd/generator"
  "/usr/local/lib/systemd/system"
  "/usr/lib/systemd/system"
  "/run/systemd/generator.late"
  "/run/systemd/generator.early"
)

prefijos_de_guion() {
  # systemd.unit(5): para "foo-bar-baz.service", los drop-ins se buscan en
  # foo-bar-baz.service.d/, foo-bar-.service.d/ y foo-.service.d/. $1 =
  # nombre SIN sufijo. Imprime un prefijo por línea, del más largo al más
  # corto. Función PURA: sólo stdout, nunca toca `fallo`.
  local base="$1" partes n i j prefijo
  IFS='-' read -ra partes <<< "$base"
  n=${#partes[@]}
  echo "$base"
  for ((i = n - 1; i >= 1; i--)); do
    prefijo=""
    for ((j = 0; j < i; j++)); do
      prefijo="${prefijo}${partes[j]}-"
    done
    echo "$prefijo"
  done
}

enumerar_dropins_en_disco() {
  # SÓLO PARA MODO DE PRUEBA (RAIZ_PRUEBA no vacío): enumeración a mano de
  # las 12 rutas × prefijos con guion × genérico de tipo -- no hay systemd
  # real al que preguntarle por un árbol de /tmp. En producción, este
  # chequeo lo hace `obtener_reales_disco_produccion` (systemd-delta), no
  # esta función.
  #
  # $1=unidad $2=nombre-sin-sufijo $3=tipo $4=raíz. FUNCIÓN PURA (MAJOR-A,
  # ronda 4): imprime la lista por stdout, nunca toca `fallo`; devuelve 1
  # si algún directorio existente no se pudo LISTAR (el llamador decide
  # qué hacer con eso).
  local unidad="$1" nombre_sin_sufijo="$2" tipo="$3" raiz="$4"
  local prefijos=() dirs_relativos=() p rel ruta_base dir conf
  local hubo_error=0

  mapfile -t prefijos < <(prefijos_de_guion "$nombre_sin_sufijo")
  for p in "${prefijos[@]}"; do
    dirs_relativos+=("${p}.${tipo}.d")
  done
  dirs_relativos+=("${tipo}.d")

  if [ -f "$raiz/etc/systemd/system/$unidad" ]; then
    echo "/etc/systemd/system/$unidad"
  fi
  for ruta_base in "${UNIT_PATHS_SYSTEMD[@]}"; do
    for rel in "${dirs_relativos[@]}"; do
      dir="$raiz$ruta_base/$rel"
      if [ -e "$dir" ] || [ -L "$dir" ]; then
        if [ ! -d "$dir" ] || [ ! -r "$dir" ] || [ ! -x "$dir" ]; then
          echo "NO SE PUDO LEER: $ruta_base/$rel (existe pero no es un directorio legible/listable)" >&2
          hubo_error=1
          continue
        fi
        for conf in "$dir"/*.conf; do
          [ -e "$conf" ] || continue
          echo "$ruta_base/$rel/$(basename -- "$conf")"
        done
      fi
    done
  done
  return "$hubo_error"
}

obtener_reales_disco_produccion() {
  # SÓLO PRODUCCIÓN (RAIZ_PRUEBA vacío). La verdad la da systemd mismo:
  # `systemctl show -p FragmentPath` para el fragmento BASE activo (si
  # está servido desde system.control/transient en vez de
  # /etc/systemd/system/<u>, esto no matchea el manifiesto -> DIFERENCIA,
  # sin que `systemd-delta` tenga que resolverlo) + `systemd-delta` para
  # los drop-ins reales (ya resolvió las 12 rutas y los prefijos con
  # guion). FUNCIÓN PURA (MAJOR-A): stdout = lista, código de salida =
  # estado, nunca toca `fallo`.
  local unidad="$1" fragmento
  if ! fragmento="$(systemctl show -p FragmentPath --value "$unidad" 2>&1)"; then
    echo "SYSTEMCTL SHOW (FragmentPath) FALLÓ para $unidad: $fragmento" >&2
    return 1
  fi
  if [ -z "$fragmento" ]; then
    echo "systemctl show -p FragmentPath no devolvió nada para $unidad (¿no existe la unidad?)" >&2
    return 1
  fi
  echo "$fragmento"
  systemd-delta --no-pager --type=overridden,extended,redirected,masked,equivalent 2>/dev/null \
    | grep -E "^\[EXTENDED\][[:space:]]+/etc/systemd/system/${unidad}[[:space:]]" \
    | sed -E 's/^\[EXTENDED\][[:space:]]+[^[:space:]]+[[:space:]]+(->|→)[[:space:]]+//'
  return 0
}

obtener_reales_cargado() {
  # SÓLO PRODUCCIÓN. Lo que systemd tiene CARGADO ahora mismo:
  # `systemctl show -p FragmentPath -p DropInPaths -p NeedDaemonReload`.
  # `NeedDaemonReload` distinto de "no" es la señal REAL (no un aviso de
  # texto libre por stderr) de que esta vista puede no reflejar el disco.
  # FUNCIÓN PURA (MAJOR-A): stdout = lista, código de salida = estado,
  # nunca toca `fallo`.
  local unidad="$1" salida archivo_err err need_reload hubo_error=0
  archivo_err="$(mktemp)"
  if ! salida="$(systemctl show -p FragmentPath -p DropInPaths -p NeedDaemonReload --value "$unidad" 2>"$archivo_err")"; then
    echo "SYSTEMCTL SHOW FALLÓ para $unidad (¿la unidad no existe?): $(cat -- "$archivo_err")" >&2
    rm -f -- "$archivo_err"
    return 1
  fi
  err="$(cat -- "$archivo_err")"; rm -f -- "$archivo_err"
  if [ -n "$err" ]; then
    echo "AVISO DE SYSTEMCTL (no descartado) para $unidad: $err" >&2
    hubo_error=1
  fi
  need_reload="$(printf '%s\n' "$salida" | sed -n '3p')"
  if [ "$need_reload" != no ]; then
    echo "NeedDaemonReload=$need_reload para $unidad -- lo cargado por systemd puede no reflejar el disco" >&2
    hubo_error=1
  fi
  printf '%s\n' "$salida" | sed -n '1p'
  printf '%s\n' "$salida" | sed -n '2p' | tr ' ' '\n' | sed '/^$/d'
  return "$hubo_error"
}

reportar_diferencia() {
  # Llamada DIRECTA: puede tocar `fallo`.
  local unidad="$1" origen="$2" esperados="$3" reales="$4" sobran faltan
  sobran="$(comm -13 <(printf '%s\n' "$esperados") <(printf '%s\n' "$reales"))"
  faltan="$(comm -23 <(printf '%s\n' "$esperados") <(printf '%s\n' "$reales"))"
  echo "DIFERENCIA ($origen) entre lo que $unidad aplica de verdad y el manifiesto:" >&2
  if [ -n "$sobran" ]; then
    echo "  DE MÁS (participa del arranque y el manifiesto no lo sabe):" >&2
    while IFS= read -r x; do echo "    $x" >&2; done <<< "$sobran"
  fi
  if [ -n "$faltan" ]; then
    echo "  el manifiesto los pide y NO participan del arranque:" >&2
    while IFS= read -r x; do echo "    $x" >&2; done <<< "$faltan"
  fi
  fallo=1
}

unidades="$(awk -F'\t' '$2 ~ /^\/etc\/systemd\/system\/[^\/]+\.(service|timer)$/ { n=split($2,a,"/"); print a[n] }' "$MANIFIESTO" | sort -u)"

while IFS= read -r unidad; do
  [ -z "$unidad" ] && continue
  nombre_sin_sufijo="${unidad%.*}"
  tipo="${unidad##*.}"

  esperados="$( { echo "/etc/systemd/system/$unidad"; awk -F'\t' -v pref="/etc/systemd/system/$unidad.d/" \
    'index($2, pref) == 1 {print $2}' "$MANIFIESTO"; } | sort -u)"

  # a) Disco.
  if [ -z "$RAIZ_PRUEBA" ]; then
    if ! reales_disco="$(obtener_reales_disco_produccion "$unidad" | sort -u)"; then
      fallo=1
    fi
  else
    if ! reales_disco="$(enumerar_dropins_en_disco "$unidad" "$nombre_sin_sufijo" "$tipo" "$RAIZ_PRUEBA" | sort -u)"; then
      fallo=1
    fi
  fi
  if [ "$reales_disco" != "$esperados" ]; then
    reportar_diferencia "$unidad" "disco" "$esperados" "$reales_disco"
  fi

  # b) Cargado -- sólo producción (preguntarle a systemctl por un árbol de
  #    prueba no tendría sentido).
  if [ -z "$RAIZ_PRUEBA" ]; then
    if ! reales_cargado="$(obtener_reales_cargado "$unidad" | sort -u)"; then
      fallo=1
    fi
    if [ "$reales_cargado" != "$esperados" ]; then
      reportar_diferencia "$unidad" "cargado por systemd" "$esperados" "$reales_cargado"
    fi
  fi
done <<< "$unidades"

if [ "$fallo" -ne 0 ]; then
  echo "verificar-arranque-instalado: hay diferencias entre el repo y lo instalado -- ver arriba" >&2
  exit 1
fi

echo "verificar-arranque-instalado: repo e instalado coinciden ($revisados archivos)"
