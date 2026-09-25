#!/usr/bin/env bash
# ops/verificar-arranque-instalado.sh [RAIZ_PRUEBA]
#
# Compara la configuración de arranque de systemd que vive en el repo
# contra la que está instalada y CARGADA en esta máquina. Sólo lectura: no
# escribe en /etc, no reinicia nada, no hace daemon-reload. Sale 0 ó no
# según lo que hay, nunca cambia nada.
#
# RONDA 5 -- UN SOLO CAMINO (decisión del coordinador tras la ronda 4: dos
# implementaciones distintas -- producción con `systemd-delta`, pruebas
# con una enumeración a mano -- significaba que las pruebas NUNCA
# ejercitaban el código que corre en producción. Además, medido en
# hall9000: `systemd-delta` NO ve todo lo que `systemd-analyze unit-paths`
# sí ve -- generator/transient/system.control y los prefijos con guion no
# aparecen en su salida de la misma forma que en la enumeración directa).
#
# La capa DISCO es SIEMPRE `enumerar_dropins_en_disco`, con UNA sola
# implementación para producción y pruebas -- sólo cambia la raíz:
#   - Producción: RAIZ="/" (el `/etc`, `/run`, `/usr/lib` reales).
#   - Pruebas: RAIZ=<árbol bajo /tmp> (`RAIZ_PRUEBA`).
# Recorre, para CADA unidad del manifiesto, las 12 rutas reales de
# `systemd-analyze unit-paths` -- versionadas acá como constante (ver
# UNIT_PATHS_SYSTEMD) y comparadas contra la lista real cuando este guion
# corre en el host de producción (tests/test_verificar_arranque_instalado.py::
# test_unit_paths_versionadas_coinciden_con_la_real, para que no derive) --
# y en cada una busca: el FRAGMENTO base (`<unidad>` exacto) y los DROP-INS
# (`<unidad>.d/*.conf`, en cada prefijo con guion de systemd.unit(5) y en
# el genérico `service.d/`/`timer.d/`). El manifiesto declara UNA sola
# ubicación válida para el fragmento (`/etc/systemd/system/<u>`) -- un
# fragmento en CUALQUIER otra ruta (mayor prioridad, que de verdad lo
# reemplazaría, o menor prioridad, un duplicado dormido que se activaría
# solo si el de /etc alguna vez desaparece) es una DIFERENCIA.
#
# La capa CARGADO (SÓLO producción: preguntarle a systemd por un árbol de
# `/tmp` no tiene sentido) usa `systemctl show -p FragmentPath
# -p DropInPaths -p NeedDaemonReload` y exige `NeedDaemonReload=no` -- la
# señal REAL y estructurada de "lo cargado puede no reflejar el disco".
#
# MAJOR-A (ronda 4, sostenido): ninguna función asigna `fallo=1` (variable
# del script principal) DESDE DENTRO de una sustitución de comando
# `$(...)` -- esa asignación vive en un SUBSHELL y se pierde en cuanto
# termina. Las funciones sólo imprimen su LISTA por stdout y señalan su
# ESTADO por código de salida; el LLAMADOR, fuera de cualquier subshell,
# es quien pone `fallo=1` (`if ! x="$(funcion ...)"; then fallo=1; fi`).
#
# MINOR-2 (ronda 5): ninguna función se traga un error con `2>/dev/null`
# seguido de un `return 0` incondicional -- cualquier fallo real de
# enumeración propaga un código de salida distinto de cero. Nombres de
# unidad NUNCA se usan como patrón de regex (evita el problema clásico de
# que el "." del nombre matchee cualquier carácter): todas las
# comparaciones contra un nombre de unidad son con `[ = ]`/`-f`/`-e`.
#
# El manifiesto (repo -> instalado) vive en ops/manifiesto-arranque-instalado.tsv,
# para que este guion y tests/test_arranque_instalado.py lean EXACTAMENTE
# la misma lista.
#
# Salida: 0 si TODO lo de abajo se cumple. Distinto de 0 (2 si no hay
# root disponible; 1 en cualquier otro fallo), con CADA diferencia
# impresa, si no -- y sigue revisando el resto (una unidad que
# `systemctl`/la enumeración no puedan resolver no aborta la corrida).
#   1. cada archivo del manifiesto existe en el repo y en lo instalado, y
#      son byte a byte idénticos (el guion .sh además con el bit
#      ejecutable en los dos lados);
#   2. lo instalado NUNCA es un symlink, ni el archivo ni su directorio
#      padre;
#   3. propietario root:root en archivos y directorios, modo 644 en
#      archivos (755 en el .sh) y 755 en los directorios que los
#      contienen;
#   4. NINGÚN fragmento ni drop-in de más participa del arranque de cada
#      unidad del manifiesto (disco, SIEMPRE) y lo que systemd tiene
#      CARGADO coincide, con `NeedDaemonReload=no` (sólo producción).
set -euo pipefail
export LC_ALL=C

# RAIZ_PRUEBA: el argumento tal como llega. MINOR-1 (ronda 5): se
# normaliza con `realpath -m` (no exige que exista -- un RAIZ_PRUEBA de
# prueba típicamente no existe todavía) y, si resuelve a "/" (vacío,
# "/tmp/../", un symlink que apunte ahí, lo que sea), se trata EXACTAMENTE
# igual que si no se hubiera pasado nada: modo PRODUCCIÓN, con la capa
# CARGADO incluida. Sin esto, alguien podría pasar un RAIZ_PRUEBA que en
# los hechos apunta a la raíz real y el guion lo trataría como "modo de
# prueba" -- salteándose la capa cargado -- mientras la capa disco igual
# termina leyendo el /etc real.
RAIZ_PRUEBA_ARG="${1:-}"
if [ -n "$RAIZ_PRUEBA_ARG" ]; then
  RAIZ_PRUEBA_ARG="$(realpath -m -- "$RAIZ_PRUEBA_ARG")"
  [ "$RAIZ_PRUEBA_ARG" = / ] && RAIZ_PRUEBA_ARG=""
fi

# --- Corre ENTERO como root, sólo lectura --------------------------------
if [ "$(id -u)" -ne 0 ]; then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "verificar-arranque-instalado: no hay sudo disponible -- no se puede correr como root. Fallo cerrado." >&2
    exit 2
  fi
  if ! sudo -n true 2>/dev/null; then
    echo "verificar-arranque-instalado: sudo -n no alcanza (¿pide contraseña?) -- no se puede correr como root. Fallo cerrado." >&2
    exit 2
  fi
  exec sudo -n "$0" "$RAIZ_PRUEBA_ARG"
fi

if [ -z "$RAIZ_PRUEBA_ARG" ]; then
  ES_PRODUCCION=1
  RAIZ_DISCO="/"
else
  ES_PRODUCCION=0
  RAIZ_DISCO="$RAIZ_PRUEBA_ARG"
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
  instalada_real="${RAIZ_DISCO%/}$instalada"
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

# --- Ningún fragmento ni drop-in de más participa del arranque -----------
#
# Las 12 rutas reales de `systemd-analyze unit-paths` en hall9000 (systemd
# 259), EN ORDEN DE PRIORIDAD (la primera gana si hay un fragmento
# duplicado). Versionadas acá porque `systemd-analyze` no se puede correr
# contra un árbol de prueba -- tests/test_verificar_arranque_instalado.py
# tiene un test que compara esta lista contra la real cuando corre en el
# host de producción, para que no derive en silencio si una versión nueva
# de systemd cambia el orden o agrega una ruta.
UNIT_PATHS_SYSTEMD=(
  "/etc/systemd/system.control"
  "/run/systemd/system.control"
  "/run/systemd/transient"
  "/run/systemd/generator.early"
  "/etc/systemd/system"
  "/etc/systemd/system.attached"
  "/run/systemd/system"
  "/run/systemd/system.attached"
  "/run/systemd/generator"
  "/usr/local/lib/systemd/system"
  "/usr/lib/systemd/system"
  "/run/systemd/generator.late"
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
  # ÚNICA implementación -- producción (RAIZ="/") y pruebas (RAIZ=árbol
  # bajo /tmp) pasan por acá, nunca dos caminos distintos (ronda 5, causa
  # raíz del rechazo de la ronda 4). $1=unidad $2=nombre-sin-sufijo
  # $3=tipo $4=raíz.
  #
  # Imprime, una ruta de PRODUCCIÓN por línea (sin el prefijo de raíz):
  #   - el FRAGMENTO base -- cualquier ruta de las 12 donde exista un
  #     archivo con el nombre EXACTO de la unidad (no sólo la primera: el
  #     manifiesto declara una única ubicación válida,
  #     /etc/systemd/system/<u>, así que cualquier OTRA -- de más
  #     prioridad, que de verdad reemplazaría al fragmento real, o de
  #     menos, un duplicado dormido -- es una diferencia que el llamador
  #     detecta comparando contra "esperados");
  #   - los DROP-INS -- cada `*.conf` real bajo cualquier directorio que
  #     systemd.unit(5) busque para esta unidad (prefijos con guion +
  #     genérico de tipo), en cualquiera de las 12 rutas.
  # NUNCA mira contenido de archivo, sólo existencia y nombre (BLOCK-1,
  # ronda 3: por diseño, esto no puede confundir una línea de comentario
  # con un archivo real).
  #
  # FUNCIÓN PURA (MAJOR-A): nunca toca `fallo`, sólo devuelve estado por
  # código de salida. MINOR-2 (ronda 5): ningún error se descarta con
  # `2>/dev/null` + `return 0` -- cualquier ruta que exista pero no se
  # pueda leer/listar hace `hubo_error=1`, propagado en el `return` final.
  # Nombres de unidad SIEMPRE comparados con `[ = ]`/`-f`, nunca como
  # patrón de regex.
  local unidad="$1" nombre_sin_sufijo="$2" tipo="$3" raiz="$4"
  local raiz_efectiva="${raiz%/}"
  local prefijos=() dirs_relativos=() p rel ruta_base dir conf archivo_base
  local hubo_error=0

  mapfile -t prefijos < <(prefijos_de_guion "$nombre_sin_sufijo")
  for p in "${prefijos[@]}"; do
    dirs_relativos+=("${p}.${tipo}.d")
  done
  dirs_relativos+=("${tipo}.d")

  # Fragmento base: TODAS las rutas donde haya un archivo con el nombre
  # exacto (comparación de ruta, nunca regex).
  for ruta_base in "${UNIT_PATHS_SYSTEMD[@]}"; do
    archivo_base="$raiz_efectiva$ruta_base/$unidad"
    if [ -e "$archivo_base" ] || [ -L "$archivo_base" ]; then
      if [ ! -f "$archivo_base" ]; then
        echo "RUTA DE FRAGMENTO NO ES UN ARCHIVO REGULAR: $ruta_base/$unidad" >&2
        hubo_error=1
        continue
      fi
      echo "$ruta_base/$unidad"
    fi
  done

  # Drop-ins.
  for ruta_base in "${UNIT_PATHS_SYSTEMD[@]}"; do
    for rel in "${dirs_relativos[@]}"; do
      dir="$raiz_efectiva$ruta_base/$rel"
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

  # a) Disco -- SIEMPRE, la misma función, sólo cambia la raíz.
  if ! reales_disco="$(enumerar_dropins_en_disco "$unidad" "$nombre_sin_sufijo" "$tipo" "$RAIZ_DISCO" | sort -u)"; then
    fallo=1
  fi
  if [ "$reales_disco" != "$esperados" ]; then
    reportar_diferencia "$unidad" "disco" "$esperados" "$reales_disco"
  fi

  # b) Cargado -- sólo producción.
  if [ "$ES_PRODUCCION" = 1 ]; then
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
