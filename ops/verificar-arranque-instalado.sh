#!/usr/bin/env bash
# ops/verificar-arranque-instalado.sh
#
# Compara, archivo por archivo, la configuración de arranque de systemd que
# vive en el repo contra la que está instalada en /etc y /usr/local/sbin de
# esta máquina. Sólo lectura: no escribe en /etc, no reinicia nada. En
# producción corre `systemctl show` (lectura pura, sin efecto -- auditoría
# escalón 3, ronda 3, BLOCK-1: se probó primero si el entorno lo permite, y
# lo permite; si algún día no, la alternativa documentada es `systemctl cat`
# tomando SÓLO líneas de cabecera reales, nunca contenido). Sale 0 ó no
# según lo que hay, nunca cambia nada.
#
# Decisión de Fernando (2026-09-25, opción A): versionar en el repo el
# arranque de systemd que hoy sólo vivía en /etc de hall9000, y añadir esta
# verificación para que el drift entre repo e instalado se note solo.
#
# El manifiesto (repo -> instalado) vive en un archivo aparte,
# ops/manifiesto-arranque-instalado.tsv, para que este guion y
# tests/test_arranque_instalado.py lean EXACTAMENTE la misma lista -- dos
# listas que puedan divergir son peor que una sola que puede fallar.
#
# Salida: 0 si TODO lo de abajo se cumple. Distinto de 0, con CADA
# diferencia impresa, si no (y sigue revisando el resto -- MINOR-2: una
# unidad que systemctl no puede mostrar no aborta el guion, sólo cuenta
# como fallo):
#   1. cada archivo del manifiesto existe en el repo y en lo instalado, y
#      son byte a byte idénticos (el guion .sh además con el bit ejecutable
#      en los dos lados);
#   2. lo instalado NUNCA es un symlink, ni el archivo ni su directorio
#      padre (un symlink podría apuntar a otra cosa sin que ningún cmp lo
#      note si el destino coincidiera hoy y cambiara mañana);
#   3. propietario root:root en archivos y directorios, modo 644 en
#      archivos (755 en el .sh) y 755 en los directorios que los contienen;
#   4. NINGÚN archivo de más participa del arranque de cada unidad del
#      manifiesto -- en DOS sentidos independientes, los dos exigidos
#      (auditoría escalón 3, ronda 3, MAJOR-1: `systemctl show`/`cat`
#      reflejan lo CARGADO, no el disco -- un intruso recién copiado sin
#      `daemon-reload` pasaría en verde si sólo se mirara eso):
#      a) lo que hay REALMENTE en disco: las 12 rutas de
#         `systemd-analyze unit-paths` (systemd 259, hall9000) × cada
#         prefijo con guion de la unidad × el genérico de tipo -- NUNCA
#         mirando contenido de archivo, sólo nombres (BLOCK-1: una versión
#         anterior raspaba `systemctl cat` con grep '^# /' y confundía
#         líneas de COMENTARIO dentro de un .conf, como
#         "# /srv/jax-prod/jax/.venv/bin/python (el mismo intérprete...)",
#         con la cabecera real que antepone systemd -- eso ya no puede
#         pasar: esta enumeración no lee contenido de ningún archivo);
#      b) lo que systemd tiene CARGADO: `systemctl show -p FragmentPath
#         -p DropInPaths --value <unidad>` (lectura pura). Si systemctl
#         avisa algo por stderr (p.ej. "changed on disk"), NO se descarta:
#         cuenta como fallo.
set -euo pipefail
# LC_ALL=C: rutas de archivo se comparan byte a byte, no con reglas de
# colación de es_ES/en_US -- sin esto, `sort -u`/`comm` bajo un locale
# UTF-8 pueden desacordar sobre qué es "orden" y `comm` se niega a
# comparar ("input is not in sorted order"), aunque para un ojo humano
# las dos listas se vean ordenadas.
export LC_ALL=C

REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
MANIFIESTO="$REPO/ops/manifiesto-arranque-instalado.tsv"

# RAIZ_PRUEBA: vacío por defecto -- SIN esto, ninguna línea de este guion
# cambia, sigue leyendo /etc, /run, /usr/lib reales y consultando el
# systemd real de esta máquina. Con RAIZ_PRUEBA puesto, la enumeración de
# disco se corre contra un árbol de prueba bajo /tmp Y se DESACTIVA la
# consulta a systemctl (no tendría sentido: systemctl sólo conoce el /etc
# real, nunca un árbol de prueba) -- existe SÓLO para poder ejercitar la
# lógica de este guion sin escribir NUNCA en el /etc real. No se documenta
# en el manifiesto ni en el runbook de instalación: nadie lo usa para
# instalar, sólo para probar este guion antes de confiar en él.
RAIZ_PRUEBA="${RAIZ_PRUEBA:-}"

[ -f "$MANIFIESTO" ] || {
  echo "verificar-arranque-instalado: no se encontró el manifiesto: $MANIFIESTO" >&2
  exit 1
}

fallo=0
revisados=0

verificar_propietario_y_modo() {
  # $1 = ruta real a stat-ear (con RAIZ_PRUEBA si aplica), $2 = ruta a
  # mostrar en los mensajes (siempre la de producción, nunca la de prueba),
  # $3 = "1" si es el .sh (755), si no 644.
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
  # El directorio que CONTIENE cada archivo instalado tampoco puede ser un
  # symlink ni tener dueño o modo raros -- root:root 755, siempre, para los
  # directorios que este árbol versiona. $1 = ruta real del ARCHIVO
  # instalado (con RAIZ_PRUEBA si aplica), $2 = ruta del archivo a mostrar
  # (siempre la de producción).
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

# --- Ningún archivo de más participa del arranque (M2) ------------------
#
# Las 12 rutas reales de `systemd-analyze unit-paths` en hall9000 (systemd
# 259).
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
  # systemd.unit(5): para una unidad "foo-bar-baz.service", los drop-ins se
  # buscan en foo-bar-baz.service.d/, foo-bar-.service.d/ y foo-.service.d/
  # -- el nombre completo y cada truncado después de un guion, de derecha a
  # izquierda, sin bajar de un componente. $1 = nombre SIN sufijo (ej.
  # jax-memory-worker). Imprime un prefijo por línea, empezando por el
  # nombre completo.
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
  # $1 = unidad, $2 = nombre sin sufijo, $3 = tipo (service|timer),
  # $4 = raíz (RAIZ_PRUEBA o vacío = disco real). Imprime, una ruta de
  # PRODUCCIÓN por línea (sin el prefijo de raíz), la unidad base si
  # existe y cada *.conf real bajo cualquier directorio que systemd.unit(5)
  # busque para esta unidad, en cualquiera de las 12 rutas. NUNCA mira
  # contenido de archivo -- sólo existencia y nombre (BLOCK-1: por diseño,
  # esto no puede confundir una línea de comentario con un archivo).
  # Un directorio que EXISTE pero no se puede LISTAR (-r/-x) es un FALLO
  # explícito, no un "no había nada que mirar".
  local unidad="$1" nombre_sin_sufijo="$2" tipo="$3" raiz="$4"
  local prefijos=() dirs_relativos=() p rel ruta_base dir conf

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
          fallo=1
          continue
        fi
        for conf in "$dir"/*.conf; do
          [ -e "$conf" ] || continue
          echo "$ruta_base/$rel/$(basename -- "$conf")"
        done
      fi
    done
  done
}

obtener_reales_cargado() {
  # $1 = unidad. Imprime, una ruta de producción por línea, lo que systemd
  # tiene CARGADO ahora mismo: FragmentPath + cada DropInPath
  # (`systemctl show`, lectura pura, sin tocar contenido de ningún archivo
  # -- inmune por diseño al BLOCK-1). Si systemctl avisa algo por stderr
  # (p.ej. unidad con archivos "changed on disk"), NO se descarta: se
  # imprime y cuenta como fallo (MAJOR-1 -- es justo la señal de que esta
  # vista puede no reflejar el disco). Si systemctl falla (unidad
  # inexistente, systemctl roto), mensaje claro; la función no imprime
  # nada más y el llamador sigue con la unidad siguiente (MINOR-2).
  local unidad="$1" salida archivo_err err
  archivo_err="$(mktemp)"
  if ! salida="$(systemctl show -p FragmentPath -p DropInPaths --value "$unidad" 2>"$archivo_err")"; then
    echo "SYSTEMCTL SHOW FALLÓ para $unidad (¿la unidad no existe?): $(cat -- "$archivo_err")" >&2
    fallo=1
    rm -f -- "$archivo_err"
    return
  fi
  err="$(cat -- "$archivo_err")"
  rm -f -- "$archivo_err"
  if [ -n "$err" ]; then
    echo "AVISO DE SYSTEMCTL (no descartado) para $unidad: $err" >&2
    fallo=1
  fi
  printf '%s\n' "$salida" | sed -n '1p'
  printf '%s\n' "$salida" | sed -n '2p' | tr ' ' '\n' | sed '/^$/d'
}

reportar_diferencia() {
  # $1 = unidad, $2 = origen ("disco" o "cargado por systemd"),
  # $3 = esperados (multilínea), $4 = reales (multilínea).
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
}

unidades="$(awk -F'\t' '$2 ~ /^\/etc\/systemd\/system\/[^\/]+\.(service|timer)$/ { n=split($2,a,"/"); print a[n] }' "$MANIFIESTO" | sort -u)"

while IFS= read -r unidad; do
  [ -z "$unidad" ] && continue
  nombre_sin_sufijo="${unidad%.*}"
  tipo="${unidad##*.}"

  esperados="$( { echo "/etc/systemd/system/$unidad"; awk -F'\t' -v pref="/etc/systemd/system/$unidad.d/" \
    'index($2, pref) == 1 {print $2}' "$MANIFIESTO"; } | sort -u)"

  # a) Lo que hay REALMENTE en disco -- siempre, con RAIZ_PRUEBA (vacío en
  #    producción = disco real; un árbol de prueba en tests).
  reales_disco="$(enumerar_dropins_en_disco "$unidad" "$nombre_sin_sufijo" "$tipo" "$RAIZ_PRUEBA" | sort -u)"
  if [ "$reales_disco" != "$esperados" ]; then
    reportar_diferencia "$unidad" "disco" "$esperados" "$reales_disco"
    fallo=1
  fi

  # b) Lo que systemd tiene CARGADO -- SÓLO en producción (RAIZ_PRUEBA
  #    vacío): preguntarle a systemctl por un árbol de prueba no tendría
  #    sentido, sólo conoce el /etc real de esta máquina.
  if [ -z "$RAIZ_PRUEBA" ]; then
    reales_cargado="$(obtener_reales_cargado "$unidad" | sort -u)"
    if [ "$reales_cargado" != "$esperados" ]; then
      reportar_diferencia "$unidad" "cargado por systemd" "$esperados" "$reales_cargado"
      fallo=1
    fi
  fi
done <<< "$unidades"

if [ "$fallo" -ne 0 ]; then
  echo "verificar-arranque-instalado: hay diferencias entre el repo y lo instalado -- ver arriba" >&2
  exit 1
fi

echo "verificar-arranque-instalado: repo e instalado coinciden ($revisados archivos)"
