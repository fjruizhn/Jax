#!/usr/bin/env bash
# ops/verificar-arranque-instalado.sh
#
# Compara, archivo por archivo, la configuración de arranque de systemd que
# vive en el repo contra la que está instalada en /etc y /usr/local/sbin de
# esta máquina. Sólo lectura: no escribe en /etc, no corre systemctl más
# allá de `systemctl cat` (la regla de esta rama), no reinicia nada. Sale 0
# ó no según lo que hay, nunca cambia nada.
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
# diferencia impresa, si no:
#   1. cada archivo del manifiesto existe en el repo y en lo instalado, y
#      son byte a byte idénticos (el guion .sh además con el bit ejecutable
#      en los dos lados);
#   2. lo instalado NUNCA es un symlink (auditoría escalón 3, m3: un symlink
#      podría apuntar a otra cosa sin que el cmp de arriba lo note si el
#      destino coincidiera hoy y cambiara mañana);
#   3. propietario root:root y modo 644 (755 para el .sh) en cada archivo instalado
#      (m3);
#   4. NINGÚN archivo de más participa del arranque de cada unidad del
#      manifiesto -- en PRODUCCIÓN la verdad la da `systemctl cat <unidad>`
#      (permitido) mismo: sus líneas `# /ruta` son EXACTAMENTE los archivos
#      que systemd fusiona de verdad, con toda su jerarquía de rutas y
#      prefijos con guion ya resuelta (auditoría escalón 3, M2 ronda 2: la
#      primera versión de este chequeo sólo miraba 3 rutas fijas por
#      unidad y un intruso en cualquiera de las otras 9 rutas de
#      `systemd-analyze unit-paths`, o en un prefijo con guion como
#      `jax-.service.d/`, pasaba con rc=0 -- reproducido antes de este
#      arreglo).
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
# cambia, sigue leyendo /etc, /run, /usr/lib reales. Existe SÓLO para poder
# ejercitar la lógica de este guion (symlink, dueño/modo, drop-ins de más)
# contra un árbol de prueba bajo /tmp, sin escribir NUNCA en el /etc real --
# la regla de esta rama es "producción, sólo lectura". No se documenta en el
# manifiesto ni en el runbook de instalación: nadie lo usa para instalar,
# sólo para probar este guion antes de confiar en él.
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
  # MINOR-4 (auditoría escalón 3, ronda 2): el directorio que CONTIENE cada
  # archivo instalado tampoco puede ser un symlink (podría apuntar a un
  # árbol distinto sin que ningún cmp de archivo lo note) ni tener dueño o
  # modo raros -- root:root 755, siempre, para los directorios que este
  # árbol versiona. $1 = ruta real del ARCHIVO instalado (con RAIZ_PRUEBA
  # si aplica), $2 = ruta del archivo a mostrar (siempre la de producción).
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

# --- Ningún archivo de más participa del arranque (M2, ronda 2) --------
#
# Las 12 rutas reales de `systemd-analyze unit-paths` en hall9000 (systemd
# 259) -- constante acá SOLO para el modo de prueba (ver abajo); en
# producción no hace falta: `systemctl cat` ya las conoce todas.
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

unidades="$(awk -F'\t' '$2 ~ /^\/etc\/systemd\/system\/[^\/]+\.(service|timer)$/ { n=split($2,a,"/"); print a[n] }' "$MANIFIESTO" | sort -u)"

while IFS= read -r unidad; do
  [ -z "$unidad" ] && continue
  nombre_sin_sufijo="${unidad%.*}"
  tipo="${unidad##*.}"

  esperados="$( { echo "/etc/systemd/system/$unidad"; awk -F'\t' -v pref="/etc/systemd/system/$unidad.d/" \
    'index($2, pref) == 1 {print $2}' "$MANIFIESTO"; } | sort -u)"

  if [ -z "$RAIZ_PRUEBA" ]; then
    # Producción: la verdad la da systemd mismo. `systemctl cat` imprime
    # una línea "# /ruta" por cada archivo que de verdad fusiona -- ya
    # resolvió toda la jerarquía de prefijos con guion y la prioridad de
    # /etc/systemd/system.control. `sort -u` porque algunas versiones de
    # systemd repiten la línea de la unidad base (visto en systemd 259);
    # no afecta la comparación por conjunto.
    reales="$(systemctl cat "$unidad" 2>/dev/null | grep '^# /' | sed 's/^# //' | sort -u)"
  else
    # Prueba: no hay systemd real al que preguntarle -- se recorre a mano
    # la enumeración completa (MINOR-1: un directorio que existe pero no
    # se puede LISTAR es un fallo, no un "no había nada que mirar").
    mapfile -t prefijos < <(prefijos_de_guion "$nombre_sin_sufijo")
    dirs_relativos=()
    for p in "${prefijos[@]}"; do
      dirs_relativos+=("${p}.${tipo}.d")
    done
    dirs_relativos+=("${tipo}.d")

    reales_lista=""
    if [ -f "$RAIZ_PRUEBA/etc/systemd/system/$unidad" ]; then
      reales_lista="$reales_lista
/etc/systemd/system/$unidad"
    fi
    for ruta_base in "${UNIT_PATHS_SYSTEMD[@]}"; do
      for rel in "${dirs_relativos[@]}"; do
        dir="$RAIZ_PRUEBA$ruta_base/$rel"
        if [ -e "$dir" ] || [ -L "$dir" ]; then
          if [ ! -d "$dir" ] || [ ! -r "$dir" ] || [ ! -x "$dir" ]; then
            echo "NO SE PUDO LEER: $ruta_base/$rel (existe pero no es un directorio legible/listable)" >&2
            fallo=1
            continue
          fi
          for conf in "$dir"/*.conf; do
            [ -e "$conf" ] || continue
            reales_lista="$reales_lista
$ruta_base/$rel/$(basename "$conf")"
          done
        fi
      done
    done
    reales="$(printf '%s\n' "$reales_lista" | sed '/^$/d' | sort -u)"
  fi

  if [ "$reales" != "$esperados" ]; then
    sobran="$(comm -13 <(printf '%s\n' "$esperados") <(printf '%s\n' "$reales"))"
    faltan="$(comm -23 <(printf '%s\n' "$esperados") <(printf '%s\n' "$reales"))"
    echo "DIFERENCIA entre lo que $unidad aplica de verdad y el manifiesto:" >&2
    if [ -n "$sobran" ]; then
      echo "  DE MÁS (participa del arranque y el manifiesto no lo sabe):" >&2
      while IFS= read -r x; do echo "    $x" >&2; done <<< "$sobran"
    fi
    if [ -n "$faltan" ]; then
      echo "  el manifiesto los pide y NO participan del arranque:" >&2
      while IFS= read -r x; do echo "    $x" >&2; done <<< "$faltan"
    fi
    fallo=1
  fi
done <<< "$unidades"

if [ "$fallo" -ne 0 ]; then
  echo "verificar-arranque-instalado: hay diferencias entre el repo y lo instalado -- ver arriba" >&2
  exit 1
fi

echo "verificar-arranque-instalado: repo e instalado coinciden ($revisados archivos)"
